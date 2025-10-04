#pragma once
#include "kernels_common.cuh"

// cp.async 16-byte copy (cache-global)
__device__ inline void cp_async_16(void *smem_dst, const void *global_src) {
  unsigned long long dst_s, src_g;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"l"(dst_s),
               "l"(src_g));
}

// cp.async 8-byte copy (cache-all, since cg doesn't support 8B)
__device__ inline void cp_async_8(void *smem_dst, const void *global_src) {
  unsigned long long dst_s, src_g;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"l"(dst_s),
               "l"(src_g));
}

// cp.async 4-byte copy (cache-all, since cg doesn't support 4B)
__device__ inline void cp_async_4(void *smem_dst, const void *global_src) {
  unsigned long long dst_s, src_g;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"l"(dst_s),
               "l"(src_g));
}

// Wait for cp.async operations to complete
template <int group_size = 0> __device__ inline void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(group_size));
}

// Commit cp.async group - explicitly commit pending cp.async operations
__device__ inline void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

template <int Stages>
__device__ inline void epilogue_wait_and_signal(int stage_idx, int lane_id,
                                                unsigned long long *fwd_bar,
                                                integer_sequence<>) {}

template <int Stages, int I, int... Is>
__device__ inline void epilogue_wait_and_signal(int stage_idx, int lane_id,
                                                unsigned long long *fwd_bar,
                                                integer_sequence<I, Is...>) {
  int prev_stage_idx = stage_idx - Stages + I;
  if (prev_stage_idx >= 0) {
    const int slot = prev_stage_idx % Stages;
    cp_async_wait_group<Stages - 1 - I>();
    __syncwarp();
    if (lane_id == 0) {
      mbarrier_arrive(&fwd_bar[slot]);
    }
    __syncwarp();
  }
  epilogue_wait_and_signal<Stages>(stage_idx, lane_id, fwd_bar,
                                   integer_sequence<Is...>{});
}

template <int CHUNK_BYTES>
__device__ inline void cp_async_chunk(uint8_t *dst_slot,
                                      const uint8_t *src_chunk, int lane_id) {
  // Each thread in the warp copies a portion of the chunk
  // Notice that the minimal transfer size of cp.async.cg is 16 bytes.
  // ! We assume CHUNK_BYTES is divisble by cp_async_cg_bytes

  constexpr int cp_async_cg_bytes = 16;
  static_assert(CHUNK_BYTES % cp_async_cg_bytes == 0,
                "CHUNK_BYTES must be divisible by cp_async_cg_bytes");
  for (int b = lane_id * cp_async_cg_bytes; b < CHUNK_BYTES;
       b += 32 * cp_async_cg_bytes) {

    const uint8_t *src_ptr = src_chunk + b;
    uint8_t *dst_ptr = dst_slot + b;
    cp_async_16(dst_ptr, src_ptr);
  }
}

// ---- cp.async Kernel with Multiple Producer Warps ----
template <int Stages, int CHUNK_BYTES, int REPEAT, int NUM_PRODUCER_WARPS = 1>
__global__ void cp_async_bw_kernel(const uint8_t *__restrict__ src,
                                   unsigned long long *__restrict__ sink,
                                   size_t total_bytes) {
  extern __shared__ __align__(16) uint8_t smem[];

  __shared__ unsigned long long fwd_bar[Stages];
  __shared__ unsigned long long bwd_bar[Stages];

  if (threadIdx.x == 0) {
    for (int s = 0; s < Stages; ++s) {
      mbarrier_init(&fwd_bar[s], 1);
      mbarrier_init(&bwd_bar[s], 1); // Only consumer signals backward
    }
  }
  __syncthreads();

  const size_t total_chunks = total_bytes / CHUNK_BYTES;
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  int sum = 0;

  static_assert(Stages >= NUM_PRODUCER_WARPS,
                "Stages must be at least NUM_PRODUCER_WARPS");

  // Multiple Producer warps: each warp handles different chunks
  if (warp_id < NUM_PRODUCER_WARPS) { // Producer warps

    // Each producer warp processes chunks with stride
    for (size_t chunk_idx = blockIdx.x + warp_id * gridDim.x,
                stage_idx = warp_id;
         chunk_idx < total_chunks * REPEAT;
         chunk_idx += NUM_PRODUCER_WARPS * gridDim.x,
                stage_idx += NUM_PRODUCER_WARPS) {

      const int slot = int(stage_idx % Stages);
      uint8_t *dst_slot = smem + slot * CHUNK_BYTES;
      const uint8_t *src_chunk = src + (chunk_idx % total_chunks) * CHUNK_BYTES;

      // Wait for this slot to be free
      unsigned want_free = (stage_idx % (Stages * 2)) < Stages;
      if (lane_id == 0) {
        wait(&bwd_bar[slot], want_free);
      }
      __syncwarp();

      cp_async_chunk<CHUNK_BYTES>(dst_slot, src_chunk, lane_id);

      // Wait for this warp's cp.async operations to complete
      cp_async_wait_group<0>();
      __syncwarp();

      // Each producer warp signals completion
      if (lane_id == 0) {
        mbarrier_arrive(&fwd_bar[slot]);
      }
    }
  }

  // Consumer warp: wait for data ready, then consume and free slot
  else if (warp_id ==
           NUM_PRODUCER_WARPS) { // Consumer warp (first warp after producers)
    if (lane_id == 0) {          // Only one thread for simplicity
      int stage_idx, i;
      for (i = blockIdx.x, stage_idx = 0; i < total_chunks * REPEAT;
           i += gridDim.x, stage_idx++) {
        const int slot = int(stage_idx % Stages);
        uint8_t *dst_slot = smem + slot * CHUNK_BYTES;

        unsigned want_ready = (stage_idx % (Stages * 2)) >= Stages;
        wait(&fwd_bar[slot], want_ready);

        auto p32 = reinterpret_cast<const int *>(dst_slot);
        sum += *p32;

        mbarrier_arrive(&bwd_bar[slot]);
      }

      // printf("Block %d, chunks %d my chunks %d\n", blockIdx.x,
      //        int(total_chunks), int(stage_idx));
      sink[blockIdx.x] = sum;
    }
  }
}