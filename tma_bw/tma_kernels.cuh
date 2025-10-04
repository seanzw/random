#pragma once
#include "kernels_common.cuh"

// TMA bulk copy
template <int bytes>
__device__ inline void cp_async_bulk(void *smem_dst, const void *global_src,
                                     unsigned long long *bar_addr) {
  unsigned long long dst_s, src_g, bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(dst_s) : "l"(smem_dst));
  asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_g) : "l"(global_src));
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));

  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1], %3, [%2];" ::"l"(dst_s),
               "l"(src_g), "l"(bar_s), "n"(bytes));
}

// ---- TMA Kernel ----
template <int Stages, int CHUNK_BYTES, int REPEAT>
__global__ void tma_bw_kernel(const uint8_t *__restrict__ src,
                              unsigned long long *__restrict__ sink,
                              size_t total_bytes) {
  extern __shared__ __align__(16) uint8_t smem[];

  __shared__ unsigned long long fwd_bar[Stages];
  __shared__ unsigned long long bwd_bar[Stages];

  if (threadIdx.x == 0) {
    for (int s = 0; s < Stages; ++s) {
      mbarrier_init(&fwd_bar[s], 1);
      mbarrier_init(&bwd_bar[s], 1);
    }
  }
  __syncthreads();

  const size_t total_chunks = total_bytes / CHUNK_BYTES;
  int sum = 0;

  // Producer: wait for slot to be free, then issue copy
  if (threadIdx.x == PRODUCER_THREAD) {
    for (size_t i = blockIdx.x, stage_idx = 0; i < total_chunks * REPEAT;
         i += gridDim.x, stage_idx++) {
      const int slot = int(stage_idx % Stages);
      uint8_t *dst_slot = smem + slot * CHUNK_BYTES;
      const uint8_t *src_chunk = src + (i % total_chunks) * CHUNK_BYTES;

      unsigned want_free = (stage_idx % (Stages * 2)) < Stages;
      wait(&bwd_bar[slot], want_free);

      mbarrier_arrive_expect_tx(&fwd_bar[slot], CHUNK_BYTES);
      cp_async_bulk<CHUNK_BYTES>(dst_slot, src_chunk, &fwd_bar[slot]);
    }
  }

  // Consumer: wait for data ready, then consume and free slot
  else if (threadIdx.x == CONSUMER_THREAD) {
    for (size_t i = blockIdx.x, stage_idx = 0; i < total_chunks * REPEAT;
         i += gridDim.x, stage_idx++) {
      const int slot = int(stage_idx % Stages);
      uint8_t *dst_slot = smem + slot * CHUNK_BYTES;

      unsigned want_ready = (stage_idx % (Stages * 2)) >= Stages;
      wait(&fwd_bar[slot], want_ready);

      auto p32 = reinterpret_cast<const int *>(dst_slot);
      sum += *p32;

      mbarrier_arrive(&bwd_bar[slot]);
    }
    sink[blockIdx.x] = sum;
  }
}