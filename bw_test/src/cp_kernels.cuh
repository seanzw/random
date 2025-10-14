#pragma once

// ---- Ampere-compatible Synchronization ----
// Waits for the barrier's phase to match the expected parity.
__device__ inline void wait(volatile unsigned int *bar_addr, unsigned expected_parity) {
    // Spin-wait until the lowest bit of the barrier value matches the expected one.
    // The 'volatile' keyword prevents the compiler from optimizing away this read.
    while ((*bar_addr & 1) != expected_parity) {
    }
}

// Arrives at the barrier, incrementing its value atomically.
__device__ inline void arrive(unsigned int *bar_addr) {
    // Atomically increment the barrier. This signals to the waiting warp.
    atomicAdd(bar_addr, 1);
}

// ---- Simplified Copy Kernel ----
template <int CHUNK_BYTES>
__device__ inline void cp_chunk(uint8_t *dst_slot, const uint8_t *src_chunk,
                                int lane_id) {
    // Use normal load instruction with float4 (16 bytes)
    constexpr int load_bytes = 16;
    static_assert(CHUNK_BYTES % load_bytes == 0,
                  "CHUNK_BYTES must be divisible by load_bytes");
    for (int b = lane_id * load_bytes; b < CHUNK_BYTES; b += 32 * load_bytes) {

      const float4 *src_ptr = reinterpret_cast<const float4 *>(src_chunk + b);
      float4 *dst_ptr = reinterpret_cast<float4 *>(dst_slot + b);
      *dst_ptr = *src_ptr;
  }
}

template <int CHUNK_BYTES, int REPEAT, int STAGES, int NUM_PRODUCER_WARPS = 1,
          int NUM_CONSUMER_WARPS = 1>
__global__ void cp_bw_kernel(const uint8_t *__restrict__ src,
                             unsigned long long *__restrict__ sink,
                             size_t total_bytes) {
  extern __shared__ __align__(16) uint8_t smem[];

  __shared__ volatile unsigned int fwd_bar[STAGES];
  __shared__ volatile unsigned int bwd_bar[STAGES];

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      fwd_bar[s] = 1;
      bwd_bar[s] = 1;
    }
  }

  __syncthreads();

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const int total_chunks = total_bytes / CHUNK_BYTES;
  const int producer_warp_chunk_idx_start = blockIdx.x + warp_id * gridDim.x;
  const int producer_warp_chunk_idx_step = NUM_PRODUCER_WARPS * gridDim.x;
  /**
   * Calculate how many chunks this warp will process.
   */
  const int producer_stages =
      (total_chunks * REPEAT + producer_warp_chunk_idx_step - 1 -
       producer_warp_chunk_idx_start) /
      producer_warp_chunk_idx_step;

  const int consumer_warp_id = warp_id - NUM_PRODUCER_WARPS;
  const int consumer_warp_chunk_idx_start =
      blockIdx.x + consumer_warp_id * gridDim.x;
  const int consumer_warp_chunk_idx_step = NUM_CONSUMER_WARPS * gridDim.x;
  const int consumer_stages =
      (total_chunks * REPEAT + consumer_warp_chunk_idx_step - 1 -
       consumer_warp_chunk_idx_start) /
      consumer_warp_chunk_idx_step;

  // constexpr int NUM_INITIAL_STAGES = Stages / NUM_PRODUCER_WARPS;

  int sum = 0;

  static_assert(STAGES >= NUM_PRODUCER_WARPS,
                "Stages must be at least NUM_PRODUCER_WARPS");
  static_assert(STAGES % NUM_PRODUCER_WARPS == 0,
                "Stages must be divisible by NUM_PRODUCER_WARPS");

  // Multiple Producer warps: each warp handles different chunks
  if (warp_id < NUM_PRODUCER_WARPS) { // Producer warps
      // Non-TMA methods
      // Each producer warp processes chunks with stride
      for (int stage_idx = 0; stage_idx < producer_stages; stage_idx++) {
        auto real_stage_idx = stage_idx * NUM_PRODUCER_WARPS + warp_id;
        auto real_chunk_idx = stage_idx * producer_warp_chunk_idx_step +
                              producer_warp_chunk_idx_start;

        const int slot = real_stage_idx % STAGES;
        uint8_t *dst_slot = smem + slot * CHUNK_BYTES;
        const uint8_t *src_chunk =
            src + (real_chunk_idx % total_chunks) * CHUNK_BYTES;

        unsigned want_free = (real_stage_idx % (STAGES * 2)) < STAGES;
        // printf("want_free: %u, real_stage_idx: %d, slot: %d, warp_id: %d, lane_id: %d\n",
        //        want_free, real_stage_idx, slot, warp_id, lane_id);
        // printf("fwd_bar[%d]: %u, bwd_bar[%d]: %u\n", slot, fwd_bar[slot], slot, bwd_bar[slot]);
        wait(&bwd_bar[slot], want_free);

        cp_chunk<CHUNK_BYTES>(dst_slot, src_chunk, lane_id);

        __syncwarp();

        if (lane_id == 0) {
          arrive(const_cast<unsigned int*>(&fwd_bar[slot]));
        }
    }
  }

  // Consumer warp: wait for data ready, then consume and free slot
  else if (warp_id >= NUM_PRODUCER_WARPS) {

    if (lane_id == 0) { // Only one thread for simplicity

      for (int stage_idx = 0; stage_idx < consumer_stages; stage_idx++) {
        auto real_stage_idx = stage_idx * NUM_CONSUMER_WARPS + consumer_warp_id;

        const int slot = real_stage_idx % STAGES;
        uint8_t *dst_slot = smem + slot * CHUNK_BYTES;

        unsigned want_ready = (real_stage_idx % (STAGES * 2)) >= STAGES;
        wait(&fwd_bar[slot], want_ready);

        auto p32 = reinterpret_cast<const int *>(dst_slot);
        sum += *p32;

        arrive(const_cast<unsigned int*>(&bwd_bar[slot]));
      }

      sink[blockIdx.x] = sum;
    }
  }
}