
// nvcc -std=c++17 -arch=sm_90 -O3 tma_bw_test.cu
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

constexpr int PRODUCER_THREAD = 0;
constexpr int CONSUMER_THREAD = 32;

// ---- Hopper PTX helpers ----
__device__ inline void mbarrier_init(unsigned long long *bar_addr,
                                     unsigned expected_arrivals) {

  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_addr));

  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(expected_arrivals));
}

__device__ inline void mbarrier_arrive(unsigned long long *bar_addr) {
  unsigned long long bar_s;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(bar_s) : "l"(bar_addr));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"l"(bar_s));
}

__device__ inline void mbarrier_arrive_expect_tx(unsigned long long *bar_addr,
                                                 unsigned expected_tx) {
  void const *const ptr = bar_addr;
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
      "r"(expected_tx));
}

__device__ inline void wait(unsigned long long *bar_addr,
                            unsigned expected_parity) {
  void const *const ptr = bar_addr;
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(expected_parity));
}

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

// ---- Kernel: TMA bandwidth test with variable pipeline stages ----
template <int Stages, int CHUNK_BYTES>
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

  //   printf("Th %d inited\n", threadIdx.x);
  const size_t total_chunks = total_bytes / CHUNK_BYTES;
  // Adjust the start.
  int sum = 0;

  // Producer: wait for slot to be free, then issue copy
  if (threadIdx.x == PRODUCER_THREAD) {
    for (size_t i = blockIdx.x, stage_idx = 0; i < total_chunks;
         i += gridDim.x, stage_idx++) {
      const int slot = int(stage_idx % Stages);
      uint8_t *dst_slot = smem + slot * CHUNK_BYTES;
      const uint8_t *src_chunk = src + i * CHUNK_BYTES;

      unsigned want_free = (stage_idx % (Stages * 2)) < Stages;
      wait(&bwd_bar[slot], want_free);

      mbarrier_arrive_expect_tx(&fwd_bar[slot], CHUNK_BYTES);
      cp_async_bulk<CHUNK_BYTES>(dst_slot, src_chunk, &fwd_bar[slot]);
    }
  }

  // Consumer: wait for data ready, then consume and free slot
  else if (threadIdx.x == CONSUMER_THREAD) {
    for (size_t i = blockIdx.x, stage_idx = 0; i < total_chunks;
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

static void check(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

template <int Stages, int CHUNK_BYTES>
void run_tma_bw(const uint8_t *d_src, unsigned long long *d_sink,
                size_t total_bytes, dim3 grid, dim3 block) {
  const size_t shmem_bytes = Stages * CHUNK_BYTES;

  int max_shmem = 0;
  cudaDeviceGetAttribute(&max_shmem,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  // printf("Max shared memory per block: %d bytes\n", max_shmem);

  if (shmem_bytes + 4096 > max_shmem) {
    printf("Stages=%2d | Chunk=%4d | Out of SHMEM\n", Stages, CHUNK_BYTES);
    return;
  }

  cudaFuncSetAttribute(tma_bw_kernel<Stages, CHUNK_BYTES>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shmem_bytes + 4096 // + 4kB for safety
  );

  // Warm-up
  tma_bw_kernel<Stages, CHUNK_BYTES>
      <<<grid, block, shmem_bytes>>>(d_src, d_sink, total_bytes);
  check(cudaGetLastError(), "warm-up launch");
  check(cudaDeviceSynchronize(), "warm-up sync");

  // Timed run
  const int num_iters = 10;
  float total_ms = 0.0f;
  cudaEvent_t start, stop;
  for (int iter = 0; iter < num_iters; ++iter) {
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    check(cudaEventRecord(start), "record start");
    tma_bw_kernel<Stages, CHUNK_BYTES>
        <<<grid, block, shmem_bytes>>>(d_src, d_sink, total_bytes);
    check(cudaGetLastError(), "timed launch");
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    total_ms += ms;
  }
  float avg_ms = total_ms / num_iters;
  double seconds = avg_ms / 1e3;
  double gbps = (double)total_bytes / seconds / 1e9;
  printf("Stages=%2d | Chunk=%4d | Time=%.3f ms | BW=%.2f GB/s\n", Stages,
         CHUNK_BYTES, avg_ms, gbps);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <int CHUNK_BYTES>
void run_all_stages(const uint8_t *d_src, unsigned long long *d_sink,
                    size_t total_bytes, dim3 grid, dim3 block) {
  printf("Hopper TMA bulk + forward/backward mbarrier | CHUNK=%d B | "
         "total=%zu B\n",
         CHUNK_BYTES, total_bytes);
  run_tma_bw<1, CHUNK_BYTES>(d_src, d_sink, total_bytes, grid, block);
  run_tma_bw<2, CHUNK_BYTES>(d_src, d_sink, total_bytes, grid, block);
  run_tma_bw<4, CHUNK_BYTES>(d_src, d_sink, total_bytes, grid, block);
  run_tma_bw<8, CHUNK_BYTES>(d_src, d_sink, total_bytes, grid, block);
  run_tma_bw<16, CHUNK_BYTES>(d_src, d_sink, total_bytes, grid, block);
  run_tma_bw<32, CHUNK_BYTES>(d_src, d_sink, total_bytes, grid, block);
}

int main() {
  const size_t total_bytes = size_t(64) * 1024 * 1024; // 64 MiB

  // Query GPU for maximum number of SMs
  int num_sms = 0;
  check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0),
        "get SM count");
  printf("Detected %d SMs on GPU\n", num_sms);

  uint8_t *d_src = nullptr;
  unsigned long long *d_sink = nullptr;
  check(cudaMalloc(&d_src, total_bytes), "malloc d_src");
  check(cudaMalloc(&d_sink, sizeof(unsigned long long)), "malloc d_sink");

  std::vector<uint8_t> h_src(total_bytes);
  for (size_t i = 0; i < total_bytes; ++i)
    h_src[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xFF);
  check(cudaMemcpy(d_src, h_src.data(), total_bytes, cudaMemcpyHostToDevice),
        "H2D");

  const dim3 grid(num_sms);
  const dim3 block(64); // thread 0 producer, thread 32 consumer

  run_all_stages<256>(d_src, d_sink, total_bytes, grid, block);
  run_all_stages<512>(d_src, d_sink, total_bytes, grid, block);
  run_all_stages<1024>(d_src, d_sink, total_bytes, grid, block);
  run_all_stages<2048>(d_src, d_sink, total_bytes, grid, block);
  run_all_stages<4096>(d_src, d_sink, total_bytes, grid, block);
  run_all_stages<8192>(d_src, d_sink, total_bytes, grid, block);
  run_all_stages<16384>(d_src, d_sink, total_bytes, grid, block);

  cudaFree(d_src);
  cudaFree(d_sink);
  return 0;
}