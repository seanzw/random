// Fixed B200 producer-consumer TMA bandwidth point.
// nvcc -std=c++17 -arch=sm_100a -O3 b200_tma_test.cu

#include "benchmark_framework.cuh"
#include "kernel_wrappers.cuh"

int main() {
  const size_t total_bytes = size_t(16) * 1024 * 1024; // 16 MiB

  constexpr int stages = 12;
  constexpr int chunk_bytes = 16384; // 16 KiB per TMA transaction
  constexpr int repeat = 256;
  constexpr int warmup_iters = 0;
  constexpr int num_iters = 1;

  TestData test_data(total_bytes);
  const dim3 grid(test_data.get_num_sms()); // 148 CTAs on B200
  const dim3 block(256);                    // 4 producer + 4 consumer warps

  printf("=== B200 TMA Test ===\n");
  printf("Configuration: Stages=%d, Chunk=%d B, Repeat=%d, Warmup=%d, "
         "Iters=%d\n",
         stages, chunk_bytes, repeat, warmup_iters, num_iters);

  TMAKernelWrapper<4, 4> tma_wrapper;
  BandwidthBenchmark tma_bench(test_data, grid, block, tma_wrapper,
                               "B200 TMA Test");

  tma_bench.template run_single_config<stages, chunk_bytes, repeat>(
      warmup_iters, num_iters);

  return 0;
}
