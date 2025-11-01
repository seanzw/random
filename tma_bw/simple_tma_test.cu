// Simple single-point test for TMA
// nvcc -std=c++17 -arch=sm_120 -O3 simple_tma_test.cu

#include "benchmark_framework.cuh"
#include "kernel_wrappers.cuh"

int main() {
  const size_t total_bytes = size_t(64) * 1024 * 1024; // 64 MiB
  
  // Simple configuration
  constexpr int stages = 4;
  constexpr int chunk_bytes = 1024;
  constexpr int repeat = 8;
  constexpr int warmup_iters = 3;
  constexpr int num_iters = 5;

  TestData test_data(total_bytes);
  const dim3 grid(test_data.get_num_sms());
  const dim3 block(64); // 1 producer + 1 consumer warp

  printf("=== Simple TMA Test ===\n");
  printf("Configuration: Stages=%d, Chunk=%d B, Repeat=%d, Warmup=%d, Iters=%d\n",
         stages, chunk_bytes, repeat, warmup_iters, num_iters);

  TMAKernelWrapper<1, 1> tma_wrapper;
  BandwidthBenchmark tma_bench(test_data, grid, block, tma_wrapper,
                              "TMA Simple Test");

  // Run single configuration
  tma_bench.template run_single_config<stages, chunk_bytes, repeat>(warmup_iters, num_iters);

  return 0;
}