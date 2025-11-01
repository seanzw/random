// Simple single-point test for cp.async
// nvcc -std=c++17 -arch=sm_120 -O3 simple_cp_async_test.cu

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

  printf("=== Simple cp.async Test ===\n");
  printf("Configuration: Stages=%d, Chunk=%d B, Repeat=%d, Warmup=%d, Iters=%d\n",
         stages, chunk_bytes, repeat, warmup_iters, num_iters);

  CPAsyncKernelWrapper<1, 1> cp_async_wrapper;
  BandwidthBenchmark cp_async_bench(test_data, grid, block, cp_async_wrapper,
                                   "cp.async Simple Test");

  // Run single configuration
  cp_async_bench.template run_single_config<stages, chunk_bytes, repeat>(warmup_iters, num_iters);

  return 0;
}