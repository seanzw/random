// Simple single-point test for Normal Load
// nvcc -std=c++17 -arch=sm_120 -O3 simple_normal_load_test.cu

#include "benchmark_framework.cuh"
#include "kernel_wrappers.cuh"

int main() {
  const size_t total_bytes = size_t(64) * 1024 * 1024; // 64 MiB

  // Simple configuration
  constexpr int stages = 4;
  constexpr int chunk_bytes = 1024;
  constexpr int repeat = 1;
  constexpr int warmup_iters = 0;
  constexpr int num_iters = 1;

  TestData test_data(total_bytes);
  const dim3 grid(test_data.get_num_sms());
  const dim3 block(64); // 1 producer + 1 consumer warp

  printf("=== Simple Normal Load Test ===\n");
  printf(
      "Configuration: Stages=%d, Chunk=%d B, Repeat=%d, Warmup=%d, Iters=%d\n",
      stages, chunk_bytes, repeat, warmup_iters, num_iters);

  NormalLoadKernelWrapper<1, 1> normal_load_wrapper;
  BandwidthBenchmark normal_load_bench(
      test_data, grid, block, normal_load_wrapper, "Normal Load Simple Test");

  // Run single configuration
  normal_load_bench.template run_single_config<stages, chunk_bytes, repeat>(
      warmup_iters, num_iters);

  return 0;
}