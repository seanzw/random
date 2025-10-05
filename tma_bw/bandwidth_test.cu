// nvcc -std=c++17 -arch=sm_120 -O3 bandwidth_test.cu
#include "benchmark_framework.cuh"
#include "cp_kernels.cuh"
#include "tma_kernels.cuh"

// Kernel wrapper for TMA
struct TMAKernelWrapper {
  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void set_shmem_size(size_t shmem_bytes) {
    cudaFuncSetAttribute(tma_bw_kernel<Stages, CHUNK_BYTES, REPEAT>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_bytes);
  }

  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void launch(dim3 grid, dim3 block, size_t shmem_bytes, const uint8_t *src,
              unsigned long long *sink, size_t total_bytes) {
    tma_bw_kernel<Stages, CHUNK_BYTES, REPEAT>
        <<<grid, block, shmem_bytes>>>(src, sink, total_bytes);
  }
};

// Kernel wrapper for cp with configurable producer warps
template <int NUM_PRODUCER_WARPS = 1, bool USE_CP_ASYNC = true>
struct CPKernelWrapper {
  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void set_shmem_size(size_t shmem_bytes) {
    cudaFuncSetAttribute(cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT,
                                      NUM_PRODUCER_WARPS, USE_CP_ASYNC>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_bytes);
  }

  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void launch(dim3 grid, dim3 block, size_t shmem_bytes, const uint8_t *src,
              unsigned long long *sink, size_t total_bytes) {
    cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT, NUM_PRODUCER_WARPS, USE_CP_ASYNC>
        <<<grid, block, shmem_bytes>>>(src, sink, total_bytes);
  }
};

// Kernel wrapper for normal load (using cp kernel with USE_CP_ASYNC=false)
template <int NUM_PRODUCER_WARPS = 1>
using NormalLoadKernelWrapper = CPKernelWrapper<NUM_PRODUCER_WARPS, false>;

int main() {
  const size_t total_bytes = size_t(64) * 1024 * 1024; // 64 MiB
  constexpr int repeat = 16;

  TestData test_data(total_bytes);
  const dim3 grid(test_data.get_num_sms());

  // TMA benchmarks (64 threads: 2 warps)
  const dim3 tma_block(64);
  printf("\n=== TMA Bandwidth Test ===\n");
  TMAKernelWrapper tma_wrapper;
  BandwidthBenchmark tma_bench(test_data, grid, tma_block, tma_wrapper,
                               "Hopper TMA bulk + forward/backward mbarrier");

  tma_bench.run_all_stages<256, repeat>();
  tma_bench.run_all_stages<512, repeat>();
  tma_bench.run_all_stages<1024, repeat>();
  tma_bench.run_all_stages<2048, repeat>();
  tma_bench.run_all_stages<4096, repeat>();
  tma_bench.run_all_stages<8192, repeat>();
  tma_bench.run_all_stages<16384, repeat>();

  // cp.async benchmarks with 1 producer warp (64 threads: 2 warps)
  const dim3 cp_async_1_block(64);
  printf("\n=== cp.async Bandwidth Test (1 Producer Warp) ===\n");
  CPKernelWrapper<1, true> cp_async_wrapper_1;
  BandwidthBenchmark cp_async_bench_1(test_data, grid, cp_async_1_block,
                                      cp_async_wrapper_1,
                                      "cp.async + 1 producer warp");

  cp_async_bench_1.run_all_stages<256, repeat>();
  cp_async_bench_1.run_all_stages<512, repeat>();
  cp_async_bench_1.run_all_stages<1024, repeat>();
  cp_async_bench_1.run_all_stages<2048, repeat>();
  cp_async_bench_1.run_all_stages<4096, repeat>();
  cp_async_bench_1.run_all_stages<8192, repeat>();
  cp_async_bench_1.run_all_stages<16384, repeat>();

  // cp.async benchmarks with 2 producer warps (96 threads: 3 warps)
  const dim3 cp_async_2_block(96);
  printf("\n=== cp.async Bandwidth Test (2 Producer Warps) ===\n");
  CPKernelWrapper<2, true> cp_async_wrapper_2;
  BandwidthBenchmark cp_async_bench_2(test_data, grid, cp_async_2_block,
                                      cp_async_wrapper_2,
                                      "cp.async + 2 producer warps");

  using power_of_two_sequence_2 = power_of_two_sequence<2, 32>;
  cp_async_bench_2.run_all_stages<256, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<512, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<1024, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<2048, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<4096, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<8192, repeat, power_of_two_sequence_2>();
  cp_async_bench_2.run_all_stages<16384, repeat, power_of_two_sequence_2>();

  // cp.async benchmarks with 4 producer warps (160 threads: 5 warps)
  const dim3 cp_async_4_block(160);
  using power_of_two_sequence_4 = power_of_two_sequence<4, 32>;
  printf("\n=== cp.async Bandwidth Test (4 Producer Warps) ===\n");
  CPKernelWrapper<4, true> cp_async_wrapper_4;
  BandwidthBenchmark cp_async_bench_4(test_data, grid, cp_async_4_block,
                                      cp_async_wrapper_4,
                                      "cp.async + 4 producer warps");

  cp_async_bench_4.run_all_stages<256, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<512, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<1024, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<2048, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<4096, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<8192, repeat, power_of_two_sequence_4>();
  cp_async_bench_4.run_all_stages<16384, repeat, power_of_two_sequence_4>();

  // Normal load benchmarks with 1 producer warp (64 threads: 2 warps)
  const dim3 normal_load_1_block(64);
  printf("\n=== Normal Load Bandwidth Test (1 Producer Warp) ===\n");
  NormalLoadKernelWrapper<1> normal_load_wrapper_1;
  BandwidthBenchmark normal_load_bench_1(test_data, grid, normal_load_1_block,
                                         normal_load_wrapper_1,
                                         "Normal load + 1 producer warp");

  normal_load_bench_1.run_all_stages<256, repeat>();
  normal_load_bench_1.run_all_stages<512, repeat>();
  normal_load_bench_1.run_all_stages<1024, repeat>();
  normal_load_bench_1.run_all_stages<2048, repeat>();
  normal_load_bench_1.run_all_stages<4096, repeat>();
  normal_load_bench_1.run_all_stages<8192, repeat>();
  normal_load_bench_1.run_all_stages<16384, repeat>();

  // Normal load benchmarks with 2 producer warps (96 threads: 3 warps)
  const dim3 normal_load_2_block(96);
  printf("\n=== Normal Load Bandwidth Test (2 Producer Warps) ===\n");
  NormalLoadKernelWrapper<2> normal_load_wrapper_2;
  BandwidthBenchmark normal_load_bench_2(test_data, grid, normal_load_2_block,
                                         normal_load_wrapper_2,
                                         "Normal load + 2 producer warps");

  normal_load_bench_2.run_all_stages<256, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<512, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<1024, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<2048, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<4096, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<8192, repeat, power_of_two_sequence_2>();
  normal_load_bench_2.run_all_stages<16384, repeat, power_of_two_sequence_2>();

  // Normal load benchmarks with 4 producer warps (160 threads: 5 warps)
  const dim3 normal_load_4_block(160);
  printf("\n=== Normal Load Bandwidth Test (4 Producer Warps) ===\n");
  NormalLoadKernelWrapper<4> normal_load_wrapper_4;
  BandwidthBenchmark normal_load_bench_4(test_data, grid, normal_load_4_block,
                                         normal_load_wrapper_4,
                                         "Normal load + 4 producer warps");

  normal_load_bench_4.run_all_stages<256, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<512, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<1024, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<2048, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<4096, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<8192, repeat, power_of_two_sequence_4>();
  normal_load_bench_4.run_all_stages<16384, repeat, power_of_two_sequence_4>();

  // Normal load benchmarks with 8 producer warps (288 threads: 9 warps)
  const dim3 normal_load_8_block(288);
  using power_of_two_sequence_8 = power_of_two_sequence<8, 32>;
  printf("\n=== Normal Load Bandwidth Test (8 Producer Warps) ===\n");
  NormalLoadKernelWrapper<8> normal_load_wrapper_8;
  BandwidthBenchmark normal_load_bench_8(test_data, grid, normal_load_8_block,
                                         normal_load_wrapper_8,
                                         "Normal load + 8 producer warps");

  normal_load_bench_8.run_all_stages<256, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<512, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<1024, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<2048, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<4096, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<8192, repeat, power_of_two_sequence_8>();
  normal_load_bench_8.run_all_stages<16384, repeat, power_of_two_sequence_8>();

  // Normal load benchmarks with 16 producer warps (544 threads: 17 warps)
  const dim3 normal_load_16_block(544);
  using power_of_two_sequence_16 = power_of_two_sequence<16, 32>;
  printf("\n=== Normal Load Bandwidth Test (16 Producer Warps) ===\n");
  NormalLoadKernelWrapper<16> normal_load_wrapper_16;
  BandwidthBenchmark normal_load_bench_16(test_data, grid, normal_load_16_block,
                                          normal_load_wrapper_16,
                                          "Normal load + 16 producer warps");

  normal_load_bench_16.run_all_stages<256, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<512, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<1024, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<2048, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<4096, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<8192, repeat, power_of_two_sequence_16>();
  normal_load_bench_16.run_all_stages<16384, repeat, power_of_two_sequence_16>();

  return 0;
}