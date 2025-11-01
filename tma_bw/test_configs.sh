#!/bin/bash

# Script to easily test different configurations for simple tests

echo "=== Testing Simple Configurations ==="

echo
echo "Testing Normal Load with different configurations:"
echo "Config 1: stages=2, chunk=512, repeat=4, warmup=1, iters=3"

# Create a temporary test file
cat > temp_normal_test.cu << 'EOF'
#include "benchmark_framework.cuh"
#include "cp_kernels.cuh"

template <int NUM_PRODUCER_WARPS = 1, int NUM_CONSUMER_WARPS = 1,
          CP_METHOD METHOD = CP_METHOD::CP_ASYNC>
struct CPKernelWrapper {
  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void set_shmem_size(size_t shmem_bytes) {
    cudaFuncSetAttribute(
        cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT, NUM_PRODUCER_WARPS,
                     NUM_CONSUMER_WARPS, METHOD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
  }

  template <int Stages, int CHUNK_BYTES, int REPEAT>
  void launch(dim3 grid, dim3 block, size_t shmem_bytes, const uint8_t *src,
              unsigned long long *sink, size_t total_bytes) {
    cp_bw_kernel<Stages, CHUNK_BYTES, REPEAT, NUM_PRODUCER_WARPS,
                 NUM_CONSUMER_WARPS, METHOD>
        <<<grid, block, shmem_bytes>>>(src, sink, total_bytes);
  }
};

template <int NUM_PRODUCER_WARPS = 1, int NUM_CONSUMER_WARPS = 1>
using NormalLoadKernelWrapper =
    CPKernelWrapper<NUM_PRODUCER_WARPS, NUM_CONSUMER_WARPS,
                    CP_METHOD::NORMAL_LOAD>;

int main() {
  const size_t total_bytes = size_t(32) * 1024 * 1024; // 32 MiB
  
  constexpr int stages = 2;
  constexpr int chunk_bytes = 512;
  constexpr int repeat = 4;
  constexpr int warmup_iters = 1;
  constexpr int num_iters = 3;

  TestData test_data(total_bytes);
  const dim3 grid(test_data.get_num_sms());
  const dim3 block(64);

  printf("Custom Configuration Test\n");
  printf("Configuration: Stages=%d, Chunk=%d B, Repeat=%d, Warmup=%d, Iters=%d\n",
         stages, chunk_bytes, repeat, warmup_iters, num_iters);

  NormalLoadKernelWrapper<1, 1> normal_load_wrapper;
  BandwidthBenchmark normal_load_bench(test_data, grid, block, normal_load_wrapper,
                                      "Custom Normal Load Test");

  normal_load_bench.template run_single_config<stages, chunk_bytes, repeat>(warmup_iters, num_iters);

  return 0;
}
EOF

# Compile and run
nvcc -std=c++17 -arch=sm_120 -O3 -lcudart temp_normal_test.cu -o temp_normal_test.out
./temp_normal_test.out

# Clean up
rm -f temp_normal_test.cu temp_normal_test.out

echo
echo "=== Simple test script completed ==="