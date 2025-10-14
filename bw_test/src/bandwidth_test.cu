#include "benchmark_framework.cuh"
#include "cp_kernels.cuh"

// A simplified kernel wrapper for the normal load method
template <int NUM_PRODUCER_WARPS = 1, int NUM_CONSUMER_WARPS = 1>
struct NormalLoadKernelWrapper {
  template <int CHUNK_BYTES, int REPEAT, int STAGES>
  void set_shmem_size(size_t shmem_bytes) {
    cudaFuncSetAttribute(
        cp_bw_kernel<CHUNK_BYTES, REPEAT, STAGES, NUM_PRODUCER_WARPS, NUM_CONSUMER_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
  }

  template <int CHUNK_BYTES, int REPEAT, int STAGES>
  void launch(dim3 grid, dim3 block, size_t shmem_bytes, const uint8_t *src,
              unsigned long long *sink, size_t total_bytes) {
    cp_bw_kernel<CHUNK_BYTES, REPEAT, STAGES, NUM_PRODUCER_WARPS, NUM_CONSUMER_WARPS>
        <<<grid, block, shmem_bytes>>>(src, sink, total_bytes);
  }
};


int main() {
  const size_t total_bytes = size_t(64) * 1024 * 1024; // 64 MiB
  constexpr int repeat = 16;
  constexpr int stages = 16;

  TestData test_data(total_bytes);
  const dim3 grid(test_data.get_num_sms());
  // 8 producer warps and 8 consumer warps
  const dim3 block(512);

  printf("\n=== Normal Load Bandwidth Test (8 Producer Warps) ===\n");
  NormalLoadKernelWrapper<8, 8> normal_load_wrapper;
  BandwidthBenchmark bench(test_data, grid, block, normal_load_wrapper, "Normal load");

  // Run benchmark for different chunk sizes
  // bench.run_single_config<256, repeat>();
  // bench.run_single_config<512, repeat>();
  // bench.run_single_config<1024, repeat>();
  // bench.run_single_config<2048, repeat>();
  bench.run_single_config<4096, repeat, stages>();
  // bench.run_single_config<8192, repeat>();
  // bench.run_single_config<16384, repeat>();

  return 0;
}