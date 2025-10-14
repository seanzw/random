#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

// Error checking utility
static void check(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Test data setup
class TestData {
private:
  uint8_t *d_src;
  unsigned long long *d_sink;
  size_t total_bytes;
  int num_sms;

public:
  TestData(size_t bytes) : total_bytes(bytes) {
    // Query GPU info
    check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0),
          "get SM count");
    printf("Detected %d SMs on GPU\n", num_sms);
    // Allocate GPU memory
    check(cudaMalloc(&d_src, total_bytes), "malloc d_src");
    check(cudaMalloc(&d_sink, sizeof(unsigned long long) * num_sms),
          "malloc d_sink");
    // Initialize data
    std::vector<uint8_t> h_src(total_bytes);
    for (size_t i = 0; i < total_bytes; ++i)
      h_src[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xFF);
    check(cudaMemcpy(d_src, h_src.data(), total_bytes, cudaMemcpyHostToDevice),
          "H2D");
  }

  void realloc() {
    cudaFree(d_src);
    cudaFree(d_sink);
    check(cudaMalloc(&d_src, total_bytes), "malloc d_src");
    check(cudaMalloc(&d_sink, sizeof(unsigned long long) * num_sms),
          "malloc d_sink");
    // Initialize data
    std::vector<uint8_t> h_src(total_bytes);
    for (size_t i = 0; i < total_bytes; ++i)
      h_src[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xFF);
    check(cudaMemcpy(d_src, h_src.data(), total_bytes, cudaMemcpyHostToDevice),
          "H2D");
  }

  ~TestData() {
    cudaFree(d_src);
    cudaFree(d_sink);
  }

  const uint8_t *get_src() const { return d_src; }
  unsigned long long *get_sink() const { return d_sink; }
  size_t get_total_bytes() const { return total_bytes; }
  int get_num_sms() const { return num_sms; }
};

// Benchmark framework
template <typename KernelFunc> class BandwidthBenchmark {
protected:
  TestData &data;
  dim3 grid, block;
  KernelFunc kernel_func;
  const char *kernel_name;

public:
  BandwidthBenchmark(TestData &data, dim3 g, dim3 b, KernelFunc func,
                     const char *name)
      : data(data), grid(g), block(b), kernel_func(func), kernel_name(name) {}

  template <int CHUNK_BYTES, int REPEAT, int STAGES> void run_single_config() {
    const size_t shmem_bytes = STAGES * CHUNK_BYTES;

    int max_shmem = 0;
    cudaDeviceGetAttribute(&max_shmem,
                           cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);

    if (shmem_bytes + 4096 > max_shmem) {
      printf("Chunk=%5d | Out of SHMEM\n", CHUNK_BYTES);
      return;
    }

    kernel_func.template set_shmem_size<CHUNK_BYTES, REPEAT, STAGES>(shmem_bytes + 4096);
    data.realloc(); // Reset cache

    // Warm-up
    kernel_func.template launch<CHUNK_BYTES, REPEAT, STAGES>(
        grid, block, shmem_bytes, data.get_src(), data.get_sink(),
        data.get_total_bytes());
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

      kernel_func.template launch<CHUNK_BYTES, REPEAT, STAGES>(
          grid, block, shmem_bytes, data.get_src(), data.get_sink(),
          data.get_total_bytes());

      check(cudaGetLastError(), "timed launch");
      check(cudaEventRecord(stop), "record stop");
      check(cudaEventSynchronize(stop), "event sync");

      float ms = 0.0f;
      check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
      total_ms += ms;

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    float avg_ms = total_ms / num_iters;
    double seconds = avg_ms / 1e3;
    double gbps = (double)data.get_total_bytes() * REPEAT / seconds / 1e9;
    printf("%s | Chunk=%5d | Time=%7.3f ms | BW=%.2f GB/s\n", kernel_name,
           CHUNK_BYTES, avg_ms, gbps);
  }
};