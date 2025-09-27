#include <cuda_runtime.h>
#include <stdio.h>

__global__ void local_mem_ptx_test(unsigned long long *out) {
  // Per-thread local variable; taking its address forces local memory
  // allocation
  unsigned long long slot =
      (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);

  // Convert the generic pointer (&slot) to a local-space address
  unsigned long long addr;
  asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(addr) : "l"(&slot));

  // Repeated local loads: LD L.64
  unsigned long long sum = 0;
#pragma unroll 256
  for (int i = 0; i < 256; ++i) {
    unsigned long long tmp;
    asm volatile("ld.local.u64 %0, [%1];" : "=l"(tmp) : "l"(addr));
    asm volatile("st.local.u64 [%0], %1;" ::"l"(addr), "l"(slot));

    sum += tmp;
  }

  // Prevent optimization; write result to global memory
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main() {
  const int N = 256;
  unsigned long long *d_out, h_out[N];
  cudaMalloc(&d_out, N * sizeof(unsigned long long));
  local_mem_ptx_test<<<1, N>>>(d_out);
  cudaMemcpy(h_out, d_out, N * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  printf("Sample output: %llu\n", (unsigned long long)h_out[0]);
  cudaFree(d_out);
  return 0;
}