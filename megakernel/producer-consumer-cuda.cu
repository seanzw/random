#include <stdio.h>
#include <cuda.h>
#include <cuda/barrier>

struct semaphores_t {
    cuda::barrier<cuda::thread_scope_block> A_sem;
    cuda::barrier<cuda::thread_scope_block> B_sem;
};

__device__ void Producer(semaphores_t& sem) {
    sem.A_sem.wait_parity(true);
    printf("Produce 1\n");
    sem.B_sem.arrive();

    sem.A_sem.wait_parity(false);
    printf("Produce 2\n");
    sem.B_sem.arrive();

    sem.A_sem.wait_parity(true);
    printf("Produce 3\n");
    sem.B_sem.arrive();

    sem.A_sem.wait_parity(false);
    printf("Produce 4\n");
    sem.B_sem.arrive();
}

__device__ void Consumer(semaphores_t& sem) {
    sem.B_sem.wait_parity(false);
    printf("Consume 1\n");
    sem.A_sem.arrive();

    sem.B_sem.wait_parity(true);
    printf("Consume 2\n");
    sem.A_sem.arrive();

    sem.B_sem.wait_parity(false);
    printf("Consume 3\n");
    sem.A_sem.arrive();

    sem.B_sem.wait_parity(true);
    printf("Consume 4\n");
    sem.A_sem.arrive();
}

__device__ void Producer(semaphores_t& sem, int iters){
    for (int i = 0; i < iters; ++i) {
        int parity = (i & 1) == 0 ? 1 : 0;
        sem.A_sem.wait_parity(parity);
        sem.B_sem.arrive();
    }
}

__device__ void Consumer(semaphores_t& sem, int iters){
    for (int i = 0; i < iters; ++i) {
        int parity = (i & 1) == 0 ? 0 : 1;
        sem.B_sem.wait_parity(parity);
        sem.A_sem.arrive();
    }
}

__global__ void Kernel(int iters) {
    __shared__ semaphores_t sem;

    if (threadIdx.x == 0) {
        init(&sem.A_sem, 1);
        init(&sem.B_sem, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        Producer(sem, iters);
    }
    if (threadIdx.x == 32) {
        Consumer(sem, iters);
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Kernel<<<1, 64>>>(1);
    cudaDeviceSynchronize();


    cudaEventRecord(start);
    Kernel<<<1, 64>>>(10000000);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("CUDA version elapsed time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
