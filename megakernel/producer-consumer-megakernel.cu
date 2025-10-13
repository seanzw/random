#include <stdio.h>
#include <cuda.h>
#include <cuda/barrier>

struct semaphore {
private:
    uint64_t value;
};

struct semaphores_t {
    semaphore A_sem;
    semaphore B_sem;
};

__device__ static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count=0) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}

__device__ static inline void invalidate_semaphore(semaphore& bar) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    asm volatile (
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

__device__ static inline void arrive(semaphore& sem) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :
        : "r"(mbar_ptr)
        : "memory"
    );
}

__device__ static inline void arrive(semaphore& sem, uint32_t count) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

__device__ static inline void wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures to save instruction issue slots
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ void Producer(semaphores_t& sem){
    wait(sem.A_sem, true);
    printf("Produce 1\n");
    arrive(sem.B_sem);

    wait(sem.A_sem, false);
    printf("Produce 2\n");
    arrive(sem.B_sem);

    wait(sem.A_sem, true);
    printf("Produce 3\n");
    arrive(sem.B_sem);

    wait(sem.A_sem, false);
    printf("Produce 4\n");
    arrive(sem.B_sem);
}

__device__ void Consumer(semaphores_t& sem){
    wait(sem.B_sem, false);
    printf("Consume 1\n");
    arrive(sem.A_sem);

    wait(sem.B_sem, true);
    printf("Consume 2\n");
    arrive(sem.A_sem);

    wait(sem.B_sem, false);
    printf("Consume 3\n");
    arrive(sem.A_sem);

    wait(sem.B_sem, true);
    printf("Consume 4\n");
    arrive(sem.A_sem);
}

__device__ void Producer(semaphores_t& sem, int iters){
    for (int i = 0; i < iters; ++i) {
        int parity = (i & 1) == 0 ? 1 : 0;
        wait(sem.A_sem, parity);
        arrive(sem.B_sem);
    }
}

__device__ void Consumer(semaphores_t& sem, int iters){
    for (int i = 0; i < iters; ++i) {
        int parity = (i & 1) == 0 ? 0 : 1;
        wait(sem.B_sem, parity);
        arrive(sem.A_sem);
    }
}

__global__ void Kernel(int iters){
    __shared__ semaphores_t sem;
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0){
        init_semaphore(sem.A_sem, 1);
        init_semaphore(sem.B_sem, 1);
    }
    __syncthreads();
    if(tid == 0){
        Producer(sem, iters);
    }
    if(tid == 32){
        Consumer(sem, iters);
    }
    __syncthreads();
    if(tid == 0){
        invalidate_semaphore(sem.A_sem);
        invalidate_semaphore(sem.B_sem);  
    }
}

int main() {
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Kernel<<<1, 64>>>(1);
    cudaDeviceSynchronize();


    cudaEventRecord(start);
    Kernel<<<1, 64>>>(10000000);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("MegaKernel version elapsed time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}