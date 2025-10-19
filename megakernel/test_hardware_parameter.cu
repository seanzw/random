#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("\nDevice %d: \"%s\"\n", i, deviceProp.name);
        printf("  Total shared memory per block:         %zu bytes\n", 
               deviceProp.sharedMemPerBlock);
        printf("  Shared memory per multiprocessor:      %zu bytes\n", 
               deviceProp.sharedMemPerMultiprocessor);
        printf("  Max shared memory per block optin:     %zu bytes\n", 
               deviceProp.sharedMemPerBlockOptin);
        
        printf("  Max threads per block:                 %d\n", 
               deviceProp.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor:        %d\n", 
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Number of multiprocessors:             %d\n", 
               deviceProp.multiProcessorCount);
    }

    return 0;
}