NVCC_FLAGS = -std=c++17 -arch=sm_120 -O3


bandwidth_test.out: ../src/bandwidth_test.cu ../src/cp_kernels.cuh ../src/benchmark_framework.cuh
	nvcc $(NVCC_FLAGS) ../src/bandwidth_test.cu -o $@

all: bandwidth_test.out

clean:
	rm -f *.out
	rm -f *.ncu-rep