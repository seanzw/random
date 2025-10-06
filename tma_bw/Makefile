NVCC_FLAGS = -std=c++17 -arch=sm_120 -O3

bandwidth_test.out: bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh
	nvcc $(NVCC_FLAGS) bandwidth_test.cu -o $@

all: bandwidth_test.out

clean:
	rm -f *.out
