local_mem_write_policy.out: test.cu
	nvcc -arch=sm_120 $< -O3 -std=c++17 -o $@
