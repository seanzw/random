NVCC_FLAGS = -std=c++17 -O3 -arch=compute_86 -code=compute_86,sm_86
LDFLAGS = -lcudart

bandwidth_test.out: ../src/bandwidth_test.cu ../src/cp_kernels.cuh ../src/benchmark_framework.cuh
	nvcc $(NVCC_FLAGS) -o $@ $< $(LDFLAGS)

all: bandwidth_test.out

clean:
	rm -f *.out
	rm -rf checkpoint_files
	rm -f _app_cuda*
	rm -f _cuobjdump*
	rm -f gpgpu_inst_stats.txt
	rm -f *.1.sm_86.ptx
	rm -f *.1.sm_86.ptxas