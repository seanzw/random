NVCC_FLAGS ?= -std=c++17 -arch=sm_90a -O3 -lcudart -Xptxas=-dlcm=cg

dsm_h200.out: benchmark.cu kernels.cuh common.cuh
	nvcc $(NVCC_FLAGS) -lineinfo -Xptxas=-v benchmark.cu -o $@

dsm_h200.sass: dsm_h200.out
	cuobjdump --dump-sass $< > $@

clean:
	rm -f dsm_h200.out dsm_h200.sass

.PHONY: clean
