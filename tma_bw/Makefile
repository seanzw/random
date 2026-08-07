NVCC_FLAGS = -std=c++17 -arch=sm_120 -O3 -lcudart
H200_NVCC_FLAGS ?= -std=c++17 -arch=sm_90a -O3 -lcudart -Xptxas=-dlcm=cg

bandwidth_test.out: bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(NVCC_FLAGS) bandwidth_test.cu -o $@

# Hopper binaries with disjoint cache-capacity regimes. The 40 MiB L2 working
# set fits in H200 L2 but exceeds per-SM L1 capacity; -dlcm=cg bypasses L1 for
# compiler-generated normal loads. The 1 GiB HBM stream is much larger than L2
# and is traversed only once per kernel launch.
bandwidth_test_h200_l2.out: bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(H200_NVCC_FLAGS) -DBENCH_L2_MODE=1 -DBENCH_FORCE_CG=1 -DBENCH_TOTAL_MIB=40 -DBENCH_REPEAT=32 bandwidth_test.cu -o $@

bandwidth_test_h200_hbm.out: bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(H200_NVCC_FLAGS) -DBENCH_HBM_MODE=1 -DBENCH_FORCE_CG=1 -DBENCH_TOTAL_MIB=1024 -DBENCH_REPEAT=1 -DBENCH_REALLOCATE_EACH_CONFIG=0 -DBENCH_FLUSH_L2_MIB=256 bandwidth_test.cu -o $@

h200-l2: bandwidth_test_h200_l2.out

h200-hbm: bandwidth_test_h200_hbm.out

h200: h200-l2 h200-hbm

# Isolated test targets
tma_test.out: bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(NVCC_FLAGS) -DONLY_TMA bandwidth_test.cu -o $@

cp_async_test.out: bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(NVCC_FLAGS) -DONLY_CP_ASYNC bandwidth_test.cu -o $@

normal_load_test.out: bandwidth_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(NVCC_FLAGS) -DONLY_NORMAL_LOAD bandwidth_test.cu -o $@

# Simple single-point test targets
simple_normal_load_test.out: simple_normal_load_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(NVCC_FLAGS) simple_normal_load_test.cu -o $@

simple_tma_test.out: simple_tma_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(NVCC_FLAGS) simple_tma_test.cu -o $@

simple_cp_async_test.out: simple_cp_async_test.cu kernels_common.cuh cp_kernels.cuh benchmark_framework.cuh kernel_wrappers.cuh
	nvcc $(NVCC_FLAGS) simple_cp_async_test.cu -o $@

all: bandwidth_test.out tma_test.out cp_async_test.out normal_load_test.out simple_normal_load_test.out simple_tma_test.out simple_cp_async_test.out

# PTX extraction targets
bandwidth_test.ptx: bandwidth_test.out
	cuobjdump --dump-ptx $< > $@

tma_test.ptx: tma_test.out
	cuobjdump --dump-ptx $< > $@

cp_async_test.ptx: cp_async_test.out
	cuobjdump --dump-ptx $< > $@

normal_load_test.ptx: normal_load_test.out
	cuobjdump --dump-ptx $< > $@

simple_normal_load_test.ptx: simple_normal_load_test.out
	cuobjdump --dump-ptx $< > $@

simple_tma_test.ptx: simple_tma_test.out
	cuobjdump --dump-ptx $< > $@

simple_cp_async_test.ptx: simple_cp_async_test.out
	cuobjdump --dump-ptx $< > $@

# Extract PTX from all binaries
ptx: bandwidth_test.ptx tma_test.ptx cp_async_test.ptx normal_load_test.ptx simple_normal_load_test.ptx simple_tma_test.ptx simple_cp_async_test.ptx

# Convenience targets for running isolated tests
run-tma: tma_test.out
	./tma_test.out

run-cp-async: cp_async_test.out
	./cp_async_test.out

run-normal-load: normal_load_test.out
	./normal_load_test.out

# Convenience targets for running simple single-point tests
run-simple-normal-load: simple_normal_load_test.out
	./simple_normal_load_test.out

run-simple-tma: simple_tma_test.out
	./simple_tma_test.out

run-simple-cp-async: simple_cp_async_test.out
	./simple_cp_async_test.out

run-all: bandwidth_test.out
	./bandwidth_test.out

# Run all simple tests
run-simple-all: simple_normal_load_test.out simple_tma_test.out simple_cp_async_test.out
	@echo "=== Running all simple tests ==="
	./simple_normal_load_test.out
	./simple_tma_test.out
	./simple_cp_async_test.out

clean:
	rm -f *.out *.ptx

.PHONY: all h200 h200-l2 h200-hbm ptx run-tma run-cp-async run-normal-load run-simple-normal-load run-simple-tma run-simple-cp-async run-all run-simple-all clean
