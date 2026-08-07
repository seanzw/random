# Bandwidth Comparison: TMA vs cp.async vs Normal Load

This directory contains a unified bandwidth test comparing Hopper's TMA (Tensor Memory Accelerator) bulk operations with traditional cp.async instructions and normal memory loads, all implemented in a single unified kernel.

## Files

### Core Implementation
- `kernels_common.cuh` - Shared PTX helpers and constants (including TMA bulk copy)
- `cp_kernels.cuh` - Unified kernel supporting TMA, cp.async, and normal load methods
- `benchmark_framework.cuh` - Unified benchmarking framework
- `bandwidth_test.cu` - Main unified test driver

### Build
- `Makefile` - Build configuration for Blackwell (`sm_120`) and split H200
  L2/HBM binaries (`sm_90a`)

## Architecture

The code uses a unified kernel design with compile-time method selection:

```
kernels_common.cuh          # Shared mbarrier operations, constants, TMA bulk copy
└── cp_kernels.cuh          # Unified kernel with CP_METHOD enum:
                            #   - NORMAL_LOAD: float4 loads
                            #   - CP_ASYNC: cp.async instructions  
                            #   - TMA: bulk TMA operations

benchmark_framework.cuh     # Unified timing and memory management
└── bandwidth_test.cu       # Main driver using unified kernel
```

## Key Features

### Unified Kernel Implementation (`cp_bw_kernel`)
The unified kernel supports three copy methods via the `CP_METHOD` enum:

#### TMA Method (`CP_METHOD::TMA`)
- Uses `cp.async.bulk` instructions with mbarrier integration
- Single producer thread per warp (lane 0) issues bulk transfers
- Hardware-accelerated bulk transfers with automatic completion
- Uses `mbarrier_arrive_expect_tx` for proper synchronization
- Requires SM_90+ (Hopper architecture)

#### cp.async Method (`CP_METHOD::CP_ASYNC`)
- Uses traditional `cp.async.cg` instructions (16B per thread)
- All threads in producer warp participate in copying
- Manual work distribution across warp threads
- Explicit `cp.async.wait_group` synchronization
- Compatible with SM_80+ (Ampere and newer)

#### Normal Load Method (`CP_METHOD::NORMAL_LOAD`)
- Uses simple `float4` load instructions (16B per thread)
- All threads in producer warp participate in copying
- Standard memory hierarchy (L1/L2 cache)
- Compatible with all CUDA architectures

### Multi-Producer Support
- Configurable number of producer warps (1, 2, 4, 8, 16)
- Each method works with multiple producer warps
- Single consumer warp for all configurations
- Double-buffering pipeline with configurable stages

## Usage

```bash
# Build the test
make

# Run the test
./bandwidth_test.out

# Clean build artifacts
make clean
```

### H200: separate L2 and HBM measurements

The original `64 MiB x 16 repeats` configuration mixes cache-resident and HBM
traffic. Do not interpret its effective bandwidth as pure HBM bandwidth. Build
the two cache-capacity regimes separately:

```bash
module load cuda/12.8
make h200
```

This produces:

- `bandwidth_test_h200_l2.out`: 40 MiB working set, one warm-up launch, 32
  repeats, and `-dlcm=cg` to bypass L1. The tested H200 NVL reports 60 MiB L2.
- `bandwidth_test_h200_hbm.out`: each timed launch first reads an independent
  256 MiB eviction buffer outside the timed interval, then streams once through
  a 1 GiB working set (`repeat=1`).

Run both binaries inside an H200 allocation:

```bash
./bandwidth_test_h200_l2.out | tee bw_result_h200_l2.log
./bandwidth_test_h200_hbm.out | tee bw_result_h200_hbm.log

python process.py \
  -i bw_result_h200_l2.log \
  -o bw_result_h200_l2.png \
  --bandwidth-only \
  --title "NVIDIA H200 NVL - L2-resident"

python process.py \
  -i bw_result_h200_hbm.log \
  -o bw_result_h200_hbm.png \
  --peak-bw 4.8 \
  --title "NVIDIA H200 NVL - cold HBM stream"
```

### H200 NVL results

These results were collected on an NVIDIA H200 NVL with CUDA 12.8 in Slurm job
`2035527`. Each regime contains 375 valid measurements plus 45 configurations
that exceed shared-memory capacity, with no correctness or CUDA runtime errors.

| Method | L2-resident peak | Cold-HBM peak | HBM peak reference |
| --- | ---: | ---: | ---: |
| TMA | 7786.99 GB/s | 4287.34 GB/s | 89.3% |
| cp.async | 7748.72 GB/s | 4277.17 GB/s | 89.1% |
| Normal load | 7157.76 GB/s | 4138.69 GB/s | 86.2% |

- [Complete H200 L2 heatmap](bw_result_h200_l2.png)
- [Complete H200 cold-HBM heatmap](bw_result_h200_hbm.png)

The HBM percentages use NVIDIA's published
[4.8 TB/s H200 memory-bandwidth specification](https://www.nvidia.com/en-us/data-center/h200/).
The L2 plot intentionally reports bandwidth only because there is no equivalent
product-level L2 peak-bandwidth specification. Nsight Compute counter validation
was unavailable on the test cluster (`ERR_NVGPUCTRPERM`), so the two regimes are
separated through cache policy, capacity, warm-up, explicit eviction, and access
pattern rather than a reported hardware hit-rate counter.

## GPU Compatibility

- **Default Architecture**: sm_120 (configurable in Makefile)
- **H200 Architecture**: sm_90a (`make h200`)
- **TMA Features**: Requires Hopper+ (sm_90+)
- **cp.async Features**: Requires Ampere+ (sm_80+)

## Output

The test measures bandwidth (GB/s) for different:
- Copy methods (TMA, cp.async, normal load)
- Pipeline stages (1-32, powers of 2; minimum stage count grows with producer
  count)
- Chunk sizes (256B - 16KB)
- Producer warp counts (1, 2, 4, 8, 16)

Results show the relative performance characteristics of each memory copy
method across different configurations, with TMA, cp.async, and synchronous
loads implemented for direct comparison.
