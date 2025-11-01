# Bandwidth Comparison: TMA vs cp.async vs Normal Load

This directory contains a unified bandwidth test comparing Hopper's TMA (Tensor Memory Accelerator) bulk operations with traditional cp.async instructions and normal memory loads, all implemented in a single unified kernel.

## Files

### Core Implementation
- `kernels_common.cuh` - Shared PTX helpers and constants (including TMA bulk copy)
- `cp_kernels.cuh` - Unified kernel supporting TMA, cp.async, and normal load methods
- `benchmark_framework.cuh` - Unified benchmarking framework
- `bandwidth_test.cu` - Main unified test driver

### Build
- `Makefile` - Build configuration (targets sm_120)

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

## GPU Compatibility

- **Target Architecture**: sm_120 (configurable in Makefile)
- **TMA Features**: Requires Hopper+ (sm_90+)
- **cp.async Features**: Requires Ampere+ (sm_80+)

## Output

The test measures bandwidth (GB/s) for different:
- Copy methods (TMA, cp.async, normal load)
- Pipeline stages (2-32, powers of 2)
- Chunk sizes (256B - 16KB)
- Producer warp counts (1, 2, 4, 8, 16)

Results show the relative performance characteristics of each memory copy method across different configurations. 
- Both TMA, cp.async, synchronous load implementations for direct comparison