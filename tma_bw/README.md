# Bandwidth Comparison: TMA vs cp.async

This directory contains a unified bandwidth test comparing Hopper's TMA (Tensor Memory Accelerator) bulk operations with traditional cp.async instructions.

## Files

### Core Implementation
- `kernels_common.cuh` - Shared PTX helpers and constants
- `tma_kernels.cuh` - TMA kernel implementation 
- `cp_async_kernels.cuh` - cp.async kernel implementation
- `benchmark_framework.cuh` - Unified benchmarking framework
- `bandwidth_test.cu` - Main unified test driver

### Build
- `Makefile` - Build configuration (targets sm_120)

## Architecture

The code is structured to eliminate duplication:

```
kernels_common.cuh          # Shared mbarrier operations, constants
├── tma_kernels.cuh         # TMA-specific implementation  
└── cp_async_kernels.cuh    # cp.async-specific implementation

benchmark_framework.cuh     # Unified timing and memory management
└── bandwidth_test.cu       # Main driver using both kernels
```

## Key Differences

### TMA Implementation (`tma_bw_kernel`)
- Uses `cp.async.bulk` instructions with mbarrier integration
- Single producer thread (thread 0) and single consumer thread (thread 32)
- Hardware-accelerated bulk transfers
- Automatic mbarrier completion signaling
- Requires SM_90+ (Hopper architecture)

### cp.async Implementation (`cp_async_bw_kernel`)
- Uses traditional `cp.async` instructions (4B, 8B, 16B)
- Producer warp (32 threads) and consumer warp (32 threads)
- Manual work distribution across warp threads
- Explicit `cp.async.wait_group` synchronization
- Compatible with SM_80+ (Ampere and newer)

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
- Pipeline stages (1-32)
- Chunk sizes (256B - 16KB) 
- Both TMA and cp.async implementations for direct comparison