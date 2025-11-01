# TMA Bandwidth Test - Simple Single-Point Tests

这个项目现在包含了三个简化的单点测试，用于快速验证特定配置的性能。

## 新增功能

### 1. 参数化的 Benchmark Framework

修改了 `benchmark_framework.cuh` 以支持：
- `warmup_iters`: 预热迭代次数（默认：1）
- `num_iters`: 测试迭代次数（默认：10）

### 2. 统一的 Kernel Wrappers

创建了 `kernel_wrappers.cuh` 来集中管理所有的 kernel wrapper：
- `CPKernelWrapper`: 通用的 cp kernel wrapper
- `TMAKernelWrapper`: TMA 专用 wrapper
- `NormalLoadKernelWrapper`: Normal Load 别名
- `CPAsyncKernelWrapper`: cp.async 别名

这避免了在多个文件中重复定义相同的 wrapper 代码。

## 文件结构

- `kernel_wrappers.cuh`: 统一的 kernel wrapper 定义
- `benchmark_framework.cuh`: 参数化的 benchmark 框架
- `simple_*.cu`: 三个简单的单点测试文件
- `bandwidth_test.cu`: 完整的 benchmark 测试套件

### 3. 简单的单点测试

#### 三个新的测试文件：

1. **`simple_normal_load_test.cu`** - Normal Load 测试
2. **`simple_tma_test.cu`** - TMA 测试  
3. **`simple_cp_async_test.cu`** - cp.async 测试

每个测试都有固定的简单配置：
- Stages: 4
- Chunk size: 1024 B
- Repeat: 8
- Warmup iterations: 3
- Test iterations: 5

## 构建和运行

### 构建所有目标
```bash
make all
```

### 构建单个简单测试
```bash
make simple_normal_load_test.out
make simple_tma_test.out
make simple_cp_async_test.out
```

### 运行简单测试
```bash
# 运行单个测试
make run-simple-normal-load
make run-simple-tma
make run-simple-cp-async

# 运行所有简单测试
make run-simple-all
```

### 直接运行可执行文件
```bash
./simple_normal_load_test.out
./simple_tma_test.out
./simple_cp_async_test.out
```

## API 使用示例

### 使用新的参数化 API

```cpp
// 使用自定义的 warmup 和迭代次数
benchmark.template run_single_config<stages, chunk_bytes, repeat>(warmup_iters, num_iters);

// 运行所有阶段，使用自定义参数
benchmark.run_all_stages<chunk_bytes, repeat>(stage_sequence, warmup_iters, num_iters);
```

### 向后兼容性

原有的 API 仍然工作，使用默认值：
```cpp
// 使用默认值（warmup=1, iters=10）
benchmark.template run_single_config<stages, chunk_bytes, repeat>();
benchmark.run_all_stages<chunk_bytes, repeat>();
```

## 示例输出

```
=== Simple Normal Load Test ===
Configuration: Stages=4, Chunk=1024 B, Repeat=8, Warmup=3, Iters=5
Stages= 4 | Chunk=1024 | Warmup=3 | Iters=5 | Time=1.386 ms | BW=387.43 GB/s

=== Simple TMA Test ===
Configuration: Stages=4, Chunk=1024 B, Repeat=8, Warmup=3, Iters=5
Stages= 4 | Chunk=1024 | Warmup=3 | Iters=5 | Time=0.443 ms | BW=1211.82 GB/s

=== Simple cp.async Test ===
Configuration: Stages=4, Chunk=1024 B, Repeat=8, Warmup=3, Iters=5
Stages= 4 | Chunk=1024 | Warmup=3 | Iters=5 | Time=0.569 ms | BW=943.08 GB/s
```

## 自定义配置

你可以修改简单测试文件中的 `constexpr` 值来测试不同的配置：

```cpp
constexpr int stages = 4;           // 管道阶段数
constexpr int chunk_bytes = 1024;   // 块大小（字节）
constexpr int repeat = 8;           // 重复次数
constexpr int warmup_iters = 3;     // 预热迭代次数
constexpr int num_iters = 5;        // 测试迭代次数
```

## Makefile 目标

- `all`: 构建所有测试
- `run-simple-all`: 运行所有简单测试
- `run-simple-normal-load`: 运行 Normal Load 简单测试
- `run-simple-tma`: 运行 TMA 简单测试
- `run-simple-cp-async`: 运行 cp.async 简单测试
- `clean`: 清理生成的文件