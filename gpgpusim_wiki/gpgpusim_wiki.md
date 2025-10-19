# Notes on GPGPU-Sim

## Configuration & Statistics

*Meaning of config fields and statistics and how they relate to real gpu metrics*

Reference: [GPGPU-sim Doc](http://gpgpu-sim.org/manual/index.php/Main_Page#Configuration_Options)

| 5090 |  GPGPU-sim |  Config Value | Note |
| -- | -- | -- | -- |
| # SM | gpgpu_n_clusters | 170 | |
| base clock, memory_clock_rate | gpgpu_clock_domains \<Core Clock\>:\<Interconnect Clock\>:\<L2 Clock\>:\<DRAM Clock\> | 2010:2010:2010:14000 | Core & DRAM clock are confirmed by doc; interconnect & L2 clock just maintained the original convention (same as core clock), not confirmed.  |
| limits_max_cta_per_sm | gpgpu_shader_cta | 24 | |
| # memory controller | gpgpu_n_mem | 16 | |
| single DRAM bandwidth | gpgpu_dram_buswidth | 4 | |
| L2 cache | gpgpu_cache:dl2 \<nsets\>:\<bsize\>:\<assoc\>,\<rep\>:\<wr\>:\<alloc\>:\<wr_alloc\>,\<mshr\>:\<N\>:\<merge\>,\<mq\> | S:1536:128:32,L:B:m:L:X,A:192:4,32:0,32 | Only adjusted to make sure \<nsets\>x\<bsize\>x\<assoc\>x\<# memory controller\> = 96MB. Specific values are not confirmed. Addr mapping policy changed to XOR since original method (hash) needs manual encoding, but current number of sets is too big. |



## Code Reading

*E.g. key functions, calling chain for a memory access, how the memory bandwidth is modeled, etc.*

### PTX Opcode Parsing

采用 **X-Macro** 模式，将数据的定义与使用分离。

#### Definition

`opcodes.def` 是所有opcode信息的来源。文件中的每一行都通过macro（`OP_DEF` 或 `OP_W_DEF`）定义了一个opcode的全部meta data。

```c
// File: cuda-sim/opcodes.def
// ...
//        (1)        (2)            (3)           (4)   (5)
OP_DEF(   ADD_OP,    add_impl,      "add",        1,    1)
OP_DEF(   BRA_OP,    bra_impl,      "bra",        0,    3)
OP_W_DEF( BAR_OP,    bar_impl,      "bar.sync",   0,    3) // Warp-level instruction
// ...
```
- (1) 枚举名: ADD_OP - 在代码中使用的唯一标识符。
- (2) 实现函数: add_impl - 实现了该指令功能的函数指针。
- (3) 字符串名: "add" - PTX汇编指令名，在解析.ptx文件时进行匹配。
- (4) 目标操作数标志: 1 表示该指令有目标操作数（destination operand），0 表示没有。
- (5) 分类ID: 1 (ALU), 3 (Control) - 用于流水线调度、资源分配和性能统计。

#### 宏展开

以add为例：
* **数据源 (`opcodes.def`)**:
    ```c
    OP_DEF(ADD_OP, add_impl, "add", 1, 1)
    ```

1. 生成枚举 (`cuda-sim/opcodes.h`)
    * **宏定义**:
        ```cpp
        #define OP_DEF(OP,...) OP,
        ```
    * **展开效果**:
        ```cpp
        enum opcode_t {
            ...,
            ADD_OP,
            ...
        };
        ```
    * **目的**: 获得一个唯一的、类型安全的整数标识符 `ADD_OP`。

2. 生成字符串映射 (`cuda-sim/instructions.cc`)
    * **宏定义**:
        ```cpp
        #define OP_DEF(OP, FUNC, STR, ...) STR,
        ```
    * **展开效果**:
        ```cpp
        const char *g_opcode_string[] = {
            ...,
            "add",
            ...
        };
        ```
    * **目的**: 创建一个可以通过 `g_opcode_string[ADD_OP]` 快速查找指令名称的数组，用于调试和日志。

3. 生成执行分派表 (`cuda-sim/cuda-sim.cc`)
    * **宏定义**:
        ```cpp
        #define OP_DEF(OP, FUNC, ...) case OP: FUNC(...); break;
        ```
    * **展开效果**:
        ```cpp
        switch(opcode) {
            ...
            case ADD_OP:
                add_impl(...);
                break;
            ...
        }
        ```
    * **目的**: 构建一个高效的 `switch-case` 结构，将指令的执行请求分派到其对应的实现函数 `add_impl`。


## Other Tips

*Other useful tips*

1. 可用以下两种方式之一锁频，但都不是hard constraint，最终SM Frequency会显著小于设定值（目前观测的max gap: 1.79GHz/2.01GHz）
    1. ncu默认锁定频率到base clock (5090为2.01GHz)，可以不进行任何设置；也可以通过`ncu --clock-control base`显式指定 ([ref](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control))
    2. 先用`nvidia-smi -i $(GPU_ID) -pm 1`进入persistence mode，再用`nvidia-smi -i $(GPU_ID) -lgc $(GpuClock)`指定频率，同时设定`ncu --clock-control none` ([ref](https://stackoverflow.com/questions/64701751/can-i-fix-my-gpu-clock-rate-to-ensure-consistent-profiling-results))
       - 支持的GpuClock可使用`sudo nvidia-smi -q -d SUPPORTED_CLOCKS`查询，单位：MHz
       - profile结束后用`sudo nvidia-smi -i $(GPU_ID) -rgc`恢复GPU动态调频
       - Observation: GpuClock需略大于期望值，即若期望为2010MHz，则建议使用`nvidia-smi -i $(GPU_ID) -lgc 2011`；若输入`nvidia-smi -i $(GPU_ID) -lgc 2010`，则会在dmon中观测到clock为2002。（此行为未找到官方说明）
       - 鉴于SM Frequency为测量值(cycles/duration)，dmon监测值和ncu report值仍存在明显差距 ([ref](https://forums.developer.nvidia.com/t/sm-frequency-reported-in-nsight-compute/264271))
