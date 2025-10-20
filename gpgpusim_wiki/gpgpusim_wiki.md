# Notes on GPGPU-Sim

- [Notes on GPGPU-Sim](#notes-on-gpgpu-sim)
  - [Configuration \& Statistics](#configuration--statistics)
  - [Code Reading](#code-reading)
    - [PTX Opcode Parsing](#ptx-opcode-parsing)
      - [Calling Chain: From .ptx to function\_info](#calling-chain-from-ptx-to-function_info)
      - [Centralized Definition](#centralized-definition)
      - [Code Generation with X-Macros](#code-generation-with-x-macros)
    - [Debug: Trace](#debug-trace)
      - [Implementation](#implementation)
      - [How to Use](#how-to-use)
  - [Other Tips](#other-tips)

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

#### Calling Chain: From .ptx to function_info
![PTX Opcode Parsing Flow](figs/ptx-parsing-calling-chain.png)

#### Centralized Definition

`cuda-sim/opcodes.def` 通过Macro(`OP_DEF`,`OP_W_DEF`) ，集中定义了所有opcode的meta data。

```c
// ...
//        (1)        (2)            (3)           (4)   (5)
OP_DEF(   ADD_OP,    add_impl,      "add",        1,    1)
OP_DEF(   BRA_OP,    bra_impl,      "bra",        0,    3)
OP_W_DEF( BAR_OP,    bar_impl,      "bar.sync",   0,    3)
// ...
```
- (1) 枚举名: ADD_OP - 源码中使用的唯一ID。
- (2) 实现函数: add_impl - 执行指令功能的函数。
- (3) 字符串名: "add" - PTX指令名，用于解析.ptx的文本匹配。
- (4) 目标操作数标志: 标记是否有目标操作数 (1=有, 0=无)。
- (5) 分类ID: 指令类型 (如ALU, Control)，具体编码见注释。

#### Code Generation with X-Macros

以"add"为例：

* **Source (`cuda-sim/opcodes.def`)**: `OP_DEF(ADD_OP, add_impl, "add", 1, 1)`

1. Opcode Struct (`cuda-sim/opcodes.h`): `#define OP_DEF(OP,...) OP,`
    * **Expansion Result**:
        ```cpp
        enum opcode_t {
            ...,
            ADD_OP,
            ...
        };
        ```
    * **Purpose**: 获得一个唯一的、类型安全的整数标识符 `ADD_OP`。

2. Name Mapping (`cuda-sim/instructions.cc`): `#define OP_DEF(OP, FUNC, STR, ...) STR,`
    * **Expansion Result**:
        ```cpp
        const char *g_opcode_string[] = {
            ...,
            "add",
            ...
        };
        ```
    * **Purpose**: 创建一个可以通过 `g_opcode_string[ADD_OP]` 快速查找指令名称的数组，用于调试和日志。

3. Dispatch Table (`cuda-sim/cuda-sim.cc`): `#define OP_DEF(OP, FUNC, ...) case OP: FUNC(...); break;`
    * **Expansion Result**:
        ```cpp
        switch(opcode) {
            ...
            case ADD_OP:
                add_impl(...);
                break;
            ...
        }
        ```
    * **Purpose**: 构建一个高效的 `switch-case` 结构，将ptx opcode `add`分派到其对应的C实现函数 `add_impl()`。


### Debug: Trace

#### Implementation
1. Data Source (`trace_streams.tup`): A central file listing all available trace streams (e.g., WARP_SCHEDULER).

2. Code Generation (`trace.h`, `trace.cc`): 
   - `trace.h` 通过include `trace_streams.tup` 自动生成一个 `enum`，为每个stream提供唯一的ID。
   - `trace.cc` 同样包include `trace_streams.tup`，用于生成字符串数组，将enum ID映射到其对应的名称（例如 WARP_SCHEDULER -> "WARP_SCHEDULER"），用于日志打印。
   - trace.cc 中的 Trace::init() 函数会解析配置文件中的 -gpgpu_trace 选项，填充一个布尔数组 (trace_streams_enabled)，以记录哪些流被用户启用。

3. User Interface : `DPRINTF`
   - 在`trace.h`中定义，核心宏为 `DPRINTF(STREAM_NAME, "format string", ...)`
   - 在打印前会进行双重检，如果通过，则会自动添加cycle和stream名称作为前缀，再按照printf输出用户信息
     - 检查trace system的总开关 (Trace::enabled) 是否打开
     - 检查当前的STREAM_NAME是否在`gpgpusim.config`中被启用
  
   
#### How to Use
1. Add Trace Points: Include `trace.h` 并调用 `DPRINTF` 宏
```cpp
// Example in a scheduler file
#include "trace.h"

...
// 第一个参数是 trace_streams.tup 中定义的stream名，后续同`printf`用法
DPRINTF(WARP_SCHEDULER, "Warp %u is now stalled.\n", warp_id);
...
```

2. Enable Streams: 在`gpgpusim.config`中配置trace

```
-trace_enabled 1
-trace_components WARP_SCHEDULER,SCOREBOARD
```

3. Compile and Run: Set TRACING_ON=1 during compilation.



## Other Tips

*Other useful tips*

1. 5090可以用以下两种方式锁频，但它们都不是hard constraint，最终SM Frequency会显著小于设定值（目前观测的max gap: 1.79GHz/2.01GHz）
    1. ncu默认锁定频率到base clock (5090为2.01GHz)，可以不进行任何设置；也可以通过`ncu --clock-control base`显式指定 ([ref](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control))
    2. 先用`nvidia-smi -i $(GPU_ID) -pm 1`进入persistence mode，再用`nvidia-smi -i $(GPU_ID) -lgc $(GpuClock)`指定频率，同时设定`ncu --clock-control none` ([ref](https://stackoverflow.com/questions/64701751/can-i-fix-my-gpu-clock-rate-to-ensure-consistent-profiling-results))
       - 支持的GpuClock可使用`sudo nvidia-smi -q -d SUPPORTED_CLOCKS`查询，单位：MHz
       - profile结束后用`sudo nvidia-smi -i $(GPU_ID) -rgc`恢复GPU动态调频
       - Observation: GpuClock需略大于期望值，即若期望为2010MHz，则建议使用`nvidia-smi -i $(GPU_ID) -lgc 2011`；若输入`nvidia-smi -i $(GPU_ID) -lgc 2010`，则会在dmon中观测到clock为2002。（此行为未找到官方说明）
       - 鉴于SM Frequency为测量值(cycles/duration)，dmon监测值和ncu report值仍存在明显差距 ([ref](https://forums.developer.nvidia.com/t/sm-frequency-reported-in-nsight-compute/264271))
