# Notes on GPGPU-Sim

- [Notes on GPGPU-Sim](#notes-on-gpgpu-sim)
  - [Configuration \& Statistics](#configuration--statistics)
    - [Updated](#updated)
    - [Updated, but NOT Necessary](#updated-but-not-necessary)
    - [Unchanged, but Confirmed](#unchanged-but-confirmed)
  - [Code Reading](#code-reading)
    - [General Design Rule: Reverse-Order Execution](#general-design-rule-reverse-order-execution)
    - [Memory Subsystem](#memory-subsystem)
      - [Calling Chain (wip)](#calling-chain-wip)
    - [PTX Opcode Parsing](#ptx-opcode-parsing)
      - [Calling Chain: From .ptx to function\_info](#calling-chain-from-ptx-to-function_info)
      - [Centralized Definition](#centralized-definition)
      - [Code Generation with X-Macros](#code-generation-with-x-macros)
    - [Debug: Trace](#debug-trace)
      - [Implementation](#implementation)
      - [Usage](#usage)
  - [Other Tips](#other-tips)
    - [5090 Lock Frequency](#5090-lock-frequency)
    - [Profile Flow](#profile-flow)

## Configuration & Statistics

*Meaning of config fields and statistics and how they relate to real gpu metrics*

Reference: [GPGPU-sim Doc](http://gpgpu-sim.org/manual/index.php/Main_Page#Configuration_Options)

### Updated

| 5090 |  GPGPU-sim |  Config Value | Note |
| -- | -- | -- | -- |
| # SM | gpgpu_n_clusters | 170 | |
| core clock, memory_clock_rate | gpgpu_clock_domains \<Core Clock\>:\<Interconnect Clock\>:\<L2 Clock\>:\<DRAM Clock\> | 2580:2580:2580:14000 | Core clock is set higher than base to reduce the mismatch caused by SM core, DRAM clock is confirmed by doc; interconnect & L2 clock just maintained the original convention (same as core clock), not confirmed.  |
| limits_max_cta_per_sm | gpgpu_shader_cta | 24 | |
| num_l2s_per_fbp | gpgpu_n_sub_partition_per_mchannel | 8 | |
| single DRAM bandwidth | gpgpu_dram_buswidth | 4 | \<gpgpu_n_mem_per_ctrlr\>x\<gpgpu_dram_buswidth\>x<# memory controller\> = 512 bits = 64B |
| L2 cache | gpgpu_cache:dl2 \<nsets\>:\<bsize\>:\<assoc\>,\<rep\>:\<wr\>:\<alloc\>:\<wr_alloc\>,\<mshr\>:\<N\>:\<merge\>,\<mq\> | S:256:128:24,L:B:m:L:P,A:192:4,32:0,32 | Sectored. Only adjusted to make sure \<nsets\>x\<bsize\>x\<assoc\>x\<# memory controller\>x\<gpgpu_n_sub_partition_per_mchannel\> = 96MB. Specific values are not confirmed.|


### Updated, but NOT Necessary
| 5090 |  GPGPU-sim |  Config Value | Note |
| -- | -- | -- | -- |
| Compute Capability | gpgpu_compute_capability_major | 12 | Combined with minor, means the compute capability is 12.0 |
| Compute Capability | gpgpu_compute_capability_minor | 0 | Combined with major, means the compute capability is 12.0 |
|  | gpgpu_ptx_force_max_capability | 120 | Use compute capability 12.0 as the upper limit and select the highest version of the binary for execution |
| | gpgpu_occupancy_sm_number | 120 | Corresponding to sm_120 arch, but this parameter actually has no effect |
| | gpgpu_coalesce_arch | 120 | Corresponding to sm_120 arch, but this parameter actually has no effect |


### Unchanged, but Confirmed

| 5090 |  GPGPU-sim |  Config Value | Note |
| -- | -- | -- | -- |
| # memory controller | gpgpu_n_mem | 16 | |
| Adaptive cache | gpgpu_adaptive_cache_config | 1 | If a kernel does not utilize shared memory, all the onchip storage will be assigned to the L1D cache  |
| L1/shmem size (KB) per SM | gpgpu_unified_l1d_size | 128 | L1 cache/ Shared memory size per SM core |


## Code Reading

*E.g. key functions, calling chain for a memory access, how the memory bandwidth is modeled, etc.*

### General Design Rule: Reverse-Order Execution

  - Take SM core cycle as an example. 
  ```cpp
  // File: gpgpu-sim/shader.cc
  void shader_core_ctx::cycle() {
    if (!isactive() && get_not_completed() == 0) return;

    m_stats->shader_cycles[m_sid]++;
    writeback();
    execute();
    read_operands();
    issue();
    for (unsigned int i = 0; i < m_config->inst_fetch_throughput; ++i) {
      decode();
      fetch();
    }
  }
  ```
  真实硬件中，在时钟周期T，所有pipeline stage（e.g. IF, ID, ISSUE, REG, EX, WB）并行执行，且都依赖 T−1 周期的结果。如果simulator依照 IF → ID → ISSUE → REG → EX → WB 顺序模拟，ID 会在同一个周期 T 错误地用到 IF 刚产生的结果。因此simulator需要逆序执行，确保数据依赖正确。

  对所有`cycle()`实现，gpgpu-sim的代码都是真实硬件执行的逆序。

### Memory Subsystem

#### Calling Chain (wip)

```mermaid
graph TD;
    %% --- 节点定义 (Node Definitions) ---
    A["<b>gpgpu_sim::cycle()</b><br> 模拟器的主循环函数。分CORE, L2, ICNT, DRAM四个clock domain，驱动相应模块的周期性活动"];

    B["Cluster -> Shader Core -> EX -> LSU"];
    Ba["进入memory cycle，设定新访问请求的delay cycle并将其推入latency queue"];
    Bb["每周期推进delay cycle，在delay归0时，探测L1 cache是否命中，将miss的访问请求推入miss queue"]
    Bc["从miss queue中取出请求，注入Interconnect"]

    C["<b>memory_sub_partition::cache_cycle()</b><br> L2缓存所在内存子分区的核心驱动函数。处理进出L2的请求"];
    D["<b>l2_cache::access()</b><br> L2缓存的访问入口。处理来自互联网络（ICNT）的内存请求"];
    E["<b>memory_partition_unit::dram_cycle()</b><br> 驱动DRAM控制器和调度器的运行，向DRAM模型发送命令"];

    %% --- 边定义 (Edge Definitions / Call Chain) ---
    A --m_cluster[i]->core_cycle();--> B;
    A --> C;
    A --> E;

    subgraph SM Core and L1 Cache
        B --> Ba;
        B --> Bb;
        B --> Bc;
        Ba --> Bb --> Bc;
    end

    subgraph L2 Cache
        C --> D;
    end

    subgraph DRAM
        E;
    end

```

##### SM Core and L1 Cache
```mermaid
graph TD;
    %% --- 节点定义 (Node Definitions) ---
    C["<b>simt_core_cluster::core_cycle()</b><br> 驱动SM Cluster中所有Shader Core的流水线执行"];
    D["<b>shader_core_ctx::cycle()</b><br> 单个Shader Core的流水线主函数，逆序调用各流水线阶段(IF-ID-ISSUE-REG-EXE-WB)"];
    E["<b>shader_core_ctx::execute()</b><br> 执行(EX)阶段，驱动所有FU，处理指令计算，调用LSU"];

    F["<b>ldst_unit::cycle()</b><br> LSU顶层cycle。处理所有底层函数返回的STALL"];
    Fa["<b>ldst_unit::L1_latency_queue_cycle()</b><br> 模拟有delay的l1 cache，检测每个bank的pipeline出口是否有请求，访问L1 Cache"];
    N["<b>ldst_unit::memory_cycle()</b><br> 驱动除了constant, shared, texture之外的memory访问，发送请求或者记录停顿"];

    G["<b>l1_cache::access()</b><br> L1 Cache的访问入口。探测tag状态，分发请求给处理函数，更新统计信息，返回access_status给LSU(用于判断是否要stall)"];
    Ga["<b>tag_array::probe()</b><br> 探测tag array，判断是否HIT，若MISS则判断evict cacheline index"];

    M["<b>baseline_cache::cycle()</b><br> 从m_miss_queue中取出请求发送到下一级内存，更新端口占用状况"];
    Na["<b>mem_fetch_interface::push()</b><br> 在此采用shader_memory_interface子类，统计网络流量，将mf(请求)注入互联网络(Interconnect)" ];
    Za["<b>simt_core_cluster::icnt_inject_request_packet()</b><br> 更新统计信息，确认destination，调用外部interconnect模型接口"]
    Z["<b>::icnt_push()</b><br> GPGPU-Sim核心与具体互联网络模型 (e.g. BookSim) 之间的接口"]

    Nb["<b>ldst_unit::process_memory_access_queue_l1cache()</b><br> 若模拟l1 delay，则确定目标bank，将mf放入l1_latency_queue由后续LSU cycle调用的L1_latency_queue_cycle()处理，若有conflict则返回上游函数使其stall；若使用理想l1(无delay)，则直接进行cache access"]
    H["<b>data_cache::process_tag_probe()</b><br> 接收tag探测结果，如果是不是HIT或RESERVATION_FAIL，就处理MISS"];
    I["<b>m_wr_miss</b><br> MISS处理指针，根据配置的m_write_alloc_policy跳转"];

    Ia["<b>data_cache::<br>wr_miss_wa_naive()</b>"];
    Ib["<b>data_cache::<br>wr_miss_wa_lazy_fetch_on_read()</b><br>"];
    Ic["<b>data_cache::<br>wr_miss_wa_fetch_on_write()</b><br>"];
    Id["<b>data_cache::<br>wr_miss_no_wa()</b><br>"];

    O["<b>data_cache::send_write_request()</b><br>" 将mem_fetch对象推入m_miss_queue，后续在LSU cycle中调用的m_L1D->cycle会把mf对象取出并发往下一级内存]

    %% --- 边定义 (Edge Definitions / Call Chain) ---

        C --m_core[*it]->cycle();--> D;
        D --execute();--> E;
        E --m_fu[n]->cycle();--> F;
        F --m_L1D->cycle();--> M;
        M --m_memport->push(mf);--> Na;
        F --memory_cycle(pipe_reg, rc_fail, type);--> N;
        F --L1_latency_queue_cycle();--> Fa;
        Fa --m_L1D->access(mf_next->get_addr(), mf_next, m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle, events);--> G;
        N --If bypassL1D == true<br>m_icnt->push(mf);--> Na;
        Na --m_cluster->icnt_inject_request_packet(mf);--> Za;
        Za --::icnt_push(m_cluster_id, m_config->mem2device...);--> Z;
        N --If bypassL1D == false--> Nb;
        Nb --cache->access()--> G;
        G --m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);--> Ga;
        G --process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);--> H;
        H --(this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);--> I;
        I --WRITE_ALLOCATE--> Ia;
        I --LAZY_FETCH_ON_READ--> Ib;
        I --FETCH_ON_WRITE--> Ic;
        I --NO_WRITE_ALLOCATE--> Id;
        Ia --> O;
        Ib --> O;
        Ic --> O;
        Id --> O;
```

##### L2 Cache (wip)
##### DRAM (wip)


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

以"add"为例：Source (`cuda-sim/opcodes.def`): `OP_DEF(ADD_OP, add_impl, "add", 1, 1)`

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
   - `trace.cc` include `trace_streams.tup`生成string数组，将enum ID映射到其对应的名称（e.g. WARP_SCHEDULER -> "WARP_SCHEDULER"），用于日志打印。
   - `trace.cc` 中的 `Trace::init()` 函数会解析配置文件中的`trace_components`选项，填充一个Boolean数组 (trace_streams_enabled)，以记录哪些stream被用户启用。

3. User Interface : `DPRINTF`, `DPRINTF_NoGPU`, `DPRINTFG`
   - 在`trace.h`中定义，核心宏为 `DPRINTF(STREAM_NAME, "format string", ...)`
   - 功能：
     - 检查`gpgpusim.config`：是否`-trace_enabled==1`，当前STREAM_NAME是否在`trace_components`中启用
     - 输出cycle, STREAM_NAME, 用户printf(...)信息
   - 变体：
     - `DPRINTF_NoGPU`面向无m_gpu场景，不输出cycle
     - `DPRINTFG` 全局信息（？To be completed）
  
   
#### Usage
1. Add Trace Points: Include `trace.h` 并调用 `DPRINTF` 宏
```cpp
#include "trace.h"

...
// 第一个参数是trace_streams.tup中定义的stream名，后续同printf()用法
DPRINTF(WARP_SCHEDULER, "Warp %u is now stalled.\n", warp_id);
...
```

2. Enable Streams: 在`gpgpusim.config`中配置trace

```
-trace_enabled 1
-trace_components WARP_SCHEDULER,SCOREBOARD
```

3. Compile and Run: Set `TRACING_ON=1` during compilation.


## Other Tips

*Other useful tips*

### 5090 Lock Frequency
两种方式都不是hard constraint，最终SM Frequency会显著小于设定值（观测到的max gap: 1.79GHz/2.01GHz）
 1. ncu默认锁定频率到base clock (5090为2.01GHz)，可以不进行任何设置；也可以通过`ncu --clock-control base`显式指定 ([ref](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control))
   
 2. 先用`nvidia-smi -i $(GPU_ID) -pm 1`进入persistence mode，再用`nvidia-smi -i $(GPU_ID) -lgc $(GpuClock)`指定频率，同时设定`ncu --clock-control none` ([ref](https://stackoverflow.com/questions/64701751/can-i-fix-my-gpu-clock-rate-to-ensure-consistent-profiling-results))
    - 支持的GpuClock可使用`sudo nvidia-smi -q -d SUPPORTED_CLOCKS`查询，单位：MHz
    - profile结束后用`sudo nvidia-smi -i $(GPU_ID) -rgc`恢复GPU动态调频
    - Observation: GpuClock需略大于期望值，即若期望为2010MHz，则建议使用`nvidia-smi -i $(GPU_ID) -lgc 2011`；若输入`nvidia-smi -i $(GPU_ID) -lgc 2010`，则会在dmon中观测到clock为2002。（此行为未找到官方说明）
    - 鉴于SM Frequency为测量值(cycles/duration)，dmon监测值和ncu report值仍存在明显差距 ([ref](https://forums.developer.nvidia.com/t/sm-frequency-reported-in-nsight-compute/264271))

### Profile Flow
  1. Lock Frequency
  ```shell
  sudo nvidia-smi -i ${GPU_ID} -lgc $((CLOCK+1))
  ```
  2. Run Profiling
  - L2 Cache
    ```shell
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ncu -f -o ${workload} \
    --cache-control none \
    --replay-mode application \
    --section MemoryWorkloadAnalysis_Chart \
    --section SpeedOfLight \
    --target-processes all \
    --clock-control none \
    ${cmd}
    ```
  - DRAM
    ```shell
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ncu -f -o ${workload} \
    --section MemoryWorkloadAnalysis_Chart \
    --section SpeedOfLight \
    --target-processes all \
    --clock-control none \
    ${cmd}
    ```
  3. Reset Frequency
  ```shell
  sudo nvidia-smi -i ${GPU_ID} -rgc
  ```
