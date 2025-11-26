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
        - [SM Core and L1 Cache](#sm-core-and-l1-cache)
        - [L2 Cache (wip)](#l2-cache-wip)
        - [DRAM (wip)](#dram-wip)
    - [From Binary to cycle()](#from-binary-to-cycle)
      - [CUDA Flow](#cuda-flow)
      - [GPGPU-sim Calling Chain](#gpgpu-sim-calling-chain)
    - [PTX Opcode Parsing](#ptx-opcode-parsing)
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
    A["<b>gpgpu_sim::cycle()</b><br> GPGPU-sim的主循环函数。分CORE, L2, ICNT, DRAM四个clock domain，驱动相应模块的周期性活动"];

    B["Cluster -> Shader Core -> EX Stage -> LSU Cycle"];
    Ba["进入memory cycle，设定新访问请求的delay cycle并将其推入latency queue"];
    Bb["每周期推进delay cycle，在delay归0时，探测L1 cache是否命中，将miss的访问请求推入miss queue"]
    Bc["从miss queue中取出请求，注入Interconnect"]
    Bd["从INCT中读取完成的请求，推入LSU的response fifo"]
    Be["从response fifo取出请求；如果是write ack，更新warp“在途写”计数器；如果是其他请求，调用fill()填充对应cache，标记req已ready（可被writeback处理）"]
    Bf["轮询各层次memory，执行writeback()将数据写回reg file并更新响应依赖"]

    C["<b>memory_sub_partition::cache_cycle()</b><br> L2缓存所在内存子分区的核心驱动函数。处理进出L2的请求"];
    D["<b>Work in Progress</b><br>"];

    E["<b>memory_partition_unit::dram_cycle()</b><br> 驱动DRAM控制器和调度器的运行，向DRAM模型发送命令"];
    F["<b>Work in Progress</b><br>"];


    %% --- 边定义 (Edge Definitions / Call Chain) ---
    A --> B;
    A --> Bd;
    A --> E;
    A --> C;

    subgraph SM Core and L1 Cache
      subgraph Request path
        B --> Ba;
        B --> Bb;
        B --> Bc;
        Ba --> Bb --> Bc;
      end
      subgraph Response path
        B --> Be;
        B --> Bf;
        Bd --> Be --> Bf;
      end
    end

    subgraph L2 Cache
        C --> D;
    end

    subgraph DRAM
        E --> F;
    end

```

##### SM Core and L1 Cache
```mermaid
graph TD;
    %% --- Node Definitions ---
    A["<b>gpgpu_sim::cycle()</b><br> if (clock_mask & CORE)"];
    C["<b>simt_core_cluster::core_cycle()</b><br> 驱动SM Cluster中所有Shader Core的流水线执行"];
    D["<b>shader_core_ctx::cycle()</b><br> 单个Shader Core的流水线主函数，逆序调用各流水线阶段(IF-ID-ISSUE-REG-EXE-WB)"];
    E["<b>shader_core_ctx::execute()</b><br> 执行(EX)阶段，驱动所有FU，处理指令计算，调用LSU"];
    F["<b>ldst_unit::cycle()</b><br> LSU顶层cycle。处理所有底层函数返回的STALL"];
    Fa["<b>ldst_unit::<br>L1_latency_queue_cycle()</b><br> 模拟有delay的l1 cache，检测每个bank的pipeline出口是否有请求，访问L1 Cache"];
    N["<b>ldst_unit::memory_cycle()</b><br> 驱动除了constant, shared, texture之外的memory访问，发送请求或者记录停顿"];
    G["<b>l1_cache::access()</b><br> L1 Cache的访问入口。探测tag状态，分发请求给处理函数，更新统计信息，返回access_status给LSU(用于判断是否要stall)"];
    Ga["<b>tag_array::probe()</b><br> 探测tag array，判断是否HIT，若MISS则判断evict cacheline index"];
    M["<b>baseline_cache::cycle()</b><br> 从m_miss_queue中取出请求发送到下一级内存，更新端口占用状况"];
    Na["<b>mem_fetch_interface::push()</b><br> 在此采用shader_memory_interface子类，统计网络流量，将mf(请求)注入互联网络(Interconnect)" ];
    Za["<b>simt_core_cluster::<br>icnt_inject_request_packet()</b><br> 更新统计信息，确认destination，调用外部interconnect模型接口"]
    Z["<b>::icnt_push()</b><br> GPGPU-Sim核心与具体互联网络模型 (e.g. BookSim) 之间的接口"]
    Nb["<b>ldst_unit::<br>process_memory_access_queue_l1cache()</b><br> 若模拟l1 delay，则确定目标bank，将mf放入l1_latency_queue由后续LSU cycle调用的L1_latency_queue_cycle()处理，若有conflict则返回上游函数使其stall；若使用理想l1(无delay)，则直接进行cache access"]
    H["<b>data_cache::process_tag_probe()</b><br> 接收tag探测结果，如果是不是HIT或RESERVATION_FAIL，就处理MISS"];
    I["<b>m_wr_miss</b><br> MISS处理指针，根据配置的m_write_alloc_policy跳转"];
    Ia["<b>data_cache::<br>wr_miss_wa_naive()</b>"];
    Ib["<b>data_cache::<br>wr_miss_wa_lazy_fetch_on_read()</b><br>"];
    Ic["<b>data_cache::<br>wr_miss_wa_fetch_on_write()</b><br>"];
    Id["<b>data_cache::<br>wr_miss_no_wa()</b><br>"];
    O["<b>data_cache::send_write_request()</b><br>" 将mem_fetch对象推入m_miss_queue，后续在LSU cycle中调用的m_L1D->cycle会把mf对象取出并发往下一级内存];

    P["<b>simt_core_cluster::icnt_cycle()</b><br> ICNT与Core间的接口：Stage 1从ICNT拉取新响应，Stage 2把响应dispatch到对应的SM core"];
    R["<b>ldst_unit::fill()</b><br> 更新mf的状态，将响应推入LSU内部的m_reponse_fifo，将由LSU cycle处理"];
    Sc["<b>baseline_cache::fill()</b><br> 对于非WRITE_ACK类型的内存响应（包括texture, constant和local/global memory），调用对应的(m_L1T/m_L1C/m_L1D)->fill()来填充数据"];
    Sb["<b>m_next_global = mf;</b><br> 对于bylass L1的全局/局部内存响应，将mf写入m_global_next，将由下个周期的ldst_unit::writeback()中写回register file"];
    Sa["<b>shader_core_ctx::<br>store_ack()</b><br> 确定发出写操作的warp，调用dec_store_req()递减warp内部的m_stores_outstanding计数器"];
    Sd["<b>tag_array::fill()</b><br> 更新cacheline meta data，写入数据（通过调用cache_block_t::fill），维护dirty bit"];
    Se["<b>mshr_table::mark_ready()</b><br> 将完成的block addr推入m_current_response，后续在LSU writeback()内部轮询时可以access_ready()和next_access()读取并处理"];

    U["<b>ldst_unit::writeback()</b><br>LSU的写回仲裁逻辑，Stage 1轮询L1T, L1C, L1D, m_global_next, 选择一个完成的请求写入m_next_wb；Stage 2将m_next_wb写回寄存器、更新依赖"];
    V["<b>opndcoll_rfu_t::writeback()</b><br>操作数收集单元(Operand Collector)执行写回，将数据写入物理寄存器堆"];
    W["<b>Scoreboard::releaseRegister()</b><br>记分板释放对目标寄存器的占用"];
    X["<b>shader_core_ctx::warp_inst_complete()</b><br>标记指令完成，更新统计信息"];
    Y["在ldst_unit::cycle()中，处理m_response_fifo内的数据响应"];

    IS["<b>shader_core_ctx::issue()</b><br> 发射(IS)阶段，根据Round Robin在scheduler中轮询"];
    SCHE["<b>scheduler_unit::cycle()</b><br> 将具体指令分派给具体warp，调度原理为：优先级排序->依赖检查->资源检查->dispatch"];
    ISW["<b>shader_core_ctx::issue_warp()</b><br> 完成所在warp将issue指令的functional simulation，再根据ext_inst->op，跳转到对应的performance simulation handler"];
    EXEC["<b>exec_shader_core_ctx::<br>func_exec_inst()</b><br> 对所有指令执行execute_warp_inst_t，对内存访问指令执行generate_mem_accesses"];
    FUNC["<b>core_t::execute_warp_inst_t()</b><br> 确定warp中的active thread，计算tid，调用ptx_exec_inst，完成功能模拟后更新执行状态"];
    MEM["<b>warp_inst_t::<br>generate_mem_accesses()</b><br> 对于load/store指令，生成mem_access_t；对于shared memory访问，估算bank conflict导致的最大delay写入cycles，用于lsu cycle中shared_cycle()递减"];
    PTX["<b>ptx_thread_info::ptx_exec_inst()</b><br> 内含使用X-Macros展开得到的dispatch table，将跳转到对应指令的_impl()完成functioncal simulation"];

    %% --- Edge Definitions / Call Chain ---
    D --issue();--> IS;

    subgraph "Functional Simulation"
      IS --schedulers[j]->cycle();--> SCHE;
      SCHE --m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id, m_id);--> ISW;
      ISW --func_exec_inst(**pipe_reg);--> EXEC;
      EXEC --execute_warp_inst_t(inst);--> FUNC;
      EXEC --inst.generate_mem_accesses();--> MEM;
      FUNC --m_thread[tid]->ptx_exec_inst(inst, t);--> PTX;
    end

    A --m_cluster[i]->core_cycle();--> C;
    A --m_cluster[i]->icnt_cycle();--> P;

    C --m_core[*it]->cycle();--> D;
    D --execute();--> E;
    E --m_fu[n]->cycle();--> F;
    F --m_L1D->cycle();--> M;
    F --memory_cycle(pipe_reg, rc_fail, type);--> N;
    F --L1_latency_queue_cycle();--> Fa;
    F -- "writeback()" --> U;
    F --if (!m_response_fifo.empty())--> Y;

    subgraph "Request Path"
      M --m_memport->push(mf);--> Na;
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
    end

    subgraph "Response Path"
      P --m_core[cid]->accept_ldst_unit_response(mf);<br>m_ldst_unit->fill(mf);--> R;
      U --m_operand_collector->writeback(m_next_wb)--> V;
      V --> W;
      W --> X;
      Y --if mf->get_access_type() == TEXTURE_ACC_R / CONST_ACC_R<br>或者bypassL1D == false--> Sc;
      Y --If bypassL1D == true--> Sb;
      Y --If mf->get_type() == WRITE_ACK--> Sa;
      Sc --m_tag_array->fill(e->second.m_cache_index, time, mf);--> Sd;
      Sc --m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);--> Se;
    end
```

Note:
- warp的m_stores_outstanding计数器用于跟踪已发出的store req数量，在store req被推向ICNT时递增，在LSU cycle中解析到对应的write ack时递减；用于与membar配合。

##### L2 Cache (wip)

##### DRAM (wip)

### From Binary to cycle()

#### CUDA Flow
```mermaid
graph LR;
    %% --- 节点定义 (Node Definitions) ---
    %% Phase 1: Registration
    OS["<b>OS Loader</b><br> 系统加载binary，触发 .init_array 段中的代码"];

    Ctor["<b>__cuda_module_ctor()</b><br> nvcc 自动生成的 C++ 静态构造函数。在 main() 执行前运行，负责初始化上下文"];

    RegFat["<b>__cudaRegisterFatBinary()</b><br> 将包含 PTX/SASS 的 Fatbinary Blob 提交给 Driver<br>Input: void* fatCubin<br>Output: void** handle (模块句柄)"];

    RegFunc["<b>__cudaRegisterFunction()</b><br> 建立 Host 函数指针与 Device 代码名的映射<br>Input: handle, host_ptr, device_name<br>Output: Runtime 内部哈希表条目"];

    RegVar["<b>__cudaRegisterVar()</b><br> 注册全局变量 (__device__, __constant__)<br>Input: handle, host_var_ptr, name, size"];

    %% Phase 2: Execution
    Main["<b>int main()</b><br> 用户程序入口。此时 Runtime 已知晓所有 Kernel 信息"];

    UserCall["<b>Kernel Syntax</b><br> 用户代码: kernel<<<Dg, Db>>>(args)<br>编译器将其展开为 Runtime API 调用"];

    Launch["<b>cudaLaunchKernel()</b><br> 统一的 Kernel 启动 API (或 cudaLaunch 组合)<br>Input: func_ptr (Host指针), grid, block, args<br>Output: 向 Driver 发送 Launch 请求"];

    Lookup["<b>Runtime Lookup</b><br> Runtime 使用 func_ptr 在内部哈希表中查找对应的 device_name (由 Phase 1 注册)"];

    DriverExec["<b>cuLaunchKernel()</b><br> Driver API。驱动程序调度真正的硬件指令 (SASS) 到 GPU 执行"];

    %% --- 边定义 (Edge Definitions / Call Chain) ---
    
    %% Phase 1: Register Chain
    OS --> Ctor;
    Ctor --> RegFat;
    RegFat --> RegFunc;
    RegFat --> RegVar;

    %% Link to Phase 2
    RegFunc --> Main;
    RegVar --> Main;

    %% Phase 2: Execution Chain
    Main --> UserCall;
    UserCall --> Launch;
    Launch --> Lookup;
    Lookup --> DriverExec;

    %% Data Dependency (Logical Link)
    RegFunc -.-> Lookup;

    %% --- 子图分组 (Subgraphs) ---
    subgraph Phase 1: Registration
        OS;
        Ctor;
        RegFat;
        RegFunc;
        RegVar;
    end

    subgraph Phase 2: Execution
        Main;
        UserCall;
        Launch;
        Lookup;
        DriverExec;
    end
```
#### GPGPU-sim Calling Chain

```mermaid
graph TD;
    %% --- 节点定义 (Node Definitions) ---
    %% Root Entry
    Entry["<b>__cudaRegisterFatBinary(void *fatCubin)</b><br> CUDA 运行时入口：注册 fatbin，返回 fatbin handle"];
    
    Internal["<b>cudaRegisterFatBinaryInternal(...)</b><br> 内部包装"];
    
    Impl["<b>cudaRegisterFatBiaryInternal_impl(...)</b><br> 核心注册逻辑：确保上下文就绪，注册 fatbin，首次触发代码提取"];

    FuncEntry["<b>__cudaRegisterFunction(...)</b><br> Host 端函数注册入口"];

    %% --- Subgraph: Initialization ---
    InitCtx["<b>GPGPU_Context()</b><br>"];
    InitSim["<b>GPGPUSim_Context(ctx)</b><br>"];
    SimInit["<b>gpgpu_context::GPGPUSim_Init()</b><br> 完成硬件模块配置与pthread启动"];

    %% --- Subgraph: PTX Extraction (Modern Path) ---
    RegFatAPI["<b>api->cuobjdumpRegisterFatBinary(...)</b><br> 记录句柄映射；若 handle==1 则触发 cuobjdumpInit"];

    DumpInit["<b>cuda_runtime_api::cuobjdumpInit()</b><br> 注册回调函数: auto ctx_extract_code_func = [=]() { extract_code_using_cuobjdump(); };"];

    DumpInitInt["<b>cuda_runtime_api::cuobjdumpInit_internal()</b><br>"];

    Extract["<b>extract_code_using_cuobjdump()</b><br> 注册回调函数: auto ctx_extract_ptx_func = [=](CUctx_st *context) {extract_ptx_files_using_cuobjdump(context);};"];

    ExtractWrapper["<b>extract_code_using_cuobjdump_internal()</b><br> 判断是否设置CUOBJDUMP_SIM_FILE，若无则执行cuobjdump"];

    ExtractSingle["<b>extract_ptx_files_using_cuobjdump()<br>extract_ptx_files_using_cuobjdump_internal()</b><br>核心提取逻辑：<br>curr dir/<br>├── _cuobjdump_list_ptx_... <-- [ptx文件名列表]<br>├── my_app.1.sm_75.ptx    <-- [导出ptx]<br>└── my_app.2.sm_80.ptx    <-- [导出ptx]<br>并构建 version_filename 映射表(arch->filename)"];

    %% --- Subgraph: PTX Parsing (Direct File Load) ---
    RegFunc["<b>cudaRegisterFunctionInternal(...)</b><br> 注册 Kernel 时触发解析"];

    ParseBin["<b>gpgpu_context::cuobjdumpParseBinary(handle)</b><br> 从 version_filename 中找到最匹配当前 GPU 架构的 .ptx 文件名，解析并生成 symbol_table，并注册（context->add_binary）"];

    LoadFile["<b>gpgpu_ptx_sim_load_ptx_from_filename(...)</b><br> 打开并读取选定的 .ptx 文本文件"];
    PTXParse["<b>init_parser()</b><br> 解析ptx文件，构建ptx_instruction写入function_info，存入symbol_table"];

    PtxDecode["<b>ptx_parser->decode(...) / ptx_parse()</b><br> Bison/Yacc 生成 IR (function_info, symbol_table)"];

    PTXINFOLF["<b>gpgpu_ptx_info_load_from_filename(...)</b><br> 调用 ptxas 并解析 ptxinfo 文件，提取资源使用情况"];


    %% --- 边定义 (Edge Definitions / Call Chain) ---
    
    %% Main Flow
    Entry --> Internal;
    Internal --> Impl;

    %% Initialization Branch
    Impl --1. 初始化模拟器全局上下文--> InitCtx;
    Impl --2. 初始化底层 CUDA 上下文 (CUctx_st)--> InitSim;
    InitSim --> SimInit;

    %% Extraction Branch (Simplified for CUDA > 6.0)
    Impl --3. 提取ptx (CUDA 4.0+ 默认启用 cuobjdump)--> RegFatAPI;
    RegFatAPI --> DumpInit;
    DumpInit --> DumpInitInt;
    DumpInitInt --> Extract;
    Extract --> ExtractWrapper;
    ExtractWrapper --> ExtractSingle;

    %% Parsing Branch (Simplified: No Sections, No PTXPlus)
    Impl -.-> RegFunc;
    FuncEntry --> RegFunc;
    
    RegFunc --> ParseBin;
    
    %% 直接从 ParseBin 到 LoadFile (跳过 Section 查找)
    ParseBin --> LoadFile;
    
    LoadFile --> PTXParse;
    PTXParse --> PtxDecode;
    PtxDecode --写回Symbol Table--> PTXParse;

    ParseBin --> PTXINFOLF;
    %% --- 子图分组 (Subgraphs) ---
    
    subgraph Initialization
        InitCtx;
        InitSim;
        SimInit;
    end

    subgraph PTX_Extraction
        RegFatAPI;
        DumpInit;
        DumpInitInt;
        Extract;
        ExtractWrapper;
        ExtractSingle;
    end

    subgraph PTX_Parsing
        RegFunc;
        ParseBin;
        LoadFile;
        PtxDecode;
        PTXINFOLF;
        PTXParse;
    end

    %% 样式调整
    style ExtractSingle text-align:left,font-family:monospace
```

### PTX Opcode Parsing

<!-- ![PTX Opcode Parsing Flow](figs/ptx-parsing-calling-chain.png) -->

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
