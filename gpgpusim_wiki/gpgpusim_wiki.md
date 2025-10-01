# Notes on GPGPU-Sim

## Configuration & Statistics

*Meaning of config fields and statistics and how they relate to real gpu metrics*

Reference: [GPGPU-sim Doc](http://gpgpu-sim.org/manual/index.php/Main_Page#Configuration_Options)

| 5090 |  GPGPU-sim |  Config Value | Note |
| -- | -- | -- | -- |
| multiprocessor_count | gpgpu_n_clusters | 170 | |
| base clock, memory_clock_rate | gpgpu_clock_domains | 2010:2010:2010:14000 | \<Core Clock\>:\<Interconnect Clock\>:\<L2 Clock\>:\<DRAM Clock\>  |
| limits_max_cta_per_sm | gpgpu_shader_cta | 24 | |


## Code Reading

*E.g. key functions, calling chain for a memory access, how the memory bandwidth is modeled, etc.*

## Other Tips

*Other useful tips*

1. 系统有一套c++运行时环境，而anaconda虚拟环境中也有一套c++运行时环境，命令行中（ldd,g++,nvcc等）默认使用的是系统环境，而python运行时使用的是anaconda环境，如果两者版本不匹配就有可能链接出错。查看版本的方法是：strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_。如果anaconda环境的版本过低，应该使用conda install -c conda-forge libstdcxx-ng来升级。

2. ncu Profiling可以用以下两种方式之一来锁定频率，但都不是hard constraint，最终SM Frequency（测量值）仍会显著小于设定值（目前观测的max gap: 1.79GHz/2.01GHz）([ref](https://forums.developer.nvidia.com/t/sm-frequency-reported-in-nsight-compute/264271))
    1. ncu默认将clock锁定到base clock (5090为2.01GHz)，可以不进行任何设置；也可以通过`ncu --clock-control base`显式指定 ([ref](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control))
    2. 先用`nvidia-smi -i 0 -pm 1`启用persistence mode，再用`nvidia-smi -lgc minGpuClock,maxGpuClock`指定频率，同时设定`ncu --clock-control none` ([ref](https://stackoverflow.com/questions/64701751/can-i-fix-my-gpu-clock-rate-to-ensure-consistent-profiling-results))