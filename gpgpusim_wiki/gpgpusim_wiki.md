# Notes on GPGPU-Sim

## Configuration & Statistics

*Meaning of config fields and statistics and how they relate to real gpu metrics*

## Code Reading

*E.g. key functions, calling chain for a memory access, how the memory bandwidth is modeled, etc.*

## Other Tips

*Other useful tips*

1. 系统有一套c++运行时环境，而anaconda虚拟环境中也有一套c++运行时环境，命令行中（ldd,g++,nvcc等）默认使用的是系统环境，而python运行时使用的是anaconda环境，如果两者版本不匹配就有可能链接出错。查看版本的方法是：strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_。如果anaconda环境的版本过低，应该使用conda install -c conda-forge libstdcxx-ng来升级。