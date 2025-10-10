1. 系统有一套c++运行时环境，而anaconda虚拟环境中也有一套c++运行时环境，命令行中（ldd,g++,nvcc等）默认使用的是系统环境，而python运行时使用的是anaconda环境，如果两者版本不匹配就有可能链接出错。查看版本的方法是：strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_。如果anaconda环境的版本过低，应该使用conda install -c conda-forge libstdcxx-ng来升级。

2. 信号量的行为和官方文档不完全一致。通过一个简单的[Producer and Consumer demo](producer-consumer-demo.cu)测试可以发现，其真实行为满足：
    - mbarrier的初始相位为true。
    - 只要phaseParity和当前mbarrier的相位相同，就将waitComplete置为true，而不管pending arrival计数器的值。
    - 每执行一次arrive操作，pending arrival计数器就减一，如果pending arrival计数器归0，就触发相位反转，然后将pending arrival计数器重置为expected arrival。
