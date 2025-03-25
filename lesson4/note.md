# 编译时增加设置

```
#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif
```

最前面指定，加编译时可以加上

```
nvcc -USE_DP add.cu -o addDouble.exe  
```
手动设置real的位置

float 219.633ms
double 230.ms
float应该比double 快一点


# 在之前cuda的里

checkapi里的 app函数
double: 18.6512
float: 5.68195

# cuda程序高性能的必要
1. 数据传输比例小，尽可能的减少memcpu从 host到device的
2. 核函数里的算数强度越高越好
3. 核函数中定义的线程数目多，增大并发性

# 优化时的思路
1. 减少主机与设备的数据传输
2. 提高核函数算数强度
3. 增大并行规模

# 和函数里的cuda内置函数
cmath 里的库，只能在cpu ram内存上用
cuda 里的库，就可以在device上的数据用了
cuda-math-api里使用（直接gpt就可以了）