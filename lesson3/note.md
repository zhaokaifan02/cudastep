# cuda调试

正常情况下，所有的核函数，是在不同线程里并行计算的，但是这不好调试，有两种策略，将异步的转化为同步的

1. 在要调试的核函数里加上
```
cudaDeviceSynchronize(); 
```

2. 将临时环境变量设置
```
CUDA_LAUNCH_BLOCKING = 1;
```

