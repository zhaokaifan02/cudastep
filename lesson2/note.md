# record time
```
#include <chrono>
auto start = std::chrono::high_resolution_clock::now();
//code

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end -start);
std::cout << "run time: " << duration.count() << " us" << std::endl;
```

# 执行空间标识符
__global__ 叫核函数 主机调用，设备中执行。如果使用动态并行，可以在核函数中调用自己或其他核函数
__device__ 设备函数，只能被核函数或者其他设备函数调用，在设备中执行
__host__ 就是主机端的普通函数

对于一个函数 它可以同时被 __host__ 和 __device__ 同时修饰
既可以被标准c++调用，也可以被主机里函数调用，减少冗余代码(但没啥用)

__device__ 和 __global__ 不能同时使用
__host__ 和 __global__ 也不能同时使用