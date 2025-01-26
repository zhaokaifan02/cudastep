#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>

const double EPSLION = 1.0E-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
__global__ void add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

__global__ void add(const double *x, const double *y, double *z)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    // bid表示在第几个块，dim每个block有几块，
    // 这两相乘表示前面跳过了多少
    // threadIdx表示在这个块里前面跳了几个
    z[index] = x[index] + y[index];
    // 因为z是一个double*的 表示一次跳sizeofdouble个位置
}

// 其实这样写会有一点问题
//  当总元素个数N，是blockDim的整数倍，也就是说正好可以均等的 每份blockDim个
// 一共分成 N/blockDim个 grid， 也就是说gird size 为N/blockDIm

// 但这样问题就出现了，如果N不是blockDim的整数倍时会出现一下情况
// 当要处理的数组长度N = 10^8 + 1时 取一个线程块的最大线程数为N
//  100000000 / 128  = 781250
//  100000001 / 128 = 781250 ...1 并且计算机里的除是舍去小数的除法，所以真正开辟空间时，会把最后一个丢掉

// 因此为了避免这个情况，我们可以这样操作
//  int grid_size = (N-1)/block_size + 1; //在保留荣誉的情况下多开一个线程块，这样就能避免这个问题，同时还需要修改add操作

__global__ void add(const double *x, const double *y, double *z, const int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        z[i] = x[i] + y[i];
    }
}
// 这是一种常见的bug，编写代码时要注意

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    const int N = 100000001;
    const int M = sizeof(double) * N; // 开辟的数组内存
    double *cpu_x = (double *)malloc(M);
    double *cpu_y = (double *)malloc(M);
    double *cpu_z = (double *)malloc(M); // 开辟存储的数组
    for (int i = 0; i < N; i++)
    {
        cpu_x[i] = a;
        cpu_y[i] = b;
    } // 初始化数字,在cpu(ram)内存中开辟的

    double *gpu_x, *gpu_y, *gpu_z; // 开辟一个数组，作为GPU和CPU的媒介
    cudaMalloc((void **)&gpu_x, M);
    cudaMalloc((void **)&gpu_y, M);
    cudaMalloc((void **)&gpu_z, M);

    // gpu里也开辟内存
    cudaMemcpy(gpu_x, cpu_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, cpu_y, M, cudaMemcpyHostToDevice);
    const int block_size = 128;
    const int grid_size = (N - 1) / block_size + 1;
    // 执行
    add<<<grid_size, block_size>>>(gpu_x, gpu_y, gpu_z, N);
    // 放回cpu ram里
    cudaMemcpy(cpu_z, gpu_z, M, cudaMemcpyDeviceToHost);

    check(cpu_z, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "gpu runtime : " << duration.count() << " us" << std::endl;
    free(cpu_x);
    free(cpu_y);
    free(cpu_z);

    cudaFree(gpu_x);
    cudaFree(gpu_y);
    cudaFree(gpu_z);
    return 0;
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N; i++)
    {
        if (fabs(z[i] - c) > EPSLION)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "HAS ERRORS" : "NO ERRORS");
}