#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>

const double EPSLION = 1.0E-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
__global__ void add(const double*x, const double* y,double* z);
void check (const double* z, const int N);
int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    const int N = 100000000;
    const int M = sizeof(double)*N; //开辟的数组内存
    double *cpu_x  = (double*)malloc(M);
    double *cpu_y = (double*)malloc(M);
    double *cpu_z = (double*)malloc(M); //开辟存储的数组
    for(int i = 0;i<N;i++)
    {
        cpu_x[i] =a;
        cpu_y[i] = b; 
    } //初始化数字,在cpu(ram)内存中开辟的

    double *gpu_x, *gpu_y, *gpu_z; //开辟一个数组，作为GPU和CPU的媒介

    cudaMalloc((void **)&gpu_x,M); 
    cudaMalloc((void**)&gpu_y,M);
    cudaMalloc((void**)&gpu_z,M); //通过这个媒介和cpu交互
    //cudaMalloc 指向指针的指针， 
    // double* d_x 理解为本地开一个double* 类型的地址，这个double* 放的是数组的受地震
    // &d_x 数组首地址的地址，然后转化为void **
    //像gpu里分配一块内存

    cudaMemcpy(gpu_x,cpu_x,M,cudaMemcpyHostToDevice); //从host到device
    //这里就和pytorch一样了，用device来表述gpu
    cudaMemcpy(gpu_y,cpu_y,M,cudaMemcpyHostToDevice);
    //把这两个装载到gpu上
    //cudaMemcpy(目标，起点，长度,方法)
    const int block_size = 128; //先设定一个一个线程块里最多128个
    const int grid_size = N/block_size; //因为一共N个元素，所以分成N/128个线程块
    //相当于每个线程只处理一个加法
    add<<<grid_size,block_size>>>(gpu_x,gpu_y,gpu_z); //传给和函数的指针必须指向device内存 就是GPU里的 
    //核函数不能成为一个类的成员
    //以前核函数之间不能相互调用
    //3.5之后就可以了  动态并行机制 dynamic parallelism
    cudaMemcpy(cpu_z,gpu_z,M,cudaMemcpyDeviceToHost); //再把GPU拿回CPU


    check(cpu_z,N);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end -start);
    std::cout << "gpu runtime : " << duration.count() << " us" << std::endl;
    
    
    //内存销毁
    free(cpu_x);
    free(cpu_y);
    free(cpu_z);
    cudaFree(gpu_x);
    cudaFree(gpu_y);
    cudaFree(gpu_z);
    return 0;
}
//gpu runtime : 1306450 us
__global__ void add(const double *x,const  double* y, double* z) 
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x; 
    //bid表示在第几个块，dim每个block有几块，
    //这两相乘表示前面跳过了多少
    //threadIdx表示在这个块里前面跳了几个
    z[index] = x[index] + y[index];
    //因为z是一个double*的 表示一次跳sizeofdouble个位置
}
void check(const double* z,const int N)
{
    bool has_error = false;
    for(int i = 0;i<N;i++)
    {
        if(fabs(z[i] -c )>EPSLION)
        {
            has_error = true;
        }
    }
    printf("%s\n",has_error?"HAS ERRORS":"NO ERRORS");
}
