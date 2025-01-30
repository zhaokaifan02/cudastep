#include <stdio.h>

__device__ int d_x = 1;
__device__ int d_y[2]; //静态内存

__global__ void my_kernel(const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;// 在和函数里开辟的就是寄存器。
    //一个寄存器占有32bit 4字节所以一个int变量正好站一个寄存器
    //开一个double的最要两个寄存器，他是最快的
    //这个线程结束，这个寄存器的生命周期就结束

    //局部内存
    double a[20]; //寄存器放不下的就在局部内存里 local memory
    
    d_y[0] +=d_x;
    d_y[1] += d_x;
    printf("d_x = %d,  d_y[0] = %d,  d_y[1] = %d .\n",d_x,d_y[0],d_y[1]);
}

int main()
{
    int h_y[2] = {10,20};
    cudaMemcpyToSymbol(d_y,h_y,sizeof(int)*2); //修改全局内存
    const int N = 2; //常量内存，适合在传输大小时使用的。一个核函数最多4kb
    my_kernel<<<1,1>>>(N); //就这样用，可读不可写
    cudaDeviceSynchronize(); //清空缓存

    cudaMemcpyFromSymbol(h_y, d_y,sizeof(int)*2);
    printf("h_y[0] = %d, h_y[1] = %d .\n", h_y[0],h_y[1]);
    return 0;
}