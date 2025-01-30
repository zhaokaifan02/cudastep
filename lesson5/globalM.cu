#include <stdio.h>

int main()
{
    double* h_x;
    double *d_x,*d_y;
    const int N = 100000000;
    const int M = sizeof(double1)*N;
    h_x = (double*)malloc(M);
    for(int i = 0;i<N;i++)
    {
        h_x[i] = 1.2;
    }
    cudaMalloc((void**)&d_x,M);
    cudaMalloc((void**)&d_y,M);

    cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice); //主机拷贝到device 显存中
    cudaMemcpy(d_y,d_x,M,cudaMemcpyDeviceToDevice); //显存到显存
    //全局内存可读可写，只要进了显存，在核函数中就是共享的

    //gird上的所有block的线程，都可以访问和函数的数据

    //所以全局内存上的数据，生命周期在主机
    //主机cudaMalloc分配
    //cudaFree 结束
    cudaFree(d_x);
    cudaFree(d_y);

    //同样的我们可以不采取memcpy的方法，
    //在定义函数时，直接将其放到device上

    __device__ double x;
    x = 5;
    __device__ double* x_array[N]; //固定长度数组，直接开辟到空间上

    //含函数中，可以直接对静态全局内存变量进行访问，不需要将其以参数形式传给核函数。不可再主机函数中访问直接访问静态内存变量，但是可以拷贝比如

    

}