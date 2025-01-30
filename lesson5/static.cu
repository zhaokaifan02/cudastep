#include <stdio.h>

__device__ int d_x = 1;
__device__ int d_y[2]; //静态内存

__global__ void my_kernel()
{
    d_y[0] +=d_x;
    d_y[1] += d_x;
    printf("d_x = %d,  d_y[0] = %d,  d_y[1] = %d .\n",d_x,d_y[0],d_y[1]);
}

int main()
{
    int h_y[2] = {10,20};
    cudaMemcpyToSymbol(d_y,h_y,sizeof(int)*2); //修改全局内存
    my_kernel<<<1,1>>>();
    cudaDeviceSynchronize(); //清空缓存

    cudaMemcpyFromSymbol(h_y, d_y,sizeof(int)*2);
    printf("h_y[0] = %d, h_y[1] = %d .\n", h_y[0],h_y[1]);
    return 0;
}