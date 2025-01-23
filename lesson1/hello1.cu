#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int gDim = gridDim.x; //相当于一个内置变量获取传入的 gridDim
    const int bDim = blockDim.x; //内置变量获得blockDim
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    //内建变量
    //blockIdx.x 表示这个线程在哪个块 gridDim表示有几个快 从0到gridDim.x-1
    //threadIdx.x 表示在表示在具体的block里的哪一个 从0 到 blockDimx-1
    printf("HELLO WORLD GPU! block %d and thread %d!\n",bid,tid);
    /**
     * 有时候先1有时候先0
     * 说明线程块的执行是独立的，先执行完的那个线程块先输出，
    */

   /**
    * 推广多网络这种blockDim.x 这个是很像c++的结构体的
   */
  
}

int main(void)
{
    hello_from_gpu<<<2,4>>>();
    cudaDeviceSynchronize();
    return 0;
}
//<<<线程块个数， 每个线程块里线程的个数
// 或者理解为网格大小, 线程块大小
// 网格里放着线程块，线程块里放着线程
//<<改成2 4 后相当于创建了2个线程块，每个线程块里4个线程，总共8个线程
//所有输出8条hello gpu
//<<<grid_size,block_size>>>

