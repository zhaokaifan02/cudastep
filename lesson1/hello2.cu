#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bx = blockIdx.x; //这个blockIdx理解为为java的内置变量，有自己的属性
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    printf("Hello world from GPU! from block( %d %d %d ), thread( %d %d)\n", bx,by ,bz, tx, ty);      
}

int main()
{
    const dim3 grid_size(3,3,3); 
    //girdsize可以理解为一个三个元素的结构体，比如这里就是开辟了一个3*3*3像立方体一样的网格块，一共9个线程块
    const dim3 block_size(2,4);
    /**
     * 这相当于9个线程块里的每一个都是2*4 一共8个线程
     * 逻辑结构上是一个3*3*3的三阶魔方，每个方块里，都是一个2*4的平面
     * 为什么说是逻辑结构呢，因为实际上物理存储中，他并不是在内存中真的开一个3*3*3
     * 因为在某些边角上会有极大的浪费
    */

    hello_from_gpu<<<grid_size,block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}

/**
 * 受限于图灵架构的GPU限制
 * 网格大小的限制为
 * x 2^31- 1
 * y 2^16-1 65536
 * z 2^16-1 65536
 * 也就是说我们可以开辟的线程块的个数是有限的
 * 
 * 每个线程块里所放的线程数最多为1024个，其中在此前提下，依然有一些单方向要求
 * x 1024
 * y 1024
 * z 64
 * 单个线程块一定只有1024个线程！！ 
*/

/**
 * 新问题来了，逻辑上像一个模仿一样的线程块。物理内存到底是怎么排的呢
 * 像多维数组一样，本质上还是一维的，取决于按行还是按列存储
 * 回忆回忆，前面讲的
 * gridDim和blockDim，如果他是三维的话，那么
 * 真实的线程id为
 * int tid = threadIdx.z *
*/