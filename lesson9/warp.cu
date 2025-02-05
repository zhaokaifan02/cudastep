#include "error.cuh"
#include <stdio.h>

const unsigned WIDTH = 8;
const unsigned BLOCK_SIZE = 16;
const unsigned FULL_MASK = 0xffffffff; 

__global__ void  test_warp_primitives(void);

int main(int argc, char **argv)
{
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK(cudaDeviceSynchronize()); //用printf时必要的
    return 0;
}

__global__ void  test_warp_primitives(void)
{
    int tid = threadIdx.x; //线程id
    int lane_id = tid % WIDTH; //块内id
    //WIDTH理解为一种逻辑warp分割，真实的warp永远是32不可改的，但是我们可以人为调整来理解代码
    if (tid == 0) printf("threadIdx.x: ");
    printf("%2d ", tid); 
    if (tid == 0) printf("\n");

    if (tid == 0) printf("lane_id:     ");
    printf("%2d ", lane_id);
    if (tid == 0) printf("\n");

    unsigned mask1 = __ballot_sync(FULL_MASK, tid > 0); //参与的同一个warp内，标记所有tid>0的
    unsigned mask2 = __ballot_sync(FULL_MASK, tid == 0);//标记tid等于0的

    if (tid == 0) printf("FULL_MASK = %x\n", FULL_MASK);
    if (tid == 1) printf("mask1     = %x\n", mask1);
    if (tid == 0) printf("mask2     = %x\n", mask2);

    int result = __all_sync(FULL_MASK, tid); //如果所有参与现成的tid都不为0时才返回1
    if (tid == 0) printf("all_sync (FULL_MASK): %d\n", result);

    result = __all_sync(mask1, tid); //参与线程都不为0才返回1
    if (tid == 1) printf("all_sync     (mask1): %d\n", result);

    result = __any_sync(FULL_MASK, tid); //参与线程有一个不都为0时才返回1
    if (tid == 0) printf("any_sync (FULL_MASK): %d\n", result);

    result = __any_sync(mask2, tid);  //参与线程有一个不都为0时才返回1
    if (tid == 0) printf("any_sync     (mask2): %d\n", result);

    int value = __shfl_sync(FULL_MASK, tid, 2, WIDTH); //获得束内编号为2的线程的tid
    if (tid == 0) printf("shfl:      ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH); //获得t-1现成的tid值
    if (tid == 0) printf("shfl_up:   ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH); //获得t+1线程的tid值
    if (tid == 0) printf("shfl_down: ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");

    value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH); //
    if (tid == 0) printf("shfl_xor:  ");
    printf("%2d ", value);
    if (tid == 0) printf("\n");
}
