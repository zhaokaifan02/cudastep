#include "error.cuh"
#include <cuda_runtime.h>
//cuda有自己的及时方式 cuda event
int main()
{
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); 
    cudaEventQuery(start);

    //需要及时的代码模块

    //结束的模板
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); //相当于清空缓冲器
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop); //算出时间差
    printf("Time = %g ms \n", elapsed_time);
    //清空内存
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


}

