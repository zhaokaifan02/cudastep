#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

const int NUM_REPEATS = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main()
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real *)malloc(M);
    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.23;
    }

    timing(x, N);

    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}

real reduce(const real *x, const int N)
{
    real sum = 0.0;
    for (int i = 0; i < N; ++i)
    {
        sum += x[i];
    }
    return sum;
}
//单精度float 时会出问题 因为float的精度问题,大数吃小数这种情况
// 双精度double， 只是最后几问有问题


//思路
//binary reduction
//设置一个offset 偏置，设置为长度的一半 比如10个元素，长度就是offset就是5
//比如0的offset就是 0+5 对应5 ，全部折叠一下就完成了结果
//全部折叠一次后， N变成了N/2，再offset折一半
__global__ void reduce(real *d_x, int N)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    for(int offset = N/2; offset>0;offset/=2)
    {
        if(i<offset)
            d_x[i] += d_x[i+offset];
    }
}
//这样有问题，这种想法是顺序的，建立在理想情况下
//若干线程同时执行完第一次折半
//再同时执行第二次折半
//但实际情况确实，第一个线程还没折完，第二个就折第二次了，结果往往是错的

//cuda提供了一种同步函数
__global__ void reduce_global(real* d_x, real* d_y) 
{
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x; 
    //d_x 是数组的首地址， blockDim.x* blockIdx.x 是当block线程块拿到的首地址
    for(int offset = blockDim.x>>1; offset>0;offset>>=1)
    {
        //>>1 相当于÷2
        if(tid<offset)
        {
            x[tid] +=x[tid+offset];
        }
        __syncthreads(); //同步单线程块里的线程，每个线程块缩减offset时，都要等待所有的线程执行完这次求和
    }
    if(tid==0)
    {
        d_y[blockIdx.x] = x[0];
    }
    //最终结果是把线程块给从100000000 缩小到了 GIRD_szie ,只求出了单个线程块的和
}