#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real *h_x, real *d_x, const int method);


int main(int argc, char *argv[])
{
    int  method = std::atoi(argv[1]); // 使用 std::atoi 转换
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, method);



    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}

real reduce_global_syncthreads(real* d_x)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; //划分大小
    const int ymem = sizeof(real) * grid_size; //开空间计算后的y
    // const int smem = sizeof(real) * BLOCK_SIZE; //现成的大小
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem); //

    reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost)); //拷回host里
    real result = 0.0;
    for(int i = 0;i<grid_size;i++)
    {
        result +=h_y[i];
    }
    
    free(h_y);
    cudaFree(d_y);

    return result;

}

__global__ void reduce_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x; //线程的id
    const int bid = blockIdx.x; //块线程块的id
    const int n = bid * blockDim.x + tid; //实际拿到的总体的下标，也是要访问的
    __shared__ real s_y[128]; //线程块里的共享内存
    s_y[tid] = (n < N) ? d_x[n] : 0.0; //看看是否越界，如果没越界，就把要算的x塞进来
    //比如真实的是129，拿过来的线程id就是1，所以s_y[1]换成这个x
    __syncthreads(); //所有线程都把这个这个share放好后再说

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset]; //在逻辑上一样，只不过把这里给升级成共享内存了，所以很快
        }
        __syncthreads();//一次折完后，再缩短
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
    
}

real reduce_shared_syncthreads(real* d_x)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; //划分大小
    const int ymem = sizeof(real) * grid_size; //开空间计算后的y
    // const int smem = sizeof(real) * BLOCK_SIZE; //现成的大小
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem); //

    reduce_shared<<<grid_size,BLOCK_SIZE>>>(d_x,d_y);

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost)); //拷回host里
    real result = 0.0;
    for(int i = 0;i<grid_size;i++)
    {
        result +=h_y[i];
    }
    
    free(h_y);
    cudaFree(d_y);
    return result;
}

__global__ void  reduce_dynamic(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[]; //调用时需要参数
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

real reduce_shared_dynamic_syncthreads(real* d_x)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; //划分大小
    const int ymem = sizeof(real) * grid_size; //开空间计算后的y
    const int smem = sizeof(real) * BLOCK_SIZE; //要开的共享内存y的大小
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem); //

    reduce_dynamic<<<grid_size,BLOCK_SIZE,smem>>>(d_x,d_y);

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost)); //拷回host里
    real result = 0.0;
    for(int i = 0;i<grid_size;i++)
    {
        result +=h_y[i];
    }
    
    free(h_y);
    cudaFree(d_y);
    return result;
}





void timing(real *h_x, real *d_x, const int method)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);
        switch(method)
        {
            case 0:
            sum = reduce_global_syncthreads(d_x);
            break;
            case 1:
            sum = reduce_shared_syncthreads(d_x);
            break;
            case 3:
            sum = reduce_shared_dynamic_syncthreads(d_x);
            break;
            default:
            printf("wrong method\n");
            exit(-1);
            break;
        }
        
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