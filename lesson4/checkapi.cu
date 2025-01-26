#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

#include "error.cuh"
#include <math.h>
#include <stdio.h>

const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
__global__ void add(const real* x, const real *y, real *z, const int N);
void check(const real* z, const int N);

int main()
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *h_x = (real*)malloc(M);
    real *h_y = (real*)malloc(M);
    real *h_z = (real*)malloc(M);

    for(int i = 0;i<N;i++)
    {
        h_x[i] = a;
        h_y[i] = b;
    }

    real *d_x,*d_y,*d_z;
    //纯套路开辟空间
    CHECK(cudaMalloc((void**)&d_x,M));
    CHECK(cudaMalloc((void**)&d_y,M));
    CHECK(cudaMalloc((void**)&d_z,M));

    CHECK(cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpy(d_y,h_y,M,cudaMemcpyHostToDevice));
    const int block_size = 128;
    const int grid_size = (N-1)/128 + 1;

    //计时模板1
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); 
    cudaEventQuery(start);

    add<<<grid_size,block_size>>>(d_x,d_y,d_z,N);
    //即时模板2
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); //相当于清空缓冲器
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop); //算出时间差
    printf("Time = %g ms \n", elapsed_time);
    //清空内存
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
    CHECK(cudaMemcpy(h_z,d_z,M,cudaMemcpyDeviceToHost));
    check(h_z,N);

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    CHECK(cudaFree(d_z));
    return 0;
}



__global__ void add(const real *x, const real *y, real *z, const int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        z[i] = x[i] + y[i];
    }
}

void check(const real *z, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N; i++)
    {
        if (fabs(z[i] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "HAS ERRORS" : "NO ERRORS");
}