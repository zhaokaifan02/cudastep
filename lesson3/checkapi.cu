#include "error.cuh"
//又是一个编译小贴士，cu需要先编译成-o，而cuh不需要提前编译，这点和cpp是一样的
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
__global__ void add(const double* x, const double *y, double *z, const int N);
void check(const double* z, const int N);

int main()
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*)malloc(M);
    double *h_y = (double*)malloc(M);
    double *h_z = (double*)malloc(M);

    for(int i = 0;i<N;i++)
    {
        h_x[i] = a;
        h_y[i] = b;
    }

    double *d_x,*d_y,*d_z;
    //纯套路开辟空间
    CHECK(cudaMalloc((void**)&d_x,M));
    CHECK(cudaMalloc((void**)&d_y,M));
    CHECK(cudaMalloc((void**)&d_z,M));

    CHECK(cudaMemcpy(d_x,h_x,M,cudaMemcpyDeviceToHost)); //这里应该会报错，因为d_x是cudaMalloc里的，在device上，写反了
    CHECK(cudaMemcpy(d_y,h_y,M,cudaMemcpyHostToDevice));
    const int block_size = 128;
    const int grid_size = (N-1)/128 + 1;

    add<<<grid_size,block_size>>>(d_x,d_y,d_z,N);

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



__global__ void add(const double *x, const double *y, double *z, const int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        z[i] = x[i] + y[i];
    }
}

void check(const double *z, const int N)
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