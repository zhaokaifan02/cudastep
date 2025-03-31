#include <stdio.h>
#include <iostream>
#define N 2048 // 矩阵大小 N x N

// CUDA 核函数：执行矩阵乘法
__global__ void matrixMul(int *A, int *B, int *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int k = 0; k < n; k++)
    {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

int main()
{
    int size = N * N * sizeof(int);
    int *h_A = (int *)malloc(size); // 示例矩阵 A
    int *h_B = (int *)malloc(size); // 示例矩阵 B
    int *h_C = (int *)malloc(size); // 结果矩阵 C
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1;
        h_B[i] = 1;
    }

    int *d_A, *d_B, *d_C; // 显卡里的
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // 拷贝到GPU中
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32); // 每个线程块有32乘32 个线程
    dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 结束的模板
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 相当于清空缓冲器
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop); // 算出时间差
    printf("GPU：Time = %g ms \n", elapsed_time);
    // 清空内存
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
