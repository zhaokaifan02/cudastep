#include <stdio.h>
#include <iostream>
#define N 2048 // 矩阵大小 N x N

void matrixMulCpu(int *A, int *B, int *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < n; k++)
            {
                int a = A[i * n + k];
                int b = B[k * n + j];
                sum += a * b;
            }
            C[i * n + j] = sum;
        }
    }
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    matrixMulCpu(h_A, h_B, h_C, N);

    // 结束的模板
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 相当于清空缓冲器
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop); // 算出时间差
    printf("CPU: Time = %g ms \n", elapsed_time);
    // 清空内存
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}