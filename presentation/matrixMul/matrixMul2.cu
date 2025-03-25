#include <stdio.h>
#include <iostream>
#define N 2048 // 矩阵大小 N x N

// CUDA 核函数：执行矩阵乘法
#define TILE_SIZE 32  // 每个线程块的大小

__global__ void matrixMulOptimized(int *A, int *B, int *C, int n) {
    __shared__ int tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;

    // 遍历子矩阵块
    for (int k = 0; k < n / TILE_SIZE; k++) {
        // **线程块中的线程共同加载 A 和 B 的数据到共享内存**
        tile_A[threadIdx.y][threadIdx.x] = A[row * n + (k * TILE_SIZE + threadIdx.x)];
        tile_B[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads(); // **同步，确保所有线程都加载完毕**

        // **计算该线程的结果**
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads(); // **同步，确保所有线程都用完当前共享内存**
    }

    // **写入结果**
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}


int main()
{
    int size = N * N * sizeof(int);
    int *h_A  = (int*)malloc(size);// 示例矩阵 A
    int *h_B = (int*)malloc(size); // 示例矩阵 B
    int *h_C = (int*)malloc(size); // 结果矩阵 C
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

    dim3 threadsPerBlock(32, 32); // 每个线程块有16乘16个线程
    dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    matrixMulOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 结束的模板
    cudaDeviceSynchronize();
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); // 相当于清空缓冲器
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop); // 算出时间差
    printf("GPU:Time = %g ms \n", elapsed_time);
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
