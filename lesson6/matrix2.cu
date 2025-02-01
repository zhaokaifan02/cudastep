#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;  //避免出现误差的平均执行次数
const int TILE_DIM = 32; //一个片的大小

void timing(const real *d_A, real *d_B, const int N, const int task);
__global__ void copy(const real *A, real *B, const int N);
__global__ void transpose1(const real *A, real *B, const int N);
__global__ void transpose2(const real *A, real *B, const int N);
__global__ void transpose3(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: %s N\n", argv[0]); //argv[0]表示当前的程序的名字
        exit(1);
    }
    const int N = atoi(argv[1]); //数组的单边大小

    const int N2 = N * N; //实际数组的大小，每个元素都是一个real
    const int M = sizeof(real) * N2; //开这个数组，逻辑上二维，物理上一维
    real *h_A = (real *) malloc(M); //开辟数组
    real *h_B = (real *) malloc(M);
    for (int i = 0; i < N2; ++i)
    {
        h_A[i] = i; //数组初始化
    }
    real *d_A, *d_B; //device里的数据
    CHECK(cudaMalloc(&d_A, M)); //开辟空间
    CHECK(cudaMalloc(&d_B, M)); //开辟空间， 时刻把握逻辑上的二维，物理上的一维
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice)); //拷贝过来

    printf("\ncopy:\n");
    timing(d_A, d_B, N, 0);
    printf("\ntranspose with coalesced read:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose with coalesced write:\n");
    timing(d_A, d_B, N, 2);
    printf("\ntranspose with coalesced write and __ldg read:\n");
    timing(d_A, d_B, N, 3);

    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
    if (N <= 10)
    {
        printf("A =\n");
        print_matrix(N, h_A);
        printf("\nB =\n");
        print_matrix(N, h_B);
    }

    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}
// 经典的测试模板
void timing(const real *d_A, real *d_B, const int N, const int task)
{
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM; //计算开几个块
    const int grid_size_y = grid_size_x; //
    const dim3 block_size(TILE_DIM, TILE_DIM); //二维的线程，都是逻辑上的 好理解
    const dim3 grid_size(grid_size_x, grid_size_y); // 二维的线程块

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (task)
        {
            case 0:
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 3:
                transpose3<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop)); //执行消耗的时间
        printf("Time = %g ms.\n", elapsed_time);

        if (repeat > 0)
        {
            t_sum += elapsed_time; //计算执行消耗时间和
            t2_sum += elapsed_time * elapsed_time; //t2范数
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS; //平均一下
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave); //方差
    printf("Time = %g +- %g ms.\n", t_ave, t_err);
}

__global__ void copy(const real *A, real *B, const int N)
{
    //blockId(0,0) threadId(0,0)
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x; //比如blockId是(0,0)表示第一个block，总偏移就是0，线程id
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y; //y方向偏移量，从逻辑上更好看了
    const int index = ny * N + nx; //算出x方向偏移+y方向偏移，物理杀那个就是ny*N，因为一行的实际长度是N +nx
    if (nx < N && ny < N)
    {
        B[index] = A[index]; //完成复制任务
    }
}

__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
    //从A中读取是顺序的，B中写入不是顺序的
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
    //A中读取不是顺序的，B中写入是顺序的
}

__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]); //在使用开普勒与麦克斯韦架构时需要使用到这个
    }
}

void print_matrix(const int N, const real *A)
{
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            printf("%g\t", A[ny * N + nx]);
        }
        printf("\n");
    }
}

//总结，在拿去全局内存时，最核心的优化思路就是
//1. 读写时合并访问
//2. 用共享内存