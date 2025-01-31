#define TILE_DIM 32
#define real double
#include <stdio.h>
__global__ void copy(const real*A, real *B, const int N)
{
    const int nx = blockIdx.x *TILE_DIM + threadIdx.x; 
    const int ny = blockIdx.y *TILE_DIM + threadIdx.y;
    
    const int index = ny*N + nx; //数据的实际位置

    if(nx<N && ny<N)
    {
        B[index] = A[index];
    }
}

void copyMatrix(double* d_A,double* d_B,const int N)
{
    //N是N*N矩阵的大小
    //TILE是我们一个线程块要处理的一个矩阵片，因为担心不能完美分片导致浪费，
    const int grid_size_x = (N+TILE_DIM-1)/TILE_DIM; //决定有几个线程块
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM,TILE_DIM); //每一片都哦是这么大 32*32 = 1024 正好一个块里的线程开满，每个线程都只处理一个复制
    const dim3 grid_size(grid_size_x,grid_size_y); //每个gird只处理一个片
    copy<<<grid_size,block_size>>>(d_A,d_B,N);
}
    