#include "error.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

int N; // number of atoms
const int NUM_REPEATS = 20; // number of timings
const int MN = 10; // maximum number of neighbors for each atom 邻居的最大个数
const real cutoff = 1.9; // in units of Angstrom 判断是否为邻居的参数
const real cutoff_square = cutoff * cutoff; //A^2

void read_xy(std::vector<real>& v_x, std::vector<real>& v_y)
{
    std::ifstream infile("xy.txt");
    std::string line, word;
    if(!infile)
    {
        std::cout << "Cannot open xy.txt" << std::endl;
        exit(1);
    }
    while (std::getline(infile, line))
    {
        std::istringstream words(line); //分词
        if(line.length() == 0)
        {
            continue;
        } //如果读到空行就继续执行
        for (int i = 0; i < 2; i++) //每一行就两个元素
        {
            if(words >> word)
            {
                if(i == 0)
                {
                    v_x.push_back(std::stod(word)); //写进x
                }
                if(i==1)
                {
                    v_y.push_back(std::stod(word)); //写进y
                }
            }
            else
            {
                std::cout << "Error for reading xy.txt" << std::endl;
                exit(1);
            }
        }
    }
    infile.close(); //关掉读取的线程
}

void find_neighbor_cpu(int *NN, int *NL, const real* x, const real* y)
{
    for (int n = 0; n < N; n++)
    {
        NN[n] = 0;  //原子的邻居个数初始化都为0
    }

    for (int n1 = 0; n1 < N; ++n1) //处理所有的原子
    {
        real x1 = x[n1]; //拿到要处理的原子的坐标
        real y1 = y[n1];
        for (int n2 = n1 + 1; n2 < N; ++n2)  //对称矩阵不需要从0遍历，从我的下一个开始遍历
        {
            real x12 = x[n2] - x1;  //
            real y12 = y[n2] - y1; //xy的差
            real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                NL[n1 * MN + NN[n1]++] = n2; //n1的邻居加上n2这个坐标，同时n1的邻居++
                NL[n2 * MN + NN[n2]++] = n1; //因为是对称的，所以同样的n2的邻居也是n1
            }
        }
    }
}

__global__ void find_neighbor_gpu_atomic(int *d_NN,int *d_NL, const real*d_x, const real* d_y, const int N, const real cutoff_suqre)
{
    int index = threadIdx.x + blockDim.x*blockIdx.x; //拿到要处理的那个下标
    if(index<N)
    {
        d_NN[index] = 0;
        real x1 = d_x[index];
        real y1 = d_y[index];
        for(int n = index+1;n<N;++n)
        {
            real x12 =  d_x[n] - x1;
            real y12 =  d_y[n] - y1;
            real distance_square = x12*x12 + y12*y12;
            if(distance_square<cutoff_suqre)
            {
                //可以被作为
                const int temp1 = atomicAdd(&d_NN[index],1); //temp1的返回值是原来的值
                d_NL[index*MN + temp1] = n;
                const int temp2 = atomicAdd(&d_NN[n],1);
                d_NL[n*MN+temp2] = index;
            }
        }
    }
}

void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y,const int N)
{
    real* h_x = (real*)malloc(sizeof(real)*N);
    real* h_y = (real*)malloc(sizeof(real)*N);
    for(int i = 0;i<x.size();i++)
    {
        h_x[i] = x[i];
        h_y[i] = y[i];
    }//初始化h_x与h_y

    real *d_x,*d_y;
    int *d_NN,*d_NL;
    cudaMalloc((void**)&d_x,sizeof(real)*N);
    cudaMalloc((void**)&d_y,sizeof(real)*N);
    cudaMalloc((void**)&d_NN,sizeof(int)*N);
    cudaMalloc((void**)&d_NL,sizeof(int)*N*MN);
    cudaMemcpy(d_x,h_x,sizeof(real)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,sizeof(real)*N,cudaMemcpyHostToDevice);
    const int grid_size = (N-1)/128 + 1;
    const int block_size = 128;
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        while(cudaEventQuery(start)!=cudaSuccess){}
        //执行
        find_neighbor_gpu_atomic<<<grid_size,block_size>>>(d_NN,d_NL,d_x,d_y,N,cutoff_square);
        
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    cudaMemcpy(NN,d_NN,sizeof(int)*N,cudaMemcpyDeviceToHost);
    cudaMemcpy(NL,d_NL,sizeof(int)*N*MN,cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_NL);
    cudaFree(d_NN);
}

void print_neighbor(const int *NN, const int *NL)
{
    std::ofstream outfile("neighbor.txt");
    if (!outfile)
    {
        std::cout << "Cannot open neighbor.txt" << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        if (NN[n] > MN) //如果n大于MN了，说明邻居设少了，不符合现实情况
        {
            std::cout << "Error: MN is too small." << std::endl;
            exit(1);
        }
        outfile << NN[n]; //邻居个数写进去
        for (int k = 0; k < MN; ++k)
        {
            if(k < NN[n]) //不断将这个原子的邻居写进去
            {
                outfile << " " << NL[n * MN + k]; //得重新学一下王道的数组了
            }
            else
            {
                outfile << " NaN";
            }
        }
        outfile << std::endl;
    }
    outfile.close();
}

int main(void)
{
    std::vector<real> x, y; //开两个vector，记录坐标在一个平面上x和y的坐标用两个向量存储 //vector的长度就是原子的个数
    read_xy(x, y); 
    N = x.size();
    int *NN = (int*) malloc(N * sizeof(int)); //表示邻居的个数
    int *NL = (int*) malloc(N * MN * sizeof(int)); //n*MN+k =拿到的是第n个原子的第k个坐标，就是二维数组的一维本质
    
    timing(NN, NL, x, y,N);
    print_neighbor(NN, NL);

    free(NN);
    free(NL);
    return 0;
}