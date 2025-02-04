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

void read_xy(std::vector<real>& x, std::vector<real>& y);
void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y);
void print_neighbor(const int *NN, const int *NL);

int main(void)
{
    std::vector<real> x, y; //开两个vector，记录坐标在一个平面上x和y的坐标用两个向量存储 //vector的长度就是原子的个数
    read_xy(x, y); 
    N = x.size();
    int *NN = (int*) malloc(N * sizeof(int)); //表示邻居的个数
    int *NL = (int*) malloc(N * MN * sizeof(int)); //n*MN+k =拿到的是第n个原子的第k个坐标，就是二维数组的一维本质
    
    timing(NN, NL, x, y);
    print_neighbor(NN, NL);

    free(NN);
    free(NL);
    return 0;
}

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

void find_neighbor(int *NN, int *NL, const real* x, const real* y)
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

void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y)
{
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        while(cudaEventQuery(start)!=cudaSuccess){}
        //执行
        find_neighbor(NN, NL, x.data(), y.data());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
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