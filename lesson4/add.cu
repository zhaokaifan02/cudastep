#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono> //c++11的高精度时间库
// const double EPSILON = 1.0e-15; //不需要了
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;
void add(const real* x, const real *y, real* z, const int N);
void check(const real* z, const int N);
//哪怕没有核函数也可以用cu
int main()
{
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int N = 100000000;
    const int M = sizeof(real)*N; //开辟的数组内存
    real *x  = (real*)malloc(M);
    real *y = (real*)malloc(M);
    real *z = (real*)malloc(M); //开辟存储的数组
    for(int i = 0;i<N;i++)
    {
        x[i] =a;
        y[i] = b; 
    } //初始化数字


    cudaEventRecord(start); 
    cudaEventQuery(start);
    //执行的代码
    add(x,y,z,N);
    //结束的模板
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); //相当于清空缓冲器
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time,start,stop); //算出时间差
    printf("Time = %g ms \n", elapsed_time);
    //清空内存
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    check(z,N);

    
    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const real*x, const real*y, real*z, const int N)
{
    for(int i = 0;i<N;i++)
    {
        z[i] = x[i]+y[i];
    }
}

void check(const real* z,const int N)
{
    bool has_error = false;
    for(int i = 0;i<N;i++)
    {
        if(fabs(z[i] -c )>EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n",has_error?"HAS ERRORS":"NO ERRORS");
}

//cpu runtime : 1006144 us
//gpu runtime : 1306450 us