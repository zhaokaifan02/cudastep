#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono> //c++11的高精度时间库
const double EPSILON = 1.0e-15; //特殊的表示法表示0.0001 15个0
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void add(const double* x, const double *y, double* z, const int N);
void check(const double* z, const int N);

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    const int N = 100000000;
    const int M = sizeof(double)*N; //开辟的数组内存
    double *x  = (double*)malloc(M);
    double *y = (double*)malloc(M);
    double *z = (double*)malloc(M); //开辟存储的数组
    for(int i = 0;i<N;i++)
    {
        x[i] =a;
        y[i] = b; 
    } //初始化数字
    
    add(x,y,z,N);
    check(z,N);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end -start);
    std::cout << "cpu runtime : " << duration.count() << " us" << std::endl;
    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const double*x, const double*y, double*z, const int N)
{
    for(int i = 0;i<N;i++)
    {
        z[i] = x[i]+y[i];
    }
}

void check(const double* z,const int N)
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