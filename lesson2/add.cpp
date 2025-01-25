#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0e-15; //特殊的表示法表示0.0001 15个0
const double a = 1.23;
const double b = 2.34;
const double d = 3.57;
void add(const double* x, const double *y, const double* z, const int N);
void check(const double* z, const int N);

int main()
{
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
    
    return 0;
}

void add(const double*x, const double*y, double*z, const int N)
{
    
}