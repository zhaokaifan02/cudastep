
__device__ double add1_device(const double x, const double y) 
{
    //前面提过global控制的函数是不能有返回值的
    //但是gpu里的device就可以有返回值
    return x+y;
}

__global__ void add1(const double* x, const double *y, double* z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n<N)
        z[n] = add1_device(x[n],y[n]); //核函数里调用device的
}

//同指针的设备函数
__device__ void add2_device(const double x, const double y,  double* z)
{
    *z = x+y;
}

__global__ void add2(const double* x, const double *y, double *z,const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n<N)
        add2_device(x[n],y[n],&z[n]); //z[n]的地址 
}

//使用引用 refername c++的特色吧
__device__ void add3_device(const double x, const double y, double &z)
{
    z = x+y;
}

__global__ void add3(const double *x, const double*y ,double*z , const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n<N)
        add3_device(x[n],y[n],z[n]); //z[n]的地址 
}
