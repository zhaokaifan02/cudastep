#include <stdlib.h>
#include <stdio.h>
#include "common.cuh"
int main(int argc, char **argv)
{
    int nx = 5;     // 晶格在每个方向上的单元数
    int Ne = 20000; // 平衡阶段的步数， equilibration
    int Np = 20000; // 生产阶段的步数， production

    if (argc != 3)
    {
        printf("Usage: %s nx Ne\n", argv[0]);
        exit(1);
    }
    else
    {
        nx = atoi(argv[1]);
        Ne = atoi(argv[2]);
        Np = Ne;
    }
    int N = 4 * nx * nx * nx; //Ne是面心立方，每个晶胞4个原子，一共nx*nx*nx个晶胞，所以4*nx*nx*nx个原子
    int Ns = 100; // 采样次数，我们100个 
    int MN = 200;
    real T_0 = 60.0;
    real ax = 5.385;
}