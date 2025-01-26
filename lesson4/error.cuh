#pragma once
#include <stdio.h>
//为什么不用 iostream？ 因为cuda里不让cout

#define CHECK(call) \
do \
{ \
    const cudaError_t error_code = call;\
    if(error_code != cudaSuccess)\
    {\
        printf("CUDA ERROR:\n");\
        printf("    File:       %s\n", __FILE__);\
        printf("    Line:       %d\n", __LINE__);\
        printf("    ERROR code: %d\n", error_code);\
        printf("    ERROR text: %s\n", cudaGetErrorString(error_code));\
        exit(1);\
    }\
} while (0);\
//这是c++多行宏的定义与解析

//在使用CHECK(cudaFree(gpu_x));时，会展开并自动判断 
