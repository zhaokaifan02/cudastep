#ifndef CUDATIMER_HPP
#define CUDATIMER_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <string>

class CudaTimer {
private:
    cudaEvent_t start, stop;
    float elapsedTime;

public:
    CudaTimer();
    ~CudaTimer();

    void startTimer();
    void stopTimer();
    float getElapsedTime() const;
    void printElapsedTime(const std::string& message = "Elapsed time") const;
};

#endif // CUDATIMER_HPP