#include "CudaTimer.hpp"

CudaTimer::CudaTimer() : elapsedTime(0.0f) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaTimer::startTimer() {
    cudaEventRecord(start, 0);
}

void CudaTimer::stopTimer() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
}

float CudaTimer::getElapsedTime() const {
    return elapsedTime;
}

void CudaTimer::printElapsedTime(const std::string& message) const {
    std::cout << message << ": " << elapsedTime << " ms" << std::endl;
}
