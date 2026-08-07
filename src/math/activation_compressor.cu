#include <iostream>

__global__ void applySigmoidKernel(float* activation, const int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i >= N) return;

    activation[i] = 1.0f / (1.0f + std::exp(-activation[i]));
}


void applySigmoidGPU(float* activation, const int N) {
    int blocks = (N + 255) / 256;

        
    applySigmoidKernel<<<blocks, 256>>>(
        activation,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "applySigmoidKernel launch failed: "
                << cudaGetErrorString(err) << '\n';
    }
}