#include <iostream>

__global__ void matmulKernel(
    const float* activation, 
    const float* weight, 
    const float* bias, 
    float* next_activation, 
    const int n1, 
    const int n2
) {

    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if(k >= n2) return;

    float sum = 0.0f;
    for(int n = 0; n < n1; ++n) {
        sum += weight[k * n1 + n] * activation[n];
    }
    next_activation[k] = sum + bias[k];
}

void matmulGPU(
    const float* activation, 
    const float* weight, 
    const float* bias, 
    float* next_activation, 
    const int n1, 
    const int n2
) {
    int blocks = (n2 + 255) / 256;

        
    matmulKernel<<<blocks, 256>>>(
        activation,
        weight,
        bias,
        next_activation,
        n1,
        n2
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "applySigmoidKernel launch failed: "
                << cudaGetErrorString(err) << '\n';
    }
}