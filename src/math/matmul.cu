#include <iostream>

__global__ void matmulKernel(
    const float* activation, 
    const float* weight, 
    const float* bias, 
    float* next_activation, 
    const int n1, 
    const int n2,
    const int num_act
) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row >= n2 || col >= num_act) return;

    float sum = 0.0f;

    for(int n = 0; n < n1; ++n) {
        sum += weight[row * n1 + n] * activation[col * n1 + n];
    }
    next_activation[col * n2 + row] = sum + bias[row];
}

void matmulGPU(
    const float* activation, 
    const float* weight, 
    const float* bias, 
    float* next_activation, 
    const int n1, 
    const int n2,
    const int num_act
) {
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid((num_act + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n2 + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(
        activation,
        weight,
        bias,
        next_activation,
        n1,
        n2,
        num_act
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "matmulKernel launch failed: "
                << cudaGetErrorString(err) << '\n';
    }
}