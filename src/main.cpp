#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>   

#include "params/read_params.h"
#include "math/activation_compressor.h"
#include "math/matmul.h"


int main()
{
    std::cout << "==================================================" << std::endl;
    std::cout << "     Jesse's CUDA MNIST INFERENCE ACCELERATOR     " << std::endl;
    std::cout << "==================================================" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    // ========================================
    // INITIALIZING PARAMETERS
    // ========================================

    double t_initialization = 0.0;
    double t_cudaMalloc = 0.0;
    double t_forward_prop = 0.0;
    double t_initialization_loop = 0.0;
    double t_output = 0.0;
    double t_pcie = 0.0;
    double t_DtoH = 0.0;
    double t_HtoD = 0.0;
    double t_fmem = 0.0;

    const int num_iterations = 100000;

    const int num_act = 1;

    std::cout << "Initializing Parameters . . . " << std::endl;

    const std::int32_t num_neur1 = 784;
    const std::int32_t num_neur2 = 16;
    const std::int32_t num_neur3 = 16;
    const std::int32_t num_neur4 = 10;

    float* W1;
    float* W2;
    float* W3;

    float* B1;
    float* B2;
    float* B3;

    W1 = load_bin("../input/w1.bin", (num_neur2 * num_neur1));
    W2 = load_bin("../input/w2.bin", (num_neur3 * num_neur2));
    W3 = load_bin("../input/w3.bin", (num_neur4 * num_neur3));

    B1 = load_bin("../input/b1.bin", num_neur2);
    B2 = load_bin("../input/b2.bin", num_neur3);
    B3 = load_bin("../input/b3.bin", num_neur4);

    // ========================================
    // INITIALIZING ACTIVATIONS & INPUT VECTOR
    // ========================================

    float* X;
    std::vector<float> A1(16 * num_act, 0.0f);
    std::vector<float> A2(16 * num_act, 0.0f);
    std::vector<float> A3(10 * num_act, 0.0f);

    auto t1 = std::chrono::high_resolution_clock::now();

    t_initialization += std::chrono::duration<double>(t1 - t0).count();

    // ========================================
    // DEVICE MEMORY ALLOCATION
    // ========================================

    float* d_X;
    float* d_A1;
    float* d_A2;
    float* d_A3;

    float* d_W1;
    float* d_W2;
    float* d_W3;

    float* d_B1;
    float* d_B2;
    float* d_B3;

    cudaMalloc(&d_X, 784 * num_act * sizeof(float));
    cudaMalloc(&d_A1, 16 * num_act * sizeof(float));
    cudaMalloc(&d_A2, 16 * num_act * sizeof(float));
    cudaMalloc(&d_A3, 10 * num_act * sizeof(float));

    cudaMalloc(&d_W1, 784 * 16 * sizeof(float));
    cudaMalloc(&d_W2, 16 * 16 * sizeof(float));
    cudaMalloc(&d_W3, 16 * 10 * sizeof(float));

    cudaMalloc(&d_B1, 16 * sizeof(float));
    cudaMalloc(&d_B2, 16 * sizeof(float));
    cudaMalloc(&d_B3, 10 * sizeof(float));

    auto t2 = std::chrono::high_resolution_clock::now();

    t_cudaMalloc += std::chrono::duration<double>(t2 - t1).count();

    cudaMemcpy(d_W1, W1, 784 * 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, W3, 16 * 10 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_B1, B1, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, B2, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B3, B3, 10 * sizeof(float), cudaMemcpyHostToDevice);

    auto t3 = std::chrono::high_resolution_clock::now();

    t_HtoD += std::chrono::duration<double>(t3 - t2).count();

    // ========================================
    // MAIN LOOP
    // ========================================
    
    for(int step = 0; step < num_iterations; ++step) {

        auto t_loop_start = std::chrono::high_resolution_clock::now();

        X = load_bin("../input/test_case.bin", 784 * num_act);

        auto t4 = std::chrono::high_resolution_clock::now();

        cudaMemcpy(d_X, X, 784*num_act*sizeof(float), cudaMemcpyHostToDevice);

        auto t5 = std::chrono::high_resolution_clock::now();

        matmulGPU(d_X, d_W1, d_B1, d_A1, 784, 16, num_act);

        applySigmoidGPU(d_A1, 16*num_act);

        matmulGPU(d_A1, d_W2, d_B2, d_A2, 16, 16, num_act);

        applySigmoidGPU(d_A2, 16*num_act);

        matmulGPU(d_A2, d_W3, d_B3, d_A3, 16, 10, num_act);

        applySigmoidGPU(d_A3, 10*num_act);

        cudaDeviceSynchronize();

        auto t6 = std::chrono::high_resolution_clock::now();

        cudaMemcpy(A3.data(), d_A3, 10*num_act*sizeof(float), cudaMemcpyDeviceToHost);

        auto t7 = std::chrono::high_resolution_clock::now();

        delete[] X;

        auto t8 = std::chrono::high_resolution_clock::now();

        t_fmem += std::chrono::duration<double>(t8 - t7).count();

        t_HtoD += std::chrono::duration<double>(t5 - t4).count();

        t_DtoH += std::chrono::duration<double>(t7 - t6).count();

        t_initialization_loop += std::chrono::duration<double>(t4 - t_loop_start).count();

        t_forward_prop += std::chrono::duration<double>(t6 - t5).count();

        std::cout << "Step: " << (step + 1) << "/" << num_iterations << std::endl;

        auto t9 = std::chrono::high_resolution_clock::now();

        t_output += std::chrono::duration<double>(t9 - t8).count();
    }

    auto t10 = std::chrono::high_resolution_clock::now();

    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_W3);

    cudaFree(d_B1);
    cudaFree(d_B2);
    cudaFree(d_B3);

    cudaFree(d_X);
    cudaFree(d_A1);
    cudaFree(d_A2);
    cudaFree(d_A3);

    auto t11 = std::chrono::high_resolution_clock::now();

    t_fmem += std::chrono::duration<double>(t11 - t10).count();

    t_pcie = t_HtoD + t_DtoH;

    double t_loop_total = t_forward_prop + t_initialization_loop + t_output + t_pcie;

    auto t12 = std::chrono::high_resolution_clock::now();

    double t_total = std::chrono::duration<double>(t12 - t0).count();

    // ========================================
    // OUTPUT DATA
    // ========================================

    std::cout << "==================================================" << std::endl;
    std::cout << "                Profiling Results                 " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::cout << "  1. Parameter Initialization: " << t_initialization << std::endl;
    std::cout << "  2. Input Initialization:     " << t_initialization_loop << std::endl;
    std::cout << "  3. Device MemAlloc:          " << t_cudaMalloc << std::endl;
    std::cout << "  4. Forward Propagation:      " << t_forward_prop << std::endl;
    std::cout << "  5. PCIe Transfers:           " << t_pcie << std::endl;
    std::cout << "      5a. Host to Device:      " << t_HtoD << std::endl;
    std::cout << "      5b. Device to Host:      " << t_DtoH << std::endl;
    std::cout << "  6. Output:                   " << t_output << std::endl;

    std::cout << "==================================================" << std::endl;
    std::cout << "  Main Loop Runtime:           " << t_loop_total << std::endl;
    std::cout << "  Full Program Runtime:        " << t_total << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}