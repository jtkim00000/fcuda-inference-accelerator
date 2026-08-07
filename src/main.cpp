#include <iostream>
#include <vector>
#include <chrono>

#include "params/read_params.h"
#include "math/activation_compressor.h"
#include "math/matmul.h"


int main()
{
    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "==================================================" << std::endl;
    std::cout << "     Jesse's CUDA MNIST INFERENCE ACCELERATOR     " << std::endl;
    std::cout << "==================================================" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    // ========================================
    // INITIALIZING PARAMETERS
    // ========================================

    double t_initialization = 0.0;

    const int num_iterations = 100000;

    const int num_act = 512;

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
    // MAIN LOOP
    // ========================================

    double t_forward_prop = 0.0;
    double t_initialization_loop = 0.0;
    double t_output = 0.0;
    
    for(int step = 0; step < num_iterations; ++step) {

        auto t_loop_start = std::chrono::high_resolution_clock::now();

        X = load_bin("../input/test_case.bin", 784*num_act);

        auto t2 = std::chrono::high_resolution_clock::now();

        matmul(X, W1, B1, A1.data(), 784, 16, num_act);

        applySigmoid(A1.data(), 16*num_act);

        matmul(A1.data(), W2, B2, A2.data(), 16, 16, num_act);

        applySigmoid(A2.data(), 16*num_act);

        matmul(A2.data(), W3, B3, A3.data(), 16, 10, num_act);

        applySigmoid(A3.data(), 10*num_act);

        auto t3 = std::chrono::high_resolution_clock::now();

        t_initialization_loop += std::chrono::duration<double>(t2 - t_loop_start).count();

        t_forward_prop += std::chrono::duration<double>(t3 - t2).count();

        std::cout << "Step: " << (step + 1) << "/" << num_iterations << std::endl;

        auto t4 = std::chrono::high_resolution_clock::now();

        t_output += std::chrono::duration<double>(t4 - t3).count();

        delete[] X;
    }

    double t_loop_total = t_forward_prop + t_initialization_loop + t_output;

    auto t5 = std::chrono::high_resolution_clock::now();

    double t_total = std::chrono::duration<double>(t5 - t0).count();

    // ========================================
    // OUTPUT DATA
    // ========================================

    std::cout << "==================================================" << std::endl;
    std::cout << "                Profiling Results                 " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::cout << "  1. Parameter Initialization: " << t_initialization << std::endl;
    std::cout << "  2. Input Initialization:     " << t_initialization_loop << std::endl;
    std::cout << "  3. Forward Propagation:      " << t_forward_prop << std::endl;
    std::cout << "  4. Output:                   " << t_output << std::endl;

    std::cout << "==================================================" << std::endl;
    std::cout << "  Main Loop Runtime:           " << t_loop_total << std::endl;
    std::cout << "  Full Program Runtime:        " << t_total << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}