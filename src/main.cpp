#include <iostream>
#include <vector>
#include <chrono>

#include "params/read_params.h"

int main()
{
    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "==============================================" << std::endl;
    std::cout << "   Jesse's CUDA MNIST INFERENCE ACCELERATOR   " << std::endl;
    std::cout << "==============================================" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

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
    W3 = load_bin("../input/w3.bin", (num_neur4 * num_neur4));

    B1 = load_bin("../input/b1.bin", num_neur2);
    B2 = load_bin("../input/b2.bin", num_neur3);
    B3 = load_bin("../input/b3.bin", num_neur4);

    auto t1 = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 16; ++i) {
        std::cout << B1[i] << std::endl;
    }

    return 0;
}