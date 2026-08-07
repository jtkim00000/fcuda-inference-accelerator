#include <iostream>
#include <cmath>

float* apply_sigmoid(float* activation, const int N) {

    for(int i = 0; i < N; ++i) {
        activation[i] = 1.0f / (1.0f + std::exp(-activation[i]));
    }

    return activation;
}