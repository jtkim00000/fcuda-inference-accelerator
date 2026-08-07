
float* matmul(const float* activation, const float* weight, const float* bias, float* next_activation, const int n1, const int n2) {

    for(int k = 0; k < n2; ++k) {
        float sum = 0.0f;
        for(int n = 0; n < n1; ++n) {
            sum += weight[k * n1 + n] * activation[n];
        }
        next_activation[k] = sum + bias[k];
    }

    return next_activation;
}