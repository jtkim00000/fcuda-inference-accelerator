
// dimensions: (n2 x n1) * (n1 x num_act)
void matmul(
    const float* activation, 
    const float* weight, 
    const float* bias, 
    float* next_activation, 
    const int n1, 
    const int n2,
    const int num_act
) {

    for(int row = 0; row < n2; ++row) {
        for(int col = 0; col < num_act; ++col) {
            float sum = 0.0f;
            for(int n = 0; n < n1; ++n) {
                sum += weight[row * n1 + n] * activation[n * num_act + col];
            }
            next_activation[row * num_act + col] = sum + bias[row];
        }
    }
}