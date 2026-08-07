

void matmul(
    const float* activation, 
    const float* weight, 
    const float* bias, 
    float* next_activation, 
    const int n1, 
    const int n2,
    const int num_act
);

void matmulGPU(
    const float* activation, 
    const float* weight, 
    const float* bias, 
    float* next_activation, 
    const int n1, 
    const int n2
);