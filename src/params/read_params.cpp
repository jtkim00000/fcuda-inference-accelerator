#include <iostream>

float* load_bin(const char* path, size_t num_elements) {
    float* data = (float*)malloc(num_elements * sizeof(float));
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }
    size_t read = fread(data, sizeof(float), num_elements, f);
    if (read != num_elements) {
        fprintf(stderr, "Expected %zu elements, got %zu from %s\n", num_elements, read, path);
        exit(1);
    }
    fclose(f);
    return data;
}