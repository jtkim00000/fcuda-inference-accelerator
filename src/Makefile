CXX  = g++-13
NVCC = nvcc

CXXFLAGS  = -O3 -std=c++17 -Wall
NVCCFLAGS = -O3 -std=c++17 -arch=sm_89

TARGET = execute

CPP_OBJ = \
	main.o \
	params/read_params.o

CUDA_OBJ = \
	math/matmul.o \
	math/activation_compressor.o

OBJ = $(CPP_OBJ) $(CUDA_OBJ)

all: $(TARGET)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

params/read_params.o: params/read_params.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

math/matmul.o: math/matmul.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

math/activation_compressor.o: math/activation_compressor.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean