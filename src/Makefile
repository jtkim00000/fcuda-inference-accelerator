
execute: main.o read_params.o activation_compressor.o matmul.o
	g++-13 main.o read_params.o activation_compressor.o matmul.o -o execute

main.o: main.cpp
	g++-13 -c main.cpp

read_params.o: params/read_params.cpp params/read_params.h
	g++-13 -c params/read_params.cpp

activation_compressor.o: math/activation_compressor.cpp math/activation_compressor.h
	g++-13 -c math/activation_compressor.cpp

matmul.o: math/matmul.cpp math/matmul.h
	g++-13 -c math/matmul.cpp

clean:
	rm *.o execute