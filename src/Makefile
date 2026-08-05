
execute: main.o read_params.o
	g++-13 main.o read_params.o -o execute

main.o: main.cpp
	g++-13 -c main.cpp

read_params.o: params/read_params.cpp params/read_params.h
	g++-13 -c params/read_params.cpp

clean:
	rm *.o execute