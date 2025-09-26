all: main

main: main.cpp neuralFunctions.h neuralFunctions.cpp ThreadPool.h ThreadPool.cpp
	g++ -I eigen -o main main.cpp neuralFunctions.cpp ThreadPool.cpp
	