all: main

main: main.cpp neuralFunctions.h neuralFunctions.cpp
	g++ -I eigen -o main main.cpp neuralFunctions.cpp
	