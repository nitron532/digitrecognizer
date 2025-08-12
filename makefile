all: main

main: main.cpp neuralFunctions.h
	g++ -I eigen -o main main.cpp neuralFunctions.cpp