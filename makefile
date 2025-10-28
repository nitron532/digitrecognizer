all: main

main: main.cpp neuralFunctions.h neuralFunctions.cpp ThreadPool.h ThreadPool.cpp
	g++ -I eigen -o main main.cpp neuralFunctions.cpp ThreadPool.cpp -O3 -fno-math-errno -march=native -fopenmp -DNDEBUG -g
	
# 	-o3 -fno-math-errno -march=native -fopenmp -dndebug
#make optimized option?