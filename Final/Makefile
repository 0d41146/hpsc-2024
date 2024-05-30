openmp: openmp.cpp
	g++ openmp.cpp -fopenmp -O3

acc: acc.cpp
	nvc++ acc.cpp -acc -Minfo -gpu=managed

cuda: cuda.cu
	nvcc cuda.cu

cpp: 10_cavity.cpp
	g++ 10_cavity.cpp -O3
