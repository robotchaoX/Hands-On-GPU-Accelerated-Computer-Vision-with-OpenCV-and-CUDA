#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

//Defining number of elements in Array
#define N	50000 // 任意大

//Defining Kernel function for vector addition
__global__ void gpuAdd(int *d_a, int *d_b, int *d_c) {
	//Getting block index of current kernel
	int tid = threadIdx.x + blockIdx.x * blockDim.x;	
	while (tid < N) // 固定线程数量，N大小未知
	{
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += blockDim.x * gridDim.x; // 递增启动的线程数量
	}
}

int main(void) {
	//Defining host arrays
	int h_a[N], h_b[N], h_c[N];
	//Defining device pointers
	int *d_a, *d_b, *d_c;
	// allocate the memory
	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));
	//Initializing Arrays
	for (int i = 0; i < N; i++) {
		h_a[i] = 2 * i*i;
		h_b[i] = i;
	}
	// Copy input arrays from host to device memory
	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
	//Calling kernels with N blocks and one thread per block, passing device pointers as parameters
	gpuAdd << <512, 512 >> >(d_a, d_b, d_c); 
	//Copy result back to host memory from device memory
	cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	int Correct = 1;
	printf("Vector addition on GPU \n");
	//Printing result on console
	for (int i = 0; i < N; i++) {
		if ((h_a[i] + h_b[i] != h_c[i]))
		{
			Correct = 0;
		}
	}
	if (Correct == 1)
	{
		printf("GPU has computed Sum Correctly\n");
	}
	else
	{
		printf("There is an Error in GPU Computation\n");
	}
	//Free up memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
