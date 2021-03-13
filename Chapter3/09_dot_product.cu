#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024
#define threadsPerBlock 512

__global__ void gpu_dot(float *d_a, float *d_b, float *d_c) {
	//Declare shared memory
	__shared__ float partial_sum[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//Calculate index for shared memory 
	int index = threadIdx.x;
	//Calculate Partial Sum
	float sum = 0;
	while (tid < N) 
	{
		sum += d_a[tid] * d_b[tid];
		tid += blockDim.x * gridDim.x;
	}
	// Store partial sum in shared memory
	partial_sum[index] = sum; // 每个线程的部分和结果
	// synchronize threads 
	__syncthreads();
	// Calculating partial sum for whole block in reduce operation
	int i = blockDim.x / 2;
	while (i != 0) {
		if (index < i)
			partial_sum[index] += partial_sum[index + i]; // 循环相邻奇偶相加
		__syncthreads();
		i /= 2;
	}
	//Store block partial sum in global memory
	if (index == 0)
		d_c[blockIdx.x] = partial_sum[0]; // 当前块的总体部分和结果
}

int main(void) {
	//Declare Host Array
	float *h_a, *h_b, h_c, *partial_sum;
	//Declare device Array
	float *d_a, *d_b, *d_partial_sum;
	//Calculate total number of blocks per grid
	int block_calc = (N + threadsPerBlock - 1) / threadsPerBlock; // 除法向上取整
	int blocksPerGrid = (32 < block_calc ? 32 : block_calc); // 取块数量限制32？？
	// allocate memory on the host side
	h_a = (float*)malloc(N * sizeof(float));
	h_b = (float*)malloc(N * sizeof(float));
	partial_sum = (float*)malloc(blocksPerGrid * sizeof(float));
	// allocate the memory on the device
	cudaMalloc((void**)&d_a, N * sizeof(float));
	cudaMalloc((void**)&d_b, N * sizeof(float));
	cudaMalloc((void**)&d_partial_sum, blocksPerGrid * sizeof(float));
	// fill the host array with data
	for (int i = 0; i<N; i++) {
		h_a[i] = i;
		h_b[i] = 2;
	}
	//Copy Array from host to device
	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
	//Call kernel 
	gpu_dot << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_partial_sum);
	// copy the array back to host memory
	cudaMemcpy(partial_sum, d_partial_sum, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
	//Calculate final dot product on host
	h_c = 0;
	for (int i = 0; i<blocksPerGrid; i++) {
		h_c += partial_sum[i];
	}
	printf("The computed dot product is: %f\n", h_c);
#define cpu_sum(x) (x*(x+1))
	if (h_c == cpu_sum((float)(N - 1)))
	{
		printf("The dot product computed by GPU is correct\n");
	}
	else
	{
		printf("Error in dot product computation");
	}
	// free memory on host and device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_partial_sum);
	free(h_a);
	free(h_b);
	free(partial_sum);
}