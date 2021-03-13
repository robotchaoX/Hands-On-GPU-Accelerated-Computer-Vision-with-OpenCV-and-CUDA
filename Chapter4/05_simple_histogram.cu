#include <stdio.h>
#include <cuda_runtime.h>

// 元素总个数
//Defining number of elements in Array
#define SIZE 1000
// 元素分布范围，每个元素值0-15
#define NUM_BIN 16

__global__ void histogram_without_atomic(int *d_b, int *d_a)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int item = d_a[tid];
	if (tid < SIZE)
	{
		// error 有1000个线程同时对16个内存位置进行写入，必须原子操作
		d_b[item]++;
	}
}

__global__ void histogram_atomic(int *d_b, int *d_a)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	// 0-1000数组内的值
	int item = d_a[tid];
	if (tid < SIZE)
	{
		// 原子操作，对应写入结果0-15下标+1
		atomicAdd(&(d_b[item]), 1);
	}
}

int main()
{
	// generate the input array on the host
	int h_a[SIZE];
	for (int i = 0; i < SIZE; i++) {
		h_a[i] = i % NUM_BIN;
	}
	int h_b[NUM_BIN];
	// 定义并初始化结果数组
	for (int i = 0; i < NUM_BIN; i++) {
		h_b[i] = 0;
	}
	// declare GPU memory pointers
	int * d_a;
	int * d_b;
	// allocate GPU memory
	cudaMalloc((void **)&d_a, SIZE * sizeof(int));
	cudaMalloc((void **)&d_b, NUM_BIN * sizeof(int));
	// transfer the arrays to the GPU
	cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);
	// launch the kernel
	histogram_without_atomic << <((SIZE+NUM_BIN-1) / NUM_BIN), NUM_BIN >> >(d_b, d_a);
	//histogram_atomic << <((SIZE+NUM_BIN-1) / NUM_BIN), NUM_BIN >> >(d_b, d_a);
	// copy back the sum from GPU
	cudaMemcpy(h_b, d_b, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Histogram using 16 bin without shared Memory is: \n");
	for (int i = 0; i < NUM_BIN; i++) {
		printf("bin %d: count %d\n", i, h_b[i]);
	}
	// free GPU memory allocation
	cudaFree(d_a);
	cudaFree(d_b);
	return 0;
}
