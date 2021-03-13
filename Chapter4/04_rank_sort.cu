#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//Defining number of elements in Array
#define arraySize 5
//Defining number of thread per block
#define threadPerBlock 5

// 秩排序算法，对于数组中每个元素，统计小于它的元素个数
__global__ void addKernel(int *d_a, int *d_b)
{
	// 当前元素在排序后数组中的位置
	int count = 0;
	// 块中当前线程索引
	int tid = threadIdx.x;
	// 所有块中当前线程唯一索引
	int ttid = blockIdx.x * threadPerBlock + tid; // threadPerBlock=blockDim.x
	// 每个线程的当前元素
	int val = d_a[ttid];
	// 共享内存，减少全局内存访问
	__shared__ int cache[threadPerBlock];
	// 循环直到整个数组计算完，每次计算一个块，步长是一个块线程大小
	for (int i = tid; i < arraySize; i += threadPerBlock) { // threadPerBlock=blockDim.x
		// 填充共享内存
		cache[tid] = d_a[i];
		__syncthreads();
		// 在一个块内，和当前val变量比较，统计block内比其小的数量
		for (int j = 0; j < threadPerBlock; ++j)
			if (val > cache[j])
				count++;
		__syncthreads();
	}
	// 当前线程元素val在排序后数组的位置
	d_b[count] = val;
}

int main()
{
	//Defining host arrays
	int h_a[arraySize] = { 5, 9, 3, 4, 8 };
	int h_b[arraySize];
	//Defining device pointers
	int *d_a, *d_b;
	// allocate the memory
	cudaMalloc((void**)&d_b, arraySize * sizeof(int));
	cudaMalloc((void**)&d_a, arraySize * sizeof(int));
	// Copy input vector from host memory to GPU buffers.
	cudaMemcpy(d_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<arraySize/threadPerBlock, threadPerBlock>>>(d_a, d_b);
	// 设备同步
	cudaDeviceSynchronize();
	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(h_b, d_b, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	printf("The Enumeration sorted Array is: \n");
	for (int i = 0; i < arraySize; i++) {
		printf("%d\n", h_b[i]);
	}
	cudaFree(d_a);
	cudaFree(d_b);
	return 0;
}
