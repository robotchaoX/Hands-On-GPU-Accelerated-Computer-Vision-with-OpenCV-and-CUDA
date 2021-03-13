//Matrix multiplication using shared and non shared kernal
#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// tile 每个块的共享存储区的存储空间一般很小，对数据进行分块，分块进行计算
// 网格维度(size/TILE_SIZE, size/TILE_SIZE)
// 块维度(TILE_SIZE, TILE_SIZE)
#define TILE_SIZE 2

//Matrix multiplication using non shared kernel
__global__ void gpu_Matrix_Mul_nonshared(float *d_a, float *d_b, float *d_c, const int size)
{
	int row, col; // Matrix size
	col = TILE_SIZE * blockIdx.x + threadIdx.x; // TILE_SIZE = blockDim.x
	row = TILE_SIZE * blockIdx.y + threadIdx.y; // TILE_SIZE = blockDim.y
	// col = blockDim.x * blockIdx.x + threadIdx.x;
	// row = blockDim.y * blockIdx.y + threadIdx.y;
	// printf("col:%d, row:%d\n", col,row);
	// 为计算result[row,col]结果,需整个矩阵row行，col列点乘
	for (int k = 0; k< size; k++)
	{
		d_c[row*size + col] += d_a[row * size + k] * d_b[k * size + col]; // 行列点乘结果
	}
}

// Matrix multiplication using shared kernel
__global__ void gpu_Matrix_Mul_shared(float *d_a, float *d_b, float *d_c, const int size)
{
	int row, col;
	//Defining Shared Memory
	__shared__ float shared_a[TILE_SIZE][TILE_SIZE];
	__shared__ float shared_b[TILE_SIZE][TILE_SIZE];
	col = TILE_SIZE * blockIdx.x + threadIdx.x; // TILE_SIZE = blockDim.x
	row = TILE_SIZE * blockIdx.y + threadIdx.y; // TILE_SIZE = blockDim.y
	for (int i = 0; i< size / TILE_SIZE; i++) // 循环size/TILE_SIZE次，可计算tile*tile方块矩阵结果
	{
		// 填充共享内存
		shared_a[threadIdx.y][threadIdx.x] = d_a[row* size + (i*TILE_SIZE + threadIdx.x)];
		shared_b[threadIdx.y][threadIdx.x] = d_b[(i*TILE_SIZE + threadIdx.y) * size + col];
		__syncthreads(); 
		// 计算tile*tile方块矩阵的向量点乘,tile内需row行，col列点乘
		for (int j = 0; j<TILE_SIZE; j++)
			d_c[row*size + col] += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
		__syncthreads(); 
	}
}

// main routine
int main()
{
	// 矩阵大小 4×4
	const int size = 4;
	//Define Host Array
	float h_a[size][size], h_b[size][size],h_result[size][size];
	//Defining device Array
	float *d_a, *d_b, *d_result; 
	//Initialize host Array
	for (int i = 0; i<size; i++)
	{
		for (int j = 0; j<size; j++)
		{
			h_a[i][j] = i;
			h_b[i][j] = j;
		}
	}
	// declare and allocate GPU memory
	cudaMalloc((void **)&d_a, size*size*sizeof(int));
	cudaMalloc((void **)&d_b, size*size * sizeof(int));
	cudaMalloc((void **)&d_result, size*size* sizeof(int));
	//copy host array to device array
	cudaMemcpy(d_a, h_a, size*size* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size*size* sizeof(int), cudaMemcpyHostToDevice);
	//Define grid and block dimensions
	dim3 dimGrid(size / TILE_SIZE, size / TILE_SIZE, 1);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
	printf("dimGrid:(%d*%d), dimBlock:(%d*%d) \n",size / TILE_SIZE,size / TILE_SIZE, TILE_SIZE, TILE_SIZE);
	//Call kernel
	//gpu_Matrix_Mul_nonshared << <dimGrid, dimBlock >> > (d_a, d_b, d_result, size);
	gpu_Matrix_Mul_shared << <dimGrid, dimBlock >> > (d_a, d_b, d_result, size);
	// copy the array back to host memory
	cudaMemcpy(h_result, d_result, size*size * sizeof(int),	cudaMemcpyDeviceToHost);
	printf("The result of Matrix multiplication is: \n");
	for (int i = 0; i< size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			printf("%f   ", h_result[i][j]);
		}
		printf("\n");
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	return 0;
}
