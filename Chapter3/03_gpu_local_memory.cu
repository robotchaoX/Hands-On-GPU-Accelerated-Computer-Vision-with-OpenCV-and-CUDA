#include <stdio.h>

// Defining number of elements in Array
#define N 5

__global__ void gpu_local_memory(int d_in)
{
	// Define local memory
	int t_local;
	t_local = d_in * threadIdx.x;
	printf("Value of Local variable in current thread is: %d \n", t_local);
}

int main(int argc, char **argv)
{
	printf("Use of Local Memory on GPU:\n");
	// launch the kernel
	gpu_local_memory << <1, N >> >(5);  
	// waiting for all kernels to finish
	cudaDeviceSynchronize();
	return 0;
}
