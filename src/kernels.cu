#include <stdio.h>

__global__ void forward_step1(float *weight_D, float *a_D, float *res1_D, unsigned int columns) {
	
	unsigned int tid = blockDim.x*threadIdx.y + threadIdx.x;
	
	unsigned int i = blockIdx.z;
	
	unsigned int j = (gridDim.x*blockIdx.y+blockIdx.x)*blockDim.x*blockDim.y + tid;
	
	__shared__ float partial_sums[1024];
	
	if (j < columns) {
		partial_sums[tid] = a_D[j]*weight_D[i*columns+j];
	} else {
		partial_sums[tid] = 0;
	}
	__syncthreads();
	if (tid < 512) { partial_sums[tid] += partial_sums[tid+512]; } __syncthreads();
	if (tid < 256) { partial_sums[tid] += partial_sums[tid+256]; } __syncthreads();
	if (tid < 128) { partial_sums[tid] += partial_sums[tid+128]; } __syncthreads();
	if (tid <  64) { partial_sums[tid] += partial_sums[tid+ 64]; } __syncthreads();
	if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads();
	if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads();
	if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads();
	if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads();
	if (tid <   2) { partial_sums[tid] += partial_sums[tid+  2]; } __syncthreads();
	if (tid <   1) {
		res1_D[i*64 + gridDim.x*blockIdx.y + blockIdx.x] = partial_sums[0]+partial_sums[1];
	}
	
}

__global__ void forward_step2(float *res1_D, float *bias_D, float *a_D) {
	
	unsigned int i = blockIdx.z;
	unsigned int tid = blockDim.x*threadIdx.y + threadIdx.x;
	
	__shared__ float partial_sums[64];
	
	partial_sums[tid] = res1_D[64*i+tid];
	__syncthreads();
	if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads();
	if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads();
	if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads();
	if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads();
	if (tid <   2) { partial_sums[tid] += partial_sums[tid+  2]; } __syncthreads();
	if (tid <   1) {
		partial_sums[0] += partial_sums[1]+bias_D[i];
		a_D[i] = 1/(1+expf(-partial_sums[0]));
	}
	
}

__global__ void output_error(float *aL_D, float *y_D, float *deltaL_D) {
	unsigned int i = blockIdx.z;
	
	deltaL_D[i] = (aL_D[i]-y_D[i])*aL_D[i]*(1-aL_D[i]);
	/*
	unsigned int tid = blockDim.x*threadIdx.y + threadIdx.x;
	__shared__ float results[2]
	if (tid == 0) {
		results[0] = a[i]-y[i];
	} else if (tid == 1) {
		results[1] = 1-a[i];
	}
	__syncthreads();
	
	deltaL_D[i] = results[0]*results[1]*aL_D[i];
	*/
}

__global__ void backward_step1(float *weight_D, float *delta_D, float *res1_D, unsigned int columns, unsigned int rows) {
	unsigned int i = blockIdx.z;
	unsigned int tid = blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int j = (gridDim.x*blockIdx.y+blockIdx.x)*blockDim.x*blockDim.y + tid;
	
	__shared__ float partial_sums[1024];
	
	if (j<rows) {
		partial_sums[tid] = weight_D[j*columns+i]*delta_D[j];
	} else {
		partial_sums[tid] = 0;
	}
	
	__syncthreads();
	if (tid < 512) { partial_sums[tid] += partial_sums[tid+512]; } __syncthreads();
	if (tid < 256) { partial_sums[tid] += partial_sums[tid+256]; } __syncthreads();
	if (tid < 128) { partial_sums[tid] += partial_sums[tid+128]; } __syncthreads();
	if (tid <  64) { partial_sums[tid] += partial_sums[tid+ 64]; } __syncthreads();
	if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads();
	if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads();
	if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads();
	if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads();
	if (tid <   2) { partial_sums[tid] += partial_sums[tid+  2]; } __syncthreads();
	if (tid <   1) {
		res1_D[i*64 + gridDim.x*blockIdx.y + blockIdx.x] = partial_sums[0]+partial_sums[1];
	}
}

__global__ void backward_step2(float *res1_D, float *a_D, float *delta_D) {
	
	unsigned int i = blockIdx.z;
	unsigned int tid = blockDim.x*threadIdx.y + threadIdx.x;
	
	__shared__ float partial_sums[64];
	
	partial_sums[tid] = res1_D[64*i+tid];
	
	__syncthreads();
	if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads();
	if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads();
	if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads();
	if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads();
	if (tid <   2) { partial_sums[tid] += partial_sums[tid+  2]; } __syncthreads();
	if (tid <   1) {
		partial_sums[0] += partial_sums[1];
		delta_D[i] = partial_sums[0]*a_D[i]*(1-a_D[i]);		//dσ(t)/dt = σ(t)*(1-σ(t))
	}
	
}

//this is supposed to run with pthreads.
__global__ void weight_gradient(float *a_D, float *delta_D, float *weightG_D, unsigned int columns/*(a_previous)*/, unsigned int rows/*delta*/) {
	
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (i<columns && j<rows) {
		weightG_D[j*columns+i] = a_D[i]*delta_D[j];
	}
	
}


//sums 1024 elements of arrays of size size
__global__ void sum_of_1024(float *wG_or_bG, float *result_D, unsigned int size, unsigned int samples) {
	
	unsigned int tid = blockDim.x*threadIdx.y +threadIdx.x;
	unsigned int cell = gridDim.y*blockIdx.z + blockIdx.y;
	unsigned int sample_id = blockIdx.x*1024+tid;
	
	__shared__ float partial_sums[1024];
	
	if (sample_id>=samples) {
		partial_sums[tid] = 0;
	} else {
		partial_sums[tid] = wG_or_bG[sample_id*size+cell];
	}

	__syncthreads();
	if (tid < 512) { partial_sums[tid] += partial_sums[tid+512]; } __syncthreads();
	if (tid < 256) { partial_sums[tid] += partial_sums[tid+256]; } __syncthreads();
	if (tid < 128) { partial_sums[tid] += partial_sums[tid+128]; } __syncthreads();
	if (tid <  64) { partial_sums[tid] += partial_sums[tid+ 64]; } __syncthreads();
	if (tid <  32) { partial_sums[tid] += partial_sums[tid+ 32]; } __syncthreads();
	if (tid <  16) { partial_sums[tid] += partial_sums[tid+ 16]; } __syncthreads();
	if (tid <   8) { partial_sums[tid] += partial_sums[tid+  8]; } __syncthreads();
	if (tid <   4) { partial_sums[tid] += partial_sums[tid+  4]; } __syncthreads();
	if (tid <   2) { partial_sums[tid] += partial_sums[tid+  2]; } __syncthreads();
	if (tid <   1) {
		result_D[cell+blockIdx.x*size] = partial_sums[0]+partial_sums[1];
	}

}

__global__ void grad_desc(float *wG_or_bG, float *w_or_b, unsigned int size, unsigned int samples, float learning_rate) {
	unsigned int global_id = (blockDim.x*blockDim.y*blockDim.z)*(gridDim.x*gridDim.y*blockIdx.z + gridDim.x*blockIdx.y + blockIdx.x) + (blockDim.x*blockDim.y)*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
	if (global_id < size) {
		w_or_b[global_id] = w_or_b[global_id] - learning_rate*wG_or_bG[global_id]/samples;
	}
}