#include <cuda.h>
#include "kernels.cu"
#include <stdio.h>

int forward_step_wrap(float *weight, float *bias, float *a, float *a_new, unsigned int columns, unsigned int rows) {
	float *weight_D, *bias_D, *a_D, *res1_D;
	if (cudaMalloc((void **)&weight_D, columns*rows*sizeof(float)) != cudaSuccess) {
		return -1;
	}
	if (cudaMalloc((void **)&bias_D, rows*sizeof(float)) != cudaSuccess) {
		return -2;
	}
	if (cudaMalloc((void **)&a_D, ((columns>rows)*columns + (rows>=columns)*rows)*sizeof(float)) != cudaSuccess) {
		return -3;
	}
	if (cudaMalloc((void **)&res1_D, 64*rows*sizeof(float)) != cudaSuccess) {
		return -4;
	}
	
	dim3 block(32, 32, 1);
	dim3 grid(8, 8, rows);
	cudaMemcpy(weight_D, weight, columns*rows*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(a_D, a, columns*sizeof(float), cudaMemcpyHostToDevice);
	forward_step1<<<grid, block>>>(weight_D, a_D, res1_D, columns);
	cudaMemcpy(bias_D, bias, rows*sizeof(float), cudaMemcpyHostToDevice);
	block.x = 8;
	block.y = 8;
	grid.x = 1;
	grid.y = 1;
	forward_step2<<<grid, block>>>(res1_D, bias_D, a_D);
	cudaMemcpy(a_new, a_D, rows*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(weight_D);
	cudaFree(bias_D);
	cudaFree(a_D);
	cudaFree(res1_D);
	
	return 0;
}

int output_error_wrap(float *aL, float *y, float *deltaL, unsigned int output_size) {
	
	float *aL_D, *y_D, *deltaL_D;
	if (cudaMalloc((void **)&aL_D, output_size*sizeof(float)) != cudaSuccess) {
		return -1;
	}
	if (cudaMalloc((void **)&y_D, output_size*sizeof(float)) != cudaSuccess) {
		return -2;
	}
	if (cudaMalloc((void **)&deltaL_D, output_size*sizeof(float)) != cudaSuccess) {
		return -3;
	}
	
	dim3 block(1, 1, 1);
	dim3 grid(1, 1, output_size);
	cudaMemcpy(aL_D, aL, output_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y_D, y, output_size*sizeof(float), cudaMemcpyHostToDevice);
	output_error<<<grid, block>>>(aL_D, y_D, deltaL_D);
	cudaMemcpy(deltaL, deltaL_D, output_size*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(aL_D);
	cudaFree(y_D);
	cudaFree(deltaL_D);
	
	return 0;
}

int backward_step_wrap(float *weight, float *a, float *delta, float *delta_new, unsigned int columns, unsigned int rows) {
	float *weight_D, *a_D, *delta_D, *res1_D;
	if (cudaMalloc((void **)&weight_D, columns*rows*sizeof(float)) != cudaSuccess) {
		return -1;
	}
	if (cudaMalloc((void **)&a_D, columns*sizeof(float)) != cudaSuccess) {
		return -2;
	}
	if (cudaMalloc((void **)&delta_D, ((columns>rows)*columns + (rows>=columns)*rows)*sizeof(float)) != cudaSuccess) {
		return -3;
	}
	if (cudaMalloc((void **)&res1_D, 64*columns*sizeof(float)) != cudaSuccess) {
		return -4;
	}
	
	dim3 block(32, 32, 1);
	dim3 grid(8, 8, columns);
	
	cudaMemcpy(weight_D, weight, columns*rows*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(delta_D, delta, rows*sizeof(float), cudaMemcpyHostToDevice);
	backward_step1<<<grid, block>>>(weight_D, delta_D, res1_D, columns, rows);
	cudaMemcpy(a_D, a, columns*sizeof(float), cudaMemcpyHostToDevice);
	
	block.x = 8;
	block.y = 8;
	grid.x = 1;
	grid.y = 1;
	
	backward_step2<<<grid, block>>>(res1_D, a_D, delta_D);
	cudaMemcpy(delta_new, delta_D, columns*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(weight_D);
	cudaFree(a_D);
	cudaFree(delta_D);
	cudaFree(res1_D);
	
	return 0;
	
}

int sum_weights_wrap_test(float *weightGs, float *result1, unsigned int columns, unsigned int rows, unsigned int samples) {
	float *weightGs_D, *result_D;
	unsigned int size = columns*rows;
	unsigned int gridx = samples/1024+1;

	if (cudaMalloc((void **)&weightGs_D, size*samples*sizeof(float)) != cudaSuccess) {
		return -1;
	}
	if (cudaMalloc((void **)&result_D, size*gridx*sizeof(float)) != cudaSuccess) {
		return -2;
	}

	dim3 block(32, 32, 1);
	dim3 grid(gridx, columns, rows);
	cudaMemcpy(weightGs_D, weightGs, size*samples*sizeof(float), cudaMemcpyHostToDevice);
	
	sum_of_1024<<<grid, block>>>(weightGs_D, result_D, size, samples);
	grid.x = 1;
	sum_of_1024<<<grid, block>>>(result_D, result_D, size, gridx);
	cudaMemcpy(result1, result_D, size*gridx*sizeof(float), cudaMemcpyDeviceToHost);
	
	return 0;
}

int gradient_descent_wrap(float *w_or_b, float *wG_or_bG, unsigned int columns, unsigned int rows, unsigned int samples, float heta) {
	float *w_or_b_D, *wG_or_bG_D;
	unsigned int size = columns*rows;
	if (cudaMalloc((void **)&w_or_b_D, size*sizeof(float)) != cudaSuccess) {
		return -1;
	}
	if (cudaMalloc((void **)&wG_or_bG_D, size*sizeof(float)) != cudaSuccess) {
		return -2;
	}

	dim3 block(32, 32, 1);
	dim3 grid(columns/32+1, rows/32+1, 1);

	cudaMemcpy(w_or_b_D, w_or_b, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(wG_or_bG_D, wG_or_bG, size*sizeof(float), cudaMemcpyHostToDevice);
	grad_desc<<<grid, block>>>(w_or_b_D, wG_or_bG_D, size, samples, heta);
	cudaMemcpy(w_or_b, w_or_b_D, size*sizeof(float), cudaMemcpyDeviceToHost);

	return 0;
}

int weight_gradient_wrap(float *a, float *delta, float *weightG, /*size of a*/unsigned int columns, /*size of delta*/unsigned int rows) {
	
	float *a_D, *delta_D, *weightG_D;
	if (cudaMalloc((void **)&a_D, columns*sizeof(float)) != cudaSuccess) {
		return -1;
	}
	if (cudaMalloc((void **)&delta_D, rows*sizeof(float)) != cudaSuccess) {
		return -2;
	}
	if (cudaMalloc((void **)&weightG_D, columns*rows*sizeof(float)) != cudaSuccess) {
		return -3;
	}

	dim3 block(32, 32, 1);
	dim3 grid(columns/32+1, rows/32+1, 1);

	cudaMemcpy(a_D, a, columns*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(delta_D, delta, rows*sizeof(float), cudaMemcpyHostToDevice);

	weight_gradient<<<grid, block>>>(a_D, delta_D, weightG_D, columns, rows);

	cudaError error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s", cudaGetErrorString(error));
	}
	cudaMemcpy(weightG, weightG_D, columns*rows*sizeof(float), cudaMemcpyDeviceToHost);

	return 0;

}