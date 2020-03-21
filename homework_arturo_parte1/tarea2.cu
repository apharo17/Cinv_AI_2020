#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>


#define N 1024 //It has to be a power of 2, and the max is 1024
//Local minima is +-4.49341

#define MAX_STEPS 1000
#define XMAX 10.0
#define XMIN (-10.0)



__global__ void init(unsigned int seed, curandState_t *states) {

  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}



__global__ void sim_ann(curandState_t *states, float *d_res_arr){

	float x;
	float cost;
	float frac;
	float temp;
	float t;
	float delta;
	float x_;
	float cost_;
	float p;

	x = (XMAX-XMIN)*curand_uniform(&states[blockIdx.x]) + XMIN;

	cost = sin(x)/x;

	for (int step=0; step<MAX_STEPS; step++) {

		frac = ((float)step)/MAX_STEPS;

		//Get temperature
		t = (1.0-frac > 0.01) ? 1.0-frac : 0.01;

		//Get random neighbour
    	delta = ((XMAX-XMIN)*frac/10.0)*curand_uniform(&states[blockIdx.x]) - ((XMAX-XMIN)*frac/10.0)/2.0;

    	temp = (x+delta < XMAX) ? x+delta : XMAX;
    	x_ = (temp > XMIN) ? temp : XMIN;
    	
    	cost_ = sin(x_)/x_;

    	if (cost_ < cost)
    		p = 1.0;
    	else
    		p = expf(double(-(cost_-cost)/t));

    	temp = curand_uniform(&states[blockIdx.x]);
    	if (p > temp)
    		x = x_;
    		cost = cost_;
	}

	d_res_arr[blockIdx.x] = x;
}



__global__ void get_best(float *d_res_arr, float *d_res) {
    
    int k = threadIdx.x;
 	float cost1, cost2;
    extern __shared__ float sdata[];

    sdata[k] = d_res_arr[k];
	__syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s >>= 1) {
        if(k < s) {
            cost1 = sin(sdata[k])/sdata[k];
            cost2 = sin(sdata[k+s])/sdata[k+s];
            if(cost1 > cost2) {
                sdata[k] = sdata[k + s];
            }
        }
        __syncthreads();
    }

    if (k==0)
    	(*d_res) = sdata[0];
}




int main(void)
{
	float elapsedTime, bw;
	cudaEvent_t start, stop;

	float res;
	float *d_res;
	float *d_res_arr;
	curandState_t* states;
	
	cudaMalloc((void**) &states, sizeof(curandState_t)*N);
	cudaMalloc((void**)&d_res_arr, sizeof(float)*N);
	cudaMalloc((void**)&d_res, sizeof(float));


	// Creating events to estimate execution time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); // Starting clock
	bw = 0.0;


	init<<<N,1>>>(time(0), states);
	sim_ann<<<N,1>>>(states, d_res_arr);
	get_best<<<1,N,N*sizeof(float)>>>(d_res_arr, d_res);

	cudaDeviceSynchronize();

	cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);


	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);


    cudaFree(states);
    cudaFree(d_res_arr);
    cudaFree(d_res);

	printf("Best solution: %.5f\n", res);

	bw = 1.0*N/(elapsedTime);
	printf("Simulated Annealing GPU execution time: %7.3f ms, Throughput %6.3f KFLOPS\n", elapsedTime, bw);

	return 0;
}

