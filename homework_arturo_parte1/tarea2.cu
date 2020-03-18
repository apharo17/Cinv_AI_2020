#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

#define MAX_STEPS 1000
#define XMAX 10.0
#define XMIN (-10.0)
#define N 10


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

	cost = (x-3)*(x-3);
	//cost = (x+3)*(x+3);

	for (int step=0; step<MAX_STEPS; step++) {

		frac = ((float)step)/MAX_STEPS;

		//Get temperature
		t = (1.0-frac > 0.01) ? 1.0-frac : 0.01;

		//Get random neighbour
    	delta = ((XMAX-XMIN)*frac/10.0)*curand_uniform(&states[blockIdx.x]) - ((XMAX-XMIN)*frac/10.0)/2.0;

    	temp = (x+delta < XMAX) ? x+delta : XMAX;
    	x_ = (temp > XMIN) ? temp : XMIN;
    	
    	cost_ = (x_-3)*(x_-3);
    	//cost_ = (x_+3)*(x_+3);

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


int main(void)
{
	float res;
	float res_arr[N];
	float *d_res_arr;
	cudaMalloc((void**)&d_res_arr, sizeof(float)*N);

	curandState_t* states;
	cudaMalloc((void**) &states, sizeof(curandState_t)*N);
	init<<<N,1>>>(time(0), states);

	sim_ann<<<N,1>>>(states, d_res_arr);

	cudaDeviceSynchronize();
	cudaMemcpy(res_arr, d_res_arr, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaFree(d_res_arr);

    float cost, cost_, res_;
    res = res_arr[0];
    cost = (res-3)*(res-3);
    for (int i=1; i<N; i++) {
    	printf("%.4f\n",res_arr[i]);
    	res_ = res_arr[i];
    	cost_ = (res_-3)*(res_-3);
    	if (cost_ < cost)
    	{
    		res = res_;
    		cost = cost_;
    	}
    }

	printf("Solution: %.4f\n",res);

	return 0;
}

