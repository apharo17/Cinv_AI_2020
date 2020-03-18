#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

#define MAX_STEPS 1000
#define XMAX 10.0
#define XMIN (-10.0)


__global__ void sim_ann(unsigned int seed, float *d_res){

	float x;
	float cost;
	float frac;
	float temp;
	float t;
	float delta;
	float x_;
	float cost_;
	float p;

	curandState_t state;
  	curand_init(seed, 0, 0, &state);

	x = (XMAX-XMIN)*curand_uniform(&state) + XMIN;
	cost = (x-3)*(x-3);
	//cost = (x+3)*(x+3);

	for (int step=0; step<MAX_STEPS; step++) {

		frac = ((float)step)/MAX_STEPS;

		//Get temperature
		t = (1.0-frac > 0.01) ? 1.0-frac : 0.01;

		//Get random neighbour
    	delta = ((XMAX-XMIN)*frac/10.0)*curand_uniform(&state) - ((XMAX-XMIN)*frac/10.0)/2.0;
    	temp = (x+delta < XMAX) ? x+delta : XMAX;
    	x_ = (temp > XMIN) ? temp : XMIN;
    	
    	cost_ = (x_-3)*(x_-3);
    	//cost_ = (x_+3)*(x_+3);

    	if (cost_ < cost)
    		p = 1.0;
    	else
    		p = expf(double(-(cost_-cost)/t));

    	if (p > curand_uniform(&state))
    		x = x_;
    		cost = cost_;
	}

	(*d_res) = x;
}


int main(void)
{
	float res;
	float *d_res;
	cudaMalloc((void**)&d_res, sizeof(float));

	sim_ann<<<1,1>>>(time(NULL), d_res);

	cudaDeviceSynchronize();
	cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_res);

	printf("Solution: %.4f\n",res);

	return 0;
}

