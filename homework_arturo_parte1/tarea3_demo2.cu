#include <stdio.h>
#include <math.h>
#include <time.h>

#define NPOINTS 1024 //It has to be a power of two (from 64 to 1024)
#define NDIMS 5
#define ITERS 1000

float points_x[NPOINTS][NDIMS];
float points_x_copy[NPOINTS][NDIMS];
float points_y[NPOINTS];
float w[NDIMS];

double rand_gen() {
   // return a uniformly distributed random value
   return ((double)(rand())+1.)/((double)(RAND_MAX)+1.);
}

double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}


void print_points() {
	for (int i=0; i<NPOINTS; i++) {
		for (int j=0; j<NDIMS; j++) {
			printf("%.3f\t", points_x[i][j]);
		}
		printf("---> %.3f\n", points_y[i]);
	}
}


float compute_cost() {
	float cost, dotprod;
	
	cost = 0.0;
	for (int i=0; i<NPOINTS; i++) {
		dotprod = 0.0;
		for (int j=0; j<NDIMS; j++) {
			dotprod += w[j]*points_x[i][j];
		}
		cost += (dotprod-points_y[i])*(dotprod-points_y[i]);
	}
	cost /= (1.0*NPOINTS);

	return cost;
}




__global__ void compute_score(float *d_points_x, float *d_w, float *d_score)
{
    int point_idx = blockIdx.x; //index for point
    int dim_idx = threadIdx.x; //index for dimension
    float res = 0.0;

    extern __shared__ float sdata[];

    sdata[dim_idx] = d_w[dim_idx] * d_points_x[point_idx*blockDim.x + dim_idx];
    __syncthreads();

    if (dim_idx == 0) {
    	for (int k=0; k<NDIMS; k++)
    		res += sdata[k];
    		d_score[point_idx] = res;
    }
}


__global__ void substract_y(float *d_score, float *d_points_y)
{
    int i = threadIdx.x; //index for point

    d_score[i] -=  d_points_y[i];
}


__global__ void scale_x(float *d_score, float *d_points_x, float *d_points_x_copy)
{
    int i = blockIdx.x; //index for point
    int j = threadIdx.x; //index for element

    d_points_x_copy[i*blockDim.x+j] = 2.0*d_score[i]*d_points_x[i*blockDim.x+j];
}


__global__ void compute_gradient(float *d_points_x_copy, float *d_partial_grad)
{
	int block_idx = blockIdx.x; //index for block
	int point_idx = threadIdx.x; //index for point
	int dim_idx = threadIdx.y; //index for dimension

    __shared__ float sdata[64][NDIMS];

    sdata[point_idx][dim_idx] = d_points_x_copy[(block_idx+1)*point_idx*NDIMS + dim_idx];
    __syncthreads();

	for (unsigned int s=1; s < blockDim.x; s *= 2) {
		int index = 2 * s * point_idx;
		if (index < blockDim.x) {
			sdata[index][dim_idx] += sdata[index + s][dim_idx];
		}
		__syncthreads();
	}

	if (point_idx == 0)
		d_partial_grad[block_idx*NDIMS + dim_idx] = sdata[0][dim_idx]/64.0;
}




void gradientDescent() {
	
	float eta = 0.01;
	float gradient[NDIMS];
	float partial_grad[NPOINTS/64][NDIMS];

	for (int i=0; i<NDIMS; i++) {
		w[i] = 0.;
		gradient[i] = 0.;
	}

	float *d_points_x, *d_w, *d_score, *d_points_x_copy, *d_points_y, *d_partial_grad;
	cudaMalloc((void**)&d_points_x, sizeof(float)*NPOINTS*NDIMS);
	cudaMalloc((void**)&d_w, sizeof(float)*NDIMS);
	cudaMalloc((void**)&d_score, sizeof(float)*NPOINTS);
	cudaMalloc((void**)&d_points_x_copy, sizeof(float)*NPOINTS*NDIMS);
	cudaMalloc((void**)&d_points_y, sizeof(float)*NPOINTS);
	cudaMalloc((void**)&d_partial_grad, sizeof(float)*(NPOINTS/64)*NDIMS);


	cudaMemcpy(d_points_x, points_x, sizeof(float)*NPOINTS*NDIMS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_points_y, points_y, sizeof(float)*NPOINTS, cudaMemcpyHostToDevice);

	for (int k=0; k<ITERS; k++) {

		cudaMemcpy(d_w, w, sizeof(float)*NDIMS, cudaMemcpyHostToDevice);
		
		compute_score<<<NPOINTS,NDIMS,NDIMS*sizeof(float)>>>(d_points_x, d_w, d_score);

		substract_y<<<1,NPOINTS>>>(d_score, d_points_y);
		
		scale_x<<<NPOINTS,NDIMS>>>(d_score, d_points_x, d_points_x_copy);

		dim3 threads(64,NDIMS);
		compute_gradient<<<(NPOINTS/64),threads>>>(d_points_x_copy, d_partial_grad);

		cudaMemcpy(partial_grad, d_partial_grad, sizeof(float)*(NPOINTS/64)*NDIMS, cudaMemcpyDeviceToHost);

		for (int i=0; i<(NPOINTS/64); i++) {
			for (int j=0; j<NDIMS; j++) {
				gradient[j] += partial_grad[i][j];
			}
		}
		for (int i=0; i<NDIMS; i++) {
			gradient[i] /= (NPOINTS/64.0);
		}

		//Update weights
		for (int i=0; i<NDIMS; i++) {
			w[i] -= eta*gradient[i];
		}
	}

	cudaDeviceSynchronize();

	cudaFree(d_points_x);
	cudaFree(d_w);
	cudaFree(d_score);
	cudaFree(d_points_x_copy);
	cudaFree(d_points_y);
	cudaFree(d_partial_grad);

	printf("Total of points: %d\n",NPOINTS);
	printf("Weights: ");
	for (int i=0; i<NDIMS; i++) {
		printf("%.4f\t", w[i]);
	}
	printf("\n");
	printf("Final cost: %.4f\n", compute_cost());
	
}


int main() {
	
	float true_w[NDIMS] = {1, 2, 3, 4, 5};

	srand(time(0));

	//Create dataset
	float dotprod;
	for (int i=0; i<NPOINTS; i++) {
		for (int j=0; j<NDIMS; j++) {
			points_x[i][j] = (float) normalRandom();
		}
	}
	for (int i=0; i<NPOINTS; i++) {
		dotprod = 0.0;
		for (int j=0; j<NDIMS; j++) {
			dotprod += true_w[j]*points_x[i][j];
		}
		points_y[i] = dotprod + (float)normalRandom()*0.2;
	}


	
	gradientDescent();

	return 0;
}