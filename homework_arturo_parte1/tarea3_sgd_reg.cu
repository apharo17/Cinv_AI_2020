#include <stdio.h>
#include <math.h>
#include <time.h>

#include <stdio.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define NPOINTS 64
#define NDIMS 5
#define ITERS 10

float points_x[NPOINTS][NDIMS];
float points_y[NPOINTS];


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


__global__ void init(unsigned int seed, curandState_t *states) {

  curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}


__global__ void fisher_yates(curandState_t *states, int *d_permuted) {
  
  int id  = threadIdx.x;
  int temp;

  extern __shared__ int perm_shared[];
  perm_shared[2*id] = 2*id;
  perm_shared[2*id+1] = 2*id+1;
  __syncthreads();

  unsigned int shift = 1;
  unsigned int pos = id*2;  
  while(shift <= blockDim.x)
  {
      if (curand(&states[id]) & 1) {
        temp = perm_shared[pos];
        perm_shared[pos] = perm_shared[pos+shift];
        perm_shared[pos+shift] = temp;
      }
      
      shift = shift << 1;
      pos = (pos & ~shift) | ((pos & shift) >> 1);
      __syncthreads();
  }

  d_permuted[2*id] = perm_shared[2*id];
  d_permuted[2*id+1] = perm_shared[2*id+1];
}


void sgradientDescent() {
	float eta = 0.01;
	float w[NDIMS];
	float dotprod, temp;
	float gradient[NDIMS];

	for (int i=0; i<NDIMS; i++)
		w[i] = 0;

	curandState_t* states;
  	cudaMalloc((void**) &states, sizeof(curandState_t)*(NPOINTS/2));
  	init<<<(NPOINTS/2),1>>>(time(0), states);

  	int indexes[NPOINTS];
  	int index;
  	int *d_indexes;
  	cudaMalloc((void**) &d_indexes, sizeof(int)*NPOINTS);

	
	for (int k=0; k<ITERS; k++) {

		fisher_yates<<<1,(NPOINTS/2),NPOINTS>>>(states, d_indexes);
		cudaMemcpy(indexes, d_indexes, sizeof(int)*NPOINTS, cudaMemcpyDeviceToHost);

		//Calculate the gradient of the cost
		for (int i=0; i<NPOINTS; i++) {
			
			index = indexes[i];

			dotprod = 0.0;
			for (int j=0; j<NDIMS; j++) {
				dotprod += w[j]*points_x[index][j];
			}

			temp = 2.0*(dotprod-points_y[index]);
			
			for (int j=0; j<NDIMS; j++) {
				gradient[j] = temp*points_x[index][j];
			}
			for (int j=0; j<NDIMS; j++) {
				w[j] -= eta*gradient[j];
			}
		}

		printf("Iteration %d\n", k);
		printf("Weights:\t");
		for (int i=0; i<NDIMS; i++) {
			printf("%.4f\t", w[i]);
		}
		printf("\n");
	}

}


int main() {
	
	float true_w[NDIMS] = {1, 2, 3, 4, 5};

	srand(time(0));

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
		points_y[i] = dotprod + (float)normalRandom()*.2;
	}
	
	//print_points();
	sgradientDescent();

	return 0;
}