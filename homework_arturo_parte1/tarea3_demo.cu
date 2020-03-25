#include <stdio.h>
#include <math.h>
#include <time.h>

#define NPOINTS 128
#define NDIMS 5
#define ITERS 1000

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


void gradientDescent() {
	float eta = 0.01;
	float w[NDIMS];
	float dotprod, cost, temp;
	float gradient[NDIMS];

	for (int i=0; i<NDIMS; i++)
		w[i] = 0;

	
	for (int k=0; k<ITERS; k++) {

		//Calculate the cost
		cost = 0.0;
		for (int i=0; i<NPOINTS; i++) {
			dotprod = 0.0;
			for (int j=0; j<NDIMS; j++) {
				dotprod += w[j]*points_x[i][j];
			}
			cost += (dotprod-points_y[i])*(dotprod-points_y[i]);
		}
		cost /= (1.0*NPOINTS);


		//Calculate the gradient of the cost
		for (int i=0; i<NPOINTS; i++) {
			dotprod = 0.0;
			for (int j=0; j<NDIMS; j++) {
				dotprod += w[j]*points_x[i][j];
			}

			temp = 2.0*(dotprod-points_y[i]);
			
			for (int j=0; j<NDIMS; j++) {
				gradient[j] += temp*points_x[i][j];
			}
		}
		for (int i=0; i<NDIMS; i++) {
			gradient[i] /= (1.0*NPOINTS);
		}


		//Update weights
		for (int i=0; i<NDIMS; i++) {
			w[i] -= eta*gradient[i];
		}


		if (k<10) {
			printf("Iteration %d, Cost = %.4f\n", k, cost);
			printf("Weights:\t");
			for (int i=0; i<NDIMS; i++) {
				printf("%.4f\t", w[i]);
			}
			printf("\n");
		}

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
	gradientDescent();

	return 0;
}