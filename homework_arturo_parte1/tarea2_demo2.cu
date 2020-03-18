#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CYCLES 50
#define TRIALS 50
#define XMAX 25
#define XMIN -25

double f(double x) {
	//return pow(x-3,2);
	if (x==0)
        return 1;
    return sin(x)/x;
}


double random() {
	return (double) rand() / (RAND_MAX);
}


int main(void) {
	
	srand(time(0));

	double x_start = 9.0;
	int num_solutions = 0;
	
	double p_start = 0.7;
	double p_end = 0.001;
	double t_start = -1.0/log(p_start);
	double t_end = -1.0/log(p_end);
	double frac = pow((t_end/t_start), 1.0/(CYCLES-1.0));

	double x_arr[CYCLES+1];
	double xi;

	x_arr[0] = x_start;

	xi = x_start;
	num_solutions++;

	//Current best result so far
	double xc = x_arr[0];
	double fc = f(xi);
	double fc_arr[CYCLES+1];
	fc_arr[0] = fc;
	double t = t_start;
	double DeltaE_avg = 0.0;


	//--------------------------
	double temp;
	double DeltaE;
	double p;
	bool accept;

	for(int i=0; i<CYCLES; i++) {
		printf("Cycle: %d\tTemperature: %.4f\n", i+1, t);

		for(int j=0; j<TRIALS; j++) {

			//Generate trial point
			xi = xc + (random() - 0.5);

			//Clip to upper and lower bounds
			temp = (xi < XMAX) ? xi : XMAX;
			xi = (temp > XMIN) ? temp : XMIN;
			DeltaE = fabs(f(xi)-fc);
			
			if (f(xi) > fc) {

				if (i==0 && j==0)
					DeltaE_avg = DeltaE;

				p = exp(-DeltaE/(DeltaE_avg * t));

				accept = (random() < p);
			}
			else {
				accept = true;
			}

			if (accept) {
				xc = xi;
				fc = f(xc);
				num_solutions++; 
				DeltaE_avg = (DeltaE_avg * (num_solutions-1.0) + DeltaE) / num_solutions;
			}
		}

		x_arr[i+1] = xc;
		fc_arr[i+1] = fc;
		printf("x: %.4f\tf: %.4f\n", xc, fc);

		//Lower the temperature for next cycle
		t *= frac;
	}


	printf("Best solution: %.4f\n", xc);
	printf("Best objective: %.4f\n", fc);


	return 0;
}