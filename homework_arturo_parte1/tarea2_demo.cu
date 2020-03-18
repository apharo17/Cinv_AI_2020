#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CYCLES 50
#define TRIALS 50


float f(float x0, float x1) {
	return 0.2 + pow(x0, 2) + pow(x1, 2) - 0.1*cos(6.0*3.1415*x0) - 0.1*cos(6.0*3.1415*x1);
}


float random() {
	return (float) rand() / (RAND_MAX);
}


int main(void) {
	
	srand(time(0));

	float x0_start = 0.8;
	float x1_start = -0.5;
	int num_solutions = 0;
	
	float p_start = 0.7;
	float p_end = 0.001;
	float t_start = -1.0/log(p_start);
	float t_end = -1.0/log(p_end);
	float frac = pow((t50/t1), 1.0/(CYCLES-1.0));

	float x0_arr[CYCLES+1];
	float x1_arr[CYCLES+1];
	float x0_i;
	float x1_i;

	x0_arr[0] = x0_start;
	x1_arr[0] = x1_start;

	x0_i = x0_start;
	x1_i = x1_start;
	num_solutions++;

	//Current best result so far
	float xc0 = x0_arr[0];
	float xc1 = x1_arr[0];
	float fc = f(x0_i, x1_i);
	float fs[CYCLES+1];
	fs[0] = fc;
	float t = t_start;
	float DeltaE_avg = 0.0;


	//--------------------------
	float temp;
	float DeltaE;
	float p;
	bool accept;

	for(int i=0; i<CYCLES; i++) {
		printf("Cycle: %d\tTemperature: %.4f\n", i+1, t);

		for(int j=0; j<TRIALS; j++) {

			//Generate trial points
			x0_i = xc0 + random() - 0.5;
			x1_i = xc1 + random() - 0.5;

			//Clip to upper and lower bounds
			temp = (x0_i < 1.0) ? x0_i : 1.0;
			x0_i = (temp > -1.0) ? temp : -1.0;
			temp = (x1_i < 1.0) ? x1_i : 1.0;
			x1_i = (temp > -1.0) ? temp : -1.0;
			DeltaE = fabs(f(x0_i, x1_i)-fc);
			
			if (f(x0_i, x1_i) > fc) {

				if (i==0 && j==0)
					DeltaE_avg = DeltaE;

				p = pow(-DeltaE/(DeltaE_avg * t));

				accept = (random() < p);
			}
			else {
				accept = true;
			}

			if (accept) {
				xc0 = x0_i;
				xc1 = x1_i;
				fc = f(xc0, xc1);
				num_solutions++; 
				DeltaE_avg = (DeltaE_avg * (num_solutions-1.0) + DeltaE) / num_solutions;
			}
		}

		x0_arr[i+1] = xc0;
		x1_arr[i+1] = xc1;
		fs[i+1] = fc;

		//Lower the temperature for next cycle
		t *= frac;
	}


	printf("Best solution: %.4f, %.4f\n", xc0, xc1);
	printf("Best objective: %.4f\n", fc);


	return 0;
}