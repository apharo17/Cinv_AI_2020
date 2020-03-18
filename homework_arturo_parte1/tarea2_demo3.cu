#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_STEPS 1000
#define XMAX 10.0
#define XMIN (-10.0)


float f(float x) {
	return pow((double)(x-3),2);
}


int main() {
	
	float x;
	float cost;
	
	float frac;
	float temp;
	float t;
	float delta;
	float x_;
	float cost_;
	float p;

	srand(time(0));

	temp = (1.0*rand()/RAND_MAX);
	temp = (1.0*rand()/RAND_MAX);
	x = (XMAX-XMIN)*temp + XMIN;
	cost = f(x);

	for (int step=0; step<MAX_STEPS; step++) {
		//printf("step: %d\t x: %.4f\t cost: %.4f\n", step, x, cost);

		frac = ((float)step)/MAX_STEPS;

		//Get temperature
		t = (1.0-frac > 0.01) ? 1.0-frac : 0.01;

		//Get random neighbour
		temp = (1.0*rand()/RAND_MAX);
    	delta = ((XMAX-XMIN)*frac/10.0)*temp - ((XMAX-XMIN)*frac/10.0)/2.0;
    	temp = (x+delta < XMAX) ? x+delta : XMAX;
    	x_ = (temp > XMIN) ? temp : XMIN;
    	
    	cost_ = f(x_);

    	if (cost_ < cost)
    		p = 1.0;
    	else
    		p = exp(double(-(cost_-cost)/t));

    	temp = (1.0*rand()/RAND_MAX);
    	if (p > temp)
    		x = x_;
    		cost = cost_;

	}

	printf("x: %.4f\t cost: %.4f\n", x, f(x));

	return 0;
}