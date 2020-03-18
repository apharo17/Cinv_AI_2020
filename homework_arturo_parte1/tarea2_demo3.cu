#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_STEPS 1000
#define XMAX 10.0
#define XMIN (-10.0)

double random() {
	return (double) rand() / (RAND_MAX);
}


double f(double x) {
	return x*x;
}


int main() {
	
	double x;
	double cost;
	
	double frac;
	double temp;
	double t;
	double amp;
	double delta;
	double x_;
	double cost_;
	double p;

	srand(time(0));

	temp = (1.0*rand()/RAND_MAX);
	//printf("%.4f\n",temp);
	//printf("%d\n",rand());
	//printf("%d\n",RAND_MAX);
	//printf("%.4f\n",(1.0*rand()/RAND_MAX));
	temp = (1.0*rand()/RAND_MAX);
	//printf("%.4f\n",temp);

	x = (XMAX-XMIN)*temp + XMIN;
	cost = f(x);

	for (int step=0; step<MAX_STEPS; step++) {
		//printf("step: %d\t x: %.4f\t cost: %.4f\n", step, x, cost);

		frac = ((double)step)/MAX_STEPS;

		//Get temperature
		t = (1.0-frac > 0.01) ? 1.0-frac : 0.01;

		//Get random neighbour
		amp = (XMAX-XMIN) * frac / 10.0;
		temp = (1.0*rand()/RAND_MAX);
		printf("%.4f\n",temp);
    	delta = amp*temp - amp/2.0;
    	temp = (x+delta < XMAX) ? x+delta : XMAX;
    	x_ = (temp > XMIN) ? temp : XMIN;
    	
    	cost_ = f(x_);

    	if (cost_ < cost)
    		p = 1.0;
    	else
    		p = exp(-(cost_-cost)/t);

    	temp = (1.0*rand()/RAND_MAX);
		printf("%.4f\n",temp);
    	if (p > temp)
    		x = x_;
    		cost = cost_;

	}

	printf("x: %.4f\t cost: %.4f\n", x, cost);

	return 0;
}