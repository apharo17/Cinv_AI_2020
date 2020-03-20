#include<stdio.h>
#define N 4


__global__ void add(float *d_a, float *d_b, int *d_i) {
	float temp;
	int i = *d_i;
	int j = blockIdx.x + i + 1;
	

	//temp = d_a[j][i]/d_a[i][i];
	temp = d_a[j*N+i]/d_a[i*N+i];
	for (int k=i; k<N; k++)
	{
		//d_a[j][k] = d_a[j][k] - d_a[i][k]*temp;
		d_a[j*N+k] = d_a[j*N+k] - d_a[i*N+k]*temp;
	}
	d_b[j] = d_b[j] - d_b[i]*temp;
}


float a[N][N];
float b[N];


void print_matrix() {

	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			printf("%.2f\t",a[i][j]);
		}
		printf("%.2f\n",b[i]);
	}
	printf("\n\n");
}


int main(void) {

	a[0][0]=1;
	a[0][1]=-2;
	a[0][2]=2;
	a[0][3]=-3;
	b[0]=15;

	a[1][0]=3;
	a[1][1]=4;
	a[1][2]=-1;
	a[1][3]=1;
	b[1]=-6;

	a[2][0]=2;
	a[2][1]=-3;
	a[2][2]=2;
	a[2][3]=-1;
	b[2]=17;

	a[3][0]=1;
	a[3][1]=1;
	a[3][2]=-3;
	a[3][3]=-2;
	b[3]=-7;

	
	print_matrix();

	float *d_a;
	float *d_b;
	int *d_i;
	cudaMalloc((void**)&d_a, N*N*sizeof(float));
	cudaMalloc((void**)&d_b, N*sizeof(float));
	cudaMalloc((void**)&d_i, sizeof(int));
	cudaMemcpy(d_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

	for (int i=0; i<N-1; i++)
	{
		cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
		add<<<N-i-1,1>>>(d_a, d_b, d_i);
		cudaDeviceSynchronize();
		cudaMemcpy(a, d_a, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);
		print_matrix();			
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_i);

	return 0;
}