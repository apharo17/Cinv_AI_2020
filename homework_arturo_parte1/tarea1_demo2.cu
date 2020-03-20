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


void print_sysequ() {

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

	FILE* fp;

    fp = fopen("linsys_4_a.txt","r");
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        if (j == N-1)
          fscanf(fp,"%f\n",&a[i][j]);
        else
          fscanf(fp,"%f ",&a[i][j]);
      }
    }
    fclose(fp);

    fp = fopen("linsys_4_b.txt","r");
    for (int i=0; i<N; i++) {
      fscanf(fp,"%f\n",&b[i]);
    }
    fclose(fp);
	print_sysequ();



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
		print_sysequ();			
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_i);

	return 0;
}