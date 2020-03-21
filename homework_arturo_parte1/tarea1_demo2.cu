#include<stdio.h>
#define N 1000


char input_file[]= "linsys_1000.txt";
float a[N][N];
float b[N];



__global__ void add(float *d_a, float *d_b, int *d_i) {
	float temp;
	int i = *d_i;
	
	//BlockId: the equation
	int j = blockIdx.x + i + 1;
	
	//ThreadId: column
	int k = threadIdx.x + i;

	temp = d_a[j*N+i]/d_a[i*N+i];
	
	if (k==N)
		d_b[j] = d_b[j] - d_b[i]*temp;

	else
		d_a[j*N+k] = d_a[j*N+k] - d_a[i*N+k]*temp;
	
}



void save_sysequ() {

	FILE* fp;

	fp = fopen("output.txt","w");
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N+1; j++)
		{
			if (j == N)
	          fprintf(fp, "%.2f\n", b[i]);
	        else
	          fprintf(fp, "%.2f\t", a[i][j]);
		}
	}
	fclose(fp);
}



void load_sysequ() {
	
	FILE* fp;

    fp = fopen(input_file,"r");
    for (int i=0; i<N; i++) {
      for (int j=0; j<N+1; j++) {
        if (j == N)
          fscanf(fp,"%f\n",&b[i]);
        else
          fscanf(fp,"%f ",&a[i][j]);
      }
    }
    fclose(fp);

}



int main(void) {

	float *d_a;
	float *d_b;
	int *d_i;

	load_sysequ();

	cudaMalloc((void**)&d_a, N*N*sizeof(float));
	cudaMalloc((void**)&d_b, N*sizeof(float));
	cudaMalloc((void**)&d_i, sizeof(int));
	cudaMemcpy(d_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

	for (int i=0; i<N-1; i++)
	{
		cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
		add<<<N-i-1,N-i+1>>>(d_a, d_b, d_i);		
	}

	cudaDeviceSynchronize();
	cudaMemcpy(a, d_a, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_i);

	save_sysequ();

	return 0;
}