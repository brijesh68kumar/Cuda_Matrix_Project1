#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/* change dimension size as needed */
const int dimension = 1024 ;
const int blocksize = 20;
const int K = 16;

__global__ void gpuMM(float *A, float *B, float *C,int N)
{
	
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	float sum = 0;
	for (int n = 0; n < N; ++n)
	    sum =sum+ A[row*N+n]*B[n*N+col];

	C[row*N+col] = sum;
	for (int n = 0; n < N; ++n)
	printf("%d\t",C[row*dimension+col]);
}


int main(int argc, char *argv[])
{
		cudaEvent_t start1, stop1;
		float time;

		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
        
		int i, j;
        float *A, *B, *C;// start, end;
		float *Ad, *Bd, *Cd;

        A = (float*)malloc(dimension*dimension*sizeof(float));
        B = (float*)malloc(dimension*dimension*sizeof(float));
        C = (float*)malloc(dimension*dimension*sizeof(float));		
		
        srand(292);

        for(i = 0; i < dimension; i++)
                for(j = 0; j < dimension; j++)
                {   
                        A[dimension*i+j] = 1;
                        B[dimension*i+j] = 2;
                       // C[dimension*i+j] = 0.0;
                }   
				
		cudaMalloc( (void**)&Ad, dimension*dimension*sizeof(float) );
		cudaMemcpy( Ad, A, dimension*dimension*sizeof(float), cudaMemcpyDefault );

		cudaMalloc( (void**)&Bd, dimension*dimension*sizeof(float) );
		cudaMemcpy( Bd, B, dimension*dimension*sizeof(float), cudaMemcpyDefault );
		
		cudaMalloc( (void**)&Cd, dimension*dimension*sizeof(float) );

		dim3 threadBlock(blocksize,blocksize);
		dim3 grid(K,K);
		
        //start = timestamp();
		cudaEventRecord( start1, 0 );

		gpuMM<<<grid,threadBlock>>>( Ad,Bd,Cd,dimension);
		cudaMemcpy( Cd, C, dimension*dimension*sizeof(float), cudaMemcpyDefault );

        //end = timestamp();

		cudaEventRecord( stop1, 0 );
		cudaEventSynchronize( stop1 );
		
		cudaEventElapsedTime( &time, start1, stop1 );
		
		//print(C,N);
		printf("\nTime to calculate results on GPU: %f ms.\n", time);
		cudaEventDestroy( start1 );
		cudaEventDestroy( stop1 );
        

        free(A);
        free(B);
        free(C);
		cudaFree(Ad);
		cudaFree(Bd);
		cudaFree(Cd);

        return 0;
}
