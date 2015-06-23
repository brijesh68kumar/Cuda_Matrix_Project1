#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/* change dimension size as needed */
const int dimension = 32 ;
const int blocksize = 10;
const int K = 1;


struct timeval tv; 

__global__ void gpuMM(double *A, double *B, double *C, int N)
{
	// Matrix multiplication for NxN matrices C=A*B
	// Each thread computes a single element of C
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	float sum = 0.f;
	for (int n = 0; n < N; ++n)
	    sum += A[row*N+n]*B[n*N+col];

	C[row*N+col] = sum;
}


double timestamp()
{
        double t;
        gettimeofday(&tv, NULL);
        t = tv.tv_sec + (tv.tv_usec/1000000.0);
        return t;
}

int main(int argc, char *argv[])
{
        int i, j;
        double *A, *B, *C, start, end;
		double *Ad, *Bd, *Cd;

        A = (double*)malloc(dimension*dimension*sizeof(double));
        B = (double*)malloc(dimension*dimension*sizeof(double));
        C = (double*)malloc(dimension*dimension*sizeof(double));		
		
        srand(292);

        for(i = 0; i < dimension; i++)
                for(j = 0; j < dimension; j++)
                {   
                        A[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
                        B[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
                        C[dimension*i+j] = 0.0;
                }   
				
		cudaMalloc( (void**)&Ad, dimension*dimension*sizeof(double) );
		cudaMemcpy( Ad, A, dimension*dimension*sizeof(double), cudaMemcpyHostToDevice );

		cudaMalloc( (void**)&Bd, dimension*dimension*sizeof(double) );
		cudaMemcpy( Bd, B, dimension*dimension*sizeof(double), cudaMemcpyHostToDevice );
		
		cudaMalloc( (void**)&Cd, dimension*dimension*sizeof(double) );

		dim3 threadBlock(blocksize,blocksize);
		dim3 grid(K,K);
		
        start = timestamp();

		gpuMM<<<grid,threadBlock>>>( Ad,Bd,Cd,dimension);

        end = timestamp();
		
		cudaMemcpy(C,Cd,dimension*dimension*sizeof(double),cudaMemcpyDeviceToHost);

        printf("\nsecs:%f\n", end-start);

        free(A);
        free(B);
        free(C);
		cudaFree(Ad);
		cudaFree(Bd);
		cudaFree(Cd);

        return 0;
}
