#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/* change dimension size as needed */
const int dimension = 512 ;
const int blocksize = 32;
const int K = 1;


struct timeval tv; 

__global__ void gpuMM(float *A, float *B, float *C, int N)
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


/*float timestamp()
{
        float t;
        gettimeofday(&tv, NULL);
        t = tv.tv_sec + (tv.tv_usec/1000000.0);
        return t;
}
*/

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
                        A[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
                        B[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
                        C[dimension*i+j] = 0.0;
                }   
				
		cudaMalloc( (void**)&Ad, dimension*dimension*sizeof(float) );
		cudaMemcpy( Ad, A, dimension*dimension*sizeof(float), cudaMemcpyHostToDevice );

		cudaMalloc( (void**)&Bd, dimension*dimension*sizeof(float) );
		cudaMemcpy( Bd, B, dimension*dimension*sizeof(float), cudaMemcpyHostToDevice );
		
		cudaMalloc( (void**)&Cd, dimension*dimension*sizeof(float) );

		dim3 threadBlock(blocksize,blocksize);
		dim3 grid(K,K);
		
        //start = timestamp();
		cudaEventRecord( start1, 0 );

		gpuMM<<<grid,threadBlock>>>( Ad,Bd,Cd,dimension);

        //end = timestamp();

		cudaEventRecord( stop1, 0 );
		cudaEventSynchronize( stop1 );
		
		cudaEventElapsedTime( &time, start1, stop1 );

		printf("\nsecs:%f\n", time);
		cudaEventDestroy( start1 );
		cudaEventDestroy( stop1 );

		cudaMemcpy(C,Cd,dimension*dimension*sizeof(float),cudaMemcpyDeviceToHost);

        //printf("\nsecs:%f\n", end-start);

        free(A);
        free(B);
        free(C);
		cudaFree(Ad);
		cudaFree(Bd);
		cudaFree(Cd);

        return 0;
}
