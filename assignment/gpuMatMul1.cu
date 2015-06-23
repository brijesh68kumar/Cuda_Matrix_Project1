#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/* change dimension size as needed */
const int dimension = 4096 ;
const int blocksize = 10;
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


int main(int argc, char *argv[])
{
		cudaEvent_t start_i, stop_i,start_mc_h2d, stop_mc_h2d,start_mc_d2h, stop_mc_d2h,start_pl, stop_pl;
		float time_i,time_mc_h2d,time_mc_d2h,time_pl;

		cudaEventCreate(&start_i);
		cudaEventCreate(&stop_i);
		
		cudaEventCreate(&start_mc_h2d);
		cudaEventCreate(&stop_mc_h2d);

		cudaEventCreate(&start_mc_d2h);
		cudaEventCreate(&stop_mc_d2h);
		
		cudaEventCreate(&start_pl);
		cudaEventCreate(&stop_pl);
		
		int i, j;
        float *A, *B, *C;// start, end;
		float *Ad, *Bd, *Cd;

		cudaEventRecord( start_i, 0 );
		
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
				
		cudaEventRecord( stop_i, 0 );
		cudaEventSynchronize( stop_i );
		
		cudaEventElapsedTime( &time_i, start_i, stop_i );


		cudaEventRecord( start_mc_h2d, 0 );
		
		cudaMalloc( (void**)&Ad, dimension*dimension*sizeof(float) );
		cudaMemcpy( Ad, A, dimension*dimension*sizeof(float), cudaMemcpyHostToDevice );

		cudaMalloc( (void**)&Bd, dimension*dimension*sizeof(float) );
		cudaMemcpy( Bd, B, dimension*dimension*sizeof(float), cudaMemcpyHostToDevice );
		
		cudaMalloc( (void**)&Cd, dimension*dimension*sizeof(float) );

		cudaEventRecord( stop_mc_h2d, 0 );
		cudaEventSynchronize( stop_mc_h2d );
		
		cudaEventElapsedTime( &time_mc_h2d, start_mc_h2d, stop_mc_h2d );

		

        //start = timestamp();

		cudaEventRecord( start_pl, 0 );
		dim3 threadBlock(blocksize,blocksize);
		dim3 grid(K,K);
		
		gpuMM<<<grid,threadBlock>>>( Ad,Bd,Cd,dimension);

        //end = timestamp();

		cudaEventRecord( stop_pl, 0 );
		cudaEventSynchronize( stop_pl );
		
		cudaEventElapsedTime( &time_pl, start_pl, stop_pl );

		cudaEventRecord( start_mc_d2h, 0 );
		cudaMemcpy(C,Cd,dimension*dimension*sizeof(float),cudaMemcpyDeviceToHost);
		cudaEventRecord( stop_mc_d2h, 0 );
		cudaEventSynchronize( stop_mc_d2h );
		
		cudaEventElapsedTime( &time_mc_d2h, start_mc_d2h, stop_mc_d2h );

		
		printf("\n IT : %f   ", time_i);
		printf(" MCT : %f", ( time_mc_d2h + time_mc_h2d ) );
		printf(" PLT:%f ", time_pl);
		printf(" Total:%f.... \n\n", (time_pl + time_mc_d2h + time_mc_h2d+time_i));

		
        //printf("\nsecs:%f\n", end-start);

		cudaEventDestroy( start_i );
		cudaEventDestroy( stop_i );

		cudaEventDestroy( start_mc_d2h );
		cudaEventDestroy( stop_mc_d2h );

		cudaEventDestroy( start_mc_h2d );
		cudaEventDestroy( stop_mc_h2d );
		
		cudaEventDestroy( start_pl );
		cudaEventDestroy( stop_pl );
		
        free(A);
        free(B);
        free(C);
		cudaFree(Ad);
		cudaFree(Bd);
		cudaFree(Cd);

        return 0;
}
