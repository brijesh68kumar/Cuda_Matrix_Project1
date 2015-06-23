#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/* change dimension size as needed */
const int dimension = 512;
struct timeval tv; 

double timestamp()
{
        double t;
        gettimeofday(&tv, NULL);
        t = tv.tv_sec + (tv.tv_usec/1000000.0);
        return t;
}

int main(int argc, char *argv[])
{

		cudaEvent_t start1, stop1;
		float time;

		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
        
        int i, j, k;
        double *A, *B, *C, begin, start, end;
	
	begin = timestamp();

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

        start = timestamp();

		cudaEventRecord( start1, 0 );

        for(i = 0; i < dimension; i++)
                for(j = 0; j < dimension; j++)
                        for(k = 0; k < dimension; k++)
                                C[dimension*i+j] += A[dimension*i+k] *
                                        B[dimension*k+j];

        end = timestamp();
		cudaEventRecord( stop1, 0 );
		cudaEventSynchronize( stop1 );
		
		cudaEventElapsedTime( &time, start1, stop1 );

		printf("\ncuda secs:%f\n\n", time);
		cudaEventDestroy( start1 );
		CudaEventDestroy( stop1 );

        printf("\n cpu secs:%f\n\n", end-start);
	printf("init %f\n",(start - begin));
        free(A);
        free(B);
        free(C);

        return 0;
}
