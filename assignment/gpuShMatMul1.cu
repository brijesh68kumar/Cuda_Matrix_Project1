#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/* change dimension size as needed */
const int dimension = 4096 ;
const int blocksize = 64;
const int K = 4;
const int tilewidth = 2 ;


struct timeval tv; 

__global__ void gpuSmMM( float *Ad , float *Bd , float *Cd , int dimention )
{

        //Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
          __shared__ float Ads [tilewidth][tilewidth] ;
          __shared__ float Bds [tilewidth][tilewidth] ;
         // calculate thread id
          unsigned int col = tilewidth*blockIdx.x + threadIdx.x ;
          unsigned int row = tilewidth*blockIdx.y + threadIdx.y ;
        for (int m = 0 ; m<dimention/tilewidth ; m++ ) // m indicate number of phase
			{
            Ads[threadIdx.y][threadIdx.x] =  Ad[row*dimention + (m*tilewidth + threadIdx.x)]  ;
            Bds[threadIdx.y][threadIdx.x] =  Bd[ ( m*tilewidth + threadIdx.y) * dimention + col] ;
         __syncthreads() ; // for syncronizeing the threads
         // Do for tile
				for ( int k1 = 0; k1<tilewidth ; k1++ )
                       Cd[row*dimention + col]+= Ads[threadIdx.x][k1] * Bds[k1][threadIdx.y] ;
         __syncthreads() ; // for syncronizeing the threads

			}
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
		
		gpuSmMM<<<grid,threadBlock>>>( Ad,Bd,Cd,dimension);

        //end = timestamp();

		cudaEventRecord( stop_pl, 0 );
		cudaEventSynchronize( stop_pl );
		
		cudaEventElapsedTime( &time_pl, start_pl, stop_pl );

		cudaEventRecord( start_mc_d2h, 0 );
		cudaMemcpy(C,Cd,dimension*dimension*sizeof(float),cudaMemcpyDeviceToHost);
		cudaEventRecord( stop_mc_d2h, 0 );
		cudaEventSynchronize( stop_mc_d2h );
		
		cudaEventElapsedTime( &time_mc_d2h, start_mc_d2h, stop_mc_d2h );

		
		//printf("IT: %f ", time_i);
		printf("MC: %f ", ( time_mc_d2h + time_mc_h2d ) );
		printf("PLT: %f \n ", time_pl);
		//printf("T:%f .... \n\n", (time_pl + time_mc_d2h + time_mc_h2d+time_i));

		
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
