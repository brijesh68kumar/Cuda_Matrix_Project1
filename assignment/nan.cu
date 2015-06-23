#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

// Thread block sizes
#define BLOCK_SIZE 32
#define TILE_WIDTH 6
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define MATRIX_SIZE  128 *BLOCK_SIZE

#define WA (MATRIX_SIZE) // Matrix A width
#define HA (MATRIX_SIZE) // Matrix A height
#define WB (MATRIX_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
        //data[i] = rand() %2 ;
}

void printOutput(float *C, char a){
    int i=0;
    printf("Printing %c\n", a);
    for(i=0;i<100;i++){
        printf("%f\t", *C++);
    }
}
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
   // Calculate the row index of the Pd element and M
   int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
   // Calculate the column idenx of Pd and N
   int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

   float Pvalue = 0;
   // each thread computes one element of the block sub-matrix
   for (int k = 0; k < Width; ++k){
       Pvalue += Md[Row*Width+k] * Nd[k*Width+Col];
   }

   Pd[Row*Width+Col] = Pvalue;
}



__global__ void MatrixMulKernelTiled(float* Md, float* Nd, float* Pd, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    // Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < Width/TILE_WIDTH; ++m) {
    // Collaborative loading of Md and Nd tiles into shared memory
        Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH + ty)*Width];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    Pd[Row*Width+Col] = Pvalue;
}



void MatrixMulOnHost(float* M, float* N, float* P, int Width)
{
    for (int i = 0; i < Width; ++i)
        for (int j = 0; j < Width; ++j) {
            double sum = 0;
            for (int k = 0; k < Width; ++k) {
                double a = M[i * Width + k];
                double b = N[k * Width + j];
                sum += a * b;
            }
            P[i * Width + j] = sum;
        }
}


void runMatrixWithOutShared(float *h_A, float *h_B, unsigned int mem_size_A, unsigned int mem_size_B)
{
   /* cudaEvent_t start, stop;
    //printOutput(h_A,a);
    //printOutput(h_B,b);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);*/

    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,cudaMemcpyHostToDevice);
    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, HC / threads.y);
    
    //cudaEventRecord(start);
    // execute the kernel
    MatrixMulKernel<<< grid, threads >>>(d_A, d_B, d_C, WB);
    cudaError_t  code = cudaGetLastError();
    if (code != cudaSuccess) 
      printf ("Cuda error -- %s\n", cudaGetErrorString(code)); 
    //cudaEventRecord(stop);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,cudaMemcpyDeviceToHost);
    //char c='c';
    //printOutput(h_C,c);
    //float kernelRunTime = 0;
    //cudaEventElapsedTime(&kernelRunTime, start, stop);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //return kernelRunTime;
}

void runMatrixWithShared(float *h_A, float *h_B, unsigned int mem_size_A, unsigned int mem_size_B){

   /* cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);*/
    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_B, h_B, mem_size_B,cudaMemcpyHostToDevice);
    

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    //dim3 dimThreads(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(WC / dimThreads.x, HA / dimThreads.y);

    dim3 dimThreads(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(MATRIX_SIZE / TILE_WIDTH,MATRIX_SIZE / TILE_WIDTH);

    //cudaEventRecord(start);
    MatrixMulKernelTiled<<<dimGrid, dimThreads>>>(d_A, d_B, d_C,WB);
    //cudaEventRecord(stop);
    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,cudaMemcpyDeviceToHost);
    //cudaEventSynchronize(stop);

    //printOutput(h_C,c);

    //float kernelRunTime = 0;

    //cudaEventElapsedTime(&kernelRunTime, start, stop);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //return kernelRunTime;
}


int main()
{



	long long initTime;
	struct timeval initstime,initetime;

    //gettimeofday(&totalstime,0);

    cudaSetDevice(1);
    cudaDeviceProp  prop;
    cudaGetDeviceProperties( &prop, 1);
    printf( "Name:  %s\n\n", prop.name );

    printf("Matrix Size =%dX%d \n",MATRIX_SIZE,MATRIX_SIZE);
    printf("Tile Width = %dX%d\n",TILE_WIDTH,TILE_WIDTH);


    gettimeofday(&initstime,0);

    srand(2014);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);

    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    gettimeofday(&initetime,0);

    initTime = (initetime.tv_sec-initstime.tv_sec)*1000000LL + initetime.tv_usec-initstime.tv_usec;

    printf("Init Time : %lld\n",initTime );

/*----------------PARALLEL EXECUTION BEGINS HERE ----------------------------*/
    
    /*printf("Starting Without Shared Memory\n\n");
    runMatrixWithOutShared(h_A,h_B,mem_size_A,mem_size_B);
*/
/*------------------PARALLEL EXECUTION ENDS HERE ----------------------------*/    
    


/* -----------------TILING EXECUTION BEGINS HERE-----------------------------*/
    
   printf("Starting With Shared Memory\n\n");
    runMatrixWithShared(h_A,h_B,mem_size_A,mem_size_B);
    //printf("Mat Time With TILING: %f\n\n", matTimeWithTile);

/*------------------------------ TILING ENDS HERE ---------------------------------*/

/*
    gettimeofday(&stime,0);
    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);

    MatrixMulOnHost(h_A, h_B, reference, WB);
	gettimeofday(&etime,0);
    hosttime = (etime.tv_sec-stime.tv_sec)*1000000LL + etime.tv_usec-stime.tv_usec;

	//printf("host: %lld\ncuda: %lld\ncuda, w/copy: %lld\n", hosttime, cudatime, ctime);

    gettimeofday(&totaletime,0);
    totaltime = (totaletime.tv_sec-totalstime.tv_sec)*1000000LL + totaletime.tv_usec-totalstime.tv_usec;
    */

    // clean up memory
    free(h_A);
    free(h_B);
    //free(h_C);
    //free(reference);
    //cudaFree(d_A);
    //cudaFree(d_B);
    //cudaFree(d_C);

    //cudaThreadExit();


    printf("Total Time: %lld\n",totaltime );    
    printf("Initialization Time: %lld\n",inittime);
    printf("Copy Time: %lld\n",ctime-cudatime);
    printf("Parallel Time: %lld\n",cudatime);
    printf("Host Time: %lld\n", hosttime);
    printf("Tile Parallel Time: %lld\n",cudatiletime);
    printf("Sum %lld\n", inittime+ctime+hosttime+cudatiletime);
}


