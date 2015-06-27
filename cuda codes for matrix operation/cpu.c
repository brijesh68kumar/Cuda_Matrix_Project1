#include"stdio.h"
#include"time.h"
#include"math.h"
#include"stdlib.h"
#define S 1024
#define X 1024
#define Y 1024

    int main()
    {
        int sum = 0,i,j,k;
        int **A; //declared A[X][S]
        int **B; //declare B[S][Y]
        int **C; //resultant matrix become C[X][Y]
        struct timespec start,stop;
        double t1=0,t2=0,result=0;
  
  /*----------------------------------------------------*/
 //create array of pointers(Rows)
         A =(int **) malloc(X*sizeof(int*));
         B =(int **)malloc(S*sizeof(int*));
         C=(int **)malloc(X*sizeof(int*));
  /*----------------------------------------------------*/
  
  /*--------------------------------------------------------------------------------*/
 //allocate memory for each Row pointer
         for(i=0; i < X; i++)
        {
            A[i]=(int *)malloc(S*sizeof(int));
            C[i]=(int *)malloc(Y*sizeof(int));
        }
  
        for(i = 0; i < S; i++)
        B[i]=(int *)malloc(Y*sizeof(int));
  /*--------------------------------------------------------------------------------*/
  
        for(i = 0; i < X; i++)
        {
         for(j = 0; j < S; j++)
        {
          A[i][j] = 1; 
        }
        }
 
         for(i = 0; i < S; i++)
     {
          for(j = 0; j < Y; j++)
    {
        B[i][j] = 2;
    }
    }
 //------------------calculate Starting time----------------------
        clock_gettime(CLOCK_REALTIME,&start);
        t1 = start.tv_sec + (start.tv_nsec/pow(10,9));
 //--------------------------------------------------------------
 
         for(i = 0; i < X; i++)
    {
        for(j = 0; j < Y;j++)
    {
        sum=0;
        for(k = 0; k < S; k++)
    {
         sum = sum + A[i][k] * B[k][j];
    }
        C[i][j] = sum;
   }
   }
 //---------------calculate End time-------------------------
         clock_gettime(CLOCK_REALTIME,&stop);
        t2 = stop.tv_sec + (stop.tv_nsec/pow(10,9));
 //--------------------------------------------------------------
 // result = End_time - Start_time

        result = t2 - t1;
        for(i = 0; i < 2; i++)
    {
         for(j = 0; j < 10; j++)
    {
        printf("%d\t",C[i][j]);
    }
        printf("\n");
    }
        printf("\ntime taken =\t %lf\n",result);
        return 0;
    }
