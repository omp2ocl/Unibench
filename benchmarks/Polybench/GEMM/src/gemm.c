/**
 * gemm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU 
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br> 
*/

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

#define GPU 1

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 1024
#define NJ 1024
#define NK 1024

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  int i,j,k;
	
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  C[i*NJ + j] *= BETA;
	  
	  for (k = 0; k < NK; ++k)
	    {
	      C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
	    }
	}
    }
}

void gemm_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
  int i,j,k;
	
  #pragma omp target device(GPU) map(to: A[:NI*NK], B[:NK*NJ]) map(tofrom: C[:NI*NJ])
  #pragma omp parallel for collapse(2)
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  C[i*NJ + j] *= BETA;
	
	  for (k = 0; k < NK; ++k)
	    {
	      C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
	    }
	}
    }
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *C_OMP)
{
  int i, j;

  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NK; j++)
	{
	  A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
	}
    }

  for (i = 0; i < NK; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
	}
    }

  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
	  C_OMP[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
	}
    }
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
  int i, j, fail;
  fail = 0;
	
  // Compare C1 and C2
  for (i=0; i < NI; i++) 
    {
      for (j=0; j < NJ; j++) 
	{
	  if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
    }
  
  // Print results
  printf(">>\n   Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f%s: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, "%", fail);
}

int main(int argc, char *argv[])
{
  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* B;  
  DATA_TYPE* C;  
  DATA_TYPE* C_outputFromGpu; 

  A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
  B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
  C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
  C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

  fprintf(stdout, "<< Matrix-multiply C=alpha.A.B+beta.C >>\n");
  fprintf(stdout, "<< Size of Matrices: 1024 * 1024 >>\n");
  fprintf(stdout, "<< Computation: >>\n\n");

  fprintf(stdout, "   for (i = 0; i < 1024; i++) \n");
  fprintf(stdout, "       for (j = 0; j < 1024; j++) {\n");
  fprintf(stdout, "   	      C[i][j] *= BETA; \n");
  fprintf(stdout, "   	      for (k = 0; k < 1024; ++k) \n");
  fprintf(stdout, "   	          C[i][j] += ALPHA * A[i][k] * B[k][j];\n");
  fprintf(stdout, "       }\n\n");
  
  fprintf(stdout, "<< Initializing Matrices A, B & C ... ");
  init(A, B, C, C_outputFromGpu);
  
  fprintf(stdout, ">>\n<< Start computation of matrix C on GPU... ");
  t_start = rtclock();	
  gemm_OMP(A, B, C_outputFromGpu);
  t_end = rtclock();
  fprintf(stdout, ">>\n   GPU Runtime: %0.6lfs\n", t_end - t_start);

  fprintf(stdout, "\n<< Start computation of matrix C on CPU... ");
  t_start = rtclock();	
  gemm(A, B, C);
  t_end = rtclock();
  fprintf(stdout, ">>\n   CPU Runtime: %0.6lfs\n", t_end - t_start);
	
  fprintf(stdout, "<< Comparing Results ... ");
  compareResults(C, C_outputFromGpu);

  free(A);
  free(B);  
  free(C);  
  free(C_outputFromGpu); 

  return 0;
}

