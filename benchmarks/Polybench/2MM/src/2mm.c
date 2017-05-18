/**
 * 2mm.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU 
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *	     Luís Felipe Mattos <ra107822@students.ic.unicamp.br> 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

#define GPU 1

/* Problem size. */
# define NI 1024
# define NJ 1024
# define NK 1024
# define NL 1024

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C)
{
  int i, j;

  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NK; j++)
	{
	  A[i*NI + j] = ((DATA_TYPE) i*j) / NI;
	}
    }
  
  for (i = 0; i < NK; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  B[i*NK + j] = ((DATA_TYPE) i*(j+1)) / NJ;
	}
    }
  
  for (i = 0; i < NL; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  C[i*NL + j] = ((DATA_TYPE) i*(j+3)) / NL;
	}
    } 
}

void compareResults(DATA_TYPE *E, DATA_TYPE *E_GPU)
{
  int i,j,fail;
  fail = 0;

  for (i=0; i < NL; i++)
    {
      for (j=0; j < NI; j++)
	{
	  if (percentDiff(E[i*NI + j], E_GPU[i*NI + j]) > ERROR_THRESHOLD)
	    {
	      fail++;
	    }
	}
    }
	
  // print results
  printf(">>\n   Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f%s: %d\n", ERROR_THRESHOLD, "%", fail);
}

void mm2_cpu(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
  int i, j, k;

  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NJ; j++)
	{
	  D[i*NJ + j] = 0.0;
	  for (k = 0; k < NK; ++k)
	    {
	      D[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
	    }
	}
    }
  
  for (i = 0; i < NI; i++)
    {
      for (j = 0; j < NL; j++)
	{
	  E[i*NL + j] = 0.0;
	  for (k = 0; k < NJ; ++k)
	    {
	      E[i*NL + j] += C[i*NJ + k] * D[k*NL + j];
	    }
	}
    }
}

void mm2_OMP(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E)
{
  int i, j, k;

  #pragma omp target data device (GPU) map(from: D[:NI*NK])
  {
   #pragma omp target map(to: A[:NI*NK], B[:NK*NJ])
   #pragma omp parallel for simd collapse(2)
   for (i = 0; i < NI; i++)
     {
       for (j = 0; j < NJ; j++)
	 {
	   D[i*NJ + j] = 0.0;
	   for (k = 0; k < NK; ++k)
	     {
	       D[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
	     }
	 }
     }
	
    #pragma omp target map(to: C[:NI*NJ]) map(from: E[:NI*NL])
    #pragma omp parallel for simd collapse(2)
    for (i = 0; i < NI; i++)
      {
	for (j = 0; j < NL; j++)
	  {
	    E[i*NL + j] = 0.0;
	    for (k = 0; k < NJ; ++k)
	      {
		E[i*NL + j] += C[i*NJ + k] * D[k*NL + j];
	      }
	  }
      }
  }
}


int main(int argc, char** argv)
{
  double t_start, t_end, t_start_GPU, t_end_GPU;

  DATA_TYPE* C;
  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* D;
  DATA_TYPE* E;
  DATA_TYPE* E_GPU;

  C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
  A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
  D = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
  E = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
  E_GPU = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

  fprintf(stdout, "<< Linear Algebra: 2 Matrices Multiplications (D=A.B; E=C.D) >>\n");
  fprintf(stdout, "<< Size of Matrices: 1024 * 1024 >>\n");
  fprintf(stdout, "<< Computation: >>\n\n");
  fprintf(stdout, "   for (i = 0; i < 1024; i++) \n");
  fprintf(stdout, "       for (j = 0; j < 1024; j++) { \n");
  fprintf(stdout, "	      D[i][j] = 0.0; \n");
  fprintf(stdout, " 	      for (k = 0; k < 1024; ++k) \n");
  fprintf(stdout, "	          D[i][j] += A[i][k] * B[k][j]; \n");
  fprintf(stdout, "       } \n\n");
  fprintf(stdout, "   for (i = 0; i < 1024; i++) \n");
  fprintf(stdout, "       for (j = 0; j < 1024; j++) { \n");
  fprintf(stdout, " 	      E[i][j] = 0.0; \n");
  fprintf(stdout, "	      for (k = 0; k < 1024; ++k) \n");
  fprintf(stdout, "	          E[i][j] += C[i][k] * D[k][j]; \n");
  fprintf(stdout, "       } \n\n");

  fprintf(stdout, "<< Initializing Matrices A, B & C ... ");
  init_array(A, B, C);

  fprintf(stdout, ">>\n<< Start computation of matrices D & E on CPU...>>\n");
  t_start = rtclock();
  mm2_cpu(A, B, C, D, E);
  t_end = rtclock();
  fprintf(stdout, "   CPU Runtime: %0.6lfs\n", t_end  - t_start);

  fprintf(stdout, "<< Start computation of matrices D & E on GPU...>>\n");
  t_start_GPU = rtclock();
  mm2_OMP(A, B, C, D, E_GPU);
  t_end_GPU = rtclock();
  fprintf(stdout, "   GPU Runtime: %0.6lfs\n", t_end_GPU  - t_start_GPU);	

  fprintf(stdout, "<< Comparing Results ... ");
  compareResults(E, E_GPU);

  free(C);
  free(A);
  free(B);
  free(D);
  free(E);
  free(E_GPU);

  return 0;
}

