/**
 * covariance.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU 
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br> 
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
#define PERCENT_DIFF_ERROR_THRESHOLD 2.0

#define GPU 1

/* Problem size */
#define M 2048
#define N 2048

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* data)
{
  int i, j;

  for (i = 1; i < (M+1); i++)
    {
      for (j = 1; j < (N+1); j++)
	{
	  data[i*(N+1) + j] = ((DATA_TYPE) i*j) / M;
	}
    }
}

void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
  int i,j,fail;
  fail = 0;

  for (i=1; i < (M+1); i++)
    {
      for (j=1; j < (N+1); j++)
	{
	  if (percentDiff(symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
	    {
	      fail++;
	    }			
	}
    }
  printf(">>\n   Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f%s: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, "%", fail);
}

void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
  int i, j, j1,j2;

  /* Determine mean of column vectors of input data matrix */
  for (j = 1; j < (M+1); j++)
    {
      mean[j] = 0.0;
      for (i = 1; i < (N+1); i++)
	{
	  mean[j] += data[i*(M+1) + j];
	}
      mean[j] /= FLOAT_N;
    }
  
  /* Center the column vectors. */
  for (i = 1; i < (N+1); i++)
    {
      for (j = 1; j < (M+1); j++)
	{
	  data[i*(M+1) + j] -= mean[j];
	}
    }

  /* Calculate the m * m covariance matrix. */
  for (j1 = 1; j1 < (M+1); j1++)
    {
      for (j2 = j1; j2 < (M+1); j2++)
	{
	  symmat[j1*(M+1) + j2] = 0.0;
	  for (i = 1; i < N+1; i++)
	    {
	      symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
	    }
	  symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
	}
    }
}

void covariance_OMP(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
  int i, j, j1,j2;

  /* Determine mean of column vectors of input data matrix */
	
  #pragma omp target device (GPU) map(to: data[:(M+1)*(N+1)]) map(from: mean[:(M+1)])
  #pragma omp parallel for
  for (j = 1; j < (M+1); j++)
    {
      mean[j] = 0.0;
      for (i = 1; i < (N+1); i++)
	{
	  mean[j] += data[i*(M+1) + j];
	}
      mean[j] /= FLOAT_N;
    }
  
  /* Center the column vectors. */
  #pragma omp target map(to: mean[:(M+1)]) map(tofrom: data[:(M+1)*(N+1)])
  #pragma omp parallel for collapse(2)
  for (i = 1; i < (N+1); i++)
    {
      for (j = 1; j < (M+1); j++)
	{
	  data[i*(M+1) + j] -= mean[j];
	}
    }
  
  /* Calculate the m * m covariance matrix. */
  #pragma omp target map(to: data[:(M+1)*(M+1)]) map(from: symmat[:(M+1)*(N+1)])
  #pragma omp parallel for schedule(dynamic,8)
  for (j1 = 1; j1 < (M+1); j1++)
    {
      for (j2 = j1; j2 < (M+1); j2++)
	{
	  symmat[j1*(M+1) + j2] = 0.0;
	  for (i = 1; i < N+1; i++)
	    {
	      symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
	    }
	  symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
	}
    }
}

int main()
{
  double t_start, t_end;

  DATA_TYPE* data;
  DATA_TYPE* symmat;
  DATA_TYPE* mean;
  DATA_TYPE* symmat_outputFromGpu;	

  data = (DATA_TYPE*)malloc((M+1)*(N+1)*sizeof(DATA_TYPE));
  symmat = (DATA_TYPE*)malloc((M+1)*(M+1)*sizeof(DATA_TYPE));
  mean = (DATA_TYPE*)malloc((M+1)*sizeof(DATA_TYPE));
  symmat_outputFromGpu = (DATA_TYPE*)malloc((M+1)*(M+1)*sizeof(DATA_TYPE));	

  fprintf(stdout, "<< Covariance Computation >>\n");
  fprintf(stdout, "<< Size of data Matrix: 2048 * 2048 >>\n\n");
  fprintf(stdout, "<< Determine mean of column vectors of input data matrix: >>\n");
  fprintf(stdout, "   for (j = 1; j <= 2048; j++) {\n");
  fprintf(stdout, "       mean[j] = 0.0; \n");
  fprintf(stdout, "       for (i = 1; i <= 2048; i++) \n");
  fprintf(stdout, "	      mean[j] += data[i][j]; \n");
  fprintf(stdout, "       mean[j] /= 3214212.01; \n");
  fprintf(stdout, "   } \n");
  fprintf(stdout, "<< Center the column vectors: >> \n");
  fprintf(stdout, "   for (i = 1; i <= 2048; ++i) \n");
  fprintf(stdout, "       for (j = 1; j <= 2048; ++j) \n");
  fprintf(stdout, "           data[i][j] -= mean[j];\n");  
  fprintf(stdout, "<< Calculate de 2048*2048 covariance matrix: >>\n");
  fprintf(stdout, "   for (j1 = 1; j1 <= 2048; j1++) \n");
  fprintf(stdout, "       for (j2 = j1; j2 <= 2048; j2++) {\n");
  fprintf(stdout, "   	      symmat[j1][j2] = 0.0; \n");
  fprintf(stdout, "   	      for (i = 1; i <= 2048; i++) \n");
  fprintf(stdout, "               symmat[j1][j2] += data[i][j1] * data[i][j2];\n");
  fprintf(stdout, "           symmat[j2][j1] = symmat[j1][j2];\n");
  fprintf(stdout, "       } \n\n");

  fprintf(stdout, "<< Initializing data ... ");
  init_arrays(data);

  fprintf(stdout, ">>\n<< Start covariance computation on GPU... ");
  t_start = rtclock();
  covariance_OMP(data, symmat_outputFromGpu, mean);
  t_end = rtclock();
  fprintf(stdout, ">>\n   GPU Runtime: %0.6lfs\n", t_end - t_start);	

  init_arrays(data);

  fprintf(stdout, "\n<< Start covariance computation on CPU... ");
  t_start = rtclock();
  covariance(data, symmat, mean);
  t_end = rtclock();
  fprintf(stdout, ">>\n   CPU Runtime: %0.6lfs\n", t_end - t_start);    

  fprintf(stdout, "<< Comparing Results...");
  compareResults(symmat, symmat_outputFromGpu);

  free(data);
  free(symmat);
  free(mean);
  free(symmat_outputFromGpu);	
  
  return 0;
}

