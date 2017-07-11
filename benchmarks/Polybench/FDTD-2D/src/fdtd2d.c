/**
 * fdtd2d.c: This file was adapted from PolyBench/GPU 1.0 test suite
 * to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU 1

/* Problem size */
#define tmax 500
#define NX 2048
#define NY 2048

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
  int i, j;

  for (i = 0; i < tmax; i++)
    {
      _fict_[i] = (DATA_TYPE) i;
    }
  
  for (i = 0; i < NX; i++)
    {
      for (j = 0; j < NY; j++)
	{
	  ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
	  ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
	  hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
	}
    }
}

void init_array_hz(DATA_TYPE* hz)
{
  int i, j;
  
  for (i = 0; i < NX; i++)
    {
      for (j = 0; j < NY; j++)
	{
	  hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
	}
    }
}

void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
  int i, j, fail;
  fail = 0;
	
  for (i=0; i < NX; i++) 
    {
      for (j=0; j < NY; j++) 
	{
	  if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
    }
	
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
  int t, i, j;
	
  for (t=0; t < tmax; t++)  
    {
      for (j=0; j < NY; j++)
	{
	  ey[0*NY + j] = _fict_[t];
	}
      
      for (i = 1; i < NX; i++)
	{
	  for (j = 0; j < NY; j++)
	    {
	      ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
	    }
	}

      for (i = 0; i < NX; i++)
	{
	  for (j = 1; j < NY; j++)
	    {
	      ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
	    }
	}

      for (i = 0; i < NX; i++)
	{
	  for (j = 0; j < NY; j++)
	    {
	      hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
	    }
	}
    }
}

void runFdtd_OMP(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
  int t, i, j;

#pragma omp target device (GPU) map(to: _fict_[:tmax], ex[:(NX*(NY+1))], ey[:((NX+1)*NY)]) \
                                map(tofrom: hz[:(NX*(NY+1))])
  {
    for (t=0; t < tmax; t++)  
      {
        #pragma omp parallel for
	for (j=0; j < NY; j++)
	  {
	    ey[0*NY + j] = _fict_[t];
	  }
		
        #pragma omp parallel for collapse(2)
	for (i = 1; i < NX; i++)
	  {
	    for (j = 0; j < NY; j++)
	      {
		ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
	      }
	  }
	
        #pragma omp parallel for collapse(2)
	for (i = 0; i < NX; i++)
	  {
	    for (j = 1; j < NY; j++)
	      {
		ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
	      }
	  }
	
        #pragma omp parallel for collapse(2)
	for (i = 0; i < NX; i++)
	  {
	    for (j = 0; j < NY; j++)
	      {
		hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
	      }
	  }
      }
  }
}

int main()
{
  double t_start, t_end;

  DATA_TYPE* _fict_;
  DATA_TYPE* ex;
  DATA_TYPE* ey;
  DATA_TYPE* hz;
  DATA_TYPE* hz_outputFromGpu;

  _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
  ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
  ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
  hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
  hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

  fprintf(stdout, "<< 2-D Finite Different Time Domain Kernel >>\n");

  init_arrays(_fict_, ex, ey, hz);
  init_array_hz(hz_outputFromGpu);
  
  t_start = rtclock();
  runFdtd_OMP(_fict_, ex, ey, hz_outputFromGpu);
  t_end = rtclock();
	
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  t_start = rtclock();
  runFdtd(_fict_, ex, ey, hz);
  t_end = rtclock();
	
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
  compareResults(hz, hz_outputFromGpu);

  free(_fict_);
  free(ex);
  free(ey);
  free(hz);
  free(hz_outputFromGpu);

  return 0;
}

