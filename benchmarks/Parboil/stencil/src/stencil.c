/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/***************************************************************************
 *
 *  This benchmark was adapted to run on GPUs with OpenMP 4.0 pragmas
 *  and OpenCL driver implemented in gpuclang 2.0 (based on clang 3.5)
 *
 *  Marcio M Pereira <mpereira@ic.unicamp.br>
 *
 ***************************************************************************/

/*
 * === NOTE ===
 *
 * The Polyhedral optmizations restricts the class of loops it can manipulate
 * to sequences of imperfectly nested loops with particular constraints on the
 * loop bound and array subscript expressions.
 *
 * To allow this optimization we fixed the problem size with __STATIC__ tag
 * comment this tag to use original version.
 *
 */

#ifndef __STATIC__
  #define __STATIC__
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <inttypes.h>

#ifdef __APPLE__
  #include <machine/endian.h>
  #include <sys/malloc.h>
#else
  #include <endian.h>
  #include <malloc.h>
#endif

#include "../../common/parboil.h"
#include "../../common/polybenchUtilFuncts.h"

#if __BYTE_ORDER != __LITTLE_ENDIAN
  # error "File I/O is not implemented for this system: wrong endianness."
#endif

#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

//define the error threshold for the results "not matching"
#define ERROR_THRESHOLD 0.05

#define GPU 1

#ifdef __STATIC__
  // Define statically the problem size
  #define NX 512
  #define NY 512
  #define NZ 64
#else
  int NX, NY, NZ;
#endif

#define c0 0.1667
#define c1 0.0278

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

struct pb_Parameters *parameters;

double t_start, t_end, t_start_GPU, t_end_GPU;
float *h_Anext_GPU, *h_Anext_CPU;

void cpu_stencilGPU(float *A0, float *Anext) {
  int i, j, k;
  #pragma omp target device(GPU) \
                     map(to: A0[:NX*NY*NZ]) \
                     map(tofrom: Anext[:NX*NY*NZ])
  #pragma omp parallel for collapse(2)
  for(k=1;k<NZ-1;k++) {
    for(j=1;j<NY-1;j++) {
      for(i=1;i<NX-1;i++) {
	Anext[Index3D (NX, NY, i, j, k)] = 
	  (A0[Index3D (NX, NY, i, j, k + 1)] +
	   A0[Index3D (NX, NY, i, j, k - 1)] +
	   A0[Index3D (NX, NY, i, j + 1, k)] +
	   A0[Index3D (NX, NY, i, j - 1, k)] +
	   A0[Index3D (NX, NY, i + 1, j, k)] +
	   A0[Index3D (NX, NY, i - 1, j, k)])*c1
	  - A0[Index3D (NX, NY, i, j, k)]*c0;
      }
    }
  }
}

void cpu_stencilCPU(float *A0, float * Anext) {
  int i, j, k;
  for(k=1;k<NZ-1;k++) {
    for(j=1;j<NY-1;j++) {
      for(i=1;i<NX-1;i++) {
	Anext[Index3D (NX, NY, i, j, k)] = 
	  (A0[Index3D (NX, NY, i, j, k + 1)] +
	   A0[Index3D (NX, NY, i, j, k - 1)] +
	   A0[Index3D (NX, NY, i, j + 1, k)] +
	   A0[Index3D (NX, NY, i, j - 1, k)] +
	   A0[Index3D (NX, NY, i + 1, j, k)] +
	   A0[Index3D (NX, NY, i - 1, j, k)])*c1
	  - A0[Index3D (NX, NY, i, j, k)]*c0;
      }
    }
  }
}

void compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU) {
  int i, j, k, fail=0;

  for (k=0; k < NZ; k++) {
    for (j=0; j < NY; j++) {
      for (i=0; i < NX; i++) {
	if (percentDiff(A[Index3D (NX, NY, i, j, k)], A_GPU[Index3D (NX, NY, i, j, k)]) > ERROR_THRESHOLD) {
	  fail++;
	}
      }
    }	
  }	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) {	
  int s=0;
  int i, j, k;
  for(i=0;i<NZ;i++)
    {
      for(j=0;j<NY;j++)
	{
	  for(k=0;k<NX;k++)
	    {
	      fread(A0+s,sizeof(float),1,fp);
	      s++;
	    }
	}
    }
  return 0;
}

double stencilGPU(int argc, char** argv) {
  //declaration
  int nx,ny,nz;
  int size;
  int iteration;

  if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

  nx = atoi(argv[1]);
  if (nx<1)
    return -1;
  ny = atoi(argv[2]);
  if (ny<1)
    return -1;
  nz = atoi(argv[3]);
  if (nz<1)
    return -1;
  iteration = atoi(argv[4]);
  if(iteration<1)
    return -1;
	
  //host data
  float *h_A0;
  float *h_Anext;

  size=nx*ny*nz;
  
  h_A0=(float*)malloc(sizeof(float)*size);
  h_Anext=(float*)malloc(sizeof(float)*size);
  FILE *fp = fopen(parameters->inpFiles[0], "rb");
  read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
  memcpy (h_Anext,h_A0 ,sizeof(float)*size);

#ifndef __STATIC__
  NX = nx;
  NY = ny;
  NZ = nz;
#endif
  
  int t;
  t_start_GPU = rtclock();
  for(t=0;t<iteration;t++)
    {
      cpu_stencilGPU(h_A0, h_Anext);
      float *temp=h_A0;
      h_A0 = h_Anext;
      h_Anext = temp;
    }
  t_end_GPU = rtclock();
  float *temp=h_A0;
  h_A0 = h_Anext;
  h_Anext_GPU = temp;
		
  free (h_A0);
  return t_end_GPU - t_start_GPU;
}

double stencilCPU(int argc, char** argv) {	
  //declaration
  int nx,ny,nz;
  int size;
  int iteration;

  if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

  nx = atoi(argv[1]);
  if (nx<1)
    return -1;
  ny = atoi(argv[2]);
  if (ny<1)
    return -1;
  nz = atoi(argv[3]);
  if (nz<1)
    return -1;
  iteration = atoi(argv[4]);
  if(iteration<1)
    return -1;

  //host data
  float *h_A0;
  float *h_Anext;

  size=nx*ny*nz;

  h_A0=(float*)malloc(sizeof(float)*size);
  h_Anext=(float*)malloc(sizeof(float)*size);
  FILE *fp = fopen(parameters->inpFiles[0], "rb");
  read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
  memcpy (h_Anext,h_A0 ,sizeof(float)*size);

#ifndef __STATIC__
  NX = nx;
  NY = ny;
  NZ = nz;
#endif
  
  int t;
  t_start = rtclock();
  for(t=0;t<iteration;t++)
    {
      cpu_stencilCPU(h_A0, h_Anext);
      float *temp=h_A0;
      h_A0 = h_Anext;
      h_Anext = temp;
    }
  t_end = rtclock();
	
  float *temp=h_A0;
  h_A0 = h_Anext;
  h_Anext_CPU = temp;
		
  free (h_A0);
  return t_end - t_start;
}

int main(int argc, char** argv) {
  double t_GPU, t_CPU;

  parameters = pb_ReadParameters(&argc, argv);

  t_GPU = stencilGPU(argc, argv);
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_GPU);

  t_CPU = stencilCPU(argc, argv);
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_CPU);

  compareResults(h_Anext_GPU, h_Anext_CPU);

  pb_FreeParameters(parameters);
  free (h_Anext_GPU);
  free (h_Anext_CPU);

  return 0;
}
