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
 * sgemm.cc dense matrix-matrix multiplication
 */

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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>

#ifdef __APPLE__
  #include <sys/malloc.h>
#else
  #include <malloc.h>
#endif

#include "../../common/parboil.h"
#include "../../common/polybenchUtilFuncts.h"

#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1

#ifdef __STATIC__
  // Define statically the problem size
  #define NX 1024
  #define NY 1056
  #define NZ 992
#else
  int NX, NY;
#endif

#define alpha 1.0f
#define beta 0.0f

double t_start, t_end, t_start_GPU, t_end_GPU;
float *matC_GPU, *matC_CPU;

typedef float DATA_TYPE;

#ifdef __STATIC__
  void basicSgemmGPU( char transa, char transb, const float *A, const float *B, float *C )
#else
  void basicSgemmGPU( char transa, char transb, const float *A, const float *B, float *C, int NX, int NY, int NZ )
#endif
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  int mm, nn, i;
  
  #pragma omp target device(1)
  #pragma omp target map(to: A[:NX*NZ], B[:NY*NZ]) map(tofrom: C[:NX*NY])
  #pragma omp parallel for
  for (mm = 0; mm < NX; ++mm) {
    for (nn = 0; nn < NY; ++nn) {
      float c = 0.0f;
      for (i = 0; i < NZ; ++i) {
        float a = A[mm + i * NX]; 
        float b = B[nn + i * NY];
        c += a * b;
      }
      C[mm+nn*NX] = C[mm+nn*NX] * beta + alpha * c;
    }
  }
}

#ifdef __STATIC__
  void basicSgemmCPU(char transa, char transb, const float *A, const float *B, float *C)
#else
  void basicSgemmCPU(char transa, char transb, const float *A, const float *B, float *C, int NX, int NY, int NZ)
#endif
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  int mm, nn, i;

  for (mm = 0; mm < NX; ++mm) {
    for (nn = 0; nn < NY; ++nn) {
      float c = 0.0f;
      for (i = 0; i < NZ; ++i) {
        float a = A[mm + i * NX]; 
        float b = B[nn + i * NY];
        c += a * b;
      }
      C[mm+nn*NX] = C[mm+nn*NX] * beta + alpha * c;
    }
  }
}

void compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU)
{
  int i, j, fail=0;

  for (i=0; i < NX; i++)
    {
      for (j=0; j < NY; j++)
	{
	  if (percentDiff(A[i*NY + j], A_GPU[i*NY + j]) > ERROR_THRESHOLD) 
	    {
	      fail++;
	    }
	}
    }	
  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v) {
  std::fstream f(fn, std::fstream::in);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;

  float data;
  while (f.good() ) {
    f >> data;
    v.push_back(data);
  }
  v.pop_back(); // remove the duplicated last element

  return true;
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float>&v) {
  std::fstream f(fn, std::fstream::out);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f << nr_row << " "<<nr_col<<" ";

  float data;
  for (int i = 0; i < v.size(); ++i) {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;
}

double sgemmGPU(int argc, char *argv[]) {

  struct pb_Parameters *params;

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  // load A
  readColMajorMatrixFile(params->inpFiles[0], matArow, matAcol, matA);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2], matBcol, matBrow, matBT);

  // allocate space for C
  matC_GPU = (float *)calloc(matArow*matBcol, sizeof(float));

  //printf("matArow = %d, matAcol = %d, matBcol = %d\n", matArow, matAcol, matBcol);

  t_start_GPU = rtclock();
#ifdef __STATIC__
  basicSgemmGPU('N', 'T', &matA.front(), &matBT.front(), matC_GPU);
#else
  basicSgemmGPU('N', 'T', &matA.front(), &matBT.front(), matC_GPU, matArow, matBcol, matAcol);
#endif
  t_end_GPU = rtclock();

  // used by CompareResults
#ifndef __STATIC__
  NX = matArow;
  NY = matBcol;
#endif

  pb_FreeParameters(params);
  return t_end_GPU - t_start_GPU;
}

double sgemmCPU(int argc, char *argv[]) {

  struct pb_Parameters *params;

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  // load A
  readColMajorMatrixFile(params->inpFiles[0], matArow, matAcol, matA);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2], matBcol, matBrow, matBT);

  // allocate space for C
  matC_CPU = (float *)calloc(matArow*matBcol, sizeof(float));

  t_start = rtclock();
#ifdef __STATIC__
  basicSgemmCPU('N', 'T', &matA.front(), &matBT.front(), matC_CPU);
#else
  basicSgemmCPU('N', 'T', &matA.front(), &matBT.front(), matC_CPU, matArow, matBcol, matAcol);
#endif
  t_end = rtclock();

  pb_FreeParameters(params);
  return t_end - t_start;
}

int main (int argc, char *argv[]) {
  double t_GPU, t_CPU;

  t_GPU = sgemmGPU(argc, argv);
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_GPU);

  t_CPU = sgemmCPU(argc, argv);
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_CPU);

  compareResults(matC_GPU, matC_CPU);  
  return 0;
}
