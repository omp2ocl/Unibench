/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
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
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
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
 * Recommended gpuclang options:
 *  -O2 -lm -ffast-math -opt-poly-all
 */

#ifndef __STATIC__
  #define __STATIC__
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/time.h>

#ifdef __APPLE__
  #include <sys/malloc.h>
  #include <machine/endian.h>
#else
  #include <endian.h>
  #include <malloc.h>
#endif

#if __BYTE_ORDER != __LITTLE_ENDIAN
  # error "File I/O is not implemented for this system: wrong endianness."
#endif

#include "../../common/parboil.h"
#include "../../common/polybenchUtilFuncts.h"

#define ERROR_THRESHOLD 0.5
#define GPU_DEVICE 1

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

#ifdef __STATIC__
  // Define statically the problem size
  #define NK 2048     // K_ELEMS_PER_GRID
  #define NX 262144
#else
  int NK, NX;
#endif

double t_start, t_end, t_start_GPU, t_end_GPU;

float *Qr_GPU, *Qi_GPU;		/* Q signal (complex) */
float *Qr_CPU, *Qi_CPU;		/* Q signal (complex) */

typedef float DATA_TYPE;

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

void ComputePhiMagCPU(float* phiR, float* phiI, float* phiMag)
{
  int indexK = 0;
  for (indexK = 0; indexK < NK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

void ComputeQGPU(struct kValues *kVals,
		  float* x, float* y, float* z,
		  float *Qr, float *Qi)
{  
  int indexK, indexX;
  #pragma omp target device(1)
  #pragma omp target map(to: kVals[:NK], x[:NX], y[:NX], z[:NX]) map(tofrom: Qr[:NX], Qi[:NX])
  for (indexK = 0; indexK < NK; indexK++) {
    #pragma omp parallel for
    for (indexX = 0; indexX < NX; indexX++) {
      float expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                       kVals[indexK].Ky * y[indexX] +
                       kVals[indexK].Kz * z[indexX]);

      float cosArg = cos(expArg);
      float sinArg = sin(expArg);

      float phi = kVals[indexK].PhiMag;
      Qr[indexX] += phi * cosArg;
      Qi[indexX] += phi * sinArg;
    }
  }
}

void ComputeQCPU(struct kValues *kVals,
		  float* x, float* y, float* z,
		  float *Qr, float *Qi)
{   
  int indexK, indexX;
  for (indexK = 0; indexK < NK; indexK++) {
    for (indexX = 0; indexX < NX; indexX++) {
      float expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                       kVals[indexK].Ky * y[indexX] +
                       kVals[indexK].Kz * z[indexX]);

      float cosArg = cos(expArg);
      float sinArg = sin(expArg);

      float phi = kVals[indexK].PhiMag;
      Qr[indexX] += phi * cosArg;
      Qi[indexX] += phi * sinArg;
    }
  }
}

void createDataStructsCPU(float** phiMag, float** Qr, float** Qi)
{
  *phiMag = (float* ) malloc(NK * sizeof(float));
  *Qr = (float*) malloc(NX * sizeof (float));
  memset((void *)*Qr, 0, NX * sizeof(float));
  *Qi = (float*) malloc(NX * sizeof (float));
  memset((void *)*Qi, 0, NX * sizeof(float));
}

void inputData(char* fName, int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI)
{
  int numK, numX;
  FILE* fid = fopen(fName, "r");

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }
  fread (&numK, sizeof (int), 1, fid);
  *_numK = numK;
  fread (&numX, sizeof (int), 1, fid);
  *_numX = numX;
  *kx = (float *) malloc(numK * sizeof (float));
  fread (*kx, sizeof (float), numK, fid);
  *ky = (float *) malloc(numK * sizeof (float));
  fread (*ky, sizeof (float), numK, fid);
  *kz = (float *) malloc(numK * sizeof (float));
  fread (*kz, sizeof (float), numK, fid);
  *x = (float *) malloc(numX * sizeof (float));
  fread (*x, sizeof (float), numX, fid);
  *y = (float *) malloc(numX * sizeof (float));
  fread (*y, sizeof (float), numX, fid);
  *z = (float *) malloc(numX * sizeof (float));
  fread (*z, sizeof (float), numX, fid);
  *phiR = (float *) malloc(numK * sizeof (float));
  fread (*phiR, sizeof (float), numK, fid);
  *phiI = (float *) malloc(numK * sizeof (float));
  fread (*phiI, sizeof (float), numK, fid);
  fclose (fid); 
}

void compareResults(DATA_TYPE *A, DATA_TYPE *A_GPU, DATA_TYPE *B, DATA_TYPE *B_GPU)
{
  int i,fail=0;

  for (i=0; i < NX; i++)
    {
      if (percentDiff(A[i], A_GPU[i]) > ERROR_THRESHOLD) 
	{
	  fail++;
	}
    }

  for (i=0; i < NX; i++)
    {
      if (percentDiff(B[i], B_GPU[i]) > ERROR_THRESHOLD) 
	{
	  fail++;
	}
    }
	
  // print results
  printf(">>\n   Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f%s: %d\n", ERROR_THRESHOLD, "%", fail);
}

double mriqGPU(int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  struct kValues* kVals;

  struct pb_Parameters *params;

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
  
  /* Read in data */
  fprintf(stdout, "<< Reading data ... ");
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

#ifndef __STATIC__
  NK = numK;
  NX = numX;
#endif
  
  /* Create CPU data structures */
  createDataStructsCPU(&phiMag, &Qr_GPU, &Qi_GPU);
  ComputePhiMagCPU(phiR, phiI, phiMag);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  int k;
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  fprintf(stdout, ">>\n<< Start computation on GPU... ");
  t_start_GPU = rtclock();
  ComputeQGPU(kVals, x, y, z, Qr_GPU, Qi_GPU);
  t_end_GPU = rtclock();

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (phiMag);
  free (kVals);

  return t_end_GPU - t_start_GPU;
}

double mriqCPU(int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  struct kValues* kVals;

  struct pb_Parameters *params;

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
    {
      fprintf(stderr, "Expecting one input filename\n");
      exit(-1);
    }
  
  /* Read in data */
  inputData(params->inpFiles[0],
	    &original_numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
	{
	  fprintf(stderr, "Expecting an integer parameter\n");
	  exit(-1);
	}

      numK = MIN(inputK, original_numK);
    }

#ifndef __STATIC__
  NK = numK;
  NX = numX;
#endif

  /* Create CPU data structures */
  createDataStructsCPU(&phiMag, &Qr_CPU, &Qi_CPU);
  ComputePhiMagCPU(phiR, phiI, phiMag);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  int k;
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  fprintf(stdout, "\n<< Start computation on CPU... ");
  t_start = rtclock();
  ComputeQCPU(kVals, x, y, z, Qr_CPU, Qi_CPU);
  t_end = rtclock();

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (phiMag);
  free (kVals);

  return t_end - t_start;
}

int main (int argc, char *argv[]) {
  double t_GPU, t_CPU;

  fprintf(stdout, "<< Creating the Q data structure for fast convolution-based\n");
  fprintf(stdout, "   Hessian multiplication for arbitrary k-space trajectories.>>\n");
  fprintf(stdout, "<< Elements per Grid: 2048 >>\n\n");
  fprintf(stdout, "   for (indexK = 0; indexK < 2048; indexK++) \n");
  fprintf(stdout, "       for (indexX = 0; indexX < 262144; indexX++) { \n");
  fprintf(stdout, "           float expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +\n");
  fprintf(stdout, "                          kVals[indexK].Ky * y[indexX] + \n");
  fprintf(stdout, "                          kVals[indexK].Kz * z[indexX]);\n");
  fprintf(stdout, "           float cosArg = cos(expArg);\n");
  fprintf(stdout, "           float sinArg = sin(expArg);\n");
  fprintf(stdout, "           float phi = kVals[indexK].PhiMag;\n");
  fprintf(stdout, "           Qr[indexX] += phi * cosArg;\n");
  fprintf(stdout, "           Qi[indexX] += phi * sinArg;\n");
  fprintf(stdout, "       } \n\n");

  t_GPU = mriqGPU(argc, argv);
  fprintf(stdout, ">>\n   GPU Runtime: %0.6lfs\n", t_GPU);

  t_CPU = mriqCPU(argc, argv);
  fprintf(stdout, ">>\n   CPU Runtime: %0.6lfs\n", t_CPU);

  fprintf(stdout, "\n<< Comparing Results...");
  compareResults(Qr_CPU, Qr_GPU, Qi_CPU, Qi_GPU);

  free(Qr_GPU);
  free(Qi_GPU);
  free(Qr_CPU);
  free(Qi_CPU);

  return 0;
}
