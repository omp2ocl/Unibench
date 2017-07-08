/**
 * gesummv.c: This file was adapted from PolyBench/GPU 1.0 test
 * suite to run on GPU with OpenMP 4.0 pragmas and OpenCL driver.
 *
 * http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Contacts: Marcio M Pereira <mpereira@ic.unicamp.br>
 *           Rafael Cardoso F Sousa <rafael.cardoso@students.ic.unicamp.br>
 *           Luís Felipe Mattos <ra107822@students.ic.unicamp.br>
 */

#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "../../common/polybenchUtilFuncts.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU 1

/* Problem size */
#define N 8192

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0)
 */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y) {

  for (int i = 0; i < N; i++) {
    DATA_TYPE tmp = 0;
    y[i] = 0;
    for (int j = 0; j < N; j++) {
      tmp = A[i * N + j] * x[j] + tmp;
      y[i] = B[i * N + j] * x[j] + y[i];
    }
    y[i] = ALPHA * tmp + BETA * y[i];
  }
}

void gesummv_OMP(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y) {

#pragma omp target device(GPU) map(to : A[:N * N], B[:N * N], x[:N])      \
                               map(from : y[:N])
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    DATA_TYPE tmp = 0;
    y[i] = 0;
    for (int j = 0; j < N; j++) {
      tmp = A[i * N + j] * x[j] + tmp;
      y[i] = B[i * N + j] * x[j] + y[i];
    }
    y[i] = ALPHA * tmp + BETA * y[i];
  }
}

void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x) {
  int i, j;

  for (i = 0; i < N; i++) {
    x[i] = ((DATA_TYPE)i) / N;

    for (j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)i * j) / N;
      B[i * N + j] = ((DATA_TYPE)i * j) / N;
    }
  }
}

void compareResults(DATA_TYPE *y, DATA_TYPE *y_outputFromGpu) {
  int i, fail;
  fail = 0;

  for (i = 0; i < (N); i++) {
    if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
      fail++;
    }
  }
  // Print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f "
         "Percent: %d\n",
         PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

int main(int argc, char *argv[]) {
  double t_start, t_end;

  DATA_TYPE *A;
  DATA_TYPE *B;
  DATA_TYPE *x;
  DATA_TYPE *y;
  DATA_TYPE *y_outputFromGpu;

  A = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  B = (DATA_TYPE *)malloc(N * N * sizeof(DATA_TYPE));
  x = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE *)malloc(N * sizeof(DATA_TYPE));

  fprintf(stdout, "<< Scalar, Vector and Matrix Multiplication >>\n");

  init(A, B, x);

  t_start = rtclock();
  gesummv_OMP(A, B, x, y_outputFromGpu);
  t_end = rtclock();
  fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

  init(A, B, x);

  t_start = rtclock();
  gesummv(A, B, x, y);
  t_end = rtclock();
  fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

  compareResults(y, y_outputFromGpu);

  free(A);
  free(B);
  free(x);
  free(y);
  free(y_outputFromGpu);

  return 0;
}
