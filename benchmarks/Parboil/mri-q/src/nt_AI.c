/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 *  MRI-Q: Magnetic Resonance Imaging
 *         Computes a matrix Q, representing the scanner configuration for
 *         calibration, used in a 3D magnetic resonance image reconstruction
 *         algorithms in non-Cartesian space.
 *
 ***************************************************************************/

/***************************************************************************
 *
 *  This benchmark was adapted to run on GPUs with OpenMP 4.0 pragmas
 *  and OpenCL driver implemented in gpuclang 2.1 (based on clang 3.5)
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
 * The Polyhedral optmization used in gpuclang restricts the class of loops it
 * can manipulate to sequences of imperfectly nested loops with particular
 * constraints on the loop bound and array subscript expressions.
 *
 * To allow this optimization we fixed the problem size with __STATIC__ tag
 * Comment this tag to use the original version.
 *
 * Recommended gpuclang options:
 *  -O3 -lm -ffast-math -opt-poly=tile
 */

#ifndef __STATIC__
#define __STATIC__
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <sys/malloc.h>
#include <machine/endian.h>
#else
#include <endian.h>
#include <malloc.h>
#endif

#if _POSIX_VERSION >= 200112L
#include <sys/time.h>
#endif

#if __BYTE_ORDER != __LITTLE_ENDIAN
#error "File I/O is not implemented for this system: wrong endianness."
#endif

#define SMALL_FLOAT_VAL 0.00000001f
#define ERROR_THRESHOLD 0.5
#define GPU_DEVICE 1

#define PI 3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

#ifdef __STATIC__
// Define statically the problem size
#define NK 2048 // K_ELEMS_PER_GRID
#define NX 262144
#else
int NK, NX;
#endif

double t_start, t_end, t_start_GPU, t_end_GPU;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

float absVal(float a) {
  if (a < 0)
    return (a * -1);
  else
    return a;
}

float percentDiff(double val1, double val2) {
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
    return 0.0f;
  else
    return 100.0f *
           (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
}

/* Command line parameters for benchmark */
struct pb_Parameters {
  char *outFile;   /* If not NULL, the raw output of the
                    * computation should be saved to this
                    * file. The string is owned. */
  char **inpFiles; /* A NULL-terminated array of strings
                    * holding the input file(s) for the
                    * computation.  The array and strings
                    * are owned. */
};

/* A time or duration. */
#if _POSIX_VERSION >= 200112L
typedef unsigned long long pb_Timestamp; /* time in microseconds */
#else
#error "Timestamps not implemented"
#endif

enum pb_TimerState {
  pb_Timer_STOPPED,
  pb_Timer_RUNNING,
};

struct pb_Timer {
  enum pb_TimerState state;
  pb_Timestamp elapsed; /* Amount of time elapsed so far */
  pb_Timestamp init;    /* Beginning of the current time interval,
                         * if state is RUNNING.  End of the last
                         * recorded time interfal otherwise.  */
};

/* Execution time is assigned to one of these categories. */
enum pb_TimerID {
  pb_TimerID_NONE = 0,
  pb_TimerID_IO,         /* Time spent in input/output */
  pb_TimerID_KERNEL,     /* Time spent computing on the device,
                          * recorded asynchronously */
  pb_TimerID_COPY,       /* Time spent synchronously moving data
                          * to/from device and allocating/freeing
                          * memory on the device */
  pb_TimerID_DRIVER,     /* Time spent in the host interacting with the
                          * driver, primarily for recording the time
                          * spent queueing asynchronous operations */
  pb_TimerID_COPY_ASYNC, /* Time spent in asynchronous transfers */
  pb_TimerID_COMPUTE,    /* Time for all program execution other
                          * than parsing command line arguments,
                          * I/O, kernel, and copy */
  pb_TimerID_OVERLAP,    /* Time double-counted in asynchronous and
                          * host activity: automatically filled in,
                          * not intended for direct usage */
  pb_TimerID_LAST        /* Number of timer IDs */
};

/* Dynamic list of asynchronously tracked times between events */
struct pb_async_time_marker_list {
  char *label;             // actually just a pointer to a string
  enum pb_TimerID timerID; /* The ID to which the interval beginning
                            * with this marker should be attributed */
  void *marker;
  // cudaEvent_t marker; 		/* The driver event for this marker */
  struct pb_async_time_marker_list *next;
};

struct pb_SubTimer {
  char *label;
  struct pb_Timer timer;
  struct pb_SubTimer *next;
};

struct pb_SubTimerList {
  struct pb_SubTimer *current;
  struct pb_SubTimer *subtimer_list;
};

/* A set of timers for recording execution times. */
struct pb_TimerSet {
  enum pb_TimerID current;
  struct pb_async_time_marker_list *async_markers;
  pb_Timestamp async_begin;
  pb_Timestamp wall_begin;
  struct pb_Timer timers[pb_TimerID_LAST];
  struct pb_SubTimerList *sub_timer_list[pb_TimerID_LAST];
};

/* Free an array of owned strings. */
static void free_string_array(char **string_array) {
  char **p;

  if (!string_array)
    return;
  for (p = string_array; *p; p++)
    free(*p);
  free(string_array);
}

/* Parse a comma-delimited list of strings into an
 * array of strings. */
static char **read_string_array(char *in) {
  char **ret;
  int i;
  int count;       /* Number of items in the input */
  char *substring; /* Current substring within 'in' */

  /* Count the number of items in the string */
  count = 1;
  for (i = 0; in[i]; i++)
    if (in[i] == ',')
      count++;

  /* Allocate storage */
  ret = (char **)malloc((count + 1) * sizeof(char *));

  /* Create copies of the strings from the list */
  substring = in;
  for (i = 0; i < count; i++) {
    char *substring_end;
    int substring_length;

    /* Find length of substring */
    for (substring_end = substring;
         (*substring_end != ',') && (*substring_end != 0); substring_end++)
      ;

    substring_length = substring_end - substring;

    /* Allocate memory and copy the substring */
    ret[i] = (char *)malloc(substring_length + 1);
    memcpy(ret[i], substring, substring_length);
    ret[i][substring_length] = 0;

    /* go to next substring */
    substring = substring_end + 1;
  }
  ret[i] = NULL; /* Write the sentinel value */

  return ret;
}

struct argparse {
  int argc;    /* Number of arguments.  Mutable. */
  char **argv; /* Argument values.  Immutable. */

  int argn;        /* Current argument number. */
  char **argv_get; /* Argument value being read. */
  char **argv_put; /* Argument value being written.
                    * argv_put <= argv_get. */
};

static void initialize_argparse(struct argparse *ap, int argc, char **argv) {
  ap->argc = argc;
  ap->argn = 0;
  ap->argv_get = ap->argv_put = ap->argv = argv;
}

static void finalize_argparse(struct argparse *ap) {
  /* Move the remaining arguments */
  for (; ap->argn < ap->argc; ap->argn++)
    *ap->argv_put++ = *ap->argv_get++;
}

/* Delete the current argument. */
static void delete_argument(struct argparse *ap) {
  if (ap->argn >= ap->argc) {
    fprintf(stderr, "delete_argument\n");
  }
  ap->argc--;
  ap->argv_get++;
}

/* Go to the next argument.  Also, move the current argument to its
 * final location in argv. */
static void next_argument(struct argparse *ap) {
  if (ap->argn >= ap->argc) {
    fprintf(stderr, "next_argument\n");
  }
  /* Move argument to its new location. */
  *ap->argv_put++ = *ap->argv_get++;
  ap->argn++;
}

static int is_end_of_arguments(struct argparse *ap) {
  return ap->argn == ap->argc;
}

static char *get_argument(struct argparse *ap) { return *ap->argv_get; }

static char *consume_argument(struct argparse *ap) {
  char *ret = get_argument(ap);
  delete_argument(ap);
  return ret;
}

void pb_FreeParameters(struct pb_Parameters *p) {
  char **cpp;

  free(p->outFile);
  free_string_array(p->inpFiles);
  free(p);
}

struct pb_Parameters *pb_ReadParameters(int *_argc, char **argv) {
  char *err_message;
  struct argparse ap;
  struct pb_Parameters *ret =
      (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));

  /* Initialize the parameters structure */
  ret->outFile = NULL;
  ret->inpFiles = (char **)malloc(sizeof(char *));
  ret->inpFiles[0] = NULL;

  /* Each argument */
  initialize_argparse(&ap, *_argc, argv);
  while (!is_end_of_arguments(&ap)) {
    char *arg = get_argument(&ap);

    /* Single-character flag */
    if ((arg[0] == '-') && (arg[1] != 0) && (arg[2] == 0)) {
      delete_argument(&ap); /* This argument is consumed here */

      switch (arg[1]) {
      case 'o': /* Output file name */
        if (is_end_of_arguments(&ap)) {
          err_message = "Expecting file name after '-o'\n";
          goto error;
        }
        free(ret->outFile);
        ret->outFile = strdup(consume_argument(&ap));
        break;
      case 'i': /* Input file name */
        if (is_end_of_arguments(&ap)) {
          err_message = "Expecting file name after '-i'\n";
          goto error;
        }
        ret->inpFiles = read_string_array(consume_argument(&ap));
        break;
      case '-': /* End of options */
        goto end_of_options;
      default:
        err_message = "Unexpected command-line parameter\n";
        goto error;
      }
    } else {
      /* Other parameters are ignored */
      next_argument(&ap);
    }
  } /* end for each argument */

end_of_options:
  *_argc = ap.argc; /* Save the modified argc value */
  finalize_argparse(&ap);

  return ret;

error:
  fputs(err_message, stderr);
  pb_FreeParameters(ret);
  return NULL;
}

int pb_Parameters_CountInputs(struct pb_Parameters *p) {
  int n;

  for (n = 0; p->inpFiles[n]; n++)
    ;
  return n;
}

/*****************************************************************************/
/* Timer routines */

static void accumulate_time(pb_Timestamp *accum, pb_Timestamp start,
                            pb_Timestamp end) {
#if _POSIX_VERSION >= 200112L
  *accum += end - start;
#else
#error "Timestamps not implemented for this system"
#endif
}

#if _POSIX_VERSION >= 200112L
static pb_Timestamp get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (pb_Timestamp)(tv.tv_sec * 1000000LL + tv.tv_usec);
}
#else
#error "no supported time libraries are available on this platform"
#endif

void pb_ResetTimer(struct pb_Timer *timer) {
  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  timer->elapsed = 0;
#else
#error "pb_ResetTimer: not implemented for this system"
#endif
}

void pb_StartTimer(struct pb_Timer *timer) {
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Ignoring attempt to start a running timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
#error "pb_StartTimer: not implemented for this system"
#endif
}

void pb_StartTimerAndSubTimer(struct pb_Timer *timer,
                              struct pb_Timer *subtimer) {
  unsigned int numNotStopped = 0x3; // 11
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Warning: Timer was not stopped\n", stderr);
    numNotStopped &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_STOPPED) {
    fputs("Warning: Subtimer was not stopped\n", stderr);
    numNotStopped &= 0x2; // Zero out 2^0
  }
  if (numNotStopped == 0x0) {
    fputs("Ignoring attempt to start running timer and subtimer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;
  subtimer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    if (numNotStopped & 0x2) {
      timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }

    if (numNotStopped & 0x1) {
      subtimer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }
  }
#else
#error "pb_StartTimer: not implemented for this system"
#endif
}

void pb_StopTimer(struct pb_Timer *timer) {

  pb_Timestamp fini;

  if (timer->state != pb_Timer_RUNNING) {
    fputs("Ignoring attempt to stop a stopped timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
#error "pb_StopTimer: not implemented for this system"
#endif

  accumulate_time(&timer->elapsed, timer->init, fini);
  timer->init = fini;
}

void pb_StopTimerAndSubTimer(struct pb_Timer *timer,
                             struct pb_Timer *subtimer) {

  pb_Timestamp fini;

  unsigned int numNotRunning = 0x3; // 0b11
  if (timer->state != pb_Timer_RUNNING) {
    fputs("Warning: Timer was not running\n", stderr);
    numNotRunning &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_RUNNING) {
    fputs("Warning: Subtimer was not running\n", stderr);
    numNotRunning &= 0x2; // Zero out 2^0
  }
  if (numNotRunning == 0x0) {
    fputs("Ignoring attempt to stop stopped timer and subtimer\n", stderr);
    return;
  }

  timer->state = pb_Timer_STOPPED;
  subtimer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
#error "pb_StopTimer: not implemented for this system"
#endif

  if (numNotRunning & 0x2) {
    accumulate_time(&timer->elapsed, timer->init, fini);
    timer->init = fini;
  }

  if (numNotRunning & 0x1) {
    accumulate_time(&subtimer->elapsed, subtimer->init, fini);
    subtimer->init = fini;
  }
}

/* Get the elapsed time in seconds. */
double pb_GetElapsedTime(struct pb_Timer *timer) {
  double ret;

  if (timer->state != pb_Timer_STOPPED) {
    fputs("Elapsed time from a running timer is inaccurate\n", stderr);
  }

#if _POSIX_VERSION >= 200112L
  ret = timer->elapsed / 1e6;
#else
#error "pb_GetElapsedTime: not implemented for this system"
#endif
  return ret;
}

void pb_InitializeTimerSet(struct pb_TimerSet *timers) {
  int n;

  timers->wall_begin = get_time();

  timers->current = pb_TimerID_NONE;

  timers->async_markers = NULL;

  for (n = 0; n < pb_TimerID_LAST; n++) {
    pb_ResetTimer(&timers->timers[n]);
    timers->sub_timer_list[n] = NULL; // free first?
  }
}

void pb_AddSubTimer(struct pb_TimerSet *timers, char *label,
                    enum pb_TimerID pb_Category) {

  struct pb_SubTimer *subtimer =
      (struct pb_SubTimer *)malloc(sizeof(struct pb_SubTimer));

  int len = strlen(label);

  subtimer->label = (char *)malloc(sizeof(char) * (len + 1));
  sprintf(subtimer->label, "%s\n", label);

  pb_ResetTimer(&subtimer->timer);
  subtimer->next = NULL;

  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[pb_Category];
  if (subtimerlist == NULL) {
    subtimerlist =
        (struct pb_SubTimerList *)malloc(sizeof(struct pb_SubTimerList));
    subtimerlist->subtimer_list = subtimer;
    timers->sub_timer_list[pb_Category] = subtimerlist;
  } else {
    // Append to list
    struct pb_SubTimer *element = subtimerlist->subtimer_list;
    while (element->next != NULL) {
      element = element->next;
    }
    element->next = subtimer;
  }
}

void pb_SwitchToSubTimer(struct pb_TimerSet *timers, char *label,
                         enum pb_TimerID category) {

  // switchToSub( NULL, NONE
  // switchToSub( NULL, some
  // switchToSub( some, some
  // switchToSub( some, NONE -- tries to find "some" in NONE's sublist, which
  // won't be printed

  struct pb_Timer *topLevelToStop = NULL;
  if (timers->current != category && timers->current != pb_TimerID_NONE) {
    // Switching to subtimer in a different category needs to stop the top-level
    // current, different categoried timer.
    // NONE shouldn't have a timer associated with it, so exclude from branch
    topLevelToStop = &timers->timers[timers->current];
  }

  struct pb_SubTimerList *subtimerlist =
      timers->sub_timer_list[timers->current];
  struct pb_SubTimer *curr =
      (subtimerlist == NULL) ? NULL : subtimerlist->current;

  if (timers->current != pb_TimerID_NONE) {
    if (curr != NULL && topLevelToStop != NULL) {
      pb_StopTimerAndSubTimer(topLevelToStop, &curr->timer);
    } else if (curr != NULL) {
      pb_StopTimer(&curr->timer);
    } else {
      pb_StopTimer(topLevelToStop);
    }
  }

  subtimerlist = timers->sub_timer_list[category];
  struct pb_SubTimer *subtimer = NULL;

  if (label != NULL) {
    subtimer = subtimerlist->subtimer_list;
    while (subtimer != NULL) {
      if (strcmp(subtimer->label, label) == 0) {
        break;
      } else {
        subtimer = subtimer->next;
      }
    }
  }

  if (category != pb_TimerID_NONE) {

    if (subtimerlist != NULL) {
      subtimerlist->current = subtimer;
    }

    if (category != timers->current && subtimer != NULL) {
      pb_StartTimerAndSubTimer(&timers->timers[category], &subtimer->timer);
    } else if (subtimer != NULL) {
      // Same category, different non-NULL subtimer
      pb_StartTimer(&subtimer->timer);
    } else {
      // Different category, but no subtimer (not found or specified as NULL) --
      // unprefered way of setting topLevel timer
      pb_StartTimer(&timers->timers[category]);
    }
  }

  timers->current = category;
}

void pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer) {
  /* Stop the currently running timer */
  if (timers->current != pb_TimerID_NONE) {
    struct pb_SubTimer *currSubTimer = NULL;
    struct pb_SubTimerList *subtimerlist =
        timers->sub_timer_list[timers->current];

    if (subtimerlist != NULL) {
      currSubTimer = timers->sub_timer_list[timers->current]->current;
    }
    if (currSubTimer != NULL) {
      pb_StopTimerAndSubTimer(&timers->timers[timers->current],
                              &currSubTimer->timer);
    } else {
      pb_StopTimer(&timers->timers[timers->current]);
    }
  }

  timers->current = timer;

  if (timer != pb_TimerID_NONE) {
    pb_StartTimer(&timers->timers[timer]);
  }
}

void pb_PrintTimerSet(struct pb_TimerSet *timers) {

  pb_Timestamp wall_end = get_time();

  struct pb_Timer *t = timers->timers;
  struct pb_SubTimer *sub = NULL;

  int maxSubLength;

  const char *categories[] = {"IO",     "Kernel",     "Copy",
                              "Driver", "Copy Async", "Compute"};

  const int maxCategoryLength = 10;

  int i;
  for (i = 1; i < pb_TimerID_LAST - 1;
       ++i) { // exclude NONE and OVRELAP from this format
    if (pb_GetElapsedTime(&t[i]) != 0) {

      // Print Category Timer
      printf("%-*s: %f\n", maxCategoryLength, categories[i - 1],
             pb_GetElapsedTime(&t[i]));

      if (timers->sub_timer_list[i] != NULL) {
        sub = timers->sub_timer_list[i]->subtimer_list;
        maxSubLength = 0;
        while (sub != NULL) {
          // Find longest SubTimer label
          if (strlen(sub->label) > maxSubLength) {
            maxSubLength = strlen(sub->label);
          }
          sub = sub->next;
        }

        // Fit to Categories
        if (maxSubLength <= maxCategoryLength) {
          maxSubLength = maxCategoryLength;
        }

        sub = timers->sub_timer_list[i]->subtimer_list;

        // Print SubTimers
        while (sub != NULL) {
          printf(" -%-*s: %f\n", maxSubLength, sub->label,
                 pb_GetElapsedTime(&sub->timer));
          sub = sub->next;
        }
      }
    }
  }

  if (pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]) != 0)
    printf("CPU/Kernel Overlap: %f\n",
           pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]));

  float walltime = (wall_end - timers->wall_begin) / 1e6;
  printf("Timer Wall Time: %f\n", walltime);
}

void pb_DestroyTimerSet(struct pb_TimerSet *timers) {
  /* clean up all of the async event markers */
  struct pb_async_time_marker_list **event = &(timers->async_markers);
  while (*event != NULL) {
    struct pb_async_time_marker_list **next = &((*event)->next);
    free(*event);
    (*event) = NULL;
    event = next;
  }

  int i = 0;
  for (i = 0; i < pb_TimerID_LAST; ++i) {
    if (timers->sub_timer_list[i] != NULL) {
      struct pb_SubTimer *subtimer = timers->sub_timer_list[i]->subtimer_list;
      struct pb_SubTimer *prev = NULL;
      while (subtimer != NULL) {
        free(subtimer->label);
        prev = subtimer;
        subtimer = subtimer->next;
        free(prev);
      }
      free(timers->sub_timer_list[i]);
    }
  }
}

float *Qr_GPU, *Qi_GPU; /* Q signal (complex) */
float *Qr_CPU, *Qi_CPU; /* Q signal (complex) */

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

void ComputePhiMagCPU(float *phiR, float *phiI, float *phiMag) {
  int indexK = 0;
  char RST_AI1 = 0;
  RST_AI1 |= !(((void*) (phiI + 0) > (void*) (phiMag + 2048))
  || ((void*) (phiMag + 0) > (void*) (phiI + 2048)));
  RST_AI1 |= !(((void*) (phiI + 0) > (void*) (phiR + 2048))
  || ((void*) (phiR + 0) > (void*) (phiI + 2048)));
  RST_AI1 |= !(((void*) (phiMag + 0) > (void*) (phiR + 2048))
  || ((void*) (phiR + 0) > (void*) (phiMag + 2048)));
  #pragma omp target data map(to: phiI[0:2048],phiR[0:2048]) map(tofrom: phiMag[0:2048]) if(!RST_AI1)
  #pragma omp target if(!RST_AI1)
  #pragma omp parallel for 
  for (indexK = 0; indexK < NK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real * real + imag * imag;
  }
}

void ComputeQGPU(struct kValues *kVals, float *x, float *y, float *z, float *Qr,
                 float *Qi, int lNK, int lNX, double lPIx2) {
  int indexK, indexX;
  //#pragma omp target device(1)
  //#pragma omp target map(to: kVals[:NK], x[:NX], y[:NX], z[:NX]) map(tofrom:
  //Qr[:NX], Qi[:NX])
  //#pragma omp parallel for
  for (indexK = 0; indexK < lNK; indexK++) {
    for (indexX = 0; indexX < lNX; indexX++) {
      float expArg =
          lPIx2 * (kVals[indexK].Kx * x[indexX] + kVals[indexK].Ky * y[indexX] +
                   kVals[indexK].Kz * z[indexX]);

      float cosArg = cos(expArg);
      float sinArg = sin(expArg);

      float phi = kVals[indexK].PhiMag;
      Qr[indexX] += phi * cosArg;
      Qi[indexX] += phi * sinArg;
    }
  }
}

void ComputeQCPU(struct kValues *kVals, float *x, float *y, float *z, float *Qr,
                 float *Qi) {
  int indexK, indexX;
  for (indexK = 0; indexK < NK; indexK++) {
    for (indexX = 0; indexX < NX; indexX++) {
      float expArg =
          PIx2 * (kVals[indexK].Kx * x[indexX] + kVals[indexK].Ky * y[indexX] +
                  kVals[indexK].Kz * z[indexX]);

      float cosArg = cos(expArg);
      float sinArg = sin(expArg);

      float phi = kVals[indexK].PhiMag;
      Qr[indexX] += phi * cosArg;
      Qi[indexX] += phi * sinArg;
    }
  }
}

void createDataStructsCPU(float **phiMag, float **Qr, float **Qi) {
  *phiMag = (float *)malloc(NK * sizeof(float));
  *Qr = (float *)malloc(NX * sizeof(float));
  memset((void *)*Qr, 0, NX * sizeof(float));
  *Qi = (float *)malloc(NX * sizeof(float));
  memset((void *)*Qi, 0, NX * sizeof(float));
}

void inputData(char *fName, int *_numK, int *_numX, float **kx, float **ky,
               float **kz, float **x, float **y, float **z, float **phiR,
               float **phiI) {
  int numK, numX;
  FILE *fid = fopen(fName, "r");

  if (fid == NULL) {
    fprintf(stderr, "Cannot open input file\n");
    exit(-1);
  }
  fread(&numK, sizeof(int), 1, fid);
  *_numK = numK;
  fread(&numX, sizeof(int), 1, fid);
  *_numX = numX;
  *kx = (float *)malloc(numK * sizeof(float));
  fread(*kx, sizeof(float), numK, fid);
  *ky = (float *)malloc(numK * sizeof(float));
  fread(*ky, sizeof(float), numK, fid);
  *kz = (float *)malloc(numK * sizeof(float));
  fread(*kz, sizeof(float), numK, fid);
  *x = (float *)malloc(numX * sizeof(float));
  fread(*x, sizeof(float), numX, fid);
  *y = (float *)malloc(numX * sizeof(float));
  fread(*y, sizeof(float), numX, fid);
  *z = (float *)malloc(numX * sizeof(float));
  fread(*z, sizeof(float), numX, fid);
  *phiR = (float *)malloc(numK * sizeof(float));
  fread(*phiR, sizeof(float), numK, fid);
  *phiI = (float *)malloc(numK * sizeof(float));
  fread(*phiI, sizeof(float), numK, fid);
  fclose(fid);
}

void compareResults(float *A, float *A_GPU, float *B, float *B_GPU) {
  int i, fail = 0;

  for (i = 0; i < NX; i++) {
    if (percentDiff(A[i], A_GPU[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  for (i = 0; i < NX; i++) {
    if (percentDiff(B[i], B_GPU[i]) > ERROR_THRESHOLD) {
      fail++;
    }
  }

  // print results
  printf(">>\n   Non-Matching CPU-GPU Outputs Beyond Error Threshold of "
         "%4.2f%s: %d\n",
         ERROR_THRESHOLD, "%", fail);
}

double mriqGPU(int argc, char *argv[]) {
  int numX, numK;      /* Number of X and K values */
  int original_numK;   /* Number of K values in input file */
  float *kx, *ky, *kz; /* K trajectory (3D vectors) */
  float *x, *y, *z;    /* X coordinates (3D vectors) */
  float *phiR, *phiI;  /* Phi values (complex) */
  float *phiMag;       /* Magnitude of Phi */
  struct kValues *kVals;

  struct pb_Parameters *params;

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL)) {
    fprintf(stderr, "Expecting one input filename\n");
    exit(-1);
  }

  /* Read in data */
  fprintf(stdout, "<< Reading data ... ");
  inputData(params->inpFiles[0], &original_numK, &numX, &kx, &ky, &kz, &x, &y,
            &z, &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else {
    int inputK;
    char *end;
    inputK = strtol(argv[1], &end, 10);
    if (end == argv[1]) {
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

  kVals = (struct kValues *)calloc(numK, sizeof(struct kValues));
  int k;
  for (k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  fprintf(stdout, ">>\n<< Start computation on GPU... ");
  t_start_GPU = rtclock();
  ComputeQGPU(kVals, x, y, z, Qr_GPU, Qi_GPU, NK, NX, PIx2);
  t_end_GPU = rtclock();

  free(kx);
  free(ky);
  free(kz);
  free(x);
  free(y);
  free(z);
  free(phiR);
  free(phiI);
  free(phiMag);
  free(kVals);

  return t_end_GPU - t_start_GPU;
}

double mriqCPU(int argc, char *argv[]) {
  int numX, numK;      /* Number of X and K values */
  int original_numK;   /* Number of K values in input file */
  float *kx, *ky, *kz; /* K trajectory (3D vectors) */
  float *x, *y, *z;    /* X coordinates (3D vectors) */
  float *phiR, *phiI;  /* Phi values (complex) */
  float *phiMag;       /* Magnitude of Phi */
  struct kValues *kVals;

  struct pb_Parameters *params;

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL)) {
    fprintf(stderr, "Expecting one input filename\n");
    exit(-1);
  }

  /* Read in data */
  inputData(params->inpFiles[0], &original_numK, &numX, &kx, &ky, &kz, &x, &y,
            &z, &phiR, &phiI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (argc < 2)
    numK = original_numK;
  else {
    int inputK;
    char *end;
    inputK = strtol(argv[1], &end, 10);
    if (end == argv[1]) {
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

  kVals = (struct kValues *)calloc(numK, sizeof(struct kValues));
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

  free(kx);
  free(ky);
  free(kz);
  free(x);
  free(y);
  free(z);
  free(phiR);
  free(phiI);
  free(phiMag);
  free(kVals);

  return t_end - t_start;
}

int main(int argc, char *argv[]) {
  double t_GPU, t_CPU;

  fprintf(stdout,
          "<< Creating the Q data structure for fast convolution-based\n");
  fprintf(stdout,
          "   Hessian multiplication for arbitrary k-space trajectories.>>\n");
  fprintf(stdout, "<< Elements per Grid: 2048 >>\n\n");
  fprintf(stdout, "   for (indexK = 0; indexK < 2048; indexK++) \n");
  fprintf(stdout, "       for (indexX = 0; indexX < 262144; indexX++) { \n");
  fprintf(stdout,
          "           float expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +\n");
  fprintf(stdout,
          "                          kVals[indexK].Ky * y[indexX] + \n");
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

