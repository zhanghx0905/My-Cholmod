/* Refer to CHOLMOD/Demo/cholmod_l_demo.c */
#include <assert.h>
#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cholmod.h"

#define CPUTIME (SuiteSparse_time())

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define ABS(a) (((a) >= (0)) ? (a) : -(a))

#include "cholmod_function.h"

int mode = CHOLMOD_AUTO, ordering = CHOLMOD_AMD;
int trails = 10;

#define DEBUG 0

/* ff is a global variable so that it can be closed by my_handler */
FILE *ff;

/* halt if an error occurs */
static void my_handler(int status, const char *file, int line,
                       const char *message) {
  printf("cholmod error: file: %s line: %d status: %d: %s\n", file, line,
         status, message);
  if (status < 0) {
    if (ff != NULL) fclose(ff);
    exit(0);
  }
}

int main(int argc, char **argv) {
  double t, ta, tf, tot = 10000, anorm;
  FILE *f;
  cholmod_sparse *A;
  double beta[2], xlnz;
  cholmod_common Common, *cm;
  cholmod_factor *L;
  int trial;
  int ver[3];

  /* ---------------------------------------------------------------------- */
  /* get the file containing the input matrix */
  /* ---------------------------------------------------------------------- */

  ff = NULL;
  int opt;
  char *optstr = "f:t::m::o::";  // file, mode, ordering method
  while ((opt = getopt(argc, argv, optstr)) != -1) {
    switch (opt) {
      case 'f':
        if ((f = fopen(optarg, "r")) == NULL) {
          my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
                     "unable to open file");
        }
        ff = f;
        // printf("CHOLMOD Performance Test for %s\n", optarg);
        break;
      case 't':
        trails = atoi(optarg);
        break;
      case 'm':
        mode = atoi(optarg);
        break;
      case 'o':
        ordering = atoi(optarg);
        break;
      default:
        printf(
            "usage: %s -f <testcase path> -t<trails = 10> -m<mode = AUTO(1)> "
            "-o<ordering = AMD(2)>",
            argv[0]);
        exit(1);
    }
  }
  //   printf("trails %d, mode %d, ordering %d\n", trails, mode, ordering);

  /* ---------------------------------------------------------------------- */
  /* start CHOLMOD and set parameters */
  /* ---------------------------------------------------------------------- */

  cm = &Common;
  cholmod_start(cm);
  CHOLMOD_FUNCTION_DEFAULTS; /* just for testing (not required) */

  //   cm->error_handler = my_handler;
  cm->supernodal = mode;
  cm->nmethods = 1;
  cm->method[0].ordering = ordering;
  cm->postorder = 1;
  /* cm->useGPU = 1; */
  /* use default parameter settings, except for the error handler.  This
   * demo program terminates if an error occurs (out of memory, not positive
   * definite, ...).  It makes the demo program simpler (no need to check
   * CHOLMOD error conditions).  This non-default parameter setting has no
   * effect on performance. */

  /* Note that CHOLMOD will do a supernodal LL' or a simplicial LDL' by
   * default, automatically selecting the latter if flop/nnz(L) < 40. */

  beta[0] = 0;
  beta[1] = 0;

  /* ---------------------------------------------------------------------- */
  /* read in a matrix */
  /* ---------------------------------------------------------------------- */

#if DEBUG
  cholmod_version(ver);
  printf("cholmod version %d.%d.%d\n", ver[0], ver[1], ver[2]);
  SuiteSparse_version(ver);
  printf("SuiteSparse version %d.%d.%d\n", ver[0], ver[1], ver[2]);
#endif
  A = cholmod_read_sparse(f, cm);
  if (ff != NULL) {
    fclose(ff);
    ff = NULL;
  }
#if DEBUG
  anorm = cholmod_norm_sparse(A, 0, cm);
  printf("norm (A,inf) = %g\n", anorm);
  printf("norm (A,1)   = %g\n", cholmod_norm_sparse(A, 1, cm));
  cholmod_print_sparse(A, "A", cm);
#endif

  if (A->nrow > A->ncol) {
    /* Transpose A so that A'A+beta*I will be factorized instead */
    cholmod_sparse *C = cholmod_transpose(A, 2, cm);
    cholmod_free_sparse(&A, cm);
    A = C;
#if DEBUG
    printf("transposing input matrix\n");
#endif
  }
  /* ---------------------------------------------------------------------- */
  /* analyze and factorize */
  /* ---------------------------------------------------------------------- */
#if DEBUG
  if (A->stype == 0) {
    printf("Factorizing A*A'\n");
  } else {
    printf("Factorizing A\n");
  }
#endif

  for (int trail = 0; trail < trails; trail++) {
    t = CPUTIME;
    L = cholmod_analyze(A, cm);
    if (A->stype == 0) {
      cholmod_factorize_p(A, beta, NULL, 0, L, cm);
    } else {
      cholmod_factorize(A, L, cm);
    }

    tot = MIN(tot, CPUTIME - t);
    cholmod_free_factor(&L, cm);
  }

#if DEBUG
  // printf("factor flops %g nnz(L) %15.0f (w/no amalgamation)\n",
  //        cm->fl, cm->lnz);
  if (A->stype != 0) {
    printf("nnz(A):    %15.0f\n", cm->anz);
  } else {
    printf("nnz(A*A'): %15.0f\n", cm->anz);
  }
// if (cm->lnz > 0)
// {
//     printf("flops / nnz(L):  %8.1f\n", cm->fl / cm->lnz);
// }
#endif

  printf("Overall time elasped:  %12.6f s\n", tot);

  cholmod_free_sparse(&A, cm);
  cholmod_finish(cm);

  return (0);
}
