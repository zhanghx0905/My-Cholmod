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
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#include "cholmod_function.h"

/* ff is a global variable so that it can be closed by my_handler */
FILE* ff;

/* halt if an error occurs */
static void my_handler(int status, const char* file, int line,
    const char* message)
{
    printf("cholmod error: file: %s line: %d status: %d: %s\n", file, line,
        status, message);
    if (status < 0) {
        if (ff != NULL)
            fclose(ff);
        exit(0);
    }
}

int main(int argc, char** argv)
{
    double t, tot = 10000;

    cholmod_sparse* A;
    double beta[2];
    cholmod_common Common, *cm;
    cholmod_factor* L;
    int trials = 10;
    int mode = CHOLMOD_AUTO, ordering = CHOLMOD_AMD;

    /* ---------------------------------------------------------------------- */
    /* get the file containing the input matrix */
    /* ---------------------------------------------------------------------- */

    ff = NULL;
    int opt;
    char* optstr = "f:t::m::o::"; // file, mode, ordering method
    while ((opt = getopt(argc, argv, optstr)) != -1) {
        switch (opt) {
        case 'f':
            if ((ff = fopen(optarg, "r")) == NULL) {
                my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
                    "unable to open file");
            }
            break;
        case 't':
            trials = atoi(optarg);
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

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */
    /* ---------------------------------------------------------------------- */

    cm = &Common;
    cholmod_start(cm);
    CHOLMOD_FUNCTION_DEFAULTS; /* just for testing (not required) */

    cm->error_handler = my_handler;
    cm->supernodal = mode;
    cm->nmethods = 1;
    cm->method[0].ordering = ordering;
    cm->postorder = 1;
    /* cm->useGPU = 1; */

    beta[0] = 0;
    beta[1] = 0;

    /* ---------------------------------------------------------------------- */
    /* read in a matrix */
    /* ---------------------------------------------------------------------- */

    A = cholmod_read_sparse(ff, cm);
    if (ff != NULL) {
        fclose(ff);
        ff = NULL;
    }
    if (A->nrow > A->ncol) {
        /* Transpose A so that A'A+beta*I will be factorized instead */
        cholmod_sparse* C = cholmod_transpose(A, 2, cm);
        cholmod_free_sparse(&A, cm);
        A = C;
    }
    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    for (int _ = 0; _ < trials; _++) {
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

    printf("Overall time elasped:  %12.6f s\n", tot);
    cholmod_free_sparse(&A, cm);
    cholmod_finish(cm);

    return (0);
}
