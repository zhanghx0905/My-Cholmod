#include <assert.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "SuiteSparseQR_C.h"
#include "cholmod_function.h"

#define Long SuiteSparse_long
#define CPUTIME (SuiteSparse_time())
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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
    cholmod_common Common, *cc;
    cholmod_sparse* A;
    int trials = 10;
    int opt;
    char* optstr = "f:t::"; // file, mode, ordering method
    while ((opt = getopt(argc, argv, optstr)) != -1) {
        switch (opt) {
        case 'f':
            if ((ff = fopen(optarg, "r")) == NULL) {
                my_handler(CHOLMOD_INVALID, __FILE__, __LINE__, "unable to open file");
            }
            break;
        case 't':
            trials = atoi(optarg);
            break;
        default:
            printf("usage: %s -f <testcase path> -t<trails = 10>\n", argv[0]);
            exit(1);
        }
    }
    /* start CHOLMOD */
    cc = &Common;
    cholmod_l_start(cc);
    cc->error_handler = my_handler;
    CHOLMOD_FUNCTION_DEFAULTS; /* just for testing (not required) */

    /* A = mread (stdin) ; read in the sparse matrix A */
    int mtype;
    A = (cholmod_sparse*)cholmod_l_read_matrix(ff, 1, &mtype, cc);
    fclose(ff);
    ff = NULL;
    if (mtype != CHOLMOD_SPARSE) {
        printf("input matrix must be sparse\n");
        exit(1);
    }

    cholmod_sparse** Q = malloc(sizeof(cholmod_sparse*));
    cholmod_sparse** R = malloc(sizeof(cholmod_sparse*));
    Long** E = malloc(sizeof(Long*));

    double t, tot = 10000;
    for (int _ = 0; _ < trials; _++) {
        t = CPUTIME;
        int rnk = SuiteSparseQR_C_QR(
            SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A->nrow,
            A, Q, R, E, cc);
        tot = MIN(tot, CPUTIME - t);
        cholmod_l_free_sparse(Q, cc);
        cholmod_l_free_sparse(R, cc);
        if (rnk == -1) {
            printf("Something bad happened!");
            break;
        }
    }
    printf("C Overall time elasped:  %12.6f s\n", tot);
    /* free everything */
    cholmod_l_free_sparse(&A, cc);

    cholmod_l_finish(cc);
    return (0);
}