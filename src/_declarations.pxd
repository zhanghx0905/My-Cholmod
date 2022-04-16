cdef extern from "cholmod.h":
    cdef enum:
        CHOLMOD_INT, CHOLMOD_LONG
        CHOLMOD_REAL, CHOLMOD_COMPLEX
        CHOLMOD_DOUBLE
        CHOLMOD_AUTO, CHOLMOD_SIMPLICIAL, CHOLMOD_SUPERNODAL
        CHOLMOD_NATURAL, CHOLMOD_GIVEN, CHOLMOD_AMD, CHOLMOD_METIS, CHOLMOD_NESDIS, CHOLMOD_COLAMD, CHOLMOD_POSTORDERED

    ctypedef int SuiteSparse_long

    ctypedef struct cholmod_method_struct:
        int ordering

    ctypedef struct cholmod_common:
        int supernodal
        int status
        int itype
        int print
        int nmethods, postorder
        cholmod_method_struct * method
        void (*error_handler)(int status, const char * file, int line, const char * msg)

    ctypedef struct cholmod_sparse:
        size_t nrow, ncol, nzmax
        void * p # column pointers
        void * i # row indices
        void * x
        int stype # 0 = regular, -1 = lower triangular
        int itype, dtype, xtype
        int sorted
        int packed

    ctypedef struct cholmod_factor:
        size_t n
        void * Perm
        int itype, xtype
        int is_ll, is_super, is_monotonic

    int cholmod_start(cholmod_common *) except *
    int cholmod_l_start(cholmod_common *) except *

    int cholmod_finish(cholmod_common *) except *
    int cholmod_l_finish(cholmod_common *) except *

    int cholmod_free_sparse(cholmod_sparse **, cholmod_common *) except *
    int cholmod_l_free_sparse(cholmod_sparse **, cholmod_common *) except *

    int cholmod_free_factor(cholmod_factor **, cholmod_common *) except *
    int cholmod_l_free_factor(cholmod_factor **, cholmod_common *) except *

    cholmod_factor * cholmod_copy_factor(cholmod_factor *, cholmod_common *) except? NULL
    cholmod_factor * cholmod_l_copy_factor(cholmod_factor *, cholmod_common *) except? NULL

    cholmod_factor * cholmod_analyze(cholmod_sparse *, cholmod_common *) except? NULL
    cholmod_factor * cholmod_l_analyze(cholmod_sparse *, cholmod_common *) except? NULL

    int cholmod_factorize_p(cholmod_sparse *, double beta[2],
                            int * fset, size_t fsize,
                            cholmod_factor *,
                            cholmod_common *) except *
    int cholmod_l_factorize_p(cholmod_sparse *, double beta[2],
                              SuiteSparse_long * fset, size_t fsize,
                              cholmod_factor *,
                              cholmod_common *) except *

    int cholmod_change_factor(int to_xtype, int to_ll, int to_super,
                              int to_packed, int to_monotonic,
                              cholmod_factor *, cholmod_common *) except *
    int cholmod_l_change_factor(int to_xtype, int to_ll, int to_super,
                                int to_packed, int to_monotonic,
                                cholmod_factor *, cholmod_common *) except *

    cholmod_sparse * cholmod_factor_to_sparse(cholmod_factor *,
                                              cholmod_common *) except? NULL
    cholmod_sparse * cholmod_l_factor_to_sparse(cholmod_factor *,
                                                cholmod_common *) except? NULL


cdef extern from "SuiteSparseQR_C.h":
    cdef enum:
        SPQR_ORDERING_DEFAULT
        SPQR_DEFAULT_TOL, SPQR_NO_TOL

    int SuiteSparseQR_C_QR(int ordering, double tol,
                           int econ, cholmod_sparse *A,
                           cholmod_sparse **Q, cholmod_sparse **R,
                           long **E, cholmod_common *cc) except *