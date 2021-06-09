#cython: binding = True
#cython: language_level = 3

cimport numpy as np

import warnings

import numpy as np
from scipy import sparse

# initialize the numpy C API
np.import_array()

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

cdef class Common
cdef class Factor

class CholmodError(Exception):
    pass

class CholmodWarning(UserWarning):
    pass

class CholmodTypeConversionWarning(CholmodWarning, sparse.SparseEfficiencyWarning):
    pass

cdef int _integer_typenum = np.NPY_INT32
cdef object _integer_py_dtype = np.dtype(np.int32)  # class 'np.dtype[]'

cdef int _long_typenum = np.NPY_INT64
cdef object _long_py_dtype = np.dtype(np.int64)

cdef int _real_typenum = np.NPY_FLOAT64
cdef object _real_py_dtype = np.dtype(np.float64)

cdef int _complex_typenum = np.NPY_COMPLEX128
cdef object _complex_py_dtype = np.dtype(np.complex128)

cdef _require_1darray(a, dtype):
    dtype = np.dtype(dtype)
    a = np.ascontiguousarray(a, dtype=dtype)
    return a

cdef int _np_typenum_for_data(int xtype):
    if xtype == CHOLMOD_COMPLEX:
        return _complex_typenum
    elif xtype == CHOLMOD_REAL:
        return _real_typenum
    else:
        raise CholmodError("cholmod->numpy type conversion failed")

cdef int _np_typenum_for_index(int itype):
    if itype == CHOLMOD_INT:
        return _integer_typenum
    elif itype == CHOLMOD_LONG:
        return _long_typenum
    else:
        raise CholmodError("cholmod->numpy type conversion failed")

cdef type _np_dtype_for_index(int itype):
    return np.PyArray_TypeObjectFromType(_np_typenum_for_index(itype))

cdef class _CholmodSparseDestructor:
    """ Destructor for NumPy arrays based on sparse data of Cholmod """
    cdef cholmod_sparse* _sparse
    cdef Common _common

    cdef init(self, cholmod_sparse* m, Common common):
        self._sparse = m
        self._common = common

    def __dealloc__(self):
        cholmod_c_free_sparse = cholmod_l_free_sparse if self._common._use_long else cholmod_free_sparse
        cholmod_c_free_sparse(&self._sparse, &self._common._common)

cdef _cholmod_sparse_to_scipy_sparse(cholmod_sparse * m, Common common):
    """ Build a csc_matrix onto m with with appropriate destructor. 
    'm' must have been allocated by cholmod. """
    cdef np.ndarray indptr = np.PyArray_SimpleNewFromData(1, [m.ncol + 1], _np_typenum_for_index(m.itype), m.p)
    cdef np.ndarray indices = np.PyArray_SimpleNewFromData(1, [m.nzmax], _np_typenum_for_index(m.itype), m.i)
    cdef np.ndarray data = np.PyArray_SimpleNewFromData(1, [m.nzmax], _np_typenum_for_data(m.xtype), m.x)
    cdef _CholmodSparseDestructor base = _CholmodSparseDestructor()
    base.init(m, common)
    # base.__dealloc__ is called only when all 3 arrays are destructed. 
    for array in (indptr, indices, data):
        np.set_array_base(array, base)

    return sparse.csc_matrix((data, indices, indptr), shape=(m.nrow, m.ncol))

cdef void _error_handler(
        int status, const char * file, int line, const char * msg) except * with gil:
    full_msg = f"{file.decode()}:{line:d}: {msg.decode()}(code = {status})"
    if status < 0:
        raise CholmodError(full_msg)
    elif status != 0:
        warnings.warn(full_msg, CholmodWarning)

def _check_for_csc(m):
    if not sparse.isspmatrix_csc(m):
        warnings.warn(f"Slow down for converting {m.__class__.__name__} to CSC format",
                      CholmodWarning)
        m = m.tocsc()
    return m

cdef class Common:
    cdef cholmod_common _common
    cdef int _complex
    cdef int _xtype
    cdef int _use_long

    def __cinit__(self, _complex, _use_long):
        self._complex = _complex
        self._xtype = CHOLMOD_COMPLEX if _complex else CHOLMOD_REAL
        self._use_long = _use_long
        cholmod_c_start = cholmod_l_start if _use_long else cholmod_start
        cholmod_c_start(&self._common)
        self._common.print = 0
        self._common.error_handler = (<void (*)(int, const char *, int, const char *)>_error_handler)

    def __dealloc__(self):
        cholmod_c_finish = cholmod_l_finish if self._use_long else cholmod_finish
        cholmod_c_finish(&self._common)

    cdef np.ndarray _cast(self, np.ndarray arr):
        ''' Converts the array to a type consistent with Common '''
        if not issubclass(arr.dtype.type, np.number):
            raise CholmodError(f"non-numeric dtype {arr.dtype}")
        if self._complex:
            # All numeric types can be upcast to complex:
            # asfortranarray: Return an array laid out in Fortran order in memory.
            return np.asfortranarray(arr, dtype=_complex_py_dtype)
        else:
            if issubclass(arr.dtype.type, np.complexfloating):
                raise CholmodError("Refuse to downcast complex types to real")
            else:
                return np.asfortranarray(arr, dtype=_real_py_dtype)

    cdef object _init_view_sparse(self, cholmod_sparse *out, m, symmetric):
        if symmetric and m.shape[0] != m.shape[1]:
            raise CholmodError("supposedly symmetric matrix is not square!")
        m = _check_for_csc(m)
        m.sort_indices()
        cdef np.ndarray indptr = _require_1darray(m.indptr, _np_dtype_for_index(self._common.itype))
        cdef np.ndarray indices = _require_1darray(m.indices, _np_dtype_for_index(self._common.itype))
        cdef np.ndarray data = self._cast(m.data)
        out.nrow, out.ncol = m.shape
        out.nzmax = m.nnz
        out.p = indptr.data
        out.i = indices.data
        out.x = data.data
        out.stype = -1 if symmetric else 0
        out.itype = self._common.itype
        out.dtype = CHOLMOD_DOUBLE
        out.xtype = self._xtype
        out.sorted = 1
        out.packed = 1
        return m, indptr, indices, data

cdef class Factor:
    """ A Cholesky decomposition with a particular fill-reducing permutation."""

    cdef readonly Common _common
    cdef cholmod_factor * _factor

    def __dealloc__(self):
        cholmod_c_free_factor = cholmod_l_free_factor if self._common._use_long else cholmod_free_factor
        cholmod_c_free_factor(&self._factor, &self._common._common)

    def cholesky_inplace(self, A, beta=0):
        """
        Updates the Factor so that it represents the Cholesky
        decomposition of (A + beta I), rather than whatever it
        contained before.
        """
        return self._cholesky_inplace(A, True, beta=beta)

    def cholesky_AAt_inplace(self, A, beta=0):
        """
        Updates the Factor so that it represents the Cholesky
        decomposition of (AA' + beta I), rather than whatever it
        contained before.
        """
        return self._cholesky_inplace(A, False, beta=beta)

    def _cholesky_inplace(self, A, symmetric, beta=0):
        cdef cholmod_sparse c_A
        cdef object ref = self._common._init_view_sparse(&c_A, A, symmetric)
        # 两个函数参数格式不同，不要 binding 到一个变量上
        if self._common._use_long:
            cholmod_l_factorize_p(&c_A, [beta, 0], NULL, 0, self._factor, &self._common._common)
        else:
            cholmod_factorize_p(&c_A, [beta, 0], NULL, 0, self._factor, &self._common._common)

    def copy(self):
        """ Copies the current Factor """
        cholmod_c_copy_factor = cholmod_l_copy_factor if self._common._use_long else cholmod_copy_factor

        cdef cholmod_factor * c_clone = cholmod_c_copy_factor(self._factor,
                                                        &self._common._common)
        cdef Factor clone = Factor()
        clone._common = self._common
        clone._factor = c_clone
        return clone

    cdef void _ensure_L_or_LD_inplace(self, bint want_L):
        cholmod_c_change_factor = cholmod_l_change_factor if self._common._use_long else cholmod_change_factor
        # In CHOLMOD, supernodal factorizations can only be LL'.
        want_super = self._factor.is_super and want_L
        cholmod_c_change_factor(self._factor.xtype,
                                want_L, # to_ll
                                want_super,
                                True, # to_packed
                                self._factor.is_monotonic,
                                self._factor,
                                &self._common._common)

    cdef _L_or_LD(self, bint want_L):
        cholmod_c_factor_to_sparse = cholmod_l_factor_to_sparse if self._common._use_long else cholmod_factor_to_sparse

        cdef Factor f = self.copy()
        cdef cholmod_sparse * l
        f._ensure_L_or_LD_inplace(want_L)
        l = cholmod_c_factor_to_sparse(f._factor,
                                     &f._common._common)
        return _cholmod_sparse_to_scipy_sparse(l, self._common)

    def L_D(self) -> (sparse.csc_matrix, sparse.dia_matrix):
        """
        Returns the lower-triangular csc_matrix L and diagonal dia_matirx D that
        L @ D @ L.T = A[P[:, None], P[None, :]]
        """
        ld = self._L_or_LD(False)
        l = sparse.tril(ld, -1) + sparse.eye(*ld.shape)
        d = sparse.dia_matrix((ld.diagonal(), [0]), shape=ld.shape)
        return (l, d)

    def L(self) -> sparse.csc_matrix:
        """ 
        Returns the lower-triangular csc_matrix L that
        L @ L.T = A[P[:, None], P[None, :]]
        """
        return self._L_or_LD(True)

    def P(self) -> np.ndarray:
        """
        Returns the fill-reducing permutation P as a 1d vector.
        The decomposition is of A[P[:, None], P[None, :]] (or similar for AA').
        """
        cdef np.ndarray out = np.PyArray_SimpleNewFromData(1, [self._factor.n], 
            _np_typenum_for_index(self._factor.itype), self._factor.Perm)
        np.set_array_base(out, self)

        return out

_modes = {
    "simplicial": CHOLMOD_SIMPLICIAL,
    "supernodal": CHOLMOD_SUPERNODAL,
    "auto": CHOLMOD_AUTO,
}
_ordering_methods = {
    "natural": CHOLMOD_NATURAL,
    "amd": CHOLMOD_AMD,
    "metis": CHOLMOD_METIS,
    "nesdis": CHOLMOD_NESDIS,
    "colamd": CHOLMOD_COLAMD,
    "default": None,
    "best": None,
}

def cholesky(A, beta=0, mode="auto", ordering_method="default"):
    """
    Returns a `Factor` object represented the Cholesky decomposition of
        A + beta I
        
    where `A` is a sparse, symmetric, positive-definite matrix, preferably
    in CSC format, and `beta` is any real scalar (usually 0 or 1). (And
    `I` denotes the identity matrix.)
    
    Only the lower triangular part of ``A`` is used.
    """
    return _cholesky(A, True, beta=beta, mode=mode, ordering_method=ordering_method)

def cholesky_AAt(A, beta=0, mode="auto", ordering_method="default"):
    """ 
    Returns a `Factor` object represented the Cholesky decomposition of
        AA' + beta I
        
    where `A` is a sparse matrix, preferably in CSC format, 
    and `beta` is any real scalar (usually 0 or 1). (And
    `I` denotes the identity matrix.)
    """
    return _cholesky(A, False, beta=beta, mode=mode, ordering_method=ordering_method)

def _cholesky(A, symmetric, beta=0, mode='auto', ordering_method="default"):
    A = _check_for_csc(A)
    use_long = (A.indices.dtype == np.int64)
   
    cdef Common common = Common(issubclass(A.dtype.type, np.complexfloating), use_long)
    cdef cholmod_sparse c_A
    cdef object ref = common._init_view_sparse(&c_A, A, symmetric)
    if mode not in _modes:
        warnings.warn(f"Unknown mode '{mode}', switching to 'auto'")
        mode = "auto"
    common._common.supernodal = _modes[mode]  

    if ordering_method not in _ordering_methods:
        warnings.warn(f"Unknown ordering method '{ordering_method}', switching to 'default'")
        ordering_method = "default"
    if ordering_method == "default":
        common._common.nmethods = 0
    elif ordering_method == "best":
        common._common.nmethods = 9
    else:
        common._common.nmethods = 1
        common._common.method[0].ordering = _ordering_methods[ordering_method]
    common._common.postorder = (ordering_method != "natural")

    cholmod_c_analyze = cholmod_l_analyze if use_long else cholmod_analyze

    cdef cholmod_factor *c_f = cholmod_c_analyze(&c_A, &common._common)
    if c_f is NULL:
        raise CholmodError("Error in cholmod_analyze")

    cdef Factor f = Factor()
    f._common = common
    f._factor = c_f
    if common._use_long:
        cholmod_l_factorize_p(&c_A, [beta, 0], NULL, 0, f._factor, &common._common)
    else:
        cholmod_factorize_p(&c_A, [beta, 0], NULL, 0, f._factor, &common._common)
    return f

__all__ = ["cholesky", "cholesky_AAt"]
