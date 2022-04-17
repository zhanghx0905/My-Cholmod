import os
from functools import partial

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.io import mmread

assert_allclose = partial(assert_allclose, rtol=1e-5, atol=1e-8)
real_matrix = sparse.csc_matrix([[10, 0, 3, 0],
                                 [0, 5, 0, -2],
                                 [3, 0, 5, 0],
                                 [0, -2, 0, 2]])
complex_matrix = sparse.csc_matrix([[10, 0, 3 - 1j, 0],
                                    [0, 5, 0, -2],
                                    [3 + 1j, 0, 5, 0],
                                    [0, -2, 0, 2]])

_TESTDIR = os.path.join(os.path.dirname(__file__), 'test_data')


def mm_matrix(name: str) -> sparse.csc_matrix:
    matrix = mmread(os.path.join(_TESTDIR, f'{name}.mtx'))
    assert sparse.issparse(matrix)
    matrix = matrix.tocsc()
    assert matrix.indices.dtype == np.int32
    return matrix


def perm_vec_to_mat(E: np.ndarray) -> sparse.csc_matrix:
    n = len(E)
    return sparse.csc_matrix((np.ones(n), (E, np.arange(n))), shape=(n, n))
