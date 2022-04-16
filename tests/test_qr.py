from functools import partial
import numpy as np
from cholmod import qr

from scipy import sparse
from numpy.testing import assert_allclose

assert_allclose = partial(assert_allclose, rtol=1e-5, atol=1e-8)
real_matrix = sparse.csc_matrix([[10, 0, 3, 0],
                                 [0, 5, 0, -2],
                                 [3, 0, 5, 0],
                                 [0, -2, 0, 2]])
complex_matrix = sparse.csc_matrix([[10, 0, 3 - 1j, 0],
                                    [0, 5, 0, -2],
                                    [3 + 1j, 0, 5, 0],
                                    [0, -2, 0, 2]])


def mm_matrix(name: str) -> sparse.csc_matrix:
    from scipy.io import mmread
    matrix = mmread(f"./test_data/{name}.mtx")
    assert sparse.issparse(matrix)
    return matrix.tocsc()


testcases = [
    real_matrix, complex_matrix,
    *[mm_matrix(f"case{problem}") for problem in range(1, 5)]
]


def perm_vec_to_mat(E: np.ndarray) -> sparse.csc_matrix:
    n = len(E)
    return sparse.csc_matrix((np.ones(n), (E, np.arange(n))), shape=(n, n))


for case in testcases:
    Q, R, E, _ = qr(case)
    P = perm_vec_to_mat(E)
    assert_allclose((Q*R).todense(), (case * P).todense())
