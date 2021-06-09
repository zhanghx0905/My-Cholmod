''' 用于测试实现正确性的测例 '''
from functools import partial

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from cholmod import (_modes, _ordering_methods, cholesky,
                     cholesky_AAt)


modes = _modes.keys()
ordering_methods = _ordering_methods.keys()

assert_allclose = partial(assert_allclose, rtol=1e-5, atol=1e-8)

real_matrix = sparse.csc_matrix([[10, 0, 3, 0],
                                 [0, 5, 0, -2],
                                 [3, 0, 5, 0],
                                 [0, -2, 0, 2]])
complex_matrix = sparse.csc_matrix([[10, 0, 3 - 1j, 0],
                                    [0, 5, 0, -2],
                                    [3 + 1j, 0, 5, 0],
                                    [0, -2, 0, 2]])


def test_beta():
    for matrix in [real_matrix, complex_matrix]:
        for beta in [0, 1, 3.4]:
            mat_plus_beta = matrix + beta * sparse.eye(*matrix.shape)
            for use_long in [False, True]:
                if use_long:
                    mat_plus_beta.indices = np.asarray(mat_plus_beta.indices, dtype=np.int64)
                    mat_plus_beta.indptr = np.asarray(mat_plus_beta.indptr, dtype=np.int64)
                for ordering_method in ordering_methods:
                    for mode in modes:
                        f = cholesky(matrix, beta=beta, mode=mode,
                                     ordering_method=ordering_method)
                        L, P = f.L(), f.P()
                        assert np.allclose(
                            (L @ L.H).todense(),
                            mat_plus_beta.todense()[P[:, np.newaxis], P[np.newaxis, :]])


def test_natural_ordering():
    A = real_matrix
    f = cholesky(A, ordering_method="natural")
    P = f.P()
    assert_array_equal(P, np.arange(len(P)))
    f = cholesky(sparse.eye(10, 10))
    assert np.all(f.P() == np.arange(10))


def test_big_matrix():
    def mm_matrix(name):
        from scipy.io import mmread
        matrix = mmread(f"./test_data/{name}.mtx")
        if sparse.issparse(matrix):
            matrix = matrix.tocsc()
        return matrix

    for problem in range(1, 5):
        X = mm_matrix(f"case{problem}")
        XtX = (X.T * X).tocsc()
        for mode in modes:
            f1 = cholesky(XtX, mode=mode)
            f3 = f1.copy()
            f3.cholesky_inplace(XtX)
            f2 = cholesky_AAt(X.T, mode=mode)
            f4 = f2.copy()
            f4.cholesky_AAt_inplace(X.T)
            for f in (f1, f2, f3, f4):
                P, L = f.P(), f.L()
                pXtX = XtX.todense()[P[:, np.newaxis],
                                     P[np.newaxis, :]]
                assert_allclose((L @ L.T).todense(), pXtX)
                L, D = f.L_D()
                assert_allclose((L @ D @ L.T).todense(), pXtX)
                assert_allclose(np.prod(np.diag(D.todense())),
                                np.linalg.det(XtX.todense()))


if __name__ == '__main__':
    test_beta()
    test_big_matrix()
    test_natural_ordering()
    print("cholmod correctness test OK")