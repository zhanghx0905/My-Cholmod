import os
from itertools import product
from time import perf_counter

import numpy as np
from pytest import mark
from scipy import sparse

from sparsekit import cholesky, cholesky_AAt, modes, ordering_methods
from tools import complex_matrix, mm_matrix, real_matrix, assert_allclose


def test_beta():
    for matrix, beta in product([real_matrix, complex_matrix],
                                [0, 1, 3.4]):
        mat = matrix + beta * sparse.eye(*matrix.shape)
        for use_long in [False, True]:
            if use_long:
                matrix.indices = matrix.indices.astype(np.int64)
                matrix.indptr = matrix.indptr.astype(np.int64)
            for ordering_method, mode in product(ordering_methods, modes):
                f = cholesky(matrix, beta, mode, ordering_method)
                L, P = f.L(), f.P()
                assert np.allclose(
                    (L @ L.H).todense(),
                    mat.todense()[P[:, np.newaxis], P[np.newaxis, :]])


def test_big_matrix():
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


@mark.skip
def test_performance():
    testcases = {'ted_B': 100,
                 'thermomech_dM': 10,
                 'nd12k': 3,
                 'PFlow_742': 2, }

    for problem, trial in testcases.items():
        # 保证测例均为 SPD
        X = mm_matrix(problem)
        print(f"Testcase {problem}")
        print("Py Performance Test")
        elapsed = 1e10
        for _ in range(trial):
            start = perf_counter()
            cholesky(X)
            elapsed = min(elapsed, perf_counter() - start)
        print(f"Overall time elasped:  {elapsed:12.6f} s")
        print("C Performance Test")
        ctest = os.popen(
            f'./cholmod_c_test -f ./test_data/{problem}.mtx -t{trial}')
        print(ctest.read())


if __name__ == '__main__':
    test_performance()
