import os
from time import perf_counter

from pytest import mark

from sparsekit import qr
from tools import (assert_allclose, complex_matrix, mm_matrix, perm_vec_to_mat,
                   real_matrix)


def test_correctness():
    testcases = [
        real_matrix, complex_matrix,
        *[mm_matrix(f"case{problem}") for problem in range(1, 5)]
    ]

    for case in testcases:
        Q, R, E, _ = qr(case)
        P = perm_vec_to_mat(E)
        assert_allclose((Q@R).todense(), (case @ P).todense())


@mark.skip
def test_performance():
    testcases = {'ted_B': 10, 's3rmt3m3': 5}
    for case, trial in testcases.items():
        mat = mm_matrix(case)
        elapsed = 1e5
        for _ in range(trial):
            start = perf_counter()
            qr(mat)
            elapsed = min(elapsed, perf_counter() - start)
        print(f"Py Overall time elasped:  {elapsed:12.6f} s")
        ctest = os.popen(f'./qr_c_test -f ./test_data/{case}.mtx -t{trial} ')
        print(ctest.read())


if __name__ == '__main__':
    test_performance()
