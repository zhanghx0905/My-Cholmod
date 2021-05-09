''' 用于测试模块性能的测例 '''
from time import perf_counter

import numpy as np
from cholmod import cholesky
from scipy import io

PROBLEMS = ['ted_B', 's3rmt3m3', 'thermomech_dM', 'parabolic_fem']
NTRIALS = 100

for problem in PROBLEMS:
    path = f"./test_data/{problem}.mtx"
    print(f"Py Performance Test for {path}")
    matrix = io.mmread(path)
    # if matrix.shape[0] == matrix.shape[1] or (matrix.T != matrix).todense().any():
    #     X = (matrix.T @ matrix).tocsc()
    # else:
    # 保证测例均为 SPD
    X = matrix.tocsc()
    assert (X.indices.dtype == np.int32)
    elapsed = 1000.
    for _ in range(NTRIALS):
        start = perf_counter()
        cholesky(X)
        elapsed = min(elapsed, perf_counter() - start)
    print(f"Overall time elasped:  {elapsed:12.6f} s")
