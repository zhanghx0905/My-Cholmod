''' 用于测试模块性能的测例 '''
import timeit

import numpy as np
from cholmod import cholesky
from scipy import io, sparse

PROMLEMS = ['ted_B', 's3rmt3m3']
NTRIALS = 100


for problem in PROMLEMS:
    path = f"./test_data/{problem}.mtx"
    print(f"Py Performance Test for {path}")
    matrix = io.mmread(path)
    if matrix.shape[0] != matrix.shape[1] or (matrix.T != matrix).todense().any():
        X = (matrix.T @ matrix).tocsc()
    else:
        X = matrix.tocsc()

    assert (X.indices.dtype == np.int32)
    elapsed = timeit.timeit('cholesky(X)', globals=globals(), number=NTRIALS)
    elapsed /= NTRIALS
    print(f"Overall time elasped:  {elapsed:12.6f} s")
