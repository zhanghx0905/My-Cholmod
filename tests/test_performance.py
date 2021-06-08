''' 用于测试模块性能的测例 '''
import os
from time import perf_counter

import numpy as np
from cholmod import cholesky
from scipy import io

testcases = ['ted_B', 's3rmt3m3', 'thermomech_dM', 'parabolic_fem',
             ]  # 'nd24k', 'nd12k', 'boneS10', 'PFlow_742']
NTRIALS = 10

os.system('make')
os.system('rm *.o')
assert os.path.exists('./cholmod_c_test')


def read_mtx(path: str):
    matrix = io.mmread(path).tocsc()
    assert (matrix.indices.dtype == np.int32)
    return matrix


for problem in testcases:
    # 保证测例均为 SPD
    path = f"./test_data/{problem}.mtx"
    X = read_mtx(path)
    print(f"Py Performance Test for {problem}")
    elapsed = 1000.
    for _ in range(NTRIALS):
        start = perf_counter()
        cholesky(X)
        elapsed = min(elapsed, perf_counter() - start)
    print(f"Overall time elasped:  {elapsed:12.6f} s")

    ret = os.popen(f'./cholmod_c_test {path} {NTRIALS}').read()
    print(ret)
