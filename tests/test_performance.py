''' 用于测试模块性能的测例 '''
import os
from itertools import product
from time import perf_counter

import numpy as np
from cholmod import _modes, _ordering_methods, cholesky
from scipy import io

testcases = {'ted_B': 100, 's3rmt3m3': 100,
             'thermomech_dM': 10, 'parabolic_fem': 10,
             'nd12k': 3, 'boneS10': 3,
             'nd24k': 2, 'PFlow_742': 2}


os.system('make')
os.system('rm *.o')
assert os.path.exists('./cholmod_c_test')


def read_mtx(path: str):
    matrix = io.mmread(path).tocsc()
    assert (matrix.indices.dtype == np.int32)
    return matrix


modes = {key: _modes[key] for key in ['simplicial', 'supernodal']}
ordering_methods = {key: _ordering_methods[key] for key in ['amd', 'metis']}

for problem, trail in testcases.items():
    # 保证测例均为 SPD
    path = f"./test_data/{problem}.mtx"
    X = read_mtx(path)
    print(f"Testcase {problem}")

    for mode, ordering in product(modes, ordering_methods):
        print(f'mode {mode}, ordering method {ordering}')
        print("Py Performance Test")
        elapsed = 1e10
        for _ in range(trail):
            start = perf_counter()
            cholesky(X, mode=mode, ordering_method=ordering)
            elapsed = min(elapsed, perf_counter() - start)
        print(f"Overall time elasped:  {elapsed:12.6f} s")
        print("C Performance Test")
        ctest = os.popen(f'./cholmod_c_test -f {path} -t{trail} '
                         f'-m{modes[mode]} -o{ordering_methods[ordering]}')
        print(ctest.read())
