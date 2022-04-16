from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import sparse


def cholesky(A: sparse.csc_matrix,
             beta: float = 0,
             mode: str = "auto",
             ordering_method: str = "default") -> Factor: ...


def cholesky_AAt(A: sparse.csc_matrix,
                 beta: float = 0,
                 mode: str = "auto",
                 ordering_method: str = "default") -> Factor: ...


def qr(A: sparse.csc_matrix) \
    -> Tuple[sparse.csc_matrix, sparse.csc_matrix, np.ndarray, int]: ...


modes: set[str]
ordering_methods: set[str]


class Factor:
    def copy(self) -> Factor: ...
    def cholesky_inplace(self, A: sparse.csc_matrix, beta: float = 0): ...
    def cholesky_AAt_inplace(self, A: sparse.csc_matrix, beta: float = 0): ...
    def L_D(self) -> Tuple[sparse.csc_matrix, sparse.dia_matrix]: ...
    def L(self) -> sparse.csc_matrix: ...
    def P(self) -> np.ndarray: ...


class CholmodError(Exception):
    ...


class CholmodWarning(UserWarning):
    ...
