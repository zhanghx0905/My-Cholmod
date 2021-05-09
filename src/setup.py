"""
A wrapper for the SuiteSparse CHOLMOD library for sparse Cholesky
decomposition.
"""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

setup(
    packages=find_packages(),
    ext_modules=cythonize(
         Extension(
            "cholmod",
            ["./cholmod.pyx"],
            include_dirs=[
                np.get_include(),
                sys.prefix + "/include",
                "/usr/include/suitesparse",
            ],
            library_dirs=[],
            libraries=["cholmod"],
        )
    ),
)
