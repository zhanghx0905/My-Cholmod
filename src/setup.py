import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "cholmod",
        [
            "./cholmod.pyx"
        ],
        include_dirs=[
            numpy.get_include(),
            "/usr/include/suitesparse",
        ],
        define_macros=[
            ("NDEBUG",),
        ],
        extra_compile_args=[
            "-O3",
            # disable warnings caused by Cython using the deprecated
            # NumPy C-API
            "-Wno-cpp", "-Wno-unused-function"
        ],
        library_dirs=[
        ],
        libraries=[
            "cholmod", 'spqr'
        ]
    )
]
setup(
    name='cholmod',
    version='0.1',
    packages=[''],
    package_data={
        '': ["*.pyi"]
    },
    ext_modules=cythonize(extensions,
                          language_level=3,
                          compiler_directives={
                              'c_string_type': 'str',
                              'c_string_encoding': 'default'
                          })
)
