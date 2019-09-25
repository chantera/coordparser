#!/usr/bin/env python

import os
from distutils.core import setup, Extension

from Cython.Build import build_ext


dirname = os.path.dirname(os.path.abspath(__file__))


def extension(*args, **kwags):
    return Extension(
        *args, **kwags,
        language="c++",
        include_dirs=[os.path.join(dirname, "src")],
        extra_compile_args=["-std=c++11"],
    )


ext_modules = [
    extension("src.parsers.cky", sources=["src/parsers/cky.pyx"]),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
