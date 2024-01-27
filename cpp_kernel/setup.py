#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy
import os

os.environ['CC'] = 'icpx'
os.environ['CXX'] = 'icpx'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# extension module
ext = Extension("_bindings",
                ["lib.i", "lib.cc"],
                include_dirs=[numpy_include],
                extra_compile_args=["-Ofast", "-mavx", "-fprofile-ml-use", "-march=native", "-xHost"],
                )

# setup
setup(name="_bindings",
      description="Function that performs batch-parallel pathfinding (numpy.i: a SWIG Interface File for NumPy)",
      author="Grafstvo",
      version="1.0",
      ext_modules=[ext]
)
