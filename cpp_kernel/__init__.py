"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp

import setuptools.sandbox
from multiprocessing import cpu_count
import numpy as np

package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])
if True or not os.path.exists(osp.join(package_abspath, "_bindings.so")):
    print("RECOMPILING")
    # try build _bfs.so
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(
            osp.join(package_abspath, "setup.py"), ["clean", "build"]
        )
        os.system(
            "cp {}/build/lib*/*.so {}/_bindings.so".format(package_abspath, package_abspath)
        )
        assert os.path.exists(osp.join(package_abspath, "_bindings.so"))
    finally:
        os.chdir(workdir)

from . import _bindings


def sum(A, B):
    assert A.shape == B.shape
    assert A.dtype == np.float32 and B.dtype == np.float32
    result = np.empty_like(A)
    
    _bindings.sum(
        A, B, result, 32
    )
    
    return result


def triple_for(lut, b_alt, n_jobs=None):
    n_jobs = n_jobs or cpu_count()
    if n_jobs < 0:
        n_jobs = cpu_count() - n_jobs + 1
    n_jobs = min(n_jobs, 32)
    
    assert lut.dtype == np.float32
    assert b_alt.dtype == np.uint8
        
    out_features = b_alt.shape[-2]
    output_vec = np.zeros(out_features, dtype=np.float32)
    _bindings.triple_for(
        lut, b_alt, output_vec, n_jobs
    )
    return output_vec
