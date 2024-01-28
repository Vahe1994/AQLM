from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="codebook_cuda",
    ext_modules=[cpp_extension.CUDAExtension("codebook_cuda", ["codebook_cuda.cpp", "codebook_cuda_kernel.cu"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
