import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.dirname(os.path.abspath(__file__))
CUDA_KERNEL = load(
    name="codebook_cuda",
    sources=[os.path.join(CUDA_FOLDER, "cuda_kernel.cpp"), os.path.join(CUDA_FOLDER, "cuda_kernel.cu")],
)

torch.library.define(
    "aqlm::code1x16_matmat", "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("aqlm::code1x16_matmat", "default", CUDA_KERNEL.code1x16_matmat)


@torch.library.impl_abstract("aqlm::code1x16_matmat")
def code1x16_matmat_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code1x16_matmat_dequant",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code1x16_matmat_dequant", "default", CUDA_KERNEL.code1x16_matmat_dequant)


@torch.library.impl_abstract("aqlm::code1x16_matmat_dequant")
def code1x16_matmat_dequant_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code1x16_matmat_dequant_transposed",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl(
    "aqlm::code1x16_matmat_dequant_transposed", "default", CUDA_KERNEL.code1x16_matmat_dequant_transposed
)


@torch.library.impl_abstract("aqlm::code1x16_matmat_dequant_transposed")
def code1x16_matmat_dequant_transposed_meta(input, codes, codebooks, scales, bias):
    return torch.empty(
        input.shape[:-1] + (codes.shape[1] * codebooks.shape[3],), device=input.device, dtype=input.dtype
    )


torch.library.define(
    "aqlm::code2x8_matmat", "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("aqlm::code2x8_matmat", "default", CUDA_KERNEL.code2x8_matmat)


@torch.library.impl_abstract("aqlm::code2x8_matmat")
def code2x8_matmat_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code2x8_matmat_dequant",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code2x8_matmat_dequant", "default", CUDA_KERNEL.code2x8_matmat_dequant)


@torch.library.impl_abstract("aqlm::code2x8_matmat_dequant")
def code2x8_matmat_dequant_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)


torch.library.define(
    "aqlm::code2x8_matmat_dequant_transposed",
    "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor",
)

torch.library.impl("aqlm::code2x8_matmat_dequant_transposed", "default", CUDA_KERNEL.code2x8_matmat_dequant_transposed)


@torch.library.impl_abstract("aqlm::code2x8_matmat_dequant_transposed")
def code2x8_matmat_dequant_transposed_meta(input, codes, codebooks, scales, bias):
    return torch.empty(
        input.shape[:-1] + (codes.shape[1] * codebooks.shape[3],), device=input.device, dtype=input.dtype
    )
