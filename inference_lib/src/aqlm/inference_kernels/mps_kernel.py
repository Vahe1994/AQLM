import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

MPS_FOLDER = os.path.dirname(os.path.abspath(__file__))
MPS_KERNEL = load(
    name="codebook_mps",
    sources=[os.path.join(MPS_FOLDER, "mps_kernel.h"), os.path.join(MPS_FOLDER, "mps_kernel.mm")],
)

torch.library.define(
    "aqlm::code1x16_matmat_mps", "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("aqlm::code1x16_matmat_mps", "default", MPS_KERNEL.code1x16_matmat)


@torch.library.impl_abstract("aqlm::code1x16_matmat_mps")
def code1x16_matmat_meta(input, codes, codebooks, scales, bias):
    return torch.empty(input.shape[:-1] + (codes.shape[0],), device=input.device, dtype=input.dtype)
