import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

LUT_FOLDER = os.path.dirname(os.path.abspath(__file__))
LUT_KERNEL = load(
    name="lut_matmat",
    sources=[
        os.path.join(LUT_FOLDER, "lut_kernel_pytorch.cpp"),
        os.path.join(LUT_FOLDER, "lut_kernel.h"),
    ],
    extra_include_paths=["/Users/blacksamorez/reps/"],
)

torch.library.define(
    "aqlm::code2x8_lut_matmat", "(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor bias) -> Tensor"
)

torch.library.impl("aqlm::code2x8_lut_matmat", "default", LUT_KERNEL.code2x8_lut_matmat)
