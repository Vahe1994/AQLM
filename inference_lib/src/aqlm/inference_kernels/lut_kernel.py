import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

LUT_FOLDER = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(f"{LUT_FOLDER}/cmake-out/libaqlm_bindings.dylib")


@torch.library.register_fake("aqlm::code2x8_lut_matmat")
def code2x8_lut_matmat_meta(input, codes, codebooks, scales, bias=None):
    return torch.empty(
        input.shape[:-1] + (codes.shape[1],), device=input.device, dtype=input.dtype
    )
