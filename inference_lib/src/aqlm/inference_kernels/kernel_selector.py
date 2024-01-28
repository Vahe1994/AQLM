from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from aqlm.utils import _dequantize_weight, unpack_int_data

from .triton_kernel import triton_matmul


def forward_pass_quantized_linear(
    input: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if input.is_cuda:
        return triton_matmul(input, codes, codebooks, scales, bias)
    else:
        dequantized_weight = _dequantize_weight(
            unpack_int_data(codes, codebooks.shape[0].bit_length() - 1),
            codebooks,
            scales,
        )
        return F.linear(input, dequantized_weight, bias)
