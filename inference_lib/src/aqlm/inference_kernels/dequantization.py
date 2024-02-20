from typing import Optional

import torch
import torch.nn.functional as F
from aqlm.utils import _dequantize_weight, unpack_int_data
from torch import nn


def dequantize_gemm(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    dequantized_weight = _dequantize_weight(
        unpack_int_data(codes, codebooks.shape[1].bit_length() - 1),
        codebooks,
        scales,
    )
    return F.linear(input, dequantized_weight, bias)
