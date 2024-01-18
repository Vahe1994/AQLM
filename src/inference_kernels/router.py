from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference_kernels.triton_kernel import aqlm_gemm_stupid as triton_gemm
from src.utils import _dequantize_weight, unpack_int_data


def forward_pass_quantized_linear(
    input: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    if input.is_cuda:
        matmul_result = triton_gemm(input, codes, codebooks, scales)
        if bias is not None:
            matmul_result += bias
        return matmul_result
    else:
        dequantized_weight = _dequantize_weight(
            unpack_int_data(codes, codebooks.shape[0].bit_length() - 1),
            codebooks,
            scales,
        )
        return F.linear(input, dequantized_weight, bias)
