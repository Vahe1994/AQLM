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
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    match (input.is_cuda, num_codebooks, codebook_size, out_group_size, in_group_size):
        case (True, 1, 65536, 1, 8):
            from aqlm.cuda.cuda_kernel import cuda_gemm_1x16

            return cuda_gemm_1x16(input, codes, codebooks, scales, bias)
        case (True, 2, 256, 1, 8):
            from aqlm.cuda.cuda_kernel import cuda_gemm_2x8

            return cuda_gemm_2x8(input, codes, codebooks, scales, bias)
        case (True, _, _, _, _):
            return triton_matmul(input, codes, codebooks, scales, bias)
        case _:
            dequantized_weight = _dequantize_weight(
                unpack_int_data(codes, codebooks.shape[0].bit_length() - 1),
                codebooks,
                scales,
            )
            return F.linear(input, dequantized_weight, bias)


def cuda_kernel_applicable(
    is_cuda: bool,
    num_codebooks: int,
    codebook_size: int,
    out_group_size: int,
    in_group_size: int,
) -> bool:
    return is_cuda and num_codebooks == 1 and codebook_size == 2**16 and out_group_size == 1 and in_group_size == 8


def triton_kernel_applicable(
    is_cuda: bool,
) -> bool:
    return is_cuda
