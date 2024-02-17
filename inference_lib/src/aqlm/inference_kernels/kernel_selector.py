from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from aqlm.utils import _dequantize_weight, unpack_int_data


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
            from .cuda_kernel import CUDA_FOLDER

            assert (
                input.dtype == torch.float16
            ), f"please load the model with `torch_dtype=torch.float16`, as {input.dtype} is not supported on GPU yet"
            return torch.ops.aqlm_cuda_kernel.code1x16_matmat(input, codes, codebooks, scales) + (
                bias if bias is not None else 0
            )
        case (True, 2, 256, 1, 8):
            from .cuda_kernel import CUDA_FOLDER

            assert (
                input.dtype == torch.float16
            ), f"please load the model with `torch_dtype=torch.float16`, as {input.dtype} is not supported on GPU yet"
            return torch.ops.aqlm_cuda_kernel.code2x8_matmat(input, codes, codebooks, scales) + (
                bias if bias is not None else 0
            )
        case (True, _, _, _, _):
            from .triton_kernel import triton_matmul

            return triton_matmul(input, codes, codebooks, scales, bias)
        case (False, _, 256, 1, _):
            from .numba_kernel import numba_gemm_lut

            return numba_gemm_lut(input, codes, codebooks, scales, bias)
        case _:
            dequantized_weight = _dequantize_weight(
                unpack_int_data(codes, codebooks.shape[1].bit_length() - 1),
                codebooks,
                scales,
            )
            return F.linear(input, dequantized_weight, bias)


def backward_pass_quantized_linear(
    grad_output: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return forward_pass_quantized_linear(
        grad_output.contiguous(),
        codes.transpose(0, 1).contiguous(),
        codebooks.transpose(2, 3).contiguous(),
        scales.transpose(0, 1).transpose(2, 3).contiguous(),
        None,
    )


class QuantizedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.Any,
        input: torch.Tensor,
        codes: torch.IntTensor,
        codebooks: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.save_for_backward(
            input,
            codes,
            codebooks,
            scales,
            bias,
        )
        return forward_pass_quantized_linear(
            input=input,
            codes=codes,
            codebooks=codebooks,
            scales=scales,
            bias=bias,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, codes, codebooks, scales, bias = ctx.saved_tensors
        return (
            backward_pass_quantized_linear(
                grad_output=grad_output,
                codes=codes,
                codebooks=codebooks,
                scales=scales,
                bias=bias,
            ),
            None,
            None,
            None,
            None,
        )
