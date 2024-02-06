import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.dirname(os.path.abspath(__file__))
CUDA_KERNEL = load(
    name="codebook_cuda",
    sources=[os.path.join(CUDA_FOLDER, "cuda_kernel.cpp"), os.path.join(CUDA_FOLDER, "cuda_kernel.cu")],
)


def cuda_gemm_1x16(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    out_features = codes.shape[0] * out_group_size
    assert input.ndim == 2
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebook_size == 2**16
    assert num_codebooks == 1
    assert codes.dtype == torch.int16
    assert input.dtype == torch.float16 and codebooks.dtype == torch.float16

    output = torch.zeros(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        CUDA_KERNEL.code1x16_matvec(
            codes.squeeze(2), input[i].unsqueeze(-1), output[i].unsqueeze(-1), codebooks.squeeze(0, 2)
        )
    output *= scales.flatten().unsqueeze(0)
    if bias is not None:
        output += bias
    return output.reshape(input_shape[:-1] + (-1,))


def cuda_gemm_2x8(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    out_features = codes.shape[0] * out_group_size
    assert input.ndim == 2
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebook_size == 2**8
    assert num_codebooks == 2
    assert codes.dtype == torch.int8
    assert input.dtype == torch.float16 and codebooks.dtype == torch.float16

    output = torch.zeros(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        CUDA_KERNEL.code2x8_matvec(
            codes.squeeze(2), input[i].unsqueeze(-1), output[i].unsqueeze(-1), codebooks.squeeze(0, 2)
        )
    output *= scales.flatten().unsqueeze(0)
    if bias is not None:
        output += bias
    return output.reshape(input_shape[:-1] + (-1,))
