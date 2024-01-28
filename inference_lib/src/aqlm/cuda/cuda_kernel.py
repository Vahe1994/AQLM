import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load

CUDA_FOLDER = os.path.dirname(os.path.abspath(__file__))
CUDA_KERNEL = load(
    name="codebook_cuda",
    sources=[os.path.join(CUDA_FOLDER, "codebook_cuda.cpp"), os.path.join(CUDA_FOLDER, "codebook_cuda_kernel.cu")],
)


def cuda_gemm_stupid(
    input: torch.Tensor,  #  [num_inputs, in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    out_features = codes.shape[0] * out_group_size
    num_input_groups = codes.shape[1]
    assert input.ndim == 2
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebooks.shape[1] == 2**16
    assert codes.dtype == torch.int16
    assert input.dtype == torch.float16 and codebooks.dtype == torch.float16

    output = torch.zeros(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        CUDA_KERNEL.code16_matvec(
            codes.squeeze(2), input[i].unsqueeze(-1), output[i].unsqueeze(-1), codebooks.squeeze(0, 2)
        )
    output *= scales.flatten().unsqueeze(0)
    if bias is not None:
        output += bias
    return output

    # codebook = torch.randn((codebook_size, in_group_size), dtype=torch.half, device=DEV)
    # A = torch.randint(codebook_size, (out_features, in_features // in_group_size), dtype=torch.int, device=DEV)
    # A_ref = torch.vstack([codebook[A[i]].flatten().unsqueeze(0) for i in range(M)])
    # A = A.to(torch.int16)
    # B = torch.randn((in_features, 1), dtype=torch.half, device=DEV)
    # C = torch.zeros((out_features, 1), dtype=torch.half, device=DEV)

    # C_ref = torch.matmul(A_ref, B)
    # codebook_cuda.code16_matvec(A, B, C, codebook)


def cuda_matmul(
    input: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    return cuda_gemm_stupid(
        input,
        codes,
        codebooks,
        scales,
        bias,
    ).reshape(input_shape[:-1] + (-1,))
