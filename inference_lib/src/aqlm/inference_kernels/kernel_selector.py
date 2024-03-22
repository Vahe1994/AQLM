import warnings
from contextlib import contextmanager
from typing import Callable, Optional

import torch


@contextmanager
def optimize_for_training():
    """
    WARNING: `optimize_for_training` is deprecated. The optimization now happens automatically at runtime.
    OBSOLETE: Use this context manager during model initialization (e.g. `.from_pretrained(...)`) to select inference kernels optimized for larger batch sizes
    """
    warnings.warn("`optimize_for_training` is deprecated. The optimization now happens automatically at runtime.")
    try:
        yield
    finally:
        return


def get_forward_pass_kernel(
    codebooks: torch.Tensor,
    optimize_for_training: bool,
) -> Callable[[torch.Tensor, torch.IntTensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    if (optimize_for_training, codebooks.device.type, num_codebooks, codebook_size, out_group_size, in_group_size) == (
        False,
        "cuda",
        1,
        65536,
        1,
        8,
    ):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x16_matmat
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (
        True,
        "cuda",
        1,
        65536,
        1,
        8,
    ):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x16_matmat_dequant
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (False, "cuda", 2, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code2x8_matmat
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (True, "cuda", 2, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code2x8_matmat_dequant
    elif (optimize_for_training, codebooks.device.type, out_group_size) == (False, "cuda", 1):
        from .triton_kernel import triton_matmul

        return triton_matmul
    elif (codebooks.device.type, codebook_size, out_group_size) == ("cpu", 256, 1):
        from .numba_kernel import numba_gemm_lut

        return numba_gemm_lut
    else:
        from .dequantization import dequantize_gemm

        return dequantize_gemm


def get_backward_pass_kernel(
    codebooks: torch.Tensor,
    optimize_for_training: bool,
) -> torch.Tensor:
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    if (optimize_for_training, codebooks.device.type, num_codebooks, codebook_size, out_group_size, in_group_size,) == (
        True,
        "cuda",
        1,
        65536,
        1,
        8,
    ):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code1x16_matmat_dequant_transposed
    elif (
        optimize_for_training,
        codebooks.device.type,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
    ) == (True, "cuda", 2, 256, 1, 8):
        from .cuda_kernel import CUDA_FOLDER

        return torch.ops.aqlm.code2x8_matmat_dequant_transposed
    else:
        forward_pass_kernel = get_forward_pass_kernel(
            codebooks=codebooks.transpose(2, 3), optimize_for_training=optimize_for_training
        )

        def _backward_pass_kernel(
            grad_output: torch.Tensor,  #  [..., in_features]
            codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
            codebooks: torch.Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
            scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
            bias: Optional[torch.Tensor],
        ) -> torch.Tensor:
            return forward_pass_kernel(
                grad_output.contiguous(),
                codes.transpose(0, 1).contiguous(),
                codebooks.transpose(2, 3).contiguous(),
                scales.transpose(0, 1).transpose(2, 3).contiguous(),
                None,
            )

        return _backward_pass_kernel
