""" This file serves as the single entry point to efficiently run FinalizedQuantizedLinear layers"""
import functools
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


class FinalizedQuantizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int,
        bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        num_out_groups = out_features // out_group_size
        num_in_groups = in_features // in_group_size
        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = 2**nbits_per_codebook

        # CODES & CODEBOOKS
        self.codebooks = nn.Parameter(
            torch.empty(
                (num_codebooks, self.codebook_size, out_group_size, in_group_size),
                **factory_kwargs,
            ),
            requires_grad=True,
        )  # [num_codebooks, codebook_size, out_group_size, in_group_size]
        self.codes = nn.Parameter(
            torch.empty(
                (num_out_groups, num_in_groups, num_codebooks),
                device=device,
                dtype=get_int_dtype(nbits_per_codebook),
            ),
            requires_grad=False,
        )  #  [num_out_groups, num_in_groups, num_codebooks]

        # SCALES
        self.scales = nn.Parameter(
            torch.empty((num_out_groups, 1, 1, 1), **factory_kwargs), requires_grad=True
        )  #  [num_out_groups, num_in_groups, 1, 1] if scale_nbits > 0 else [num_out_groups, 1, 1, 1]

        # BIAS
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return forward_pass_quantized_linear(input, self.codes, self.codebooks, self.scales, self.bias)


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


@torch.inference_mode()
def unpack_int_data(data: torch.IntTensor, nbits: int) -> torch.IntTensor:
    return data.to(torch.int64) % (2**nbits)


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
    should_script = int(os.environ.get("AQ_USE_JIT", not using_tpu))
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def _dequantize_weight(
    codes: torch.Tensor, codebooks: torch.Tensor, scales: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Decode float weights from quantization codes. Differentiable.
    :param codes: tensor of integer quantization codes, shape [*dims, num_out_groups, num_in_groups, num_codebooks]
    :param codebooks: tensor of vectors for each quantization code, [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, must be broadcastble with [*dims, out_groups, num_in_groups, out_group_size, in_group_size]
    :return: reconstructed weight tensor of shape [*dims, num_in_groups*group_size]
    """
    num_out_groups, num_in_groups, num_codebooks = codes.shape[-3:]
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    out_features = num_out_groups * out_group_size
    in_features = num_in_groups * in_group_size
    codebook_offsets = torch.arange(
        0, num_codebooks * codebook_size, codebook_size, device=codes.device
    )  # shape: [num_codebooks]
    reconstructed_weight_flat = F.embedding_bag(
        codes.flatten(0, -2) + codebook_offsets,
        codebooks.flatten(0, 1).flatten(-2, -1),
        mode="sum",
    )  # [prod(dims) * num_out_groups * num_in_groups, out_group_size * in_group_size]

    reconstructed_weight_groupwise = reconstructed_weight_flat.view(
        list(codes.shape[:-3]) + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )
    if scales is not None:
        reconstructed_weight_groupwise = reconstructed_weight_groupwise.mul(scales)
    return reconstructed_weight_groupwise.swapaxes(-3, -2).reshape(list(codes.shape[:-3]) + [out_features, in_features])


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


@triton.autotune(
    configs=[
        triton.Config({"UNUSED": 1}, num_stages=num_stages, num_warps=num_warps)
        for num_stages in (1, 2, 3, 4, 5)
        for num_warps in (1, 2, 4, 8)
    ],
    key=[
        "in_features",
        "out_features",
        "num_codebooks",
        "codebook_size",
        "out_group_size",
        "in_group_size",
        "num_input_groups",
        "num_input_groups_next_power_of_2",
        "compute_in_fp32",
    ],
)
@triton.jit
def _aqlm_gemv_simple(
    input_vec_ptr,
    output_vec_ptr,
    codes_ptr,
    codebooks_ptr,
    scales_ptr,
    bias_ptr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    num_codebooks: tl.constexpr,
    codebook_size: tl.constexpr,
    out_group_size: tl.constexpr,
    in_group_size: tl.constexpr,
    num_input_groups: tl.constexpr,
    num_input_groups_next_power_of_2: tl.constexpr,
    compute_in_fp32: tl.constexpr,
    UNUSED: tl.constexpr,
):
    # variables ending with "_i" mean "for i-th output unit"
    pid = tl.program_id(axis=0)  # [0, 1, ... {out_features-1}]

    # Stage 1: load input data
    input_vec = tl.load(
        input_vec_ptr
        + tl.arange(0, num_input_groups_next_power_of_2)[:, None, None] * in_group_size
        + tl.arange(0, in_group_size)[None, None, :],
        mask=tl.arange(0, num_input_groups_next_power_of_2)[:, None, None] < num_input_groups,
    )
    # [in_features//in_group_size, 1, group_size]
    # Note: we could simply load input_vec then reshape
    #     input_vec = tl.load(input_vec_ptr + tl.arange(0, in_features))  # [in_features]
    #     input_vec = tl.view(input_vec, [num_input_groups, 1, in_group_size])
    #     , but this does not work because tl.view may reorder elements arbitrarily; see its docstring

    # Stage 2: load integer codes for the active row
    # [in_features // in_group_size, num_codebooks]
    codes_i_ptrs = (
        codes_ptr
        + pid * num_input_groups * num_codebooks
        + tl.arange(0, num_input_groups_next_power_of_2)[:, None] * num_codebooks
        + tl.arange(0, num_codebooks)[None, :]
    )
    codes_i_mask_1d = tl.arange(0, num_input_groups_next_power_of_2) < num_input_groups

    codes_i = tl.load(codes_i_ptrs, mask=codes_i_mask_1d[:, None])  # [in_features//in_group_size, num_codebooks]
    codes_i = codes_i.to(tl.int32)
    codes_i = (codes_i) + (codes_i < 0) * codebook_size  # aka 2 ** nbits_per_codebook
    # ^-- (because codes are int16 tensors that contain uint data)

    # The following alternative does not work:
    #     codes_i = codes_i.to(tl.int32) % codebook_size # aka 2 ** nbits_per_codeboo

    # shift codes_i so that codebooks after 0th point to correct indices in codebooks_ptr
    codes_i += tl.arange(0, num_codebooks)[None, :] * codebook_size  # aka 2 ** nbits_per_codebook
    # ^-- [in_group_size, num_codebooks]

    # Stage 3: convert codes to pointers to every individual (activated) weight in codebooks
    # [in_features // in_group_size, num_codebooks, out_group_size, in_group_size]
    out_group_ix = tl.arange(0, out_group_size)[None, None, :, None]
    in_group_ix = tl.arange(0, in_group_size)[None, None, None, :]
    weight_i_ptrs = (
        codebooks_ptr
        + codes_i[:, :, None, None] * out_group_size * in_group_size
        + out_group_ix * in_group_size
        + in_group_ix
    )

    # Stage 4: reconstruct weights, multiply by inputs and write out
    weights_i = tl.load(weight_i_ptrs, mask=codes_i_mask_1d[:, None, None, None], other=0)
    if compute_in_fp32:
        weights_i = weights_i.to(tl.float32)
        input_vec = input_vec.to(tl.float32)
    # ^-- [in_features // in_group_size, num_codebooks, out_group_size, in_group_size]
    weights_i = tl.sum(weights_i, axis=1)  # sum codebooks as per additive quantization
    # ^-- [in_features // in_group_size, out_group_size, in_group_size]

    if out_group_size == 1:
        scale = tl.load(scales_ptr + pid).to(weights_i.dtype)  # scalar
        output_i = tl.sum(weights_i * input_vec) * scale
        if bias_ptr:
            output_i += tl.load(bias_ptr + pid).to(weights_i.dtype)
        tl.store(output_vec_ptr + pid, output_i.to(input_vec.dtype))
    else:
        output_i = tl.sum(tl.sum(weights_i, axis=2) * input_vec, axis=0)  # [out_group_size]
        output_i *= tl.load(scales_ptr + pid).to(weights_i.dtype)
        if bias_ptr:
            output_i += tl.load(bias_ptr + pid).to(weights_i.dtype)
        tl.store(output_vec_ptr + pid * out_group_size + tl.arange(0, out_group_size), output_i.to(input_vec.dtype))


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def aqlm_gemv_simple(
    input_vec: torch.Tensor,
    codes_i16: torch.ShortTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
):

    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input_vec.shape[1]
    out_features = codes_i16.shape[0] * out_group_size
    num_input_groups = codes_i16.shape[1]
    assert input_vec.ndim == 2 and input_vec.shape[0] == 1, "do reshape; now!"
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebooks.shape[1] < 2**32

    output_vec = torch.empty(1, out_features, device=device, dtype=dtype)
    # 1D launch kernel where each block computes output unit
    grid = lambda META: (out_features // out_group_size,)
    _aqlm_gemv_simple[grid](
        input_vec,
        output_vec,
        codes_i16,
        codebooks,
        scales,
        bias,
        in_features,
        out_features,
        num_codebooks,
        codebook_size,
        out_group_size,
        in_group_size,
        num_input_groups,
        next_power_of_2(num_input_groups),
        compute_in_fp32,
    )

    return output_vec


def aqlm_gemm_stupid(
    input: torch.Tensor,
    codes_i16: torch.ShortTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
):
    device, dtype = codebooks.device, codebooks.dtype
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = input.shape[1]
    out_features = codes_i16.shape[0] * out_group_size
    num_input_groups = codes_i16.shape[1]
    assert input.ndim == 2
    assert scales.shape == (out_features // out_group_size, 1, 1, 1)
    assert in_features % in_group_size == 0
    assert codebooks.shape[1] < 2**32

    output = torch.empty(input.shape[0], out_features, device=device, dtype=dtype)
    for i in range(input.shape[0]):
        # 1D launch kernel where each block computes output unit
        grid = lambda META: (out_features // out_group_size,)
        _aqlm_gemv_simple[grid](
            input[i],
            output[i],
            codes_i16,
            codebooks,
            scales,
            bias,
            in_features,
            out_features,
            num_codebooks,
            codebook_size,
            out_group_size,
            in_group_size,
            num_input_groups,
            next_power_of_2(num_input_groups),
            compute_in_fp32,
        )

    return output


def triton_matmul(
    input: torch.Tensor,
    codes: torch.IntTensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_in_fp32: bool = True,
) -> torch.Tensor:
    input_shape = input.shape
    input = input.reshape(-1, input_shape[-1])

    if input.shape[0] == 1:
        return aqlm_gemv_simple(
            input,
            codes,
            codebooks,
            scales,
            bias,
            compute_in_fp32,
        ).reshape(input_shape[:-1] + (-1,))
    else:
        return aqlm_gemm_stupid(
            input,
            codes,
            codebooks,
            scales,
            bias,
            compute_in_fp32,
        ).reshape(input_shape[:-1] + (-1,))
