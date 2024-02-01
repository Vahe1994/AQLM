""" Core mathematics for Additive Quantization (AQ): initialization, reconstruction and beam search"""
import torch
import torch.nn as nn
from aqlm.inference_kernels import forward_pass_quantized_linear
from aqlm.utils import get_int_dtype

import numpy as np
import numba
numba.set_num_threads(8)
COMPILED_KERNELS = {}


class QuantizedLinear(nn.Module):
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
            torch.rand((num_codebooks, self.codebook_size, out_group_size, in_group_size), **factory_kwargs) / 1e5,
            requires_grad=True,
        )  # [num_codebooks, codebook_size, out_group_size, in_group_size]
        self.codes = nn.Parameter(
            torch.zeros(
                (num_out_groups, num_in_groups, num_codebooks), device=device, dtype=get_int_dtype(nbits_per_codebook)
            ),
            requires_grad=False,
        )  #  [num_out_groups, num_in_groups, num_codebooks]

        # SCALES
        self.scales = nn.Parameter(
            torch.ones((num_out_groups, 1, 1, 1), **factory_kwargs), requires_grad=True
        )  #  [num_out_groups, 1, 1, 1]

        # BIAS
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
            
        self.numpied_codes = None
        self.numpied_codebooks = None
        self.numpied_scales = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.numpied_codes is None:
            self.numpied_codes = torch.permute(self.codes.data, (1, 0, 2)).contiguous().view(torch.uint8).numpy()
            self.numpied_codebooks = self.codebooks.data.numpy()
            self.numpied_scales = self.scales.data.numpy()
        
        in_group_size, out_features, in_features, num_codebooks, codebook_size, num_input_groups = self.in_group_size, self.out_features, self.in_features, self.num_codebooks, self.codebook_size, self.in_features // self.in_group_size
        kernel_key = (in_group_size, out_features, in_features, num_codebooks)
        if kernel_key not in COMPILED_KERNELS:
            print(f"Compiling {kernel_key=}")

            @numba.njit(parallel=True)
            def numba_gemv_lut_(x, codes_alt, codebooks, scales):
                lut = x.reshape(-1, in_group_size) @ codebooks.reshape(-1, in_group_size).T
                lut = lut.reshape(-1, num_codebooks, codebook_size)

                output_vec = np.zeros(out_features, dtype=x.dtype)
                for j in numba.prange(num_input_groups):
                    for i in range(out_features):
                        for c in range(num_codebooks):
                            output_vec[i] += lut[j, c, codes_alt[j, i, c]]
                output_vec *= scales.flatten()
                return output_vec
            
            COMPILED_KERNELS[kernel_key] = numba_gemv_lut_
        compiled_kernel = COMPILED_KERNELS[kernel_key]
            
        return torch.from_numpy(
            compiled_kernel(input.reshape(-1, self.in_features).numpy(), self.numpied_codes, self.numpied_codebooks, self.numpied_scales).reshape(input.shape[:-1] + (-1,))
        )
