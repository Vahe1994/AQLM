""" Core mathematics for Additive Quantization (AQ): initialization, reconstruction and beam search"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import trange

from .beam_search_l2 import beam_search_optimal_codes as beam_search_minimize_weight_mse
from .beam_search_xtx import beam_search_optimal_codes as beam_search_minimize_activation_mse
from .kmeans import find_nearest_cluster, fit_faiss_kmeans, fit_kmeans, fit_kmeans_1d
from .utils import IntCodes, _dequantize_weight, ellipsis, is_signed


class QuantizedLinear(nn.Module):
    def __init__(self, quantized_weight: QuantizedWeight, bias: Optional[nn.Parameter]):
        super().__init__()
        self.out_features, self.in_features = quantized_weight.out_features, quantized_weight.in_features
        self.quantized_weight = quantized_weight
        self.bias = bias
        self.use_checkpoint = False

    def _forward(self, input: torch.Tensor):
        return F.linear(input, self.quantized_weight(), self.bias)

    def forward(self, input: torch.Tensor):
        if getattr(self, "use_checkpoint", False) and torch.is_grad_enabled():
            return checkpoint(
                self._forward, input, use_reentrant=False, preserve_rng_state=False, determinism_check="none"
            )
        return self._forward(input)


class QuantizedWeight(nn.Module):
    EPS = 1e-9

    def __init__(
        self,
        *,
        reference_weight: torch.Tensor,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int = 8,
        codebook_value_nbits: int = 16,
        codebook_value_num_groups: int = 1,
        scale_nbits: int = 0,
        straight_through_gradient: Optional[bool] = None,
        code_dtype: torch.dtype = torch.int32,
        **init_kwargs,
    ):
        super().__init__()
        self.out_features, self.in_features = reference_weight.shape
        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        if nbits_per_codebook > torch.iinfo(code_dtype).bits - is_signed(code_dtype):
            raise ValueError(f"Code dtype cannot store {nbits_per_codebook} bits; please specify code_dtype manually")

        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = codebook_size = 2**nbits_per_codebook
        self.codebook_value_nbits = codebook_value_nbits
        self.codebook_value_num_groups = codebook_value_num_groups
        self.codebook_value_clusters = None

        self.scales = self.scales_clusters = self.scales_indices = None
        if straight_through_gradient is None and scale_nbits > 0:
            straight_through_gradient = scale_nbits >= 6
        self.straight_through_gradient = straight_through_gradient
        self.scale_nbits = scale_nbits

        with torch.no_grad():
            weight_groupwise = reference_weight.reshape(
                self.out_features // out_group_size, out_group_size, self.in_features // in_group_size, in_group_size
            ).swapaxes(
                1, 2
            )  # [num_out_groups, num_in_groups, out_group_size, in_group_size]

            if scale_nbits > 0:
                scales = weight_groupwise.norm(dim=(2, 3), keepdim=True) + self.EPS
            else:
                scales = weight_groupwise.flatten(1, -1).norm(dim=-1).view(-1, 1, 1, 1) + self.EPS
            # shape [num_out_groups, num_in_groups, 1, 1] if scale_nbits > 0 else [num_out_groups, num_in_groups, 1, 1]

            self.scales_are_lossless = scale_nbits == 0 or scale_nbits >= 16 or (2**scale_nbits >= scales.shape[1])
            if self.scales_are_lossless or self.straight_through_gradient:
                # ^-- this checks if scales can be preserved losslessly
                self.scales = nn.Parameter(scales, requires_grad=True)
            else:
                scales_clusters, scales_indices, _ = fit_kmeans_1d(scales.flatten(1, -1), k=2**scale_nbits)
                self.scales_clusters = nn.Parameter(scales_clusters, requires_grad=True)
                self.scales_indices = nn.Parameter(scales_indices, requires_grad=False)

            weight_for_init = (weight_groupwise / scales).swapaxes(1, 2).reshape_as(reference_weight)
            del weight_groupwise

        codes, codebooks = init_aq_kmeans(
            weight_for_init,
            num_codebooks=num_codebooks,
            out_group_size=out_group_size,
            in_group_size=in_group_size,
            codebook_size=self.codebook_size,
            **init_kwargs,
        )

        self.codebooks = nn.Parameter(
            codebooks, requires_grad=True
        )  # [num_codebooks, codebook_size, out_group_size, in_group_size]
        self.codes: Optional[nn.Parameter] = nn.Parameter(
            codes.to(code_dtype), requires_grad=False
        )  # [num_out_groups, num_in_groups, num_codebooks]
        self.codes_storage: Optional[IntCodes] = None  # storage for FSDP compatibility

    def get_codes(self) -> torch.IntTensor:
        """Get a non view to codes, regardless of how codes are stored"""
        assert (self.codes is None) != (self.codes_storage is None), "must have either .codes or storage, but not both"
        codes = self.codes if self.codes is not None else self.codes_storage()
        if torch.iinfo(codes.dtype).bits < 32:
            codes = codes.to(torch.int32)  # cast to int32 to allow indexing if codes are int16 or uint8
        return codes

    def set_codes(self, new_codes: torch.Tensor, selection: Union[slice, ellipsis, torch.Tensor] = ..., **kwargs):
        """Update codes[selection] to new_codes, regardless of their dtype and whether they are wrapped as storage"""
        assert (self.codes is None) != (self.codes_storage is None), "must have either .codes or storage, but not both"
        codes_ptr = self.codes if self.codes is not None else self.codes_storage()
        codes_ptr[selection].copy_(new_codes, **kwargs)

    def wrap_codes_for_fsdp_(self, **kwargs):
        """Make this module compatible with FullyShardedDataParallel; modifies state dict in-place"""
        assert self.codes is not None and self.codes_storage is None
        self.codes_storage, self.codes = IntCodes(self.codes, **kwargs), None

    def unwrap_codes_(self):
        """Undo the effect of wrap_codes_for_fsdp_; modifies state dict in-place"""
        assert self.codes is None and self.codes_storage is not None
        self.codes, self.codes_storage = nn.Parameter(self.codes_storage(), requires_grad=False), None

    def get_codebooks(self) -> torch.Tensor:
        """Get quantization codebooks or reconstruct them from second level quantization (see codebook_values_nbits)"""
        if self.codebook_value_nbits >= 16:
            return self.codebooks
        elif 0 < self.codebook_value_nbits < 16:
            with torch.no_grad():
                codebooks_dimshuffle = (
                    self.codebooks.reshape(
                        self.num_codebooks,
                        self.codebook_value_num_groups,
                        self.codebook_size // self.codebook_value_num_groups,
                        self.out_group_size,
                        self.in_group_size,
                    )
                    .permute(0, 1, 3, 4, 2)
                    .flatten(0, -2)
                )
                self.codebook_value_clusters, _unused, reconstructed_codebooks_dimshuffle = fit_kmeans_1d(
                    codebooks_dimshuffle,
                    k=2**self.codebook_value_nbits,
                    initial_clusters=self.codebook_value_clusters,
                )
                reconstructed_codebooks = (
                    reconstructed_codebooks_dimshuffle.view(
                        self.num_codebooks,
                        self.codebook_value_num_groups,
                        self.out_group_size,
                        self.in_group_size,
                        self.codebook_size // self.codebook_value_num_groups,
                    )
                    .permute(0, 1, 4, 2, 3)
                    .reshape_as(self.codebooks)
                )
            if torch.is_grad_enabled():
                reconstructed_codebooks = reconstructed_codebooks + (self.codebooks - self.codebooks.detach())
            return reconstructed_codebooks
        raise NotImplementedError(f"{self.codebook_value_nbits}-bit codebook values are not supported")

    def get_scales(self) -> torch.Tensor:
        """Get per-channel or per-group quantization scales or reconstruct those scales based on scales_nbits"""
        if self.scale_nbits == 0 or self.scales_are_lossless:
            return self.scales  # scales are not quantized or the quantization is lossless
        elif self.straight_through_gradient:
            with torch.no_grad():
                self.scales_clusters, _, dequantized_scales = fit_kmeans_1d(
                    self.scales.flatten(1, -1), k=2**self.scale_nbits, initial_clusters=self.scales_clusters
                )
                dequantized_scales = dequantized_scales.reshape_as(self.scales)
            if torch.is_grad_enabled() and self.scales.requires_grad:
                dequantized_scales = dequantized_scales + (self.scales - self.scales.detach())
            return dequantized_scales
        else:  # train scale codebook only
            return self.scales_clusters.gather(1, self.scales_indices)[:, :, None, None]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.out_features, self.in_features

    def forward(self, selection: Union[slice, ellipsis, torch.Tensor] = ...):
        """
        Differentably reconstruct the weight (or parts thereof) from compressed components
        :param selection: By default, reconstruct the entire weight. If selection is specified, this method will instead
            reconstruct a portion of weight for the corresponding output dimensions (used for parallelism).
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )

        """
        weight = _dequantize_weight(self.get_codes()[selection], self.get_codebooks(), self.get_scales()[selection])
        return weight

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        *,
        XTX: Optional[torch.Tensor] = None,
        reference_weight: torch.Tensor,
        selection: Union[slice, ellipsis, torch.LongTensor] = ...,
        **kwargs,
    ) -> torch:
        """
        Update own codes in-place via beam search so as to minimize squared errors. Return the updated codes.
        :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
        :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
          - if XTX is divided by dataset size, this function will return *mean* squared error
          - if XTX is not specified, this function minimizes squared error between weights, as if XTX was identity

        :note: if selection is specified, reference_weight must instead be [num_selected_out_features, in_features]
        :param selection:  By default, this function updates all codes, If selection specified, it will instead
            update only the codes for a portion of output dimensions (used for parallelism).
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        :param beam_size: consider up to this many best encoding combinations (this param is passed through via kwargs)
        :param kwargs: any additional keyword arguments are forwarded to beam_search_optimal_codes function
        :returns: the updated codes, in the same shape as self.get_codes()[selection]
        """
        codebooks = self.get_codebooks()
        prev_codes = self.get_codes()[selection]
        scales = self.get_scales()[selection]
        if XTX is not None:
            new_codes = beam_search_minimize_activation_mse(
                XTX=XTX,
                reference_weight=reference_weight,
                codebooks=codebooks,
                prev_codes=prev_codes,
                scales=scales,
                **kwargs,
            )
        else:
            new_codes = beam_search_minimize_weight_mse(
                reference_weight=reference_weight, codebooks=codebooks, prev_codes=prev_codes, scales=scales, **kwargs
            )
        self.set_codes(new_codes, selection)
        return new_codes

    def estimate_nbits_per_parameter(self) -> float:
        """Calculate the effective number of bits per original matrix parameters"""
        num_parameters = self.out_features * self.in_features
        group_size = self.out_group_size * self.in_group_size
        num_out_groups = self.out_features // self.out_group_size
        num_in_groups = self.in_features // self.in_group_size

        matrix_store = num_parameters // group_size * self.num_codebooks * self.nbits_per_codebook

        codebooks_store = self.num_codebooks * self.codebook_size * group_size * self.codebook_value_nbits
        if self.codebook_value_nbits < 16:
            codebooks_store += (
                2**self.codebook_value_nbits * self.num_codebooks * self.codebook_value_num_groups * group_size * 16
            )

        if self.scale_nbits >= 16 or 2**self.scale_nbits >= num_in_groups:  # group-wise scales in 16 bit
            scale_store = self.scale_nbits * num_out_groups * num_in_groups
        elif 0 < self.scale_nbits < 16:  # use scale quantization codebooks
            scale_store = self.scale_nbits * num_out_groups * num_in_groups
            scale_store += num_out_groups * 2**self.scale_nbits * 16
        elif self.scale_nbits == 0:  # no group-wise scales; use global 1d scales instead
            scale_store = num_out_groups * 16
        else:
            assert False

        return (matrix_store + codebooks_store + scale_store) / num_parameters

    def extra_repr(self) -> str:
        return f"{self.out_features=}, {self.in_features=}, bits_per_parameter={self.estimate_nbits_per_parameter()}"


@torch.no_grad()
def init_aq_kmeans(
    reference_weight: torch.Tensor,
    *,
    num_codebooks: int,
    out_group_size: int,
    in_group_size: int,
    codebook_size: int,
    verbose: bool = False,
    use_faiss: bool = False,
    max_points_per_centroid: Optional[int] = None,
    max_iter: int = 1000,
    devices: Optional[List[torch.device]] = None,
    **kwargs,
):
    """
    Create initial codes and codebooks using residual K-means clustering of weights
    :params reference_weight, num_codebooks, out_group_size, in_group_size, nbits, verbose: same as in QuantizedWeight
    :params use_faiss  whether to use faiss implementation of kmeans or pure torch
    :params max_point_per_centorid maximum data point per cluster
    :param kwargs: any additional params are forwarded to fit_kmeans
    """
    out_features, in_features = reference_weight.shape
    num_out_groups = out_features // out_group_size
    num_in_groups = in_features // in_group_size
    weight_residue = (
        reference_weight.reshape(num_out_groups, out_group_size, num_in_groups, in_group_size)
        .clone()
        .swapaxes(-3, -2)
        .reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
    )
    codebooks = []
    codes = []

    if max_points_per_centroid is not None:
        print("Clustering:", max_points_per_centroid * codebook_size, "points from", weight_residue.shape[0])

    for _ in trange(num_codebooks, desc="initializing with kmeans") if verbose else range(num_codebooks):
        if use_faiss:
            codebook_i, codes_i, reconstructed_weight_i = fit_faiss_kmeans(
                weight_residue,
                k=codebook_size,
                max_iter=max_iter,
                gpu=(weight_residue.device.type == "cuda"),
                max_points_per_centroid=max_points_per_centroid,
            )
        else:
            chosen_ids = None
            if max_points_per_centroid is not None:
                chosen_ids = torch.randperm(weight_residue.shape[0], device=weight_residue.device)[
                    : max_points_per_centroid * codebook_size
                ]
            codebook_i, _, _ = fit_kmeans(
                weight_residue if chosen_ids is None else weight_residue[chosen_ids, :],
                k=codebook_size,
                max_iter=max_iter,
                devices=devices,
                **kwargs,
            )
            codes_i, reconstructed_weight_i = find_nearest_cluster(weight_residue, codebook_i, devices=devices)

        codes_i = codes_i.reshape(num_out_groups, num_in_groups, 1)
        codebook_i = codebook_i.reshape(1, codebook_size, out_group_size, in_group_size)
        weight_residue -= reconstructed_weight_i
        codes.append(codes_i)
        codebooks.append(codebook_i)
        del reconstructed_weight_i
    codebooks = torch.cat(codebooks, dim=0)
    codes = torch.cat(codes, dim=-1)
    return codes, codebooks
