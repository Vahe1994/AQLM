"""Common utility functions for additive quantization"""
from __future__ import annotations

import contextlib
import functools
import os
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Union

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F

ellipsis = type(...)


def get_mean_nbits_by_codebook(codes: torch.IntTensor, huffman_group_size: int = 2):
    """
    Calculates average code length in codebooks.
    :param codes: codebook codes
    :param huffman_group_size: huffman compresssion dimension count
    """
    import huffman

    _, codebook_size, num_codebooks = codes.shape
    flat_codes_by_codebook = codes.permute(2, 0, 1).flatten(1, 2)
    code_counts = torch.zeros(
        num_codebooks, codebook_size, device=flat_codes_by_codebook.device, dtype=flat_codes_by_codebook.dtype
    ).scatter_add(
        -1, flat_codes_by_codebook, torch.ones_like(flat_codes_by_codebook)
    )  # shape: [current beam_size, num_codebooks, codebook_size], initial beam_size = 1
    code_probs = code_counts / code_counts.sum(dim=-1, keepdim=True).float()
    code_probs = code_probs.cpu().numpy()
    assert num_codebooks % huffman_group_size == 0

    mean_code_lengths = []
    for group_index in range(num_codebooks // huffman_group_size):
        group_code_probs = {(): 1}

        for codebook_index in range(group_index * huffman_group_size, (group_index + 1) * huffman_group_size):
            new_group_code_probs = {}
            for group, group_prob in group_code_probs.items():
                for code, code_prob in tuple(enumerate(code_probs[codebook_index])):
                    new_group_code_probs[group + (code,)] = group_prob * code_prob
            group_code_probs = new_group_code_probs

        huffman_codebook_i = huffman.codebook(list(group_code_probs.items()))
        codebook_mean_code_length_i = sum(
            len(huffman_codebook_i[code]) * prob for code, prob in group_code_probs.items()
        )
        mean_code_lengths.append(codebook_mean_code_length_i)
    return mean_code_lengths


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
        codes.flatten(0, -2) + codebook_offsets, codebooks.flatten(0, 1).flatten(-2, -1), mode="sum"
    )  # [prod(dims) * num_out_groups * num_in_groups, out_group_size * in_group_size]

    reconstructed_weight_groupwise = reconstructed_weight_flat.view(
        list(codes.shape[:-3]) + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )
    if scales is not None:
        reconstructed_weight_groupwise = reconstructed_weight_groupwise.mul(scales)
    return reconstructed_weight_groupwise.swapaxes(-3, -2).reshape(list(codes.shape[:-3]) + [out_features, in_features])


@contextlib.contextmanager
def using_tf32(enabled: bool):
    was_cudnn = torch.backends.cudnn.allow_tf32
    was_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    yield
    torch.backends.cudnn.allow_tf32 = was_cudnn
    torch.backends.cuda.matmul.allow_tf32 = was_matmul


def iterate_minibatches(
    *tensors: torch.Tensor,
    batch_size: int,
    allow_incomplete: bool = True,
    device: Optional[torch.device] = None,
    callback: Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] = lambda x: x,
) -> Iterator[Sequence[torch.Tensor]]:
    """
    Samples data points *forever*, in random order, with less overhead than DataLoader;
    Adapted from https://github.com/stanis-morozov/unq/blob/master/lib/utils.py
    probably implemented over9000 times in transformers, torch, etc
    :param tensors: one or more tensors with the same 0-th dimension
    :param batch_size: sample this many points with each yield
    :param allow_incomplete: if True and if dataset size is not divisible by batch size, the last batch
        may have less than :batch_size: samples to cover the entire dataset. If False, the last batch is dropped
    :param callback: optional function to be called on each batch of tensors before it is yielded to the user
    :returns: generates a tuple of minibatches from each tensor, same length as input *tensors
        If a batch contains only one tensor, this function will yield a tensor (and not a tuple/list with one tensor)
    """
    num_samples = len(tensors[0])
    assert all(len(x) == num_samples for x in tensors)
    indices = torch.randperm(num_samples, device=tensors[0].device)
    while True:
        prev_batch = None
        for batch_start in range(0, len(indices), batch_size):
            if not allow_incomplete and batch_start + batch_size > len(indices):
                break
            batch_ix = indices[batch_start : batch_start + batch_size]
            batch = callback(tuple(tensor[batch_ix].to(device, non_blocking=True) for tensor in tensors))
            if prev_batch is not None:
                yield prev_batch
            prev_batch = batch if isinstance(batch, (list, tuple)) and len(tensors) > 1 else batch[0]
            del batch
        yield prev_batch


def maybe_get_0th_element(x: Union[Any, Sequence[Any]]) -> Any:
    """
    Return first element if input is Sequence, otherwise return input
    """
    if isinstance(x, Sequence):
        return x[0]
    return x


def _extract_into_tensor(tensor_list: List[torch.Tensor], indices: Iterable[int], device=None, dtype=None):
    extracted_items = [maybe_get_0th_element(tensor_list[i]) for i in indices]
    return torch.cat(extracted_items, dim=0).to(device=device, dtype=dtype)


class IntCodes(nn.Module):
    """
    A storage for integer codes that makes them compatible with FullyShardedDataParallel,
    see https://github.com/pytorch/pytorch/issues/123528 for details
    """

    def __init__(self, codes: torch.tensor, storage_dtype: torch.dtype = torch.float64):
        super().__init__()
        assert torch.finfo(storage_dtype).bits % torch.iinfo(codes.dtype).bits == 0
        self.dtype, self.shape, self.numel = codes.dtype, codes.shape, codes.numel()
        size_ratio = torch.finfo(storage_dtype).bits // torch.iinfo(codes.dtype).bits
        codes = F.pad(codes.flatten().clone(), pad=[0, -codes.numel() % size_ratio])
        assert len(codes.untyped_storage()) == codes.nbytes  # no offset / stride / tail
        self.storage_dtype = storage_dtype
        self.data = nn.Parameter(
            torch.as_tensor(codes.untyped_storage(), device=codes.device, dtype=storage_dtype), requires_grad=False
        )

    def forward(self):
        assert self.data.is_contiguous() and self.data.dtype == self.storage_dtype
        byte_offset = self.data.storage_offset() * self.data.nbytes // self.data.numel()
        return torch.as_tensor(
            self.data.untyped_storage()[byte_offset : byte_offset + self.data.nbytes],
            device=self.data.device,
            dtype=self.dtype,
        )[: self.numel].view(*self.shape)


@contextlib.contextmanager
def one_rank_at_a_time(local: bool = False, group_size: int = 1):
    """
    In distributed setting, let only group_size processes enter at a time
    :param local: if True, the limit is enforced within each host, i.e. distributed hosts can act concurrently
    :param group_size: if more than one is specified,
    """
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE" if local else "WORLD_SIZE", 0)) if distributed else 1
    if distributed:
        torch.distributed.barrier()
    for current_group_index in range(world_size // group_size):
        if current_group_index == rank // group_size:
            yield
        if distributed:
            torch.distributed.barrier()


@contextlib.contextmanager
def master_rank_first(local: bool, master_rank: int = 0):
    distributed = torch.distributed.is_initialized()
    rank = int(os.environ.get("LOCAL_RANK" if local else "RANK", 0)) if distributed else 0
    if distributed and rank != master_rank:
        torch.distributed.barrier()
    yield
    if distributed and rank == master_rank:
        torch.distributed.barrier()


def is_signed(dtype: torch.dtype) -> bool:
    """Return True iff an integer dtype is signed"""
    try:
        return dtype.is_signed
    except RuntimeError:  # see https://github.com/pytorch/pytorch/issues/125124
        if dtype.is_floating_point:
            return torch.finfo(dtype).min < 0
        else:
            return torch.iinfo(dtype).min < 0 and torch.iinfo(dtype).max > 0
