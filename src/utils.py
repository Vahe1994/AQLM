from __future__ import annotations

import contextlib
import functools
import os
from typing import Callable, Iterator, Optional, Sequence

import torch

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
