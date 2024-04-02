from __future__ import annotations

import torch


def _calculate_code_frequencies(codes: torch.LongTensor, codebook_size: int):
    num_codebooks = codes.shape[-1]
    code_counts = torch.zeros(num_codebooks, codebook_size, dtype=torch.int64, device=codes.device)
    for codebook_index in range(num_codebooks):
        code_counts[codebook_index, :] = torch.bincount(
            codes[..., codebook_index].flatten(), minlength=codebook_size)
    return code_counts.float() / code_counts.sum(-1, keepdim=True)


def _calculate_code_entropy(codes: torch.LongTensor, codebook_size: int, eps: float = 1e-20):
    """Calculate per-codebook code entropy measured in bits (base-2)"""
    probs = _calculate_code_frequencies(codes, codebook_size)
    logprobs = torch.log2(probs.clamp_min(eps))
    return - torch.sum(probs * logprobs, dim=-1)


def _get_huffman_penalties_upper_bound(codes: torch.LongTensor, codebook_size: int, regularizer: float):
    """Compute log-probability penalties that minimize a linearized upper bound on entropy """
    import huffman
    num_codebooks = codes.shape[-1]
    penalties = torch.empty(num_codebooks, codebook_size, device=codes.device, dtype=torch.float32)
    freqs = _calculate_code_frequencies(codes, codebook_size)

    for codebook_index in range(num_codebooks):
        num_codes = torch.as_tensor(codes[..., codebook_index].numel(), device=freqs.device)
        missing_value_length = torch.log2(num_codes).item()

        huffman_codes = huffman.codebook([(i, freqs[codebook_index, i].item()) for i in range(codebook_size)])
        code_lengths = torch.as_tensor([
            len(huffman_codes.get(i, missing_value_length)) for i in range(codebook_size)],
            device=freqs.device, dtype=torch.float32)
        penalties[codebook_index] = (regularizer / num_codes) * code_lengths
    return penalties


def _get_entropy_penalties_upper_bound(codes: torch.LongTensor, codebook_size: int, regularizer: float):
    """Compute log-probability penalties that minimize a linearized upper bound on entropy """
    probs = _calculate_code_frequencies(codes, codebook_size)
    num_codes = torch.as_tensor(codes[..., 0].numel(), device=probs.device)
    logprobs = torch.log2(probs.clamp_min(1. / num_codes))
    return (- regularizer / num_codes) * logprobs


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
