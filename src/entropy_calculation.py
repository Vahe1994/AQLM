from typing import Optional, Union
import torch
import huffman

IntegerTypeTensor = Union[torch.LongTensor, torch.IntTensor, torch.ShortTensor, torch.ByteTensor]


def _calculate_code_frequencies(codes: torch.LongTensor, num_codebooks: int, nbits_per_codebook: int ):
    code_counts = torch.zeros(num_codebooks, 2**nbits_per_codebook, dtype=torch.int64, device=codes.device)
    for codebook_index in range(num_codebooks):
        code_counts[codebook_index, :] = torch.bincount(
            codes[..., codebook_index].flatten(), minlength=2**nbits_per_codebook)
    return code_counts.float() / code_counts.sum(-1, keepdim=True)


def _calculate_code_entropy(
        codes: IntegerTypeTensor, num_codebooks: int, nbits_per_codebook: int,  eps: float = 1e-20
) -> torch.Tensor:
    """Calculate per-codebook code entropy measured in bits (base-2)"""
    probs = _calculate_code_frequencies(codes, num_codebooks, nbits_per_codebook)
    logprobs = torch.log2(probs.clamp_min(eps))
    return - torch.sum(probs * logprobs, dim=-1)


def _get_huffman_penalties_upper_bound(
        codes: IntegerTypeTensor, regularizer: float, num_codebooks: int, nbits_per_codebook: int
) -> torch.Tensor:
    """Compute log-probability penalties that minimize a linearized upper bound on entropy """
    penalties = torch.empty(num_codebooks, 2 ** nbits_per_codebook, device=codes.device, dtype=torch.float32)
    probs = _calculate_code_frequencies(codes, num_codebooks, nbits_per_codebook)
    
    for codebook_index in range(num_codebooks):
        num_codes = torch.as_tensor(codes[..., codebook_index].numel(), device=probs.device)
        missing_value_length = torch.log2(num_codes).item()
        
        huffman_codes = huffman.codebook([(i, probs[codebook_index, i].item()) for i in range(2 ** nbits_per_codebook)])
        code_lengths = torch.as_tensor([
            len(huffman_codes.get(i, missing_value_length)) for i in range(2 ** nbits_per_codebook)],
            device=probs.device, dtype=torch.float32)
        penalties[codebook_index] = (regularizer / probs.shape[-1]) * code_lengths
    return penalties


def _get_entropy_penalties_upper_bound(codes: IntegerTypeTensor,
                                       regularizer: float,
                                       num_codebooks: int,
                                       nbits_per_codebook: int,
                                       eps: Optional[float] = None
                                       ) -> torch.Tensor:
    """Compute log-probability penalties that minimize a linearized upper bound on entropy """
    probs = _calculate_code_frequencies(codes, num_codebooks, nbits_per_codebook)
    num_codes = torch.as_tensor(codes[..., 0].numel(), device=probs.device)
    if eps is None:
        eps = 1. / num_codes
    logprobs = torch.log2(probs.clamp_min(eps))
    return (- regularizer / probs.shape[-1]) * logprobs
