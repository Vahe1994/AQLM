"""Beam search that minimizes ||Wref - Wq||^2 w.r.t. Wq"""
import random
import time
from typing import Optional, List

import torch
import torch.nn.functional as F

from src.aq_ops import maybe_script


@torch.inference_mode
def beam_search_optimal_codes(
        reference_weight: torch.Tensor,
        codebooks: torch.Tensor,
        prev_codes: torch.Tensor,
        scales: Optional[torch.Tensor],
        beam_size: int,
        stochastic_rounding_tau: float = 0.0,
        chunk_size_bytes: int = 2 ** 32,
        dim_rng: Optional[random.Random] = None,
        verbose: bool = False
) -> torch.Tensor:
    """
    Update codes using beam search to minimize L2 error in code values (regardless of activations)
    :param reference_weight: a target for L2 error, [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param prev_codes: previous-best integer weight codes, shape: [num_out_groups, num_in_groups, num_codebooks]
    :param scales: weight will be multiplied by this factor, shape = [num_out_groups, num_in_groups or 1, 1, 1]
    :param dim_rng: a source of randomness to (optionally) shuffle the order in which the beam search runs
      None = update dimensions and codebooks in their natural order (0, 1, ..., n)
      random.Random(optional_seed) = shuffle dimensions at random, optionally using the specified seed

    :param beam_size: consider up to this many best encoding combinations
    :param stochastic_rounding_tau: if positive, each time the algorithm chooses a code, it will have a probability
        of replacing it with the second-best choice. If the two best codes increase the error by delta1 and delta2,
        then the probability of choosing each code is P_i = delta_i ^ -1/tau / (sum_j_in_choices delta_j ^ -1/tau).
        Note that if there is a code that has zero error, the algorithm will choose allways choose such a code
    :param chunk_size_bytes: process this many candidates at a time; reduce to save memory
    :return: best quantization codes found, same shape as prev_codes

    """
    t0 = time.perf_counter()
    # reshape references, codes and codebooks so they are no longer group-wise
    out_features, in_features = reference_weight.shape
    num_out_groups, num_in_groups, num_codebooks = prev_codes.shape
    _num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    reference_weight = reference_weight.reshape(
        num_out_groups, out_group_size, num_in_groups, in_group_size
    ).permute(0, 2, 1, 3)  # [num_out_groups, num_in_groups, out_group_size, in_group_size]

    if scales is not None:
        reference_weight = reference_weight / scales
        # divide by scales; the resulting problem is equivalent to multiplying dequantized weight

    reference = reference_weight.flatten(2, 3).flatten(0, 1)
    prev_codes = prev_codes.flatten(0, -2)
    codebooks = codebooks.flatten(-2, -1).detach()
    dim_order = list(range(num_codebooks))
    if dim_rng is not None:
        dim_rng.shuffle(dim_order)
    new_codes_groupwise = _beam_search_update_codes_groupwise(
        reference=reference, codebooks=codebooks, codes=prev_codes,
        beam_size=beam_size, stochastic_rounding_tau=stochastic_rounding_tau,
        chunk_size_values=chunk_size_bytes // reference[0, 0].nbytes,
        dim_order=dim_order)
    new_codes_groupwise = new_codes_groupwise.reshape(num_out_groups, num_in_groups, num_codebooks)
    if verbose:
        print(f"Beam search for {reference_weight.numel()} weights finished in {time.perf_counter() - t0} seconds.")
    return new_codes_groupwise


@torch.inference_mode
def update_codes_and_codebooks(
        reference_weight: torch.Tensor,
        codebooks: torch.Tensor,
        prev_codes: torch.Tensor,
        scales: Optional[torch.Tensor],
        *,
        beam_size: int,
        num_iter: int,
        stochastic_rounding_tau: float = 0.0,
        dim_rng: Optional[random.Random] = None,
        chunk_size_bytes: int = 2 ** 32,
) -> (torch.Tensor, torch.IntTensor):
    """
    Update codes and codebooks (but not scales) to minimize L2 error with reference weight (regardless of activations)
    The codes are updated with beam search; codebooks are updated with closed form least squares solver

    :param reference_weight: a target for L2 error, [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param prev_codes: previous-best integer weight codes, shape: [num_out_groups, num_in_groups, num_codebooks]
    :param scales: weight will be multiplied by this factor, shape = [num_out_groups, num_in_groups or 1, 1, 1]
    :param dim_rng: a source of randomness to (optionally) shuffle the order in which the beam search runs
      None = update dimensions and codebooks in their natural order (0, 1, ..., n)
      random.Random(optional_seed) = shuffle dimensions at random, optionally using the specified seed
    :param beam_size: consider up to this many best encoding combinations

    :param num_iter: run beam search and least-squares this many times each

    :param stochastic_rounding_tau: if positive, each time the algorithm chooses a code, it will have a probability
        of replacing it with the second-best choice. If the two best codes increase the error by delta1 and delta2,
        then the probability of choosing each code is P_i = delta_i ^ -1/tau / (sum_j_in_choices delta_j ^ -1/tau).
        Note that if there is a code that has zero error, the algorithm will choose allways choose such a code
    :param chunk_size_bytes: process this many candidates at a time; reduce to save memory
    :return: updated quantization codes found by beam search, same shape as prev_codes

    :returns: a tuple (updated codebooks, updated codes)
    """

    orig_code_shape = prev_codes.shape
    orig_codebook_shape = codebooks.shape

    # reshape references, codes and codebooks so they are no longer group-wise
    out_features, in_features = reference_weight.shape
    num_out_groups, num_in_groups, num_codebooks = prev_codes.shape
    _num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    reference_weight = reference_weight.reshape(
        num_out_groups, out_group_size, num_in_groups, in_group_size
    ).permute(0, 2, 1, 3)  # [num_out_groups, num_in_groups, out_group_size, in_group_size]

    if scales is not None:
        reference_weight = reference_weight / scales
        # divide by scales; the resulting problem is equivalent to multiplying dequantized weight

    reference = reference_weight.flatten(2, 3).flatten(0, 1)
    prev_codes = prev_codes.flatten(0, -2)
    codebooks = codebooks.flatten(-2, -1).detach()
    dim_order = list(range(num_codebooks))
    if dim_rng is not None:
        dim_rng.shuffle(dim_order)

    for _ in range(num_iter):
        new_codes = _beam_search_update_codes_groupwise(
            reference=reference,
            codebooks=codebooks,
            codes=prev_codes,
            beam_size=beam_size,
            stochastic_rounding_tau=stochastic_rounding_tau,
            chunk_size_values=chunk_size_bytes // reference[0, 0].nbytes,
            dim_order=dim_order,
        )

        if (new_codes == prev_codes).all():
            break
        prev_codes = new_codes

        codebooks = _find_optimal_codebooks(
            reference,
            codebooks,
            prev_codes,
        )

    return codebooks.reshape(*orig_codebook_shape), prev_codes.reshape(*orig_code_shape)


@maybe_script
def _beam_search_update_codes_groupwise(
        reference: torch.Tensor,
        codebooks: torch.Tensor,
        codes: torch.Tensor,
        *,
        beam_size: int,
        stochastic_rounding_tau: float,
        chunk_size_values: int,
        dim_order: Optional[List[int]],
) -> torch.Tensor:
    """
    :param reference: [num_groups, group_size]
    :param codes: [num_groups, num_codebooks]
    :param codebooks: [num_codebooks, codebook_size, group_size]
    :returns: [num_groups, num_codebooks]
    """
    if stochastic_rounding_tau > 0:
        assert beam_size >= 2, "with stochastic rounding, we need at least 2 hypotheses to choose from"

    device = reference.device
    num_groups, group_size = reference.shape
    num_codebooks, codebook_size, group_size = codebooks.shape
    codebook_offsets = torch.arange(
        0, num_codebooks * codebook_size, codebook_size, device=device
    )  # shape: [num_codebooks]
    if dim_order is None:
        dim_order = list(range(num_codebooks))

    code_norms_sq = codebooks.square().sum(-1)  # [num_codebooks, codebook_size]
    beam_codes = codes.clone().unsqueeze(1)  # [num_groups, current_beam_size, num_codebooks]
    residue = reference[:, None, :] - F.embedding_bag(
        beam_codes.flatten(0, 1) + codebook_offsets, codebooks.flatten(0, 1), mode='sum'
    ).view(num_groups, 1, group_size)  # shape: [num_groups, current_beam_size, group_size]

    for i, codebook_index in enumerate(dim_order):
        current_beam_size = residue.shape[1]
        is_last_step = i == len(dim_order) - 1
        residue = residue + F.embedding(beam_codes[..., codebook_index], codebooks[codebook_index, ...])
        if beam_size > 1:
            residue_norms_sq = residue.square().sum(-1).unsqueeze(-1)  # [num_groups, current beam size, 1]
        else:
            residue_norms_sq = torch.zeros(0, device=device)  # when doing greedy search, these are const

        flat_best_indices = torch.empty(num_groups, beam_size, device=device, dtype=codes.dtype)
        flat_best_scores = torch.empty(num_groups, beam_size, device=device, dtype=residue.dtype)
        chunk_size_rows = chunk_size_values // (codebook_size * current_beam_size)
        for chunk_start in range(0, num_groups, chunk_size_rows):
            chunk_end = min(chunk_start + chunk_size_rows, num_groups)
            scores = torch.matmul(residue[chunk_start: chunk_end], codebooks[codebook_index].T)
            if beam_size > 1:
                scores = residue_norms_sq[chunk_start: chunk_end] - 2 * scores + code_norms_sq[codebook_index]
            else:
                scores = - 2 * scores + code_norms_sq[codebook_index]  # residue norms are const(j)
            # ^-- [num_groups_chunk, beam_size, codebook_size]
            flat_best_scores[chunk_start: chunk_end], flat_best_indices[chunk_start: chunk_end] = torch.topk(
                scores.flatten(1, 2), k=beam_size,
                largest=False, sorted=is_last_step or beam_size > 1
            )  # [num_groups_chunk, beam_size (+1 if stochastic rounding)]

        arange_num_groups = torch.arange(num_groups, device=device)
        best_hypo_source_ids = flat_best_indices // codebook_size
        best_hypo_codes = flat_best_indices % codebook_size
        beam_codes = beam_codes[arange_num_groups[:, None], best_hypo_source_ids, :]
        beam_codes[:, :, codebook_index] = best_hypo_codes.to(beam_codes.dtype)
        # ^-- [num_groups, beam_size, num_codebooks]

        if not is_last_step:
            residue = residue - F.embedding(beam_codes[..., codebook_index], codebooks[codebook_index, ...])

        if is_last_step and stochastic_rounding_tau > 0:
            assert flat_best_scores.shape[-1] >= 2
            # note: you can also compute erros as torch.norm(residue, dim=-1) after you subtract code embeddings
            errors = flat_best_scores.relu().sqrt()  # non-squared errors
            scores = torch.pow(errors / errors.sum(-1, keepdim=True), -1 / stochastic_rounding_tau)
            # ^-- [num_groups, beam_size]
            keep_prob = scores[:, :-1] / (scores[:, :-1] + scores[:, 1:])  # [num_groups, beam_size - 1]
            keep_prob = torch.where(torch.isinf(scores[:, :-1]), 1.0, keep_prob)  # [num_groups, beam_size - 1]
            best_code_changed = (beam_codes[:, 0, :] != codes).all(dim=-1, keepdim=True)
            keep_best = torch.logical_or(
                best_code_changed.tile(1, 3), torch.less_equal(torch.rand_like(keep_prob), keep_prob)
            )
            beam_codes = torch.where(
                keep_best[:, :, None], beam_codes[:, :-1, :], beam_codes[:, 1:, :])

    return beam_codes[:, 0, :]


def _find_optimal_codebooks(
        reference: torch.Tensor,
        codebooks: torch.Tensor,
        codes: torch.Tensor,
) -> torch.Tensor:
    num_samples = len(reference)
    num_codebooks, codebook_size, group_size = codebooks.shape

    # compute optimal codebooks via linsolve
    codebook_offsets = torch.arange(num_codebooks, device=codes.device) * codebook_size
    code_indicators = torch.sparse_coo_tensor(
        indices=torch.stack([
            torch.arange(num_samples * num_codebooks, device=codes.device) // num_codebooks,
            (codes + codebook_offsets).flatten()
        ], 0),
        values=torch.ones(num_samples * num_codebooks, device=codes.device),
        size=(num_samples, num_codebooks * codebook_size)
    )
    cooc = (code_indicators.T @ code_indicators).coalesce()
    rhs = (code_indicators.T @ reference)

    try:
        cooc = cooc.to_dense()
        cooc[torch.arange(len(cooc)), torch.arange(len(cooc))].clamp_min_(1.0)
        optimal_codebooks = (torch.linalg.lstsq(cooc, rhs)
                             ).solution.reshape(num_codebooks, codebook_size, group_size)
    except Exception as e:
        print(f"Linsolve failed with {e}")
        optimal_codebooks = codebooks
    return optimal_codebooks
