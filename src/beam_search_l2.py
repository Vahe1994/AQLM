"""Beam search that minimizes ||Wref - Wq||^2 w.r.t. Wq"""
import math
import random
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.utils import _dequantize_weight, maybe_script


@torch.inference_mode
def beam_search_optimal_codes(
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    prev_codes: torch.Tensor,
    scales: Optional[torch.Tensor],
    beam_size: int,
    stochastic_rounding_tau: float = 0.0,
    chunk_size_bytes: int = 2**32,
    dim_rng: Optional[random.Random] = None,
    force_update: bool = False,
    max_update_fraction: float = 1.0,
    code_selection_temperature: float = 0,
    trust_ratio: Optional[float] = None,
) -> torch.Tensor:
    """
    Update codes using beam search to minimize L2 error in code values (regardless of activations)
    :param reference_weight: a target for L2 error, [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param prev_codes: previous-best integer weight codes, shape: [num_output_groups, num_input_groups, num_codebooks]
    :param scales: weight will be multiplied by this factor, shape = [num_output_groups, num_input_groups or 1, 1, 1]
    :param dim_rng: a source of randomness to (optionally) shuffle the order in which the beam search runs
      None = update dimensions and codebooks in their natural order (0, 1, ..., n)
      random.Random(optional_seed) = shuffle dimensions at random, optionally using the specified seed

    :param beam_size: consider up to this many best encoding combinations
    :param stochastic_rounding_tau: if positive, each time the algorithm chooses a code, it will have a probability
        of replacing it with the second-best choice. If the two best codes increase the error by delta1 and delta2,
        then the probability of choosing each code is P_i = delta_i ^ -1/tau / (sum_j_in_choices delta_j ^ -1/tau).
        Note that if there is a code that has zero error, the algorithm will choose allways choose such a code
    :param chunk_size_bytes: process this many candidates at a time; reduce to save memory
    :param force_update: if True, the algorithm will force codes to change even if code is optimal in terms
     of mean squared error. By default, the algorithm forces *all* weights to update this way, which may change weights
     too much. To limit the numer of updated weights, set max_code_change and trust_ratio.
    :param max_update_fraction: the maximum portion of discrete code groups that *can* be updated;
        By default, all codes can be updated. If < 1, only this portion of all code groups is allowed to update.
        The algorithm selects the codes for update based on the difference between de-quantized and reference_weight.
        If there are multiple codebooks, changing any one code responsible for the group counts as code group changed.
        Note that small max_code_change also speeds up computation since not all codes need beam search.
        If the number of weights do not divide evenly, the algoritm will round the number of updates up.
    :param code_selection_temperature: only used if max_code_change > 1; by default, prioritize updating the codes with
        the largest delta = ||(reference_weight - quantized_weight) * mask_only_weights_that_depend_on_this_code||_2 .
        If temperature > 0, the updated codes are instead *sampled* at random, proportionally to delta^(1/temperature) .
    :param trust_ratio: if not None, the algorithm only admits code changes as long as they do not change too much.
        Formally, ||new_quantized_weight - prev_quantized_weight|| / ||prev_quantized_weight|| <= trust_ratio
        If this is not true, the algorithm will reset some of the new quantized weights to their old values until the
        constraint becomes satisfied. The algorithm still prioritizes changes to weights with largest delta (see above).
        If code_change_temperature > 0, the algorithm instead samples which weights to change with the same probability.
        The algorithm will always allow changing exactly *one* code in excess of trust ratio to ensure that at least
        one weight is updated. If both this and max_code_change is set, both these constraints are enforced.
    :return: the best quantization codes found within constraints, same shape as prev_codes

    """
    assert 0 < max_update_fraction <= 1 and (trust_ratio is None or trust_ratio > 0)
    # reshape references, codes and codebooks so they are no longer group-wise
    num_output_groups, num_input_groups, num_codebooks = prev_codes.shape
    _num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape

    flat_unscaled_reference = reference_weight.reshape(
        num_output_groups, out_group_size, num_input_groups, in_group_size
    ).permute(
        0, 2, 1, 3
    )  # [num_output_groups, num_input_groups, out_group_size, in_group_size]
    if scales is not None:
        flat_unscaled_reference = flat_unscaled_reference / scales
        # divide by scales; the resulting problem is equivalent to multiplying dequantized weight
    flat_unscaled_reference = flat_unscaled_reference.flatten(2, 3).flatten(0, 1)
    flat_prev_codes = prev_codes.flatten(0, -2)
    flat_codebooks = codebooks.flatten(-2, -1).detach()
    dim_order = list(range(num_codebooks))
    if dim_rng is not None:
        dim_rng.shuffle(dim_order)

    def _update_flat_codes(_flat_reference, _flat_codes):
        """update _flat_codes [num_groups, num_codebooks] to approximate _flat_reference [num_groups, group_size]"""
        if num_codebooks == 1 and beam_size == 1 and stochastic_rounding_tau == 0 and not force_update:
            # a faster algorithm for a special case of one codebook
            return _greedy_find_best_codes(
                reference=_flat_reference,
                codebook=flat_codebooks[0],
                chunk_size_values=chunk_size_bytes // _flat_reference[0, 0].nbytes,
                code_dtype=prev_codes.dtype,
            )
        else:
            return _beam_search_update_codes_groupwise(
                reference=_flat_reference,
                codebooks=flat_codebooks,
                codes=_flat_codes,
                beam_size=beam_size,
                stochastic_rounding_tau=stochastic_rounding_tau,
                force_update=force_update,
                chunk_size_values=chunk_size_bytes // _flat_reference[0, 0].nbytes,
                dim_order=dim_order,
            )

    def _groupwise_squared_norms(delta: torch.Tensor):
        """
        Given a matrix delta [out_features, in_features], compute a tensor [num_output_groups, num_input_groups] that
        contains the squared sum of elements of delta from each tile of (out_group_size, in_group_size) values.
        """
        return (
            delta.view(delta.shape[0] // out_group_size, out_group_size, delta.shape[1] // in_group_size, in_group_size)
            .square()
            .sum(dim=(1, 3))
        )

    flat_indices_to_update = prev_dequantized_weight = None
    if max_update_fraction < 1 or trust_ratio is not None:
        # precompute ordered code indices to be used for constraints on the number of updates
        prev_dequantized_weight = _dequantize_weight(prev_codes, codebooks, scales)
        num_codes_to_update = int(math.ceil(max_update_fraction * num_output_groups * num_input_groups))
        difference_with_reference_squared_norms = _groupwise_squared_norms(reference_weight - prev_dequantized_weight)
        # ^-- [num_output_groups, num_input_groups]
        if code_selection_temperature > 0:
            flat_indices_to_update = torch.pow(
                difference_with_reference_squared_norms.flatten(),
                0.5 / code_selection_temperature,
                # note: temperature is multuplied by 0.5 because sampling is proportional to norms without square
            ).multinomial(num_samples=num_codes_to_update, replacement=False)
        else:
            flat_indices_to_update = torch.topk(
                difference_with_reference_squared_norms.flatten(), k=num_codes_to_update, largest=True, sorted=True
            ).indices

    if max_update_fraction == 1:
        flat_new_codes = _update_flat_codes(flat_unscaled_reference, flat_prev_codes)
    else:
        flat_new_codes = flat_prev_codes.index_put(  # note: this is an out-of-place op that does not modify prev codes
            (flat_indices_to_update[:, None], torch.arange(num_codebooks, device=codebooks.device)[None, :]),
            _update_flat_codes(
                flat_unscaled_reference[flat_indices_to_update], flat_prev_codes[flat_indices_to_update]
            ),
        )

    if trust_ratio is not None:
        assert isinstance(flat_indices_to_update, torch.Tensor) and isinstance(prev_dequantized_weight, torch.Tensor)
        new_dequantized_weight = _dequantize_weight(flat_new_codes.view_as(prev_codes), codebooks, scales)
        weight_change_squared_norms = _groupwise_squared_norms(new_dequantized_weight - prev_dequantized_weight)
        # ^-- shape: [num_output_groups, num_input_groups]

        flat_ordered_weight_change_squared_norms = weight_change_squared_norms.flatten()[flat_indices_to_update]
        flat_ordered_cumulative_norms = flat_ordered_weight_change_squared_norms.cumsum(0).sqrt()
        # [num_codes_to_update]

        num_codes_selected = 1 + torch.searchsorted(
            flat_ordered_cumulative_norms, trust_ratio * prev_dequantized_weight.norm(), side="left"
        )
        truncated_flat_indices_to_update = flat_indices_to_update[:num_codes_selected]  # sorted most to least important
        flat_new_codes = flat_prev_codes.index_put(  # <-- note: this is an out-of-place operation
            (truncated_flat_indices_to_update[:, None], torch.arange(num_codebooks, device=codebooks.device)[None, :]),
            flat_new_codes[truncated_flat_indices_to_update],
        )
    return flat_new_codes.view_as(prev_codes)


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
    force_update: bool,
) -> torch.Tensor:
    """
    :param reference: [num_groups, group_size]
    :param codes: [num_groups, num_codebooks]
    :param codebooks: [num_codebooks, codebook_size, group_size]
    :returns: [num_groups, num_codebooks]
    """
    if stochastic_rounding_tau > 0:
        assert beam_size >= 2, "with stochastic rounding, we need at least 2 hypotheses to choose from"

    prev_codes = codes
    device = reference.device
    num_groups, group_size = reference.shape
    num_codebooks, codebook_size, group_size = codebooks.shape
    codebook_offsets = torch.arange(0, num_codebooks * codebook_size, codebook_size, device=device)  # [num_codebooks]
    original_dequantized_vectors = F.embedding_bag(
        codes + codebook_offsets, codebooks.flatten(0, 1), mode="sum"
    )  # [num_groups, group_size]
    if dim_order is None:
        dim_order = list(range(num_codebooks))

    code_norms_sq = codebooks.square().sum(-1)  # [num_codebooks, codebook_size]
    beam_codes = codes.clone().unsqueeze(1)  # [num_groups, current_beam_size, num_codebooks]
    residue = (reference - original_dequantized_vectors).view(num_groups, 1, group_size)
    # shape: [num_groups, current_beam_size, group_size]
    direction = residue.clone().view(num_groups, group_size) if force_update else torch.empty(0)

    for i, codebook_index in enumerate(dim_order):
        current_beam_size = residue.shape[1]
        is_last_step = i == len(dim_order) - 1
        # ^-- [num_groups, current_beam_size, group_size]
        residue = residue + F.embedding(beam_codes[..., codebook_index], codebooks[codebook_index, ...])
        if beam_size > 1 or stochastic_rounding_tau > 0:
            residue_norms_sq = residue.square().sum(-1).unsqueeze(-1)  # [num_groups, current beam size, 1]
        else:
            residue_norms_sq = torch.empty(0, device=device)  # when doing greedy search, these are const

        if not is_last_step:
            target_num_candidates = beam_size + int(stochastic_rounding_tau > 0)
        else:
            target_num_candidates = 2 if stochastic_rounding_tau > 0 or force_update else 1

        flat_best_indices = torch.empty(num_groups, target_num_candidates, device=device, dtype=codes.dtype)
        chunk_size_rows = chunk_size_values // (codebook_size * current_beam_size) // 32
        for chunk_start in range(0, num_groups, chunk_size_rows):
            chunk_end = min(chunk_start + chunk_size_rows, num_groups)
            scores = torch.matmul(residue[chunk_start:chunk_end], codebooks[codebook_index].T)
            if beam_size > 1 or stochastic_rounding_tau > 0:
                scores = residue_norms_sq[chunk_start:chunk_end] - 2 * scores + code_norms_sq[codebook_index]
            else:
                scores = -2 * scores + code_norms_sq[codebook_index]  # residue norms are const(j)
            # ^-- [num_groups_chunk, beam_size, codebook_size]

            flat_best_losses_chunk, flat_best_indices_chunk = torch.topk(
                scores.flatten(1, 2),
                k=target_num_candidates,
                largest=False,
                sorted=is_last_step or beam_size > 1 or stochastic_rounding_tau > 0,
            )  # [num_groups_chunk, target_num_candidates]

            if stochastic_rounding_tau > 0:
                errors = flat_best_losses_chunk.relu().sqrt()  # non-squared errors
                scores = torch.pow(errors / errors.sum(-1, keepdim=True), -1 / stochastic_rounding_tau)
                # ^-- [num_groups_chunk, beam_size + 1]
                keep_prob = scores[:, :-1] / (scores[:, :-1] + scores[:, 1:])  # [num_groups, k_best]
                keep_prob = torch.where(torch.isinf(scores[:, :-1]), 1.0, keep_prob)
                keep = torch.less_equal(torch.rand_like(keep_prob), keep_prob)
                flat_best_indices_chunk = torch.where(
                    keep, flat_best_indices_chunk[:, :-1], flat_best_indices_chunk[:, 1:]
                )

            flat_best_indices[chunk_start:chunk_end] = flat_best_indices_chunk

        arange_num_groups = torch.arange(num_groups, device=device)
        best_hypo_source_ids = flat_best_indices // codebook_size
        best_hypo_codes = flat_best_indices % codebook_size
        beam_codes = beam_codes[arange_num_groups[:, None], best_hypo_source_ids, :]
        beam_codes[:, :, codebook_index] = best_hypo_codes.to(beam_codes.dtype)
        # ^-- [num_groups, beam_size, num_codebooks]

        if not is_last_step:
            residue = residue - F.embedding(beam_codes[..., codebook_index], codebooks[codebook_index, ...])

    if force_update:
        assert beam_codes.shape[1] == 2
        best_codes = beam_codes[:, 0, :]
        second_best_codes = beam_codes[:, 1, :]
        best_code_changed = torch.ne(best_codes, prev_codes).any(dim=-1)
        return torch.where(best_code_changed.unsqueeze(-1), best_codes, second_best_codes)
    else:
        return beam_codes[:, 0, :]


@maybe_script
def _greedy_find_best_codes(
    reference: torch.Tensor, codebook: torch.Tensor, chunk_size_values: int, code_dtype: torch.dtype
) -> torch.Tensor:
    """
    :param reference: [num_groups, group_size]
    :param codebook: [codebook_size, group_size]
    :param chunk_size_values: how many values can be materialized in memory simultaneously
    :parma code_dtype the dtype of optimal codes returned by this function
    :returns: codes [num_groups, 1]
    """
    codebook_t = codebook.T.contiguous()
    chunk_size = chunk_size_values // len(codebook)
    codebook_norms_sq = codebook.square().sum(dim=-1)
    new_codes = torch.empty((len(reference),), dtype=code_dtype, device=reference.device)
    for chunk_start in range(0, len(reference), chunk_size):
        new_codes[chunk_start : chunk_start + chunk_size] = torch.addmm(
            codebook_norms_sq[None], reference[chunk_start : chunk_start + chunk_size], codebook_t, alpha=-2
        ).argmin(-1)
    return new_codes.unsqueeze(-1)


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
        indices=torch.stack(
            [
                torch.arange(num_samples * num_codebooks, device=codes.device) // num_codebooks,
                (codes + codebook_offsets).flatten(),
            ],
            0,
        ),
        values=torch.ones(num_samples * num_codebooks, device=codes.device),
        size=(num_samples, num_codebooks * codebook_size),
    )
    cooc = (code_indicators.T @ code_indicators).coalesce()
    rhs = code_indicators.T @ reference

    try:
        cooc = cooc.to_dense()
        cooc[torch.arange(len(cooc)), torch.arange(len(cooc))].clamp_min_(1.0)
        optimal_codebooks = (torch.linalg.lstsq(cooc, rhs)).solution.reshape(num_codebooks, codebook_size, group_size)
    except Exception as e:
        print(f"Linsolve failed with {e}")
        optimal_codebooks = codebooks
    return optimal_codebooks
