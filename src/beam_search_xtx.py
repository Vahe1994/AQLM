""" Beam search that minimizes ||XWref - XWq||^2 w.r.t. Wq codes """
import random
from typing import Optional, Tuple

import torch
from torch.nn import functional as F
from tqdm.asyncio import trange

from src.utils import _dequantize_weight, maybe_script


@torch.inference_mode()
def beam_search_optimal_codes(
    *,
    XTX: torch.Tensor,
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    prev_codes: torch.IntTensor,
    scales: Optional[torch.Tensor],
    beam_size: int,
    dim_rng: Optional[random.Random] = None,
    sparsity_regularizer: float = 0,
    verbose: bool,
):
    """
    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if XTX is divided by dataset size, this function will return *mean* squared error
    :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_siz, in_group_size]
    :param prev_codes: previous-best integer weight codes, shape: [num_out_groups, num_in_groups, num_codebooks]
    :param scales: weight will be multiplied by this factor, shape = [num_out_groups, num_in_groups or 1, 1, 1]
    :param dim_rng: a source of randomness to (optionally) shuffle the order in which the beam search runs
      None = update dimensions and codebooks in their natural order (0, 1, ..., n)
      random.Random(optional_seed) = shuffle dimensions at random, optionally using the specified seed

    :param beam_size: consider up to this many best encoding combinations
    :param sparsity_regularizer: subtract this value from beam search objective each time you have a zero code somewhere
    :param verbose: if True, draw a progressbar and periodically print best loss
    :return: best quantization codes found, same shape as prev_codes

    :intuition: the beam search needs to produce weight codes that minimize MSE error
    - the codes are of shape [out_features / out_group_size, in_features / in_group_size, num_codebooks]

    Out of those three dimensions, out_features is "independent", i.e. changing code in
    one output feature does not increase the MSE error for another feature. Therefore,
    beam search for different output features can run in independently in parallel.

    Neither (in_features / in_group_size) nor (num_codebooks) dimension are independent:
    - changing the encoding for one feature can compensate the error from encoding another, OBC-style
    - for a single weight group, changing code in one codebook can affect the optimal choice in another codebook
    Therefore, beam search must go in a double loop over (in_features/in_group_size) and (num_codebooks) dimensions

    This leaves one choice: which dimension used for outer loop, and which one goes is in the inner loop?
    Due to the nature of beam search, interactions between dimensions of inner loop will be explored better.
    We chose to use (in_features/in_group_size) in the outer loop and (num_codebooks) for the inner loop.
    This is based on an intuition from GPTQ: you can get decent performance by quantizing each input unit ...
    ... greedily --- GPTQ does not change quantizations for previously quantized features and works fine.
    Therefore, we believe that we can also use a greedy approach to compensate error between input features.
    In turn, we believe that the codes used to encode the same weights (additively) are more inter-dependent.
    This should be treated as an educated guess with no proof and no ablation (as of the time of writing).

    """
    num_out_groups, num_in_groups, num_codebooks = prev_codes.shape
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    in_features = num_in_groups * in_group_size
    out_features = num_out_groups * out_group_size
    assert reference_weight.shape == (out_features, in_features)
    prev_weight = _dequantize_weight(prev_codes, codebooks, scales)

    # initialize all beam codes as previous codes - so they can be updated during beam search
    beam_codes = prev_codes.unsqueeze(0)
    # beam_codes shape: [current beam_size, num_out_groups, num_in_groups, num_codebooks], initial beam_size = 1
    beam_weights = prev_weight.unsqueeze(0)
    # beam_weights shape: [current beam_size, out_features, in_features], initial beam size = 1

    beam_losses = (
        _channelwise_squared_error(XTX, prev_weight, reference_weight)
        .reshape(1, num_out_groups, out_group_size)
        .sum(-1)
    )
    # beam_losses shape: [current beam_size, num_out_groups], initial beam_size = 1
    if sparsity_regularizer != 0:
        beam_losses = beam_losses - sparsity_regularizer * (prev_codes == 0).sum(dim=(-1, -2))[None, :]

    if verbose:
        progressbar = trange(num_in_groups * num_codebooks)

    def _make_range(n: int) -> list:
        seq = list(range(n))
        if dim_rng is not None:
            dim_rng.shuffle(seq)
        return seq

    for input_group_index in _make_range(num_in_groups):
        for codebook_index in _make_range(num_codebooks):
            ### part 1: compute losses for every possible candidate for one given codebook and input group.
            # Currently, we compute errors for all output features in parallel in a vectorized fashion.
            best_losses, best_indices = _beam_search_squared_errors(
                XTX=XTX,
                reference_weight=reference_weight,
                codebooks=codebooks,
                scales=scales,
                beam_losses=beam_losses,
                beam_codes=beam_codes,
                beam_weights=beam_weights,
                input_group_index=input_group_index,
                codebook_index=codebook_index,
                k_best=beam_size,
                sparsity_regularizer=sparsity_regularizer,
            )  # [current beam_size, codebook_size, num_out_groups]

            # part 2: select beam_size new best codes and re-arrange beam to account for the fact that ...
            # ... sometimes two or more top candidates originate from the same source in previous beam
            beam_codes, beam_weights, beam_losses = _beam_search_select_best(
                beam_codes=beam_codes,
                beam_weights=beam_weights,
                codebooks=codebooks,
                scales=scales,
                input_group_index=input_group_index,
                codebook_index=codebook_index,
                best_losses=best_losses,
                best_indices=best_indices,
                beam_size=beam_size,
            )

            if verbose:
                progressbar.update()
                if (input_group_index * num_codebooks + codebook_index) % verbose != 0:
                    continue  # if update is an integer, compute metrics every (this many) beam search steps
                best_loss = beam_losses.min(0).values.sum().item() / out_features
                info = f"in_group {input_group_index} / {num_in_groups} "
                info += f"| codebook {codebook_index} / {num_codebooks} "
                if sparsity_regularizer == 0:
                    info += f"| loss {best_loss:.10f}"
                else:  # un-regularize to restore MSE loss, report sparsity rate
                    num_zero_codes = (beam_codes[0] == 0).sum().item()
                    best_loss = best_loss + sparsity_regularizer / out_features * num_zero_codes
                    sparsity = num_zero_codes / prev_codes.numel()
                    info += f"| loss {best_loss:.5f} | sparse {sparsity * 100:.1f}% |"

                progressbar.desc = info
    return beam_codes[0]


@maybe_script
def _beam_search_squared_errors(
    XTX: torch.Tensor,
    reference_weight: torch.Tensor,
    codebooks: torch.Tensor,
    scales: Optional[torch.Tensor],
    beam_losses: torch.Tensor,
    beam_codes: torch.Tensor,
    beam_weights: torch.Tensor,
    input_group_index: int,
    codebook_index: int,
    k_best: int,
    sparsity_regularizer: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MSE or sum-of-squared-error losses for all possible ways to replace quantization codes for one input group
     and one codebook. Works in parallel for all output-dimension groups.

    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if both XTX *and* beam_loses are divided by dataset size, this function will return mean squared error
    :param reference_weight: original weight matrix that is being quantized, shape: [out_features, in_features]
    :param codebooks: look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, [num_out_groups, num_in_groups, 1, 1]

    :param beam_losses: sum-of-squared-error for each hypothesis in beam and for each output channel;
        shape: [beam_size, num_out_groups]
    :param beam_codes: a tensor with best weight codes, shape: [beam_size, num_out_groups, num_in_groups, num_codebooks]
    :param beam_weights: a tensor with de-quantized beam_codes, shape: [beam_size, out_features, in_features]
    :param input_group_index: an index of one group of in_features that is being re-encoded
    :param codebook_index: an index of one codebook for that group of features that is being re-encoded
    :return: tuple(Tensor, Tensor) of 3d tensor of shape = [beam_size, k_best, num_out_groups].
        First one is float tensor of losses of k_best lowest square errors for each beam and out_group
        Second one is int64 tensor of indices of k_best lowest square errors for each beam and out_group

    :note: The code computes MSE using the square-of-difference expansion
     ||X@W.T - sum_i X@(Bi@Ci).T||^2 = ||X@W.T||^2 - 2 <X@W.T, sum_i X@(Bi@Ci).T> + ||sum_i X@Bi@Ci||^2
    where X[nsamples,in_features] is calibration data, W[out_features, in_features] is the reference weight,
       C[num_codebooks, codebook_size, in_features] are learned codebooks (Ci has shape [codebook_size, out_features])
       B[num_codebooks, out_features, codebook_size] are one-hot encoded indices (quantization codes)
    The formula above uses a single group per output "neuron" and a single group.
    The algorithm below generalizes the formula for multiple groups and codebooks.

    Furthermore, the algorithm does not compute the entire formula. Instead, it begins from some baseline loss
    and computes the change in loss from changing a single code to every possible altearnative code.
    When computing the changed loss, the algorithm only computes the few affected parts of the loss formula above.
    """
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape
    out_features = num_out_groups * out_group_size

    input_group_slice = slice(input_group_index * in_group_size, (input_group_index + 1) * in_group_size)

    prev_codes_part = beam_codes[:, :, input_group_index, codebook_index]  # [beam_size, num_out_groups]

    if scales is not None:
        scales_part = scales[:, input_group_index % scales.shape[1], :, :]  # [num_out_groups, 1, 1]
    else:
        scales_part = torch.empty(0, device=XTX.device)
    prev_part_dequantized = F.embedding(prev_codes_part, codebooks[codebook_index].flatten(-2, -1)).view(
        beam_size, out_features, in_group_size
    )  # previous codes de-quantized

    prev_weight_part = prev_part_dequantized
    if scales is not None:
        prev_weight_part = (
            prev_weight_part.view(beam_size, num_out_groups, out_group_size, in_group_size)
            .mul(scales_part)
            .view(beam_size, out_features, in_group_size)
        )

    cand_weights = codebooks[codebook_index]  # [codebook_size, out_group_size, in_group_size], all replacement codes

    delta_weight_without_part = reference_weight - beam_weights
    delta_weight_without_part[:, :, input_group_slice] += prev_weight_part

    # dWTXTX is equivalent to < X @ (W - \sum BiCi except current codebook), X @ SOMETHING >
    dWTXTXg = delta_weight_without_part @ XTX[..., input_group_slice]  # [beam_size, out_features, in_group_size]
    # below: use torch.matmul to compute broadcasted batch matrix multiplication; see matmul docs

    XnewBkC_norms_sq = torch.bmm(
        (cand_weights.flatten(0, 1) @ XTX[input_group_slice, input_group_slice]).view(
            codebook_size, 1, out_group_size * in_group_size
        ),
        cand_weights.view(codebook_size, out_group_size * in_group_size, 1),
    ).reshape(
        codebook_size, 1
    )  # [codebook_size, num_out_groups]
    if scales is not None:
        XnewBkC_norms_sq = XnewBkC_norms_sq.mul(scales_part.square().reshape(1, num_out_groups))

    best_losses = torch.empty(
        (beam_size, k_best, num_out_groups), dtype=XTX.dtype, device=XTX.device
    )  # shape: [beam_size, k_best, num_out_groups]
    best_indices = torch.empty(
        (beam_size, k_best, num_out_groups),
        dtype=torch.int64,
        device=XTX.device,
    )
    for beam_id in range(beam_size):
        dot_products = (
            torch.einsum(
                "mg,og->mo",
                cand_weights.reshape(codebook_size, out_group_size * in_group_size),
                dWTXTXg[beam_id].view(num_out_groups, out_group_size * in_group_size),
            )
            .sub_(
                torch.einsum(
                    "og,og->o",
                    prev_part_dequantized[beam_id].reshape(num_out_groups, out_group_size * in_group_size),
                    dWTXTXg[beam_id].view(num_out_groups, out_group_size * in_group_size),
                ).view(1, num_out_groups)
            )
            .view(codebook_size, num_out_groups)
        )
        if scales is not None:
            dot_products = dot_products.mul_(scales_part.reshape(1, num_out_groups))

        XoldBkC_norms_sq = torch.bmm(
            (prev_weight_part[beam_id] @ XTX[input_group_slice, input_group_slice]).view(
                num_out_groups, 1, out_group_size * in_group_size
            ),
            prev_weight_part[beam_id].view(num_out_groups, out_group_size * in_group_size, 1),
        ).reshape(1, num_out_groups)

        # finally, combine them to get MSE
        candidate_squared_errors = (
            beam_losses[beam_id, None, :] - 2 * dot_products + XnewBkC_norms_sq - XoldBkC_norms_sq
        )  # shape: [codebook_size, num_out_groups]

        if sparsity_regularizer != 0:
            candidate_squared_errors += sparsity_regularizer * (prev_codes_part[beam_id] == 0).to(XTX.dtype)[None, :]
            candidate_squared_errors[0, :] -= sparsity_regularizer

        best_beam_squared_errors, best_beam_indices = torch.topk(
            candidate_squared_errors, k_best, dim=0, largest=False, sorted=False
        )
        best_losses[beam_id] = best_beam_squared_errors
        best_indices[beam_id] = best_beam_indices

    return best_losses, best_indices


@maybe_script
def _beam_search_select_best(
    beam_codes: torch.Tensor,
    beam_weights: torch.Tensor,
    codebooks: torch.Tensor,
    scales: Optional[torch.Tensor],
    input_group_index: int,
    codebook_index: int,
    best_losses: torch.Tensor,
    best_indices: torch.Tensor,
    beam_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select top-:beam_size: and reorder beam accordingly, return new beam
    :param beam_codes: a tensor with best weight codes, shape: [beam_size, num_out_groups, num_in_groups, num_codebooks]
    :param beam_weights: a tensor with de-quantized beam_codes, shape: [beam_size, out_features, in_features]
    :param codebooks: a tensor with look-up tables of codes, shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
    :param scales: weight will be multiplied by this factor, [num_out_groups, num_in_groups, 1, 1]

    :param input_group_index: an index of one group of in_features that is being re-encoded
    :param codebook_index: an index of one codebook for that group of features that is being re-encoded
    :param best_losses: a 3d tensor of losses of k_best lowest square errors for each beam and out group,
        shape = [beam_size, k_best, num_out_groups]
    :param best_indices: a 3d tensor of indices of k_best lowest square errors for each beam and out group,
        shape = [beam_size, k_best, num_out_groups]
    :param beam_size: how many top hypotheses should be selected

    :returns: new (beam_codes, beam_weights, beam_losses)
    """
    dtype = best_losses.dtype
    device = best_losses.device
    _prev_beam_size, k_best, num_out_groups = best_losses.shape
    _prev_beam_size, out_features, in_features = beam_weights.shape
    _prev_beam_size, num_out_groups, num_in_groups, num_codebooks = beam_codes.shape
    flat_best = best_losses.flatten(0, 1).topk(dim=0, k=beam_size, largest=False)
    best_hypo_source_ids = flat_best.indices // k_best
    arange_out_groups = torch.arange(num_out_groups, device=device)
    best_hypo_codes = best_indices.flatten(0, 1)[flat_best.indices, arange_out_groups].reshape(
        beam_size, num_out_groups
    )
    # ^-- shape: [beam_size, num_out_groups]

    # reorder beam codes and weights
    new_beam_codes = torch.full(
        size=(len(best_hypo_codes), num_out_groups, num_in_groups, num_codebooks),
        fill_value=-1,
        dtype=beam_codes.dtype,
        device=device,
    )  # [beam_size, num_out_groups, num_in_groups, num_codebooks]
    new_beam_weights = torch.empty(len(best_hypo_codes), out_features, in_features, dtype=dtype, device=device)

    for beam_index in range(len(best_hypo_codes)):
        new_beam_codes[beam_index, :, ...] = beam_codes[best_hypo_source_ids[beam_index, :], arange_out_groups, ...]
        new_beam_codes[beam_index, :, input_group_index, codebook_index] = best_hypo_codes[beam_index, :]
        new_beam_weights[beam_index, :, :] = _dequantize_weight(new_beam_codes[beam_index, ...], codebooks, scales)

    # Note: the code above can be further accelerated by 1) vectorzing loop and ...
    # ... 2) updating new_beam_weights only for the chosen input group
    return new_beam_codes, new_beam_weights, flat_best.values


@maybe_script
def _channelwise_squared_error(XTX: torch.Tensor, weight: torch.Tensor, reference_weight: torch.Tensor):
    """
    Compute per-channel squared error between X @ weight_or_weights and X @ reference_weight
    :param XTX: pairwise products of input features matmul(X.transpose(), X), shape: [in_features, in_features]
    :note: if XTX is divided by dataset size, this function will return *mean* squared error
    :param weight: predicted/reconstructed weights of shape [*dims, out_features, in_features]
    :param reference_weight: reference weight of shape [out_features, in_features]
    :return: per-channel squared errors of shape [*dims, out_features]
    """
    XW_norm_square = torch.matmul(weight[..., :, None, :], (weight @ XTX)[..., :, :, None]).flatten(-3)
    XWreference_norm_square = torch.bmm(reference_weight[:, None, :], (reference_weight @ XTX)[:, :, None]).flatten(-3)
    dot_product = torch.matmul((reference_weight @ XTX)[:, None, :], weight[..., :, :, None]).flatten(-3)
    return XW_norm_square - 2 * dot_product + XWreference_norm_square
