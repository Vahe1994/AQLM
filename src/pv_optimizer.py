"""Module containing utilities for straight-through fine-tuning of language models"""
import contextlib
import dataclasses
import random
import time
from collections import defaultdict
from enum import Enum, auto
from typing import Optional, Dict, Tuple, List, Any, Sequence, Iterator

import torch
import torch.nn as nn
import torch.distributed
from torch.optim.optimizer import StateDict

from src.aq import QuantizedWeight
from src.configurable_adam import ConfigurableAdamW


class ParameterRole(Enum):
    QUANTIZED_PARAMETER = auto()   # entire quantized weight, in a de-quantized form
    QUANTIZED_REPRESENTATION_PARAMETER = auto()  # part of quantized weight inner parameters, e.g. codebooks or scales
    NON_QUANTIZED_PARAMETER = auto()


class StraightThroughAdamW(ConfigurableAdamW):
    """
    A wrapper for a PyTorch optimizer that can perform updates on quantized and/or de-quantized parameters
    :param update_non_quantized_params: how to update parameters that are not directly linked to a QuantizedWeight.
        This may include biases, embeddings/heads, normalization layers or parts of the model that were not quantized.
    :param update_codebooks_and_scales: how to update continuous params of QuantizedWeight: codebooks and scales.
        This should be a dictionary of optimizer kwargs. In the latter case, these
            keyword arguments will be used when configuring optimizer for this specific parameter group
    :param update_codes: how to update codes in each QuantizedWeight with beam search and straight-through grad.
        This should be a dictionary of configurations, similar to
    :param max_code_change_per_step: max portion of discrete code groups that can be updated; only affects codes
    :param beam_size: beam search width used only when updating codes. See beam_size in aq.py
    :param stochastic_rounding_tau: if above 0, use stochastic rounding with this temperature. See aq.py
    :param dequantized_dtype: use this dtype when accumulating updates to de-quantized weight matrices

    :param sharded: if set, split all QuantizedWeights and their corresponding states between ranks (distributed only)
       Note: unlike FSDP, every QuantizedWeight is assigned fully to one rank (and not split into smaller parts)

    If sharded, the full list of directly optimized parameters for one rank is:
      - non-quantized weights from named_dequantized_weights
      - straight-through buffers for the assigned subset of quantized weights
      - continuous representation parameters for the assigned subset of quantized weights (codebooks and scales)

    The straight-through buffers are used to update the discrete representation parameters (codes) for QuantizedWeights

    The de-quantized versions of quantized weights in named_dequantzied_weights are *not* optimized directly.
    Instead, after each step updates codes, codebooks and scales, they are set to the latest dequantized weight.
    """

    def __init__(self,
                 named_dequantized_params: Dict[str, nn.Parameter],
                 named_quantized_params: Dict[str, QuantizedWeight],
                 *,
                 update_non_quantized_parameters: Optional[dict] = None,
                 update_codebooks_and_scales: Optional[dict] = None,
                 update_codes: Optional[dict] = None,
                 beam_size: int,
                 max_code_change_per_step: float,
                 stochastic_rounding_tau: float = 0,
                 delta_decay: float = 0,
                 dequantized_dtype: Optional[torch.dtype] = None,
                 sharded: bool = False,
                 verbose: bool = False,
                 **kwargs):
        self.sharded = sharded
        assert all(name in named_dequantized_params for name in named_quantized_params), "param names mismatch"
        assert all(isinstance(qw, QuantizedWeight) for qw in named_quantized_params.values())
        if sharded:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = _split_quantized_weights_between_ranks(named_quantized_params)
        param_groups, all_optimized_params = self._select_optimized_parameters(
            named_dequantized_params=named_dequantized_params,
            named_quantized_params=named_quantized_params,
            update_non_quantized_parameters=update_non_quantized_parameters,
            update_codebooks_and_scales=update_codebooks_and_scales,
            update_codes=update_codes,
            dequantized_dtype=dequantized_dtype)

        super().__init__(param_groups, **kwargs)
        self.ordered_quantized_weight_names = tuple(sorted(named_quantized_params.keys()))
        self.optimized_param_to_name = {param: name for name, param in all_optimized_params.items()}
        self.quantized_weights_by_name = {name: qw for name, qw in named_quantized_params.items()
                                          if isinstance(qw, (QuantizedWeight, YourQuantizedWeightIsInAnotherRank))}
        self.straight_through_buffer_by_name = {name: all_optimized_params[name]
                                                for name in self.quantized_weights_by_name.keys()
                                                if name in all_optimized_params}
        self.dequantized_weights_by_name = {name: param for name, param in named_dequantized_params.items()
                                            if name in named_quantized_params}
        if sharded:
            self.sharded_param_sizes_by_rank = _get_sharded_param_sizes_by_rank(named_dequantized_params)

        self.should_update_non_quantized_parameters = update_non_quantized_parameters is not None
        self.should_update_codebooks_and_scales = update_codebooks_and_scales is not None
        self.should_update_codes = update_codes is not None
        self.beam_size = beam_size
        self.max_code_change_per_step = max_code_change_per_step
        self.stochastic_rounding_tau = stochastic_rounding_tau
        self.delta_decay = delta_decay
        self.verbose = verbose

    def _select_optimized_parameters(
            self, named_dequantized_params, named_quantized_params, dequantized_dtype,
            update_non_quantized_parameters: Optional[dict], update_codebooks_and_scales: Optional[dict],
            update_codes: Optional[dict]) -> Tuple[List[Dict[str, Any]], Dict[str, nn.Parameter]]:
        """Choose which version of parameter to optimize: the parameter itself or a straight-through buffer"""
        non_quantized_params, quantized_params, quantized_representation_params = dict(), dict(), dict()
        for name, param in named_dequantized_params.items():
            if name not in named_quantized_params or isinstance(named_quantized_params[name], torch.Tensor):
                non_quantized_params[name] = param
            elif isinstance(named_quantized_params[name], QuantizedWeight):
                quantized_weight = named_quantized_params[name]
                with torch.no_grad():
                    dequantized_weight = quantized_weight()
                dequantized_weight = nn.Parameter(dequantized_weight.to(dtype=dequantized_dtype),
                                                  requires_grad=dequantized_weight.requires_grad)
                quantized_params[name] = dequantized_weight  # accumulator for optimizer updates; sharded alongside FSDP
                for subparam_name, subparam in quantized_weight.named_parameters():
                    full_name = f'{name}.{subparam_name}'
                    assert full_name not in quantized_representation_params, full_name
                    quantized_representation_params[full_name] = subparam
            elif isinstance(named_quantized_params[name], YourQuantizedWeightIsInAnotherRank):
                assert self.sharded  # running sharded optimizer, this weight should be optimized by another rank
            else:
                raise RuntimeError(f"Unxpected quantized param type {type(named_quantized_params[name])}")

        total_params = len(set(non_quantized_params) | set(quantized_params) | set(quantized_representation_params))
        assert total_params == len(non_quantized_params) + len(quantized_params) + len(quantized_representation_params)
        param_groups = []
        all_optimized_params = dict()
        if update_non_quantized_parameters is not None:
            all_optimized_params.update(non_quantized_params)
            param_groups.append(dict(params=list(non_quantized_params.values()),
                                     role=ParameterRole.NON_QUANTIZED_PARAMETER,
                                     **update_non_quantized_parameters))
        if update_codebooks_and_scales is not None:
            all_optimized_params.update(quantized_representation_params)
            param_groups.append(dict(params=list(quantized_representation_params.values()),
                                     role=ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER,
                                     **update_codebooks_and_scales))
        if update_codes is not None:
            all_optimized_params.update(quantized_params)
            param_groups.append(dict(params=list(quantized_params.values()),
                                     role=ParameterRole.QUANTIZED_PARAMETER,
                                     **update_codes))
        assert len(param_groups) > 0, ("Please set at least one of update_codes, update_codebooks_and_scales "
                                       "or update_non_quantized_parameters")
        return param_groups, all_optimized_params

    def step(self, *args, **kwargs):
        with print_runtime_stats("_propagate_grads_to_optimized_parameters", enabled=self.verbose):
            self._propagate_grads_to_optimized_parameters()
        with print_runtime_stats("super().step", enabled=self.verbose):
            original_output = super().step(*args, **kwargs)
        with print_runtime_stats("_optimize_quantized_weights", enabled=self.verbose):
            self._optimize_quantized_weights()
        with print_runtime_stats("_update_dequantized_weights", enabled=self.verbose):
            self._update_dequantized_weights()
        return original_output

    def _aggregate_gradients_for_dequantized_weights(self):
        """move gradients from dequantized params to straight-through buffers. If sharded, gather grads across ranks"""
        async_ops = list()
        aggregated_grads_by_name = dict()
        for name in self.ordered_quantized_weight_names:
            grad = self.dequantized_weights_by_name[name].grad
            if grad is None:
                assert self.dequantized_weights_by_name[name].numel() == 0
                grad = torch.zeros_like(self.dequantized_weights_by_name[name])
            assert grad is not None, name
            if not self.sharded:
                aggregated_grads_by_name[name] = grad
            else:
                quantized_weight = self.quantized_weights_by_name[name]
                own_rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                if isinstance(quantized_weight, QuantizedWeight):
                    shard_sizes: Sequence[int] = self.sharded_param_sizes_by_rank[name]
                    combined_grad_buffer = torch.full(
                        [quantized_weight.out_features, quantized_weight.in_features], fill_value=torch.nan,
                        dtype=grad.dtype, device=grad.device)
                    assert sum(shard_sizes) == combined_grad_buffer.numel()
                    gather_buffers = list(combined_grad_buffer.view(-1).split_with_sizes(shard_sizes))
                    assert all(part.untyped_storage().data_ptr() == combined_grad_buffer.untyped_storage().data_ptr()
                               for part in gather_buffers)
                    for i in range(world_size):
                        if i != own_rank:
                            async_ops.append(torch.distributed.irecv(gather_buffers[i], src=i))
                        else:
                            gather_buffers[i].copy_(grad)
                    aggregated_grads_by_name[name] = combined_grad_buffer
                else:
                    assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)
                    destination_rank = self.quantized_weights_by_name[name].rank
                    async_ops.append(torch.distributed.isend(grad.flatten(), destination_rank))

        for handle in async_ops:
            handle.wait()
        if self.verbose:
            for name, grad in aggregated_grads_by_name.items():
                print(end=f'aggregated grad norm for {name}: {grad.norm().item()}\n')
        return aggregated_grads_by_name

    @torch.no_grad()
    def _propagate_grads_to_optimized_parameters(self):
        """Ensure that every optimized parameter receives gradient"""
        aggregated_grads_by_name = self._aggregate_gradients_for_dequantized_weights()
        for param_group in self.param_groups:
            for param in param_group['params']:
                name = self.optimized_param_to_name[param]
                if param_group['role'] == ParameterRole.QUANTIZED_PARAMETER:
                    assert param is self.straight_through_buffer_by_name[name]
                    # pass gradients to straight-through update buffer or (possibly offloaded) quantized parameter
                    grad_wrt_dequantized_parameter = aggregated_grads_by_name[name]
                    assert grad_wrt_dequantized_parameter.shape == param.shape
                    param.grad = grad_wrt_dequantized_parameter.to(dtype=param.dtype, device=param.device)

                elif param_group['role'] == ParameterRole.NON_QUANTIZED_PARAMETER:
                    assert name not in self.dequantized_weights_by_name and name not in self.quantized_weights_by_name
                elif param_group['role'] == ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER:
                    assert name not in self.dequantized_weights_by_name
                    assert self.should_update_codebooks_and_scales
                    # gradients w.r.t quantized representation parameters are computed below via backprop
                else:
                    raise RuntimeError(f"Unexpected param role: {param_group['role']}")

        if self.should_update_codebooks_and_scales:
            # propagate gradients from dequantized weights to quantization parameters so they can be updated in step;
            # if sharded, every rank propagates gradients only for the QuantizedWeight instances owned by this rank
            with torch.enable_grad():
                for name, quantized_weight in self.quantized_weights_by_name.items():
                    if isinstance(quantized_weight, QuantizedWeight):
                        grad = aggregated_grads_by_name[name]
                        quantized_weight.forward().backward(grad)

    @torch.no_grad()
    def _optimize_quantized_weights(self):
        """Update discrete state representations to approximate straight through buffers"""
        # note: if sharded, this only updates the subset of quantized weights that are assigned to local rank

        for param_group in self.param_groups:
            if param_group['role'] == ParameterRole.QUANTIZED_PARAMETER:
                for param in param_group['params']:
                    name = self.optimized_param_to_name[param]
                    quantized_weight = self.quantized_weights_by_name[name]
                    reference_weight = param    # dequantized weight after optimizer updates
                    assert isinstance(quantized_weight, QuantizedWeight)

                    prev_codes = quantized_weight.get_codes().clone()  # [num_output_groups, num_input_groups]
                    new_codes = quantized_weight.beam_search_update_codes_(
                        reference_weight=reference_weight,
                        beam_size=self.beam_size,
                        stochastic_rounding_tau=self.stochastic_rounding_tau,
                        max_change_fraction=self.max_code_change_per_step,
                        dim_rng=random.Random(None),
                    )  # note: this updates quantized_weight.get_codes()[...] in-place
                    if self.delta_decay != 0:
                        reference_weight[...] = (
                            self.delta_decay * quantized_weight() + (1 - self.delta_decay) * reference_weight
                        )

                    if self.verbose:
                        code_change_rate = torch.not_equal(prev_codes, new_codes).any(-1).float().mean().item()
                        maybe_distributed_msg = ""
                        if torch.distributed.is_initialized():
                            maybe_distributed_msg = f" (rank {torch.distributed.get_rank()})"
                        maybe_limit_msg = ""
                        if self.max_code_change_per_step is not None:
                            maybe_limit_msg = f"(limit {self.max_code_change_per_step})"
                        maybe_individual_msg = ""
                        if quantized_weight.num_codebooks > 1:
                            subcode_change = torch.not_equal(prev_codes, new_codes).float().mean().item()
                            maybe_individual_msg = f" | individual code change {subcode_change}\n"
                        maybe_delta_msg = ""
                        if self.delta_decay != 1:
                            _dequantized_weight = quantized_weight()
                            delta_norm = (reference_weight - _dequantized_weight).norm().item()
                            relative_error = delta_norm / max(_dequantized_weight.norm().item(), 1e-9)
                            maybe_delta_msg = (f"\t||quantized_weight - optimized_weight|| / ||quantized_weight||"
                                               f" = {relative_error}\n")
                        print(end=f"Updated codes for {name}{maybe_distributed_msg}:\n\tFraction of weights with at "
                                  f"least one code change: {code_change_rate} {maybe_limit_msg}{maybe_individual_msg}\n"
                                  f"{maybe_delta_msg}\n")

    @torch.no_grad()
    def _update_dequantized_weights(self):
        """Assign dequantized weight buffers to latest quantized weights after codebook/scale/code updates"""
        own_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        async_ops = list()
        for name in self.ordered_quantized_weight_names:
            quantized_weight = self.quantized_weights_by_name[name]
            dequantized_weight_buffer = self.dequantized_weights_by_name[name]
            dequantized_weight_buffer.fill_(float('nan'))  # this is to ensure that the update reaches the buffer

            if not self.sharded:
                dequantized_weight_buffer[...] = quantized_weight()

            else:
                if isinstance(quantized_weight, QuantizedWeight):
                    new_dequantized_weight = quantized_weight().to(dequantized_weight_buffer.dtype)
                    shard_sizes: Sequence[int] = self.sharded_param_sizes_by_rank[name]
                    assert sum(shard_sizes) == new_dequantized_weight.numel()
                    new_dequantized_weight_parts = new_dequantized_weight.flatten().split_with_sizes(shard_sizes)
                    for i in range(world_size):
                        if i != own_rank:
                            async_ops.append(torch.distributed.isend(new_dequantized_weight_parts[i], dst=i))
                        else:
                            dequantized_weight_buffer.copy_(new_dequantized_weight_parts[i])

                else:
                    assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)
                    source_rank = self.quantized_weights_by_name[name].rank
                    async_ops.append(torch.distributed.irecv(dequantized_weight_buffer, src=source_rank))
        for handle in async_ops:
            handle.wait()

    def zero_grad(self, set_to_none: bool = True, *args, **kwargs) -> None:
        super().zero_grad(set_to_none=set_to_none, *args, **kwargs)
        for param in self.dequantized_weights_by_name.values():
            # dequantized weights are not in param_groups, but they still accumulate grads; reset them manually
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def iterate_local_quantized_weights(self) -> Iterator[Tuple[str, QuantizedWeight]]:
        """Iterate over (name, QuantizedWeight) pairs for all quantized weights trained by this optimizer and rank"""
        for name, quantized_weight in self.quantized_weights_by_name.items():
            if isinstance(quantized_weight, QuantizedWeight):  # skip YourQuantizedWeightIsInAnotherRank if sharded
                yield name, quantized_weight

    def state_dict(self) -> StateDict:
        state_dict = super().state_dict()
        assert "quantized_weight_state_dicts" not in state_dict
        state_dict["quantized_weight_state_dicts"] = {
            name: quantized_weight.state_dict() for name, quantized_weight in self.iterate_local_quantized_weights()
        }
        state_dict["straight_through_buffers"] = dict(self.straight_through_buffer_by_name)
        # note: the de-quantized params are not saved here; instead, they are saved with model.state_dict
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        quantized_weight_state_dicts: Dict[str, StateDict] = dict(state_dict.pop("quantized_weight_state_dicts"))
        for name, quantized_weight in self.iterate_local_quantized_weights():
            quantized_weight.load_state_dict(quantized_weight_state_dicts.pop(name))
        assert len(quantized_weight_state_dicts) == 0, f"unused keys: {quantized_weight_state_dicts.keys()}"

        straight_through_buffers = state_dict.pop("straight_through_buffers")
        assert all(name in straight_through_buffers for name in self.straight_through_buffer_by_name)
        for name, loaded_values in straight_through_buffers.items():
            self.straight_through_buffer_by_name[name][...] = loaded_values
        super().load_state_dict(state_dict)


def _get_sharded_param_sizes_by_rank(named_dequantized_params: Dict[str, torch.Tensor]) -> Dict[str, Sequence[int]]:
    """For each parameter name, return a tuple of sizes (numbers of elements) this parameter across all FSDP ranks"""
    assert torch.distributed.is_initialized()
    own_dequantized_param_shard_size = {name: param.numel() for name, param in named_dequantized_params.items()}
    world_size = torch.distributed.get_world_size()
    gathered_list = [{} for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_list, own_dequantized_param_shard_size)
    assert all(name in sizes_dict for sizes_dict in gathered_list for name in own_dequantized_param_shard_size)
    dequantized_param_sizes_by_rank = dict()
    for name in named_dequantized_params.keys():
        dequantized_param_sizes_by_rank[name] = [gathered_list[rank][name] for rank in range(world_size)]
    return dequantized_param_sizes_by_rank


def _split_quantized_weights_between_ranks(quantized_weights: Dict[str, QuantizedWeight]):
    """
    Split all quantized weights between ranks in a distributed setup; uses greedy knapsack heuristic.
    Note that unlike FSDP, this heuristic will always assign the entire quantized weight to one rank.

    :param quantized_weights: a dictionary [parameter_name] -> QuantizedWeight
    :returns: a dictionary similar to quantized weights or pointers to different ranks.
        If your rank stores this quantized weight for [name], then returned_dict[name] is quantized_weights[name]
        Otherwise, returned_dict[name] = YourQuantizedWeightIsInAnotherRank(rank=where_it_is_stored)
    """
    assert torch.distributed.is_initialized()
    own_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    all_quantized_weights: Dict[QuantizedWeight, List[str]] = defaultdict(list)
    for name, quantized_weight in quantized_weights.items():
        all_quantized_weights[quantized_weight].append(name)

    # order quantized weights in a rank-agnostic way: order by (param size desc, linked param name asc)
    def _compute_size(qw: QuantizedWeight) -> float:
        return qw.out_features * qw.in_features * qw.estimate_nbits_per_parameter()
    ordered_quantized_weights = sorted(
        all_quantized_weights, key=lambda qw: (-_compute_size(qw), min(all_quantized_weights[qw]))
    )
    assert len(ordered_quantized_weights) > 0, "internal error: could not find any linked QuantizedWeight in state"

    # split between ranks
    quantized_weight_to_rank = dict()
    total_size_by_rank = [0 for _ in range(world_size)]
    for quantized_weight in ordered_quantized_weights:
        least_busy_rank = min(range(world_size), key=lambda rank: total_size_by_rank[rank])
        total_size_by_rank[least_busy_rank] += _compute_size(quantized_weight)
        quantized_weight_to_rank[quantized_weight] = least_busy_rank

    checksum = tuple((min(all_quantized_weights[qw]), quantized_weight_to_rank[qw], _compute_size(qw))
                     for qw in ordered_quantized_weights)
    checksums = [() for _ in range(world_size)]
    torch.distributed.all_gather_object(checksums, checksum)
    assert checksums[own_rank] == checksum, (checksums, own_rank, checksum)
    assert all(other_checksum == checksum for other_checksum in checksums), checksums

    sharded_quantized_weights = dict()
    for name, quantized_weight in list(quantized_weights.items()):
        target_rank = quantized_weight_to_rank[quantized_weight]
        if target_rank == own_rank:
            sharded_quantized_weights[name] = quantized_weight
        else:
            sharded_quantized_weights[name] = YourQuantizedWeightIsInAnotherRank(target_rank)
    return sharded_quantized_weights


@dataclasses.dataclass(init=True, frozen=True)
class YourQuantizedWeightIsInAnotherRank:
    """This replaces quantized weights that are not held on this rank"""
    rank: int


@contextlib.contextmanager
def print_runtime_stats(operation_name: str, enabled: bool = True):
    if not enabled:
        yield
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if torch.device.type == 'cuda':
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    yield
    if torch.device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(end=f"rank{rank} {operation_name} took {time.perf_counter() - start_time}\n")


