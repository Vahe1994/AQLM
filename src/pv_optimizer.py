"""Module containing utilities for straight-through fine-tuning of language models"""
import random
from enum import Enum, auto
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed
import torch.nn as nn
from torch.optim.optimizer import StateDict

from src.aq import QuantizedWeight
from src.configurable_adam import ConfigurableAdamW
from src.pv_utils import YourQuantizedWeightIsInAnotherRank, print_runtime_stats


class ParameterRole(Enum):
    QUANTIZED_PARAMETER = auto()  # entire quantized weight, in a de-quantized form
    QUANTIZED_REPRESENTATION_PARAMETER = auto()  # part of quantized weight inner parameters, e.g. codebooks or scales
    NON_QUANTIZED_PARAMETER = auto()


class StraightThroughAdamW(ConfigurableAdamW):
    """
    A wrapper for a PyTorch optimizer that can perform updates on quantized and/or de-quantized parameters
    :param update_non_quantized_params: how to update parameters that are not directly linked to a QuantizedWeight.
        This may include biases, embeddings/heads, normalization layers or parts of the model that were not quantized.
        This should be either None (do not update) or a dictionary of optimizer kwargs. In the latter case, these
        keyword arguments will be used when configuring optimizer for this specific parameter group.
    :param update_codebooks_and_scales: how to update continuous params of QuantizedWeight: codebooks and scales.
        This should be either None (do not update) or a dictionary of optimizer kwargs. In the latter case, these
        keyword arguments will be used when configuring optimizer for this specific parameter group.
    :param update_codes: how to update codes in each QuantizedWeight with beam search and straight-through grad.
        This should be either None (do not update codes) or a dictionary of hyperparameter, similarly to above.
    :param delta_decay: determines whether to use straight-through estimation, direct optimization or a mixture thereof
        - if delta_decay == 1, do not use straight-through estimation. In this regime, the optimizer first updates
         de-quantized weights as though they were continuous, then uses modified weights to update codes, codebooks and
         scales; at the end of each step, the optimizer overwrites de-quantized weights to a de-quantization of the
         possibly updated quantized representations (codes, codebooks, scales).
        - if delta_decay == 0, use standard straight-through estimation. In this regime, the optimizer creates
        an internal set of straight-through buffers in the shape of de-quantized weights. The optimizer trains these
        buffers as though they were continuous; the quantized weights are then updated to minimize the L2 distance to
        these straight-through buffers; finally, the optimizer updates de-quantized weights from the quantized versions.
        - if delta_decay is between 0 and 1, use penalized straight-through estimation. The optimizer acts as though
        using standard straight-through estimation (see delta_decay == 0), but after every step, the straight-through
        buffers are set to (1 - delta_decay) * straight_through_buffer + delta_decay * quantized_weight.

    :param max_code_change_per_step: max portion of discrete code groups that can be updated; only affects codes
    :param code_trust_ratio: the maximum relative change to quantized weights per step, as a fraction of weight norm;
        see details in src/beam_search_l2.py, and in particular, beam_search_optimal_codes docstring.
    :param code_selection_temperature: if max_code_change or code_trust_ratio is set, the optimizer will by default
        prioritize updating codes with the largest delta = ||dequantized_weight_after_sgd_step - quantized_weight||_2 .
        If code_selection_temperature is above 0, it will instead sample codes randomly in proportion to the same
        delta ^ (1 / temperature). If temperature is very high, the optimizer will choose codes uniformly at random.
    :param force_code_update: if True, beam search will force codes to change even if code is optimal in
        terms of mean squared error. By default, the algorithm forces *all* weights to update this way, which may change
        weights too much. To limit the numer of updated weights, set max_code_change and trust_ratio.
    :param stochastic_rounding_tau: if above 0, use stochastic rounding with this temperature. See aq.py

    :param beam_size: beam search width used only when updating codes. See beam_size in aq.py

    :param straight_through_buffer_dtype: use this dtype when accumulating updates to de-quantized weight matrices
        Used only if delta_decay != 1.

    """

    def __init__(
        self,
        named_dequantized_params: Dict[str, nn.Parameter],
        named_quantized_params: Dict[str, Union[QuantizedWeight, YourQuantizedWeightIsInAnotherRank]],
        *,
        update_non_quantized_parameters: Optional[dict] = None,
        update_codebooks_and_scales: Optional[dict] = None,
        update_codes: Optional[dict] = None,
        beam_size: int,
        delta_decay: float = 1,
        max_code_change_per_step: float,
        code_trust_ratio: Optional[float] = None,
        code_selection_temperature: float = 0,
        force_code_update: bool = False,
        stochastic_rounding_tau: float = 0,
        straight_through_buffer_dtype: Optional[torch.dtype] = None,
        verbose: bool = False,
        **kwargs,
    ):
        assert 0 <= delta_decay <= 1
        assert all(
            isinstance(qw, (QuantizedWeight, YourQuantizedWeightIsInAnotherRank))
            for qw in named_quantized_params.values()
        )
        assert all(name in named_dequantized_params for name in named_quantized_params), "param names mismatch"

        self.sharded = not all(isinstance(qw, QuantizedWeight) for qw in named_quantized_params.values())
        self.is_straight_through = delta_decay != 1
        if verbose and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            print(end=f"PV optimizer init:\n\tAre quantized weights sharded? : {self.sharded}.\n")
            print(end=f"\tOptimizing {('without', 'with')[self.is_straight_through]} straight-through buffers\n")
        param_groups, all_optimized_params = self._select_optimized_parameters(
            named_dequantized_params=named_dequantized_params,
            named_quantized_params=named_quantized_params,
            update_non_quantized_parameters=update_non_quantized_parameters,
            update_codebooks_and_scales=update_codebooks_and_scales,
            update_codes=update_codes,
            straight_through_buffer_dtype=straight_through_buffer_dtype,
        )

        super().__init__(param_groups, **kwargs)
        self.ordered_quantized_weight_names = tuple(sorted(named_quantized_params.keys()))
        self.optimized_param_to_name = {param: name for name, param in all_optimized_params.items()}
        self.quantized_weights_by_name = {
            name: qw
            for name, qw in named_quantized_params.items()
            if isinstance(qw, (QuantizedWeight, YourQuantizedWeightIsInAnotherRank))
        }
        self.straight_through_buffer_by_name = (
            {
                name: all_optimized_params[name]
                for name in self.quantized_weights_by_name.keys()
                if name in all_optimized_params
            }
            if self.is_straight_through
            else {}
        )
        self.dequantized_weights_by_name = {
            name: param for name, param in named_dequantized_params.items() if name in named_quantized_params
        }
        if self.sharded:
            self.sharded_param_sizes_by_rank = _get_sharded_param_sizes_by_rank(named_dequantized_params)
            self.target_rank_by_name = {
                name: qw.rank if isinstance(qw, YourQuantizedWeightIsInAnotherRank) else torch.distributed.get_rank()
                for name, qw in self.quantized_weights_by_name.items()
            }

        self.should_update_non_quantized_parameters = update_non_quantized_parameters is not None
        self.should_update_codebooks_and_scales = update_codebooks_and_scales is not None
        self.should_update_codes = update_codes is not None

        self.delta_decay = delta_decay
        self.max_code_change_per_step = max_code_change_per_step
        self.code_trust_ratio = code_trust_ratio
        self.force_code_update = force_code_update
        self.code_selection_temperature = code_selection_temperature
        self.stochastic_rounding_tau = stochastic_rounding_tau
        self.beam_size = beam_size
        self.verbose = verbose

    def _select_optimized_parameters(
        self,
        named_dequantized_params,
        named_quantized_params,
        straight_through_buffer_dtype,
        update_non_quantized_parameters: Optional[dict],
        update_codebooks_and_scales: Optional[dict],
        update_codes: Optional[dict],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, nn.Parameter]]:
        """Choose which version of parameter to optimize: the parameter itself or a straight-through buffer"""
        non_quantized_params, quantized_params, quantized_representation_params = dict(), dict(), dict()
        for name, param in named_dequantized_params.items():
            if name not in named_quantized_params or isinstance(named_quantized_params[name], torch.Tensor):
                non_quantized_params[name] = param
            elif isinstance(named_quantized_params[name], QuantizedWeight):
                quantized_weight = named_quantized_params[name]
                if self.is_straight_through:  # create an accumulator for optimizer updates; sharded alongside FSDP
                    with torch.no_grad():
                        dequantized_weight = quantized_weight()
                    dequantized_weight = nn.Parameter(
                        dequantized_weight.to(dtype=straight_through_buffer_dtype),
                        requires_grad=dequantized_weight.requires_grad,
                    )
                else:
                    dequantized_weight = param
                quantized_params[name] = dequantized_weight
                for subparam_name, subparam in quantized_weight.named_parameters():
                    full_name = f"{name}.{subparam_name}"
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
            param_groups.append(
                dict(
                    params=list(non_quantized_params.values()),
                    role=ParameterRole.NON_QUANTIZED_PARAMETER,
                    **update_non_quantized_parameters,
                )
            )
        if update_codebooks_and_scales is not None:
            all_optimized_params.update(quantized_representation_params)
            param_groups.append(
                dict(
                    params=list(quantized_representation_params.values()),
                    role=ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER,
                    **update_codebooks_and_scales,
                )
            )
        if update_codes is not None:
            all_optimized_params.update(quantized_params)
            param_groups.append(
                dict(params=list(quantized_params.values()), role=ParameterRole.QUANTIZED_PARAMETER, **update_codes)
            )
        assert len(param_groups) > 0, (
            "Please set at least one of update_codes, update_codebooks_and_scales " "or update_non_quantized_parameters"
        )
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
        """collect full parameter gradients from fsdp-sharded parameters, return dict[name -> grad]"""
        grad_shards_by_name = dict()

        for name in self.ordered_quantized_weight_names:
            if self.dequantized_weights_by_name[name].grad is None:
                assert self.dequantized_weights_by_name[name].numel() == 0
                self.dequantized_weights_by_name[name].grad = torch.zeros_like(self.dequantized_weights_by_name[name])
            grad = self.dequantized_weights_by_name[name].grad
            assert grad is not None, name
            grad_shards_by_name[name] = grad

        if self.sharded:
            aggregated_grads_by_name = _aggregate_tensors_by_name(
                grad_shards_by_name,
                self.sharded_param_sizes_by_rank,
                self.target_rank_by_name,
                name_order=self.ordered_quantized_weight_names,
            )
        else:
            aggregated_grads_by_name = grad_shards_by_name

        aggregated_grads_by_name = {
            name: grad.view(self.quantized_weights_by_name[name].shape)
            for name, grad in aggregated_grads_by_name.items()
        }
        if self.verbose:
            for name, grad in aggregated_grads_by_name.items():
                print(end=f"aggregated grad norm for {name}: {grad.norm().item()}\n")
        return aggregated_grads_by_name

    def _aggregate_dequantized_weights(self):
        """collect full (possibly optimizer-updated) dequantized weights"""
        if not self.sharded:
            return self.dequantized_weights_by_name
        dequantized_flat_param_shards = {
            name: param.data.flatten() for name, param in self.dequantized_weights_by_name.items()
        }
        flat_aggregated_params_by_name = _aggregate_tensors_by_name(
            dequantized_flat_param_shards,
            self.sharded_param_sizes_by_rank,
            self.target_rank_by_name,
            name_order=self.ordered_quantized_weight_names,
        )
        aggregated_params_by_name = {
            name: param.view(self.quantized_weights_by_name[name].shape)
            for name, param in flat_aggregated_params_by_name.items()
        }
        return aggregated_params_by_name

    @torch.no_grad()
    def _propagate_grads_to_optimized_parameters(self):
        """Ensure that every optimized parameter receives gradient"""
        aggregated_grads_by_name = self._aggregate_gradients_for_dequantized_weights()
        for param_group in self.param_groups:
            for param in param_group["params"]:
                name = self.optimized_param_to_name[param]
                if param_group["role"] == ParameterRole.QUANTIZED_PARAMETER:
                    if self.is_straight_through:
                        assert param is self.straight_through_buffer_by_name[name]
                        # pass gradients to straight-through update buffer or (possibly offloaded) quantized parameter
                        grad_wrt_dequantized_parameter = aggregated_grads_by_name[name]
                        assert grad_wrt_dequantized_parameter.shape == param.shape
                        param.grad = grad_wrt_dequantized_parameter.to(dtype=param.dtype, device=param.device)
                    else:
                        assert len(self.straight_through_buffer_by_name) == 0, self.straight_through_buffer_by_name
                        assert param.grad is not None
                elif param_group["role"] == ParameterRole.NON_QUANTIZED_PARAMETER:
                    assert name not in self.dequantized_weights_by_name and name not in self.quantized_weights_by_name
                elif param_group["role"] == ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER:
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
                        quantized_weight.forward().backward(aggregated_grads_by_name[name])

    @torch.no_grad()
    def _optimize_quantized_weights(self):
        """Update discrete state representations to approximate straight through buffers"""
        # note: if sharded, this only updates the subset of quantized weights that are assigned to local rank
        remaining_quantized_weights = {
            name: qw for name, qw in self.quantized_weights_by_name.items() if isinstance(qw, QuantizedWeight)
        }
        if self.is_straight_through:
            reference_weights_by_name = self.straight_through_buffer_by_name
        else:
            reference_weights_by_name = self._aggregate_dequantized_weights()

        for param_group in self.param_groups:
            if param_group["role"] == ParameterRole.QUANTIZED_PARAMETER:
                for param in param_group["params"]:
                    # param is either a dequantized weight or a special straight-through buffer (if is_straight_through)
                    name = self.optimized_param_to_name[param]
                    quantized_weight = remaining_quantized_weights.pop(name)
                    reference_weight = reference_weights_by_name[name]
                    assert reference_weight.shape == quantized_weight.shape, (
                        reference_weight.shape,
                        quantized_weight.shape,
                    )
                    assert isinstance(quantized_weight, QuantizedWeight)

                    prev_codes = quantized_weight.get_codes().clone()  # [num_output_groups, num_input_groups]
                    new_codes = quantized_weight.beam_search_update_codes_(
                        reference_weight=reference_weight,
                        beam_size=self.beam_size,
                        stochastic_rounding_tau=self.stochastic_rounding_tau,
                        max_update_fraction=self.max_code_change_per_step,
                        force_update=self.force_code_update,
                        code_selection_temperature=self.code_selection_temperature,
                        trust_ratio=self.code_trust_ratio,
                        dim_rng=random.Random(None),
                    )  # note: this updates quantized_weight codes in-place
                    if self.delta_decay != 0 and self.is_straight_through:
                        self.straight_through_buffer_by_name[name][...] = (
                            self.delta_decay * quantized_weight() + (1 - self.delta_decay) * reference_weight
                        )
                        # if not is_straight_throuh, param will be properly updated in _update_dequantized_weights

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
                            maybe_individual_msg = f" | overall change {subcode_change:.8f}"
                        maybe_delta_msg = ""
                        if self.delta_decay != 1:
                            _dequantized_weight = quantized_weight()
                            delta_norm = (reference_weight - _dequantized_weight).norm().item()
                            relative_error = delta_norm / max(_dequantized_weight.norm().item(), 1e-9)
                            maybe_delta_msg = (
                                f"\t||quantized_weight - optimized_weight|| / ||quantized_weight||"
                                f" = {relative_error}\n"
                            )
                        print(
                            end=f"Updated codes for {name}{maybe_distributed_msg}:\n\tFraction of weights with at "
                            f"least one code change: {code_change_rate:.8f} "
                            f"{maybe_limit_msg}{maybe_individual_msg}\n{maybe_delta_msg}\n"
                        )
        assert len(remaining_quantized_weights) == 0

    @torch.no_grad()
    def _update_dequantized_weights(self):
        """Assign dequantized weight buffers to latest quantized weights after codebook/scale/code updates"""
        own_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        async_ops = list()
        for name in self.ordered_quantized_weight_names:
            quantized_weight = self.quantized_weights_by_name[name]
            dequantized_weight_buffer = self.dequantized_weights_by_name[name]
            dequantized_weight_buffer.fill_(float("nan"))  # this is to ensure that the update reaches the buffer

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
        state_dict["straight_through_buffers"] = dict(self.straight_through_buffer_by_name)  # may be empty
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


def _aggregate_tensors_by_name(
    sharded_tensors_by_name: Dict[str, torch.Tensor],
    shard_sizes_by_name: Dict[str, Sequence[int]],
    target_rank_by_name: Dict[str, int],
    name_order: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    :param sharded_tensors_by_name: a dictionary from string to flat (1d) tensors available on the current shard
    :note: the keys should be the same across ranks and go in the same order; if not, use ordered_names
    :param shard_sizes_by_name: a dictionary from name to a list of sizes (numel) for this key across ranks
    :param target_rank_by_name: a dictionary from name to a rank that this name should be aggregated to
    :param name_order: if specified, this defines the order in which devices go over named shards
    """
    assert torch.distributed.is_initialized()
    own_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    aggregated_tensors_by_name = dict()
    async_ops = list()

    for name in sorted(sharded_tensors_by_name.keys()) if name_order is None else name_order:
        shard = sharded_tensors_by_name[name]
        assert shard.ndim == 1
        destination_rank = target_rank_by_name[name]
        shard_sizes: Sequence[int] = shard_sizes_by_name[name]
        if destination_rank == own_rank:
            total_numel = sum(shard_sizes)
            combined_buffer = torch.full((total_numel,), fill_value=torch.nan, dtype=shard.dtype, device=shard.device)
            gather_buffers = list(combined_buffer.split_with_sizes(shard_sizes))
            assert all(
                part.untyped_storage().data_ptr() == combined_buffer.untyped_storage().data_ptr()
                for part in gather_buffers
            )
            for i in range(world_size):
                if shard_sizes[i] == 0:
                    continue  # optimization: this handles FSDP where some param/grad shards are empty
                elif i != own_rank:
                    async_ops.append(torch.distributed.irecv(gather_buffers[i], src=i))
                else:
                    gather_buffers[i].copy_(shard)
            aggregated_tensors_by_name[name] = combined_buffer
        else:
            if shard_sizes[own_rank] == 0:
                continue
            async_ops.append(torch.distributed.isend(shard, destination_rank))

    for handle in async_ops:
        handle.wait()
    return aggregated_tensors_by_name
