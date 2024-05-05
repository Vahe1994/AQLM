"""Module containing utilities for straight-through fine-tuning of language models"""
import contextlib
import random
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from itertools import chain
from typing import Optional, Dict, Union, Tuple

import torch
import torch.nn as nn
import transformers
from torch.optim.optimizer import params_t

from src.aq import QuantizedLinear, QuantizedWeight


def create_dequantized_model(
        model: transformers.PreTrainedModel, *,
        reuse_non_quantized: bool,
        dequantized_dtype: Optional[torch.dtype] = None
):
    """
    Create a version of the model where all QuanizedWeight and derivative layers are de-quantized and cast to dtype.
    :param model: model to be dequantzied (out-of-place)
    :param reuse_non_quantized: if True, any non-quantized parameters and buffers are reused for de-quantized model;
        otherwise (default) they are copied and linked in the returned dictionary
    :returns: a model (converted out-of-place) and a mapping (dict) from de-quantized to master parameters
    """
    memo = dict()  # for deepcopy with replacement
    master_parameters = dict()
    all_quantized_weight_parameters = set()

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            assert module not in master_parameters and id(module) not in memo, f"{name} is converted more than once"
            quantized_weight = module.quantized_weight

            dequantized_module = nn.Linear(
                module.in_features, module.out_features, bias=module.bias is not None,
                dtype=dequantized_dtype if dequantized_dtype is not None else quantized_weight.get_codebooks().dtype,
                device=next(quantized_weight.parameters()).device
            )
            with torch.no_grad():
                dequantized_module.weight[...] = quantized_weight()
                dequantized_module.weight.requires_grad = any(p.requires_grad for p in quantized_weight.parameters())

                if module.bias is not None and not reuse_non_quantized:
                    dequantized_module.bias[...] = module.bias
                    dequantized_module.bias.requires_grad = dequantized_module.bias.requires_grad
                elif module.bias is not None and reuse_non_quantized:
                    dequantized_module.bias = module.bias

            memo[id(module)] = dequantized_module
            master_parameters[f"{name}.weight"] = quantized_weight
            if dequantized_module.bias is not module.bias:
                master_parameters[f"{name}.weight"] = module.bias
            all_quantized_weight_parameters |= set(quantized_weight.parameters())
            assert all(param in {dequantized_module.weight, dequantized_module.bias}
                       for param in dequantized_module.parameters())

    for name, param_or_buffer in chain(model.named_parameters(), model.named_buffers()):
        if name in master_parameters or param_or_buffer in all_quantized_weight_parameters:
            continue  # parameter already accounted for in the previous loop
        assert name not in master_parameters, name
        assert id(param_or_buffer) not in memo, name
        if reuse_non_quantized:
            new_param_or_buffer = param_or_buffer
        elif isinstance(param_or_buffer, nn.Parameter):
            new_param_or_buffer = nn.Parameter(param_or_buffer.data.clone(), param_or_buffer.requires_grad)
        else:
            new_param_or_buffer = param_or_buffer.detach().clone().requires_grad_(param_or_buffer.requires_grad)
        if new_param_or_buffer is not param_or_buffer:
            master_parameters[name] = new_param_or_buffer
        memo[id(param_or_buffer)] = new_param_or_buffer

    dequantized_model = deepcopy(model, memo=memo)

    for name, module in dequantized_model.named_modules():
        assert not isinstance(module, QuantizedWeight), (f"Dequantized model should not have quantized weights, "
                                                         f"but found {name} that is {module}")
    if reuse_non_quantized:
        assert all(isinstance(master, QuantizedWeight) for master in master_parameters.values())
    verify_dequantized_model(dequantized_model, master_parameters)
    return dequantized_model, master_parameters


def verify_dequantized_model(dequantized_model: nn.Module, master_parameters: dict):
    """Test that the dequantized model parameters still match the dequantized_to_master dictionary"""
    unmatched_master_parameters = set(master_parameters.keys())
    for name, param_or_buffer in chain(dequantized_model.named_parameters(), dequantized_model.named_buffers()):
        if name not in master_parameters:
            continue  # non-quantized weight
        master_param_or_buffer = master_parameters[name]
        assert param_or_buffer.shape == master_param_or_buffer.shape
        unmatched_master_parameters.remove(name)
    assert len(unmatched_master_parameters) == 0, f"Found unmatched tensors: {unmatched_master_parameters}"


class ParameterRole(Enum):
    QUANTIZED_PARAMETER = auto()   # entire quantized weight, in a de-quantized form
    QUANTIZED_REPRESENTATION_PARAMETER = auto()  # part of quantized weight inner parameters, e.g. codebooks or scales
    NON_QUANTIZED_PARAMETER = auto()


class StraightThroughAdamW(torch.optim.AdamW):
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
    """
    EXTRA_STATE_KEYS = ['name', 'param_version_that_accumulates_grad', 'quantized_weight']

    def __init__(self,
                 named_dequantized_params: Dict[str, nn.Parameter],
                 named_master_params: Dict[str, Union[QuantizedWeight, nn.Parameter]],
                 *,
                 update_non_quantized_parameters: Optional[dict] = None,
                 update_codebooks_and_scales: Optional[dict] = None,
                 update_codes: Optional[dict] = None,
                 beam_size: int,
                 max_code_change_per_step: float,
                 stochastic_rounding_tau: float = 0,
                 delta_decay: float = 1.0,
                 dequantized_dtype: Optional[torch.dtype] = None,
                 **kwargs):
        param_groups, all_optimized_params = self._select_optimized_parameters(
            named_dequantized_params=named_dequantized_params,
            named_master_params=named_master_params,
            update_non_quantized_parameters=update_non_quantized_parameters,
            update_codebooks_and_scales=update_codebooks_and_scales,
            update_codes=update_codes,
            dequantized_dtype=dequantized_dtype)

        super().__init__(param_groups, **kwargs)
        self._link_parameters_in_optimizer_state(all_optimized_params, named_dequantized_params, named_master_params)
        self.should_update_non_quantized_parameters = update_non_quantized_parameters is not None
        self.should_update_codebooks_and_scales = update_codebooks_and_scales is not None
        self.should_update_codes = update_codes is not None
        self.beam_size = beam_size
        self.max_code_change_per_step = max_code_change_per_step
        self.stochastic_rounding_tau = stochastic_rounding_tau
        self.delta_decay = delta_decay

    def _select_optimized_parameters(self, named_dequantized_params, named_master_params, dequantized_dtype,
                                     update_non_quantized_parameters: Optional[dict],
                                     update_codebooks_and_scales: Optional[dict],
                                     update_codes: Optional[dict]) -> Tuple[params_t, Dict[str, nn.Parameter]]:
        """Choose which version of parameter to optimize: the parameter itself or a straight-through buffer"""
        non_quantized_params, quantized_params, quantized_representation_params = dict(), dict(), dict()
        for name, param in named_dequantized_params.items():
            if name not in named_master_params:
                non_quantized_params[name] = param
            elif name in named_master_params and isinstance(named_master_params[name], nn.Parameter):
                non_quantized_params[name] = named_master_params[name]
            elif name in named_master_params and isinstance(named_master_params[name], QuantizedWeight):
                dequantized_weight = named_master_params[name]()
                dequantized_weight = nn.Parameter(dequantized_weight.to(dtype=dequantized_dtype),
                                                  requires_grad=dequantized_weight.requires_grad)
                quantized_params[name] = dequantized_weight  # accumulator for optimizer updates; sharded alongside FSDP
                quantized_weight = named_master_params[name]
                for subparam_name, subparam in quantized_weight.named_parameters():
                    full_name = f'{name}.{subparam_name}'
                    assert full_name not in quantized_representation_params, full_name
                    quantized_representation_params[full_name] = subparam
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

    def _link_parameters_in_optimizer_state(self, all_optimized_params, named_dequantized_params, named_master_params):
        """For each optimizer parameter in state, add its name, corresponding quantized and/or offloaded parameter"""

        all_quantized_representation_parameters = dict()
        for module in named_master_params.values():
            if isinstance(module, QuantizedWeight):
                all_quantized_representation_parameters.update({param: module for param in module.parameters()})

        quantized_weight_to_dequantized = dict()
        for name, dequantized_param in named_dequantized_params.items():
            if isinstance(named_master_params.get(name), QuantizedWeight):
                assert named_master_params[name] not in quantized_weight_to_dequantized
                quantized_weight_to_dequantized[named_master_params[name]] = dequantized_param

        for name, param in all_optimized_params.items():
            self.state[param]['name'] = name
            if isinstance(named_master_params.get(name), QuantizedWeight):
                quantized_weight = named_master_params[name]
                assert isinstance(quantized_weight, QuantizedWeight)
                self.state[param]['quantized_weight'] = quantized_weight
                self.state[param]['param_version_that_accumulates_grad'] = named_dequantized_params[name]
            elif param in all_quantized_representation_parameters:
                quantized_weight = all_quantized_representation_parameters[param]
                assert isinstance(quantized_weight, QuantizedWeight)
                self.state[param]['quantized_weight'] = quantized_weight
                dequantized_param = quantized_weight_to_dequantized[quantized_weight]
                print(quantized_weight.shape, dequantized_param.shape)
                self.state[param]['param_version_that_accumulates_grad'] = dequantized_param
            else:  # non_quantized params, e.g. biases, layernorms, etc
                self.state[param]['param_version_that_accumulates_grad'] = named_dequantized_params[name]

    def step(self, *args, **kwargs):
        self._propagate_grads_to_optimized_parameters()
        with self._hide_extra_state():
            original_output = super().step(*args, **kwargs)
        self._update_quantized_weights()
        self._update_dequantized_weights()
        return original_output

    @torch.no_grad()
    def _propagate_grads_to_optimized_parameters(self):
        for param_group in self.param_groups:
            for param in param_group['params']:
                if self.state[param]['param_version_that_accumulates_grad'] is not param:
                    accumulated_grad = self.state[param]['param_version_that_accumulates_grad'].grad
                    param.grad = accumulated_grad.to(dtype=param.dtype, device=param.device)
                    # pass gradients to straight-through update buffer or (possibly offloaded) master parameter
        if self.should_update_codebooks_and_scales:
            # propagate accumulated gradients from dequantized weights to quantization parameters so they can be updated
            for param_group in self.param_groups:
                if param_group['role'] == ParameterRole.QUANTIZED_PARAMETER:
                    for param in param_group['params']:
                        with torch.enable_grad():
                            grad = self.state[param]['param_version_that_accumulates_grad'].grad
                            self.state[param]['quantized_weight'].forward().backward(grad)

    @torch.no_grad()
    def _update_quantized_weights(self):
        if torch.distributed.is_initialized():
            raise NotImplementedError("TODO FSDP SUPPORT")

        for param_group in self.param_groups:
            if param_group['role'] == ParameterRole.QUANTIZED_PARAMETER:
                for param in param_group['params']:
                    reference_weight = param    # dequantized weight after optimizer updates
                    quantized_weight = self.state[param]['quantized_weight']
                    assert isinstance(quantized_weight, QuantizedWeight)

                    prev_codes = quantized_weight.get_codes().clone()  # [num_output_groups, num_input_groups]
                    new_codes = quantized_weight.beam_search_update_codes_(
                        reference_weight=reference_weight,
                        beam_size=self.beam_size,
                        stochastic_rounding_tau=self.stochastic_rounding_tau,
                        max_change_fraction=self.max_code_change_per_step,
                        dim_rng=random.Random(None),
                    )  # note: this updates quantized_weight.get_codes()[...] in-place
                    change_rate = torch.not_equal(prev_codes, new_codes).float().mean().item()
                    print(f"{self.state[param]['name']} change rate:", change_rate)
                    if self.delta_decay < 1.0:
                        reference_weight[...] = (
                            self.delta_decay * reference_weight + (1 - self.delta_decay) * quantized_weight()
                        )

    def _update_dequantized_weights(self):
        """Apply any updates to master parameters onto dequantized parameters"""
        quantized_weights_to_update = dict()
        for param_group in self.param_groups:
            for param in param_group['params']:
                if self.state[param]['param_version_that_accumulates_grad'] is not param:
                    param.grad = None  # deference so that previous grads can be deleted on zero_grad(set_to_none=True)

                if param_group['role'] in (ParameterRole.QUANTIZED_PARAMETER,
                                           ParameterRole.QUANTIZED_REPRESENTATION_PARAMETER):
                    assert isinstance(self.state[param].get('quantized_weight'), QuantizedWeight)
                    dequantized_weight = self.state[param]['param_version_that_accumulates_grad']
                    if self.state[param]['quantized_weight'] in quantized_weights_to_update:
                        assert quantized_weights_to_update[self.state[param]['quantized_weight']] is dequantized_weight
                    quantized_weights_to_update[self.state[param]['quantized_weight']] = dequantized_weight
                else:
                    assert param_group['role'] == ParameterRole.NON_QUANTIZED_PARAMETER
                    self.state[param]['param_version_that_accumulates_grad'].data[...] = param.data

        for quantized_weight, dequantized_weight in quantized_weights_to_update.items():
            dequantized_weight.data[...] = quantized_weight().data

    @contextlib.contextmanager
    def _hide_extra_state(self):
        """Hide """
        original_state = self.state
        try:
            self.state = defaultdict(dict)
            for param, param_state in original_state.items():
                self.state[param] = {k: v for k, v in param_state.items() if k not in self.EXTRA_STATE_KEYS}
            yield
            for param, param_state in self.state.items():
                original_state[param].update(param_state)
        finally:
            self.state = original_state

    def zero_grad(self, set_to_none: bool = True, *args, **kwargs) -> None:
        super().zero_grad(set_to_none=set_to_none, *args, **kwargs)
        for param_group in self.param_groups:
            for param in param_group['params']:
                if set_to_none:
                    self.state[param]['param_version_that_accumulates_grad'].grad = None
                elif self.state[param]['param_version_that_accumulates_grad'].grad is not None:
                    self.state[param]['param_version_that_accumulates_grad'].grad.zero_()
