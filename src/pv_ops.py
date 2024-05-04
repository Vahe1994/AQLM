"""Module containing utilities for straight-through fine-tuning of language models"""
import random
from copy import deepcopy
from itertools import chain
from typing import Optional, Iterable, Dict, Union

import torch
import torch.nn as nn
import transformers
from torch.optim.optimizer import StateDict

from src.aq import QuantizedLinear, QuantizedWeight
from src.aq_ops import _dequantize_weight
from src.beam_search_l2 import beam_search_optimal_codes


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
    :returns: a model (converted out-of-place) and a mapping (dict) from de-quantized to original parameters
    """
    memo = dict()  # for deepcopy with replacement
    master_parameters = dict()

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
            assert all(param in {dequantized_module.weight, dequantized_module.bias}
                       for param in dequantized_module.parameters())

    for name, param_or_buffer in chain(model.named_parameters(), model.named_modules()):
        if name in master_parameters:
            continue  # parameter already accounted for in the previous loop
        assert name not in master_parameters
        assert id(param_or_buffer) not in memo
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
        if param_or_buffer not in master_parameters:
            continue  # non-quantized weight
        master_param_or_buffer = master_parameters[name]
        assert param_or_buffer.shape == master_param_or_buffer
        unmatched_master_parameters.remove(name)
    assert len(unmatched_master_parameters) == 0, f"Found unmatched tensors: {unmatched_master_parameters}"


class StraightThroughAdamW(torch.optim.AdamW):
    """
    A wrapper for any PyTorch optimizer that enables updates on quantized parameters

    """
    def __init__(self,
                 named_dequantized_params: Dict[str, nn.Parameter],
                 named_master_params: Dict[str, Union[QuantizedWeight, nn.Parameter]],
                 *,
                 max_code_change_per_step: float,
                 beam_size: int,
                 stochastic_rounding_tau: float = 0,
                 delta_dtype: Optional[torch.dtype] = None,
                 **kwargs):
        optimized_params = self._select_optimized_parameters(named_dequantized_params, named_master_params, delta_dtype)
        super().__init__(list(optimized_params.values()), **kwargs)
        self._link_parameters_in_optimizer_state(optimized_params, named_dequantized_params, named_master_params)
        self.max_code_change_per_step = max_code_change_per_step
        self.stochastic_rounding_tau = stochastic_rounding_tau
        self.beam_size = beam_size

    def _select_optimized_parameters(self, named_dequantized_params, named_master_params, delta_dtype):
        """Choose which version of parameter to optimize: the parameter itself or a straight-through buffer"""
        optimized_params = dict()
        for name, param in named_dequantized_params.items():
            if name not in named_master_params:
                optimized_params[name] = param
            elif name in named_master_params and isinstance(named_master_params[name], nn.Parameter):
                optimized_params[name] = named_master_params[name]
            elif name in named_master_params and isinstance(named_master_params[name], QuantizedWeight):
                delta = torch.zeros_like(param, dtype=delta_dtype, requires_grad=True)
                optimized_params[name] = delta  # accumulator for optimizer updates; sharded alongside FSDP
                # note: we track delta (difference) instead of raw weight to better handle half precision
        return optimized_params

    def _link_parameters_in_optimizer_state(self, optimized_params, named_dequantized_params, named_master_params):
        """For each optimizer parameter in state, add its name, corresponding quantized and/or offloaded parameter"""
        for name, param in optimized_params.items():
            self.state[param]['name'] = name
            self.state[param]['param_version_that_accumulates_grad'] = named_dequantized_params[name]
            self.state[param]['is_quantized'] = isinstance(named_master_params.get(name), QuantizedWeight)
            if self.state[param]['is_quantized']:
                quantized_weight = named_master_params[name]
                assert isinstance(quantized_weight, QuantizedWeight)
                self.state[param]['quantized_weight'] = quantized_weight

    def step(self, *args, **kwargs):
        for param_group in self.param_groups:
            for param in param_group['params']:
                if self.state[param]['param_version_that_accumulates_grad'] is not param:
                    param.grad = self.state[param]['param_version_that_accumulates_grad'].grad

        original_output = super().step(*args, **kwargs)
        self._update_quantized_weights()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if self.state[param]['param_version_that_accumulates_grad'] is not param:
                    param.grad = None  # deference so that previous grads can be deleted on zero_grad(set_to_none=True)
                    if self.state[param]['is_quantized']:
                        self.state[param]['param_version_that_accumulates_grad'].data[...] = param['quantized_weight']()
                    else:
                        self.state[param]['param_version_that_accumulates_grad'].data[...] = param.data
        return original_output

    @torch.no_grad()
    def _update_quantized_weights(self):
        if torch.distributed.is_initialized():
            raise NotImplementedError("TODO FSDP SUPPORT")

        for param_group in self.param_groups:
            for param in param_group['params']:
                if self.state[param]['is_quantized']:
                    delta_weight = param    # a tensor that receives optimizer updates
                    quantized_weight = self.state[param]['quantized_weight']
                    dequantized_weight = quantized_weight()
                    reference_weight = dequantized_weight + delta_weight
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
                    delta_weight[...] = reference_weight - quantized_weight()

