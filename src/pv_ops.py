"""Module containing utilities for straight-through fine-tuning of language models"""
from copy import deepcopy
from itertools import chain
from typing import Optional, Iterable, Dict

import torch
import torch.nn as nn
import transformers
from torch.optim.optimizer import StateDict

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
    """A wrapper for any PyTorch optimizer that enables updates on quantized parameters"""
    def __init__(self, named_params: Dict[str, nn.Parameter], *args,
                 named_quantized_weights: Dict[str, QuantizedWeight],
                 max_code_change_fraction: float,
                 delta_dtype: Optional[torch.dtype] = None,
                 **kwargs):
        optimized_params = dict()
        for name, param in named_params.items():
            if name in named_quantized_weights:
                param_delta = torch.zeros_like(param, dtype=delta_dtype, requires_grad=True)  # sharded alongside FSDP
                optimized_params[name] = param_delta  # accumulator for optimizer updates to dequantized weight
                # note: we store deltas (updated - quantized) instead of raw parameters to better handle half precision
            else:
                optimized_params[name] = param

        super().__init__(list(optimized_params.values()), *args, **kwargs)

        unmatched_quantized_params = dict(named_quantized_weights)
        for name, param in named_params.items():
            self.state[param]['name'] = name
            self.state[param]['is_quantized'] = name in named_quantized_weights
            if self.state[param]['is_quantized']:
                quantized_weight = unmatched_quantized_params.pop(name)
                assert isinstance(quantized_weight, QuantizedWeight)
                self.state[param]['quantized_weight'] = quantized_weight
                unmatched_quantized_params.pop(name)
        assert len(unmatched_quantized_params) == 0, f"Unmatched quantized_params: {unmatched_quantized_params.keys()}"
        self.max_code_change_fraction = max_code_change_fraction

    def step(self, *args, **kwargs):
        original_output = super().step(*args, **kwargs)
        for param_group in self.param_groups:
            for param in param_group['params']:
                if self.state[param]['is_quantized']:
                    raise NotImplementedError()



        return original_output