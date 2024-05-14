import contextlib
import time
from copy import deepcopy
from itertools import chain
from typing import Tuple, Optional, Dict

import torch
import transformers

from torch import nn as nn

from src.aq import QuantizedLinear, QuantizedWeight


def infer_block_classes(model: nn.Module, block_type: str) -> Tuple[type[nn.Module], ...]:
    """find transformer block classes that should be wrapped with inner FullyShardedDataParallel (auto_wrap_policy)"""
    transformer_block_types = []
    for module in model.modules():
        if module.__class__.__name__ == block_type:
            transformer_block_types.append(type(module))
    if not transformer_block_types:
        raise ValueError(f"Could not find {block_type} among model layers")
    transformer_block_types = tuple(transformer_block_types)
    assert any(isinstance(module, transformer_block_types) for module in model.modules())
    return transformer_block_types


def create_dequantized_model(
        model: transformers.PreTrainedModel, *,
        reuse_non_quantized: bool,
        dequantized_dtype: Optional[torch.dtype] = None
):
    """
    Create a version of the model where all QuanizedWeight and derivative layers are de-quantized and cast to dtype.
    :param model: model to be dequantized (out-of-place)
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


def get_original_named_parameters_from_fsdp_module(dequantized_model) -> Dict[str, nn.Parameter]:
    return {name.replace('_fsdp_wrapped_module.', ''): param for name, param in dequantized_model.named_parameters()}


@contextlib.contextmanager
def print_runtime_stats(operation_name: str, enabled: bool = True, device: Optional[torch.device] = None):
    if not enabled:
        yield
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if device is None:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if torch.device.type == 'cuda':
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    yield
    if torch.device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(end=f"rank{rank} {operation_name} took {time.perf_counter() - start_time}\n")
