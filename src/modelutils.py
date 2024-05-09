import math
import os
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import transformers
from accelerate import dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM

from src.aq import QuantizedWeight, QuantizedLinear

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt' and 'falcon' are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi", "mistral", "mixtral", "gemma", "cohere")


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring


def dispatch_quantized_model(model):
    num_devices = torch.cuda.device_count()
    device_map = {"model.embed_tokens": 0, "model.norm": num_devices - 1, "lm_head": 0}
    num_layers = len(get_layers(model))
    layers_per_device = math.ceil(num_layers / num_devices)
    for layer_id in range(num_layers):
        device_id = layer_id // layers_per_device
        device_map[f"model.layers.{layer_id}"] = device_id
    model = dispatch_model(model, device_map)
    # for some reason dispatch doesn't put this modules on needed device
    model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
    model.lm_head = model.lm_head.to("cuda:0")
    return model


def get_model(
    model_path, load_quantized=None, dtype="auto", device_map=None, attn_implementation=None, trust_remote_code=False
):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    model_kwargs = {}
    # this argument is avaialbe only for transformers >= 4.38.0
    if transformers.__version__ >= "4.38.0":
        model_kwargs["attn_implementation"] = attn_implementation

    with suspend_nn_inits():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            # defer distribution if loading quantized
            device_map=None if load_quantized else device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
            **model_kwargs,
        )
        if load_quantized:
            print("Initializing model with random weights...")
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
            if device_map == "auto":
                assert model.config.model_type in LLAMA_LIKE, "Dispatching is implemented only for Llama-like models."
                model = dispatch_quantized_model(model)
        else:
            print("Loading pretrained model ...")

    print("Model loaded sucсessfully ...")

    return model


def get_model_head(model):
    head = torch.nn.ModuleList()
    if model.config.model_type in LLAMA_LIKE:
        if model.model.norm is not None:
            head.append(model.model.norm)
        head.append(model.lm_head)
    elif model.config.model_type.lower() in FALCON_TYPES:
        if model.transformer.ln_f is not None:
            head.append(model.transformer.ln_f)
        head.append(model.lm_head)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            head.append(model.model.decoder.final_layer_norm)
        if model.model.decoder.project_out is not None:
            head.append(model.model.decoder.project_out)
        head.append(model.lm_head)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return head


def get_lm_logits(inps_, model):
    if model.config.model_type in LLAMA_LIKE:
        hidden_states = inps_.unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type.lower() in FALCON_TYPES:
        hidden_states = inps_.unsqueeze(0)
        if model.transformer.ln_f is not None:
            hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "opt":
        hidden_states = inps_.unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    return lm_logits


def get_layers(model):
    if model.config.model_type in LLAMA_LIKE:
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    elif model.config.model_type == "opt":
        return model.model.decoder.layers
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def find_sublayers(module, layers=(nn.Conv2d, nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_sequential_groups(model):
    if model.config.model_type in LLAMA_LIKE:
        assert "mixtral" not in model.config.model_type.lower()  # check that this is not mixtral
        return [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]
    elif model.config.model_type.lower() in FALCON_TYPES:
        return [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
        ]
    elif model.config.model_type == "opt":
        return [
            ["self_attn.q_proj"],
            ["self_attn.k_proj"],
            ["self_attn.v_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))


def read_quant_weight_from_file(load_path, block_i, layer_name, device):
    return torch.load(load_path + "/" + str(block_i) + "/" + layer_name, map_location=device)


def load_linear_layers(layer, quant_layer, model):
    layer_ident = {}
    for submodule in layer.modules():
        for child_name, child_module in submodule.named_children():
            print(child_name, "child_name", layer_ident)
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) or "norm" in child_name:
                if child_name in layer_ident:
                    layer_ident[child_name] += 1
                else:
                    layer_ident[child_name] = 1
                quant_count = 0
                print("Finding to dequantize ", child_name)
                for quant_submodule in quant_layer.modules():
                    for quant_child_name, quant_child_module in quant_submodule.named_children():
                        if quant_child_name == child_name:
                            quant_count += 1
                            if quant_count != layer_ident[child_name]:
                                continue
                            print(quant_child_name, quant_child_module)
                            if ("gate" in child_name.lower()) and ("mixtral" in model.config.model_type.lower()):
                                print("gate", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                                continue
                            if "norm" in child_name and not isinstance(child_module, (nn.Conv2d, nn.Linear)):
                                print("norm", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                            else:
                                print(child_name)
                                child_module.weight.data = (
                                    quant_child_module.quantized_weight()
                                    .data.to(child_module.weight.dtype)
                                    .to(child_module.weight.device)
                                )
                            # Bias is not taked into account
    return layer


def load_dequantized_model(model, load_path):
    """Load quantized model by dequantizing it"""
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        print("layer", layer_index)
        layer = layers[layer_index]
        quant_layer = torch.load(os.path.join(load_path, str(layer_index) + ".pth"), map_location="cpu")
        for module in quant_layer.modules():
            if isinstance(module, QuantizedWeight):
                if not hasattr(module, 'codes_storage'):
                    module.codes_storage = None  # backwards compatibility
        layers[layer_index] = load_linear_layers(layer, quant_layer, model)
    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def load_quantized_model(model, load_path):
    """Load quantized model"""

    for layer_index in range(len(model.model.layers)):
        model.model.layers[layer_index] = torch.load(
            os.path.join(load_path, str(layer_index) + ".pth"),
            map_location=model.model.layers[layer_index].input_layernorm.weight.device,
        )
        for module in model.model.layers[layer_index].modules():
            if isinstance(module, QuantizedWeight):
                if not hasattr(module, 'codes_storage'):
                    module.codes_storage = None  # backwards compatibility

    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def save_not_quantized_weights(model: nn.Module, save_dir: str):
    already_saved_weights = set()
    for layer in get_layers(model):
        for param in layer.parameters():
            already_saved_weights.add(param)
    not_quantized_weights = {
        name: param for name, param in model.named_parameters() if param not in already_saved_weights
    }
    torch.save(not_quantized_weights, os.path.join(save_dir, "not_quantized_weights.pt"))


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
