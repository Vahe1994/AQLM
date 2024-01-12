import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt' and 'falcon' are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi", "mistral")


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


def get_model(model_path, load_quantized=None, dtype="auto", model_seqlen=2048):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=True).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    else:
        dtype = getattr(torch, dtype)

    with suspend_nn_inits():
        if load_quantized:
            print("Initializing model with random weights...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  # consider trust_remote_code=True
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype).eval()
            print("Loading quantized model ...")
            model = load_quantized_model(model, load_quantized)
        else:
            print("Loading pretrained model ...")
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
    # Please verify correcttess #TODO[aqlm]
    model.seqlen = model_seqlen
    print("Model loaded sucessfully ...")

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


def load_linear_layers(layer, quant_layer):
    for submodule in layer.modules():
        for child_name, child_module in submodule.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) or "norm" in child_name:
                for quant_submodule in quant_layer.modules():
                    for quant_child_name, quant_child_module in quant_submodule.named_children():
                        if quant_child_name == child_name:
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
        layers[layer_index] = load_linear_layers(layer, quant_layer)
    model.load_state_dict(torch.load(load_path + "/not_quantized_weights.pt"), strict=False)
    return model


def load_quantized_model(model, load_path):
    """Load quantized model"""

    for layer_index in range(len(model.model.layers)):
        print(model.model.layers[layer_index].input_layernorm.weight.device)
        model.model.layers[layer_index] = torch.load(
            os.path.join(load_path, str(layer_index) + ".pth"),
            map_location=model.model.layers[layer_index].input_layernorm.weight.device,
        )
    model.load_state_dict(torch.load(os.path.join(load_path, "/not_quantized_weights.pt")), strict=False)
    return model
