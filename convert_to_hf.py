import json
import os
import re
import shutil

import torch
from tqdm.auto import trange
from transformers import AutoConfig, AutoModelForCausalLM

try:
    import safetensors
except ModuleNotFoundError:
    safetensors = None


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


@torch.inference_mode()
def pack_int_data(data: torch.IntTensor, nbits: int) -> torch.IntTensor:
    data[data >= 2 ** (nbits - 1)] -= 2**nbits
    return data.to(get_int_dtype(nbits))


def get_num_layers(config) -> int:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma":
            return config.num_hidden_layers
        case unknown_type:
            raise NotImplementedError(f"Can't get number of layers for {unknown_type}")


def get_layers_prefix(config) -> str:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma":
            return "model.layers"
        case unknown_type:
            raise NotImplementedError(f"Can't get layers prefix for {unknown_type}")


def get_converted_state_dict(config, nbits: int, in_path: os.PathLike) -> [dict, list[str]]:
    state_dict = {}
    linear_weights_not_to_quantize = []

    num_layers = get_num_layers(config)
    layers_prefix = get_layers_prefix(config)

    for i in trange(num_layers):
        layer = torch.load(os.path.join(in_path, f"{i}.pth"))
        for name, p in layer.named_parameters():
            if torch.is_floating_point(p.data):
                p.data = p.data.half()
            else:
                p.data = pack_int_data(p.data, nbits)
            if "quantized_weight." not in name:
                linear_weights_not_to_quantize.append(f"{layers_prefix}.{i}.{name}")
            else:
                name = re.sub("quantized_weight.", "", name)
            state_dict[f"{layers_prefix}.{i}.{name}"] = p.data

    for key, value in torch.load(os.path.join(in_path, "not_quantized_weights.pt")).items():
        state_dict[key] = value.half()
        linear_weights_not_to_quantize.append(key)

    if "lm_head.weight" not in linear_weights_not_to_quantize:
        linear_weights_not_to_quantize.append("lm_head.weight")

    return state_dict, linear_weights_not_to_quantize


def get_metadata(in_path: os.PathLike) -> dict:
    quant_args = torch.load(os.path.join(in_path, "args.pt"))
    return {
        "nbits_per_codebook": quant_args["nbits_per_codebook"],
        "num_codebooks": quant_args["num_codebooks"],
        "out_group_size": quant_args["out_group_size"],
        "in_group_size": quant_args["in_group_size"],
    }


def update_config(config_dict: dict, aqlm_metadata: dict[str, int], linear_weights_not_to_quantize: list[str]):
    config_dict["quantization_config"] = {
        "quant_method": "aqlm",
        "nbits_per_codebook": aqlm_metadata["nbits_per_codebook"],
        "num_codebooks": aqlm_metadata["num_codebooks"],
        "out_group_size": aqlm_metadata["out_group_size"],
        "in_group_size": aqlm_metadata["in_group_size"],
        "linear_weights_not_to_quantize": linear_weights_not_to_quantize,
    }
    config_dict["torch_dtype"] = "float16"
    return config_dict


def add_inference_code(model_type: str, save_path: os.PathLike):
    if os.path.isdir(f"./transformers/{model_type}"):
        shutil.copytree(f"./transformers/{model_type}", save_path, dirs_exist_ok=True)
    else:
        print(f"No predefined PreTrainedModel exists for {model_type}. You'll have to copy-paste some code yourself.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model",
        type=str,
        help="Path to the model to base config on, as in AutoConfig.from_pretrained()",
    )
    parser.add_argument(
        "in_path",
        type=str,
        help="Path of the checkpoint to convert",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Path to save HF compatible checkpoint to",
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Whether to save in safetensors format",
    )
    args = parser.parse_args()

    old_config = AutoConfig.from_pretrained(args.model)
    metadata = get_metadata(args.in_path)

    state_dict, linear_weights_not_to_quantize = get_converted_state_dict(
        old_config, metadata["nbits_per_codebook"], args.in_path
    )
    torch.save(state_dict, os.path.join(args.out_path, "pytorch_model.bin"))

    new_config_dict = update_config(old_config.to_diff_dict(), metadata, linear_weights_not_to_quantize)
    with open(os.path.join(args.out_path, "config.json"), "w") as config_file:
        json.dump(new_config_dict, config_file, indent=4)

    # convert to safetensors
    if args.save_safetensors:
        assert safetensors
        model = AutoModelForCausalLM.from_pretrained(args.out_path, trust_remote_code=True, torch_dtype=torch.float16)
        shutil.rmtree(args.out_path)
        model.save_pretrained(args.out_path)
