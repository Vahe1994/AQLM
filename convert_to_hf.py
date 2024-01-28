import json
import os
import re

import torch
from tqdm.auto import trange

from src.saving import add_inference_code, update_config
from src.utils import pack_int_data
from transformers import AutoConfig, PretrainedConfig


def get_num_layers(config) -> int:
    match config.model_type:
        case "llama" | "mistral":
            return config.num_hidden_layers
        case unknown_type:
            raise NotImplementedError(f"Can't get number of layers for {unknown_type}")


def get_layers_prefix(config) -> str:
    match config.model_type:
        case "llama" | "mistral":
            return "model.layers"
        case unknown_type:
            raise NotImplementedError(f"Can't get layers prefix for {unknown_type}")


def get_converted_state_dict(config, nbits: int, in_path: os.PathLike) -> dict:
    state_dict = {}

    num_layers = get_num_layers(config)
    layers_prefix = get_layers_prefix(config)

    for i in trange(num_layers):
        layer = torch.load(os.path.join(in_path, f"{i}.pth"))
        for name, p in layer.named_parameters():
            if torch.is_floating_point(p.data):
                p.data = p.data.half()
            else:
                p.data = pack_int_data(p.data, nbits)
            name = re.sub("quantized_weight.", "", name)
            state_dict[f"{layers_prefix}.{i}.{name}"] = p.data

    for key, value in torch.load(os.path.join(in_path, "not_quantized_weights.pt")).items():
        state_dict[key] = value.half()

    return state_dict


def get_metadata(in_path: os.PathLike) -> dict:
    quant_args = torch.load(os.path.join(in_path, "args.pt"))
    return {
        "nbits_per_codebook": quant_args["nbits_per_codebook"],
        "num_codebooks": quant_args["num_codebooks"],
        "out_group_size": quant_args["out_group_size"],
        "in_group_size": quant_args["in_group_size"],
    }


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
    args = parser.parse_args()

    old_config = AutoConfig.from_pretrained(args.model)
    print(type(old_config))
    metadata = get_metadata(args.in_path)

    add_inference_code(old_config.model_type, args.out_path)

    state_dict = get_converted_state_dict(old_config, metadata["nbits_per_codebook"], args.in_path)
    torch.save(state_dict, os.path.join(args.out_path, "pytorch_model.bin"))

    new_config = update_config(old_config, metadata)
    with open(os.path.join(args.out_path, "config.json"), "w") as config_file:
        json.dump(new_config.to_dict(), config_file, indent=4)
