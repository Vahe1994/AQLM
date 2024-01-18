import os
import shutil

import torch
from torch import nn

from transformers import PreTrainedModel


def update_config(model: PreTrainedModel, args):
    old_config = model.config
    old_config_type = type(old_config)
    old_model_type = old_config.model_type
    new_model_type = f"{old_model_type}_aqlm"

    class AqlmConfig(old_config_type):
        model_type = new_model_type

        def __init__(
            self,
            aqlm: dict[str, int] = {
                "nbits_per_codebook": 16,
                "num_codebooks": 1,
                "out_group_size": 8,
                "in_group_size": 1,
            },
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.aqlm = aqlm

    config_dict = old_config.to_dict()
    config_dict["auto_map"] = {
        f"AutoConfig": f"configuration_{new_model_type}.{old_config.__class__.__name__}",
        "AutoModelForCausalLM": f"modeling_{new_model_type}.{model.__class__.__name__}",
    }
    del config_dict["_name_or_path"]

    new_config = AqlmConfig(
        {
            "nbits_per_codebook": args.nbits_per_codebook,
            "num_codebooks": args.num_codebooks,
            "out_group_size": args.out_group_size,
            "in_group_size": args.in_group_size,
        }
    )
    new_config.update(config_dict)

    model.config = new_config
    model.__class__.__name__ = model.__class__.__name__ + "_AQLM"


def add_inference_code(model: PreTrainedModel, save_path: os.PathLike):
    model_type = model.config.model_type

    shutil.copytree(f"./transformers/common", save_path, dirs_exist_ok=True)
    if os.path.isdir(f"./transformers/{model_type}"):
        shutil.copytree(f"./transformers/{model_type}", save_path, dirs_exist_ok=True)
    else:
        print(f"No predefined PreTrainedModel exists for {model_type}. You'll have to copy-paste some code yourself.")


def save_fresh_model(model: PreTrainedModel, replaced_linears: list[tuple[nn.Module, str, nn.Module]], args):
    for (submodule, child_name, quantized_linear) in replaced_linears:
        setattr(submodule, child_name, quantized_linear.finalize())

    update_config(model, args)
    model.save_pretrained(args.save)
    add_inference_code(model, args.save)
