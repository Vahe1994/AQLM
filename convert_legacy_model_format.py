"""
This abomination converts between one of several quantized model formats to the same format as returned by main.py .
This code exists because we failed to produce a single data format for quantized model.
We should eventually switch to saving all models in the same data format. Once we do, this file should be deleted.
"""
import argparse
import os
import warnings
from copy import deepcopy

import torch
import transformers.models
from torch import nn

from src.aq import QuantizedLinear, QuantizedWeight
from src.modelutils import get_model, save_quantized_model
from src.utils import is_signed


def load_quantized_model_with_old_pickle(base_model_name: str, quantized_model_name: str, **kwargs):
    """Hacky way to allow compatibility between old *pickled* layers and new transformers"""
    # because patching it for the fourth time is better than writing a proper saver once >.<
    import transformers.activations

    if not hasattr(transformers.activations, "SiLUActivation"):
        transformers.activations.SiLUActivation = deepcopy(torch.nn.SiLU)
        transformers.activations.SiLUActivation.inplace = False
        # https://github.com/huggingface/transformers/issues/28496
    if not hasattr(transformers.models.llama.modeling_llama.LlamaAttention, "attention_dropout"):
        transformers.models.llama.modeling_llama.LlamaAttention.attention_dropout = 0
    quantized_model = get_model(base_model_name, None, **kwargs)
    quantized_model_src = get_model(base_model_name, quantized_model_name, **kwargs)
    for module in quantized_model_src.modules():
        if isinstance(module, QuantizedWeight) and not hasattr(module, "codes_storage"):
            module.codes_storage = None  # backwards compatibility with older pickled snapshots

    lut = {}
    for name, module in quantized_model_src.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, QuantizedWeight):
                lut[name + "." + child_name] = child_module
    print(f"found {len(lut)} quantized weight matrices")
    for name, module in quantized_model.named_modules():
        for child_name, child_module in module.named_children():
            if name + "." + child_name + ".quantized_weight" in lut:
                quantized_weight = lut.pop(name + "." + child_name + ".quantized_weight")
                assert isinstance(child_module, nn.Linear)
                setattr(module, child_name, QuantizedLinear(quantized_weight, bias=child_module.bias))
    assert not lut, list(lut.keys())
    quantized_model.load_state_dict(quantized_model_src.state_dict())
    warnings.warn("You should be ashamed of yourself.")
    return quantized_model


import functools


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def load_quantized_model_from_fdsp_checkpoint(base_model_name: str, fsdp_checkpoint_path: str, **kwargs):
    original_model = get_model(base_model_name, None, **kwargs)

    state_filenames = os.listdir(fsdp_checkpoint_path)

    non_quant_fname = "non_quantized_state_dict.pth"
    non_quant_path = os.path.join(fsdp_checkpoint_path, non_quant_fname)
    non_quant_states = torch.load(non_quant_path)

    incomp_keys = original_model.load_state_dict(non_quant_states, strict=False)
    assert not incomp_keys.unexpected_keys

    missing_keys = list()
    for module_name, module in original_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        assert not module.bias
        state_fname = f"{module_name}.weight.pth"

        if state_fname not in state_filenames:
            missing_keys.append(module_name)
            continue

        state_path = os.path.join(fsdp_checkpoint_path, state_fname)
        quantized_weight = torch.load(state_path, map_location="cpu")
        quantized_linear = QuantizedLinear(quantized_weight, bias=None)
        rsetattr(original_model, module_name, quantized_linear)

    return original_model


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--code_dtype",
        type=str,
        default=None,
        help="if specified, cast quantized layers' codes to this dtype; default = keep loaded dtype",
    )
    parser.add_argument(
        "--p_finetuned_state_dict",
        type=str,
        default=None,
        help="path to quantized model state dict saved by the old FSDP finetuning code",
    )
    parser.add_argument(
        "--pv_fsdp_dir",
        type=str,
        default=None,
        help="path to quantized model state dict saved by the old FSDP finetuning code",
    )
    parser.add_argument(
        "--monkeypatch_old_pickle",
        action="store_true",
        help="If set, load quantized_model in a hacky way that allows pickled models with older transformers/torch.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading base model.",
    )
    parser.add_argument("--save", type=str, required=True, help="Save the converted quantized model here")

    args = parser.parse_args()
    assert args.p_finetuned_state_dict or args.pv_fsdp_dir, "either one of those must be specified"
    print(f"{args.p_finetuned_state_dict=}, {args.pv_fsdp_dir=}")
    assert (args.p_finetuned_state_dict is not None) != (args.pv_fsdp_dir is not None)

    args.load_dtype = getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    args.code_dtype = getattr(torch, args.code_dtype) if args.code_dtype is not None else None

    if not args.monkeypatch_old_pickle:
        quantized_model = get_model(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
    elif args.p_finetuned_state_dict:
        quantized_model = load_quantized_model_with_old_pickle(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
    elif args.pv_fsdp_dir:
        quantized_model = load_quantized_model_from_fdsp_checkpoint(
            args.base_model,
            args.pv_fsdp_dir,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
        )

    for module in quantized_model.modules():
        if isinstance(module, QuantizedWeight):
            if not hasattr(module, "codes_storage"):
                module.codes_storage = None
            if module.codes is None:
                module.unwrap_codes_()
            assert module.codes is not None
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(args.code_dtype).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)

    if args.p_finetuned_state_dict is not None:
        state_dict = torch.load(args.p_finetuned_state_dict, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".codes_storage.data")}
        status = quantized_model.load_state_dict(state_dict, strict=False)
        assert all(key.endswith("codes") for key in status.missing_keys)
        assert not status.unexpected_keys
        del state_dict, status  # note: in this case, it is okay not to load codes since P step does not change them

    save_quantized_model(quantized_model, args.save)


if __name__ == "__main__":
    main()
