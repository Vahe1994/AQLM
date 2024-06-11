"""Utilities for saving and loading model and training state"""
import argparse
import os
from typing import Tuple

import torch
import torch.optim
import transformers
from torch import nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, StateDictType, FullStateDictConfig

from convert_legacy_model_format import load_quantized_model_with_old_pickle
from src.aq import QuantizedWeight
from src.aq_ops import is_signed, IntCodes

from src.modelutils import get_model
from src.pv_optimizer import StraightThroughAdamW
from src.pv_utils import infer_module_classes, create_dequantized_model


def load_teacher_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    """Load unquantized model with frozen parameters"""
    base_model = get_model(
        args.base_model, load_quantized=None, dtype=args.load_dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(dtype=args.load_dtype if args.load_dtype != 'auto' else None)
    base_model.train(False)
    for param in base_model.parameters():
        param.requires_grad = False

    base_model.config.use_cache = False
    transformer_block_types = infer_module_classes(base_model, args.block_type)
    return FullyShardedDataParallel(
        base_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, transformer_block_types),
        device_id=device
    )


def load_student_model(
        args: argparse.Namespace, device: torch.device, dequantize: bool
) -> Tuple[FullyShardedDataParallel, dict]:
    """
    load student model for fine-tuning. If dequantize is set, dequantize all quantized weights to accumulate full grads
    """
    if not args.monkeypatch_old_pickle:
        student_model = get_model(
            args.base_model, args.quantized_model, dtype=args.load_dtype, trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation
        ).to(args.master_dtype)  # master parameters
    else:
        student_model = load_quantized_model_with_old_pickle(
            args.base_model, args.quantized_model, dtype=args.load_dtype, trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation
        ).to(args.master_dtype)

    student_model.config.use_cache = False
    student_model.train(True)  # note: HF gradient checkpoints do not work for some models without train(True); see
    # https://github.com/huggingface/transformers/blob/2d92db8/src/transformers/models/llama/modeling_llama.py#L1006
    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        student_model.enable_input_require_grads()

    # convert QuantizedModel state dict to make it compatible with FSDP
    for name, module in student_model.named_modules():
        if isinstance(module, QuantizedWeight):
            assert module.codes is not None
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(args.code_dtype).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)
    assert any(isinstance(module, IntCodes) for module in student_model.modules())

    if dequantize:
        student_model, named_quantized_params = create_dequantized_model(
            student_model, dequantized_dtype=args.amp_dtype, reuse_non_quantized=True)
    else:
        named_quantized_params = {
            name: module for name, module in student_model.named_parameters() if isinstance(module, QuantizedWeight)
        }

    transformer_block_types = list(infer_module_classes(student_model, args.block_type))
    layernorm_types = list(transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    extra_block_types = list()
    for extra_module_name in args.wrap_separately:
        extra_block_types.extend(infer_module_classes(student_model, extra_module_name))
    block_types_to_wrap = tuple(set(transformer_block_types + layernorm_types + extra_block_types))
    if torch.distributed.get_rank() == 0:
        print(f"Blocks to be wrapped separately: {block_types_to_wrap}\n")

    mixed_precision = None
    if args.use_fsdp_amp:
        assert args.amp_dtype is not None, "requested to use_fsdp_amp, but amp_dtype is not None"
        block_types_for_amp_to_ignore = tuple(set(layernorm_types + extra_block_types))
        if torch.distributed.get_rank() == 0:
            print(f"Blocks excluded from AMP: {block_types_for_amp_to_ignore}\n")
        mixed_precision = MixedPrecision(
            param_dtype=args.amp_dtype,
            reduce_dtype=args.amp_dtype,
            _module_classes_to_ignore=block_types_for_amp_to_ignore
        )
    else:
        if torch.distributed.get_rank() == 0:
            print(f"Not using FSDP native MixedPrecision; Local amp_dtype={args.amp_dtype}.")

    student_model = FullyShardedDataParallel(
        student_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, block_types_to_wrap),
        mixed_precision=mixed_precision,
        use_orig_params=True,
        device_id=device,
    )
    return student_model, named_quantized_params


def save_training_state(
        args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Save model, optimizer state dict and training metadata to be loaded via load_training_state"""
    if args.save is None:
        return
    rank = torch.distributed.get_rank()
    os.makedirs(args.save, exist_ok=True)
    if rank == 0:
        print(f"Saving snapshot to {args.save}")
        torch.save(metadata, os.path.join(args.save, 'metadata.pt'))
    with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
        torch.save(quantized_model.state_dict(), os.path.join(args.save, f'quantized_model_state_dict_rank{rank}.pt'))
        # model saves non-quantized weights and dequantized versions of QuantizedWeight; the latter is not necessary
    torch.save(optimizer.state_dict(), os.path.join(args.save, f'optimizer_state_dict_rank{rank}.pt'))
    # optimizer state dict saves statistics QuantizedWeight instances and straight-through buffers
    if args.on_save:
        exec(args.on_save)


def load_training_state(
    args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: torch.optim.Optimizer
):
    """Load model, optimizer state dict and metadata saved via save_training_state; update parameters in-place"""
    rank = torch.distributed.get_rank()
    if args.save is None or not os.path.exists(args.save):
        if args.save is not None and rank == 0:
            print(f"No checkpoint found at {args.save}")
    else:
        with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
            # this loads non-quantized weights and de-quantized versions of QuantizedWeight instances
            state_dict_ptr = quantized_model.state_dict()
            loaded_state_dict = torch.load(os.path.join(args.save, f'quantized_model_state_dict_rank{rank}.pt'))
            with torch.no_grad():
                for key in state_dict_ptr:
                    state_dict_ptr[key].copy_(loaded_state_dict.pop(key))
                assert len(loaded_state_dict) == 0, f"Unused keys:, {tuple(loaded_state_dict.keys())}"
            del state_dict_ptr, loaded_state_dict

        # v-- loading optimizer state dict also loads all QuantizedWeights and straight-through buffers
        optimizer.load_state_dict(torch.load(
            os.path.join(args.save, f'optimizer_state_dict_rank{rank}.pt'),
            map_location='cpu'))
        metadata.update(torch.load(os.path.join(args.save, 'metadata.pt')))
        if args.eval_datasets is not None and metadata['early_stop_on'] not in args.eval_datasets:
            if rank == 0:
                print(f"Stopping criterion {metadata['early_stop_on']} is not in eval_datasets; resetting best loss.")
            metadata['early_stop_on'] = next(iter(args.eval_datasets))
            metadata['best_eval_perplexity'] = float('inf')
            metadata['best_step'] = 0
        if rank == 0:
            print(f"Loaded training state from {args.save}: {metadata}")


def save_model(args: argparse.Namespace, student_model: FullyShardedDataParallel, optimizer: torch.optim.Optimizer):
    """Save model for either P- or PV-Tuning using the appropriate saver"""
    if isinstance(optimizer, StraightThroughAdamW):
        save_pv_model(args, student_model, optimizer)
    else:
        assert any(isinstance(module, QuantizedWeight) for module in student_model.modules())
        save_p_model(args, student_model)


def save_pv_model(
    args: argparse.Namespace, dequantized_model: FullyShardedDataParallel, optimizer: StraightThroughAdamW
):
    """Save consolidated model from PV tuning, can be exported later via convert_legacy_model_format.py"""
    output_path = os.path.join(args.save, "best_model")
    os.makedirs(output_path, exist_ok=True)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    local_quantized_weight_names = set()
    for name, quantized_weight in optimizer.iterate_local_quantized_weights():
        torch.save(quantized_weight, os.path.join(output_path, f"{name}.pth"))
        local_quantized_weight_names.add(name)

    quantized_weight_names_by_rank = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(local_quantized_weight_names, quantized_weight_names_by_rank, dst=0)

    with FullyShardedDataParallel.state_dict_type(
            dequantized_model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        model_state_dict = dequantized_model.state_dict()
        if rank == 0:
            all_quantized_weight_names = set()
            for local_quantized_weight_names in quantized_weight_names_by_rank:
                all_quantized_weight_names |= set(local_quantized_weight_names)

            non_quantized_state_dict = dict()
            for name, tensor in model_state_dict.items():
                if name in all_quantized_weight_names:
                    all_quantized_weight_names.remove(name)  # do not save de-quantized versions of quantized weights
                else:
                    non_quantized_state_dict[name] = tensor
            assert len(all_quantized_weight_names) == 0, f"mismatched names: {all_quantized_weight_names}"
            torch.save(non_quantized_state_dict, os.path.join(output_path, "non_quantized_state_dict.pth"))
    torch.distributed.barrier()
    if rank == 0:
        print(f"Saved best model shards to {output_path}")


def save_p_model(args: argparse.Namespace, quantized_model: FullyShardedDataParallel):
    """Save consolidated model state dict from P-only tuning, can be exported via convert_legacy_model_format.py"""
    os.makedirs(args.save, exist_ok=True)
    rank = torch.distributed.get_rank()
    with FullyShardedDataParallel.state_dict_type(
            quantized_model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    ):
        model_state_dict = quantized_model.state_dict()
        if rank == 0:
            torch.save(model_state_dict, os.path.join(args.save, f'best_model_state_dict.pt'))
    torch.distributed.barrier()
    if rank == 0:
        print(f"Saved best model state dict to {os.path.join(args.save, f'best_model_state_dict.pt')}")
