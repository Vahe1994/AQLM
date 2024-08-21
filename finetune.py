"""
Fine-tune an LLM that was previously quantized with AQLM;
based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import argparse
import os
from contextlib import nullcontext
from functools import partial
from typing import Dict, Optional, Tuple

import datasets
import torch
import torch.distributed
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import transformers
from torch import nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    StateDictType,
)
from tqdm.auto import tqdm

from convert_legacy_model_format import load_quantized_model_with_old_pickle
from src.aq import QuantizedWeight
from src.configurable_adam import ConfigurableAdamW
from src.datautils import evaluate_perplexity, get_loaders, group_texts, split_long_texts
from src.memory_efficient_loss import compute_kl_divergence_loss_values
from src.modelutils import get_model, is_model_for_causal_lm
from src.pv_optimizer import StraightThroughAdamW
from src.pv_utils import (
    YourQuantizedWeightIsInAnotherRank,
    create_dequantized_model,
    get_original_named_parameters_from_fsdp_module,
    infer_module_classes,
    split_quantized_weights_between_ranks,
)
from src.utils import IntCodes, is_signed, master_rank_first, one_rank_at_a_time

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def add_model_args(parser: argparse.ArgumentParser):
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
        "--monkeypatch_old_pickle",
        action="store_true",
        help="If set, load quantized_model in a hacky way that allows pickled models with older transformers/torch.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--master_dtype",
        type=str,
        default="float32",
        help="data type for storing master parameters and computing optimizer updates",
    )
    parser.add_argument(
        "--embed_dtype",
        type=str,
        default=None,
        help="data type for storing master input and output embeddings; defaults to master_dtype",
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=None,
        help="if specified, runs automated mixed precision with this dtype",
    )
    parser.add_argument(
        "--straight_through_buffer_dtype",
        type=str,
        default=None,
        help="data type for storing optimized straight through buffers, defaults to master_dtype",
    )
    parser.add_argument(
        "--code_dtype",
        type=str,
        default=None,
        help="if specified, cast quantized layers' codes to this dtype; default = keep loaded dtype",
    )
    parser.add_argument(
        "--block_type",
        type=str,
        required=True,
        help="string name of a transformer layer to wrap, e.g. LlamaDecoderLayer",
    )
    parser.add_argument(
        "--wrap_separately",
        type=str,
        nargs="*",
        default=[],
        help="module classes (by name, similar to block_type) that will be wrapped in a separate fsdp instance and do "
        "not participate in FSDP AMP (if used). Applies to the student (de)quantized model, not the teacher model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument(
        "--limit_parallel_inits",
        type=int,
        default=1,
        help="this many ranks (per host) initialize their model in parallel. This parameter is meant to save host RAM.",
    )


def add_finetuning_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--update_codes",
        action="store_true",
        help="If set, train discrete codes; if not, freeze them",
    )
    parser.add_argument(
        "--update_codebooks_and_scales",
        action="store_true",
        help="If set, train continuous parameters of quantized representations; if not, freeze them",
    )
    parser.add_argument(
        "--update_non_quantized_parameters",
        action="store_true",
        help="If set, train the non-quantized model parameters (layernorm scales, biases, logits); if not, freeze them",
    )
    parser.add_argument(
        "--force_dequantize",
        action="store_true",
        help="If set, the algorithm will create a de-quantized model instead of dequantizing weights just in time even"
        "when doing p-only tuning. This version only has effect if --update_codes is not set. Setting this will"
        " make the training run faster, but it will also use substantially more memory.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="finetuning learning rate for continuous params",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.90,
        help="Adam beta1 for continuous params",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Adam beta2 for continuous params",
    )

    parser.add_argument(
        "--code_lr",
        type=float,
        default=1e-2,
        help="finetuning learning rate for discrete codes",
    )
    parser.add_argument(
        "--code_beta1",
        type=float,
        default=0.0,
        help="Adam beta1 for discrete params",
    )
    parser.add_argument(
        "--code_beta2",
        type=float,
        default=0.95,
        help="Adam beta2 for discrete params",
    )
    parser.add_argument(
        "--delta_decay",
        type=float,
        default=0.0,
        help="Determines whether to use direct training, straight-through estimation or a mixture thereof. "
        "If delta_decay is 0, use straight-through estimation. If delta_decay is 1, do not use it at all. "
        "If between 0 and 1, every straight-through buffer will decay to the quantized weight with moving average."
        " straight_through_buffer = (1 - delta_decay) * straight_through_buffer + delta_decay * quantized_weight."
        " Please refer to the docstring of StraightThroughAdam for details.",
    )
    parser.add_argument(
        "--max_code_change_per_step",
        type=float,
        default=1e-2,
        help="Maximum number of code groups that can be changed during one update to codes. "
        "This constraint is enforced on a per-tensor level. If the weight is represented with multiple codes, "
        "changing any of the codes will count towards the limit. If more than this many code groups have changed, "
        "the algorithm will rollback the changes with least update norm until the constraint is satisfied.",
    )
    parser.add_argument(
        "--code_trust_ratio",
        type=float,
        default=None,
        help="By default, the optimizer can make arbitrary changes to quantized weights. If this parameter is set,"
        "the optimizer ensures that the change to quantized weights is not too large by undoing some of the change"
        "until ||new_quantized_weights - prev_quantized_weights|| / ||prev_quantized_weight|| <= code_trust_ratio."
        " See StraightThroughAdam docstring for details.",
    )
    parser.add_argument(
        "--force_code_update",
        action="store_true",
        help="If set, force discrete codes to change in the direction of optimizer update, even if previous codes"
        "were optimal in terms of MSE. See StraightThroughAdam docstring for details. Use when delta_decay==1.",
    )
    parser.add_argument(
        "--code_selection_temperature",
        type=float,
        default=0,
        help="If max_code_change_per_step or code_trust_ratio is set and code_selection_temperature=0, beam search will"
        " prioritize updating codes that have the largest continuosu update norm. If code_selection_temperature is"
        " not 0, sample a subset of codes for update stochastically. See StraightThroughAdam for details.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size when updating codes; higher is slower but more accurate. For single codebook, use beam_size=1",
    )
    parser.add_argument(
        "--code_adam_16bit",
        action="store_true",
        help="If set, adam statistics for codes will be stored as float16 (exp_avg and v_hat) or bfloat16(exp_avg_sq)",
    )
    parser.add_argument(
        "--offload_optimizer",
        action="store_true",
        help="If set, adam statistics will be offloaded to RAM",
    )
    parser.add_argument(
        "--offload_teacher_params",
        action="store_true",
        help="If set, the teacher model will be offloaded to RAM and paged using FSDP's CPUOffload",
    )
    parser.add_argument(
        "--offload_student_params",
        action="store_true",
        help="If set, the student model will be offloaded to RAM and paged using FSDP's CPUOffload",
    )
    parser.add_argument(
        "--limit_all_gathers",
        action="store_true",
        help="sets limit_all_gathers in both FSDP instances",
    )
    parser.add_argument(
        "--forward_prefetch",
        action="store_true",
        help="sets forward_prefetech in both FSDP instances",
    )
    parser.add_argument(
        "--lamb",
        action="store_true",
        help="If set, use Lamb (aka Adam with trust ratio)",
    )
    parser.add_argument(
        "--amsgrad",
        action="store_true",
        help="if True, use the AMSGrad variant of adam/lamb",
    )
    parser.add_argument(
        "--debias",
        action="store_true",
        default=None,
        help="Whether or not to debias optimizer statistics; defaults to True for adam and False for Lamb",
    )
    parser.add_argument(
        "--no_debias",
        action="store_false",
        dest="debias",
        help="Disable optimizer debiasing (see above)",
    )
    parser.add_argument(
        "--verbose_optimizer",
        action="store_true",
        help="If set, the optimizer will print beam search results, tensors norms, etc",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size - how many samples are processed per optimizer step, between all GPUs in total",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size - how many samples are processed per GPU per forward pass",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to apply gradient checkpointing for transformer blocks",
    )
    parser.add_argument(
        "--loss_tokens_per_chunk",
        type=int,
        default=None,
        help="If specified, compute LM logits and loss using gradient checkpointing in chunks of this size."
        "This option slows down loss computation, but reduces memory usage. Recommended for large vocabularies",
    )
    parser.add_argument(
        "--use_fsdp_amp",
        action="store_true",
        help="Whether to use FSDP native mixed precision (excluding registered layernorms and --wrap_separately).",
    )
    parser.add_argument(
        "--minimize_sync",
        action="store_true",
        help="if True, accumulate microbatch gradients locally and synchronize once per optimizer step. If False, "
        "synchronize after every step. This reduces communication overhead but increases memory usage. See "
        "https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument("--save", type=str, default=None, help="Path to save training snapshot.")
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Total number of training epochs (passes over calibration data) after which the training will conclude",
    )
    parser.add_argument(
        "--print_every_steps",
        type=int,
        default=None,
        help="print training metrics once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=None,
        help="evaluate once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=None,
        help="save state once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument("--keep_best_model", action="store_true", help="Save best model state separately")
    parser.add_argument(
        "--on_save",
        type=str,
        default=None,
        help="Optional callback (python code string) to call after each saved layer. Example: when "
        "training on preemptible compute, upload partially finetuned model and resume later",
    )


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Training dataset name (from HF datasets) or path to data where to extract calibration data from",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Training dataset split name, e.g. 'train'",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir for huggingface datasets",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="If set, re-run data preprocessing even if it is cached",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of CPU workers for preprocessing and data loading",
    )
    parser.add_argument(
        "--download_num_workers",
        type=int,
        default=None,
        help="Number of CPU workers for downloading the training dataset; overrides num_workers",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Number of CPU workers for preprocessing; overrides num_workers",
    )
    parser.add_argument(
        "--preprocessing_chunk_length",
        type=int,
        default=100_000,
        help="Texts exceeding this length will be split approximately in the middle",
    )
    parser.add_argument(
        "--preprocessing_keep_in_memory",
        action="store_true",
        help="If set, do not save intermediate preprocessing steps in memory",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading base model.",
    )
    parser.add_argument(
        "--save_dataset_and_exit",
        type=str,
        default=None,
        help="If not None, save tokenized dataset to this path and exit training immediately",
    )


def prepare_training_dataset(args: argparse.Namespace, tokenizer: transformers.PreTrainedTokenizer) -> datasets.Dataset:
    if os.path.exists(args.dataset_name):
        dataset = datasets.load_from_disk(args.dataset_name)
    else:
        dataset = datasets.load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=args.split,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
            num_proc=args.download_num_workers if args.download_num_workers is not None else args.num_workers,
            streaming=False,
        )

    def is_tokenized(dataset):
        return "input_ids" in dataset.column_names

    if is_tokenized(dataset):
        if torch.distributed.get_rank() == 0:
            print("Dataset already tokenized")
            assert len(dataset[0]["input_ids"]) == args.model_seqlen
        return dataset

    text_column_name = "text" if "text" in dataset.column_names else next(iter(dataset.column_names))

    if args.preprocessing_chunk_length is not None:
        dataset = dataset.map(
            lambda examples: {
                text_column_name: split_long_texts(examples[text_column_name], args.preprocessing_chunk_length)
            },
            batched=True,
            num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
            remove_columns=list(dataset.column_names),
            keep_in_memory=args.preprocessing_keep_in_memory,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Splitting dataset over newline into chunks of ~{args.preprocessing_chunk_length} characters",
        )

    tokenized_dataset = dataset.map(
        lambda example: tokenizer(example[text_column_name]),
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        remove_columns=list(dataset.column_names),
        keep_in_memory=args.preprocessing_keep_in_memory,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    lm_dataset = tokenized_dataset.map(
        partial(group_texts, block_size=args.model_seqlen, add_labels=False),
        batched=True,
        num_proc=args.preprocessing_num_workers if args.preprocessing_num_workers is not None else args.num_workers,
        keep_in_memory=args.preprocessing_keep_in_memory,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {args.model_seqlen}",
    )
    assert is_tokenized(lm_dataset)
    return lm_dataset


def load_teacher_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    """Load unquantized model with frozen parameters"""
    model = get_model(
        args.base_model,
        load_quantized=None,
        dtype=args.load_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(dtype=args.load_dtype if args.load_dtype != "auto" else None)
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False

    model.config.use_cache = False
    transformer_block_types = infer_module_classes(model, args.block_type)

    return wrap_model_with_fsdp_(
        model,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, transformer_block_types),
        cpu_offload=CPUOffload(offload_params=args.offload_teacher_params) if args.offload_teacher_params else None,
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        device_id=device,
    )


def load_student_model(
    args: argparse.Namespace, device: torch.device, dequantize: bool
) -> Tuple[FullyShardedDataParallel, Optional[Dict[str, QuantizedWeight]]]:
    """
    load student model for fine-tuning. If dequantize is set, dequantize all quantized weights to accumulate full grads
    """
    if not args.monkeypatch_old_pickle:
        student_model = get_model(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        ).to(
            args.master_dtype
        )  # master parameters
    else:
        student_model = load_quantized_model_with_old_pickle(
            args.base_model,
            args.quantized_model,
            dtype=args.load_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        ).to(args.master_dtype)

    if args.embed_dtype != args.master_dtype:
        student_model.set_output_embeddings(student_model.get_output_embeddings().to(args.embed_dtype))
        student_model.set_input_embeddings(student_model.get_input_embeddings().to(args.embed_dtype))

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
            student_model, dequantized_dtype=args.amp_dtype, reuse_non_quantized=True
        )
    else:
        named_quantized_params = None

    transformer_block_types = list(infer_module_classes(student_model, args.block_type))
    layernorm_types = list(transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    extra_block_types = list()
    for extra_module_name in args.wrap_separately:
        extra_block_types.extend(infer_module_classes(student_model, extra_module_name))
    block_types_to_wrap = tuple(
        set(
            transformer_block_types
            + layernorm_types
            + extra_block_types
            + [
                IntCodes,
            ]
        )
    )
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
            _module_classes_to_ignore=block_types_for_amp_to_ignore,
        )
    else:
        if torch.distributed.get_rank() == 0:
            print(f"Not using FSDP native MixedPrecision; Local amp_dtype={args.amp_dtype}.")

    student_model = wrap_model_with_fsdp_(
        student_model,
        use_orig_params=True,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, block_types_to_wrap),
        cpu_offload=CPUOffload(offload_params=args.offload_student_params) if args.offload_student_params else None,
        limit_all_gathers=args.limit_all_gathers,
        forward_prefetch=args.forward_prefetch,
        mixed_precision=mixed_precision,
        device_id=device,
    )

    if named_quantized_params is not None:
        if torch.distributed.get_world_size() > 1:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = split_quantized_weights_between_ranks(
                named_quantized_params, verify_checksums=False
            )
        for quantized_weight in named_quantized_params.values():
            if isinstance(quantized_weight, QuantizedWeight):
                quantized_weight.to(device)
            else:
                assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)

    return student_model, named_quantized_params


def wrap_model_with_fsdp_(
    model: transformers.PreTrainedModel, auto_wrap_policy: callable, **kwargs
) -> FullyShardedDataParallel:
    """Wrap a model *ForCausalLM components: transformer and lm_head are wrapped as FSDP instances"""
    assert isinstance(model, transformers.PreTrainedModel) and is_model_for_causal_lm(model)
    base_model, lm_head = model.base_model, model.get_output_embeddings()

    def _modified_auto_wrap_policy(module, recurse, **kwargs):
        return auto_wrap_policy(module, recurse, **kwargs) or (module in (base_model, lm_head))

    model = FullyShardedDataParallel(model, auto_wrap_policy=_modified_auto_wrap_policy, **kwargs)

    assert isinstance(model.module, transformers.PreTrainedModel)
    assert isinstance(model.base_model, FullyShardedDataParallel)
    assert isinstance(model.get_output_embeddings(), FullyShardedDataParallel)
    return model


def trigger_fsdp_lazy_init_(
    tokenizer: transformers.PreTrainedTokenizer,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
):
    """Trigger FullyShardedDataParallel lazy init in the correct order to allow both training and eval"""
    print("Initializing FSDP root")
    dummy_batch = tokenizer("I am the monument to all your sins", return_tensors="pt")
    dummy_batch = {k: v.to(device) for k, v in dummy_batch.items()}
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        with torch.no_grad():
            teacher_model(**dummy_batch)
        (student_model(**dummy_batch).logits * 0).sum().backward()


def create_pv_optimizer(
    args: argparse.Namespace,
    student_model: FullyShardedDataParallel,
    named_quantized_params: Dict[str, QuantizedWeight],
) -> torch.optim.Optimizer:
    """Create optimizer for PV-Tuning using a de-quantized student model and a dictionary of quantized weights"""
    named_dequantized_params = get_original_named_parameters_from_fsdp_module(student_model)
    opt_device = torch.device("cpu") if args.offload_optimizer else next(student_model.parameters()).device
    assert all(name in named_dequantized_params for name in named_quantized_params)
    return StraightThroughAdamW(
        named_dequantized_params=named_dequantized_params,
        named_quantized_params=named_quantized_params,
        update_codes=dict(
            lr=args.code_lr,
            betas=(args.code_beta1, args.code_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_sq_dtype=torch.bfloat16 if args.code_adam_16bit else args.master_dtype,
            v_hat_max_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_codes
        else None,
        update_codebooks_and_scales=dict(
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype,
            exp_avg_sq_dtype=args.master_dtype,
            v_hat_max_dtype=args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_codebooks_and_scales
        else None,
        update_non_quantized_parameters=dict(
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb,
            debias=args.debias,
            amsgrad=args.amsgrad,
            compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype,
            exp_avg_sq_dtype=args.master_dtype,
            v_hat_max_dtype=args.master_dtype,
            exp_avg_device=opt_device,
            exp_avg_sq_device=opt_device,
            v_hat_max_device=opt_device,
        )
        if args.update_non_quantized_parameters
        else None,
        delta_decay=args.delta_decay,
        max_code_change_per_step=args.max_code_change_per_step,
        force_code_update=args.force_code_update,
        code_trust_ratio=args.code_trust_ratio,
        beam_size=args.beam_size,
        straight_through_buffer_dtype=args.straight_through_buffer_dtype,
        verbose=args.verbose_optimizer,
    )


def create_p_optimizer(args: argparse.Namespace, student_model: FullyShardedDataParallel) -> torch.optim.Optimizer:
    """Create optimizer for training only continuous parameters of a quantized model"""
    quantized_weight_continuous_parameters = set()
    for module in student_model.modules():
        if isinstance(module, QuantizedWeight):
            for param in module.parameters():
                if torch.is_floating_point(param) and param.requires_grad:
                    quantized_weight_continuous_parameters.add(param)
    all_trainable_params = []
    if args.update_codebooks_and_scales:
        all_trainable_params.extend(
            param for param in student_model.parameters() if param in quantized_weight_continuous_parameters
        )  # use iteration instead of simply adding list(set) to ensure deterministic order of parameters
    if args.update_non_quantized_parameters:
        all_trainable_params.extend(
            param
            for param in student_model.parameters()
            if torch.is_floating_point(param)
            and param.requires_grad
            and param not in quantized_weight_continuous_parameters
        )
    if args.update_codes:
        raise RuntimeError("When asked to update_codes, one should create_pv_optimizer, but this is create_p_optimizer")
    assert len(all_trainable_params) > 0, (
        "found no trainable parameters. Did you specify update_codes, "
        "update_codebooks_and_scales or update_non_quantized_parameters?"
    )
    opt_device = torch.device("cpu") if args.offload_optimizer else next(student_model.parameters()).device
    return ConfigurableAdamW(
        params=list(all_trainable_params),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        lamb=args.lamb,
        debias=args.debias,
        amsgrad=args.amsgrad,
        compute_dtype=args.master_dtype,
        exp_avg_dtype=args.master_dtype,
        exp_avg_sq_dtype=args.master_dtype,
        v_hat_max_dtype=args.master_dtype,
        exp_avg_device=opt_device,
        exp_avg_sq_device=opt_device,
        v_hat_max_device=opt_device,
    )


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
        torch.save(metadata, os.path.join(args.save, "metadata.pt"))
    with FullyShardedDataParallel.state_dict_type(quantized_model, StateDictType.LOCAL_STATE_DICT):
        torch.save(quantized_model.state_dict(), os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt"))
        # model saves non-quantized weights and dequantized versions of QuantizedWeight; the latter is not necessary
    torch.save(optimizer.state_dict(), os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"))
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
            loaded_state_dict = torch.load(os.path.join(args.save, f"quantized_model_state_dict_rank{rank}.pt"))
            with torch.no_grad():
                for key in state_dict_ptr:
                    state_dict_ptr[key].copy_(loaded_state_dict.pop(key))
                assert len(loaded_state_dict) == 0, f"Unused keys:, {tuple(loaded_state_dict.keys())}"
            del state_dict_ptr, loaded_state_dict

        # v-- loading optimizer state dict also loads all QuantizedWeights and straight-through buffers
        optimizer.load_state_dict(
            torch.load(os.path.join(args.save, f"optimizer_state_dict_rank{rank}.pt"), map_location="cpu")
        )
        metadata.update(torch.load(os.path.join(args.save, "metadata.pt")))
        if args.eval_datasets is not None and metadata["early_stop_on"] not in args.eval_datasets:
            if rank == 0:
                print(f"Stopping criterion {metadata['early_stop_on']} is not in eval_datasets; resetting best loss.")
            metadata["early_stop_on"] = next(iter(args.eval_datasets))
            metadata["best_eval_perplexity"] = float("inf")
            metadata["best_step"] = 0
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
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
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
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = quantized_model.state_dict()
        if rank == 0:
            torch.save(model_state_dict, os.path.join(args.save, f"best_model_state_dict.pt"))
    torch.distributed.barrier()
    if rank == 0:
        print(f"Saved best model state dict to {os.path.join(args.save, f'best_model_state_dict.pt')}")


def compute_loss_on_batch(
    batch: dict,
    teacher_model: FullyShardedDataParallel,
    student_model: FullyShardedDataParallel,
    *,
    amp_dtype: Optional[torch.dtype],
    max_tokens_per_chunk: Optional[int],
) -> torch.Tensor:
    if max_tokens_per_chunk is not None:  # chunked inference, transformer and lm head must be separate FSDP instances
        with torch.no_grad():
            teacher_hidden_states = teacher_model.base_model(**batch).last_hidden_state
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_hidden_states = student_model.base_model(**batch).last_hidden_state
            return compute_kl_divergence_loss_values(
                student_hidden_states=student_hidden_states,
                student_lm_head=student_model.get_output_embeddings(),
                teacher_hidden_states=teacher_hidden_states,
                teacher_lm_head=teacher_model.get_output_embeddings(),
                max_tokens_per_chunk=max_tokens_per_chunk,
                checkpoint_last_chunk=False,
                use_reentrant=False,
                determinism_check="none",
            ).mean()

    else:  # combined inference without gradient checkpointing
        with torch.no_grad():
            teacher_logprobs = F.log_softmax(teacher_model(**batch).logits, dim=-1)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            student_logprobs = F.log_softmax(student_model(**batch).logits, dim=-1)
            loss = F.kl_div(
                input=student_logprobs.flatten(0, -2),
                target=teacher_logprobs.flatten(0, -2),
                log_target=True,
                reduction="batchmean",
            ).mean()
        return loss


def compute_validation_perplexities(args: argparse.Namespace, model: nn.Module, eval_datasets: dict):
    rank = torch.distributed.get_rank()
    perplexities = {}
    for dataset_name, eval_dataset in eval_datasets.items():
        if rank == 0:
            print(f"Evaluating perplexity on {dataset_name} ...")
        device = next(model.parameters()).device
        original_dtype = args.load_dtype if args.load_dtype != "auto" else None
        amp_dtype = args.amp_dtype if args.amp_dtype is not None else original_dtype
        ppl = evaluate_perplexity(model, eval_dataset, args.model_seqlen, device=device, amp_dtype=amp_dtype)
        if rank == 0:
            print(f"{dataset_name} perplexity: {ppl:.9f}")
        perplexities[dataset_name] = ppl
    return perplexities


def main():
    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    parser = argparse.ArgumentParser(add_help=True)
    add_model_args(parser)
    add_data_args(parser)
    add_finetuning_args(parser)
    args = parser.parse_args()

    assert torch.distributed.is_initialized()
    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0
    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)

    args.master_dtype = getattr(torch, args.master_dtype)
    args.embed_dtype = getattr(torch, args.embed_dtype) if args.embed_dtype is not None else args.master_dtype
    args.load_dtype = getattr(torch, args.load_dtype) if args.load_dtype != "auto" else "auto"
    args.code_dtype = getattr(torch, args.code_dtype) if args.code_dtype is not None else None
    args.amp_dtype = getattr(torch, args.amp_dtype) if args.amp_dtype is not None else None

    if args.straight_through_buffer_dtype is not None:
        args.straight_through_buffer_dtype = getattr(torch, args.straight_through_buffer_dtype)
    else:
        args.straight_through_buffer_dtype = args.master_dtype

    if args.save_every_steps is not None:
        assert args.save is not None, f"save_every_steps={args.save_every_steps}, but --save path not specified"
    if args.keep_best_model:
        assert args.save is not None, f"--keep_best_model requires --save path"
        assert args.eval_every_steps is not None, f"--keep_best_model requires --eval_every_steps"
        assert args.eval_datasets is not None, f"--keep_best_model requires --eval_datasets"

    if args.wandb and rank == 0:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")})

    if rank == 0:
        print(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    assert tokenizer.eos_token_id is not None
    tokenizer.pad_token = tokenizer.eos_token

    with master_rank_first(local=True):
        dataset = prepare_training_dataset(args, tokenizer)
        if args.save_dataset_and_exit is not None:
            if rank == 0:
                dataset.save_to_disk(args.save_dataset_and_exit)

    if args.save_dataset_and_exit is not None:
        torch.distributed.barrier()
        return

    sampler = torch.utils.data.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.seed
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=transformers.default_data_collator,
    )
    eval_datasets = {
        dataset_name: get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            eval_mode=True,
        )
        for dataset_name in args.eval_datasets
    }

    use_pv_tuning = args.update_codes and not args.force_dequantize
    if rank == 0:
        print(f"Training {['without', 'with'][use_pv_tuning]} PV-Tuning")

    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        teacher_model = load_teacher_model(args, device)
        student_model, named_quantized_params = load_student_model(args, device, dequantize=use_pv_tuning)
        if rank == 0:
            print("Wrapped model:")
            print(student_model)
            for name, param in student_model.named_parameters():
                print(name, param.shape, param.dtype)

    if use_pv_tuning:
        optimizer = create_pv_optimizer(args, student_model, named_quantized_params)
    else:
        optimizer = create_p_optimizer(args, student_model)
    del named_quantized_params

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_step=0,
    )

    load_training_state(args, metadata, student_model, optimizer)
    torch.distributed.barrier()
    trigger_fsdp_lazy_init_(tokenizer, teacher_model, student_model, device, amp_dtype=args.amp_dtype)

    for current_epoch in range(args.max_epochs):
        if current_epoch < metadata["current_epoch"]:
            continue  # skip finished epochs
        sampler.set_epoch(current_epoch)

        batch_iter = tqdm(train_dataloader, desc=f"Training epoch #{current_epoch}") if rank == 0 else train_dataloader
        for batch_index, batch in enumerate(batch_iter):
            if batch_index <= metadata["microbatches_since_epoch_start"]:
                continue  # skip batches processed before checkpoint
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss_on_batch(
                batch,
                teacher_model,
                student_model,
                amp_dtype=args.amp_dtype,
                max_tokens_per_chunk=args.loss_tokens_per_chunk,
            )

            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1
            if metadata["grad_steps_accumulated"] < grad_accumulation_steps:
                with student_model.no_sync() if args.minimize_sync else nullcontext():
                    (loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()
                optimizer.step()
                optimizer.zero_grad()
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1

                if args.print_every_steps and metadata["total_optimizer_steps"] % args.print_every_steps == 0:
                    loss_numerator_and_denominator = torch.tensor(
                        [metadata["loss_numerator"], metadata["loss_denominator"]], dtype=torch.float64, device=device
                    )

                    torch.distributed.all_reduce(loss_numerator_and_denominator, op=torch.distributed.ReduceOp.SUM)
                    loss_numerator, loss_denominator = loss_numerator_and_denominator.tolist()
                    metadata["aggregated_loss"] = loss_numerator / loss_denominator
                    metadata["loss_numerator"] = metadata["loss_denominator"] = 0
                    if rank == 0:
                        print(
                            f"epoch {metadata['current_epoch']}\tbatch {batch_index}",
                            f"\t| total updates = {metadata['total_optimizer_steps']}",
                            f"\tloss = {metadata['aggregated_loss']:.9f}",
                        )

                if args.eval_every_steps and metadata["total_optimizer_steps"] % args.eval_every_steps == 0:
                    perplexity_scores = compute_validation_perplexities(args, student_model, eval_datasets)
                    for dataset_name, perplexity in perplexity_scores.items():
                        metadata[f"perplexity_{dataset_name}"] = perplexity
                    metric_name = metadata["early_stop_on"]
                    if perplexity_scores[metric_name] < metadata["best_eval_perplexity"]:
                        if rank == 0:
                            print(f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}")
                        metadata["best_eval_perplexity"] = perplexity_scores[args.eval_datasets[0]]
                        metadata["best_step"] = metadata["total_optimizer_steps"]
                        if args.keep_best_model:
                            save_model(args, student_model, optimizer)
                if args.wandb and rank == 0:
                    wandb.log(metadata, step=metadata["total_microbatches"])
                if args.save_every_steps and metadata["total_optimizer_steps"] % args.save_every_steps == 0:
                    save_training_state(args, metadata, student_model, optimizer)

        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1

    save_training_state(args, metadata, student_model, optimizer)


if __name__ == "__main__":
    main()
