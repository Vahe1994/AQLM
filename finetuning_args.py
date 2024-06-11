"""Argument names and defaults for finetune.py"""
import argparse

import torch.distributed


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
        '--monkeypatch_old_pickle',
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
        "--master_dtype",
        type=str,
        default="float32",
        help="data type for storing master parameters and computing optimizer updates",
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
        "--block_type", type=str, required=True,
        help="string name of a transformer layer to wrap, e.g. LlamaDecoderLayer"
    )
    parser.add_argument(
        '--wrap_separately', type=str, nargs='*', default=[],
        help="module classes (by name, similar to block_type) that will be wrapped in a separate fsdp instance and do "
             "not participate in FSDP AMP (if used). Applies to the student (de)quantized model, not the teacher model."
    )
    parser.add_argument(
        "--attn_implementation", type=str, default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2"
    )
    parser.add_argument(
        "--limit_parallel_inits",
        type=int,
        default=1,
        help="this many ranks (per host) initialize their model in parallel. This parameter is meant to save host RAM.",
    )


def add_finetuning_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--update_codes', action='store_true', help="If set, train discrete codes; if not, freeze them",
    )
    parser.add_argument(
        '--update_codebooks_and_scales', action='store_true',
        help="If set, train continuous parameters of quantized representations; if not, freeze them",
    )
    parser.add_argument(
        '--update_non_quantized_parameters', action='store_true',
        help="If set, train the non-quantized model parameters (layernorm scales, biases, logits); if not, freeze them",
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
        '--force_code_update',
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
             " not 0, sample a subset of codes for update stochastically. See StraightThroughAdam for details."
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size when updating codes; higher is slower but more accurate. For single codebook, use beam_size=1",
    )
    parser.add_argument(
        '--code_adam_16bit',
        action="store_true",
        help="If set, adam statistics for codes will be stored as float16 (exp_avg and v_hat) or bfloat16(exp_avg_sq)",
    )
    parser.add_argument(
        '--lamb', action='store_true', help="If set, use Lamb (aka Adam with trust ratio)",
    )
    parser.add_argument(
        '--amsgrad', action='store_true', help="if True, use the AMSGrad variant of adam/lamb",
    )
    parser.add_argument(
        '--debias', action='store_true', default=None,
        help="Whether or not to debias optimizer statistics; defaults to True for adam and False for Lamb",
    )
    parser.add_argument(
        '--no_debias', action='store_false', dest='debias', help="Disable optimizer debiasing (see above)",
    )
    parser.add_argument(
        '--verbose_optimizer',
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
        help="Whether to apply gradient checkpointing",
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


def validate_args(args: argparse.Namespace):
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0

    args.load_dtype = getattr(torch, args.load_dtype) if args.load_dtype != 'auto' else 'auto'
    args.amp_dtype = getattr(torch, args.amp_dtype) if args.amp_dtype is not None else None
    args.code_dtype = getattr(torch, args.code_dtype) if args.code_dtype is not None else None
    args.master_dtype = getattr(torch, args.master_dtype)
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
