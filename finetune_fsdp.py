"""
Fine-tune an LLM that was previously quantized with AQLM;
based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import argparse
import os
from contextlib import nullcontext
from functools import partial
from typing import Optional, Tuple

import transformers
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType, FullStateDictConfig, MixedPrecision
from tqdm.auto import tqdm

from convert_legacy_model_format import load_quantized_model_with_old_pickle
from src.aq import QuantizedWeight
from src.aq_ops import IntCodes, master_rank_first, one_rank_at_a_time, is_signed
from src.datautils import group_texts, split_long_texts, get_loaders, evaluate_perplexity
from src.modelutils import get_model
from src.pv_utils import infer_block_classes, create_dequantized_model, \
    get_original_named_parameters_from_fsdp_module, split_quantized_weights_between_ranks, \
    YourQuantizedWeightIsInAnotherRank
from src.pv_optimizer import StraightThroughAdamW

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
        help="Determines whether to use direct training, straight-throughe stimation or a mixture thereof. "
             "If delta_decay is 0, use straight-through estimation. If delta_decay is 1, do not use it at all. "
             "If between 0 and 1, every straight-through buffer will decay to the quantized weight with moving average."
             " straight_through_buffer = (1 - delta_decay) * straight_through_buffer + delta_decay * quantized_weight."
             " Please refer to the docstring of StraightThroughAdam for details.",
    )
    parser.add_argument(
        "--max_code_change_per_step",
        type=float,
        default=1e-3,
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
        '--force_directional_code_update',
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
        default=None,
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
        return 'input_ids' in dataset.column_names
    if is_tokenized(dataset):
        if torch.distributed.get_rank() == 0:
            print("Dataset already tokenized")
            assert len(dataset[0]['input_ids']) == args.model_seqlen
        return dataset

    text_column_name = 'text' if 'text' in dataset.column_names else next(iter(dataset.column_names))

    if args.preprocessing_chunk_length is not None:
        dataset = dataset.map(
            lambda examples: {text_column_name: split_long_texts(
                examples[text_column_name], args.preprocessing_chunk_length)},
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


def load_base_model(args: argparse.Namespace, device: torch.device) -> FullyShardedDataParallel:
    base_model = get_model(
        args.base_model, load_quantized=None, dtype=args.load_dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(dtype=args.load_dtype if args.load_dtype != 'auto' else None)
    base_model.train(False)
    for param in base_model.parameters():
        param.requires_grad = False

    base_model.config.use_cache = False
    transformer_block_types = infer_block_classes(base_model, args.block_type)
    return FullyShardedDataParallel(
        base_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, transformer_block_types),
        device_id=device
    )


def load_dequantized_model(args: argparse.Namespace, device: torch.device) -> Tuple[FullyShardedDataParallel, dict]:
    if not args.monkeypatch_old_pickle:
        quantized_model = get_model(
            args.base_model, args.quantized_model, dtype=args.load_dtype, trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation
        ).to(args.master_dtype)  # master parameters
    else:
        quantized_model = load_quantized_model_with_old_pickle(
            args.base_model, args.quantized_model, dtype=args.load_dtype, trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation
        ).to(args.master_dtype)

    quantized_model.config.use_cache = False
    quantized_model.train(True)  # note: HF gradient checkpoints do not work for some models without train(True); see
    # https://github.com/huggingface/transformers/blob/2d92db8/src/transformers/models/llama/modeling_llama.py#L1006
    if args.gradient_checkpointing:
        quantized_model.gradient_checkpointing_enable()
        quantized_model.enable_input_require_grads()

    # convert QuantizedModel state dict to make it compatible with FSDP
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedWeight):
            assert module.codes is not None
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(args.code_dtype).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)
    assert any(isinstance(module, IntCodes) for module in quantized_model.modules())

    dequantized_model, named_quantized_params = create_dequantized_model(
        quantized_model, dequantized_dtype=args.amp_dtype, reuse_non_quantized=True)
    del quantized_model

    transformer_block_types = infer_block_classes(dequantized_model, args.block_type)

    mixed_precision = None
    if args.amp_dtype is not None:
        mixed_precision = MixedPrecision(
            param_dtype=args.amp_dtype,
            reduce_dtype=args.amp_dtype,
            _module_classes_to_ignore=transformer_block_types + tuple(transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
        )
    fsdp_model = FullyShardedDataParallel(
        dequantized_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, transformer_block_types),
        mixed_precision=mixed_precision,
        use_orig_params=True,
        device_id=device,
    )
    return fsdp_model, named_quantized_params


def compute_loss_on_batch(
        batch: dict, base_model: nn.Module, quantized_model: nn.Module, amp_dtype: Optional[torch.dtype]
) -> torch.Tensor:
    with torch.no_grad():
        teacher_logprobs = F.log_softmax(base_model(**batch).logits, dim=-1)
    with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
        student_logprobs = F.log_softmax(quantized_model(**batch).logits, dim=-1)
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
        original_dtype = args.load_dtype if args.load_dtype != 'auto' else None
        amp_dtype = args.amp_dtype if args.amp_dtype is not None else original_dtype
        ppl = evaluate_perplexity(model, eval_dataset, args.model_seqlen, device=device, amp_dtype=amp_dtype)
        if rank == 0:
            print(f"{dataset_name} perplexity: {ppl:.9f}")
        perplexities[dataset_name] = ppl
    return perplexities


def _load_state(args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: StraightThroughAdamW):
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


def _save_state(args: argparse.Namespace, metadata: dict, quantized_model: nn.Module, optimizer: StraightThroughAdamW):
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


def _save_model(args: argparse.Namespace, dequantized_model: FullyShardedDataParallel, optimizer: StraightThroughAdamW):
    """Save consolidated model state dict"""
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
            print(f"Saved best model to {output_path}")


def main():
    parser = argparse.ArgumentParser(add_help=True)
    add_model_args(parser)
    add_data_args(parser)
    add_finetuning_args(parser)
    args = parser.parse_args()

    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    assert args.batch_size is not None, "please specify batch size"
    assert args.batch_size % world_size == 0
    if args.microbatch_size is None:
        args.microbatch_size = args.batch_size // world_size
    assert args.batch_size % (world_size * args.microbatch_size) == 0
    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)
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
            exit()

    sampler = torch.utils.data.DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.seed)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.microbatch_size, num_workers=args.num_workers, sampler=sampler,
        collate_fn=transformers.default_data_collator
    )
    eval_datasets = {dataset_name: get_loaders(
        dataset_name, seed=args.seed, model_path=args.base_model, seqlen=args.model_seqlen, eval_mode=True,
        ) for dataset_name in args.eval_datasets
    }

    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        base_model = load_base_model(args, device)
        dequantized_model, named_quantized_params = load_dequantized_model(args, device)
        if rank == 0:
            print("Wrapped model:")
            print(dequantized_model)
            for name, param in dequantized_model.named_parameters():
                print(name, param.shape, param.dtype)
        named_dequantized_params = get_original_named_parameters_from_fsdp_module(dequantized_model)
        assert all(name in named_dequantized_params for name in named_quantized_params)

        if world_size > 1:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = split_quantized_weights_between_ranks(
                named_quantized_params, verify_checksums=False)
        for quantized_weight in named_quantized_params.values():
            if isinstance(quantized_weight, QuantizedWeight):
                quantized_weight.to(device)
            else:
                assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)

    optimizer = StraightThroughAdamW(
        named_dequantized_params=named_dequantized_params,
        named_quantized_params=named_quantized_params,
        update_codes=dict(
            lr=args.code_lr, betas=(args.code_beta1, args.code_beta2),
            lamb=args.lamb, debias=args.debias, amsgrad=args.amsgrad, compute_dtype=args.master_dtype,
            exp_avg_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_sq_dtype=torch.bfloat16 if args.code_adam_16bit else args.master_dtype,
            v_hat_max_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
        ),
        update_codebooks_and_scales=dict(
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb, debias=args.debias, amsgrad=args.amsgrad, compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype, exp_avg_sq_dtype=args.master_dtype, v_hat_max_dtype=args.master_dtype,
        ),
        update_non_quantized_parameters=dict(
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb, debias=args.debias, amsgrad=args.amsgrad, compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype, exp_avg_sq_dtype=args.master_dtype, v_hat_max_dtype=args.master_dtype,
        ),
        delta_decay=args.delta_decay,
        max_code_change_per_step=args.max_code_change_per_step,
        force_directional_code_update=args.force_directional_code_update,
        code_trust_ratio=code_trust_ratio,
        beam_size=args.beam_size,
        straight_through_buffer_dtype=args.straight_through_buffer_dtype,
        verbose=args.verbose_optimizer,
    )
    del named_quantized_params

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float('nan'),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float('inf'),
        best_step=0,
    )

    _load_state(args, metadata, dequantized_model, optimizer)
    torch.distributed.barrier()

    for current_epoch in range(args.max_epochs):
        if current_epoch < metadata['current_epoch']:
            continue  # skip finished epochs
        sampler.set_epoch(current_epoch)

        batch_iter = tqdm(train_dataloader, desc=f"Training epoch #{current_epoch}") if rank == 0 else train_dataloader
        for batch_index, batch in enumerate(batch_iter):
            if batch_index <= metadata['microbatches_since_epoch_start']:
                continue  # skip batches processed before checkpoint
            metadata['microbatches_since_epoch_start'] += 1
            metadata['total_microbatches'] += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss_on_batch(batch, base_model, dequantized_model, amp_dtype=args.amp_dtype)

            metadata['loss_numerator'] += loss.item()
            metadata['loss_denominator'] += 1
            metadata['grad_steps_accumulated'] += 1
            if metadata['grad_steps_accumulated'] < grad_accumulation_steps:
                with dequantized_model.no_sync() if args.minimize_sync else nullcontext():
                    (loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()
                optimizer.step()
                optimizer.zero_grad()
                metadata['grad_steps_accumulated'] = 0
                metadata['total_optimizer_steps'] += 1

                if args.print_every_steps and metadata['total_optimizer_steps'] % args.print_every_steps == 0:
                    loss_numerator_and_denominator = torch.tensor(
                        [metadata['loss_numerator'], metadata['loss_denominator']], dtype=torch.float64, device=device)

                    torch.distributed.all_reduce(loss_numerator_and_denominator, op=torch.distributed.ReduceOp.SUM)
                    loss_numerator, loss_denominator = loss_numerator_and_denominator.tolist()
                    metadata['aggregated_loss'] = loss_numerator / loss_denominator
                    metadata['loss_numerator'] = metadata['loss_denominator'] = 0
                    if rank == 0:
                        print(f"epoch {metadata['current_epoch']}\tbatch {batch_index}",
                              f"\t| total updates = {metadata['total_optimizer_steps']}",
                              f"\tloss = {metadata['aggregated_loss']:.9f}")

                if args.eval_every_steps and metadata['total_optimizer_steps'] % args.eval_every_steps == 0:
                    perplexity_scores = compute_validation_perplexities(args, dequantized_model, eval_datasets)
                    for dataset_name, perplexity in perplexity_scores.items():
                        metadata[f'perplexity_{dataset_name}'] = perplexity
                    metric_name = metadata['early_stop_on']
                    if perplexity_scores[metric_name] < metadata['best_eval_perplexity']:
                        if rank == 0:
                            print(f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}")
                        metadata['best_eval_perplexity'] = perplexity_scores[args.eval_datasets[0]]
                        metadata['best_step'] = metadata['total_optimizer_steps']
                        if args.keep_best_model:
                            _save_model(args, dequantized_model, optimizer)
                if args.wandb and rank == 0:
                    wandb.log(metadata, step=metadata['total_microbatches'])
                if args.save_every_steps and metadata['total_optimizer_steps'] % args.save_every_steps == 0:
                    _save_state(args, metadata, dequantized_model, optimizer)

        metadata['microbatches_since_epoch_start'] = 0
        metadata['current_epoch'] += 1

    _save_state(args, metadata, dequantized_model, optimizer)


if __name__ == "__main__":
    main()
