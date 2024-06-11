"""
Fine-tune an LLM that was previously quantized with AQLM;
based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import argparse
import os
from contextlib import nullcontext
from functools import partial
from typing import Optional, Dict

import transformers
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel
from tqdm.auto import tqdm

from finetuning_args import add_model_args, add_finetuning_args, add_data_args, validate_args
from finetuning_saveload import load_teacher_model, load_student_model, load_training_state, save_training_state, save_model
from src.aq import QuantizedWeight
from src.aq_ops import master_rank_first, one_rank_at_a_time
from src.configurable_adam import ConfigurableAdamW
from src.datautils import group_texts, split_long_texts, get_loaders, evaluate_perplexity
from src.pv_utils import get_original_named_parameters_from_fsdp_module, split_quantized_weights_between_ranks, \
    YourQuantizedWeightIsInAnotherRank
from src.pv_optimizer import StraightThroughAdamW

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


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


def create_pv_optimizer(args: argparse.Namespace, student_model: FullyShardedDataParallel,
                        named_quantized_params: Dict[str, QuantizedWeight]) -> torch.optim.Optimizer:
    """Create optimizer for PV-Tuning using a de-quantized student model and a dictionary of quantized weights"""
    named_dequantized_params = get_original_named_parameters_from_fsdp_module(student_model)
    assert all(name in named_dequantized_params for name in named_quantized_params)
    return StraightThroughAdamW(
        named_dequantized_params=named_dequantized_params,
        named_quantized_params=named_quantized_params,
        update_codes=dict(
            lr=args.code_lr, betas=(args.code_beta1, args.code_beta2),
            lamb=args.lamb, debias=args.debias, amsgrad=args.amsgrad, compute_dtype=args.master_dtype,
            exp_avg_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
            exp_avg_sq_dtype=torch.bfloat16 if args.code_adam_16bit else args.master_dtype,
            v_hat_max_dtype=torch.float16 if args.code_adam_16bit else args.master_dtype,
        ) if args.update_codes else None,
        update_codebooks_and_scales=dict(
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb, debias=args.debias, amsgrad=args.amsgrad, compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype, exp_avg_sq_dtype=args.master_dtype, v_hat_max_dtype=args.master_dtype,
        ) if args.update_codebooks_and_scales else None,
        update_non_quantized_parameters=dict(
            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
            lamb=args.lamb, debias=args.debias, amsgrad=args.amsgrad, compute_dtype=args.master_dtype,
            exp_avg_dtype=args.master_dtype, exp_avg_sq_dtype=args.master_dtype, v_hat_max_dtype=args.master_dtype,
        ) if args.update_non_quantized_parameters else None,
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
    all_trainable_params = set()
    if args.update_codebooks_and_scales:
        all_trainable_params |= quantized_weight_continuous_parameters
    if args.update_non_quantized_parameters:
        non_quantized_parameters = {param for param in student_model.parameters()
                                    if param not in quantized_weight_continuous_parameters
                                    and torch.is_floating_point(param) and param.requires_grad}
        all_trainable_params |= non_quantized_parameters
    if args.update_codes:
        raise RuntimeError("When asked to update_codes, one should create_pv_optimizer, but this is create_p_optimizer")
    assert len(all_trainable_params) > 0, ("found no trainable parameters. Did you specify update_codes, "
                                           "update_codebooks_and_scales or update_non_quantized_parameters?")
    return ConfigurableAdamW(
        params=list(all_trainable_params),
        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
        lamb=args.lamb, debias=args.debias, amsgrad=args.amsgrad, compute_dtype=args.master_dtype,
        exp_avg_dtype=args.master_dtype, exp_avg_sq_dtype=args.master_dtype, v_hat_max_dtype=args.master_dtype,
    )


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
    validate_args(args)

    grad_accumulation_steps = args.batch_size // (world_size * args.microbatch_size)

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

    use_pv_tuning = args.update_codes is not None
    if rank == 0 and use_pv_tuning:
        print("Training with PV-Tuning, updating discrete codes")
    elif rank == 0 and not use_pv_tuning:
        print("Training without PV-Tuning, updating continuous parameters only")

    with one_rank_at_a_time(local=True, group_size=args.limit_parallel_inits):
        base_model = load_teacher_model(args, device)
        student_model, named_quantized_params = load_student_model(args, device, dequantize=use_pv_tuning)
        if rank == 0:
            print("Wrapped model:")
            print(student_model)
            for name, param in student_model.named_parameters():
                print(name, param.shape, param.dtype)

        if world_size > 1 and use_pv_tuning:
            # distributed pv: each rank holds a subset of all quantized weights; the rest are replaced with pointers
            named_quantized_params = split_quantized_weights_between_ranks(
                named_quantized_params, verify_checksums=False)
        else:
            named_quantized_params = {}  # not needed; delete to save memory

        for quantized_weight in named_quantized_params.values():
            if isinstance(quantized_weight, QuantizedWeight):
                quantized_weight.to(device)
            else:
                assert isinstance(quantized_weight, YourQuantizedWeightIsInAnotherRank)

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
        aggregated_loss=float('nan'),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float('inf'),
        best_step=0,
    )

    load_training_state(args, metadata, student_model, optimizer)
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
            loss = compute_loss_on_batch(batch, base_model, student_model, amp_dtype=args.amp_dtype)

            metadata['loss_numerator'] += loss.item()
            metadata['loss_denominator'] += 1
            metadata['grad_steps_accumulated'] += 1
            if metadata['grad_steps_accumulated'] < grad_accumulation_steps:
                with student_model.no_sync() if args.minimize_sync else nullcontext():
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
                    perplexity_scores = compute_validation_perplexities(args, student_model, eval_datasets)
                    for dataset_name, perplexity in perplexity_scores.items():
                        metadata[f'perplexity_{dataset_name}'] = perplexity
                    metric_name = metadata['early_stop_on']
                    if perplexity_scores[metric_name] < metadata['best_eval_perplexity']:
                        if rank == 0:
                            print(f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}")
                        metadata['best_eval_perplexity'] = perplexity_scores[args.eval_datasets[0]]
                        metadata['best_step'] = metadata['total_optimizer_steps']
                        if args.keep_best_model:
                            save_model(args, student_model, optimizer)
                if args.wandb and rank == 0:
                    wandb.log(metadata, step=metadata['total_microbatches'])
                if args.save_every_steps and metadata['total_optimizer_steps'] % args.save_every_steps == 0:
                    save_training_state(args, metadata, student_model, optimizer)

        metadata['microbatches_since_epoch_start'] = 0
        metadata['current_epoch'] += 1

    save_training_state(args, metadata, student_model, optimizer)


if __name__ == "__main__":
    main()
