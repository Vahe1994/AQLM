"""Early prototype of FSDP fine-tuning; TODO clean-up imports"""
import argparse
from tqdm.auto import tqdm
import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel

from src.aq import QuantizedWeight
from src.aq_ops import IntCodes
from src.modelutils import get_model

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False



def add_finetuning_args(parser: argparse.ArgumentParser):
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
    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="number of samples",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--eval_model_seqlen",
        type=int,
        default=None,
        help="Model seqlen on validation. By default is equal to model_seqlen.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="size of validation split",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4"],
        help="Datasets to run evaluation on",
    )
    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="finetuning learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.90,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Adam beta2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to apply gradient checkpointing",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Whether to use amp",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized statistics.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    parser.add_argument(
        "--block_type", type=str, required=True,
        help="string name of a transformer layer to wrap, e.g. LlamaDecoderLayer"
    )
    parser.add_argument(
        "--attn_implementation", type=str, default=None,
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    add_finetuning_args(parser)
    args = parser.parse_args()

    args.microbatch_size = args.microbatch_size or args.batch_size
    if args.dtype != 'auto':
        args.dtype = getattr(torch, args.dtype)

    assert torch.cuda.is_available() and torch.distributed.is_available()
    torch.distributed.init_process_group()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    base_model = get_model(
        args.base_model, load_quantized=None, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(dtype=args.dtype if args.dtype != 'auto' else None)

    base_model.train(True)
    # TODO disable and set all params requires_grad to False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

    transformer_block_types = []
    for module in base_model.modules():
        if module.__class__.__name__ == args.block_type:
            transformer_block_types.append(type(module))
    if not transformer_block_types:
        raise ValueError(f"Could not find {args.block_type} among model layers")
    transformer_block_types = tuple(transformer_block_types)
    assert any(isinstance(module, transformer_block_types) for module in base_model.modules())

    base_model = FullyShardedDataParallel(
        base_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, transformer_block_types),
        device_id=device
    )

    quantized_model = get_model(
        args.base_model, args.quantized_model, dtype=args.dtype, trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation
    ).to(torch.float32)  # master parameters

    quantized_model.train(True)  # note: HF gradient checkpoints do not work for some models without train(True); see
    # https://github.com/huggingface/transformers/blob/2d92db8/src/transformers/models/llama/modeling_llama.py#L1006
    if args.gradient_checkpointing:
        quantized_model.gradient_checkpointing_enable()
        quantized_model.enable_input_require_grads()

    # convert QuantizedModel state dict to make it compatible with FSDP
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedWeight):
            assert module.codes is not None
            if not hasattr(module, 'codes_storage'):
                module.codes_storage = None  # backwards compatibility with older snapshots
            module.codes = nn.Parameter(module.codes.to(torch.int32), requires_grad=module.codes.requires_grad)
            module.wrap_codes_for_fsdp_()
            assert module.codes is None and isinstance(module.codes_storage, IntCodes)

    assert any(isinstance(module, transformer_block_types) for module in quantized_model.modules())
    assert any(isinstance(module, IntCodes) for module in quantized_model.modules())

    quantized_model = FullyShardedDataParallel(
        quantized_model,
        auto_wrap_policy=lambda module, recurse, **_: recurse or isinstance(module, (IntCodes,) + transformer_block_types),
        use_orig_params=True,
        device_id=device,
    )
    print(quantized_model)
    for name, param in quantized_model.named_parameters():
        print(name, param.shape, param.dtype)



    input_ids = torch.arange(4 * 2048).reshape(-1, 2048).to(device)
    for i in tqdm(range(100)):
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            y = base_model(input_ids)
        y.logits.norm().backward()
    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)


