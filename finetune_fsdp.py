"""Early prototype of FSDP fine-tuning; TODO clean-up imports"""
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import time
import random
from tqdm.auto import trange, tqdm
import numpy as np
import ipynbname  # pip install ipynbname

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from argparse import Namespace

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from src.aq import QuantizedWeight, QuantizedLinear
from aq_engine import AQEngine
from src.modelutils import get_model
from src.datautils import get_loaders
from finetune import kl_div


def _run_p_step(args, base_model, quantized_model, codebook_optimizer, train_data, device):
    assert len(train_data) % args.codebook_grad_accumulation_steps == 0
    codebook_optimizer.zero_grad()
    with tqdm(train_data, desc="P step") as progress:
        total_loss = 0.0
        for i, batch in enumerate(progress):
            batch = torch.as_tensor(batch, device=device)
            with torch.no_grad():
                teacher_logits = base_model(batch).logits
            with torch.cuda.amp.autocast(enabled=args.autocast_dtype is None, dtype=args.autocast_dtype):
                student_logits = quantized_model(batch).logits
                loss = kl_div(student_logits, teacher_logits, temp=1.0)

            (loss / args.codebook_grad_accumulation_steps).backward()
            if (i + 1) % args.codebook_grad_accumulation_steps == 0:
                codebook_optimizer.step()
                codebook_optimizer.zero_grad()

            total_loss = loss.item() / (i + 1) + total_loss * i / (i + 1)
            progress.desc = f"P step: updating continuous params, loss = {total_loss:.9f}"
            del student_logits, teacher_logits, loss
        print("Average loss (after P step):", total_loss)
        assert (i + 1) % args.codebook_grad_accumulation_steps == 0
        codebook_optimizer.zero_grad(set_to_none=True)
        return total_loss


if __name__ == '__main__':
    dist.init_process_group("nccl")
    rank = os.environ["LOCAL_RANK"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)


    class args:
        base_model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        quant_model = "/extra_disk_1/jheuristic/tinyllama-3t-1x14/"
        dtype = 'bfloat16'
        model_seqlen = 1024  # can be 2048 for 1.1B, 4096-8192 for larger models

        dataset = 'pajama'
        nsamples = 128
        seed = 42
        beam_size = 1

        code_lr = 3e-5
        code_betas = (0.0, 0.95)
        codebook_lr = 1e-5
        codebook_betas = (0.9, 0.95)
        codebook_grad_accumulation_steps = 8

        autocast_dtype = torch.bfloat16  # bfloat16 or None (not using grad scaler!)
        training_dtype = torch.float32
        accumulator_dtype = torch.float64
        eval_after_v_step = True  # run slow evaluation after every V step; takes time; use for debugging only


    train_data = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.base_model,
        seqlen=args.model_seqlen,
    )

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=2 ** 16
    )

    base_model = get_model(
        args.base_model,
        None,
        args.dtype,
        args.model_seqlen,
    )

    base_model = FSDP(base_model, auto_wrap_policy=my_auto_wrap_policy, device_id=device)

    quantized_model = get_model(
        args.base_model,
        args.quant_model,
        args.dtype,
        args.model_seqlen,
    )

    quantized_model = quantized_model.to(args.training_dtype)


    def wrap_if_code(module, recurse, *args, **kwargs):
        if recurse:
            return True
        return isinstance(module, nn.ParameterList) and len(module) == 1  # TODO double-check


    quantized_model = FSDP(quantized_model, auto_wrap_policy=wrap_if_code, device_id=device, use_orig_params=True)

    codebook_optimizer = torch.optim.Adam(
        quantized_model.parameters(),
        lr=args.codebook_lr,
        betas=args.codebook_betas,
        amsgrad=True,
    )

    for epoch in range(1, 100):
        _run_p_step(args, base_model, quantized_model, codebook_optimizer, train_data, device)

    dist.destroy_process_group()
