import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")
from tqdm import trange

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=10,
        help="Number of benchmark iterations.",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=1,
        help="Input length.",
    )
    parser.add_argument(
        "--output_length",
        type=int,
        default=128,
        help="Output length.",
    )
    parser.add_argument(
        "--replicate_first_block",
        action="store_true",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
    )

    args = parser.parse_args()


def make_shared_model(model, device='cuda'):
    num_layers = len(model.model.layers)
    layer = model.model.layers[0].to(device)
    model.model.layers = nn.ModuleList([])
    model = model.to(device)
    for i in trange(num_layers, desc='Copying block parameters'):
        new_layer = type(layer)(model.config, i).to(device)
        for new_layer_param, layer_param in zip(new_layer.parameters(), layer.parameters()):
            new_layer_param.data = layer_param.data
        new_layer.self_attn.layer_idx = i
        model.model.layers.append(new_layer)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(add_help=True)

    aqlm_model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        torch_dtype="auto",
        low_cpu_mem_usage=args.low_cpu_mem_usage
    )

    if args.replicate_first_block:
        make_shared_model(aqlm_model, device)
    else: 
        aqlm_model = aqlm_model.to(device)

    prompt = torch.randint(low=0, high=aqlm_model.config.vocab_size, size=(1, args.input_length), device=device)

    for i in range(args.warmup_iters + args.benchmark_iters):
        output = aqlm_model.generate(
            prompt, 
            min_new_tokens=args.output_length, 
            max_new_tokens=args.output_length
        )
        if i == args.warmup_iters - 1:
            torch.cuda.synchronize()
            t_s = time.perf_counter()
    torch.cuda.synchronize()
    t_e = time.perf_counter()

    tokens_per_second = args.benchmark_iters * args.output_length / (t_e - t_s) 
    print(f"<Tokens per second> = {tokens_per_second:.2f}")
