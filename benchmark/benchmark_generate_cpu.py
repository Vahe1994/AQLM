import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import warnings

warnings.filterwarnings("ignore")

import torch

torch.set_num_threads(8)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=None,
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
        default=3,
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
    args = parser.parse_args()

    device = "cpu"

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32)
    if args.num_codebooks is not None:
        config.aqlm["num_codebooks"] = args.num_codebooks
    if args.in_group_size is not None:
        config.aqlm["in_group_size"] = args.in_group_size
    if args.nbits_per_codebook is not None:
        config.aqlm["nbits_per_codebook"] = args.nbits_per_codebook

    real_num_layers = config.num_hidden_layers
    if "meta-llama" in args.model:
        config.num_hidden_layers = 1
    aqlm_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float32)

    if "meta-llama" in args.model:
        aqlm_model.config.num_hidden_layers = real_num_layers
        layer = aqlm_model.model.layers[0]
        aqlm_model.model.layers = nn.ModuleList([])
        for i in range(real_num_layers):
            another_layer = type(layer)(config, i)

            another_layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data
            another_layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data
            another_layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data
            another_layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data
            another_layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data
            another_layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data
            another_layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data

            another_layer.self_attn.layer_idx = i
            aqlm_model.model.layers.append(another_layer)

        aqlm_model.model.config.num_hidden_layers = real_num_layers

    prompt = torch.randint(low=0, high=aqlm_model.config.vocab_size, size=(1, args.input_length), device=device)

    for i in range(args.warmup_iters + args.benchmark_iters):
        aqlm_model.generate(prompt, min_new_tokens=args.output_length, max_new_tokens=args.output_length)
        if i == args.warmup_iters - 1:
            t_s = time.perf_counter()
    t_e = time.perf_counter()

    tokens_per_second = args.benchmark_iters * args.output_length / (t_e - t_s)
    print(f"<Tokens per second> = {tokens_per_second:.3f}")
