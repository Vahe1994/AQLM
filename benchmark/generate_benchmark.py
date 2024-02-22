import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

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
        "--real_model",
        action="store_true",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
    )

    args = parser.parse_args()


def load_model(model_name, device="cuda"):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)


def load_shared_model(model_name, device="cuda"):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    layer = model.model.layers[0]
    for i in trange(1, num_layers, desc="Copying block parameters"):
        new_layer = type(layer)(model.config, i).to(device)
        for new_layer_param, layer_param in zip(new_layer.parameters(), layer.parameters()):
            new_layer_param.data = layer_param.data
        new_layer.self_attn.layer_idx = i
        model.model.layers.append(new_layer)
    return model


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(add_help=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    if args.real_model:
        aqlm_model = load_model(args.model, device)
    else:
        aqlm_model = load_shared_model(args.model, device)

    prompt = torch.randint(low=0, high=aqlm_model.config.vocab_size, size=(1, args.input_length), device=device)

    for i in range(args.warmup_iters + args.benchmark_iters):
        output = aqlm_model.generate(prompt, min_new_tokens=args.output_length, max_new_tokens=args.output_length)
        if i == args.warmup_iters - 1:
            torch.cuda.synchronize(device)
            t_s = time.perf_counter()
    torch.cuda.synchronize(device)
    t_e = time.perf_counter()

    tokens_per_second = args.benchmark_iters * args.output_length / (t_e - t_s)
    print(f"<Tokens per second> = {tokens_per_second:.2f}")
