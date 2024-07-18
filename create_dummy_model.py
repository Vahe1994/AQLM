import argparse
import json

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--model",
        type=str,
        help="Name of the base model used for reference",
    )
    parser.add_argument(
        "--override_config",
        type=str,
        help="Path to the config with params to be overrided",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Path to save HF compatible checkpoint to",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="torch dtype",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code",
    )

    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    with open(args.override_config, "r") as f:
        override_config = json.load(f)
    for key, value in override_config.items():
        print(f"Set {key} to {value}")
        setattr(config, key, value)
    # Create model
    model = AutoModelForCausalLM.from_config(config, torch_dtype=getattr(torch, args.dtype))
    print("dtype:", next(model.parameters()).dtype)
    # Save model
    model.save_pretrained(args.save)
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.save)
