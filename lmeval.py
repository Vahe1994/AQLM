import argparse
import fnmatch
import json
import logging
import os
import sys

import torch

sys.path.append("./lm-evaluation-harness")
import lm_eval.models
from lm_eval import evaluator, tasks, utils

from src.modelutils import load_dequantized_model

try:
    import wandb

    wandb_installed = True
except ModuleNotFoundError:
    wandb_installed = False

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name if not load.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        choices=[2048, 4096],
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized model.")

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented
    if args.log_wandb:
        assert args.exp_name or args.load
        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    if args.model_args is None:
        args.model_args = ""

    lm = lm_eval.models.get_model(args.model).create_from_arg_string(
        args.model_args, dict(batch_size=args.batch_size, device=args.device)
    )
    print("lm.device", lm.device)
    if hasattr(lm.model, "hf_device_map"):
        print("Model device map:\n", lm.model.hf_device_map)

    if args.load:
        print("Loading quantized model ...")
        lm.model = load_dequantized_model(lm.model, args.load)

    results = evaluator.simple_evaluate(
        model=lm,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=True,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        log_wandb=args.log_wandb,
    )
    if not isinstance(results["config"]["model"], str):
        results["config"]["model"] = results["config"]["model"].model.config._name_or_path
    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
