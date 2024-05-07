import argparse
import os
import shutil
import copy

import torch

from src.modelutils import get_layers, get_model, save_not_quantized_weights
from src.aq import QuantizedLinear


def get_submodule(model, path):
    for name in path.split('.'):
        model = getattr(model, name)
    return model


def patch_orig_model(orig_model, quant_model, path):
    path = path.split('.')
    assert len(path) >= 2

    father = get_submodule(orig_model, '.'.join(path[:-1]))
    linear_name = path[-1]

    quant_linear = get_submodule(quant_model, '.'.join(path))

    setattr(father, linear_name, quant_linear)


def main():
    parser = argparse.ArgumentParser(add_help=True)
    # Model params
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quant_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--test-model",
        action="store_true",
        help="Test model on small empty input.",
    )
    parser.add_argument(
        "--skip-fix",
        action="store_true",
        help="Skip fix (for testing purposes).",
    )
    # Save params
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized statistics.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    args = parser.parse_args()

    orig_model = get_model(args.base_model, None, args.dtype, "cpu", trust_remote_code=args.trust_remote_code)

    quant_model = get_model(
        args.base_model, args.quant_model, args.dtype, "cpu", trust_remote_code=args.trust_remote_code
    )

    if args.skip_fix:
        fixed_model = quant_model
        print('Skipped model fix')
    else:
        quantized_linear_names = [
            name for name, module in quant_model.named_modules()
            if isinstance(module, QuantizedLinear)
        ]
        for quantized_linear_name in quantized_linear_names:
            patch_orig_model(orig_model, quant_model, quantized_linear_name)
        fixed_model = orig_model
        print('Fixed model')

    # save model
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for layer_index, layer in enumerate(get_layers(fixed_model)):
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            torch.save(layer, layer_save_path)
        save_not_quantized_weights(fixed_model, args.save)
        # copy args
        shutil.copy(os.path.join(args.quant_model, "args.pt"), os.path.join(args.save, "args.pt"))
        print('Saved model')

    if args.test_model:
        # Changes model dtype, save should be before test
        fixed_model.float()(torch.tensor([0], dtype=torch.int32).reshape(1, 1))
        print('Model is ok')


if __name__ == '__main__':
    main()
