"""
This abomination converts between one of several quantized model formats to the same format as returned by main.py .
This code exists because we failed to produce a single data format for quantized model.
We should eventually switch to saving all models in the same data format. Once we do, this file should be deleted.
"""
import argparse

import torch
from torch import nn

from src.aq import QuantizedWeight, QuantizedLinear
from src.aq_ops import is_signed
from src.modelutils import get_model, save_quantized_model
from src.datautils import get_loaders
from main import perplexity_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
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
        "--code_dtype",
        type=str,
        default=None,
        help="if specified, cast quantized layers' codes to this dtype; default = keep loaded dtype",
    )
    parser.add_argument(
        "--p_finetuned_state_dict",
        type=str,
        default=None,
        help="path to quantized model state dict saved by the old FSDP finetuning code",
    )
    parser.add_argument("--save", type=str, required=True, help="Save the converted quantized model here")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert args.p_finetuned_state_dict is not None, "for now, this converter only accepts state dicts from P step"

    quantized_model = get_model(args.base_model, load_quantized=args.quantized_model, trust_remote_code=True)
    for module in quantized_model.modules():
        if isinstance(module, QuantizedWeight):
            if not hasattr(module, 'codes_storage'):
                module.codes_storage = None
            assert module.codes is not None
            if args.code_dtype is not None:
                assert module.nbits_per_codebook <= torch.iinfo(args.code_dtype).bits - is_signed(args.code_dtype)
                module.codes = nn.Parameter(module.codes.to(args.code_dtype), requires_grad=module.codes.requires_grad)

    if args.p_finetuned_state_dict is not None:
        state_dict = torch.load(args.p_finetuned_state_dict, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.codes_storage.data')}
        status = quantized_model.load_state_dict(state_dict, strict=False)
        assert all(key.endswith('codes') for key in status.missing_keys)
        assert not status.unexpected_keys
        del state_dict, status

    save_quantized_model(quantized_model, args.save)

