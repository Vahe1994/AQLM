"""Utility functions for fine-tuning quantized models using the PV algorithm"""
import functools

import torch
import torch.nn as nn

from src.aq import QuantizedWeight


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


class _DequantizedLayerWrapper(nn.Module):
    def __init__(self, layer: nn.Module, quantized_weight: QuantizedWeight):
        super().__init__()
        self.layer = layer
        self._aq_weight_tuple = (quantized_weight,)  # save in a way that does not register submodule

    @property
    def quantized_weight(self):
        return self._aq_weight_tuple[0]

    def forward(self, input, *args, **kwargs):
        return self.layer(input, *args, **kwargs)


def create_dequantized_model_and_optimizer(args: Namespace, base_model: nn.Module, quantized_model: nn.Module):
    module_name_to_engine = dict()
    base_model_dtype = next(base_model.parameters()).dtype
    memo_remove_quantized_weights = {
        id(module): None for module in quantized_model.modules()
        if isinstance(module, QuantizedWeight)
    }
    print(f"Found {len(memo_remove_quantized_weights)} quantized layers to tie")
    dequantized_model = deepcopy(quantized_model, memo_remove_quantized_weights).to(base_model_dtype)
    for param in dequantized_model.parameters():
        param.requires_grad = False  # only quantized weight matrices accumulate grad

    for name, module in list(dequantized_model.named_modules()):
        if not isinstance(module, QuantizedLinear):  # TODO add conv on merge
            continue
        assert module.quantized_weight is None, "sanity check failed: replacement did not work properly"
        quantized_weight = rgetattr(quantized_model, name).quantized_weight
        assert isinstance(quantized_weight, QuantizedWeight), type(quantized_weight)

        replacer = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        replacer.weight = nn.Parameter(quantized_weight().to(base_model_dtype), requires_grad=True)
        if module.bias is not None:
            replacer.bias = nn.Parameter(module.bias, requires_grad=False)

        wrapped_replacer = _DequantizedLayerWrapper(replacer, quantized_weight)
        module_name_to_engine[name] = wrapped_replacer
        rsetattr(dequantized_model, name, wrapped_replacer)

    code_optimizer = AQLMCodeSGD(
        module_name_to_engine, lr=args.code_lr, delta_decay=args.delta_decay, beam_size=args.beam_size,
        betas=args.code_betas, amsgrad=True, lamb=True, dtype=torch.float32)
    return dequantized_model, code_optimizer


@torch.no_grad()
def update_dequantized_model_(dequantized_model, quantized_model):
    """Copy updated params onto dequantized model in-place"""
    for name, module in dequantized_model.named_modules():
        if isinstance(module, _DequantizedLayerWrapper):
            assert module.quantized_weight is rgetattr(quantized_model, name).quantized_weight
        # if this assert fails, your quantized and dequantized models are not tied properly and will not train!

    quantized_parameters = dict(quantized_model.named_parameters())
    for name, param in dequantized_model.named_parameters():
        if name in quantized_parameters:
            param.data[...] = quantized_parameters[name].to(param.dtype)  # biases, normalization, etc
        else:
            assert name.endswith('.layer.weight'), name
            quantized_linear = rgetattr(quantized_model, name[:-len('.layer.weight')])
            assert isinstance(quantized_linear, QuantizedLinear)
            param.data[...] = quantized_linear.quantized_weight()

