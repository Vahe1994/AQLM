"""Utility functions for fine-tuning quantized models using the PV algorithm"""
import functools
import random
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

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


class CodeOptimizer(torch.optim.Optimizer):
    """An adam-like optimizer for discrete codes"""
    def __init__(
            self,
            engines: dict[str, _DequantizedLayerWrapper],
            *,
            lr: float,
            beam_size: int,
            betas: tuple[float],
            delta_feedback: float,
            eps: float = 1e-8,
            amsgrad: bool = False,
            lamb: bool = False,
            statistics_dtype: Optional[torch.dtype] = None,
    ):
        self.engines = engines
        self.lr = lr
        self.beam_size = beam_size
        self.betas = betas
        self.eps = eps
        self.delta_feedback = delta_feedback
        self.amsgrad = amsgrad
        self.lamb = lamb

        if delta_feedback > 0:
            self.delta = {
                name: torch.zeros_like(engine.layer.weight, requires_grad=False, dtype=statistics_dtype)
                for name, engine in self.engines.items()
            }
        if betas[0]:
            self.first_momentum = {
                name: torch.zeros_like(engine.layer.weight, requires_grad=False, dtype=statistics_dtype)
                for name, engine in self.engines.items()
            }
        if betas[1]:
            self.second_momentum = {
                name: torch.zeros_like(engine.layer.weight, requires_grad=False, dtype=statistics_dtype)
                for name, engine in self.engines.items()
            }
        if self.amsgrad:
            self.v_hat_max = {
                name: torch.zeros_like(engine.layer.weight, requires_grad=False, dtype=dtype)
                for name, engine in self.engines.items()
            }

        self._step_i = 0

    @torch.no_grad()
    def step(self, verbose: int = 1, **kwargs) -> None:
        self._step_i += 1
        engines_iter = self.engines.items()
        if verbose:
            engines_iter = tqdm(engines_iter, desc="Running beam search")
        for name, engine in engines_iter:
            grad = engine.layer.weight.grad

            if self.betas[0]:
                self.first_momentum[name] = self.betas[0] * self.first_momentum[name] + (1 - self.betas[0]) * grad
                m_hat = self.first_momentum[name] / (1 - self.betas[0] ** self._step_i)
            else:
                m_hat = grad

            if self.betas[1]:
                self.second_momentum[name] = self.betas[1] * self.second_momentum[name] + (
                            1 - self.betas[1]) * grad.pow(2)
                v_hat = self.second_momentum[name] / (1 - self.betas[1] ** self._step_i)
            else:
                v_hat = grad.pow(2)

            if self.amsgrad:
                self.v_hat_max[name] = torch.maximum(self.v_hat_max[name], v_hat)
                v_hat = self.v_hat_max[name]

            weight = engine.layer.weight.float()
            update = m_hat / (torch.sqrt(v_hat) + self.eps)
            if self.lamb:
                update_norm = torch.norm(update, p='fro')
                weight_norm = torch.norm(weight, p='fro')
                trust_ratio = weight_norm / update_norm
                update *= trust_ratio

            self.delta[name][...] = self.delta[name] * self.delta_feedback - self.lr * update
            target_weight = (weight + self.delta[name])
            engine.quantized_weight.beam_search_update_codes_(
                XTX=None, reference_weight=target_weight, dim_rng=random.Random(None), verbose=verbose > 1, **kwargs
            )
            engine.layer.weight[...] = engine.quantized_weight()  # de-quantize again
            self.delta[name][...] += weight - engine.quantized_weight()

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        for name, engine in self.engines.items():
            if set_to_none:
                engine.layer.weight.grad = None
            else:
                engine.layer.weight.grad.zero_()


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

