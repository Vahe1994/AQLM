import random
from typing import Optional

import torch
from tqdm import tqdm

from src.pv_ops import _DequantizedLayerWrapper


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
                name: torch.zeros_like(engine.layer.weight, requires_grad=False, dtype=statistics_dtype)
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

            update = m_hat / (torch.sqrt(v_hat) + self.eps)
            weight = engine.quantized_weight().to(update.dtype)
            if self.lamb:
                update_norm = torch.norm(update, p='fro')
                weight_norm = torch.norm(weight, p='fro')
                trust_ratio = weight_norm / update_norm
                update *= trust_ratio

            self.delta[name][...] = self.delta[name] * self.delta_feedback - self.lr * update
            target_weight = (weight + self.delta[name])
            engine.quantized_weight.beam_search_update_codes_(
                reference_weight=target_weight, dim_rng=random.Random(None), verbose=verbose > 1, **kwargs
            )  # beam search updates codes to minimize ||target_weight - quantized_weight()||^2
            engine.layer.weight[...] = engine.quantized_weight()  # de-quantize again
            self.delta[name][...] = target_weight - engine.quantized_weight()

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        for name, engine in self.engines.items():
            if set_to_none:
                engine.layer.weight.grad = None
            else:
                engine.layer.weight.grad.zero_()
