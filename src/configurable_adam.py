import math

import torch
from typing import Tuple, Iterable, Union, Optional


class ConfigurableAdamW(torch.optim.Optimizer):
    r"""
    A version of Adam optimizer that supports custom parameter dtypes, amsgrad, lamb or rmsprop on per-group basis.
    Adam and Amsgrad based on https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
    Lamb flag based on https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py

    :param exp_avg_dtype: dtype for storing first moments; only created if betas[0] != 0; defaults to param dtype
    :param exp_avg_sq_dtype: dtype for storing second moments; only created if betas[0] != 0; defaults to param dtype
    :param v_hat_max_dtype: dtype for storing maximum v_hat; only created amsgrad=True; defaults to param dtype
    :param compute_dtype: dtype for optimizer step computation; defaults to param dtype
    """

    def __init__(
            self,
            params: Iterable[Union[torch.Tensor, dict]],
            lr: float = 1e-3,
            betas: Tuple[float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0,
            debias: bool = True,
            amsgrad: bool = False,
            lamb: bool = False,
            clamp_value: Optional[float] = None,
            compute_dtype: Optional[torch.dtype] = None,
            exp_avg_dtype: Optional[torch.dtype] = None,
            exp_avg_sq_dtype: Optional[torch.dtype] = None,
            v_hat_max_dtype: Optional[torch.dtype] = None,

    ) -> None:
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, debias=debias, amsgrad=amsgrad, lamb=lamb,
            clamp_value=clamp_value, compute_dtype=compute_dtype, exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype, v_hat_max_dtype=v_hat_max_dtype,
        )
        super().__init__(params, defaults)

    def _maybe_init_state(self, param: torch.Tensor, group: dict) -> dict:
        state = self.state[param]
        state["step"] = 0
        if group['betas'][0] != 0:
            state["exp_avg"] = torch.zeros_like(
                param, dtype=group['exp_avg_dtype'], memory_format=torch.preserve_format)
        if group['betas'][1] != 0:
            state["exp_avg_sq"] = torch.zeros_like(
                param, dtype=group['exp_avg_sq_dtype'], memory_format=torch.preserve_format)
        if group['amsgrad']:
            state["v_hat_max"] = torch.zeros_like(
                param, dtype=group['v_hat_max_dtype'], memory_format=torch.preserve_format)
        return state

    def step(self, closure: Optional[callable] = None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                assert not grad.is_sparse, f"{self} does not support sparse gradients"

                state = self._maybe_init_state(p, group)
                state["step"] += 1
                beta1, beta2 = group["betas"]
                compute_dtype = group["compute_dtype"]
                grad = grad.to(compute_dtype)

                # Decay the first and second moment running average coefficient
                update = grad
                if beta1 != 0:
                    exp_avg = state["exp_avg"].to(compute_dtype)
                    exp_avg = exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    if exp_avg is not state["exp_avg"]:
                        state["exp_avg"].copy_(exp_avg)
                    update = exp_avg

                if beta2 != 0:
                    exp_avg_sq = state["exp_avg_sq"].to(compute_dtype)
                    exp_avg_sq = exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if exp_avg_sq is not state["exp_avg_sq"]:
                        state["exp_avg_sq"].copy_(exp_avg_sq)
                    if group["amsgrad"]:
                        exp_avg_sq = torch.maximum(exp_avg_sq, state["v_hat_max"], out=exp_avg_sq)
                        state["v_hat_max"].copy_(exp_avg_sq)
                    update /= exp_avg_sq.sqrt().add(group["eps"])

                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])  # note: this is later multiplied by learning rate

                update_scale = -group["lr"]
                debias_factor = 1
                # below: to save compute, we update scalar coefficient to account for debias/lamb/.. and multiply once
                if group["debias"]:
                    mt_debias = 1. / (1 - beta1 ** state["step"]) if beta1 != 0 else 1
                    vt_debias = 1. / math.sqrt(1 - beta2 ** state["step"]) if beta2 != 0 else 1
                    debias_factor = mt_debias / vt_debias
                    update_scale *= debias_factor

                if group["lamb"]:
                    weight_norm = torch.norm(p.data.to(compute_dtype))
                    update_norm = torch.norm(update) * debias_factor
                    # note: lamb cancels-out debias unless clamp_value is st
                    if group["clamp_value"] is not None:
                        weight_norm = torch.clamp_max_(weight_norm, group["clamp_value"])
                    if weight_norm == 0 or update_norm == 0:
                        trust_ratio = 1
                    else:
                        trust_ratio = weight_norm / update_norm
                    update_scale *= trust_ratio

                p.data.add_(update, alpha=update_scale)
        return loss
