import math

import torch
from typing import Tuple, Iterable, Union, Optional

from src.aq_ops import maybe_script


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
                compute_dtype = group.get("compute_dtype") or p.dtype
                grad = grad.to(compute_dtype)

                # Decay the first and second moment running average coefficient
                update = _inner_adam_step_and_update_statistics(
                    p, grad, state.get("exp_avg", grad), state.get("exp_avg_sq", grad), state.get("v_hat_max", grad),
                    compute_dtype, group["weight_decay"], beta1, beta2, group["amsgrad"], group["eps"]
                )

                update_scale = -group["lr"]
                # below: to save compute, we update scalar coefficient to account for debias/lamb/.. and multiply once
                if group["debias"]:
                    mt_debias = 1. / (1 - beta1 ** state["step"]) if beta1 != 0 else 1
                    vt_debias = 1. / math.sqrt(1 - beta2 ** state["step"]) if beta2 != 0 else 1
                    debias_factor = mt_debias / vt_debias
                    update_scale *= debias_factor

                if group["lamb"]:
                    weight_norm = torch.norm(p.data.to(compute_dtype))
                    update_norm = torch.norm(update)
                    # note: lamb does not count debiasing when computing trust ratio
                    if group["clamp_value"] is not None:
                        weight_norm = torch.clamp_max_(weight_norm, group["clamp_value"])
                    if weight_norm == 0 or update_norm == 0:
                        trust_ratio = 1
                    else:
                        trust_ratio = weight_norm / update_norm
                    update_scale *= trust_ratio

                p.data.add_(update, alpha=update_scale)
        return loss


@maybe_script
def _inner_adam_step_and_update_statistics(
    p: torch.Tensor, grad: torch.Tensor,
    exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, v_hat_max: torch.Tensor,
    compute_dtype: torch.dtype, weight_decay: float, beta1: float, beta2: float, amsgrad: bool, eps: float
    ):
    grad = grad.to(compute_dtype, copy=True)
    stored_exp_avg, stored_exp_avg_sq, stored_v_hat_max = exp_avg, exp_avg_sq, v_hat_max

    if beta1 != 0:
        exp_avg = exp_avg.to(compute_dtype) * beta1 + grad * (1 - beta1)
        if isinstance(stored_exp_avg, torch.Tensor):
            stored_exp_avg.copy_(exp_avg, non_blocking=True)
        update = exp_avg
    else:
        assert exp_avg is None
        update = grad.clone()

    if beta2 != 0:
        exp_avg_sq = exp_avg_sq.to(compute_dtype) * beta2 + grad.square() * (1 - beta2)
        stored_exp_avg.copy_(exp_avg_sq, non_blocking=True)
        if amsgrad:
            assert v_hat_max is not None
            exp_avg_sq = torch.maximum(exp_avg_sq, v_hat_max, out=exp_avg_sq)
            stored_v_hat_max.copy_(exp_avg_sq, non_blocking=True)
        else:
            assert v_hat_max is None
        update /= exp_avg_sq.sqrt().add(eps)
    else:
        assert exp_avg_sq is None

    if weight_decay != 0:
        update += update.add(p, alpha=weight_decay)  # note: this is later multiplied by -lr, see step
    return update
