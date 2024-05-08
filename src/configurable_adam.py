import math
from contextlib import contextmanager

import torch
from typing import Tuple, Iterable, Union, Optional

from src.aq_ops import maybe_script


class ConfigurableAdamW(torch.optim.Optimizer):
    r"""
    A version of Adam optimizer that supports custom parameter dtypes, amsgrad, lamb or rmsprop on per-group basis.
    Adam and Amsgrad based on https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
    Lamb flag based on https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
    This was tested to match Adam and Lamb exactly for torch 2.3.0 (when compute_dtypes are all None)

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
            debias: Optional[bool] = None,
            amsgrad: bool = False,
            lamb: bool = False,
            clamp_value: Optional[float] = None,
            compute_dtype: Optional[torch.dtype] = None,
            exp_avg_dtype: Optional[torch.dtype] = None,
            exp_avg_sq_dtype: Optional[torch.dtype] = None,
            v_hat_max_dtype: Optional[torch.dtype] = None,
            exp_avg_device: torch.device = None,
            exp_avg_sq_device: torch.device = None,
            v_hat_max_device: torch.device = None,
    ) -> None:
        debias = debias if debias is not None else not lamb
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, debias=debias, amsgrad=amsgrad, lamb=lamb,
            clamp_value=clamp_value, compute_dtype=compute_dtype, exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype, v_hat_max_dtype=v_hat_max_dtype, exp_avg_device=exp_avg_device,
            exp_avg_sq_device=exp_avg_sq_device, v_hat_max_device=v_hat_max_device,
        )
        super().__init__(params, defaults)

    def _maybe_init_state(self, param: torch.Tensor, group: dict) -> dict:
        group["state_offload_device"]
        state = self.state[param]
        if "step" not in state:
            state["step"] = 0
        if group["betas"][0] != 0 and "exp_avg" not in state:
            pin_memory = group["exp_avg_device"] == torch.device("cpu")
            state["exp_avg"] = torch.zeros_like(
                param, dtype=group['exp_avg_dtype'],
                memory_format=torch.preserve_format,
                device=group["exp_avg_device"],
                pin_memory=pin_memory)
        if group["betas"][1] not in (0, 1) and "exp_avg_sq" not in state:
            pin_memory = group["exp_avg_sq_device"] == torch.device("cpu")
            state["exp_avg_sq"] = torch.zeros_like(
                param, dtype=group['exp_avg_sq_dtype'],
                memory_format=torch.preserve_format,
                device=group["exp_avg_sq_device"],
                pin_memory=pin_memory)
        if group["betas"] and "v_hat_max" not in state:
            pin_memory = group["v_hat_max_device"] == torch.device("cpu")
            state["v_hat_max"] = torch.zeros_like(
                param, dtype=group['v_hat_max_dtype'],
                memory_format=torch.preserve_format,
                device=group["v_hat_max_device"],
                pin_memory=pin_memory)
        return state

    @torch.no_grad()
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

                with (
                    fetch_to_device(state.get("exp_avg", p)) as exp_avg,
                    fetch_to_device(state.get("exp_avg_sq", p)) as exp_avg_sq,
                    fetch_to_device(state.get("v_hat_max", p)) as v_hat_max,
                ):
                    state = self._maybe_init_state(p, group)
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    compute_dtype = group.get("compute_dtype") or p.dtype

                    if not group["lamb"] and group["weight_decay"] != 0:
                        p.data = p.data.mul_(1 - group["lr"] * group["weight_decay"])
                        # adam weight decay is not scaled by bias correction

                    # Decay the first and second moment running average coefficient
                    update = _inner_adam_step_and_update_statistics(
                        p, grad, exp_avg, exp_avg_sq, v_hat_max,
                        beta1, beta2, group["eps"], group["amsgrad"], compute_dtype
                    )

                    if group["lamb"] and group["weight_decay"] != 0:
                        update = update.add(p, alpha=group["weight_decay"])
                        # lamb weight decay is later multiplied by -lr * trust_ratio * bias_correction

                    update_scale = -group["lr"]
                    # below: to save compute, we update scalar coefficient to account for debias/lamb/.. and multiply once
                    if group["debias"]:
                        mt_debias = 1. / (1 - beta1 ** state["step"]) if beta1 != 0 else 1
                        vt_debias = 1. / math.sqrt(1 - beta2 ** state["step"]) if beta2 != 0 else 1
                        bias_correction = mt_debias / vt_debias
                        update_scale *= bias_correction

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
        p: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, v_hat_max: torch.Tensor,
        beta1: float, beta2: float, eps: float, amsgrad: bool, compute_dtype: torch.dtype,
):
    grad = grad.to(compute_dtype, copy=True)
    stored_exp_avg, stored_exp_avg_sq, stored_v_hat_max = exp_avg, exp_avg_sq, v_hat_max
    if beta1 != 0:
        exp_avg = exp_avg.to(compute_dtype) * beta1 + grad * (1 - beta1)
        stored_exp_avg.copy_(exp_avg, non_blocking=True)
        update = exp_avg
    else:
        update = grad.clone()

    if beta2 == 1:
        pass
    else:
        if beta2 == 0:
            exp_avg_sq = grad.square()
        else:
            exp_avg_sq = exp_avg_sq.to(compute_dtype) * beta2 + grad.square() * (1 - beta2)
            stored_exp_avg_sq.copy_(exp_avg_sq, non_blocking=True)

        if amsgrad:
            exp_avg_sq = torch.maximum(exp_avg_sq, v_hat_max, out=exp_avg_sq)
            stored_v_hat_max.copy_(exp_avg_sq, non_blocking=True)

        update /= exp_avg_sq.sqrt().add(eps)

    return update


@contextmanager
def fetch_to_device(x: torch.Tensor, device: torch.device):
    fetched = x.to(device, non_blocking=True)
    try:
        yield fetched
    finally:
        x.copy_(fetched, non_blocking=True)
