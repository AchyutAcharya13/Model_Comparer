import math

import torch
from torch.optim import Optimizer


class AdaBound(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        final_lr=0.1,
        gamma=1e-3,
        eps=1e-8,
        weight_decay=0.0,
        amsbound=False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if final_lr <= 0.0:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if gamma <= 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super().__init__(params, defaults)

        # Keep each group's initial lr so final_lr scales consistently.
        self.base_lrs = [group["lr"] for group in self.param_groups]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            beta1, beta2 = group["betas"]
            amsbound = group["amsbound"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdaBound does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsbound:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                # Dynamic bound schedule from AdaBound paper.
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (1 - 1 / (group["gamma"] * t + 1))
                upper_bound = final_lr * (1 + 1 / (group["gamma"] * t))

                bounded_step = torch.full_like(denom, step_size).div_(denom)
                bounded_step.clamp_(lower_bound, upper_bound)
                p.data.addcmul_(exp_avg, bounded_step, value=-1.0)

        return loss


def get_optimizer(name, params, **kwargs):
    normalized = name.lower()
    if normalized == "adam":
        return torch.optim.Adam(params, **kwargs)
    if normalized == "sgd":
        return torch.optim.SGD(params, **kwargs)
    if normalized in {"adabound", "adabs"}:
        return AdaBound(params, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")