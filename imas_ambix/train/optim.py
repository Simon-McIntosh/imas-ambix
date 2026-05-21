"""Optimizer and LR-scheduler helpers for the WHAM training loop.

All torch imports are deferred to function bodies so this module can be
imported without a GPU / torch environment (useful in config-only contexts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def build_adamw(
    model_params: list,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Return an AdamW optimizer configured for Llama-style training.

    Parameters
    ----------
    model_params:
        Parameter groups or raw parameter list to optimise.
    lr:
        Peak learning rate (applied as the initial LR; the scheduler
        drives warm-up and decay).
    weight_decay:
        L2 regularisation weight (default 0.1 in the v0 recipe).
    betas:
        AdamW exponential decay coefficients ``(beta1, beta2)``.
        Default ``(0.9, 0.95)`` follows the Llama-2 standard used in
        ``plans/world-model-v0.md`` §4.1.
    eps:
        Numerical stability epsilon.

    Returns
    -------
    torch.optim.AdamW
    """
    import torch

    return torch.optim.AdamW(
        model_params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )


def build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_frac: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up followed by cosine decay to ``min_lr_frac × peak_lr``.

    The schedule is implemented as a ``LambdaLR`` multiplier:

    * ``step < warmup_steps``  →  LR rises linearly from 0 to ``peak_lr``.
    * ``warmup_steps ≤ step ≤ max_steps``  →  cosine decay from ``peak_lr``
      to ``min_lr_frac × peak_lr``.
    * ``step > max_steps``  →  LR is clamped at ``min_lr_frac × peak_lr``.

    Parameters
    ----------
    optimizer:
        The optimizer whose LR is managed.
    warmup_steps:
        Number of steps for the linear warm-up phase.
    max_steps:
        Total number of training steps (end of the cosine curve).
    min_lr_frac:
        Fraction of peak LR to maintain after the cosine decay completes.
        Default 0.1 (10 % of peak) per ``plans/world-model-v0.md`` §4.1.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    """
    import math

    import torch

    def lr_lambda(current_step: int) -> float:
        # Linear warm-up
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # After max_steps: hold at min fraction
        if current_step >= max_steps:
            return min_lr_frac

        # Cosine decay from warmup_steps → max_steps
        decay_steps = max_steps - warmup_steps
        progress = float(current_step - warmup_steps) / float(max(1, decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale: 1.0 → min_lr_frac
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
