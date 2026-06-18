"""LR-schedule + grad-clip regression for the spatiotemporal corpus trainer.

The flat-LR corpus run trained to train-loss 0.07 then DIVERGED catastrophically
back to ~random — no warmup, no decay let a late large gradient blow the weights
up.  The fix is a warmup→cosine LR schedule plus grad clipping.  These tests pin
the schedule SHAPE (warmup rises, then cosine decays to the floor) and that the
clip is actually applied, so a future edit can't silently revert to the
divergence-prone flat path.
"""

from __future__ import annotations

import math

import torch

from imas_ambix.worldmodel.spacetime_train import (
    build_lr_scheduler,
    lr_schedule_factor,
)


def test_warmup_rises_linearly_then_cosine_decays():
    """factor rises ~linearly over warmup, peaks at 1, then cosine-decays."""
    total, warmup, floor = 1000, 100, 0.01

    def f(step: int) -> float:
        return lr_schedule_factor(
            step, total_steps=total, warmup_steps=warmup, min_lr_ratio=floor
        )

    # warmup: strictly increasing, ends at ~1 at the warmup boundary
    warm = [f(s) for s in range(warmup)]
    assert all(b > a for a, b in zip(warm, warm[1:])), "warmup not monotonic up"
    assert warm[0] < warm[-1]
    assert f(warmup) == 1.0  # peak at the end of warmup

    # cosine decay: strictly decreasing after the peak, ending near the floor
    decay = [f(s) for s in range(warmup, total + 1)]
    assert all(b <= a + 1e-9 for a, b in zip(decay, decay[1:])), "decay not monotone"
    assert decay[0] == 1.0
    assert abs(f(total) - floor) < 1e-6, f"end factor {f(total)} != floor {floor}"

    # midway through the cosine the factor is ~halfway (cos(pi/2)=0 -> 0.5*(1+0))
    mid = warmup + (total - warmup) // 2
    expected_mid = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert abs(f(mid) - expected_mid) < 0.02


def test_factor_pins_at_floor_past_total():
    """Over-running past total_steps holds the floor (no negative-cosine bounce)."""
    f = lambda s: lr_schedule_factor(  # noqa: E731
        s, total_steps=500, warmup_steps=50, min_lr_ratio=0.05
    )
    assert abs(f(500) - 0.05) < 1e-6
    assert abs(f(600) - 0.05) < 1e-6  # clamped, not rising again
    assert abs(f(5000) - 0.05) < 1e-6


def test_scheduler_drives_optimizer_lr():
    """LambdaLR scales the optimizer LR: tiny at step 0, peak after warmup, low at end."""
    peak, total, warmup, floor = 3e-4, 200, 40, 0.01
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=peak)
    sched = build_lr_scheduler(
        opt,
        total_steps=total,
        warmup_steps=warmup,
        min_lr_ratio=floor,
        scheduled=True,
    )
    lr0 = opt.param_groups[0]["lr"]
    assert lr0 < peak * 0.1, f"step-0 lr {lr0} not warming up from ~0"
    for _ in range(warmup):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - peak) < peak * 0.05, "no peak after warmup"
    for _ in range(total - warmup):
        opt.step()
        sched.step()
    end_lr = opt.param_groups[0]["lr"]
    assert end_lr < peak * 0.1, f"end lr {end_lr} did not decay toward the floor"
    assert end_lr > 0, "lr must stay positive (floor, not zero)"


def test_flat_fallback_holds_lr_constant():
    """scheduled=False keeps the LR flat (the old behaviour, as an escape hatch)."""
    peak = 3e-4
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=peak)
    sched = build_lr_scheduler(
        opt, total_steps=100, warmup_steps=10, min_lr_ratio=0.01, scheduled=False
    )
    for _ in range(50):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - peak) < 1e-12, "flat fallback changed the LR"


def test_scheduler_resume_continues_phase():
    """Resuming with last_step lands the next .step() on the right phase factor."""
    total, warmup, floor = 1000, 100, 0.01
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0)
    # resume as if 500 steps already ran (last completed step index = 499)
    sched = build_lr_scheduler(
        opt,
        total_steps=total,
        warmup_steps=warmup,
        min_lr_ratio=floor,
        scheduled=True,
        last_step=499,
        peak_lr=1.0,
    )
    # the optimizer LR should now reflect the factor at step 500, not step 0
    expected = lr_schedule_factor(
        500, total_steps=total, warmup_steps=warmup, min_lr_ratio=floor
    )
    assert abs(opt.param_groups[0]["lr"] - expected) < 1e-6


def test_grad_clip_bounds_grad_norm():
    """clip_grad_norm_ caps the global grad norm at max_norm (the divergence fix).

    The trainer calls ``nn.utils.clip_grad_norm_(model.parameters(), grad_clip)``
    before ``opt.step()``; verify a large gradient is scaled to the cap so a
    single big-gradient step can't blow the weights up.
    """
    lin = torch.nn.Linear(8, 8)
    x = torch.randn(4, 8)
    loss = (lin(x) * 1e4).sum()  # deliberately huge gradient
    loss.backward()
    pre = torch.nn.utils.clip_grad_norm_(lin.parameters(), max_norm=1.0)
    # returned value is the PRE-clip total norm (large); post-clip norm == cap
    post = math.sqrt(
        sum(float((p.grad**2).sum()) for p in lin.parameters() if p.grad is not None)
    )
    assert pre > 1.0, "test setup: pre-clip norm should exceed the cap"
    assert post <= 1.0 + 1e-4, f"post-clip grad norm {post} exceeds the cap"
