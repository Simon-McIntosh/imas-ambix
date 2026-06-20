"""History-token corruption + control-dropout for closed-loop-robust training.

Why this exists (the open-loop DRIFT failure)
----------------------------------------------
The signal-conditioned camera transformer is trained TEACHER-FORCED: every step
it sees the TRUE preceding frames and predicts the next.  At rollout time it
instead consumes its OWN previously-predicted frames as context.  Those
predictions carry small errors, and because the model has NEVER been shown a
slightly-wrong context during training, the errors compound — the rollout drifts
off the data manifold over a long horizon (coherent for a few frames, then
degrades).  This is exactly the exposure-bias / open-loop drift that the GameNGen
recipe fixes for a diffusion frame model: corrupt the conditioning frames during
training and condition on the corruption level, so the model LEARNS to correct
the errors it will later feed itself.

This module is the discrete-token analogue of that recipe, plus classifier-free
guidance support:

* :func:`corrupt_context_tokens` randomly REPLACES a fraction of the *context*
  frame token positions with random in-vocabulary token ids (a discrete noising
  of the history).  Only frames ``< context_frames`` are corrupted — the frames
  the rollout will hand back to itself; the forecast-window frames and the
  prediction TARGET are left untouched (the model must still predict the true
  next frame from a corrupted history).  The per-sample corruption RATE is
  quantised to a small bin index and returned, so the model can condition on
  "how corrupt is this history" via a learned per-level embedding — at inference
  the bin is 0 ("trust the context"), which the model has also seen (rate 0 is a
  valid training draw).

* :func:`sample_corruption_rates` draws the per-sample corruption rate for a
  batch (a fraction of samples get rate 0 so the clean regime stays well-trained)
  and maps each rate to its bin index.

* :func:`sample_control_dropout` decides, per sample, whether to ZERO the
  actuator/plan + measured-signal conditioning for this step (classifier-free
  guidance: a model trained with the conditioning sometimes dropped can, at
  inference, extrapolate the conditioned prediction away from the unconditioned
  one — which is what makes the pulse schedule a load-bearing, steerable control
  rather than a weak hint).

All operations are deterministic given a ``torch.Generator`` so a training step
is reproducible and a DDP rank's corruption is independent of another rank's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContextCorruptionConfig:
    """Knobs for the history-corruption + control-dropout fine-tune.

    Attributes
    ----------
    max_rate:
        Largest fraction of context tokens replaced (the rate is drawn uniformly
        in ``[0, max_rate]`` for the corrupted fraction of samples).
    levels:
        Number of discrete corruption-rate bins the model conditions on (the
        learned per-level embedding has this many rows).  Bin 0 always means
        "rate 0 / clean"; the remaining bins partition ``(0, max_rate]``.
    clean_fraction:
        Fraction of samples per step kept at rate 0 (bin 0) so the clean regime —
        the regime the first rollout frames are in — stays well-trained.
    control_dropout:
        Probability per sample of zeroing the plan + signal conditioning this
        step (classifier-free guidance).  0 disables it.
    """

    max_rate: float = 0.30
    levels: int = 8
    clean_fraction: float = 0.25
    control_dropout: float = 0.15

    @property
    def enabled(self) -> bool:
        return self.levels > 1 and self.max_rate > 0.0

    def rate_to_bin(self, rate: float) -> int:
        """Map a corruption rate in ``[0, max_rate]`` to a bin index ``[0, levels)``.

        Rate exactly 0 is bin 0 (clean).  A positive rate falls in one of the
        ``levels - 1`` equal-width sub-bins partitioning ``(0, max_rate]``, so the
        embedding row a sample uses is a faithful, monotone code of its rate.
        """
        if self.levels <= 1 or rate <= 0.0 or self.max_rate <= 0.0:
            return 0
        frac = min(max(rate / self.max_rate, 0.0), 1.0)
        # (0, 1] -> bins 1..levels-1 (bin 0 reserved for the clean rate-0 case).
        b = 1 + int(frac * (self.levels - 1 - 1e-9))
        return int(min(max(b, 1), self.levels - 1))


def sample_corruption_rates(
    batch_size: int,
    cfg: ContextCorruptionConfig,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw per-sample corruption rates + their bin indices for one batch.

    A ``clean_fraction`` of the samples get rate 0 (bin 0); the rest get a rate
    drawn uniformly in ``(0, max_rate]``.  Returns ``(rates (B,) float,
    bins (B,) long)`` on ``device``.
    """
    dev = device or torch.device("cpu")
    if not cfg.enabled:
        return (
            torch.zeros(batch_size, device=dev),
            torch.zeros(batch_size, dtype=torch.long, device=dev),
        )
    u = torch.rand(batch_size, generator=generator, device=dev)
    clean_draw = torch.rand(batch_size, generator=generator, device=dev)
    is_clean = clean_draw < cfg.clean_fraction
    rates = torch.where(is_clean, torch.zeros_like(u), u.clamp_min(1e-3) * cfg.max_rate)
    bins = torch.tensor(
        [cfg.rate_to_bin(float(r)) for r in rates.tolist()],
        dtype=torch.long,
        device=dev,
    )
    return rates, bins


def corrupt_context_tokens(
    frames: torch.Tensor,
    rates: torch.Tensor,
    *,
    context_frames: int,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace a per-sample fraction of CONTEXT frame tokens with random ids.

    ``frames`` is ``(B, T, S)`` LOCAL token ids; ``rates`` is ``(B,)`` the
    fraction of context-token positions to replace for each sample.  Only frames
    ``< context_frames`` are touched (the history the rollout re-feeds itself);
    the forecast frames and the next-frame TARGET are left intact, so the model
    must predict the TRUE next frame from a noised history.  Replacement ids are
    uniform in ``[0, vocab_size)`` — a discrete noising, not a structured mask.

    Returns a NEW tensor (the input is not modified in place).
    """
    if context_frames < 1 or frames.ndim != 3:
        return frames
    b, t, s = frames.shape
    ctx = int(min(context_frames, t))
    if ctx < 1:
        return frames
    out = frames.clone()
    # per-(sample, context-frame, position) Bernoulli mask at the sample's rate.
    rate = rates.to(frames.device).view(b, 1, 1).clamp(0.0, 1.0)
    draw = torch.rand((b, ctx, s), generator=generator, device=frames.device)
    mask = draw < rate  # (B, ctx, S)
    if not bool(mask.any()):
        return out
    rand_ids = torch.randint(
        0, int(vocab_size), (b, ctx, s), generator=generator, device=frames.device
    )
    out[:, :ctx][mask] = rand_ids[mask]
    return out


def sample_control_dropout(
    batch_size: int,
    cfg: ContextCorruptionConfig,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Per-sample boolean: zero this sample's plan + signal conditioning?

    Returns ``(B,)`` bool on ``device``; ``True`` means drop the conditioning for
    that sample this step (classifier-free guidance).  All ``False`` when
    ``control_dropout`` is 0.
    """
    dev = device or torch.device("cpu")
    if cfg.control_dropout <= 0.0:
        return torch.zeros(batch_size, dtype=torch.bool, device=dev)
    return torch.rand(batch_size, generator=generator, device=dev) < cfg.control_dropout


def apply_control_dropout(
    plan: torch.Tensor | None,
    signals: dict[str, torch.Tensor] | None,
    drop: torch.Tensor,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
    """Zero the plan + every signal block for the samples flagged in ``drop``.

    Zeroing sets a sample's conditioning tokens to PAD id 0 (a real in-vocab id,
    so the embedding lookup is valid) — the model then sees an all-PAD plan /
    signal prefix for that sample, i.e. the unconditioned prediction.  Returns
    new tensors (inputs untouched); a ``None`` plan/signals passes through.  The
    per-stream params stay in the autograd graph because the block is still
    present (just PAD-valued), so DDP uniformity is preserved.
    """
    if not bool(drop.any()):
        return plan, signals
    new_plan = plan
    if plan is not None and plan.numel() and plan.shape[1] > 0:
        new_plan = plan.clone()
        new_plan[drop] = 0
    new_signals = signals
    if signals:
        new_signals = {}
        for name, block in signals.items():
            nb = block.clone()
            if nb.numel() and nb.shape[1] > 0:
                nb[drop] = 0
            new_signals[name] = nb
    return new_plan, new_signals


__all__ = [
    "ContextCorruptionConfig",
    "apply_control_dropout",
    "corrupt_context_tokens",
    "sample_control_dropout",
    "sample_corruption_rates",
]
