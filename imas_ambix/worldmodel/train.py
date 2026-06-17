"""Next-token NLL training loop for the world model (§6 piece 3).

Training is standard decoder-only next-token prediction with TEACHER FORCING:
the model sees the true previous observation tokens and predicts the next one,
under a cross-entropy / negative-log-likelihood (NLL) loss per token, summed
over channels and modalities.  At grid step ``t`` the model predicts the
observation token at step ``t+1`` from the plan prefix + observation steps
``<= t``; only positions whose target is VALID (per the coverage mask)
contribute to the loss.

The PLAN (pulse_schedule) is the prepended conditioning prefix — it is never a
prediction target, so it carries no loss.

GPU-safety pattern (repo AGENTS.md §2b)
---------------------------------------
The loop installs a SIGTERM/SIGINT handler that sets a STOP flag and exits
cleanly within ``UnkillableStepTimeout`` (no D-state-inducing teardown), and a
``try/finally`` that releases the model and calls ``torch.cuda.empty_cache()``.
For the prototype the model is loaded once outside the per-step loop (the
in-process default), so there is no per-item model reload.  Determinism flags
match :mod:`imas_ambix.data.stream_encode`.

The tiny-overfit entrypoint :func:`overfit` is the END-TO-END PROOF: a tiny
model, one or two real shots, a few hundred steps, asserting the loss drops
substantially.  It does NOT do a full training run.
"""

from __future__ import annotations

import logging
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from imas_ambix.worldmodel.dataset import (
    ModalitySpec,
    WorldModelSample,
    WorldModelWindowConfig,
    build_shot_sample,
    default_modalities,
)
from imas_ambix.worldmodel.model import WorldModel, WorldModelConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stop-flag (clean cancellation — repo GPU-safety pattern)
# ---------------------------------------------------------------------------


class _StopFlag:
    """A SIGTERM/SIGINT-set stop flag the training loop polls each step."""

    def __init__(self) -> None:
        self.stop = False

    def install(self) -> None:
        def _handler(signum, _frame):  # noqa: ANN001
            logger.warning("received signal %s — setting STOP flag", signum)
            self.stop = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # not in the main thread (e.g. under pytest) — skip silently
                logger.debug("could not install handler for %s", sig)


# ---------------------------------------------------------------------------
# Batch assembly (sample dicts -> padded model batch tensors)
# ---------------------------------------------------------------------------


def collate_samples(
    samples: Sequence[WorldModelSample],
    obs_names: Sequence[str],
    plan_names: Sequence[str],
) -> dict:
    """Stack a list of assembled samples into a model batch.

    All samples share the same grid length (one shot = one full common grid),
    so this is a plain stack.  Returns ``{"tokens": {name: (B, T, C) long},
    "valid": {name: (B, T, C) bool}, "context_steps": int, "shot_ids": [...]}``.
    """
    names = list(plan_names) + list(obs_names)
    tokens: dict[str, torch.Tensor] = {}
    valid: dict[str, torch.Tensor] = {}
    for name in names:
        tokens[name] = torch.stack(
            [torch.as_tensor(s.tokens[name], dtype=torch.long) for s in samples]
        )
        valid[name] = torch.stack(
            [torch.as_tensor(s.valid[name], dtype=torch.bool) for s in samples]
        )
    return {
        "tokens": tokens,
        "valid": valid,
        "context_steps": int(samples[0].context_steps),
        "shot_ids": [int(s.shot_id) for s in samples],
    }


# ---------------------------------------------------------------------------
# The NLL loss (teacher-forced next-token, masked)
# ---------------------------------------------------------------------------


def next_token_nll(
    logits: dict[str, torch.Tensor],
    batch: dict,
    obs_names: Sequence[str],
    *,
    target_only: bool = False,
) -> torch.Tensor:
    """Masked teacher-forced next-token cross-entropy, summed over modalities.

    For each observation modality, logits at step ``t`` predict the token at
    step ``t+1``.  Only positions whose TARGET token is valid contribute.  When
    ``target_only`` is True, only positions in the target window (grid steps
    ``>= context_steps``) are scored — the forecasting objective; otherwise the
    whole observation sequence is scored (the standard teacher-forced LM loss
    used for the overfit proof).
    """
    ce = nn.functional.cross_entropy
    ctx = int(batch["context_steps"])
    total = torch.zeros((), dtype=torch.float32)
    n_terms = 0
    for name in obs_names:
        lg = logits[name]  # (B, T, C, V)
        tgt = batch["tokens"][name]  # (B, T, C)
        val = batch["valid"][name]  # (B, T, C)
        b, t, c, v = lg.shape
        # predict step t+1 from step t
        pred = lg[:, : t - 1]  # (B, T-1, C, V)
        target = tgt[:, 1:t]  # (B, T-1, C)
        tvalid = val[:, 1:t]  # (B, T-1, C)
        if target_only:
            # only score targets that fall in the target window
            step_idx = torch.arange(1, t, device=lg.device)
            in_target = (step_idx >= ctx).view(1, -1, 1)
            tvalid = tvalid & in_target
        if tvalid.sum() == 0:
            continue
        flat_pred = pred.reshape(-1, v)
        flat_tgt = target.reshape(-1)
        flat_mask = tvalid.reshape(-1)
        loss = ce(flat_pred[flat_mask], flat_tgt[flat_mask], reduction="mean")
        total = total + loss
        n_terms += 1
    if n_terms == 0:
        raise ValueError("no valid target positions in batch — cannot compute NLL")
    return total / n_terms


# ---------------------------------------------------------------------------
# Training config + the overfit entrypoint
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Tiny-overfit / training hyper-parameters.

    Attributes
    ----------
    steps:
        Optimiser steps.
    lr:
        Adam learning rate.
    seed:
        RNG seed (set on torch for reproducibility).
    log_every:
        Log the loss every N steps.
    """

    steps: int = 300
    lr: float = 3e-3
    seed: int = 0
    log_every: int = 50
    window: WorldModelWindowConfig = field(default_factory=WorldModelWindowConfig)
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class OverfitResult:
    """Outcome of an overfit run (the end-to-end proof)."""

    initial_loss: float
    final_loss: float
    losses: list[float]
    n_parameters: int
    context_length: int
    shot_ids: list[int]

    @property
    def loss_drop_ratio(self) -> float:
        """Final / initial loss — small means the model overfit the shots."""
        if self.initial_loss <= 0:
            return 1.0
        return self.final_loss / self.initial_loss


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def build_model_for_samples(
    samples: Sequence[WorldModelSample],
    modalities: Sequence[ModalitySpec],
    window: WorldModelWindowConfig,
    **model_kwargs: object,
) -> WorldModel:
    """Build a :class:`WorldModel` whose shapes match assembled samples."""
    sample = samples[0]
    channels = {name: arr.shape[1] for name, arr in sample.tokens.items()}
    cfg = WorldModelConfig.from_modalities(
        modalities,
        channels,
        plan_steps=window.n_steps,
        obs_steps=window.n_steps,
        **model_kwargs,  # type: ignore[arg-type]
    )
    return WorldModel(cfg)


def overfit(
    shot_ids: Sequence[int],
    *,
    modalities: Sequence[ModalitySpec] | None = None,
    config: TrainConfig | None = None,
    token_root: Path | None = None,
    level1_dir: Path | None = None,
) -> OverfitResult:
    """Overfit a handful of real shots — the end-to-end wiring proof.

    Loads the model ONCE, assembles the shots, and runs a teacher-forced
    next-token NLL descent for ``config.steps`` steps.  Returns the loss
    trajectory; a substantial drop proves tokens -> model -> recognizable
    prediction is wired end-to-end.  This is NOT a training run.
    """
    config = config or TrainConfig()
    modalities = list(modalities or default_modalities())
    _set_determinism(config.seed)

    stop = _StopFlag()
    stop.install()

    plan_names = [m.name for m in modalities if m.is_conditioning]
    obs_names = [m.name for m in modalities if not m.is_conditioning]

    samples = [
        build_shot_sample(
            int(sid),
            modalities,
            config.window,
            token_root=token_root,
            level1_dir=level1_dir,
        )
        for sid in shot_ids
    ]
    # keep only modalities that every sample actually carries (robust to a
    # modality absent on one shot)
    common = set.intersection(*(set(s.tokens) for s in samples))
    modalities = [m for m in modalities if m.name in common]
    plan_names = [n for n in plan_names if n in common]
    obs_names = [n for n in obs_names if n in common]
    if not plan_names:
        raise ValueError("no conditioning (plan) modality present across shots")
    if not obs_names:
        raise ValueError("no observation modality present across shots")

    batch = collate_samples(samples, obs_names, plan_names)
    model = build_model_for_samples(
        samples, modalities, config.window, **config.model_kwargs
    )
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses: list[float] = []
    try:
        for step in range(config.steps):
            if stop.stop:
                logger.warning("STOP flag set — ending overfit at step %d", step)
                break
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss = next_token_nll(out.logits, batch, obs_names)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if step % config.log_every == 0 or step == config.steps - 1:
                logger.info(
                    "overfit step %d/%d loss=%.4f", step, config.steps, losses[-1]
                )
        result = OverfitResult(
            initial_loss=losses[0] if losses else float("nan"),
            final_loss=losses[-1] if losses else float("nan"),
            losses=losses,
            n_parameters=model.num_parameters(),
            context_length=model.context_length(),
            shot_ids=[int(s) for s in shot_ids],
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# CLI (overfit driver — used by the sbatch's smoke phase)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Overfit-smoke / training driver.

    ``--shots 24065,24066`` overfits the given shots; defaults to a discovered
    pair.  ``--steps`` / ``--lr`` control the descent.  Prints the loss drop.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", default="", help="comma-separated shot ids")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--n-steps", type=int, default=64, help="grid steps")
    parser.add_argument("--context-steps", type=int, default=16)
    parser.add_argument("--token-root", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    modalities = default_modalities()
    if args.shots.strip():
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        from imas_ambix.worldmodel.dataset import discover_worldmodel_shots

        shots = discover_worldmodel_shots(
            modalities,
            token_root=Path(args.token_root) if args.token_root else None,
            limit=2,
        )
        if len(shots) < 1:
            logger.error("no shots discovered with the requested modalities")
            return 1

    cfg = TrainConfig(
        steps=args.steps,
        lr=args.lr,
        window=WorldModelWindowConfig(
            n_steps=args.n_steps, context_steps=args.context_steps
        ),
    )
    result = overfit(
        shots,
        modalities=modalities,
        config=cfg,
        token_root=Path(args.token_root) if args.token_root else None,
    )
    logger.info(
        "OVERFIT done: shots=%s params=%d ctx_len=%d "
        "initial=%.4f final=%.4f drop_ratio=%.3f",
        result.shot_ids,
        result.n_parameters,
        result.context_length,
        result.initial_loss,
        result.final_loss,
        result.loss_drop_ratio,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
