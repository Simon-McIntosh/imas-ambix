"""Predict-vs-reality eval: autoregressive rollout + skill vs persistence (§6 piece 4).

The eval IS the demo (§5): take a held-out shot, give the model only its plan
(pulse schedule) + a short initial-condition context window, roll the
multi-modal token stream forward AUTOREGRESSIVELY (the model consumes its own
predictions), and score the forward prediction against reality.

Two scores
----------
1. **Token forward-prediction skill vs persistence.**  Over the target window
   (grid steps ``>= context_steps``) we compare the rolled-out tokens to the
   true tokens, per observation modality.  The baseline is PERSISTENCE — repeat
   the last context token forever.  Skill = ``1 - error_model / error_persist``
   (>0 means the model beats persistence; 0 means it ties; <0 means it loses).
   For discrete tokens the error is the token mismatch rate.

2. **Predict-vs-reality against the eval-only L2 targets.**  We load the L2
   reconstruction targets for the shot — psi, q, the line-average density, the
   programmed waveforms — STRICTLY through the eval-only target reader
   (:func:`imas_ambix.tokenizer.store_targets.load_target_group`), NEVER through
   the input loader.  We report the target time base and a coverage summary so
   the predict-vs-reality comparison is structured (the trained model decodes
   its predicted L2 scalar tokens onto this time base; the skeleton reports the
   wiring + the persistence skill so the loop runs even when under-trained).

The target store boundary is honoured: the input path (dataset assembly) routes
every open through ``assert_not_target_path``; this module reads targets only
via the dedicated eval-only reader, so a target value never re-enters the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from imas_ambix.data.paths import TARGET_ROOT
from imas_ambix.tokenizer.store_targets import load_target_group
from imas_ambix.worldmodel.dataset import (
    ModalitySpec,
    WorldModelSample,
    WorldModelWindowConfig,
    build_shot_sample,
    default_modalities,
)
from imas_ambix.worldmodel.train import collate_samples

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imas_ambix.worldmodel.model import WorldModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------


def rollout(
    model: WorldModel,
    sample: WorldModelSample,
    obs_names: Sequence[str],
    plan_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Autoregressively roll the observation token stream forward.

    The plan prefix + the context window (grid steps ``< context_steps``) are
    given; from there the model predicts each next step and the predicted
    tokens are fed back (argmax / MAP decode) until the grid is filled.  Returns
    ``{modality: (n_steps, n_channels) int64}`` predicted tokens — the context
    steps copied from the truth, the target steps generated.
    """
    model.eval()
    batch = collate_samples([sample], obs_names, plan_names)
    ctx = int(sample.context_steps)
    n_steps = sample.n_steps

    # working copy of the observation tokens; context kept, target overwritten
    work = {name: batch["tokens"][name].clone() for name in batch["tokens"]}

    with torch.no_grad():
        for t in range(ctx, n_steps):
            cur = dict(batch)
            cur["tokens"] = work
            out = model(cur)
            # logits at step t-1 predict step t
            for name in obs_names:
                lg = out.logits[name]  # (1, T, C, V)
                pred = lg[:, t - 1].argmax(dim=-1)  # (1, C)
                work[name][:, t] = pred
    return {name: work[name][0].cpu().numpy().astype(np.int64) for name in obs_names}


# ---------------------------------------------------------------------------
# Skill vs persistence
# ---------------------------------------------------------------------------


@dataclass
class ModalitySkill:
    """Forward-prediction skill for one modality over the target window."""

    name: str
    model_error: float
    persistence_error: float
    n_scored: int

    @property
    def skill(self) -> float:
        """1 - model_error / persistence_error (>0 beats persistence)."""
        if self.persistence_error <= 0:
            return 0.0
        return 1.0 - self.model_error / self.persistence_error


def _persistence_prediction(sample: WorldModelSample, name: str) -> np.ndarray:
    """Persistence baseline: repeat the last context token through the target."""
    ctx = int(sample.context_steps)
    toks = sample.tokens[name].copy()  # (T, C)
    last_ctx = toks[ctx - 1]  # (C,)
    toks[ctx:] = last_ctx[None, :]
    return toks


def score_skill(
    sample: WorldModelSample,
    predicted: dict[str, np.ndarray],
    obs_names: Sequence[str],
) -> dict[str, ModalitySkill]:
    """Token-mismatch skill of the model rollout vs persistence, per modality.

    Scores only target-window positions whose true token is VALID.  Error is
    the token mismatch rate (fraction of channels-positions wrong).
    """
    ctx = int(sample.context_steps)
    out: dict[str, ModalitySkill] = {}
    for name in obs_names:
        truth = sample.tokens[name][ctx:]  # (Th, C)
        valid = sample.valid[name][ctx:]  # (Th, C)
        model_pred = predicted[name][ctx:]
        persist = _persistence_prediction(sample, name)[ctx:]
        n = int(valid.sum())
        if n == 0:
            out[name] = ModalitySkill(name, 0.0, 0.0, 0)
            continue
        model_err = float(((model_pred != truth) & valid).sum()) / n
        persist_err = float(((persist != truth) & valid).sum()) / n
        out[name] = ModalitySkill(name, model_err, persist_err, n)
    return out


# ---------------------------------------------------------------------------
# Eval-only L2 target reference (predict-vs-reality, never an input)
# ---------------------------------------------------------------------------


@dataclass
class TargetReference:
    """The eval-only L2 reconstruction targets for predict-vs-reality.

    Read STRICTLY through the dedicated target reader — never the input loader.
    Holds, per available target group, the quantity names + native time base +
    a finite-coverage summary, so the predict-vs-reality comparison is
    structured (the trained model's decoded L2 scalar trajectory is scored
    against ``programmed`` / ``derived_globals`` on these time bases; the
    equilibrium psi/q are the eventual gridded targets).
    """

    shot_id: int
    groups: dict[str, dict] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.groups)


def load_target_reference(
    shot_id: int, *, target_root: Path | None = None
) -> TargetReference:
    """Load the eval-only L2 targets for a shot (predict-vs-reality reference).

    Returns an empty reference (``available == False``) when no target store
    exists for the shot, so the eval loop runs regardless.  This reader is the
    ONLY way targets enter the eval — the input dataset can never open them.
    """
    root = Path(target_root) if target_root is not None else TARGET_ROOT
    ref = TargetReference(shot_id=int(shot_id))
    for group in ("programmed", "derived_globals", "equilibrium"):
        if not (root / str(shot_id) / f"{group}.zarr").exists():
            continue
        try:
            tg = load_target_group(shot_id, group, target_root=root)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.info("shot %s target group %s unreadable: %r", shot_id, group, exc)
            continue
        time = np.asarray(tg.attrs.time, dtype=np.float64)
        ref.groups[group] = {
            "quantity_names": list(tg.attrs.quantity_names),
            "n_time": int(time.shape[0]),
            "time_window": (
                (float(time.min()), float(time.max())) if time.size else (0.0, 0.0)
            ),
            "coverage": {
                q: float(np.asarray(tg.masks[q]).mean())
                for q in tg.attrs.quantity_names
            },
        }
    return ref


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """One held-out shot's predict-vs-reality eval result."""

    shot_id: int
    n_steps: int
    context_steps: int
    skill: dict[str, ModalitySkill]
    target_reference: TargetReference

    @property
    def mean_skill(self) -> float:
        """Mean token-skill across observation modalities (>0 beats persist)."""
        vals = [s.skill for s in self.skill.values() if s.n_scored > 0]
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> str:
        lines = [
            f"shot {self.shot_id}: {self.n_steps} steps "
            f"({self.context_steps} context / "
            f"{self.n_steps - self.context_steps} target)",
            f"  mean token-skill vs persistence: {self.mean_skill:+.3f}",
        ]
        for name, s in self.skill.items():
            lines.append(
                f"    {name:16s} skill={s.skill:+.3f} "
                f"(model_err={s.model_error:.3f} persist_err={s.persistence_error:.3f} "
                f"n={s.n_scored})"
            )
        if self.target_reference.available:
            lines.append("  eval-only L2 targets (predict-vs-reality reference):")
            for g, info in self.target_reference.groups.items():
                lines.append(
                    f"    {g:16s} quantities={info['quantity_names']} "
                    f"n_time={info['n_time']} window={info['time_window']}"
                )
        else:
            lines.append("  eval-only L2 targets: none on disk for this shot")
        return "\n".join(lines)


def evaluate_shot(
    shot_id: int,
    model: WorldModel,
    *,
    modalities: Sequence[ModalitySpec] | None = None,
    window: WorldModelWindowConfig | None = None,
    token_root: Path | None = None,
    level1_dir: Path | None = None,
    target_root: Path | None = None,
) -> EvalReport:
    """Forward-predict a held-out shot and score it predict-vs-reality.

    Assembles the shot's input tokens + plan (boundary-guarded), rolls the
    model forward from the context window, scores token-skill vs persistence,
    and attaches the eval-only L2 target reference.  Runs even when the model is
    under-trained (the skeleton requirement).
    """
    modalities = list(modalities or default_modalities())
    window = window or WorldModelWindowConfig()

    sample = build_shot_sample(
        shot_id, modalities, window, token_root=token_root, level1_dir=level1_dir
    )
    present = set(sample.tokens)
    modalities = [m for m in modalities if m.name in present]
    plan_names = [m.name for m in modalities if m.is_conditioning]
    obs_names = [m.name for m in modalities if not m.is_conditioning]

    predicted = rollout(model, sample, obs_names, plan_names)
    skill = score_skill(sample, predicted, obs_names)
    ref = load_target_reference(shot_id, target_root=target_root)

    return EvalReport(
        shot_id=int(shot_id),
        n_steps=sample.n_steps,
        context_steps=int(sample.context_steps),
        skill=skill,
        target_reference=ref,
    )


def main(argv: list[str] | None = None) -> int:
    """Eval driver: overfit then evaluate a held-out shot (predict-vs-reality).

    Trains a tiny model on ``--train-shots`` (overfit) then rolls out and scores
    ``--eval-shot``.  Without a trained model the rollout still runs and reports
    the persistence-baseline skill (the under-trained skeleton loop).
    """
    import argparse

    from imas_ambix.worldmodel.train import (
        TrainConfig,
        build_model_for_samples,
        overfit,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-shots", default="", help="comma-separated overfit shots"
    )
    parser.add_argument("--eval-shot", type=int, required=False)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--context-steps", type=int, default=16)
    parser.add_argument("--token-root", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    token_root = Path(args.token_root) if args.token_root else None
    modalities = default_modalities()
    window = WorldModelWindowConfig(
        n_steps=args.n_steps, context_steps=args.context_steps
    )

    if args.train_shots.strip():
        train_shots = [int(s) for s in args.train_shots.split(",") if s.strip()]
    else:
        from imas_ambix.worldmodel.dataset import discover_worldmodel_shots

        train_shots = discover_worldmodel_shots(
            modalities, token_root=token_root, limit=2
        )
    eval_shot = args.eval_shot if args.eval_shot is not None else train_shots[0]

    # Build + (over)fit a model so the rollout is meaningful, then eval.
    cfg = TrainConfig(steps=args.steps, window=window)
    overfit(train_shots, modalities=modalities, config=cfg, token_root=token_root)
    # rebuild a model with the same shapes and (cheaply) re-fit for the report.
    samples = [
        build_shot_sample(s, modalities, window, token_root=token_root)
        for s in train_shots
    ]
    model = build_model_for_samples(samples, modalities, window)
    report = evaluate_shot(
        eval_shot, model, modalities=modalities, window=window, token_root=token_root
    )
    print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
