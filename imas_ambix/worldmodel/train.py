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
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from imas_ambix.worldmodel.dataset import (
    ModalitySpec,
    WorldModelDataset,
    WorldModelSample,
    WorldModelWindowConfig,
    build_shot_sample,
    default_modalities,
    discover_worldmodel_shots,
)
from imas_ambix.worldmodel.model import (
    ModalityHeadSpec,
    WorldModel,
    WorldModelConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Default checkpoint root on the shared GPFS (survives node drains / reboots).
DEFAULT_CKPT_ROOT = Path("/work/projects/imas_gpu/worldmodel/ckpt")


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


def pad_collate_batch(
    samples: Sequence[WorldModelSample],
    obs_names: Sequence[str],
    plan_names: Sequence[str],
    channels: dict[str, int],
) -> dict:
    """Collate a list of (possibly heterogeneous) samples into a fixed batch.

    A corpus DataLoader draws shots whose per-modality channel counts may
    differ (a shot may carry fewer pf_active coils, a flaky camera, etc.).
    The model's embedding tables and heads are sized to a FIXED per-modality
    channel count (``channels[name]``), so this collate pads every sample's
    tokens to that width with :data:`~imas_ambix.worldmodel.dataset.PAD_LOCAL_ID`
    and marks the padded channels invalid — they carry no embedding signal
    (PAD id) and no loss (masked).  A modality entirely absent from a sample is
    emitted as an all-PAD, all-invalid block so the stacked batch is rectangular.

    Returns the same shape contract as :func:`collate_samples`:
    ``{"tokens": {name: (B, T, C)}, "valid": {name: (B, T, C) bool},
    "context_steps": int, "shot_ids": [...]}``.
    """
    from imas_ambix.worldmodel.dataset import PAD_LOCAL_ID

    names = list(plan_names) + list(obs_names)
    n_steps = int(samples[0].n_steps)
    tokens: dict[str, torch.Tensor] = {}
    valid: dict[str, torch.Tensor] = {}
    for name in names:
        c = int(channels[name])
        tok_rows: list[torch.Tensor] = []
        val_rows: list[torch.Tensor] = []
        for s in samples:
            t = torch.full((n_steps, c), PAD_LOCAL_ID, dtype=torch.long)
            v = torch.zeros((n_steps, c), dtype=torch.bool)
            if name in s.tokens:
                src = torch.as_tensor(s.tokens[name], dtype=torch.long)
                sval = torch.as_tensor(s.valid[name], dtype=torch.bool)
                cc = min(c, src.shape[1])
                t[:, :cc] = src[:, :cc]
                v[:, :cc] = sval[:, :cc]
            tok_rows.append(t)
            val_rows.append(v)
        tokens[name] = torch.stack(tok_rows)
        valid[name] = torch.stack(val_rows)
    return {
        "tokens": tokens,
        "valid": valid,
        "context_steps": int(samples[0].context_steps),
        "shot_ids": [int(s.shot_id) for s in samples],
    }


def _wm_collate_fn(
    batch: Sequence[WorldModelSample],
    obs_names: Sequence[str],
    plan_names: Sequence[str],
    channels: dict[str, int],
) -> dict:
    """torch DataLoader ``collate_fn`` (closes over the batch contract)."""
    return pad_collate_batch(batch, obs_names, plan_names, channels)


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
# Checkpointing + resume (drain-safe; eval-loadable)
# ---------------------------------------------------------------------------


def _config_to_dict(cfg: WorldModelConfig) -> dict:
    """Serialise a :class:`WorldModelConfig` to a plain dict for a checkpoint.

    Captures the per-modality head specs (name / vocab / channels /
    is_conditioning) and the backbone shape so the model can be rebuilt
    EXACTLY at load time — the checkpoint is self-describing and does not
    depend on re-discovering shots or re-reading the corpus.
    """
    return {
        "modalities": [
            {
                "name": m.name,
                "vocab_size": int(m.vocab_size),
                "n_channels": int(m.n_channels),
                "is_conditioning": bool(m.is_conditioning),
            }
            for m in cfg.modalities
        ],
        "d_model": int(cfg.d_model),
        "n_layers": int(cfg.n_layers),
        "n_heads": int(cfg.n_heads),
        "d_ff": int(cfg.d_ff),
        "dropout": float(cfg.dropout),
        "plan_steps": int(cfg.plan_steps),
        "obs_steps": int(cfg.obs_steps),
    }


def _config_from_dict(d: dict) -> WorldModelConfig:
    """Rebuild a :class:`WorldModelConfig` from :func:`_config_to_dict`."""
    heads = [
        ModalityHeadSpec(
            name=m["name"],
            vocab_size=int(m["vocab_size"]),
            n_channels=int(m["n_channels"]),
            is_conditioning=bool(m["is_conditioning"]),
        )
        for m in d["modalities"]
    ]
    return WorldModelConfig(
        modalities=heads,
        d_model=int(d["d_model"]),
        n_layers=int(d["n_layers"]),
        n_heads=int(d["n_heads"]),
        d_ff=int(d["d_ff"]),
        dropout=float(d["dropout"]),
        plan_steps=int(d["plan_steps"]),
        obs_steps=int(d["obs_steps"]),
    )


def save_checkpoint(
    out_dir: Path,
    *,
    model: WorldModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    window: WorldModelWindowConfig,
    extra: dict | None = None,
) -> Path:
    """Atomically write a self-describing checkpoint to ``out_dir/latest.pt``.

    The checkpoint carries the model state-dict, the optimiser state-dict, the
    step counter, the model config (so :func:`load_model_from_checkpoint` can
    rebuild the exact model without the corpus), and the window config.  Written
    to a temp file then ``os.replace``-d so a SIGKILL mid-write never leaves a
    half-written ``latest.pt`` (drain-safety: the GPU node has a drain history).
    A monotonic ``ckpt-<step>.pt`` snapshot is also dropped for history.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "model_config": _config_to_dict(model.config),
        "window": {
            "n_steps": int(window.n_steps),
            "context_steps": int(window.context_steps),
            "grid_window": window.grid_window,
        },
        "extra": dict(extra or {}),
    }
    final = out_dir / "latest.pt"
    tmp = out_dir / f".latest.pt.{os.getpid()}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)  # atomic on POSIX same-filesystem
    # best-effort monotonic snapshot (non-atomic; latest.pt is the source of truth)
    try:
        torch.save(payload, out_dir / f"ckpt-{int(step):08d}.pt")
    except OSError as exc:  # noqa: BLE001 — snapshot is a nicety, not load-bearing
        logger.warning("could not write step snapshot: %r", exc)
    return final


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint payload (the dict written by :func:`save_checkpoint`)."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def load_model_from_checkpoint(
    path: Path, *, map_location: str = "cpu"
) -> tuple[WorldModel, dict]:
    """Rebuild the :class:`WorldModel` from a checkpoint and load its weights.

    Returns ``(model, payload)``.  This is the eval entry point — ``eval.py``
    (or any consumer) gets a ready-to-roll model with no corpus dependency.
    The model is rebuilt from the saved config so embedding-table and head
    shapes match the checkpointed weights exactly.
    """
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = _config_from_dict(payload["model_config"])
    model = WorldModel(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    return model, payload


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """Return ``out_dir/latest.pt`` if it exists, else None (resume probe)."""
    p = Path(out_dir) / "latest.pt"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Corpus trainer (a DataLoader over MANY shots — the real training loop)
# ---------------------------------------------------------------------------


@dataclass
class CorpusTrainConfig:
    """Corpus-training hyper-parameters (distinct from the overfit config).

    Attributes
    ----------
    steps:
        Optimiser steps (a step is one batch).  The DataLoader is cycled until
        this many steps run, so it does NOT re-descend a single fixed batch.
    batch_size:
        Shots per batch.
    lr, weight_decay:
        AdamW learning rate + decoupled weight decay.
    seed:
        RNG seed (torch + DataLoader shuffle).
    log_every, ckpt_every, eval_every:
        Logging / checkpoint / held-out-eval cadences, in steps.
    num_workers:
        DataLoader worker processes (0 = main-process load; safe for the tiny
        synthetic unit test).
    window:
        Common-grid + context/target split.
    model_kwargs:
        Backbone overrides (``d_model``, ``n_layers``, ...).
    grad_clip:
        Max global grad-norm (0 disables).
    n_eval_shots:
        Held-out shots scored at each eval.
    """

    steps: int = 2000
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0
    log_every: int = 25
    ckpt_every: int = 200
    eval_every: int = 500
    num_workers: int = 0
    window: WorldModelWindowConfig = field(default_factory=WorldModelWindowConfig)
    model_kwargs: dict = field(default_factory=dict)
    grad_clip: float = 1.0
    n_eval_shots: int = 1


@dataclass
class CorpusTrainResult:
    """Outcome of a corpus-training run."""

    steps_run: int
    initial_loss: float
    final_loss: float
    losses: list[float]
    n_parameters: int
    context_length: int
    n_train_shots: int
    n_eval_shots: int
    eval_skills: list[tuple[int, float]]  # (step, mean_skill) at each eval
    checkpoint_path: str | None


def _resolve_channels(
    shot_ids: Sequence[int],
    modalities: Sequence[ModalitySpec],
    window: WorldModelWindowConfig,
    *,
    token_root: Path | None,
    level1_dir: Path | None,
    probe: int = 8,
) -> tuple[list[ModalitySpec], dict[str, int]]:
    """Probe a few shots to fix the per-modality channel counts + common set.

    The model's tables/heads must be sized once.  We assemble up to ``probe``
    shots, keep only modalities present in EVERY probed shot, and take the MAX
    channel count seen for each (the pad collate pads narrower shots up).
    Returns ``(kept_modalities, {name: n_channels})``.
    """
    samples: list[WorldModelSample] = []
    for sid in shot_ids[:probe]:
        try:
            samples.append(
                build_shot_sample(
                    int(sid),
                    modalities,
                    window,
                    token_root=token_root,
                    level1_dir=level1_dir,
                )
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.info("channel probe: shot %s unreadable: %r", sid, exc)
            continue
    if not samples:
        raise ValueError("channel probe: no shot assembled — cannot size the model")
    common = set.intersection(*(set(s.tokens) for s in samples))
    kept = [m for m in modalities if m.name in common]
    channels = {
        m.name: max(int(s.tokens[m.name].shape[1]) for s in samples) for m in kept
    }
    return kept, channels


def _build_corpus_model(
    modalities: Sequence[ModalitySpec],
    channels: dict[str, int],
    window: WorldModelWindowConfig,
    **model_kwargs: object,
) -> WorldModel:
    """Build a :class:`WorldModel` sized to the resolved channel counts."""
    cfg = WorldModelConfig.from_modalities(
        modalities,
        channels,
        plan_steps=window.n_steps,
        obs_steps=window.n_steps,
        **model_kwargs,  # type: ignore[arg-type]
    )
    return WorldModel(cfg)


def train_corpus(
    shot_ids: Sequence[int],
    *,
    modalities: Sequence[ModalitySpec] | None = None,
    config: CorpusTrainConfig | None = None,
    out_dir: Path | None = None,
    token_root: Path | None = None,
    level1_dir: Path | None = None,
    eval_shot_ids: Sequence[int] | None = None,
    device: str | None = None,
    resume: bool = True,
) -> CorpusTrainResult:
    """Train the plan-conditioned world model on a CORPUS of shots.

    A torch ``DataLoader`` draws DISTINCT shots (shuffled, cycled to the step
    budget), each forward pass scores the TARGET-WINDOW forecasting objective
    (``target_only`` next-token NLL — predict the target window from the plan
    prefix + context window), and an AdamW optimiser descends.  The model is
    loaded ONCE outside the loop (the in-process default; AGENTS.md §2b).

    Drain-safety (AGENTS.md §2a / §2b)
    ----------------------------------
    * a SIGTERM/SIGINT handler sets a STOP flag; the loop checkpoints + exits
      cleanly within ``UnkillableStepTimeout``;
    * a checkpoint (model + optimiser + step) is written every
      ``config.ckpt_every`` steps to ``out_dir`` (atomic ``latest.pt``);
    * on entry the latest checkpoint is RESUMED if present (``resume=True``);
    * a ``try/finally`` releases the model + empties the CUDA cache.

    Periodically (``config.eval_every``) a held-out shot is rolled out and the
    skill-vs-persistence number is logged, so the demo metric is visible as
    training progresses.
    """
    config = config or CorpusTrainConfig()
    modalities = list(modalities or default_modalities())
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir()
    _set_determinism(config.seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    stop = _StopFlag()
    stop.install()

    plan_names = [m.name for m in modalities if m.is_conditioning]
    obs_names = [m.name for m in modalities if not m.is_conditioning]
    if not plan_names:
        raise ValueError("no conditioning (plan) modality requested")
    if not obs_names:
        raise ValueError("no observation modality requested")

    # Size the model ONCE from a probe of the corpus.
    kept, channels = _resolve_channels(
        shot_ids,
        modalities,
        config.window,
        token_root=token_root,
        level1_dir=level1_dir,
    )
    plan_names = [n for n in plan_names if n in channels]
    obs_names = [n for n in obs_names if n in channels]
    if not plan_names or not obs_names:
        raise ValueError("probe found no common plan+obs modalities across the corpus")

    model = _build_corpus_model(kept, channels, config.window, **config.model_kwargs)
    model.to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # ── Resume from the latest checkpoint, if present ───────────────────────
    start_step = 0
    eval_skills: list[tuple[int, float]] = []
    if resume:
        latest = find_latest_checkpoint(out_dir)
        if latest is not None:
            payload = load_checkpoint(latest)
            try:
                model.load_state_dict(payload["model_state_dict"])
                if payload.get("optimizer_state_dict") is not None:
                    opt.load_state_dict(payload["optimizer_state_dict"])
                start_step = int(payload.get("step", 0))
                eval_skills = list(payload.get("extra", {}).get("eval_skills", []))
                logger.info("RESUMED from %s at step %d", latest, start_step)
            except (KeyError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "checkpoint %s incompatible (%r) — starting fresh", latest, exc
                )
                start_step = 0

    # ── DataLoader over MANY distinct shots ─────────────────────────────────
    dataset = WorldModelDataset(
        shot_ids, kept, config.window, token_root=token_root, level1_dir=level1_dir
    )
    chan = dict(channels)

    def _collate(batch):  # noqa: ANN001, ANN202
        return _wm_collate_fn(batch, obs_names, plan_names, chan)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )

    model.train()
    losses: list[float] = []
    step = start_step
    ckpt_path: Path | None = find_latest_checkpoint(out_dir) if resume else None
    t_last = time.time()
    try:
        done = False
        while not done:
            for batch in loader:
                if stop.stop or step >= config.steps:
                    done = True
                    break
                batch = _batch_to_device(batch, dev)
                opt.zero_grad(set_to_none=True)
                out = model(batch)
                loss = next_token_nll(out.logits, batch, obs_names, target_only=True)
                loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()
                losses.append(float(loss.detach()))
                step += 1

                if step % config.log_every == 0 or step == config.steps:
                    rate = config.log_every / max(time.time() - t_last, 1e-6)
                    t_last = time.time()
                    logger.info(
                        "corpus step %d/%d loss=%.4f (%.2f steps/s)",
                        step,
                        config.steps,
                        losses[-1],
                        rate,
                    )

                if config.ckpt_every > 0 and step % config.ckpt_every == 0:
                    ckpt_path = save_checkpoint(
                        out_dir,
                        model=model,
                        optimizer=opt,
                        step=step,
                        window=config.window,
                        extra={"eval_skills": eval_skills},
                    )
                    logger.info("checkpoint saved at step %d -> %s", step, ckpt_path)

                if (
                    config.eval_every > 0
                    and step % config.eval_every == 0
                    and eval_shot_ids
                ):
                    mean_skill = _run_periodic_eval(
                        model,
                        eval_shot_ids,
                        kept,
                        config.window,
                        token_root=token_root,
                        level1_dir=level1_dir,
                        n=config.n_eval_shots,
                    )
                    eval_skills.append((step, mean_skill))
                    logger.info(
                        "EVAL @ step %d: mean token-skill vs persistence = %+.4f",
                        step,
                        mean_skill,
                    )
                    model.train()

        # ── Final checkpoint (always — including on STOP) ───────────────────
        ckpt_path = save_checkpoint(
            out_dir,
            model=model,
            optimizer=opt,
            step=step,
            window=config.window,
            extra={"eval_skills": eval_skills},
        )
        logger.info("final checkpoint at step %d -> %s", step, ckpt_path)

        result = CorpusTrainResult(
            steps_run=step - start_step,
            initial_loss=losses[0] if losses else float("nan"),
            final_loss=losses[-1] if losses else float("nan"),
            losses=losses,
            n_parameters=model.num_parameters(),
            context_length=model.context_length(),
            n_train_shots=len(dataset),
            n_eval_shots=len(eval_shot_ids or []),
            eval_skills=eval_skills,
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _default_out_dir() -> Path:
    """Resolve the checkpoint out-dir, keyed by SLURM job id when available."""
    run_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("WM_RUN_ID") or "local"
    return DEFAULT_CKPT_ROOT / str(run_id)


def _batch_to_device(batch: dict, dev: torch.device) -> dict:
    """Move a collated batch's token / valid tensors to ``dev`` (in place dict)."""
    if dev.type == "cpu":
        return batch
    batch = dict(batch)
    batch["tokens"] = {k: v.to(dev) for k, v in batch["tokens"].items()}
    batch["valid"] = {k: v.to(dev) for k, v in batch["valid"].items()}
    return batch


def _run_periodic_eval(
    model: WorldModel,
    eval_shot_ids: Sequence[int],
    modalities: Sequence[ModalitySpec],
    window: WorldModelWindowConfig,
    *,
    token_root: Path | None,
    level1_dir: Path | None,
    n: int,
) -> float:
    """Roll out held-out shots and return the mean token-skill vs persistence.

    Imports :mod:`imas_ambix.worldmodel.eval` lazily (it imports back from this
    module — keeps the import graph acyclic at module-load time).  Eval runs on
    CPU on a clone of the model's weights so the device transfer in rollout is
    simple and the trained model is untouched (still on its training device).
    """
    from imas_ambix.worldmodel.eval import evaluate_shot

    model.eval()
    skills: list[float] = []
    # eval rollout assembles tensors on CPU; run the model on CPU for it.
    cpu_model = WorldModel(model.config)
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()
    for sid in list(eval_shot_ids)[:n]:
        try:
            report = evaluate_shot(
                int(sid),
                cpu_model,
                modalities=modalities,
                window=window,
                token_root=token_root,
                level1_dir=level1_dir,
            )
            skills.append(report.mean_skill)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.info("periodic eval: shot %s skipped: %r", sid, exc)
            continue
    del cpu_model
    if not skills:
        return 0.0
    return float(sum(skills) / len(skills))


# ---------------------------------------------------------------------------
# CLI (overfit driver — used by the sbatch's smoke phase)
# ---------------------------------------------------------------------------


def _cmd_overfit(args) -> int:  # noqa: ANN001
    """Overfit a handful of shots — the end-to-end wiring proof (NOT training)."""
    modalities = default_modalities()
    token_root = Path(args.token_root) if args.token_root else None
    if args.shots.strip():
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        shots = discover_worldmodel_shots(modalities, token_root=token_root, limit=2)
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
    result = overfit(shots, modalities=modalities, config=cfg, token_root=token_root)
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


def _resolve_corpus_shots(args, modalities, token_root) -> list[int]:  # noqa: ANN001
    """Resolve the train shot list: explicit ``--shots``, else discover."""
    if args.shots.strip():
        return [int(s) for s in args.shots.split(",") if s.strip()]
    return discover_worldmodel_shots(
        modalities, token_root=token_root, limit=args.n_shots
    )


def _cmd_corpus(args) -> int:  # noqa: ANN001
    """Train the plan-conditioned world model on the corpus (the real loop)."""
    modalities = default_modalities()
    if getattr(args, "modalities", ""):
        want = [s.strip() for s in args.modalities.split(",") if s.strip()]
        by_name = {m.name: m for m in modalities}
        unknown = [w for w in want if w not in by_name]
        if unknown:
            logger.error(
                "unknown --modalities %s; available: %s", unknown, sorted(by_name)
            )
            return 1
        modalities = [by_name[w] for w in want]
        logger.info("modality subset: %s", [m.name for m in modalities])
    token_root = Path(args.token_root) if args.token_root else None
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()

    shots = _resolve_corpus_shots(args, modalities, token_root)
    if len(shots) < 1:
        logger.error("no shots discovered/given for corpus training")
        return 1

    # Held-out eval shots: a tail slice not used for training, unless given.
    if args.eval_shots.strip():
        eval_shots = [int(s) for s in args.eval_shots.split(",") if s.strip()]
        train_shots = shots
    elif len(shots) > 1:
        n_eval = max(1, min(args.n_eval_shots, len(shots) // 5))
        eval_shots = shots[-n_eval:]
        train_shots = shots[:-n_eval]
    else:
        eval_shots = shots
        train_shots = shots

    cfg = CorpusTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        n_eval_shots=args.n_eval_shots,
        window=WorldModelWindowConfig(
            n_steps=args.n_steps, context_steps=args.context_steps
        ),
    )
    logger.info(
        "corpus train: %d train shots, %d eval shots, out_dir=%s",
        len(train_shots),
        len(eval_shots),
        out_dir,
    )
    result = train_corpus(
        train_shots,
        modalities=modalities,
        config=cfg,
        out_dir=out_dir,
        token_root=token_root,
        eval_shot_ids=eval_shots,
        resume=not args.no_resume,
    )
    logger.info(
        "CORPUS TRAIN done: steps_run=%d params=%d ctx_len=%d "
        "initial=%.4f final=%.4f ckpt=%s",
        result.steps_run,
        result.n_parameters,
        result.context_length,
        result.initial_loss,
        result.final_loss,
        result.checkpoint_path,
    )
    if result.eval_skills:
        logger.info("eval skill trajectory: %s", result.eval_skills)
    return 0


def main(argv: list[str] | None = None) -> int:
    """World-model training driver.

    Subcommands::

        python -m imas_ambix.worldmodel.train corpus --steps 5000 ...   # REAL train
        python -m imas_ambix.worldmodel.train overfit --shots 24065 ... # wiring proof

    ``corpus`` runs a DataLoader over many distinct shots with checkpointing,
    resume, and periodic held-out eval — the demoable trainer.  ``overfit``
    descends on a handful of fixed shots (the end-to-end wiring proof, NOT a
    training run).
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    # --- corpus (the real trainer) ---
    pc = sub.add_parser("corpus", help="train on a corpus of many shots")
    pc.add_argument("--shots", default="", help="comma-separated train shot ids")
    pc.add_argument("--n-shots", type=int, default=512, help="discovered-shot cap")
    pc.add_argument("--eval-shots", default="", help="comma-separated held-out shots")
    pc.add_argument("--n-eval-shots", type=int, default=2)
    pc.add_argument("--steps", type=int, default=2000)
    pc.add_argument("--batch-size", type=int, default=8)
    pc.add_argument("--lr", type=float, default=1e-3)
    pc.add_argument("--weight-decay", type=float, default=0.01)
    pc.add_argument("--n-steps", type=int, default=128, help="grid steps")
    pc.add_argument("--context-steps", type=int, default=32)
    pc.add_argument("--log-every", type=int, default=25)
    pc.add_argument("--ckpt-every", type=int, default=200)
    pc.add_argument("--eval-every", type=int, default=500)
    pc.add_argument("--num-workers", type=int, default=4)
    pc.add_argument("--out-dir", default=None, help="checkpoint dir (default GPFS)")
    pc.add_argument("--token-root", default=None)
    pc.add_argument("--no-resume", action="store_true", help="ignore latest.pt")
    pc.add_argument(
        "--modalities",
        default="",
        help="comma-separated subset of default_modalities by name "
        "(e.g. pulse_schedule,summary,pf_active,interferometer); empty = all",
    )
    pc.set_defaults(func=_cmd_corpus)

    # --- overfit (the wiring proof) ---
    po = sub.add_parser("overfit", help="overfit a few shots (wiring proof)")
    po.add_argument("--shots", default="", help="comma-separated shot ids")
    po.add_argument("--steps", type=int, default=300)
    po.add_argument("--lr", type=float, default=3e-3)
    po.add_argument("--n-steps", type=int, default=64, help="grid steps")
    po.add_argument("--context-steps", type=int, default=16)
    po.add_argument("--token-root", default=None)
    po.set_defaults(func=_cmd_overfit)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
