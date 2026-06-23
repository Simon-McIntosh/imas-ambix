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
# Distributed data-parallel (DDP) — opt-in via the launcher, never a rewrite
# ---------------------------------------------------------------------------
#
# The single-device path is the DEFAULT and is UNCHANGED: when the run is not
# launched under torchrun / SLURM-distributed (``WORLD_SIZE`` unset or == 1),
# ``DistEnv.from_environment`` reports ``world_size == 1`` and every DDP branch
# in ``train_corpus`` is a no-op (no process group, no wrapper, no sampler).
# Multi-GPU is opted into ONLY by the launch (``torchrun --nproc_per_node=2``),
# which sets RANK / LOCAL_RANK / WORLD_SIZE; each rank then pins its own card,
# trains a DISJOINT shard (DistributedSampler), and only RANK 0 logs /
# checkpoints / evals.  Every rank hits every collective symmetrically — a
# rank-asymmetric collective desynchronises the NCCL ring and hangs the job
# (AGENTS.md §2a), so the barriers below are placed on ALL ranks.


@dataclass
class DistEnv:
    """Resolved distributed-launch environment (single-proc when not launched).

    Attributes
    ----------
    rank:
        Global process rank (0 when single-proc).
    local_rank:
        Node-local rank == the GPU index this process pins (0 when single-proc).
    world_size:
        Total process count (1 when single-proc — the default single-device
        path).
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        """True only under a real multi-process launch (world_size > 1)."""
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """RANK 0 — the only rank that logs / checkpoints / evals."""
        return self.rank == 0

    @classmethod
    def from_environment(cls) -> DistEnv:
        """Read RANK / LOCAL_RANK / WORLD_SIZE from the launcher env.

        torchrun and SLURM set these; absent (or WORLD_SIZE == 1) means the
        ordinary single-device run — the back-compat default.
        """
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        if ws <= 1:
            return cls(rank=0, local_rank=0, world_size=1)
        return cls(
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=ws,
        )


def init_distributed(dist_env: DistEnv) -> None:
    """Init the NCCL process group + pin this rank's GPU (no-op single-proc).

    Called once per process before the model is built.  On CUDA each rank pins
    ``cuda:local_rank`` so DDP's allreduce and the per-rank DataLoader land on
    the right card.  A no-op when ``not dist_env.enabled``.
    """
    if not dist_env.enabled:
        return
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(dist_env.local_rank)
        backend = "nccl"
    else:  # CPU integration/smoke fallback (gloo) — keeps the path testable
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    logger.info(
        "DDP init: rank %d/%d local_rank %d backend %s",
        dist_env.rank,
        dist_env.world_size,
        dist_env.local_rank,
        backend,
    )


def shutdown_distributed(dist_env: DistEnv) -> None:
    """Destroy the process group (no-op single-proc).  Always safe to call."""
    if not dist_env.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception as exc:  # noqa: BLE001 — teardown must not raise
            logger.warning("DDP final barrier note: %r", exc)
        dist.destroy_process_group()


def _barrier(dist_env: DistEnv) -> None:
    """Symmetric barrier across ALL ranks (no-op single-proc).

    MUST be reached by every rank — an asymmetric call hangs the NCCL ring
    (AGENTS.md §2a).  Callers place it identically on every rank.
    """
    if not dist_env.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def _unwrap(model: nn.Module) -> WorldModel:
    """Return the underlying :class:`WorldModel`, through a DDP wrapper if any.

    ``encode`` / ``channel_logits`` / ``channel_query`` live on the
    ``WorldModel`` itself; DDP wraps only ``forward``, so the chunked loss path
    (which calls ``encode`` directly) must reach the inner module.  Works
    whether ``model`` is a bare ``WorldModel`` or a ``DistributedDataParallel``.
    """
    return getattr(model, "module", model)  # type: ignore[return-value]


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
# Chunked NLL — the same loss, never materialising all-channel logits
# ---------------------------------------------------------------------------


def chunked_next_token_nll(
    model: WorldModel,
    obs_hidden: torch.Tensor,
    batch: dict,
    obs_names: Sequence[str],
    *,
    target_only: bool = False,
    chunk_channels: int = 16,
) -> torch.Tensor:
    """Masked teacher-forced next-token NLL computed CHANNEL-CHUNK at a time.

    Thin wrapper around :meth:`WorldModel.chunked_nll` (the single source of
    truth for the numerics) — kept as a module-level function so the overfit
    smoke + the equivalence tests call it directly.  Numerically EQUAL to
    :func:`next_token_nll` (the per-modality ``cross_entropy(reduction="mean")``
    over the valid (position, channel) pairs, then the mean over modalities),
    but the per-channel logits are built one channel-chunk at a time and freed,
    so peak head memory is ``~chunk_channels × vocab`` not
    ``n_channels × vocab`` — the only way the full-resolution rbb head (256
    channels × 2^18 vocab) fits.  ``model`` is the unwrapped :class:`WorldModel`
    (see :func:`_unwrap` under DDP).
    """
    return _unwrap(model).chunked_nll(
        obs_hidden,
        batch,
        obs_names,
        target_only=target_only,
        chunk_channels=chunk_channels,
    )


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
    loss_chunk_channels: int = 16


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
            # Chunked loss path (memory-safe for a full-resolution camera head):
            # encode once, then accumulate NLL channel-chunk at a time.  For the
            # small synthetic shots this is identical to next_token_nll on the
            # full logits; for a full-res rbb overfit it is the only path that
            # fits.  ``target_only=False`` keeps the standard teacher-forced LM
            # loss the overfit proof descends.
            obs_hidden = model.encode(batch)
            loss = chunked_next_token_nll(
                model,
                obs_hidden,
                batch,
                obs_names,
                target_only=False,
                chunk_channels=config.loss_chunk_channels,
            )
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
    prefetch_factor:
        DataLoader batches each worker stages ahead (>=4 floor when workers > 0)
        so the loader runs ahead of the GPU; ignored when ``num_workers == 0``.
    loss_chunk_channels:
        Channel-chunk size for the chunked next-token NLL (see
        :func:`chunked_next_token_nll`).  Peak head memory is
        ``~chunk × vocab``, so this caps the full-resolution camera head's
        logit footprint without changing the loss value.
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
    prefetch_factor: int = 4
    loss_chunk_channels: int = 16


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
    """Fix the per-modality channel widths for ALL declared modalities.

    The model's tables/heads must be sized ONCE, and they are built for EVERY
    declared modality unconditionally (cameras/HF/L2 heads always exist) — NOT
    for the subset a tiny probe happened to intersect.  Cameras live in high
    shot-ids, so a probe over the first few shots may carry none of them; a
    probe-intersection sizing would silently drop the camera/HF heads and
    collapse the all-streams model.  Here every declared modality is KEPT and
    its fixed channel width resolved robustly:

    * the MAX channel count seen across the probed shots that actually carried
      the modality (so a present modality is sized to real data), else
    * the spec's :meth:`ModalitySpec.fixed_channel_width` — a structural
      constant for cameras (the 16×16 frame grid at the camera stride) and for
      any modality pinning ``n_channels`` — so a modality ABSENT from every
      probe shot (a camera not in the low-id probe band) is STILL sized
      correctly, never zero/absent, else
    * ``1`` as a last-resort non-zero floor (a degenerate signal_hf group with
      neither a probe sample nor a declared width).

    A shot carrying more/fewer channels than the fixed width is pad/truncated by
    :func:`pad_collate_batch`; a shot lacking the modality entirely is the
    all-PAD masked block.  Returns ``(declared_modalities, {name: width})``.
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

    # KEEP every declared modality (unconditional heads) — the probe only fixes
    # widths, it does NOT decide which modalities exist.
    kept = list(modalities)
    channels: dict[str, int] = {}
    for m in kept:
        probed = [int(s.tokens[m.name].shape[1]) for s in samples if m.name in s.tokens]
        if probed:
            channels[m.name] = max(probed)
            continue
        fixed = m.fixed_channel_width()
        channels[m.name] = int(fixed) if fixed and fixed >= 1 else 1
        logger.info(
            "channel probe: modality %s absent from all %d probed shots — "
            "sizing from spec fixed width = %d",
            m.name,
            len(samples),
            channels[m.name],
        )
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
    cache_dir: str | os.PathLike | None = None,
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

    # ── Distributed-launch detection (opt-in; single-proc default unchanged) ──
    dist_env = DistEnv.from_environment()
    init_distributed(dist_env)

    if device is None:
        # each rank pins its OWN card (cuda:local_rank); single-proc -> cuda:0
        device = f"cuda:{dist_env.local_rank}" if torch.cuda.is_available() else "cpu"
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

    base_model = _build_corpus_model(
        kept, channels, config.window, **config.model_kwargs
    )
    base_model.to(dev)
    # Log the resolved backbone size + parameter budget at STARTUP (before the
    # loop) so a scaled run is auditable from the first line — the documented
    # contract is that the param count and context length are the LIVE numbers,
    # never guessed (model.num_parameters / model.context_length).  Rank-0 only.
    if dist_env.is_main:
        logger.info(
            "model built on %s: d_model=%d n_layers=%d n_heads=%d d_ff=%d "
            "dropout=%.3g | params=%d (%.2fM) ctx_len=%d | world_size=%d "
            "modalities=%s channels=%s",
            dev,
            base_model.config.d_model,
            base_model.config.n_layers,
            base_model.config.n_heads,
            base_model.config.d_ff,
            base_model.config.dropout,
            base_model.num_parameters(),
            base_model.num_parameters() / 1e6,
            base_model.context_length(),
            dist_env.world_size,
            [m.name for m in kept],
            {k: int(v) for k, v in channels.items()},
        )

    # ── Resume from the latest checkpoint, if present ───────────────────────
    # Load into the BASE model BEFORE the DDP wrap so every rank starts from the
    # same weights (DDP broadcasts rank-0's state at construction anyway, but
    # loading on all ranks keeps the optimiser-state load symmetric).
    start_step = 0
    eval_skills: list[tuple[int, float]] = []
    opt = torch.optim.AdamW(
        base_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    if resume:
        latest = find_latest_checkpoint(out_dir)
        if latest is not None:
            payload = load_checkpoint(latest)
            try:
                base_model.load_state_dict(payload["model_state_dict"])
                if payload.get("optimizer_state_dict") is not None:
                    opt.load_state_dict(payload["optimizer_state_dict"])
                start_step = int(payload.get("step", 0))
                eval_skills = list(payload.get("extra", {}).get("eval_skills", []))
                if dist_env.is_main:
                    logger.info("RESUMED from %s at step %d", latest, start_step)
            except (KeyError, RuntimeError, ValueError) as exc:
                if dist_env.is_main:
                    logger.warning(
                        "checkpoint %s incompatible (%r) — starting fresh",
                        latest,
                        exc,
                    )
                start_step = 0

    # ── Wrap in DDP (data-parallel) when launched multi-process ─────────────
    # DDP wraps forward + all-reduces grads; the chunked loss path calls
    # ``encode`` on the UNWRAPPED model (``_unwrap``), so the backbone forward
    # still runs through DDP's autograd hooks via the loss .backward().  The
    # single-proc path leaves ``model`` as the bare WorldModel (no wrapper).
    if dist_env.enabled:
        from torch.nn.parallel import DistributedDataParallel

        # find_unused_parameters=True is REQUIRED here: per shot only the
        # modalities actually present receive a gradient (an absent camera /
        # HF stream is an all-PAD, masked block → its head + channel_query get
        # NO grad that step), and the DistributedSampler gives each rank a
        # DIFFERENT shot shard, so DIFFERENT params are unused on each rank.
        # Without this flag DDP's reducer raises "Expected to have finished
        # reduction in the prior iteration" because it expects every param to
        # participate every iteration.  The flag makes the reducer mark the
        # unused params ready so the all-reduce completes symmetrically.
        ddp_kwargs: dict = {"find_unused_parameters": True}
        if dev.type == "cuda":
            ddp_kwargs["device_ids"] = [dist_env.local_rank]
            ddp_kwargs["output_device"] = dist_env.local_rank
        model: nn.Module = DistributedDataParallel(base_model, **ddp_kwargs)
    else:
        model = base_model
    core = _unwrap(model)  # the WorldModel, for encode / channel_query / config

    # ── DataLoader over MANY distinct shots ─────────────────────────────────
    # The assembled-sample cache (opt-in via cache_dir / WM_CACHE_DIR) lets the
    # DataLoader workers torch.load a node-local NVMe file after the first epoch
    # instead of re-reading + re-assembling per-shot tokens from GPFS every epoch
    # (the camera frame reads dominate) — keeps the single H200 continuously fed.
    dataset = WorldModelDataset(
        shot_ids,
        kept,
        config.window,
        token_root=token_root,
        level1_dir=level1_dir,
        cache_dir=cache_dir,
    )
    chan = dict(channels)

    def _collate(batch):  # noqa: ANN001, ANN202
        return _wm_collate_fn(batch, obs_names, plan_names, chan)

    # ── Per-rank sharding (DDP) — each rank trains a DISJOINT shot shard ──────
    # The DistributedSampler partitions the shot list across ranks so no shot is
    # trained twice per epoch; set_epoch (below) reshuffles each cycle.  The
    # per-rank DataLoader + the node-local NVMe cache + the CUDA prefetcher are
    # UNCHANGED — each rank has its own loader; the cache on /scratch_local is
    # shared and atomic (os.replace), and the disjoint shards mean ranks mostly
    # populate DISJOINT cache entries (a concurrent same-shot write across ranks
    # just overwrites byte-identically — no torn read).  Single-proc keeps the
    # legacy shuffle=True loader.
    sampler = None
    if dist_env.enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist_env.world_size,
            rank=dist_env.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )

    # Keep the worker pool + the node-local NVMe cache WARM across epochs and
    # let the loader run ahead of the GPU: persistent_workers avoids re-forking
    # (and re-warming the cache) every epoch, prefetch_factor builds a backlog of
    # ready batches per worker, and pin_memory stages them in page-locked host
    # memory so the prefetcher's async H2D copy (below) actually overlaps compute.
    # pin_memory only helps on CUDA; skip it on CPU.
    use_workers = config.num_workers > 0
    loader_kwargs: dict = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        # a sampler is mutually exclusive with shuffle — the DistributedSampler
        # owns shuffling under DDP; the single-proc loader keeps shuffle=True.
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=_collate,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=(dev.type == "cuda"),
        # Use 'fork' (not the py3.14 forkserver/spawn default) so the local
        # collate closure + the dataset need no pickling; the workers do CPU
        # zarr reads only (never CUDA), so forking after the model is on the GPU
        # is safe — the established AGENTS.md §2b data-worker pattern.
        multiprocessing_context="fork" if use_workers else None,
    )
    if use_workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(4, config.prefetch_factor)
    loader = torch.utils.data.DataLoader(**loader_kwargs)

    # Overlap the next batch's host->device copy with the current batch's
    # compute via a double-buffered side CUDA stream (no-op pass-through on CPU),
    # so the GPU is fed continuously rather than stalling on each H2D transfer.
    prefetcher = CudaPrefetcher(loader, dev)

    model.train()
    losses: list[float] = []
    step = start_step
    ckpt_path: Path | None = find_latest_checkpoint(out_dir) if resume else None
    t_last = time.time()
    epoch = 0
    try:
        done = False
        while not done:
            # reshuffle the per-rank shard each pass (DDP requirement) so a long
            # run does not see the same shard order every cycle; no-op single-proc.
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch += 1
            for batch in prefetcher:
                if stop.stop or step >= config.steps:
                    done = True
                    break
                opt.zero_grad(set_to_none=True)
                # Memory-safe path: the CHUNKED next-token NLL is computed
                # INSIDE forward (loss_spec) so the full-resolution camera head
                # (256 ch × 2^18 vocab) never materialises all-channel logits
                # AND a DDP wrapper drives forward — letting its reducer see the
                # full backbone+head autograd graph and all-reduce every grad.
                # Single-proc: ``model`` is the bare WorldModel; identical loss.
                loss = model(
                    batch,
                    loss_spec={
                        "obs_names": obs_names,
                        "target_only": True,
                        "chunk_channels": config.loss_chunk_channels,
                    },
                )
                loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()
                losses.append(float(loss.detach()))
                step += 1

                if dist_env.is_main and (
                    step % config.log_every == 0 or step == config.steps
                ):
                    rate = config.log_every / max(time.time() - t_last, 1e-6)
                    t_last = time.time()
                    logger.info(
                        "corpus step %d/%d loss=%.4f (%.2f steps/s, world=%d)",
                        step,
                        config.steps,
                        losses[-1],
                        rate,
                        dist_env.world_size,
                    )

                if config.ckpt_every > 0 and step % config.ckpt_every == 0:
                    # Symmetric barrier so no rank races ahead while rank-0
                    # writes; rank-0 ONLY writes (the unwrapped model's weights).
                    _barrier(dist_env)
                    if dist_env.is_main:
                        ckpt_path = save_checkpoint(
                            out_dir,
                            model=core,
                            optimizer=opt,
                            step=step,
                            window=config.window,
                            extra={"eval_skills": eval_skills},
                        )
                        logger.info(
                            "checkpoint saved at step %d -> %s", step, ckpt_path
                        )
                    _barrier(dist_env)

                if (
                    config.eval_every > 0
                    and step % config.eval_every == 0
                    and eval_shot_ids
                ):
                    # rank-0 ONLY evals (a clone on CPU); other ranks wait at the
                    # barrier so the collective ring stays symmetric.
                    _barrier(dist_env)
                    if dist_env.is_main:
                        mean_skill = _run_periodic_eval(
                            core,
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
                    _barrier(dist_env)

        # ── Final checkpoint (always — including on STOP) ───────────────────
        # rank-0 ONLY; barrier so every rank reaches the same point before/after.
        _barrier(dist_env)
        if dist_env.is_main:
            ckpt_path = save_checkpoint(
                out_dir,
                model=core,
                optimizer=opt,
                step=step,
                window=config.window,
                extra={"eval_skills": eval_skills},
            )
            logger.info("final checkpoint at step %d -> %s", step, ckpt_path)
        _barrier(dist_env)

        result = CorpusTrainResult(
            steps_run=step - start_step,
            initial_loss=losses[0] if losses else float("nan"),
            final_loss=losses[-1] if losses else float("nan"),
            losses=losses,
            n_parameters=core.num_parameters(),
            context_length=core.context_length(),
            n_train_shots=len(dataset),
            n_eval_shots=len(eval_shot_ids or []),
            eval_skills=eval_skills,
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutdown_distributed(dist_env)
    return result


def _default_out_dir() -> Path:
    """Resolve the checkpoint out-dir, keyed by SLURM job id when available."""
    run_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("WM_RUN_ID") or "local"
    return DEFAULT_CKPT_ROOT / str(run_id)


class CudaPrefetcher:
    """Double-buffered async host->device prefetcher overlapping copy + compute.

    Wraps a batch iterator: while the model computes on the CURRENT batch (on the
    default stream), the NEXT batch's per-modality tensors (tokens + valid masks)
    are copied to the GPU with ``.to(device, non_blocking=True)`` on a SIDE CUDA
    stream — so the H2D transfer overlaps compute instead of serialising before
    it.  Before a batch is handed out, the default stream
    ``wait_stream(side)``-s so the copy is guaranteed complete, and the consumed
    tensors are ``record_stream``-d so their pinned-memory buffers are not freed
    while the side-stream copy is still in flight.

    The non-tensor batch fields (``context_steps``, ``shot_ids``) are carried
    through unchanged.

    On a CPU device there are no CUDA streams, so this DEGRADES to a plain
    pass-through iterator (``device.type != "cuda"``) — the synthetic CPU tests
    and any CPU run get the identical batches with no CUDA dependency.
    """

    def __init__(self, loader, device: torch.device) -> None:
        self._loader = loader
        self._device = device
        self._enabled = device.type == "cuda" and torch.cuda.is_available()
        self._stream = torch.cuda.Stream() if self._enabled else None

    def __iter__(self):
        if not self._enabled:
            # CPU (or no CUDA): plain pass-through — identical batches, no streams.
            yield from self._loader
            return

        it = iter(self._loader)
        next_batch = self._preload(it)
        while next_batch is not None:
            # ensure the side-stream copy of THIS batch is done before use, then
            # tie the copied tensors to the default stream so their pinned source
            # buffers stay alive until the compute that consumes them is queued.
            torch.cuda.current_stream().wait_stream(self._stream)
            batch = next_batch
            for d in (batch["tokens"], batch["valid"]):
                for t in d.values():
                    t.record_stream(torch.cuda.current_stream())
            # kick off the copy of the FOLLOWING batch while this one computes.
            next_batch = self._preload(it)
            yield batch

    def _preload(self, it) -> dict | None:
        """Copy the next batch to the GPU on the side stream (or None at end)."""
        try:
            batch = next(it)
        except StopIteration:
            return None
        out = dict(batch)
        with torch.cuda.stream(self._stream):
            out["tokens"] = {
                k: v.to(self._device, non_blocking=True)
                for k, v in batch["tokens"].items()
            }
            out["valid"] = {
                k: v.to(self._device, non_blocking=True)
                for k, v in batch["valid"].items()
            }
        return out


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
    eval_list = list(eval_shot_ids)[:n]
    for sid in eval_list:
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
            # Log a WARNING (not info) so a skip is VISIBLE — a skip now means a
            # genuine read failure or an eval shot that shares no scorable
            # modality with the model, not the old silently-padded crash.
            logger.warning("periodic eval: shot %s NOT scored: %r", sid, exc)
            continue
    del cpu_model
    if not skills:
        # No eval shot was scorable — surface loudly.  This usually means the
        # held-out eval shots were drawn from a band that does not carry the
        # model's streams (see the camera-bearing eval-shot selection in
        # ``_cmd_corpus``); the run continues but the metric is absent, not 0.
        logger.warning(
            "periodic eval: NONE of %d eval shots %s could be scored — "
            "the skill metric is ABSENT this round (check eval-shot selection)",
            len(eval_list),
            eval_list,
        )
        return float("nan")
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


def _select_eval_shots(
    shots: Sequence[int],
    modalities: Sequence[ModalitySpec],
    *,
    n_eval: int,
    token_root: Path | None,
) -> tuple[list[int], list[int]]:
    """Split the corpus into (train, eval), biasing eval toward camera-bearing.

    A predict-vs-reality demo must SCORE the camera modalities, so the held-out
    eval shots are chosen to actually CARRY cameras (with a spread across the id
    range), not tail-sliced from a band that happens to be camera-free.  Returns
    ``(train_shots, eval_shots)`` with the eval shots removed from training.

    Falls back to a spread of plain shots when the corpus carries no cameras
    (or none of the declared modalities is a camera).
    """
    from imas_ambix.worldmodel.dataset import _modality_store_present  # noqa: PLC0415

    ids = [int(s) for s in shots]
    if len(ids) <= 1:
        return ids, ids
    n_eval = max(1, min(n_eval, len(ids) // 5 or 1))
    root = Path(token_root) if token_root is not None else None

    camera_mods = [m for m in modalities if m.kind == "camera"]
    cam_bearing: list[int] = []
    if camera_mods:
        cam_bearing = [
            sid
            for sid in ids
            if any(_modality_store_present(sid, m, root) for m in camera_mods)
        ]

    pool = cam_bearing if cam_bearing else ids
    # spread the eval picks across the id range of the pool (evenly-spaced
    # indices over the SORTED pool) so eval is not clustered at one end.
    ordered = sorted(set(pool))
    if n_eval >= len(ordered):
        eval_shots = list(ordered)
    else:
        idx = (
            [round(i * (len(ordered) - 1) / (n_eval - 1)) for i in range(n_eval)]
            if n_eval > 1
            else [len(ordered) // 2]
        )
        eval_shots = [ordered[i] for i in sorted(set(idx))]
    eval_set = set(eval_shots)
    train_shots = [sid for sid in ids if sid not in eval_set]
    if not train_shots:  # tiny corpus — fall back to overlapping eval
        train_shots = ids
    return train_shots, eval_shots


def _corpus_model_kwargs(args) -> dict:  # noqa: ANN001
    """Collect the backbone-size overrides from the corpus CLI args.

    Only knobs the user explicitly set (non-None) are returned, so an unset
    knob falls through to the :class:`WorldModelConfig` field default — i.e.
    omitting every ``--d-model/--n-layers/--n-heads/--dropout`` reproduces the
    prior tiny-default model exactly (back-compat).

    When ``--d-model`` IS given we also derive ``d_ff = 4 * d_model`` (the
    standard transformer feed-forward ratio) unless the caller pins ``--d-ff``,
    so scaling the width up scales the MLP up too rather than leaving it at the
    256 default and starving the bigger model.
    """
    kw: dict[str, object] = {}
    if getattr(args, "d_model", None) is not None:
        kw["d_model"] = int(args.d_model)
    if getattr(args, "n_layers", None) is not None:
        kw["n_layers"] = int(args.n_layers)
    if getattr(args, "n_heads", None) is not None:
        kw["n_heads"] = int(args.n_heads)
    if getattr(args, "dropout", None) is not None:
        kw["dropout"] = float(args.dropout)
    if getattr(args, "d_ff", None) is not None:
        kw["d_ff"] = int(args.d_ff)
    elif "d_model" in kw:
        kw["d_ff"] = 4 * int(kw["d_model"])
    return kw


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

    # Held-out eval shots.  Unless given explicitly, pick shots that actually
    # CARRY cameras (spread across the id range) so the predict-vs-reality demo
    # scores the camera modalities — NOT a tail slice of the (camera-first
    # reordered) corpus, whose tail is the camera-free core-only band.
    if args.eval_shots.strip():
        eval_shots = [int(s) for s in args.eval_shots.split(",") if s.strip()]
        train_shots = shots
    else:
        train_shots, eval_shots = _select_eval_shots(
            shots, modalities, n_eval=args.n_eval_shots, token_root=token_root
        )

    # Backbone-size overrides.  Unset (None) => the WorldModelConfig field
    # default is used, so omitting all four reproduces the prior behaviour
    # (back-compat).  ``d_ff`` is not a separate CLI knob: when the width is
    # scaled up we follow the standard transformer 4x feed-forward ratio so a
    # bigger ``d_model`` actually grows the MLP (otherwise d_ff stays 256 and
    # the model is width-starved); the WorldModelConfig default (256) stands
    # when ``--d-model`` is unset.
    model_kwargs = _corpus_model_kwargs(args)

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
        prefetch_factor=args.prefetch_factor,
        loss_chunk_channels=args.loss_chunk_channels,
        window=WorldModelWindowConfig(
            n_steps=args.n_steps, context_steps=args.context_steps
        ),
        model_kwargs=model_kwargs,
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
        cache_dir=getattr(args, "cache_dir", None),
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
    pc.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="DataLoader batches each worker stages ahead (>=4 floor; "
        "ignored when --num-workers 0)",
    )
    pc.add_argument(
        "--loss-chunk-channels",
        type=int,
        default=16,
        help="channel-chunk size for the chunked next-token NLL (peak head "
        "memory ~chunk*vocab). Larger = faster when memory allows (e.g. the "
        "full-res rbb head on a half-empty card); smaller = leaner.",
    )
    pc.add_argument("--out-dir", default=None, help="checkpoint dir (default GPFS)")
    pc.add_argument("--token-root", default=None)
    pc.add_argument("--no-resume", action="store_true", help="ignore latest.pt")
    pc.add_argument(
        "--cache-dir",
        default=None,
        help="node-local NVMe dir for the assembled-sample cache (opt-in; "
        "e.g. /scratch_local/wm_token_cache). Default: WM_CACHE_DIR env, "
        "else OFF (assemble every access). Caches the fully-assembled per-shot "
        "sample so later epochs skip the GPFS re-read + re-assembly.",
    )
    # Backbone-size knobs.  Default None => the WorldModelConfig field default
    # (the tiny prototype) — omit all four for the prior behaviour; set them to
    # size the transformer UP so the H200s are actually used.
    pc.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="transformer width (default: WorldModelConfig default, tiny)",
    )
    pc.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="transformer depth (default: WorldModelConfig default)",
    )
    pc.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="attention heads (must divide d-model; default WorldModelConfig)",
    )
    pc.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help="feed-forward width (default: 4*d-model when --d-model set, else 256)",
    )
    pc.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="dropout probability (default: WorldModelConfig default, 0.0)",
    )
    pc.add_argument(
        "--modalities",
        default="",
        help="comma-separated subset of default_modalities by name "
        "(e.g. pulse_schedule,summary,pf_active,interferometer,gas_injection,"
        "soft_x_rays,xma,xim,xsx,rbb,rba,rco,rgb,rgc); empty = ALL streams",
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
