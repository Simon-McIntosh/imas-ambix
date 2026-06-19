"""Training + overfit for the signal-conditioned spatiotemporal camera model.

The v2 trainer: same next-frame teacher-forced chunked cross-entropy as v1, but
the model is a :class:`SignalSpacetimeTransformer` that conditions on the MEASURED
diagnostic streams (magnetics / density / soft_x_rays / L2 groups) alongside the
plan.  The DDP wiring, LR schedule (warmup→cosine), grad-clip, drain-safe
checkpoint/resume, and held-out eval are reused from
:mod:`imas_ambix.worldmodel.spacetime_train` — this module changes only:

* the model build (signal-aware config + per-stream channel sizing);
* the collate (stacks the per-shot signal dicts, padding to the batch-max steps
  / channels per stream — absent streams are emitted as zero-step omissions so
  the model conditions only on present streams);
* the per-rank channel probe is broadcast so every DDP rank builds the SAME
  signal-stream set + widths (a transient per-rank read miss can never give two
  ranks different model shapes — that would abort DDP's param check).

GPU-safety (repo AGENTS.md §2b) is inherited: model loaded once, SIGTERM-clean
STOP flag, try/finally release + empty_cache, cudnn deterministic, bf16 autocast.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    discover_camera_shots,
    plan_vocab,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    SignalModalitySpec,
    SignalSpacetimeDataset,
    SignalSpacetimeSample,
    assemble_signal_window,
    default_signal_modalities,
    probe_signal_channels,
    stream_specs_from_modalities,
)
from imas_ambix.worldmodel.spacetime_model_v2 import (
    SignalSpacetimeConfig,
    SignalSpacetimeTransformer,
    SignalStreamSpec,
)

# Reuse the v1 helpers verbatim (NOT modified) — distributed env, LR schedule,
# autocast, stop flag, checkpoint plumbing.
from imas_ambix.worldmodel.spacetime_train import (
    DEFAULT_CKPT_ROOT,
    DistEnv,
    OverfitResult,
    _AutocastCtx,
    _barrier,
    _broadcast_int,
    _init_distributed,
    _set_determinism,
    _shutdown_distributed,
    _StopFlag,
    _unwrap,
    build_lr_scheduler,
    find_latest_checkpoint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model build (signal-aware)
# ---------------------------------------------------------------------------


def build_signal_model(
    window: SpacetimeWindowConfig,
    *,
    plan_channels: int,
    signal_streams: Sequence[SignalStreamSpec],
    n_signal_steps: int,
    max_frames: int | None = None,
    **model_kwargs: object,
) -> SignalSpacetimeTransformer:
    """Build a :class:`SignalSpacetimeTransformer` sized to the window + streams.

    ``max_frames`` must cover plan steps + every present stream's signal steps +
    the camera frames; it defaults to ``n_plan + len(streams)*n_signal_steps +
    n_frames`` with a little slack.
    """
    n_streams = len(signal_streams)
    if max_frames is None:
        max_frames = (
            window.n_plan + n_streams * int(n_signal_steps) + window.n_frames + 2
        )
    cfg = SignalSpacetimeConfig(
        max_frames=int(max_frames),
        plan_vocab=plan_vocab(),
        plan_channels=int(plan_channels),
        signal_streams=tuple(signal_streams),
        n_signal_steps=int(n_signal_steps),
        **model_kwargs,  # type: ignore[arg-type]
    )
    return SignalSpacetimeTransformer(cfg)


def _plan_channels_for(samples: Sequence[SignalSpacetimeSample]) -> int:
    widths = [int(s.plan.shape[1]) for s in samples if s.plan.ndim == 2 and s.plan.size]
    return max(widths) if widths else 0


# ---------------------------------------------------------------------------
# Collate (stack windows + per-stream signals)
# ---------------------------------------------------------------------------


def collate_signal_windows(
    samples: Sequence[SignalSpacetimeSample],
    *,
    stream_names: Sequence[str] | None = None,
) -> dict:
    """Stack frames + plan + per-stream signals into a model batch.

    Frames stack directly (fixed window).  The plan is padded to the batch-max
    (P, C) exactly as in v1.  Each signal stream is stacked across the samples
    that carry it, padded to the batch-max (steps, channels) for that stream with
    PAD id 0; a sample missing a stream contributes an all-PAD block so the batch
    tensor is rectangular.  A stream NO sample in the batch carries is omitted
    from ``signals`` entirely.

    ``stream_names`` (optional) pins the stream set + order to the model's so the
    batch always presents the full stream set the model expects (an all-PAD block
    for a stream absent from this batch keeps the model's per-stream params in the
    graph, DDP-uniform).  When ``None`` only streams present in the batch appear.
    """
    frames = torch.stack(
        [torch.as_tensor(s.frames, dtype=torch.long) for s in samples]
    )  # (B, T, S)

    # plan — pad to batch-max (P, C), PAD 0 (v1 behaviour).
    plans = [np.asarray(s.plan, dtype=np.int64) for s in samples]
    max_p = max((pl.shape[0] for pl in plans if pl.ndim == 2 and pl.size), default=0)
    max_c = max((pl.shape[1] for pl in plans if pl.ndim == 2 and pl.size), default=0)
    if max_p == 0 or max_c == 0:
        plan_t = torch.zeros((len(samples), 0, 0), dtype=torch.long)
    else:
        out = np.zeros((len(samples), max_p, max_c), dtype=np.int64)
        for i, pl in enumerate(plans):
            if pl.ndim == 2 and pl.size:
                out[i, : pl.shape[0], : pl.shape[1]] = pl
        plan_t = torch.as_tensor(out, dtype=torch.long)

    # signals — per stream, stack to batch-max (steps, channels), PAD 0.
    if stream_names is None:
        names: list[str] = []
        for s in samples:
            for k in s.signals:
                if k not in names:
                    names.append(k)
    else:
        names = list(stream_names)

    signals: dict[str, torch.Tensor] = {}
    for name in names:
        present = [s.signals.get(name) for s in samples]
        widths = [a.shape for a in present if a is not None and a.size]
        if not widths:
            # no sample in this batch carries the stream — skip (the model's
            # zero-touch keeps its params in the graph anyway).
            continue
        max_s = max(w[0] for w in widths)
        max_ch = max(w[1] for w in widths)
        block = np.zeros((len(samples), max_s, max_ch), dtype=np.int64)
        for i, a in enumerate(present):
            if a is not None and a.size:
                block[i, : a.shape[0], : a.shape[1]] = a
        signals[name] = torch.as_tensor(block, dtype=torch.long)

    return {
        "frames": frames,
        "plan": plan_t,
        "signals": signals,
        "context_frames": int(samples[0].context_frames),
        "shot_ids": [int(s.shot_id) for s in samples],
    }


def _batch_to(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["frames"] = batch["frames"].to(device, non_blocking=True)
    out["plan"] = batch["plan"].to(device, non_blocking=True)
    out["signals"] = {
        k: v.to(device, non_blocking=True) for k, v in batch["signals"].items()
    }
    return out


# ---------------------------------------------------------------------------
# Overfit (the GATE) — proves it LEARNS + signals are LOAD-BEARING
# ---------------------------------------------------------------------------


@dataclass
class OverfitV2Config:
    steps: int = 400
    lr: float = 3e-4
    seed: int = 0
    log_every: int = 25
    chunk: int = 4096
    n_signal_steps: int = 4
    window: SpacetimeWindowConfig = field(default_factory=SpacetimeWindowConfig)
    model_kwargs: dict = field(default_factory=dict)
    modalities: list[SignalModalitySpec] = field(
        default_factory=default_signal_modalities
    )


def overfit_signal(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    config: OverfitV2Config | None = None,
    token_root: Path | None = None,
    device: str | None = None,
) -> tuple[OverfitResult, SignalSpacetimeTransformer, list[SignalSpacetimeSample]]:
    """Overfit a handful of shots with signal conditioning — the GATE.

    Returns ``(result, model, samples)``.  The caller then runs the
    signal-ablation check (a full-signals forward vs a signals-zeroed forward
    must differ) to PROVE the conditioning is load-bearing, not silently dropped.
    """
    config = config or OverfitV2Config()
    _set_determinism(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stop = _StopFlag()
    stop.install()

    samples = [
        assemble_signal_window(
            int(sid),
            config.window,
            config.modalities,
            config.n_signal_steps,
            camera=camera,
            token_root=token_root,
        )
        for sid in shot_ids
    ]
    plan_ch = _plan_channels_for(samples)
    # channel widths seen across the overfit samples (no probe shots needed).
    channels: dict[str, int] = {}
    for s in samples:
        for name, arr in s.signals.items():
            channels[name] = max(channels.get(name, 0), int(arr.shape[1]))
    streams = stream_specs_from_modalities(config.modalities, channels)
    stream_names = [st.name for st in streams]
    batch = _batch_to(collate_signal_windows(samples, stream_names=stream_names), dev)

    model = build_signal_model(
        config.window,
        plan_channels=plan_ch,
        signal_streams=streams,
        n_signal_steps=config.n_signal_steps,
        **config.model_kwargs,
    ).to(dev)
    model.train()
    logger.info(
        "overfit-v2 on %s: params=%d (%.1fM) n_frames=%d plan_ch=%d "
        "streams=%s shots=%s",
        dev,
        model.num_parameters(),
        model.num_parameters() / 1e6,
        config.window.n_frames,
        plan_ch,
        [(st.name, st.channels) for st in streams],
        list(shot_ids),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    losses: list[float] = []
    try:
        for step in range(config.steps):
            if stop.stop:
                logger.warning("STOP — ending overfit at step %d", step)
                break
            opt.zero_grad(set_to_none=True)
            with _AutocastCtx(dev):
                loss = model(batch, loss_spec={"chunk": config.chunk})
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if step % config.log_every == 0 or step == config.steps - 1:
                logger.info(
                    "overfit-v2 step %d/%d loss=%.4f", step, config.steps, losses[-1]
                )
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    result = OverfitResult(
        initial_loss=losses[0] if losses else float("nan"),
        final_loss=losses[-1] if losses else float("nan"),
        losses=losses,
        n_parameters=model.num_parameters(),
        shot_ids=[int(s) for s in shot_ids],
    )
    return result, model, samples


@torch.no_grad()
def signal_ablation_delta(
    model: SignalSpacetimeTransformer,
    batch: dict,
    *,
    chunk: int = 4096,
) -> tuple[float, float]:
    """Loss WITH signals vs loss with signals ZEROED — proves load-bearing.

    Returns ``(loss_full, loss_zeroed)``.  ``loss_zeroed`` replaces every signal
    block with PAD id 0 (a real, in-vocab token id, so it does not crash) — if
    the conditioning is genuinely feeding the prediction the two losses DIFFER.  A
    NEGLIGIBLE delta means the signals are silently dropped (a wiring bug).
    """
    model.eval()
    with _AutocastCtx(batch["frames"].device):
        full = float(model(batch, loss_spec={"chunk": chunk}).detach())
    zeroed = dict(batch)
    zeroed["signals"] = {k: torch.zeros_like(v) for k, v in batch["signals"].items()}
    with _AutocastCtx(batch["frames"].device):
        zero = float(model(zeroed, loss_spec={"chunk": chunk}).detach())
    model.train()
    return full, zero


# ---------------------------------------------------------------------------
# Corpus trainer (DDP-aware; single-proc default unchanged)
# ---------------------------------------------------------------------------


@dataclass
class CorpusV2Config:
    steps: int = 30000
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.1
    seed: int = 0
    log_every: int = 25
    ckpt_every: int = 500
    eval_every: int = 1000
    num_workers: int = 4
    prefetch_factor: int = 4
    grad_clip: float = 1.0
    chunk: int = 4096
    n_eval_shots: int = 2
    n_signal_steps: int = 4
    lr_schedule: bool = True
    warmup_steps: int = 800
    min_lr_ratio: float = 0.01
    window: SpacetimeWindowConfig = field(default_factory=SpacetimeWindowConfig)
    model_kwargs: dict = field(default_factory=dict)
    random_window: bool = True
    modalities: list[SignalModalitySpec] = field(
        default_factory=default_signal_modalities
    )


@dataclass
class CorpusResult:
    steps_run: int
    initial_loss: float
    final_loss: float
    losses: list[float]
    n_parameters: int
    n_train_shots: int
    eval_errors: list[tuple[int, float]]
    checkpoint_path: str | None
    best_eval: float | None = None
    stream_names: list[str] = field(default_factory=list)


def _default_out_dir() -> Path:
    run_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("WM_RUN_ID") or "local"
    return DEFAULT_CKPT_ROOT / f"spacetime-v2-{run_id}"


def _config_to_dict(cfg: SignalSpacetimeConfig) -> dict:
    return {
        "vocab_size": int(cfg.vocab_size),
        "grid_h": int(cfg.grid_h),
        "grid_w": int(cfg.grid_w),
        "max_frames": int(cfg.max_frames),
        "plan_vocab": int(cfg.plan_vocab),
        "plan_channels": int(cfg.plan_channels),
        "d_model": int(cfg.d_model),
        "n_layers": int(cfg.n_layers),
        "n_heads": int(cfg.n_heads),
        "d_ff": int(cfg.d_ff),
        "dropout": float(cfg.dropout),
        "n_signal_steps": int(cfg.n_signal_steps),
        "signal_streams": [
            {"name": st.name, "vocab": int(st.vocab), "channels": int(st.channels)}
            for st in cfg.signal_streams
        ],
    }


def save_checkpoint_v2(
    out_dir: Path,
    *,
    model: SignalSpacetimeTransformer,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    window: SpacetimeWindowConfig,
    extra: dict | None = None,
    name: str = "latest.pt",
    snapshot: bool = True,
) -> Path:
    """Atomic self-describing v2 checkpoint (records the signal-stream set)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "model_config": _config_to_dict(model.config),
        "window": {
            "n_frames": int(window.n_frames),
            "n_plan": int(window.n_plan),
            "context_frames": int(window.context_frames),
            "frame_stride": int(window.frame_stride),
        },
        "extra": dict(extra or {}),
    }
    final = out_dir / name
    tmp = out_dir / f".{name}.{os.getpid()}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    if snapshot:
        try:
            torch.save(payload, out_dir / f"ckpt-{int(step):08d}.pt")
        except OSError as exc:  # noqa: BLE001
            logger.warning("could not write step snapshot: %r", exc)
    return final


def _config_from_dict_v2(d: dict) -> SignalSpacetimeConfig:
    """Rebuild a :class:`SignalSpacetimeConfig` from a saved checkpoint config.

    Reconstructs the per-stream :class:`SignalStreamSpec` list (each stream's own
    local vocab + channel width) so the rebuilt model has the IDENTICAL parameter
    set as the trained one — the signal embedding tables can then be loaded by
    name without an unexpected/missing-key error (the failure the v1 loader hit).
    """
    streams = tuple(
        SignalStreamSpec(
            name=str(s["name"]), vocab=int(s["vocab"]), channels=int(s["channels"])
        )
        for s in d.get("signal_streams", [])
    )
    scalar_fields = {
        k: d[k]
        for k in d
        if k in SignalSpacetimeConfig.__dataclass_fields__
        and k not in ("signal_streams",)
    }
    return SignalSpacetimeConfig(signal_streams=streams, **scalar_fields)


def load_signal_model_from_checkpoint(
    path: Path, *, map_location: str = "cpu"
) -> tuple[SignalSpacetimeTransformer, dict]:
    """Rebuild a v2 model from a checkpoint + load its weights (eval entry).

    The v2 counterpart of
    :func:`imas_ambix.worldmodel.spacetime_train.load_model_from_checkpoint`,
    which CANNOT load a v2 checkpoint (it builds a plain
    :class:`~imas_ambix.worldmodel.spacetime_model.SpacetimeTransformer` and so
    rejects the ``signal_*`` state-dict keys).  This rebuilds the
    :class:`SignalSpacetimeTransformer` from the recorded signal-stream set first,
    so every signal embedding loads by name.
    """
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = _config_from_dict_v2(payload["model_config"])
    model = SignalSpacetimeTransformer(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    return model, payload


# ---------------------------------------------------------------------------
# Signal-aware prediction (teacher-forced + autoregressive) — for the decode gate
# ---------------------------------------------------------------------------


def _sample_stream_names(sample: SignalSpacetimeSample) -> list[str]:
    return list(sample.signals.keys())


def _decode_frame(
    model: SignalSpacetimeTransformer,
    hidden_prev: torch.Tensor,
    *,
    chunk: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Pick one frame's tokens — argmax when ``temperature<=0`` else top-p sample.

    A single entry point so the teacher-forced and autoregressive paths share the
    SAME token-selection rule: ``temperature <= 0`` (the default) reproduces the
    deterministic mode-seeking argmax baseline; a positive temperature draws a
    nucleus-truncated sample (the mode-escape that M1 tests).
    """
    if temperature is None or temperature <= 0.0:
        return model.chunked_argmax_frame(hidden_prev, chunk=chunk)
    return model.chunked_sample_frame(
        hidden_prev,
        temperature=temperature,
        top_p=top_p,
        chunk=chunk,
        generator=generator,
    )


@torch.no_grad()
def teacher_forced_signal_frames(
    model: SignalSpacetimeTransformer,
    sample: SignalSpacetimeSample,
    *,
    stream_names: Sequence[str] | None = None,
    chunk: int = 4096,
    device: torch.device | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Teacher-forced next-frame prediction WITH measured-signal conditioning.

    The signal counterpart of
    :func:`imas_ambix.worldmodel.spacetime_train.teacher_forced_frames`: it feeds
    the TRUE frames plus the plan AND the measured signals, then reads the
    per-token prediction (frame ``t`` predicted from the hidden at ``t-1``).
    Returns ``(T, S)`` LOCAL token ids; frame 0 is truth (no predecessor).

    ``stream_names`` pins the stream set/order to the model's (so an all-PAD block
    is presented for any stream this sample lacks, keeping the embedding tables in
    the graph); defaults to the streams the sample carries.  ``temperature`` /
    ``top_p`` select the token rule: ``temperature <= 0`` (default) is the greedy
    argmax baseline, a positive temperature is a nucleus sample.
    """
    model.eval()
    dev = device or next(model.parameters()).device
    names = (
        list(stream_names) if stream_names is not None else _sample_stream_names(sample)
    )
    batch = _batch_to(collate_signal_windows([sample], stream_names=names), dev)
    hidden = model._forward_tokens(
        batch["frames"], batch.get("plan"), batch.get("signals")
    )  # (1, T, S, d)
    t = hidden.shape[1]
    s = hidden.shape[2]
    out = np.zeros((t, s), dtype=np.int64)
    out[0] = np.asarray(sample.frames[0], dtype=np.int64)
    for ti in range(1, t):
        pred = _decode_frame(
            model,
            hidden[:, ti - 1],
            chunk=chunk,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )  # (1, S)
        out[ti] = pred[0].cpu().numpy().astype(np.int64)
    return out


@torch.no_grad()
def autoregressive_signal_dream(
    model: SignalSpacetimeTransformer,
    sample: SignalSpacetimeSample,
    *,
    stream_names: Sequence[str] | None = None,
    chunk: int = 4096,
    device: torch.device | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Autoregressive rollout FEEDING THE SIGNALS — the v2 forecast.

    The signal counterpart of
    :func:`imas_ambix.worldmodel.spacetime_train.autoregressive_dream`: the model
    keeps the leading ``context_frames`` TRUE frames, then rolls forward consuming
    its OWN predicted frames while conditioning on the plan AND the measured
    signals at every step.  The signals are FIXED context for the whole window
    (sub-sampled across the window span at assembly), so they are collated once and
    re-fed each step — the rollout always sees the measured plasma state, which is
    the whole point of v2.  Returns ``(T, S)`` LOCAL token ids (context = truth).

    ``temperature`` / ``top_p`` select the token rule per generated frame:
    ``temperature <= 0`` (default) reproduces the deterministic argmax baseline;
    a positive temperature draws a nucleus-truncated sample, so that repeated
    calls (different ``generator`` state) yield an ENSEMBLE of distinct coherent
    rollouts — the object a distributional metric scores against persistence.
    """
    model.eval()
    dev = device or next(model.parameters()).device
    ctx = int(sample.context_frames)
    t_total = int(sample.frames.shape[0])

    names = (
        list(stream_names) if stream_names is not None else _sample_stream_names(sample)
    )
    batch = _batch_to(collate_signal_windows([sample], stream_names=names), dev)
    plan = batch.get("plan")
    signals = batch.get("signals")

    gen = np.asarray(sample.frames, dtype=np.int64).copy()  # seed with truth
    for ti in range(ctx, t_total):
        cur = torch.as_tensor(gen[:ti][None], dtype=torch.long, device=dev)  # (1,ti,S)
        hidden = model._forward_tokens(cur, plan, signals)  # (1, ti, S, d)
        pred = _decode_frame(
            model,
            hidden[:, ti - 1],
            chunk=chunk,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )  # (1, S)
        gen[ti] = pred[0].cpu().numpy().astype(np.int64)
    return gen


@torch.no_grad()
def _eval_reconstruction(
    model: SignalSpacetimeTransformer,
    eval_shots: Sequence[int],
    window: SpacetimeWindowConfig,
    modalities: Sequence[SignalModalitySpec],
    n_signal_steps: int,
    stream_names: Sequence[str],
    *,
    camera: str,
    token_root: Path | None,
    n: int,
    chunk: int,
    device: torch.device,
) -> float:
    """Mean teacher-forced next-frame token-mismatch on held-out shots (0..1)."""
    model.eval()
    rates: list[float] = []
    for sid in list(eval_shots)[:n]:
        try:
            sample = assemble_signal_window(
                int(sid),
                window,
                modalities,
                n_signal_steps,
                camera=camera,
                token_root=token_root,
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("eval shot %s NOT assemblable: %r", sid, exc)
            continue
        batch = _batch_to(
            collate_signal_windows([sample], stream_names=stream_names), device
        )
        hidden = model._forward_tokens(
            batch["frames"], batch.get("plan"), batch.get("signals")
        )
        t = hidden.shape[1]
        truth = np.asarray(sample.frames, dtype=np.int64)
        ctx = int(sample.context_frames)
        pred = np.zeros_like(truth)
        pred[0] = truth[0]
        for ti in range(1, t):
            pred[ti] = (
                model.chunked_argmax_frame(hidden[:, ti - 1], chunk=chunk)[0]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
        rates.append(float((pred[ctx:] != truth[ctx:]).mean()))
    model.train()
    return float(np.mean(rates)) if rates else float("nan")


def train_corpus(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    config: CorpusV2Config | None = None,
    out_dir: Path | None = None,
    token_root: Path | None = None,
    eval_shot_ids: Sequence[int] | None = None,
    device: str | None = None,
    resume: bool = True,
) -> CorpusResult:
    """Train the signal-conditioned camera transformer on a CORPUS of shots.

    DDP-aware: each rank pins its card, trains a disjoint shot shard, and the
    chunked next-frame NLL is computed inside forward so DDP all-reduces every
    grad.  The plan-channel count AND the signal-stream set + widths are decided
    on rank 0 and broadcast so every rank builds the IDENTICAL model (a per-rank
    probe miss can never give two ranks different shapes).
    """
    config = config or CorpusV2Config()
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir()
    _set_determinism(config.seed)

    env = DistEnv.from_environment()
    _init_distributed(env)
    if device is None:
        device = f"cuda:{env.local_rank}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    stop = _StopFlag()
    stop.install()

    # Probe plan-channels + signal-stream widths on a few shots.
    probe: list[SignalSpacetimeSample] = []
    for sid in list(shot_ids)[:8]:
        try:
            probe.append(
                assemble_signal_window(
                    int(sid),
                    config.window,
                    config.modalities,
                    config.n_signal_steps,
                    camera=camera,
                    token_root=token_root,
                )
            )
        except (ValueError, FileNotFoundError, KeyError):
            continue
    if not probe:
        raise ValueError("no shot assembled in the probe — cannot size the model")
    plan_ch = _plan_channels_for(probe)
    channels = probe_signal_channels(
        list(shot_ids)[:16],
        config.window,
        config.modalities,
        config.n_signal_steps,
        camera=camera,
        token_root=token_root,
    )

    # Broadcast every shaping quantity from rank 0 so all ranks build the SAME
    # model (a per-rank GPFS read miss must not desync the param set -> DDP abort).
    plan_ch = _broadcast_int(env, plan_ch)
    for m in config.modalities:
        channels[m.name] = _broadcast_int(env, int(channels.get(m.name, 0)))
    streams = stream_specs_from_modalities(config.modalities, channels)
    stream_names = [st.name for st in streams]

    base_model = build_signal_model(
        config.window,
        plan_channels=plan_ch,
        signal_streams=streams,
        n_signal_steps=config.n_signal_steps,
        **config.model_kwargs,
    ).to(dev)
    if env.is_main:
        logger.info(
            "model-v2 on %s: params=%d (%.1fM) d_model=%d n_layers=%d n_heads=%d "
            "n_frames=%d plan_ch=%d n_signal_steps=%d streams=%s world=%d",
            dev,
            base_model.num_parameters(),
            base_model.num_parameters() / 1e6,
            base_model.config.d_model,
            base_model.config.n_layers,
            base_model.config.n_heads,
            config.window.n_frames,
            plan_ch,
            config.n_signal_steps,
            [(st.name, st.channels) for st in streams],
            env.world_size,
        )

    opt = torch.optim.AdamW(
        base_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    start_step = 0
    eval_errors: list[tuple[int, float]] = []
    best_eval = float("inf")
    if resume:
        latest = find_latest_checkpoint(out_dir)
        if latest is not None:
            payload = torch.load(str(latest), map_location="cpu", weights_only=False)
            try:
                base_model.load_state_dict(payload["model_state_dict"])
                if payload.get("optimizer_state_dict"):
                    opt.load_state_dict(payload["optimizer_state_dict"])
                start_step = int(payload.get("step", 0))
                extra = payload.get("extra", {})
                eval_errors = list(extra.get("eval_errors", []))
                best_eval = float(extra.get("best_eval", float("inf")))
                if env.is_main:
                    logger.info("RESUMED from %s at step %d", latest, start_step)
            except (KeyError, RuntimeError, ValueError) as exc:
                if env.is_main:
                    logger.warning(
                        "checkpoint %s incompatible (%r) — fresh", latest, exc
                    )

    scheduler = build_lr_scheduler(
        opt,
        total_steps=config.steps,
        warmup_steps=config.warmup_steps,
        min_lr_ratio=config.min_lr_ratio,
        scheduled=config.lr_schedule,
        last_step=start_step - 1,
        peak_lr=config.lr,
    )
    if env.is_main:
        logger.info(
            "LR schedule: %s peak=%.2e warmup=%d cosine→%.2e over %d steps "
            "(grad_clip=%.2f weight_decay=%.3f dropout=%s)",
            "warmup+cosine" if config.lr_schedule else "FLAT (fallback)",
            config.lr,
            config.warmup_steps,
            config.lr * config.min_lr_ratio,
            config.steps,
            config.grad_clip,
            config.weight_decay,
            base_model.config.dropout,
        )

    if env.enabled:
        from torch.nn.parallel import DistributedDataParallel

        ddp_kwargs: dict = {}
        if dev.type == "cuda":
            ddp_kwargs["device_ids"] = [env.local_rank]
            ddp_kwargs["output_device"] = env.local_rank
        # Every plan + signal param is touched each step (the zero-touch guards),
        # so param usage is uniform across ranks; find_unused_parameters stays True
        # as a safe guard while multi-rank is shaken out.
        ddp_kwargs["find_unused_parameters"] = True
        model: nn.Module = DistributedDataParallel(base_model, **ddp_kwargs)
    else:
        model = base_model
    core = _unwrap(model)

    dataset = SignalSpacetimeDataset(
        shot_ids,
        config.window,
        config.modalities,
        config.n_signal_steps,
        camera=camera,
        token_root=token_root,
        random_window=config.random_window,
        seed=config.seed,
    )
    sampler = None
    if env.enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=env.world_size,
            rank=env.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=True,
        )
    use_workers = config.num_workers > 0

    def _collate(samples):  # noqa: ANN001 — pins the full model stream set/order
        return collate_signal_windows(samples, stream_names=stream_names)

    loader_kwargs: dict = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=_collate,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=(dev.type == "cuda"),
        multiprocessing_context="fork" if use_workers else None,
    )
    if use_workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(4, config.prefetch_factor)
    loader = torch.utils.data.DataLoader(**loader_kwargs)

    model.train()
    losses: list[float] = []
    step = start_step
    ckpt_path: Path | None = find_latest_checkpoint(out_dir) if resume else None
    t_last = time.time()
    epoch = 0
    try:
        done = False
        while not done:
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch += 1
            for batch in loader:
                if stop.stop or step >= config.steps:
                    done = True
                    break
                batch = _batch_to(batch, dev)
                opt.zero_grad(set_to_none=True)
                with _AutocastCtx(dev):
                    loss = model(
                        batch,
                        loss_spec={
                            "chunk": config.chunk,
                            "context_frames": config.window.context_frames,
                        },
                    )
                loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                step += 1

                if env.is_main and (
                    step % config.log_every == 0 or step == config.steps
                ):
                    rate = config.log_every / max(time.time() - t_last, 1e-6)
                    t_last = time.time()
                    cur_lr = opt.param_groups[0]["lr"]
                    logger.info(
                        "corpus-v2 step %d/%d loss=%.4f lr=%.3e (%.2f st/s world=%d)",
                        step,
                        config.steps,
                        losses[-1],
                        cur_lr,
                        rate,
                        env.world_size,
                    )

                if config.ckpt_every > 0 and step % config.ckpt_every == 0:
                    _barrier(env)
                    if env.is_main:
                        ckpt_path = save_checkpoint_v2(
                            out_dir,
                            model=core,
                            optimizer=opt,
                            step=step,
                            window=config.window,
                            extra={
                                "eval_errors": eval_errors,
                                "best_eval": best_eval,
                                "stream_names": stream_names,
                            },
                        )
                        logger.info("checkpoint @ step %d -> %s", step, ckpt_path)
                    _barrier(env)

                if (
                    config.eval_every > 0
                    and step % config.eval_every == 0
                    and eval_shot_ids
                ):
                    _barrier(env)
                    if env.is_main:
                        err = _eval_reconstruction(
                            core,
                            eval_shot_ids,
                            config.window,
                            config.modalities,
                            config.n_signal_steps,
                            stream_names,
                            camera=camera,
                            token_root=token_root,
                            n=config.n_eval_shots,
                            chunk=config.chunk,
                            device=dev,
                        )
                        eval_errors.append((step, err))
                        logger.info(
                            "EVAL @ step %d: mean TF next-frame token-mismatch = %.4f",
                            step,
                            err,
                        )
                        if np.isfinite(err) and err < best_eval:
                            best_eval = float(err)
                            best_path = save_checkpoint_v2(
                                out_dir,
                                model=core,
                                optimizer=opt,
                                step=step,
                                window=config.window,
                                extra={
                                    "eval_errors": eval_errors,
                                    "best_eval": best_eval,
                                    "is_best": True,
                                    "stream_names": stream_names,
                                },
                                name="best.pt",
                                snapshot=False,
                            )
                            logger.info(
                                "BEST eval %.4f @ step %d -> %s",
                                best_eval,
                                step,
                                best_path,
                            )
                    model.train()
                    _barrier(env)

        _barrier(env)
        if env.is_main:
            ckpt_path = save_checkpoint_v2(
                out_dir,
                model=core,
                optimizer=opt,
                step=step,
                window=config.window,
                extra={
                    "eval_errors": eval_errors,
                    "best_eval": best_eval,
                    "stream_names": stream_names,
                },
            )
            logger.info("final checkpoint @ step %d -> %s", step, ckpt_path)
        _barrier(env)

        result = CorpusResult(
            steps_run=step - start_step,
            initial_loss=losses[0] if losses else float("nan"),
            final_loss=losses[-1] if losses else float("nan"),
            losses=losses,
            n_parameters=core.num_parameters(),
            n_train_shots=len(dataset),
            eval_errors=eval_errors,
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
            best_eval=best_eval if np.isfinite(best_eval) else None,
            stream_names=stream_names,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _shutdown_distributed(env)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_corpus(args) -> int:  # noqa: ANN001
    token_root = Path(args.token_root) if args.token_root else None
    window = SpacetimeWindowConfig(
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
    )
    span = (window.n_frames - 1) * window.frame_stride + 1

    explicit_eval = [int(s) for s in args.eval_shots.split(",") if s.strip()]
    if args.shots.strip():
        pool = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        limit = args.n_shots + len(explicit_eval) if explicit_eval else args.n_shots
        pool = discover_camera_shots(
            camera=args.camera, token_root=token_root, min_frames=span, limit=limit
        )
    if len(pool) < 2:
        logger.error("need >= 2 shots; discovered %d", len(pool))
        return 1

    if explicit_eval:
        eval_shots = explicit_eval
        train_shots = [s for s in pool if s not in set(explicit_eval)]
        if not args.shots.strip():
            train_shots = train_shots[: args.n_shots]
    else:
        n_eval = max(1, min(args.n_eval_shots, len(pool) // 5 or 1))
        eval_shots = pool[-n_eval:]
        train_shots = pool[:-n_eval] or pool

    n_eval = len(eval_shots)
    overlap = set(train_shots) & set(eval_shots)
    if overlap:
        logger.error("train/eval overlap (NOT disjoint): %s", sorted(overlap))
        return 1
    logger.info(
        "held-out eval shots %s are DISJOINT from %d train shots (verified)",
        eval_shots,
        len(train_shots),
    )

    model_kwargs: dict = {}
    for k in ("d_model", "n_layers", "n_heads", "d_ff", "dropout"):
        v = getattr(args, k, None)
        if v is not None:
            model_kwargs[k] = v

    cfg = CorpusV2Config(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=not args.flat_lr,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        chunk=args.chunk,
        n_eval_shots=n_eval,
        n_signal_steps=args.n_signal_steps,
        window=window,
        model_kwargs=model_kwargs,
    )
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    logger.info(
        "spacetime-v2 corpus: %d train / %d eval shots (held out %s) "
        "n_signal_steps=%d out=%s",
        len(train_shots),
        len(eval_shots),
        eval_shots,
        args.n_signal_steps,
        out_dir,
    )
    result = train_corpus(
        train_shots,
        camera=args.camera,
        config=cfg,
        out_dir=out_dir,
        token_root=token_root,
        eval_shot_ids=eval_shots,
        resume=not args.no_resume,
    )
    logger.info(
        "CORPUS-V2 DONE: steps=%d params=%d initial=%.4f final=%.4f best_eval=%s "
        "streams=%s ckpt=%s",
        result.steps_run,
        result.n_parameters,
        result.initial_loss,
        result.final_loss,
        result.best_eval,
        result.stream_names,
        result.checkpoint_path,
    )
    if result.eval_errors:
        logger.info("eval mismatch trajectory: %s", result.eval_errors)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command")
    pc = sub.add_parser("corpus", help="train the signal-conditioned camera model")
    pc.add_argument("--shots", default="")
    pc.add_argument("--n-shots", type=int, default=9000)
    pc.add_argument("--camera", default=REFERENCE_CAMERA)
    pc.add_argument("--n-eval-shots", type=int, default=4)
    pc.add_argument("--eval-shots", default="")
    pc.add_argument("--steps", type=int, default=30000)
    pc.add_argument("--batch-size", type=int, default=4)
    pc.add_argument("--lr", type=float, default=3e-4)
    pc.add_argument("--weight-decay", type=float, default=0.1)
    pc.add_argument("--warmup-steps", type=int, default=800)
    pc.add_argument("--min-lr-ratio", type=float, default=0.01)
    pc.add_argument("--grad-clip", type=float, default=1.0)
    pc.add_argument("--flat-lr", action="store_true")
    pc.add_argument("--n-frames", type=int, default=24)
    pc.add_argument("--n-plan", type=int, default=8)
    pc.add_argument("--context-frames", type=int, default=8)
    pc.add_argument("--frame-stride", type=int, default=1)
    pc.add_argument(
        "--n-signal-steps",
        type=int,
        default=4,
        help="conditioning steps each measured stream is sub-sampled to",
    )
    pc.add_argument("--log-every", type=int, default=25)
    pc.add_argument("--ckpt-every", type=int, default=500)
    pc.add_argument("--eval-every", type=int, default=1000)
    pc.add_argument("--num-workers", type=int, default=4)
    pc.add_argument("--chunk", type=int, default=4096)
    pc.add_argument("--d-model", type=int, default=None)
    pc.add_argument("--n-layers", type=int, default=None)
    pc.add_argument("--n-heads", type=int, default=None)
    pc.add_argument("--d-ff", type=int, default=None)
    pc.add_argument("--dropout", type=float, default=None)
    pc.add_argument("--out-dir", default=None)
    pc.add_argument("--token-root", default=None)
    pc.add_argument("--no-resume", action="store_true")
    pc.set_defaults(func=_cmd_corpus)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if getattr(args, "func", None) is None:
        p.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
