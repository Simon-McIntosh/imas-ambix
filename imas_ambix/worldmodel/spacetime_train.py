"""Training + overfit for the spatiotemporal camera transformer.

Next-frame teacher-forced cross-entropy over the frozen rbb LFQ tokens.  Two
entrypoints:

* :func:`overfit` — the GATE.  A small model on 2-4 real rbb shots descended to
  a very low loss; the caller then decodes the teacher-forced reconstruction to
  confirm spatial coherence BEFORE any corpus run.
* :func:`train_corpus` — the real run.  A torch ``DataLoader`` over many shots,
  AdamW, DDP-aware (single-proc default unchanged), drain-safe checkpointing +
  resume, periodic held-out reconstruction-error logging.

GPU-safety (repo AGENTS.md §2b)
-------------------------------
* model loaded ONCE outside the per-step loop;
* SIGTERM/SIGINT sets a STOP flag → clean exit within ``UnkillableStepTimeout``;
* per-step watchdog is implicit (the loop checks STOP every step);
* ``try/finally`` releases the model + ``torch.cuda.empty_cache()``;
* cudnn deterministic, ``set_float32_matmul_precision("high")``, bf16 autocast
  on CUDA (H200 tensor cores).
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeFrameDataset,
    SpacetimeSample,
    SpacetimeWindowConfig,
    assemble_window,
    discover_camera_shots,
    plan_vocab,
)
from imas_ambix.worldmodel.spacetime_model import (
    SpacetimeConfig,
    SpacetimeTransformer,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_CKPT_ROOT = Path("/work/projects/imas_gpu/worldmodel/ckpt")


# ---------------------------------------------------------------------------
# Stop flag (clean cancel)
# ---------------------------------------------------------------------------


class _StopFlag:
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
                logger.debug("could not install handler for %s", sig)


# ---------------------------------------------------------------------------
# Distributed env (mirrors worldmodel.train — single-proc default unchanged)
# ---------------------------------------------------------------------------


@dataclass
class DistEnv:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls) -> DistEnv:
        ws = int(os.environ.get("WORLD_SIZE", "1"))
        if ws <= 1:
            return cls()
        return cls(
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=ws,
        )


def _init_distributed(env: DistEnv) -> None:
    if not env.enabled:
        return
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(env.local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    logger.info(
        "DDP init: rank %d/%d local_rank %d backend %s",
        env.rank,
        env.world_size,
        env.local_rank,
        backend,
    )


def _shutdown_distributed(env: DistEnv) -> None:
    if not env.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DDP final barrier note: %r", exc)
        dist.destroy_process_group()


def _barrier(env: DistEnv) -> None:
    if not env.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def _unwrap(model: nn.Module) -> SpacetimeTransformer:
    return getattr(model, "module", model)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Collate (stack windows into a batch)
# ---------------------------------------------------------------------------


def collate_windows(samples: Sequence[SpacetimeSample]) -> dict:
    """Stack camera-frame windows + plan prefixes into a model batch.

    All windows share ``n_frames`` (the dataset draws fixed-length windows) so
    frames stack directly.  Plans may differ in step/channel count across shots
    (a missing plan is empty); they are padded to the batch-max (P, C) with
    PAD id 0 — plan padding only adds zero-signal conditioning lanes, it is
    never a loss target.
    """
    frames = torch.stack(
        [torch.as_tensor(s.frames, dtype=torch.long) for s in samples]
    )  # (B, T, S)
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
    return {
        "frames": frames,
        "plan": plan_t,
        "context_frames": int(samples[0].context_frames),
        "shot_ids": [int(s.shot_id) for s in samples],
    }


def _batch_to(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["frames"] = batch["frames"].to(device, non_blocking=True)
    out["plan"] = batch["plan"].to(device, non_blocking=True)
    return out


# ---------------------------------------------------------------------------
# Determinism + model build
# ---------------------------------------------------------------------------


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def build_model(
    window: SpacetimeWindowConfig,
    *,
    plan_channels: int,
    max_frames: int | None = None,
    **model_kwargs: object,
) -> SpacetimeTransformer:
    """Build a :class:`SpacetimeTransformer` sized to the window + plan.

    ``max_frames`` must cover plan steps + camera frames; defaults to
    ``window.n_plan + window.n_frames`` with a little slack.
    """
    if max_frames is None:
        max_frames = window.n_plan + window.n_frames + 2
    cfg = SpacetimeConfig(
        max_frames=int(max_frames),
        plan_vocab=plan_vocab(),
        plan_channels=int(plan_channels),
        **model_kwargs,  # type: ignore[arg-type]
    )
    return SpacetimeTransformer(cfg)


def _plan_channels_for(samples: Sequence[SpacetimeSample]) -> int:
    """Per-step plan channel count seen across samples (0 = unconditioned)."""
    widths = [int(s.plan.shape[1]) for s in samples if s.plan.ndim == 2 and s.plan.size]
    return max(widths) if widths else 0


@dataclass
class _AutocastCtx:
    """bf16 autocast on CUDA, no-op elsewhere (H200 tensor cores)."""

    device: torch.device

    def __enter__(self):
        if self.device.type == "cuda":
            self._ctx = torch.autocast("cuda", dtype=torch.bfloat16)
            return self._ctx.__enter__()
        self._ctx = None
        return None

    def __exit__(self, *exc):
        if self._ctx is not None:
            return self._ctx.__exit__(*exc)
        return False


# ---------------------------------------------------------------------------
# Overfit (the GATE)
# ---------------------------------------------------------------------------


@dataclass
class OverfitConfig:
    steps: int = 400
    lr: float = 3e-4
    seed: int = 0
    log_every: int = 25
    chunk: int = 4096
    window: SpacetimeWindowConfig = field(default_factory=SpacetimeWindowConfig)
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class OverfitResult:
    initial_loss: float
    final_loss: float
    losses: list[float]
    n_parameters: int
    shot_ids: list[int]

    @property
    def loss_drop_ratio(self) -> float:
        if self.initial_loss <= 0:
            return 1.0
        return self.final_loss / self.initial_loss


def overfit(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    config: OverfitConfig | None = None,
    token_root: Path | None = None,
    device: str | None = None,
) -> tuple[OverfitResult, SpacetimeTransformer, list[SpacetimeSample]]:
    """Overfit a handful of rbb shots — the spatial-coherence GATE.

    Returns ``(result, model, samples)`` so the caller can immediately decode
    the overfit model's teacher-forced reconstruction (the gate's PASS check).
    The model is left on ``device``.
    """
    config = config or OverfitConfig()
    _set_determinism(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stop = _StopFlag()
    stop.install()

    samples = [
        assemble_window(int(sid), config.window, camera=camera, token_root=token_root)
        for sid in shot_ids
    ]
    plan_ch = _plan_channels_for(samples)
    batch = _batch_to(collate_windows(samples), dev)

    model = build_model(config.window, plan_channels=plan_ch, **config.model_kwargs).to(
        dev
    )
    model.train()
    logger.info(
        "overfit model on %s: params=%d (%.1fM) n_frames=%d plan_ch=%d shots=%s",
        dev,
        model.num_parameters(),
        model.num_parameters() / 1e6,
        config.window.n_frames,
        plan_ch,
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
                    "overfit step %d/%d loss=%.4f", step, config.steps, losses[-1]
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


# ---------------------------------------------------------------------------
# Teacher-forced + autoregressive prediction (for the decode gate + demo)
# ---------------------------------------------------------------------------


@torch.no_grad()
def teacher_forced_frames(
    model: SpacetimeTransformer,
    sample: SpacetimeSample,
    *,
    chunk: int = 4096,
    device: torch.device | None = None,
) -> np.ndarray:
    """Teacher-forced next-frame prediction over the whole window.

    Feeds the TRUE frames and reads the per-token argmax: frame ``t`` (``t>=1``)
    is predicted from the hidden state at frame ``t-1``.  Returns ``(T, S)``
    LOCAL token ids; frame 0 is copied from truth (no predecessor).
    """
    model.eval()
    dev = device or next(model.parameters()).device
    batch = _batch_to(collate_windows([sample]), dev)
    hidden = model._forward_tokens(batch["frames"], batch.get("plan"))  # (1,T,S,d)
    t = hidden.shape[1]
    s = hidden.shape[2]
    out = np.zeros((t, s), dtype=np.int64)
    out[0] = np.asarray(sample.frames[0], dtype=np.int64)
    for ti in range(1, t):
        pred = model.chunked_argmax_frame(hidden[:, ti - 1], chunk=chunk)  # (1,S)
        out[ti] = pred[0].cpu().numpy().astype(np.int64)
    return out


@torch.no_grad()
def autoregressive_dream(
    model: SpacetimeTransformer,
    sample: SpacetimeSample,
    *,
    chunk: int = 4096,
    device: torch.device | None = None,
) -> np.ndarray:
    """Autoregressive rollout: keep the context frames, generate the rest.

    The model is given the leading ``context_frames`` TRUE frames + the plan,
    then rolls forward consuming its OWN predicted frames.  Each new frame is
    decoded in one parallel pass from the previous frame's hidden state (the
    next-frame factorisation).  Returns ``(T, S)`` LOCAL token ids — the context
    frames are the truth, the rest the dream.
    """
    model.eval()
    dev = device or next(model.parameters()).device
    ctx = int(sample.context_frames)
    t_total = int(sample.frames.shape[0])

    plan = collate_windows([sample])["plan"].to(dev)
    gen = np.asarray(sample.frames, dtype=np.int64).copy()  # seed with truth
    for ti in range(ctx, t_total):
        # feed frames [0, ti) (truth in context, generated beyond) -> predict ti
        cur = torch.as_tensor(gen[:ti][None], dtype=torch.long, device=dev)  # (1,ti,S)
        hidden = model._forward_tokens(cur, plan)  # (1, ti, S, d)
        pred = model.chunked_argmax_frame(hidden[:, ti - 1], chunk=chunk)  # (1,S)
        gen[ti] = pred[0].cpu().numpy().astype(np.int64)
    return gen


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _config_to_dict(cfg: SpacetimeConfig) -> dict:
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
    }


def _config_from_dict(d: dict) -> SpacetimeConfig:
    return SpacetimeConfig(
        **{k: d[k] for k in d if k in SpacetimeConfig.__dataclass_fields__}
    )


def save_checkpoint(
    out_dir: Path,
    *,
    model: SpacetimeTransformer,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    window: SpacetimeWindowConfig,
    extra: dict | None = None,
) -> Path:
    """Atomic self-describing checkpoint to ``out_dir/latest.pt`` (+ snapshot)."""
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
    final = out_dir / "latest.pt"
    tmp = out_dir / f".latest.pt.{os.getpid()}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    try:
        torch.save(payload, out_dir / f"ckpt-{int(step):08d}.pt")
    except OSError as exc:  # noqa: BLE001
        logger.warning("could not write step snapshot: %r", exc)
    return final


def load_model_from_checkpoint(
    path: Path, *, map_location: str = "cpu"
) -> tuple[SpacetimeTransformer, dict]:
    """Rebuild the model from a checkpoint + load its weights (eval entry)."""
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = _config_from_dict(payload["model_config"])
    model = SpacetimeTransformer(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    return model, payload


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    p = Path(out_dir) / "latest.pt"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Corpus trainer (DDP-aware; single-proc default unchanged)
# ---------------------------------------------------------------------------


@dataclass
class CorpusConfig:
    steps: int = 8000
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.05
    seed: int = 0
    log_every: int = 25
    ckpt_every: int = 500
    eval_every: int = 1000
    num_workers: int = 4
    prefetch_factor: int = 4
    grad_clip: float = 1.0
    chunk: int = 4096
    n_eval_shots: int = 2
    window: SpacetimeWindowConfig = field(default_factory=SpacetimeWindowConfig)
    model_kwargs: dict = field(default_factory=dict)
    random_window: bool = True


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


def _default_out_dir() -> Path:
    run_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("WM_RUN_ID") or "local"
    return DEFAULT_CKPT_ROOT / f"spacetime-{run_id}"


@torch.no_grad()
def _eval_reconstruction(
    model: SpacetimeTransformer,
    eval_shots: Sequence[int],
    window: SpacetimeWindowConfig,
    *,
    camera: str,
    token_root: Path | None,
    n: int,
    chunk: int,
    device: torch.device,
) -> float:
    """Mean teacher-forced next-frame token-mismatch on held-out shots (0..1).

    A coherent model has LOW mismatch (predicted tokens match truth).  Reported
    as a quick proxy metric during training (lower is better).
    """
    model.eval()
    rates: list[float] = []
    for sid in list(eval_shots)[:n]:
        try:
            sample = assemble_window(
                int(sid), window, camera=camera, token_root=token_root
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning("eval shot %s NOT assemblable: %r", sid, exc)
            continue
        pred = teacher_forced_frames(model, sample, chunk=chunk, device=device)
        truth = np.asarray(sample.frames, dtype=np.int64)
        ctx = int(sample.context_frames)
        mism = float((pred[ctx:] != truth[ctx:]).mean())
        rates.append(mism)
    model.train()
    if not rates:
        return float("nan")
    return float(np.mean(rates))


def train_corpus(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    config: CorpusConfig | None = None,
    out_dir: Path | None = None,
    token_root: Path | None = None,
    eval_shot_ids: Sequence[int] | None = None,
    device: str | None = None,
    resume: bool = True,
) -> CorpusResult:
    """Train the spatiotemporal camera transformer on a CORPUS of shots.

    DDP-aware: when launched under torchrun (``WORLD_SIZE > 1``) each rank pins
    its card, trains a disjoint shot shard (DistributedSampler), and the loss is
    the chunked next-frame NLL computed inside forward so DDP's reducer
    all-reduces every grad.  Rank-0 only logs / checkpoints / evals (symmetric
    barriers keep the ring in sync).  Single-proc is the default and unchanged.
    """
    config = config or CorpusConfig()
    out_dir = Path(out_dir) if out_dir is not None else _default_out_dir()
    _set_determinism(config.seed)

    env = DistEnv.from_environment()
    _init_distributed(env)
    if device is None:
        device = f"cuda:{env.local_rank}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    stop = _StopFlag()
    stop.install()

    # probe a few shots for the plan channel count (model sizing).
    probe: list[SpacetimeSample] = []
    for sid in list(shot_ids)[:8]:
        try:
            probe.append(
                assemble_window(
                    int(sid), config.window, camera=camera, token_root=token_root
                )
            )
        except (ValueError, FileNotFoundError, KeyError):
            continue
    if not probe:
        raise ValueError("no shot assembled in the probe — cannot size the model")
    plan_ch = _plan_channels_for(probe)

    base_model = build_model(
        config.window, plan_channels=plan_ch, **config.model_kwargs
    ).to(dev)
    if env.is_main:
        logger.info(
            "model on %s: params=%d (%.1fM) d_model=%d n_layers=%d n_heads=%d "
            "n_frames=%d plan_ch=%d world=%d",
            dev,
            base_model.num_parameters(),
            base_model.num_parameters() / 1e6,
            base_model.config.d_model,
            base_model.config.n_layers,
            base_model.config.n_heads,
            config.window.n_frames,
            plan_ch,
            env.world_size,
        )

    opt = torch.optim.AdamW(
        base_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    start_step = 0
    eval_errors: list[tuple[int, float]] = []
    if resume:
        latest = find_latest_checkpoint(out_dir)
        if latest is not None:
            payload = torch.load(str(latest), map_location="cpu", weights_only=False)
            try:
                base_model.load_state_dict(payload["model_state_dict"])
                if payload.get("optimizer_state_dict"):
                    opt.load_state_dict(payload["optimizer_state_dict"])
                start_step = int(payload.get("step", 0))
                eval_errors = list(payload.get("extra", {}).get("eval_errors", []))
                if env.is_main:
                    logger.info("RESUMED from %s at step %d", latest, start_step)
            except (KeyError, RuntimeError, ValueError) as exc:
                if env.is_main:
                    logger.warning(
                        "checkpoint %s incompatible (%r) — fresh", latest, exc
                    )

    if env.enabled:
        from torch.nn.parallel import DistributedDataParallel

        ddp_kwargs: dict = {}
        if dev.type == "cuda":
            ddp_kwargs["device_ids"] = [env.local_rank]
            ddp_kwargs["output_device"] = env.local_rank
        model: nn.Module = DistributedDataParallel(base_model, **ddp_kwargs)
    else:
        model = base_model
    core = _unwrap(model)

    dataset = SpacetimeFrameDataset(
        shot_ids,
        config.window,
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
    loader_kwargs: dict = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_windows,
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
                losses.append(float(loss.detach()))
                step += 1

                if env.is_main and (
                    step % config.log_every == 0 or step == config.steps
                ):
                    rate = config.log_every / max(time.time() - t_last, 1e-6)
                    t_last = time.time()
                    logger.info(
                        "corpus step %d/%d loss=%.4f (%.2f st/s world=%d)",
                        step,
                        config.steps,
                        losses[-1],
                        rate,
                        env.world_size,
                    )

                if config.ckpt_every > 0 and step % config.ckpt_every == 0:
                    _barrier(env)
                    if env.is_main:
                        ckpt_path = save_checkpoint(
                            out_dir,
                            model=core,
                            optimizer=opt,
                            step=step,
                            window=config.window,
                            extra={"eval_errors": eval_errors},
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
                    model.train()
                    _barrier(env)

        _barrier(env)
        if env.is_main:
            ckpt_path = save_checkpoint(
                out_dir,
                model=core,
                optimizer=opt,
                step=step,
                window=config.window,
                extra={"eval_errors": eval_errors},
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
    if args.shots.strip():
        shots = [int(s) for s in args.shots.split(",") if s.strip()]
    else:
        shots = discover_camera_shots(
            camera=args.camera,
            token_root=token_root,
            min_frames=span,
            limit=args.n_shots,
        )
    if len(shots) < 2:
        logger.error("need >= 2 shots; discovered %d", len(shots))
        return 1
    # hold out the LAST n_eval shots (deterministic, ascending order) for demo.
    n_eval = max(1, min(args.n_eval_shots, len(shots) // 5 or 1))
    eval_shots = shots[-n_eval:]
    train_shots = shots[:-n_eval] or shots

    model_kwargs: dict = {}
    for k in ("d_model", "n_layers", "n_heads", "d_ff", "dropout"):
        v = getattr(args, k, None)
        if v is not None:
            model_kwargs[k] = v

    cfg = CorpusConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        chunk=args.chunk,
        n_eval_shots=n_eval,
        window=window,
        model_kwargs=model_kwargs,
    )
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    logger.info(
        "spacetime corpus: %d train / %d eval shots (held out %s) out=%s",
        len(train_shots),
        len(eval_shots),
        eval_shots,
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
        "CORPUS DONE: steps=%d params=%d initial=%.4f final=%.4f ckpt=%s",
        result.steps_run,
        result.n_parameters,
        result.initial_loss,
        result.final_loss,
        result.checkpoint_path,
    )
    if result.eval_errors:
        logger.info("eval mismatch trajectory: %s", result.eval_errors)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command")
    pc = sub.add_parser("corpus", help="train the spatiotemporal camera transformer")
    pc.add_argument("--shots", default="")
    pc.add_argument("--n-shots", type=int, default=2000)
    pc.add_argument("--camera", default=REFERENCE_CAMERA)
    pc.add_argument("--n-eval-shots", type=int, default=4)
    pc.add_argument("--steps", type=int, default=8000)
    pc.add_argument("--batch-size", type=int, default=4)
    pc.add_argument("--lr", type=float, default=3e-4)
    pc.add_argument("--weight-decay", type=float, default=0.05)
    pc.add_argument("--n-frames", type=int, default=24)
    pc.add_argument("--n-plan", type=int, default=8)
    pc.add_argument("--context-frames", type=int, default=8)
    pc.add_argument("--frame-stride", type=int, default=1)
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
