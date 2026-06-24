"""DDP corpus trainer for the command-conditioned recurrent latent world model.

Phase-2 of the joint multi-modal plasma world model: the token backbone
(:mod:`imas_ambix.worldmodel.controllable_train`) injects the actuator plan as an
AdaLN SIDE-input the model could ignore — the powered ΔN-M controllability gate
confirmed it (true-plan rollouts ≈ random-plan rollouts).  The RSSM
(:class:`imas_ambix.worldmodel.rssm.RSSMWorldModel`) puts the command INSIDE the
recurrent transition, so a different command necessarily yields a different
rollout — controllability BY CONSTRUCTION.

This module trains that model on the SAME curated excited-window corpus the
Phase-1 controllable trainer uses, reusing its data path verbatim (the
overlapping-window manifest dataset, the controllable collate, the batch-move,
the signal-stream probe, and the masked-command-column resolver).  The RSSM's
:meth:`~imas_ambix.worldmodel.rssm.RSSMWorldModel.forward` consumes exactly the
batch the controllable collate emits:

* ``frames`` ``(B, T, S)`` — the camera tokens (the reconstruction target),
* ``actuator`` ``{"values","missing"}`` — the demanded plan = the COMMAND,
* ``signals`` ``{name: (B, Ps, Cs)}`` — the measured streams = the diagnostic
  targets,

so the RSSM ignores the controllable model's plan prefix / history-bottleneck /
scheduled-sampling machinery entirely (the command is load-bearing through the
transition, not a side-input that needs those props to bite).  The loss is the
RSSM ELBO ``camera_CE + diagnostic_weight*diagnostic_CE + beta*KL`` — already
assembled inside ``forward``.

DDP: one process per GPU (torchrun); each rank trains a disjoint shot shard via
a :class:`~torch.utils.data.distributed.DistributedSampler`; the signal-stream
widths + actuator-channel count are decided on rank 0 and broadcast so every
rank builds the IDENTICAL model.

GPU-safety (repo AGENTS.md §2b): the model is loaded ONCE outside the per-step
loop; a SIGTERM/SIGINT STOP flag flushes ``latest.pt`` cleanly within the step
(< 5 s) and a restart resumes from it; cudnn deterministic; bf16 autocast;
float32 matmul precision high.  Warm-start from a Phase-1 controllable checkpoint
loads the reusable camera token-embed / head / row-col position / diagnostic
heads (the recurrent latent core stays fresh).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from imas_ambix.worldmodel.controllable_train import (
    _batch_to,
    _ManifestWindow,
    collate_controllable_windows,
    manifest_train_windows,
    masked_command_columns,
)
from imas_ambix.worldmodel.rssm import RSSMConfig, RSSMOutput, RSSMWorldModel
from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    SignalModalitySpec,
    default_signal_modalities,
    extended_signal_modalities,
    stream_specs_from_modalities,
)
from imas_ambix.worldmodel.spacetime_train import (
    _AutocastCtx,
    _set_determinism,
    _StopFlag,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint round-trip (self-describing — config + state + optimizer + step)
# ---------------------------------------------------------------------------


def _rssm_config_to_dict(cfg: RSSMConfig) -> dict:
    """Serialise the full :class:`RSSMConfig` so the model round-trips exactly.

    The signal-stream specs are flattened to ``{name,vocab,channels}`` dicts and
    the masked-command tuple to a list (JSON / torch-save friendly), so
    :func:`load_rssm_model_from_checkpoint` reconstructs the IDENTICAL model — the
    stream set + masked-command columns included, so an eval conditions exactly as
    the trained model did.
    """
    return {
        "vocab_size": int(cfg.vocab_size),
        "grid_h": int(cfg.grid_h),
        "grid_w": int(cfg.grid_w),
        "d_model": int(cfg.d_model),
        "h_dim": int(cfg.h_dim),
        "s_dim": int(cfg.s_dim),
        "a_dim": int(cfg.a_dim),
        "cmd_hidden": int(cfg.cmd_hidden),
        "latent_hidden": int(cfg.latent_hidden),
        "decoder_hidden": int(cfg.decoder_hidden),
        "min_std": float(cfg.min_std),
        "beta": float(cfg.beta),
        "free_bits": float(cfg.free_bits),
        "diagnostic_weight": float(cfg.diagnostic_weight),
        "action_contrastive": bool(cfg.action_contrastive),
        "action_contrastive_weight": float(cfg.action_contrastive_weight),
        "contrastive_dim": int(cfg.contrastive_dim),
        "action_contrastive_temperature": float(cfg.action_contrastive_temperature),
        "signal_streams": [
            {"name": s.name, "vocab": int(s.vocab), "channels": int(s.channels)}
            for s in cfg.signal_streams
        ],
        "actuator_channels": int(cfg.actuator_channels),
        "masked_command_indices": list(cfg.masked_command_indices),
    }


def _rssm_config_from_dict(d: dict) -> RSSMConfig:
    """Rebuild an :class:`RSSMConfig` from :func:`_rssm_config_to_dict` output."""
    from imas_ambix.worldmodel.rssm import SignalStreamSpec  # noqa: PLC0415

    streams = tuple(
        SignalStreamSpec(
            name=str(s["name"]), vocab=int(s["vocab"]), channels=int(s["channels"])
        )
        for s in d.get("signal_streams", [])
    )
    return RSSMConfig(
        vocab_size=int(d["vocab_size"]),
        grid_h=int(d["grid_h"]),
        grid_w=int(d["grid_w"]),
        d_model=int(d["d_model"]),
        h_dim=int(d["h_dim"]),
        s_dim=int(d["s_dim"]),
        a_dim=int(d["a_dim"]),
        cmd_hidden=int(d["cmd_hidden"]),
        latent_hidden=int(d["latent_hidden"]),
        decoder_hidden=int(d["decoder_hidden"]),
        min_std=float(d["min_std"]),
        beta=float(d["beta"]),
        free_bits=float(d["free_bits"]),
        diagnostic_weight=float(d["diagnostic_weight"]),
        action_contrastive=bool(d.get("action_contrastive", False)),
        action_contrastive_weight=float(d.get("action_contrastive_weight", 1.0)),
        contrastive_dim=int(d.get("contrastive_dim", 128)),
        action_contrastive_temperature=float(
            d.get("action_contrastive_temperature", 0.1)
        ),
        signal_streams=streams,
        actuator_channels=int(d["actuator_channels"]),
        masked_command_indices=tuple(
            int(i) for i in d.get("masked_command_indices", ())
        ),
    )


def save_rssm_checkpoint(
    out_dir,
    *,
    model: RSSMWorldModel,
    optimizer,
    step: int,
    extra: dict | None = None,
    name: str = "latest.pt",
    snapshot: bool = True,
):
    """Atomic self-describing RSSM checkpoint (full config + state + optimizer).

    Writes ``<out_dir>/<name>`` atomically (tmp → ``os.replace``) so a SIGTERM mid
    write never leaves a torn ``latest.pt``.  ``snapshot`` also writes a
    ``ckpt-<step>.pt`` for the step.  ``extra`` carries run metadata (the
    ``stream_names`` the eval needs to build a matching batch).
    """
    import os  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "model_config": _rssm_config_to_dict(model.config),
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


def load_rssm_model_from_checkpoint(
    path, *, map_location: str = "cpu"
) -> tuple[RSSMWorldModel, dict]:
    """Rebuild an RSSM from a checkpoint + load its weights (eval / resume).

    Reconstructs the FULL :class:`RSSMConfig` (stream specs + masked-command set)
    so the model conditions exactly as the trained one.  ``strict=False`` tolerates
    a checkpoint with / without diagnostic heads against a model built the other
    way (logged).  Returns ``(model, payload)``.
    """
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = _rssm_config_from_dict(dict(payload["model_config"]))
    model = RSSMWorldModel(cfg)
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=False
    )
    if missing or unexpected:
        logger.info(
            "loaded RSSM checkpoint with strict=False: %d missing, %d unexpected keys",
            len(missing),
            len(unexpected),
        )
    model.to(map_location)
    return model, payload


def warm_start_reusable_from_checkpoint(
    model: RSSMWorldModel, ckpt_path: str
) -> dict[str, int]:
    """Load the reusable camera/head/diagnostic weights from any WM checkpoint.

    The model's own :meth:`RSSMWorldModel.warm_start_from_phase1` only recognises a
    raw ``state_dict`` or a ``{"model"|"model_state"|"state_dict": …}`` wrapper —
    but BOTH the Phase-1 controllable checkpoints (``save_controllable_checkpoint``)
    AND the RSSM checkpoints here store the weights under ``model_state_dict``, which
    that method does NOT look under, so it would silently load 0 reusable tensors
    and the warm start would be a no-op (wasting the run's inheritance of the
    trained diagnostic heads + camera head/embed).

    This wrapper recognises ``model_state_dict`` too (plus the keys the model
    already handles), strips a DDP ``module.`` prefix, and delegates to the model's
    shape-matched :meth:`RSSMWorldModel._load_reusable` — so a Phase-1 controllable
    ``latest.pt`` warm-starts correctly.  Returns the
    ``{"loaded","fresh","skipped_shape"}`` tensor counts.
    """
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    src = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model", "model_state", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                src = payload[key]
                break
    if not isinstance(src, dict):
        raise ValueError(f"could not find a state_dict in checkpoint {ckpt_path!r}")
    src = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in src.items()
    }
    return model._load_reusable(src)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RSSMCorpusConfig:
    """Knobs for the DDP corpus train of the recurrent latent world model.

    Mirrors the Phase-1 controllable corpus config's training/LR/checkpoint/DDP
    surface; the RSSM-specific knobs are ``beta`` / ``free_bits`` (the KL weight +
    per-dim floor), ``diagnostic_weight`` (the secondary diagnostic-CE weight), and
    the latent dims ``h_dim`` / ``s_dim`` (the recurrent + stochastic state widths,
    overriding the :class:`RSSMConfig` defaults).  The RSSM ignores the controllable
    model's history-bottleneck / scheduled-sampling / observation-dropout / AdaLN
    machinery — the command is load-bearing through the transition itself.
    """

    steps: int = 12000
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.1
    seed: int = 0
    log_every: int = 25
    ckpt_every: int = 250
    num_workers: int = 4
    prefetch_factor: int = 4
    grad_clip: float = 1.0
    chunk: int = 4096
    n_signal_steps: int = 4
    n_act_steps: int = 8
    lr_schedule: bool = True
    warmup_steps: int = 400
    min_lr_ratio: float = 0.02
    # RSSM objective knobs.
    beta: float = 1.0
    free_bits: float = 1.0
    diagnostic_weight: float = 0.5
    # action-contrastive on the latent (OFF -> byte-identical to the plain ELBO).
    action_contrastive: bool = False
    action_contrastive_weight: float = 1.0
    # RSSM latent-state dims (override the RSSMConfig defaults).
    h_dim: int = 256
    s_dim: int = 32
    # MASK the measured states (Ip + density + tf) out of the COMMAND vector — they
    # are responses, not commands (drive-from-commands).  True is correct.
    drop_state_channels: bool = True
    window: SpacetimeWindowConfig = field(default_factory=SpacetimeWindowConfig)
    modalities: list[SignalModalitySpec] = field(
        default_factory=extended_signal_modalities
    )
    # warm-start the reusable camera token-embed / head / row-col / diagnostic heads
    # from a Phase-1 controllable checkpoint ONCE (when no resume of THIS run exists).
    init_checkpoint: Path | None = None
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class RSSMCorpusResult:
    steps_run: int
    initial_loss: float
    final_loss: float
    losses: list[float]
    n_parameters: int
    n_train_shots: int
    checkpoint_path: str | None
    stream_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The per-step core (factored out so the smoke test drives the real path)
# ---------------------------------------------------------------------------


def _rssm_train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    *,
    chunk: int,
    grad_clip: float,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> RSSMOutput:
    """One optimiser step on the RSSM ELBO over a (moved) controllable batch.

    Zeroes grads, runs the bf16-autocast forward (the RSSM consumes ``frames`` /
    ``actuator`` / ``signals`` directly), backprops the total ELBO loss, clips +
    steps, and advances the scheduler.  Returns the :class:`RSSMOutput` so the
    caller can log the loss components.  The unwrapped model is used for the
    forward kwarg path; DDP all-reduces the grads on ``.backward()``.
    """
    optimizer.zero_grad(set_to_none=True)
    with _AutocastCtx(device):
        out = model(batch, chunk=chunk)
        loss = out.loss
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return out


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------


def build_rssm_model(
    *,
    actuator_channels: int,
    signal_streams,
    masked_command_indices: tuple[int, ...] = (),
    h_dim: int = 256,
    s_dim: int = 32,
    beta: float = 1.0,
    free_bits: float = 1.0,
    diagnostic_weight: float = 0.5,
    action_contrastive: bool = False,
    action_contrastive_weight: float = 1.0,
    **model_kwargs,
) -> RSSMWorldModel:
    """Build an :class:`RSSMWorldModel` sized to the corpus command + streams."""
    cfg = RSSMConfig(
        actuator_channels=int(actuator_channels),
        signal_streams=tuple(signal_streams),
        masked_command_indices=tuple(int(i) for i in masked_command_indices),
        h_dim=int(h_dim),
        s_dim=int(s_dim),
        beta=float(beta),
        free_bits=float(free_bits),
        diagnostic_weight=float(diagnostic_weight),
        action_contrastive=bool(action_contrastive),
        action_contrastive_weight=float(action_contrastive_weight),
        **model_kwargs,
    )
    return RSSMWorldModel(cfg)


# ---------------------------------------------------------------------------
# Corpus DDP trainer
# ---------------------------------------------------------------------------


def train_rssm_corpus(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    config: RSSMCorpusConfig | None = None,
    out_dir: Path | None = None,
    token_root: Path | None = None,
    device: str | None = None,
    resume: bool = True,
    manifest_windows: Sequence[_ManifestWindow] | None = None,
) -> RSSMCorpusResult:
    """DDP corpus train of the recurrent latent world model on EXCITED windows.

    Reuses the Phase-1 controllable corpus data path verbatim — the curated
    excited-window manifest dataset (:class:`ManifestWindowDataset`), the
    controllable collate (frames + actuator + measured signals), the batch-move,
    and the signal-stream/actuator-channel probe — but the model is the RSSM and
    the loss is its ELBO.  Each rank pins its card and trains a disjoint shard; the
    stream widths + actuator-channel count are decided on rank 0 and broadcast so
    every rank builds the IDENTICAL model.  Resume-safe: ``latest.pt`` every
    ``ckpt_every`` steps + a SIGTERM-clean flush; a restart resumes from it.
    """
    import contextlib  # noqa: PLC0415
    import time  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    from torch import nn  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_dataset import (  # noqa: PLC0415
        assemble_controllable_window,
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        ManifestWindowDataset,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        camera_frame_count,
        recording_time_span_s,
        window_span_for_shot,
    )
    from imas_ambix.worldmodel.spacetime_dataset_v2 import (  # noqa: PLC0415
        probe_signal_channels,
    )
    from imas_ambix.worldmodel.spacetime_train import (  # noqa: PLC0415
        DEFAULT_CKPT_ROOT,
        DistEnv,
        _barrier,
        _broadcast_int,
        _init_distributed,
        _unwrap,
        build_lr_scheduler,
        find_latest_checkpoint,
    )

    config = config or RSSMCorpusConfig()
    if out_dir is None:
        import os  # noqa: PLC0415

        run_id = (
            os.environ.get("SLURM_JOB_ID") or os.environ.get("WM_RUN_ID") or "local"
        )
        out_dir = DEFAULT_CKPT_ROOT / f"rssm-{run_id}"
    out_dir = _Path(out_dir)
    _set_determinism(config.seed)

    env = DistEnv.from_environment()
    _init_distributed(env)
    if device is None:
        device = f"cuda:{env.local_rank}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    stop = _StopFlag()
    stop.install()

    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ACTUATOR_CHANNELS,
    )

    act_chan_list = list(ACTUATOR_CHANNELS)

    use_manifest = manifest_windows is not None
    horizon = float(config.window.target_horizon_s)

    # Filter to windows/shots whose recording spans the horizon (an under-length
    # shot raises mid-epoch and kills a DDP rank).  Deterministic so every rank
    # computes the SAME set (DDP-safe).
    if use_manifest:
        span_count_cache: dict[tuple[int, str], tuple[float | None, int] | None] = {}

        def _span_count(sid: int, cam: str) -> tuple[float | None, int] | None:
            key = (sid, cam)
            if key in span_count_cache:
                return span_count_cache[key]
            try:
                sp = recording_time_span_s(sid, camera=cam, token_root=token_root)
                nt = int(camera_frame_count(sid, cam, token_root=token_root))
                res: tuple[float | None, int] | None = (sp, nt)
            except (FileNotFoundError, KeyError, ValueError):
                res = None
            span_count_cache[key] = res
            return res

        kept_windows: list[_ManifestWindow] = []
        scanned = 0
        for w in manifest_windows:
            w_horizon = w.horizon_s if w.horizon_s > 0 else horizon
            if w.avail_frames > 0:
                ok = w.avail_frames >= config.window.n_frames
            else:
                sc = _span_count(w.shot_id, w.camera)
                scanned += 1
                if sc is None:
                    ok = False
                else:
                    span_s, n_total = sc
                    if w_horizon > 0:
                        ok = (
                            span_s is not None
                            and span_s >= w_horizon
                            and n_total >= config.window.n_frames
                        )
                    else:
                        ok = (
                            n_total >= (config.window.n_frames - 1) * w.frame_stride + 1
                        )
            if ok:
                kept_windows.append(w)
        n_dropped = len(list(manifest_windows)) - len(kept_windows)
        shot_ids = [w.shot_id for w in kept_windows]
        if env.is_main:
            logger.info(
                "manifest-window filter: kept %d/%d excited windows with >= %d "
                "frames (dropped %d too-short; %d needed a GPFS scan)",
                len(kept_windows),
                len(list(manifest_windows)),
                config.window.n_frames,
                n_dropped,
                scanned,
            )
        if len(kept_windows) < 2:
            raise ValueError(
                f"only {len(kept_windows)} excited windows are long enough for the "
                f"~{config.window.target_horizon_s}s horizon — cannot train"
            )
        manifest_windows = kept_windows
    else:
        kept: list[int] = []
        for sid in shot_ids:
            try:
                shot_span = window_span_for_shot(
                    int(sid), config.window, camera=camera, token_root=token_root
                )
                if (
                    int(camera_frame_count(int(sid), camera, token_root=token_root))
                    >= shot_span
                ):
                    kept.append(int(sid))
            except (FileNotFoundError, KeyError, ValueError):
                continue
        if env.is_main:
            logger.info(
                "frame-count filter: kept %d/%d train shots long enough for the "
                "~%.2fs horizon window",
                len(kept),
                len(list(shot_ids)),
                config.window.target_horizon_s,
            )
        if len(kept) < 2:
            raise ValueError(
                f"only {len(kept)} train shots are long enough for the "
                f"~{config.window.target_horizon_s}s horizon window — cannot train"
            )
        shot_ids = kept

    # Probe the signal-stream widths + the actuator-channel count.  Spread the probe
    # across the corpus (the manifest is sorted by shot id; a first-N probe misses
    # era-sparse streams) and use each window's OWN camera (the unified manifest
    # mixes views).
    probe = []
    if use_manifest:
        for w in list(manifest_windows)[:8]:
            try:
                probe.append(
                    assemble_controllable_window(
                        w.shot_id,
                        config.window,
                        config.modalities,
                        config.n_signal_steps,
                        config.n_act_steps,
                        camera=w.camera,
                        token_root=token_root,
                        start_frame=w.start_frame,
                        actuator_channels=act_chan_list,
                    )
                )
            except (ValueError, FileNotFoundError, KeyError):
                continue
    else:
        for sid in list(shot_ids)[:8]:
            try:
                probe.append(
                    assemble_controllable_window(
                        int(sid),
                        config.window,
                        config.modalities,
                        config.n_signal_steps,
                        config.n_act_steps,
                        camera=camera,
                        token_root=token_root,
                        actuator_channels=act_chan_list,
                    )
                )
            except (ValueError, FileNotFoundError, KeyError):
                continue
    if not probe:
        raise ValueError("no shot assembled in the probe — cannot size the model")

    if use_manifest:
        mw = list(manifest_windows)
        n_probe = min(48, len(mw))
        step_pr = max(1, len(mw) // n_probe)
        probe_sel = mw[::step_pr][:n_probe]
        probe_shot_ids = [w.shot_id for w in probe_sel]
        probe_cameras = [w.camera for w in probe_sel]
    else:
        probe_shot_ids = list(shot_ids)[:16]
        probe_cameras = None
    channels = probe_signal_channels(
        probe_shot_ids,
        config.window,
        config.modalities,
        config.n_signal_steps,
        camera=camera,
        cameras=probe_cameras,
        token_root=token_root,
        max_probe=32,
    )
    act_channels = int(probe[0].actuator.n_channels)
    masked_idx = (
        masked_command_columns(list(probe[0].actuator.channel_keys))
        if config.drop_state_channels
        else ()
    )

    # Broadcast the model-shaping quantities so every rank builds the SAME model.
    act_channels = _broadcast_int(env, act_channels)
    for m in config.modalities:
        channels[m.name] = _broadcast_int(env, int(channels.get(m.name, 0)))
    streams = stream_specs_from_modalities(config.modalities, channels)
    stream_names = [st.name for st in streams]

    base_model = build_rssm_model(
        actuator_channels=act_channels,
        signal_streams=streams,
        masked_command_indices=masked_idx,
        h_dim=config.h_dim,
        s_dim=config.s_dim,
        beta=config.beta,
        free_bits=config.free_bits,
        diagnostic_weight=config.diagnostic_weight,
        action_contrastive=config.action_contrastive,
        action_contrastive_weight=config.action_contrastive_weight,
        **config.model_kwargs,
    ).to(dev)
    if env.is_main:
        logger.info(
            "rssm-corpus on %s: params=%d (%.1fM) d_model=%d h_dim=%d s_dim=%d "
            "actuator_ch=%d masked=%s beta=%.3f free_bits=%.3f diag_w=%.3f "
            "action_contrastive=%s ac_w=%.3f streams=%s world=%d",
            dev,
            base_model.num_parameters(),
            base_model.num_parameters() / 1e6,
            base_model.config.d_model,
            base_model.config.h_dim,
            base_model.config.s_dim,
            act_channels,
            masked_idx,
            config.beta,
            config.free_bits,
            config.diagnostic_weight,
            base_model.config.has_action_contrastive,
            config.action_contrastive_weight,
            [(st.name, st.channels) for st in streams],
            env.world_size,
        )

    opt = torch.optim.AdamW(
        base_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    start_step = 0

    # Resume THIS run's latest.pt first; else warm-start from the Phase-1 ckpt ONCE.
    latest = find_latest_checkpoint(out_dir) if resume else None
    if latest is not None:
        try:
            payload = torch.load(str(latest), map_location="cpu", weights_only=False)
            _unwrap(base_model).load_state_dict(payload["model_state_dict"])
            if payload.get("optimizer_state_dict") is not None:
                opt.load_state_dict(payload["optimizer_state_dict"])
            start_step = int(payload.get("step", 0))
            if env.is_main:
                logger.info("RESUMED from %s at step %d", latest, start_step)
        except (KeyError, RuntimeError, ValueError) as exc:
            if env.is_main:
                logger.warning("checkpoint %s incompatible (%r) — fresh", latest, exc)
    elif config.init_checkpoint is not None:
        counts = warm_start_reusable_from_checkpoint(
            _unwrap(base_model), str(config.init_checkpoint)
        )
        _unwrap(base_model).to(dev)
        if env.is_main:
            logger.info(
                "warm-started RSSM from Phase-1 checkpoint %s: %d tensors loaded, "
                "%d fresh (latent core), %d skipped (shape)",
                config.init_checkpoint,
                counts.get("loaded", 0),
                counts.get("fresh", 0),
                counts.get("skipped_shape", 0),
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

    if env.enabled:
        from torch.nn.parallel import DistributedDataParallel  # noqa: PLC0415

        ddp_kwargs: dict = {"find_unused_parameters": True}
        if dev.type == "cuda":
            ddp_kwargs["device_ids"] = [env.local_rank]
            ddp_kwargs["output_device"] = env.local_rank
        model: nn.Module = DistributedDataParallel(base_model, **ddp_kwargs)
    else:
        model = base_model
    core = _unwrap(model)

    if use_manifest:
        dataset: object = ManifestWindowDataset(
            manifest_windows,
            config.window,
            config.modalities,
            config.n_signal_steps,
            config.n_act_steps,
            camera=camera,
            token_root=token_root,
            actuator_channels=act_chan_list,
        )
    else:
        from imas_ambix.worldmodel.controllable_dataset import (  # noqa: PLC0415
            ControllableSpacetimeDataset,
        )

        dataset = ControllableSpacetimeDataset(
            shot_ids,
            config.window,
            config.modalities,
            config.n_signal_steps,
            config.n_act_steps,
            camera=camera,
            token_root=token_root,
            random_window=True,
            seed=config.seed,
            actuator_channels=act_chan_list,
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

    def _collate(samples):  # noqa: ANN001
        return collate_controllable_windows(samples, stream_names=stream_names)

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

    accum = max(1, int(config.grad_accum))

    model.train()
    losses: list[float] = []
    step = start_step
    ckpt_path: Path | None = latest
    t_last = time.time()
    epoch = 0
    micro = 0
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
                is_boundary = (micro + 1) % accum == 0
                sync_ctx = (
                    model.no_sync()
                    if (env.enabled and not is_boundary)
                    else contextlib.nullcontext()
                )
                with sync_ctx, _AutocastCtx(dev):
                    out = model(batch, chunk=config.chunk)
                    loss = out.loss
                    scaled = loss / accum
                cam_ce = float(out.camera_ce.detach())
                diag_ce = float(out.diagnostic_ce.detach())
                kl = float(out.kl.detach())
                ac = float(out.action_contrastive.detach())
                scaled.backward()
                micro += 1
                if not is_boundary:
                    continue
                micro = 0
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                losses.append(float(loss.detach()))
                step += 1

                if env.is_main and (
                    step % config.log_every == 0 or step == config.steps
                ):
                    rate = config.log_every / max(time.time() - t_last, 1e-6)
                    t_last = time.time()
                    logger.info(
                        "rssm-corpus step %d/%d loss=%.4f cam=%.4f diag=%.4f kl=%.4f "
                        "ac=%.4f lr=%.3e (%.2f st/s world=%d)",
                        step,
                        config.steps,
                        losses[-1],
                        cam_ce,
                        diag_ce,
                        kl,
                        ac,
                        opt.param_groups[0]["lr"],
                        rate,
                        env.world_size,
                    )

                if config.ckpt_every > 0 and step % config.ckpt_every == 0:
                    _barrier(env)
                    if env.is_main:
                        ckpt_path = save_rssm_checkpoint(
                            out_dir,
                            model=core,
                            optimizer=opt,
                            step=step,
                            extra={"stream_names": stream_names},
                        )
                        logger.info("checkpoint @ step %d -> %s", step, ckpt_path)
                    _barrier(env)
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    # Final checkpoint flush (covers a clean SIGTERM stop mid-run).
    _barrier(env)
    if env.is_main:
        ckpt_path = save_rssm_checkpoint(
            out_dir,
            model=core,
            optimizer=opt,
            step=step,
            extra={"stream_names": stream_names},
        )
        logger.info("final checkpoint @ step %d -> %s", step, ckpt_path)
    _barrier(env)

    return RSSMCorpusResult(
        steps_run=step - start_step,
        initial_loss=losses[0] if losses else float("nan"),
        final_loss=losses[-1] if losses else float("nan"),
        losses=losses,
        n_parameters=base_model.num_parameters(),
        n_train_shots=len(list(shot_ids)),
        checkpoint_path=str(ckpt_path) if ckpt_path else None,
        stream_names=stream_names,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    p = argparse.ArgumentParser(description="RSSM latent world-model trainer.")
    sub = p.add_subparsers(dest="command")
    pc = sub.add_parser("corpus", help="DDP train on the curated excited corpus")
    pc.add_argument(
        "--manifest",
        default="/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_unified_6cam.json",
        help="curated-window manifest; its distinct shot ids are the train pool",
    )
    pc.add_argument(
        "--shots",
        default="",
        help="explicit comma-separated train shots (overrides --manifest)",
    )
    pc.add_argument("--n-shots", type=int, default=0, help="cap train shots (0 = all)")
    pc.add_argument("--eval-shots", default="18502,18503,18504,18505")
    pc.add_argument("--camera", default="rbb")
    pc.add_argument("--token-root", default=None)
    pc.add_argument("--out-dir", default=None)
    pc.add_argument(
        "--init-checkpoint",
        default="/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221834/latest.pt",
        help="warm-start the reusable camera/head/diagnostic weights from a Phase-1 "
        "controllable checkpoint (the latent core stays fresh)",
    )
    # window
    pc.add_argument("--n-frames", type=int, default=24)
    pc.add_argument("--n-plan", type=int, default=8)
    pc.add_argument("--context-frames", type=int, default=8)
    pc.add_argument("--frame-stride", type=int, default=1)
    pc.add_argument("--target-horizon-s", type=float, default=0.25)
    pc.add_argument("--n-signal-steps", type=int, default=4)
    pc.add_argument("--n-act-steps", type=int, default=8)
    # training
    pc.add_argument("--steps", type=int, default=12000)
    pc.add_argument("--batch-size", type=int, default=4)
    pc.add_argument("--grad-accum", type=int, default=1)
    pc.add_argument("--lr", type=float, default=1e-4)
    pc.add_argument("--warmup-steps", type=int, default=400)
    pc.add_argument("--min-lr-ratio", type=float, default=0.02)
    pc.add_argument("--weight-decay", type=float, default=0.1)
    pc.add_argument("--grad-clip", type=float, default=1.0)
    pc.add_argument("--chunk", type=int, default=4096)
    pc.add_argument("--log-every", type=int, default=25)
    pc.add_argument("--ckpt-every", type=int, default=250)
    pc.add_argument("--num-workers", type=int, default=4)
    # RSSM objective + latent dims
    pc.add_argument("--beta", type=float, default=1.0, help="KL weight")
    pc.add_argument(
        "--free-bits", type=float, default=1.0, help="per-dim KL floor (nats)"
    )
    pc.add_argument(
        "--diagnostic-weight",
        type=float,
        default=0.5,
        help="weight on the per-stream diagnostic next-step CE (secondary)",
    )
    pc.add_argument(
        "--action-contrastive",
        action="store_true",
        help="turn ON the always-on action-contrastive InfoNCE on the latent — the "
        "realised state must match the TRUE-command prior rollout more than a "
        "WRONG-command one, keeping the command load-bearing (OFF = byte-identical)",
    )
    pc.add_argument(
        "--action-contrastive-weight",
        type=float,
        default=1.0,
        help="weight on the action-contrastive term in the total loss",
    )
    pc.add_argument("--h-dim", type=int, default=256, help="GRU hidden (det) state")
    pc.add_argument("--s-dim", type=int, default=32, help="stochastic latent state")
    pc.add_argument(
        "--keep-state-channels",
        action="store_true",
        help="condition on the FULL actuator vector incl plasma_current + density + "
        "tf (default MASKS those out — they are measured states, not commands)",
    )
    pc.add_argument(
        "--signal-modalities",
        choices=("default", "extended"),
        default="extended",
        help="'default' = the current measured-signal set; 'extended' (default) "
        "adds the already-tokenised HF + boundary streams (the 13-stream set)",
    )
    args = p.parse_args(argv)

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.command != "corpus":
        p.print_help()
        return 2

    held_out = {int(s) for s in args.eval_shots.split(",") if s.strip()}
    if args.shots.strip():
        train_shots = [
            int(s)
            for s in args.shots.split(",")
            if s.strip() and int(s) not in held_out
        ]
        win_list = None
    else:
        win_list = manifest_train_windows(
            _Path(args.manifest),
            held_out,
            target_horizon_s=args.target_horizon_s,
            n_frames=args.n_frames,
        )
        if args.n_shots and args.n_shots > 0:
            win_list = win_list[: args.n_shots]
        train_shots = [w.shot_id for w in win_list]
    if args.shots.strip() and args.n_shots and args.n_shots > 0:
        train_shots = train_shots[: args.n_shots]
    if len(train_shots) < 2:
        logger.error("need >= 2 train shots; got %d", len(train_shots))
        return 1
    overlap = set(train_shots) & held_out
    if overlap:
        logger.error("train/held-out overlap (NOT disjoint): %s", sorted(overlap))
        return 1
    logger.info(
        "rssm corpus train: %d %s (held-out %s DISJOINT) from %s [signals=%s]",
        len(train_shots),
        "excited windows" if win_list is not None else "train shots",
        sorted(held_out),
        args.manifest if not args.shots.strip() else "--shots",
        args.signal_modalities,
    )

    modalities = (
        extended_signal_modalities()
        if args.signal_modalities == "extended"
        else default_signal_modalities()
    )

    cfg = RSSMCorpusConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        chunk=args.chunk,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        num_workers=args.num_workers,
        n_signal_steps=args.n_signal_steps,
        n_act_steps=args.n_act_steps,
        beta=args.beta,
        free_bits=args.free_bits,
        diagnostic_weight=args.diagnostic_weight,
        action_contrastive=args.action_contrastive,
        action_contrastive_weight=args.action_contrastive_weight,
        h_dim=args.h_dim,
        s_dim=args.s_dim,
        drop_state_channels=not args.keep_state_channels,
        window=SpacetimeWindowConfig(
            n_frames=args.n_frames,
            n_plan=args.n_plan,
            context_frames=args.context_frames,
            frame_stride=args.frame_stride,
            target_horizon_s=args.target_horizon_s,
        ),
        modalities=modalities,
        init_checkpoint=_Path(args.init_checkpoint) if args.init_checkpoint else None,
    )
    result = train_rssm_corpus(
        train_shots,
        camera=args.camera,
        config=cfg,
        out_dir=_Path(args.out_dir) if args.out_dir else None,
        token_root=_Path(args.token_root) if args.token_root else None,
        manifest_windows=win_list,
    )
    logger.info(
        "rssm corpus train done: %d steps, init=%.4f final=%.4f, ckpt=%s",
        result.steps_run,
        result.initial_loss,
        result.final_loss,
        result.checkpoint_path,
    )
    return 0


__all__ = [
    "RSSMCorpusConfig",
    "RSSMCorpusResult",
    "build_rssm_model",
    "load_rssm_model_from_checkpoint",
    "main",
    "save_rssm_checkpoint",
    "train_rssm_corpus",
]


if __name__ == "__main__":
    raise SystemExit(main())
