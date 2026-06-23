"""Overfit + controllability GATE for the actuator-PLAN-conditioned camera model.

This is the cheap de-risking gate for the M4 PLAY bridge (plan
``playable-plasma-wm-v0``): does conditioning the camera model on the demanded
actuator PLAN make the controls causally LOAD-BEARING?  The first attempt
(prepended plan tokens) FAILED the gate — the model ignored the plan
(true-vs-zeroed margin ~0).  The control-conditioning survey
(``docs/control-conditioning-survey.html``) prescribes three matched fixes,
applied here TOGETHER:

* **camera-history bottleneck** (the corrected lever, highest priority) —
  independent per-frame corruption of the PAST FRAME EMBEDDINGS reaching the
  dynamics head (:mod:`imas_ambix.worldmodel.history_bottleneck`), so the
  predictable history no longer suffices and the plan must carry what it cannot.
  This REPLACES the prior gate's token-id ``context_corruption`` (which left the
  camera history near-clean — the wrong channel);
* **AdaLN-Zero per-layer plan conditioning** (in the model — the plan modulates
  every block, alpha zero-init) REPLACING the prepended plan tokens;
* **inverse-dynamics auxiliary** — predict the plan from consecutive latents
  (``inverse_dynamics_weight`` in the loss); we have 100% plan labels.

The gate metric is the Vid2World ΔN-M action-sensitivity: the decoded-rollout
(here token-rollout, a strict lower bound on the decoded response) divergence
between the TRUE plan and a RANDOM/shuffled plan on transient windows, scored
against a RANDOM-vs-RANDOM noise floor — PASS when true-vs-random clears the
floor by a clear margin.  (The legacy teacher-forced token-mismatch gate is kept
as a fast secondary check.)  CFG is kept as a test-time amplifier, used last.

GPU-safety (repo AGENTS.md §2b): model loaded once outside the per-item loop, a
SIGTERM/SIGINT STOP flag, ``try/finally`` releasing the model + empty_cache,
cudnn deterministic, bf16 autocast.  This is a 1-GPU gate (minutes), not the
6-GPU re-train.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from imas_ambix.worldmodel.actuator_plan import (
    ACTUATOR_CHANNEL_KEYS,
    N_ACTUATOR_CHANNELS,
    coil_current_channel_indices,
    find_transient_window,
    gas_puff_channel_indices,
    nbi_channel_indices,
    scale_plan_channels,
    zero_plan,
)
from imas_ambix.worldmodel.context_corruption import (
    ContextCorruptionConfig,
    apply_control_dropout,
    corrupt_context_tokens,
    sample_control_dropout,
)
from imas_ambix.worldmodel.controllable_dataset import (
    ControllableSpacetimeSample,
    assemble_controllable_window,
)
from imas_ambix.worldmodel.controllable_model import (
    ControllableSpacetimeConfig,
    ControllableSpacetimeTransformer,
)
from imas_ambix.worldmodel.history_bottleneck import (
    HistoryBottleneckConfig,
    sample_frame_strengths,
)
from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    plan_vocab,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    SignalModalitySpec,
    default_signal_modalities,
    extended_signal_modalities,
    stream_specs_from_modalities,
)
from imas_ambix.worldmodel.spacetime_train import (
    OverfitResult,
    _AutocastCtx,
    _set_determinism,
    _StopFlag,
)
from imas_ambix.worldmodel.spacetime_train_v2 import (
    _plan_channels_for,
    collate_signal_windows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model build (actuator-aware)
# ---------------------------------------------------------------------------


def architecture_from_checkpoint(checkpoint: Path) -> dict:
    """Read the backbone architecture dims from a v2 forecaster checkpoint.

    Returns ``{"d_model", "n_layers", "n_heads", "d_ff", "dropout",
    "corruption_levels"}`` from the checkpoint's ``model_config`` so the
    controllable model can be built to MATCH the forecaster (the warm start only
    loads tensors whose shapes match — a size mismatch loads nothing).  The gate
    driver applies these over its CLI model defaults when a checkpoint is given.
    """
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    mc = payload.get("model_config", {})
    keys = ("d_model", "n_layers", "n_heads", "d_ff", "dropout", "corruption_levels")
    return {k: mc[k] for k in keys if k in mc}


def _warm_start_from_forecaster(
    model: ControllableSpacetimeTransformer,
    checkpoint: Path,
    device: torch.device,
) -> tuple[int, int]:
    """Load the M2 forecaster weights into the controllable backbone (strict=False).

    The checkpoint is a v2 :class:`SignalSpacetimeTransformer` (corruption-capable)
    state dict.  Its ``blocks.N.{attn_s,attn_t,mlp}`` + the token / spatial /
    temporal / plan / signal / corruption embeddings + the tied head all load by
    NAME into the controllable model; the AdaLN blocks' affine-free norms have no
    ``ln_*.weight``/``bias`` so those checkpoint keys are dropped, and the new
    AdaLN MLP + inverse-dynamics head are not in the checkpoint (left at init — the
    zero-init gate makes the plan a no-op so the warm start IS the forecaster).
    Returns ``(n_tensors_loaded, n_new_tensors)``.
    """
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    own = model.state_dict()
    loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    # Positional tables (``frame_embed.weight``) may be LARGER in the target than
    # in the checkpoint when the horizon is extended (more camera frames → bigger
    # max_frames).  A strict shape match would DROP the table entirely and the
    # learned temporal positions would restart at random.  Instead PARTIAL-COPY:
    # the checkpoint's rows fill the low (absolute-position) indices — prefix +
    # the frames the checkpoint already saw — and the new high indices (the
    # later frames the extended horizon adds) keep their fresh init.  Rows are
    # absolute sequence positions, so this preserves the learned early-frame
    # positions and only the genuinely-new later positions start fresh.
    n_extended = 0
    for k in ("frame_embed.weight",):
        if k in own and k in state and k not in loadable:
            src, dst = state[k], own[k]
            if src.dim() == dst.dim() == 2 and src.shape[1] == dst.shape[1]:
                rows = min(src.shape[0], dst.shape[0])
                merged = dst.clone()
                merged[:rows] = src[:rows].to(merged.dtype)
                loadable[k] = merged
                n_extended += 1
                logger.info(
                    "warm-start %s: partial-copy %d/%d rows (checkpoint %d -> "
                    "target %d), high positions kept at init",
                    k,
                    rows,
                    dst.shape[0],
                    src.shape[0],
                    dst.shape[0],
                )
    missing, _unexpected = model.load_state_dict(loadable, strict=False)
    model.to(device)
    n_new = sum(1 for k in own if k not in loadable)
    return len(loadable), n_new


def build_controllable_model(
    window: SpacetimeWindowConfig,
    *,
    plan_channels: int,
    signal_streams: Sequence[SignalStreamSpec],
    n_signal_steps: int,
    actuator_channels: int = N_ACTUATOR_CHANNELS,
    n_act_steps: int = 8,
    max_frames: int | None = None,
    corruption_levels: int = 0,
    inverse_dynamics: bool = True,
    generate_diagnostics: bool = True,
    cross_modal: bool = False,
    self_predictive: bool = False,
    action_contrastive: bool = False,
    masked_command_indices: tuple[int, ...] = (),
    **model_kwargs: object,
) -> ControllableSpacetimeTransformer:
    """Build a :class:`ControllableSpacetimeTransformer` sized to window + streams.

    The actuator plan is injected via per-block AdaLN (NOT prepended as temporal
    frames), so ``max_frames`` only needs to cover the tokenised plan steps +
    every present signal stream's steps + the camera frames; it defaults to
    ``len(streams)*n_signal_steps + n_plan + n_frames`` with slack.

    ``masked_command_indices`` are the actuator-vector columns the model ZEROES
    before conditioning (the measured states Ip/density + tf) — see
    :func:`masked_command_columns`; empty conditions on the full vector.
    """
    n_streams = len(signal_streams)
    if max_frames is None:
        max_frames = (
            n_streams * int(n_signal_steps) + window.n_plan + window.n_frames + 2
        )
    cfg = ControllableSpacetimeConfig(
        max_frames=int(max_frames),
        plan_vocab=plan_vocab(),
        plan_channels=int(plan_channels),
        signal_streams=tuple(signal_streams),
        n_signal_steps=int(n_signal_steps),
        corruption_levels=int(corruption_levels),
        actuator_channels=int(actuator_channels),
        n_act_steps=int(n_act_steps),
        inverse_dynamics=bool(inverse_dynamics),
        generate_diagnostics=bool(generate_diagnostics),
        cross_modal=bool(cross_modal),
        self_predictive=bool(self_predictive),
        action_contrastive=bool(action_contrastive),
        masked_command_indices=tuple(int(i) for i in masked_command_indices),
        **model_kwargs,  # type: ignore[arg-type]
    )
    return ControllableSpacetimeTransformer(cfg)


def _controllable_config_to_dict(cfg: ControllableSpacetimeConfig) -> dict:
    """Serialise the FULL controllable config (v2 fields + the actuator/AdaLN ones).

    The v2 ``save_checkpoint_v2`` only records the v2 fields, so a controllable
    model reloaded from a v2-style ``model_config`` would lose ``actuator_channels``
    / ``n_act_steps`` / ``adaln_hidden`` / ``inverse_dynamics`` /
    ``masked_command_indices`` and could not be rebuilt.  This records them all so
    :func:`load_controllable_model_from_checkpoint` reconstructs the IDENTICAL
    model (the masked-command set included, so the eval conditions exactly as the
    trained model did).
    """
    from imas_ambix.worldmodel.spacetime_train_v2 import (  # noqa: PLC0415
        _config_to_dict,
    )

    d = _config_to_dict(cfg)
    d.update(
        {
            "actuator_channels": int(cfg.actuator_channels),
            "n_act_steps": int(cfg.n_act_steps),
            "adaln_hidden": int(cfg.adaln_hidden),
            "inverse_dynamics": bool(cfg.inverse_dynamics),
            "inv_dyn_hidden": int(cfg.inv_dyn_hidden),
            "masked_command_indices": list(cfg.masked_command_indices),
            "timescale_conditioning": bool(cfg.timescale_conditioning),
            "timescale_hidden": int(cfg.timescale_hidden),
            "camera_conditioning": bool(cfg.camera_conditioning),
            "generate_diagnostics": bool(cfg.generate_diagnostics),
            "cross_modal": bool(cfg.cross_modal),
            "self_predictive": bool(cfg.self_predictive),
            "action_contrastive": bool(cfg.action_contrastive),
            "contrastive_dim": int(cfg.contrastive_dim),
        }
    )
    return d


def save_controllable_checkpoint(
    out_dir,
    *,
    model: ControllableSpacetimeTransformer,
    optimizer,
    step: int,
    window: SpacetimeWindowConfig,
    extra: dict | None = None,
    name: str = "latest.pt",
    snapshot: bool = True,
):
    """Atomic self-describing controllable checkpoint (full config + state).

    Like ``save_checkpoint_v2`` but writes the FULL controllable ``model_config``
    (see :func:`_controllable_config_to_dict`) so the model round-trips exactly.
    """
    import os  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "model_config": _controllable_config_to_dict(model.config),
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


def load_controllable_model_from_checkpoint(
    path, *, map_location: str = "cpu"
) -> tuple[ControllableSpacetimeTransformer, dict]:
    """Rebuild a controllable model from a checkpoint + load its weights (eval).

    Reconstructs the FULL :class:`ControllableSpacetimeConfig` (signal streams +
    the actuator/AdaLN fields + the masked-command set) so the eval model
    conditions EXACTLY as the trained one.  The eval harness uses this to load the
    re-train checkpoint for the held-out ΔN-M + dream GIFs.
    """
    from imas_ambix.worldmodel.spacetime_model_v2 import (  # noqa: PLC0415
        SignalStreamSpec,
    )

    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    d = dict(payload["model_config"])
    streams = tuple(
        SignalStreamSpec(
            name=str(s["name"]), vocab=int(s["vocab"]), channels=int(s["channels"])
        )
        for s in d.get("signal_streams", [])
    )
    scalar = {
        k: d[k]
        for k in d
        if k in ControllableSpacetimeConfig.__dataclass_fields__
        and k not in ("signal_streams", "masked_command_indices")
    }
    cfg = ControllableSpacetimeConfig(
        signal_streams=streams,
        masked_command_indices=tuple(d.get("masked_command_indices", ())),
        **scalar,
    )
    model = ControllableSpacetimeTransformer(cfg)
    # strict=False: a camera-only baseline checkpoint has NO ``diagnostic_heads.*``
    # (joint generation was added later) — strict=True would crash loading it; and a
    # NEW generative checkpoint loading into a diagnostics-OFF eval model would have
    # those keys unexpected.  Tolerate both and log the shape (mirrors the
    # warm-start, which is already strict=False).
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"], strict=False
    )
    if missing or unexpected:
        logger.info(
            "loaded controllable checkpoint with strict=False: %d missing, "
            "%d unexpected keys (e.g. diagnostic_heads.* on a camera-only baseline)",
            len(missing),
            len(unexpected),
        )
    model.to(map_location)
    return model, payload


# ---------------------------------------------------------------------------
# Collate (actuator plan + the v2 signal/plan/frames batch)
# ---------------------------------------------------------------------------


def collate_actuator(samples: Sequence[ControllableSpacetimeSample]) -> dict:
    """Stack the per-sample actuator plans into ``{"values","missing"}`` tensors.

    Every sample carries the SAME actuator channel set + step count (the dataset
    sub-samples to a fixed ``n_act_steps``), so the stack is rectangular.  A
    sample whose level-1 store was unreadable contributes an all-missing plan
    (values 0, missing 1) — the model's actuator encoder then sees a zero drive
    with the missing flag set.
    """
    vals = np.stack(
        [np.asarray(s.actuator.values, dtype=np.float32) for s in samples]
    )  # (B, P, C)
    miss = np.stack(
        [np.asarray(s.actuator.missing, dtype=np.float32) for s in samples]
    )  # (B, P, C)
    return {
        "values": torch.as_tensor(vals, dtype=torch.float32),
        "missing": torch.as_tensor(miss, dtype=torch.float32),
    }


def collate_timescale_camera(samples: Sequence[ControllableSpacetimeSample]) -> dict:
    """Per-window log-Δt + camera index for the timescale / camera conditioning.

    ``frame_log_dt`` is ``(B, T)`` per-camera-frame log-Δt offset (centred on the
    reference cadence) derived from each sample's ``frame_time``; ``camera_id`` is
    ``(B,)`` long, the stable embedding index of each sample's camera (unknown →
    the reference camera).  Both are always built (cheap metadata) — the model
    consumes them only when timescale / camera conditioning is enabled, so an
    OFF model ignores them and is byte-identical.  Frames stack rectangularly, so
    the per-window Δt rows are padded/truncated to the batch frame count.
    """
    from imas_ambix.worldmodel.timescale_conditioning import (  # noqa: PLC0415
        camera_index,
        frame_dt_seconds,
        log_dt_offset,
    )

    t = int(samples[0].frames.shape[0]) if samples else 0
    log_dt = np.zeros((len(samples), t), dtype=np.float32)
    for i, s in enumerate(samples):
        off = log_dt_offset(frame_dt_seconds(s.frame_time))
        off = np.asarray(off, dtype=np.float32)
        k = min(off.shape[0], t)
        log_dt[i, :k] = off[:k]
    cam_idx = np.asarray([camera_index(s.camera) for s in samples], dtype=np.int64)
    return {
        "frame_log_dt": torch.as_tensor(log_dt, dtype=torch.float32),
        "camera_id": torch.as_tensor(cam_idx, dtype=torch.long),
    }


def collate_controllable_windows(
    samples: Sequence[ControllableSpacetimeSample],
    *,
    stream_names: Sequence[str] | None = None,
) -> dict:
    """Stack frames + plan + measured signals + the actuator plan into a batch.

    Also carries ``frame_log_dt`` (per-frame log-Δt) + ``camera_id`` (the view
    index) so a timescale / camera-conditioned model can read each window's
    cadence + camera; an OFF model ignores them (byte-identical).
    """
    signal_samples = [s.signal for s in samples]
    batch = collate_signal_windows(signal_samples, stream_names=stream_names)
    batch["actuator"] = collate_actuator(samples)
    batch.update(collate_timescale_camera(samples))
    return batch


def _batch_to(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["frames"] = batch["frames"].to(device, non_blocking=True)
    out["plan"] = batch["plan"].to(device, non_blocking=True)
    out["signals"] = {
        k: v.to(device, non_blocking=True) for k, v in batch["signals"].items()
    }
    # CLEAN diagnostic targets (when a caller supplied them pre-mask in the raw
    # batch — the trainer normally derives them post-move, so this is a no-op then).
    st = batch.get("signal_targets")
    if st is not None:
        out["signal_targets"] = {
            k: v.to(device, non_blocking=True) for k, v in st.items()
        }
    act = batch.get("actuator")
    if act is not None:
        out["actuator"] = {
            "values": act["values"].to(device, non_blocking=True),
            "missing": act["missing"].to(device, non_blocking=True),
        }
    for k in ("frame_log_dt", "camera_id"):
        if batch.get(k) is not None:
            out[k] = batch[k].to(device, non_blocking=True)
    return out


def _drop_observations(
    signals: dict[str, torch.Tensor] | None,
    drop: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Zero the measured-signal blocks for the samples flagged in ``drop``.

    The OPTIONAL-observation dropout: zeroing a sample's measured-signal tokens
    to PAD id 0 leaves the actuator plan (the drive surface) intact, so the model
    must predict that sample's camera from the PLAN alone.  Distinct from
    control-dropout (which zeroes EVERYTHING for classifier-free guidance).  A
    ``None`` / empty signals dict passes through.
    """
    if signals is None or not signals or not bool(drop.any()):
        return signals
    out: dict[str, torch.Tensor] = {}
    for name, block in signals.items():
        nb = block.clone()
        if nb.numel() and nb.shape[1] > 0:
            nb[drop] = 0
        out[name] = nb
    return out


def _mask_observations_per_stream(
    signals: dict[str, torch.Tensor] | None,
    rate: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """Per-stream INDEPENDENT observation masking — zero (→PAD) flagged streams.

    For EACH measured-signal stream independently, draw a per-sample Bernoulli
    ``(rate)`` mask of shape ``(B,)`` and zero (to PAD id 0) that stream's tokens
    for the flagged samples.  Independent draws PER STREAM, so two streams get
    DIFFERENT masks on the same sample — each stream is a diagnostic-CE target on
    ~``(1-rate)`` of steps (full coverage), and on the masked steps the model must
    DREAM that stream from the OTHER streams + commands + cameras.

    Distinct from :func:`_drop_observations`, which zeroes the WHOLE signal dict
    for a flagged sample (all-or-nothing — the ablation path).  A masked stream's
    frames STAY in the sequence (all-PAD embeddings), so the model still produces
    latents for it that the diagnostic head decodes and scores against the CLEAN
    target the trainer keeps in ``signal_targets``.

    Returns a NEW masked dict (the input is never mutated — each block is cloned).
    A ``None`` / empty signals dict passes through.
    """
    if signals is None or not signals:
        return signals
    out: dict[str, torch.Tensor] = {}
    for name, block in signals.items():
        nb = block.clone()
        if nb.numel() and nb.shape[1] > 0:
            b = int(nb.shape[0])
            drop = torch.rand(b, generator=generator, device=device) < float(rate)
            if bool(drop.any()):
                nb[drop] = 0
        out[name] = nb
    return out


def _scheduled_sampling_prob(step: int, total_steps: int, cfg) -> float:
    """Linear ramp of the scheduled-sampling probability from 0 to the max.

    0 for the first instant, rising linearly to ``scheduled_sampling_max`` at
    ``scheduled_sampling_ramp`` of training, then held at the max — the standard
    scheduled-sampling schedule (Bengio et al. 2015): start fully teacher-forced
    (the model can't predict its own frames yet), hand it more of its own
    predictions as it learns.
    """
    pmax = float(getattr(cfg, "scheduled_sampling_max", 0.0))
    if pmax <= 0.0 or total_steps <= 1:
        return 0.0
    ramp = max(1e-6, float(getattr(cfg, "scheduled_sampling_ramp", 0.5)))
    frac = (step / float(total_steps - 1)) / ramp
    return float(min(max(frac, 0.0), 1.0) * pmax)


def _diagnostic_weight(step: int, total_steps: int, cfg) -> float:
    """Linear warmup of the joint-generation diagnostic-CE weight from 0 to the max.

    Returns 0 when joint generation is OFF (``generate_diagnostics`` False) or the
    target ``diagnostic_weight`` is 0.  Otherwise ramps linearly from 0 at step 0
    to ``diagnostic_weight`` at ``diagnostic_weight_warmup_frac`` of training, then
    holds at the max.  The diagnostic CE can destabilise the camera loss early, so
    it earns its weight gradually (mirrors :func:`_scheduled_sampling_prob`).
    """
    if not bool(getattr(cfg, "generate_diagnostics", False)):
        return 0.0
    wmax = float(getattr(cfg, "diagnostic_weight", 0.0))
    if wmax == 0.0:
        return 0.0
    if total_steps <= 1:
        return wmax
    frac = float(getattr(cfg, "diagnostic_weight_warmup_frac", 0.0))
    if frac <= 0.0:
        return wmax
    progress = (step / float(total_steps - 1)) / frac
    return float(min(max(progress, 0.0), 1.0) * wmax)


@torch.no_grad()
def _scheduled_sampling_mix(
    model: ControllableSpacetimeTransformer,
    step_batch: dict,
    clean_frames: torch.Tensor,
    *,
    context_frames: int,
    prob: float,
    chunk: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace FORECAST-window context frames with the model's own predictions.

    Runs ONE no-grad teacher-forced forward over the current ``step_batch`` (the
    history bottleneck / dropouts already applied), argmax-decodes every frame,
    and for each forecast-window frame ``t >= context_frames`` replaces its tokens
    with the model's prediction-of-``t`` (the argmax of ``hidden[:, t-1]``) with
    per-(sample, frame) probability ``prob``.  Returns the mixed ``frames`` tensor
    to feed the grad forward; the loss TARGET stays ``clean_frames``.  At
    ``prob == 0`` returns the input frames unchanged.

    This makes the next-frame loss + inverse-dynamics depend on the model's OWN
    generated dynamics over the forecast window (rollout-in-the-loop), so a
    load-bearing plan has to steer the GENERATED trajectory, not just the one-step
    teacher-forced prediction — the gap the de-risk gate exposed.
    """
    frames = step_batch["frames"]
    if prob <= 0.0 or frames.ndim != 3:
        return frames
    b, t, s = frames.shape
    ctx = int(max(0, min(context_frames, t)))
    if ctx >= t:
        return frames
    model.eval()
    with _AutocastCtx(frames.device):
        hidden = model._forward_tokens(
            frames,
            step_batch.get("plan"),
            step_batch.get("signals"),
            step_batch.get("corruption_level"),
            actuator=step_batch.get("actuator"),
            context_frames=ctx,
            frame_log_dt=step_batch.get("frame_log_dt"),
            camera_id=step_batch.get("camera_id"),
        )
    model.train()
    mixed = frames.clone()
    # predict each forecast frame t (>= ctx) from hidden[:, t-1] and splice it in
    # with per-(sample, frame) Bernoulli(prob).
    for ti in range(ctx, t):
        pred = model.chunked_argmax_frame(hidden[:, ti - 1], chunk=chunk)  # (b, s)
        take = torch.rand(b, generator=generator, device=frames.device) < prob  # (b,)
        if bool(take.any()):
            mixed[take, ti] = pred[take]
    return mixed


@torch.no_grad()
def _excitation_frame_weights(
    actuator: dict | None,
    *,
    n_frames: int,
    coil_cols: Sequence[int],
    device: torch.device,
    floor: float,
    power: float,
) -> torch.Tensor | None:
    """Per-target-frame excitation weight from coil |dI/dt| — ``(B, T-1)``.

    ``actuator["values"]`` is the NORMALISED actuator vector ``(B, P_a, C)`` at the
    plan resolution (``P_a = n_act_steps``), coarser than the ``T = n_frames``
    camera frames.  We linearly resample the COIL columns from ``P_a`` to ``T``
    over the shared window-time grid, take the frame-to-frame absolute change
    summed over coil channels, and map it to a per-frame weight aligned to TARGET
    frames ``1..T-1`` (``weight[t]`` scores the transition INTO frame ``t``):

        raw[t]   = sum_c |coil_resampled[t] - coil_resampled[t-1]|     # (B, T-1)
        norm     = raw / per-sample max(raw)            # 0..1, robust to scale
        weight   = floor + (1 - floor) * norm**power    # floor..1

    A flat-top frame (coils still) keeps the ``floor`` baseline (it still teaches
    persistence); a frame where the coils swing gets weight → 1.  Returns ``None``
    when there is no actuator / no coil column / a degenerate (single-step) plan,
    so the caller falls back to the uniform loss.  All-quiet windows (max ≈ 0)
    return the uniform ``floor + (1-floor)*0`` — i.e. a constant, which the
    weighted mean treats identically to uniform.
    """
    if actuator is None:
        return None
    vals = actuator.get("values")
    if vals is None or vals.ndim != 3 or vals.shape[1] < 2:
        return None
    cols = [int(c) for c in coil_cols if 0 <= int(c) < vals.shape[2]]
    if not cols:
        return None
    t = int(n_frames)
    if t < 2:
        return None
    coil = vals[:, :, cols].to(torch.float32)  # (B, P_a, n_coil)
    b, p_a, _ = coil.shape
    # resample P_a -> T along the time axis (linear), via 1d interpolation.  Move
    # the time axis last for F.interpolate, then back.
    coil_bt = coil.permute(0, 2, 1)  # (B, n_coil, P_a)
    coil_rt = torch.nn.functional.interpolate(
        coil_bt, size=t, mode="linear", align_corners=True
    )  # (B, n_coil, T)
    coil_rt = coil_rt.permute(0, 2, 1)  # (B, T, n_coil)
    dcoil = (coil_rt[:, 1:t] - coil_rt[:, : t - 1]).abs().sum(dim=-1)  # (B, T-1)
    peak = dcoil.amax(dim=1, keepdim=True).clamp_min(1e-8)
    norm = (dcoil / peak).clamp(0.0, 1.0)
    f = float(max(0.0, min(1.0, floor)))
    w = f + (1.0 - f) * norm.pow(float(max(1e-3, power)))
    return w.to(device)


# ---------------------------------------------------------------------------
# Overfit (the GATE training phase)
# ---------------------------------------------------------------------------


@dataclass
class OverfitControllableConfig:
    steps: int = 600
    lr: float = 3e-4
    seed: int = 0
    log_every: int = 50
    chunk: int = 4096
    n_signal_steps: int = 4
    n_act_steps: int = 8
    # HIGH observation-dropout: zero the measured signals on this fraction of
    # steps so the model cannot shortcut the control->camera map via the
    # redundant realised observations and must learn to drive from the PLAN.
    observation_dropout: float = 0.8
    # PER-MODALITY masking: when True (default) each measured-signal stream is
    # masked INDEPENDENTLY (per-stream Bernoulli(observation_dropout)) and the
    # CLEAN pre-mask signals are kept as the diagnostic-CE TARGET, so every stream
    # is a CE target on ~(1-rate) of steps (full coverage) and the model learns to
    # DREAM a masked stream from the others + commands + cameras.  When False the
    # legacy ALL-OR-NOTHING ``_drop_observations`` zeroes the whole signal dict for
    # a flagged sample (the ablation path — the diagnostic CE is then 0 on ~rate of
    # steps).  ``observation_dropout`` is the per-stream rate either way.
    per_modality_masking: bool = True
    # control-dropout (CFG): zero the WHOLE conditioning (plan + actuator +
    # signals) on this fraction of steps so classifier-free guidance works.
    control_dropout: float = 0.15
    # TRANSIENT-window selection: pick each shot's window where the actuator
    # PLAN varies most (a ramp / toggle) rather than the centred flat-top window.
    # The controllability gate is only fair where the plan actually moves — a
    # flat-top window has no control variation to learn from or respond to, so
    # the gate would FALSELY fail.  Off => the centred window (debug).
    transient_windows: bool = True
    # CAMERA-HISTORY BOTTLENECK during overfit — the corrected controllability
    # lever (the prior gate's key miss).  The earlier recipe corrupted token IDS
    # of the context frames, which left the camera history near-clean (a replaced
    # id on a 2**18 codebook still embeds to a vector; at low rates most of the
    # frame survives) so the model coasted on the predictable history and ignored
    # the actuator plan (latent-action collapse).  This noises/masks the past
    # FRAME EMBEDDINGS — the channel the temporal attention actually reads —
    # independently PER FRAME (Diffusion Forcing), so the predictable history no
    # longer suffices and the plan must carry what it cannot.  The loss is always
    # scored against the CLEAN frames.  ``history_bottleneck.enabled == False``
    # (e.g. max_strength 0) disables it.
    history_bottleneck: HistoryBottleneckConfig = field(
        default_factory=HistoryBottleneckConfig
    )
    # INVERSE-DYNAMICS auxiliary weight — predict the plan from consecutive camera
    # latents (Schmidt & Jiang Prop 4.4 forces the action into the latent).  We
    # have 100% plan labels so it is cheap; 0 disables it.
    inverse_dynamics_weight: float = 1.0
    # JOINT GENERATION — grow per-stream diagnostic-prediction heads + a next-step
    # cross-entropy on the measured-signal tokens, so the model dreams the cameras
    # AND the diagnostics (a joint state model, not a camera predictor that merely
    # READS the diagnostics).  ``generate_diagnostics`` is the ablation off-switch
    # (no heads built, no objective); ``diagnostic_weight`` weights the diagnostic
    # CE in the loss.  The CE can destabilise the camera loss early, so it is warmed
    # up: a linear 0 -> diagnostic_weight ramp over ``diagnostic_weight_warmup_frac``
    # of total steps.
    generate_diagnostics: bool = True
    diagnostic_weight: float = 0.5
    diagnostic_weight_warmup_frac: float = 0.1
    # AUXILIARY contrastive objectives (default OFF — the §6 ablation turns each on).
    # Each builds its own head and adds ``<name>_weight * term`` to the loss; the
    # bool is the ablation off-switch (no head built, no objective) and the weight
    # scales the term.  cross_modal = camera<->diagnostic InfoNCE (needs signals);
    # self_predictive = BYOL/SPR latent self-prediction (no negatives);
    # action_contrastive = true-vs-random next-state margin (needs the actuator
    # drive + a second forward under a random plan — built via
    # ``_random_actuator_like`` and passed as ``actuator_random``).
    cross_modal: bool = False
    cross_modal_weight: float = 0.1
    self_predictive: bool = False
    self_predictive_weight: float = 0.1
    action_contrastive: bool = False
    action_contrastive_weight: float = 0.1
    # SCHEDULED SAMPLING / rollout-in-the-loop — closes the 1-step->rollout gap (the
    # de-risk's visible failure mode: the plan moved the teacher-forced one-step
    # prediction but NOT a free-running rollout).  With probability ramping from 0
    # to ``scheduled_sampling_max`` over ``scheduled_sampling_ramp`` fraction of
    # training, each FORECAST-window frame is replaced (for CONTEXT purposes) by
    # the model's OWN argmax prediction before the grad forward, so the next-frame
    # loss + inverse-dynamics are enforced on GENERATED dynamics — where the plan
    # must actually steer.  The loss TARGET stays the clean frames.  0 disables it
    # (pure teacher forcing).  Costs one extra no-grad forward on the steps it
    # fires (Bengio et al. 2015 scheduled sampling, batched two-pass form).
    scheduled_sampling_max: float = 0.0
    scheduled_sampling_ramp: float = 0.5
    # MASK the measured states (Ip + density + tf) out of the model's conditioning
    # (the single _plan_summary entry point) so it drives from the COMMANDS and
    # cannot read the plasma state off an always-on Ip context.  True is the
    # controllability-correct default.
    drop_state_channels: bool = True
    window: SpacetimeWindowConfig = field(default_factory=SpacetimeWindowConfig)
    model_kwargs: dict = field(default_factory=dict)
    modalities: list[SignalModalitySpec] = field(
        default_factory=default_signal_modalities
    )


def overfit_controllable(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    config: OverfitControllableConfig | None = None,
    token_root: Path | None = None,
    device: str | None = None,
    init_checkpoint: Path | None = None,
) -> tuple[
    OverfitResult,
    ControllableSpacetimeTransformer,
    list[ControllableSpacetimeSample],
    list[str],
]:
    """Overfit a handful of shots conditioned on the actuator PLAN — the GATE.

    Applies the camera-history bottleneck (so the model cannot coast on the
    predictable past) + the inverse-dynamics auxiliary + AdaLN-Zero plan
    conditioning + control-dropout (CFG).  Returns ``(result, model, samples,
    stream_names)``; the caller then runs :func:`delta_nm_gate` to PROVE the
    actuator plan is causally load-bearing.

    ``init_checkpoint`` (optional) warm-starts the backbone from the M2 FORECASTER
    checkpoint — the AdaLN space-time blocks reuse the v2 block's attention + MLP
    weights by name, and the new AdaLN MLP + inverse-dynamics head start at init
    (the zero-init gate makes that a no-op for the forecaster), so the redesign is
    tested as an ADD-ON to the trained forecaster rather than from scratch.
    """
    config = config or OverfitControllableConfig()
    _set_determinism(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stop = _StopFlag()
    stop.install()

    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        window_span_for_shot,
    )

    samples: list[ControllableSpacetimeSample] = []
    window_info: list[tuple[int, int | None, float]] = []
    for sid in shot_ids:
        start_frame = None
        var_score = 0.0
        if config.transient_windows:
            # span the ~target_horizon_s window (per-shot stride) so the transient
            # scan finds the excited region over the horizon, not a ~15ms slice.
            span = window_span_for_shot(
                int(sid), config.window, camera=camera, token_root=token_root
            )
            start_frame, var_score = find_transient_window(
                int(sid), span, camera=camera, token_root=token_root
            )
        samples.append(
            assemble_controllable_window(
                int(sid),
                config.window,
                config.modalities,
                config.n_signal_steps,
                config.n_act_steps,
                camera=camera,
                token_root=token_root,
                start_frame=start_frame,
            )
        )
        window_info.append((int(sid), start_frame, var_score))
    if config.transient_windows:
        logger.info(
            "TRANSIENT windows (actuator-plan variation per shot): %s",
            [(sid, sf, round(score, 4)) for sid, sf, score in window_info],
        )
        n_flat = sum(1 for _, sf, _ in window_info if sf is None)
        if n_flat:
            logger.warning(
                "%d/%d shots had a FLAT actuator plan (no transient window found) "
                "— they fell back to the centred window and weaken the gate",
                n_flat,
                len(window_info),
            )
    signal_samples = [s.signal for s in samples]
    plan_ch = _plan_channels_for(signal_samples)
    channels: dict[str, int] = {}
    for s in signal_samples:
        for name, arr in s.signals.items():
            channels[name] = max(channels.get(name, 0), int(arr.shape[1]))
    streams = stream_specs_from_modalities(config.modalities, channels)
    stream_names = [st.name for st in streams]
    act_channels = int(samples[0].actuator.n_channels)

    batch = _batch_to(
        collate_controllable_windows(samples, stream_names=stream_names), dev
    )

    # The history bottleneck quantises its per-frame strength to bins that feed
    # the model's history-corruption LEVEL embedding, so size the model's levels
    # to the bottleneck's (when the bottleneck is enabled).
    hb = config.history_bottleneck
    corruption_levels = int(hb.levels) if hb.enabled else 0
    masked_idx = (
        masked_command_columns(list(samples[0].actuator.channel_keys))
        if config.drop_state_channels
        else ()
    )
    model = build_controllable_model(
        config.window,
        plan_channels=plan_ch,
        signal_streams=streams,
        n_signal_steps=config.n_signal_steps,
        actuator_channels=act_channels,
        n_act_steps=config.n_act_steps,
        corruption_levels=corruption_levels,
        inverse_dynamics=config.inverse_dynamics_weight > 0.0,
        generate_diagnostics=config.generate_diagnostics,
        cross_modal=config.cross_modal,
        self_predictive=config.self_predictive,
        action_contrastive=config.action_contrastive,
        masked_command_indices=masked_idx,
        **config.model_kwargs,
    ).to(dev)
    if init_checkpoint is not None:
        n_loaded, n_new = _warm_start_from_forecaster(model, init_checkpoint, dev)
        logger.info(
            "warm-started backbone from %s: %d tensors loaded, %d new (AdaLN MLP "
            "+ inverse-dynamics head + affine-free-norm drop) left at init",
            init_checkpoint,
            n_loaded,
            n_new,
        )
    model.train()
    logger.info(
        "overfit-controllable on %s: params=%d (%.1fM) n_frames=%d plan_ch=%d "
        "actuator_ch=%d n_act_steps=%d obs_dropout=%.2f history_bottleneck=%s "
        "inv_dyn_w=%.2f streams=%s shots=%s",
        dev,
        model.num_parameters(),
        model.num_parameters() / 1e6,
        config.window.n_frames,
        plan_ch,
        act_channels,
        config.n_act_steps,
        config.observation_dropout,
        f"std={hb.noise_std} mask={hb.mask_prob} max={hb.max_strength}"
        if hb.enabled
        else "off",
        config.inverse_dynamics_weight,
        [(st.name, st.channels) for st in streams],
        list(shot_ids),
    )

    # per-step RNG for the dropout draws (reproducible).
    gen = torch.Generator(device=dev)
    drop_cfg = ContextCorruptionConfig(control_dropout=config.control_dropout)
    # action-contrastive needs the perturbable COMMAND columns (coils+sol+nbi+gas)
    # to build the WRONG-plan negative batch; resolve them by KEY (correct for the
    # filtered command vector too).
    perturbable_cols = (
        _perturbable_command_columns(list(samples[0].actuator.channel_keys))
        if config.action_contrastive
        else []
    )
    clean_frames = batch["frames"]
    ctx_frames = int(config.window.context_frames)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    losses: list[float] = []
    b = int(batch["frames"].shape[0])
    try:
        for step in range(config.steps):
            if stop.stop:
                logger.warning("STOP — ending overfit at step %d", step)
                break
            gen.manual_seed((config.seed * 7919) ^ step)
            step_batch = dict(batch)
            # CAMERA-HISTORY BOTTLENECK (the corrected lever): noise/mask the
            # context-frame EMBEDDINGS (independent per frame) so the model cannot
            # coast on a predictable camera history and MUST lean on the actuator
            # plan (which now modulates every block via AdaLN).  The model embeds
            # the CLEAN frames then corrupts the embeddings internally; the loss is
            # scored against the CLEAN frames.  We pass the per-frame strengths +
            # their bins (the bins drive the model's history-corruption level
            # embedding so it can condition on how corrupt its history is).
            if hb.enabled:
                hb_gen = torch.Generator(device=dev).manual_seed(
                    (config.seed * 104729) ^ step
                )
                strengths, bins = sample_frame_strengths(
                    b, ctx_frames, hb, generator=hb_gen, device=dev
                )
                step_batch["history_bottleneck"] = hb
                step_batch["history_strengths"] = strengths
                step_batch["history_generator"] = torch.Generator(
                    device=dev
                ).manual_seed((config.seed * 1299709) ^ step)
                step_batch["target_frames"] = clean_frames
                # the per-sample level bin = the MAX per-frame strength bin (a
                # scalar summary the corruption-level embedding conditions on).
                step_batch["corruption_level"] = bins.max(dim=1).values
            # OPTIONAL-observation masking: zero the measured signals on a high
            # fraction of samples so the model drives from the PLAN.  Keep a CLEAN
            # (pre-mask) reference as the diagnostic-CE target so the model learns
            # to DREAM a masked diagnostic, then set the (masked) model input.
            step_batch["signal_targets"] = (
                dict(batch["signals"]) if batch.get("signals") else batch.get("signals")
            )
            if config.per_modality_masking:
                # INDEPENDENT per-stream masking (full per-stream CE coverage).
                step_batch["signals"] = _mask_observations_per_stream(
                    batch["signals"],
                    config.observation_dropout,
                    generator=gen,
                    device=dev,
                )
            else:
                # ALL-OR-NOTHING (the ablation path).
                obs_drop = (
                    torch.rand(b, generator=gen, device=dev)
                    < config.observation_dropout
                )
                step_batch["signals"] = _drop_observations(batch["signals"], obs_drop)
            # control-dropout (CFG): zero the WHOLE conditioning on a fraction of
            # samples — plan + actuator + the (already obs-dropped) signals.
            ctrl_drop = sample_control_dropout(b, drop_cfg, generator=gen, device=dev)
            if bool(ctrl_drop.any()):
                plan2, signals2 = apply_control_dropout(
                    batch.get("plan"), step_batch["signals"], ctrl_drop
                )
                if plan2 is not None:
                    step_batch["plan"] = plan2
                if signals2 is not None:
                    step_batch["signals"] = signals2
                # zero the actuator drive for the dropped samples too.
                act = batch.get("actuator")
                if act is not None:
                    av = act["values"].clone()
                    am = act["missing"].clone()
                    av[ctrl_drop] = 0.0
                    step_batch["actuator"] = {"values": av, "missing": am}

            # SCHEDULED SAMPLING: splice the model's OWN predictions into the
            # forecast-window context (rollout-in-the-loop) so the loss +
            # inverse-dynamics are enforced on GENERATED dynamics.  The TARGET stays
            # the clean frames; only the model-input frames are mixed.
            ss_prob = _scheduled_sampling_prob(step, config.steps, config)
            if ss_prob > 0.0:
                ss_gen = torch.Generator(device=dev).manual_seed(
                    (config.seed * 2_750_159) ^ step
                )
                step_batch["frames"] = _scheduled_sampling_mix(
                    model,
                    step_batch,
                    clean_frames,
                    context_frames=ctx_frames,
                    prob=ss_prob,
                    chunk=config.chunk,
                    generator=ss_gen,
                )
                step_batch["target_frames"] = clean_frames

            # ACTION-CONTRASTIVE: build the WRONG-plan negative batch from the
            # (post-control-dropout) actuator drive so the second forward contrasts
            # the realised next state against a counterfactual plan.
            if config.action_contrastive and perturbable_cols:
                ac_gen = torch.Generator(device=dev).manual_seed(
                    (config.seed * 3_021_377) ^ step
                )
                step_batch["actuator_random"] = _random_actuator_batch(
                    step_batch.get("actuator"),
                    perturbable_cols=perturbable_cols,
                    generator=ac_gen,
                )

            opt.zero_grad(set_to_none=True)
            with _AutocastCtx(dev):
                out = model(
                    step_batch,
                    loss_spec={
                        "chunk": config.chunk,
                        "context_frames": ctx_frames,
                        "inverse_dynamics_weight": config.inverse_dynamics_weight,
                        "diagnostic_weight": _diagnostic_weight(
                            step, config.steps, config
                        ),
                        "cross_modal_weight": config.cross_modal_weight,
                        "self_predictive_weight": config.self_predictive_weight,
                        "action_contrastive_weight": config.action_contrastive_weight,
                        "return_components": True,
                    },
                )
                loss = out["loss"]
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            cam_nll = float(out["camera_nll"])
            diag_ce = float(out["diagnostic_ce"])
            if step % config.log_every == 0 or step == config.steps - 1:
                logger.info(
                    "overfit-controllable step %d/%d loss=%.4f cam=%.4f diag=%.4f",
                    step,
                    config.steps,
                    losses[-1],
                    cam_nll,
                    diag_ce,
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
    return result, model, samples, stream_names


# ---------------------------------------------------------------------------
# Manifest-window corpus dataset (honours B's per-shot excited-window geometry)
# ---------------------------------------------------------------------------


@dataclass
class _ManifestWindow:
    """One curated excited window: B's selected start + the per-shot stride."""

    shot_id: int
    start_frame: int
    frame_stride: int  # derived from the manifest fps to span ~target_horizon_s
    fps: float  # the shot's native cadence (from B's selector) — for logging
    # The PHYSICAL horizon (s) THIS window should span.  For full-shot windows it
    # is the shot's own ``plasma_duration_s`` (breakdown→termination), so each
    # window spans its OWN plasma phase; for the fixed-horizon manifests it is the
    # global ``target_horizon_s``.  ``assemble`` uses the TIME-BASED subsample with
    # this value (cadence-robust), NOT a literal stride.  0 -> fall back to the
    # global config horizon.
    horizon_s: float = 0.0
    # The camera this window's frames come from.  The unified multi-camera manifest
    # records a per-window ``camera_id`` (rbb / rco / rgb / rgc / rba …); the single
    # camera manifests omit it, so the field defaults to the reference camera for
    # full back-compat.  The dataset, the corpus length-filter, and the rank-0
    # sizing probe all assemble each window from THIS camera's tokens.
    camera: str = REFERENCE_CAMERA
    # Span (s) and frame count of THIS window's recording slice, derived from the
    # manifest's per-window ``frame_times`` at parse time.  When present (>0), the
    # corpus length-filter reads these directly instead of re-opening every Zarr to
    # call ``recording_time_span_s`` / ``camera_frame_count`` — eliminating a
    # ~20-min, every-launch GPFS scan (the build already measured this).  0 -> the
    # manifest omitted frame_times, so the filter falls back to the on-disk scan.
    avail_span_s: float = 0.0
    avail_frames: int = 0


def manifest_train_windows(
    manifest_path,
    held_out: set[int],
    *,
    target_horizon_s: float,
    n_frames: int,
) -> list[_ManifestWindow]:
    """Read the curated manifest into per-shot excited windows for training.

    For each manifest window (B's excitation selector already found the excited
    region + recorded its ``start_frame`` and native ``fps``), derive the per-shot
    ``frame_stride`` so the FIXED ``n_frames`` span the intended PHYSICAL horizon —
    NOT B's variable per-window ``n_frames`` (we keep rectangular batches + the
    forecaster's window shape).  Uses B's ``start_frame`` directly (no re-running
    the transient scan).

    Per-window horizon precedence (the value the TIME-BASED subsample spans):
      1. FULL-SHOT mode — when the window carries a positive ``plasma_duration_s``
         (B's full-shot selector records each shot's whole breakdown→termination
         length), the window spans THAT shot's own duration, so every window covers
         its OWN full plasma phase regardless of how long the shot is.  This is the
         genuine per-shot horizon full-shot training needs.
      2. else a single global ``target_horizon_s`` (the fixed-horizon modes).
    The per-window ``frame_stride`` is also recorded (for the corpus length filter);
    assembly itself uses the per-window ``horizon_s`` via the time-based subsample
    (cadence-robust), not the literal stride.

    Held-out shots are excluded (defence in depth — already excluded at select
    time).  Skips a window whose shot is too short for the strided window.
    """
    import json  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    data = json.loads(_Path(manifest_path).read_text())
    out: list[_ManifestWindow] = []
    n_fullshot = 0
    for w in data.get("windows", []):
        sid = int(w["shot_id"])
        if sid in held_out:
            continue
        fps = float(w.get("fps") or 0.0)
        per_window_horizon = float(w.get("plasma_duration_s") or 0.0)
        if per_window_horizon > 0.0:
            horizon_s = per_window_horizon  # full-shot: this shot's own duration
            n_fullshot += 1
        else:
            horizon_s = float(target_horizon_s or 0.0)  # global fixed horizon
        # frame_stride is for the corpus length filter only (assembly uses the
        # time-based subsample over horizon_s); derive it from the chosen horizon.
        if horizon_s > 0 and fps > 0:
            stride = max(1, int(round(horizon_s * fps / n_frames)))
        else:
            stride = int(w.get("frame_stride") or 1)
        # The build already recorded this window's frame times; derive its span +
        # frame count so the length-filter need not re-read the recording from GPFS.
        ft = w.get("frame_times")
        if ft and len(ft) >= 2:
            avail_span_s = float(ft[-1]) - float(ft[0])
            avail_frames = len(ft)
        else:
            avail_span_s = 0.0
            avail_frames = 0
        out.append(
            _ManifestWindow(
                shot_id=sid,
                start_frame=int(w.get("start_frame") or 0),
                frame_stride=stride,
                fps=fps,
                horizon_s=horizon_s,
                # per-window camera (multi-camera manifest); single-camera
                # manifests omit camera_id -> fall back to the reference camera.
                camera=str(w.get("camera_id") or REFERENCE_CAMERA),
                avail_span_s=avail_span_s,
                avail_frames=avail_frames,
            )
        )
    if n_fullshot:
        logger.info(
            "manifest: %d/%d windows use PER-SHOT full-shot horizon "
            "(plasma_duration_s); fixed n_frames=%d time-subsamples each shot's "
            "own plasma phase",
            n_fullshot,
            len(out),
            n_frames,
        )
    cameras = sorted({w.camera for w in out})
    if len(cameras) > 1:
        from collections import Counter  # noqa: PLC0415

        hist = Counter(w.camera for w in out)
        logger.info(
            "manifest: %d windows span %d cameras %s",
            len(out),
            len(cameras),
            dict(sorted(hist.items())),
        )
    return out


class ManifestWindowDataset:
    """Map-style dataset over the curated EXCITED windows (B's geometry honoured).

    Each item is assembled at the window's recorded ``start_frame`` with a
    PER-SHOT ``frame_stride`` so the fixed ``n_frames`` span ~``target_horizon_s``
    (the persistence-trap fix).  Builds a per-shot :class:`SpacetimeWindowConfig`
    (literal stride, ``target_horizon_s=0`` so :func:`assemble_window` uses that
    exact stride) and calls B's :func:`assemble_controllable_window` — B's dataset
    class + the actuator-plan reader stay untouched (the consumption-side fix).
    """

    def __init__(
        self,
        windows: Sequence[_ManifestWindow],
        config: SpacetimeWindowConfig,
        modalities: Sequence[SignalModalitySpec],
        n_signal_steps: int,
        n_act_steps: int,
        *,
        camera: str = REFERENCE_CAMERA,
        token_root=None,
        actuator_channels=None,
    ) -> None:
        self._windows = list(windows)
        self._config = config
        self._modalities = list(modalities)
        self._n_signal_steps = int(n_signal_steps)
        self._n_act_steps = int(n_act_steps)
        # dataset-wide DEFAULT camera; each item assembles from its OWN
        # ``window.camera`` (the multi-camera manifest), so this is only the
        # reference used to construct single-camera windows where camera_id was
        # absent — those windows already carry the reference camera here.
        self._camera = camera
        self._token_root = token_root
        self._actuator_channels = actuator_channels

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> ControllableSpacetimeSample:
        from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
            ACTUATOR_CHANNELS,
        )

        kw = {}
        if self._actuator_channels is not None:
            kw["actuator_channels"] = self._actuator_channels
        else:
            kw["actuator_channels"] = list(ACTUATOR_CHANNELS)
        # assemble at B's excited start_frame; assemble_window's TIME-BASED path
        # (target_horizon_s > 0) picks n_frames spanning ~the horizon from there,
        # robust to mid-shot cadence changes (a fixed per-shot literal stride
        # undershoots ~22% of windows — so we span by TIME, not by stride).  When a
        # window carries its OWN horizon (full-shot mode: each shot's
        # plasma_duration_s), span THAT — so every window covers its own whole
        # plasma phase — by overriding target_horizon_s on a per-window config copy.
        #
        # Defence-in-depth: if a window slips past the corpus-level filter and
        # still fails to assemble (a data edge case — e.g. a low-cadence
        # recording with < n_frames usable frames), fall back to the next few
        # windows rather than raising inside a DataLoader worker, which kills the
        # whole DDP rank (a single bad shot at step 50 took down a 6-GPU run).
        # Bounded so a systematically-broken corpus still fails loudly.
        n = len(self._windows)
        last_err: Exception | None = None
        for off in range(min(8, n)):
            w = self._windows[(index + off) % n]
            cfg_w = self._config
            if (
                w.horizon_s > 0
                and abs(w.horizon_s - self._config.target_horizon_s) > 1e-9
            ):
                cfg_w = replace(self._config, target_horizon_s=float(w.horizon_s))
            try:
                return assemble_controllable_window(
                    w.shot_id,
                    cfg_w,
                    self._modalities,
                    self._n_signal_steps,
                    self._n_act_steps,
                    # per-window camera (the unified multi-camera manifest); falls
                    # back to the dataset's reference camera for single-camera
                    # manifests where every window carries the reference camera.
                    camera=w.camera,
                    token_root=self._token_root,
                    start_frame=w.start_frame,
                    **kw,
                )
            except (ValueError, FileNotFoundError, KeyError) as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"ManifestWindowDataset: {min(8, n)} consecutive windows from index "
            f"{index} all failed to assemble; last error: {last_err!r}"
        )


# ---------------------------------------------------------------------------
# Corpus DDP trainer (the 6-GPU re-train on the excited corpus)
# ---------------------------------------------------------------------------


@dataclass
class ControllableCorpusConfig:
    """Knobs for the multi-shot DDP re-train of the controllable model.

    Mirrors the v2 ``CorpusV2Config`` (steps / batch / LR schedule / checkpoint /
    eval cadence / DDP workers) and adds the M4 controllability machinery: the
    camera-history embedding bottleneck, the inverse-dynamics auxiliary weight,
    scheduled sampling, and the high observation-dropout + control-dropout that
    force the model to drive from the actuator PLAN.
    """

    steps: int = 12000
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.1
    seed: int = 0
    log_every: int = 25
    ckpt_every: int = 250
    eval_every: int = 1000
    num_workers: int = 4
    prefetch_factor: int = 4
    grad_clip: float = 1.0
    chunk: int = 4096
    n_signal_steps: int = 4
    n_act_steps: int = 8
    lr_schedule: bool = True
    warmup_steps: int = 400
    min_lr_ratio: float = 0.02
    random_window: bool = True
    # M4 controllability machinery (the same levers the overfit gate validated).
    observation_dropout: float = 0.8
    # PER-MODALITY masking (default ON): each measured-signal stream masked
    # INDEPENDENTLY (per-stream Bernoulli(observation_dropout)) + the CLEAN pre-mask
    # signals kept as the diagnostic-CE target, so every stream is a CE target on
    # ~(1-rate) of steps (full coverage) and the model dreams a masked stream from
    # the others + commands + cameras.  False = the legacy all-or-nothing
    # ``_drop_observations`` (the ablation path).  See OverfitControllableConfig.
    per_modality_masking: bool = True
    control_dropout: float = 0.15
    history_bottleneck: HistoryBottleneckConfig = field(
        default_factory=HistoryBottleneckConfig
    )
    inverse_dynamics_weight: float = 1.0
    # JOINT GENERATION — per-stream diagnostic-prediction heads + next-step CE on
    # the measured-signal tokens (the model dreams cameras AND diagnostics).
    # ``generate_diagnostics`` is the ablation off-switch; ``diagnostic_weight``
    # weights the CE; the CE is warmed up linearly 0 -> diagnostic_weight over
    # ``diagnostic_weight_warmup_frac`` of total steps (it can destabilise the
    # camera loss early).
    generate_diagnostics: bool = True
    diagnostic_weight: float = 0.5
    diagnostic_weight_warmup_frac: float = 0.1
    # AUXILIARY contrastive objectives (default OFF — the §6 ablation turns each on).
    # See OverfitControllableConfig for the per-term descriptions.  cross_modal +
    # self_predictive read the camera/signal latents; action_contrastive needs the
    # actuator drive + a second forward under a random plan (``actuator_random``).
    cross_modal: bool = False
    cross_modal_weight: float = 0.1
    self_predictive: bool = False
    self_predictive_weight: float = 0.1
    action_contrastive: bool = False
    action_contrastive_weight: float = 0.1
    scheduled_sampling_max: float = 0.25
    scheduled_sampling_ramp: float = 0.5
    # EXCITATION-WEIGHT the next-frame CE by per-frame coil |dI/dt| (the operator's
    # drive).  A full-shot window is ~half flat-top frames by COUNT; without this
    # the persistence-easy flat-top frames drown the coil→plasma gradient and the
    # model re-dilutes to the 0/3 multi-window failure.  The weight up-scales
    # forecast frames where the coils move.  ``excitation_weight_floor`` is the
    # baseline weight a NON-moving frame still carries (so quiet frames are not
    # zeroed — they still teach persistence); the moving frames are scaled up to
    # 1.0.  ``excitation_weight_power`` sharpens (>1) or softens (<1) the contrast.
    # False = uniform per-frame loss (the ablation baseline).  The inverse-dynamics
    # aux already self-weights toward coil motion; this reinforces it at frame
    # resolution in the PRIMARY next-frame objective.
    excitation_weighting: bool = True
    excitation_weight_floor: float = 0.1
    excitation_weight_power: float = 1.0
    # DROP the measured STATES (plasma_current + ne_line_integrated) from the
    # actuator COMMAND vector — they are RESPONSES, not commands (they remain
    # available as v2 observation streams).  The dataset is then built on the
    # filtered :func:`command_channels` list so the actuator vector + the AdaLN MLP
    # input dim follow the data.  True is the controllability-correct default.
    drop_state_channels: bool = True
    window: SpacetimeWindowConfig = field(default_factory=SpacetimeWindowConfig)
    model_kwargs: dict = field(default_factory=dict)
    modalities: list[SignalModalitySpec] = field(
        default_factory=default_signal_modalities
    )
    # warm-start the backbone from the M2 forecaster ONCE (when no resume
    # checkpoint of THIS run exists yet) — the AdaLN MLP + inverse-dynamics head
    # start at init (zero-init gate => the warm start is the forecaster).
    init_checkpoint: Path | None = None


@dataclass
class ControllableCorpusResult:
    steps_run: int
    initial_loss: float
    final_loss: float
    losses: list[float]
    n_parameters: int
    n_train_shots: int
    checkpoint_path: str | None
    stream_names: list[str] = field(default_factory=list)


def train_controllable_corpus(
    shot_ids: Sequence[int],
    *,
    camera: str = REFERENCE_CAMERA,
    config: ControllableCorpusConfig | None = None,
    out_dir: Path | None = None,
    token_root: Path | None = None,
    device: str | None = None,
    resume: bool = True,
    manifest_windows: Sequence[_ManifestWindow] | None = None,
) -> ControllableCorpusResult:
    """DDP re-train of the controllable model on a CORPUS of EXCITED shots.

    The 6-GPU re-train the de-risk gate cleared the way for: warm-start the M2
    forecaster, fine-tune on the curated excited corpus with the camera-history
    bottleneck + inverse-dynamics + scheduled-sampling + the actuator-PLAN drive,
    so the plan becomes load-bearing for a FREE-RUNNING rollout (not just the
    one-step prediction).  Reuses the v2 DDP primitives (``DistEnv``, the LR
    scheduler, checkpoint/resume, the distributed sampler) verbatim; the model,
    dataset, and per-step controllability machinery are the controllable ones.

    ``manifest_windows`` (the corpus path): the curated EXCITED windows (B's
    per-shot ``start_frame`` + the derived per-shot stride) — each window is
    assembled at its excited start spanning ~``target_horizon_s``
    (:class:`ManifestWindowDataset`), the horizon fix.  When ``None`` the legacy
    ``shot_ids`` + random-window path is used (centred/random windows).

    DDP-aware: each rank pins its card and trains a disjoint shard; the
    plan-channel count + signal-stream widths are decided on rank 0 and broadcast
    so every rank builds the IDENTICAL model.  Resume-safe: ``latest.pt`` every
    ``ckpt_every`` steps, a SIGTERM STOP flag flushes within the step, and a
    restart resumes from ``latest.pt``.
    """
    import contextlib  # noqa: PLC0415
    import time  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    from torch import nn  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_dataset import (  # noqa: PLC0415
        ControllableSpacetimeDataset,
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

    config = config or ControllableCorpusConfig()
    if out_dir is None:
        import os  # noqa: PLC0415

        run_id = (
            os.environ.get("SLURM_JOB_ID") or os.environ.get("WM_RUN_ID") or "local"
        )
        out_dir = DEFAULT_CKPT_ROOT / f"controllable-{run_id}"
    out_dir = _Path(out_dir)
    _set_determinism(config.seed)

    env = DistEnv.from_environment()
    _init_distributed(env)
    if device is None:
        device = f"cuda:{env.local_rank}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    stop = _StopFlag()
    stop.install()

    # The actuator vector keeps the FULL channel set; the measured STATES (Ip +
    # density + tf) are MASKED OUT inside the model's _plan_summary (the single
    # conditioning entry point — train/gate/inference stay consistent), so the
    # model conditions ONLY on the commands without an always-on Ip readout it
    # could cheat through.  Ip/density remain available as the v2 observations.
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ACTUATOR_CHANNELS,
    )

    act_chan_list = list(ACTUATOR_CHANNELS)

    # Filter to windows/shots whose recording is long enough for the STRIDED
    # window (an under-length shot raises mid-epoch and kills a DDP rank).
    # Deterministic from the token-root so every rank computes the SAME set
    # (DDP-safe — no shard divergence).
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        camera_frame_count,
        recording_time_span_s,
        window_span_for_shot,
    )

    use_manifest = manifest_windows is not None
    horizon = float(config.window.target_horizon_s)
    if use_manifest:
        # MANIFEST-WINDOW path (option i): keep windows whose RECORDING spans at
        # least the horizon in TIME (assemble_window subsamples n_frames spanning
        # ~horizon from B's start_frame, robust to a variable cadence — so the
        # requirement is a time-span, not a fixed native-frame count).
        #
        # The multi-window manifest has ~3 windows per pulse, so the (span, count)
        # of a (shot, camera) recording is read repeatedly — memoise per
        # (shot_id, camera) (the recording is the same for every window of a shot
        # from the same camera) so the scan does ~unique-recordings GPFS reads, not
        # ~windows (8.7k windows over 3k shots was a ~7.5min startup).  The unified
        # manifest mixes cameras, so the key MUST include the camera — two windows
        # of the same shot from different cameras are different recordings.
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
            # the horizon THIS window must span — its own (full-shot) or global.
            w_horizon = w.horizon_s if w.horizon_s > 0 else horizon
            if w.avail_frames > 0:
                # Manifest path (no GPFS read): the build recorded this window's
                # frame_times, so its frame count is known.  The binding constraint
                # for the time-based subsample is having >= n_frames frames in the
                # window to pick from — a too-short recording makes
                # _time_spanned_indices raise inside a worker and kills the rank.
                # The window span is NOT re-checked against w_horizon: the build
                # already selected the window to span its horizon, and the camera
                # frame-slice span reads slightly under the Ip-based
                # plasma_duration_s, so that comparison would over-drop valid
                # windows (assemble clamps the last targets to the final frame).
                ok = w.avail_frames >= config.window.n_frames
            else:
                # Back-compat: manifest omitted frame_times -> scan the recording.
                sc = _span_count(w.shot_id, w.camera)
                scanned += 1
                if sc is None:
                    ok = False
                else:
                    span_s, n_total = sc
                    if w_horizon > 0:
                        # need BOTH: recording spans >= horizon in TIME, and >=
                        # n_frames frames to subsample (a low-cadence recording can
                        # span the horizon with fewer than n_frames).
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
                "frames (dropped %d too-short; %d needed a GPFS scan, %d sized from "
                "the manifest's frame_times — no scan)",
                len(kept_windows),
                len(list(manifest_windows)),
                config.window.n_frames,
                n_dropped,
                scanned,
                len(list(manifest_windows)) - scanned,
            )
        if len(kept_windows) < 2:
            raise ValueError(
                f"only {len(kept_windows)} excited windows are long enough for the "
                f"~{config.window.target_horizon_s}s horizon — cannot train"
            )
        manifest_windows = kept_windows
    else:
        # LEGACY path: per-shot horizon span derived from the shot's own cadence.
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
        n_dropped = len(list(shot_ids)) - len(kept)
        if env.is_main:
            logger.info(
                "frame-count filter: kept %d/%d train shots long enough for the "
                "~%.2fs horizon window (dropped %d short/unreadable)",
                len(kept),
                len(list(shot_ids)),
                config.window.target_horizon_s,
                n_dropped,
            )
        if len(kept) < 2:
            raise ValueError(
                f"only {len(kept)} train shots are long enough for the "
                f"~{config.window.target_horizon_s}s horizon window — cannot train"
            )
        shot_ids = kept

    # Probe plan-channels + signal-stream widths + actuator-channel count.  On the
    # manifest path the probe assembles EXACTLY as ManifestWindowDataset does — at
    # B's excited start_frame with the global config.window (target_horizon_s > 0 →
    # assemble_window's TIME-BASED subsample picks n_frames spanning ~the horizon),
    # so the sanity-log below reports the SAME spans training actually sees.  (A
    # per-shot literal stride here ran off the recording end for late start_frames
    # and clamped to ~consecutive frames — a misleading ~78ms sanity-log even
    # though training used the correct time-based windows.)
    probe: list[ControllableSpacetimeSample] = []
    if use_manifest:
        for w in manifest_windows[:8]:
            try:
                probe.append(
                    assemble_controllable_window(
                        w.shot_id,
                        config.window,
                        config.modalities,
                        config.n_signal_steps,
                        config.n_act_steps,
                        # use the WINDOW's camera (not the single CLI camera) so the
                        # probe assembles EXACTLY as ManifestWindowDataset will — the
                        # broadcast model dims must match what every rank's dataset
                        # actually produces across all cameras.
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
    # Sanity-log the ACHIEVED physical span of the probe windows (the real check
    # that the time-based subsample covers the ramp horizon, not a ~15ms slice).
    if env.is_main and config.window.target_horizon_s > 0:
        for p in probe[:4]:
            ft = np.asarray(p.frame_time, dtype=np.float64)
            span_ms = (ft[-1] - ft[0]) * 1000.0 if ft.size > 1 else float("nan")
            logger.info(
                "horizon check shot %s: %d frames span %.0f ms (target %.0f ms)",
                p.shot_id,
                ft.size,
                span_ms,
                config.window.target_horizon_s * 1000.0,
            )
    plan_ch = _plan_channels_for([p.signal for p in probe])
    # Probe shots SPREAD across the corpus, not the first N.  The manifest is
    # sorted by shot id, so the first-N shots all sit in the earliest era and MISS
    # era-sparse streams — e.g. ait (divertor heat-flux) only exists for later shot
    # ids, so a first-N probe sizes it to 0 channels and silently drops it from the
    # model.  A deterministic evenly-spaced sample spans every shot-id era (and
    # every camera era), so each present stream is sized; identical on every rank.
    # The unified manifest mixes cameras, so probe each shot with the camera the
    # dataset will actually read; the legacy path uses the single CLI camera.
    if use_manifest:
        mw = list(manifest_windows)
        n_probe = min(48, len(mw))
        step = max(1, len(mw) // n_probe)
        probe_sel = mw[::step][:n_probe]
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
    # Columns to MASK from conditioning (Ip + density + tf) — deterministic from
    # the actuator channel keys, so every rank computes the same set.  Empty when
    # drop_state_channels is off (debug: condition on the full vector).
    masked_idx = (
        masked_command_columns(list(probe[0].actuator.channel_keys))
        if config.drop_state_channels
        else ()
    )
    # Coil-current columns IN THE ACTUATOR VECTOR's coordinate system (which is the
    # filtered command-channel list when drop_state_channels is on, not the full
    # ACTUATOR_CHANNELS).  Resolve by KEY: the canonical coil keys mapped to their
    # positions in the probe's channel_keys.  Used for excitation weighting; empty
    # disables it (falls back to uniform loss).
    _coil_keys = {
        ACTUATOR_CHANNEL_KEYS[i]
        for i in coil_current_channel_indices()
        if i < len(ACTUATOR_CHANNEL_KEYS)
    }
    coil_cols = [
        i for i, k in enumerate(probe[0].actuator.channel_keys) if k in _coil_keys
    ]
    # Perturbable command columns (coils+sol+nbi+gas) for the action-contrastive
    # WRONG-plan negative — KEY-resolved, so every rank computes the same set.
    perturbable_cols = (
        _perturbable_command_columns(list(probe[0].actuator.channel_keys))
        if config.action_contrastive
        else []
    )
    if env.is_main and config.excitation_weighting:
        logger.info(
            "excitation weighting ON: %d coil columns %s (floor=%.2f power=%.2f)",
            len(coil_cols),
            coil_cols,
            config.excitation_weight_floor,
            config.excitation_weight_power,
        )

    # Broadcast every shaping quantity from rank 0 so all ranks build the SAME
    # model (a per-rank GPFS read miss must not desync the param set).
    plan_ch = _broadcast_int(env, plan_ch)
    act_channels = _broadcast_int(env, act_channels)
    for m in config.modalities:
        channels[m.name] = _broadcast_int(env, int(channels.get(m.name, 0)))
    streams = stream_specs_from_modalities(config.modalities, channels)
    stream_names = [st.name for st in streams]

    hb = config.history_bottleneck
    corruption_levels = int(hb.levels) if hb.enabled else 0
    base_model = build_controllable_model(
        config.window,
        plan_channels=plan_ch,
        signal_streams=streams,
        n_signal_steps=config.n_signal_steps,
        actuator_channels=act_channels,
        n_act_steps=config.n_act_steps,
        corruption_levels=corruption_levels,
        inverse_dynamics=config.inverse_dynamics_weight > 0.0,
        generate_diagnostics=config.generate_diagnostics,
        cross_modal=config.cross_modal,
        self_predictive=config.self_predictive,
        action_contrastive=config.action_contrastive,
        masked_command_indices=masked_idx,
        **config.model_kwargs,
    ).to(dev)
    if env.is_main:
        logger.info(
            "controllable-corpus on %s: params=%d (%.1fM) d_model=%d n_layers=%d "
            "plan_ch=%d actuator_ch=%d hb=%s inv_dyn_w=%.2f ss_max=%.2f "
            "obs_dropout=%.2f streams=%s world=%d",
            dev,
            base_model.num_parameters(),
            base_model.num_parameters() / 1e6,
            base_model.config.d_model,
            base_model.config.n_layers,
            plan_ch,
            act_channels,
            f"std={hb.noise_std}/mask={hb.mask_prob}/max={hb.max_strength}"
            if hb.enabled
            else "off",
            config.inverse_dynamics_weight,
            config.scheduled_sampling_max,
            config.observation_dropout,
            [(st.name, st.channels) for st in streams],
            env.world_size,
        )

    opt = torch.optim.AdamW(
        base_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    start_step = 0

    # Resume THIS run's latest.pt first; else warm-start the forecaster ONCE.
    latest = find_latest_checkpoint(out_dir) if resume else None
    if latest is not None:
        try:
            payload = torch.load(str(latest), map_location="cpu", weights_only=False)
            _unwrap(base_model).load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                opt.load_state_dict(payload["optimizer_state_dict"])
            start_step = int(payload.get("step", 0))
            if env.is_main:
                logger.info("RESUMED from %s at step %d", latest, start_step)
        except (KeyError, RuntimeError, ValueError) as exc:
            if env.is_main:
                logger.warning("checkpoint %s incompatible (%r) — fresh", latest, exc)
    elif config.init_checkpoint is not None:
        n_loaded, n_new = _warm_start_from_forecaster(
            _unwrap(base_model), _Path(config.init_checkpoint), dev
        )
        if env.is_main:
            logger.info(
                "warm-started backbone from %s: %d tensors loaded, %d new "
                "(AdaLN MLP + inverse-dynamics head) left at init",
                config.init_checkpoint,
                n_loaded,
                n_new,
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
        # the excited-window dataset: each item at B's start_frame + per-shot
        # stride spanning ~target_horizon_s (the horizon fix).
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
        dataset = ControllableSpacetimeDataset(
            shot_ids,
            config.window,
            config.modalities,
            config.n_signal_steps,
            config.n_act_steps,
            camera=camera,
            token_root=token_root,
            random_window=config.random_window,
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

    drop_cfg = ContextCorruptionConfig(control_dropout=config.control_dropout)
    ctx_frames = int(config.window.context_frames)
    accum = max(1, int(config.grad_accum))

    def _prepare_step_batch(batch: dict, *, gen_seed: int) -> dict:
        """History-bottleneck + obs-dropout + control-dropout + scheduled-sampling.

        Returns a batch carrying the (possibly mixed) ``frames`` for the forward,
        a CLEAN ``target_frames`` for the loss, the per-frame bottleneck strengths
        + level bin, and the dropped conditioning.  Mirrors ``overfit_controllable``
        per-step prep but for the corpus loop.
        """
        out = dict(batch)
        clean = batch["frames"]
        b = int(clean.shape[0])
        # camera-history embedding bottleneck.
        if hb.enabled:
            hb_gen = torch.Generator(device=dev).manual_seed(gen_seed ^ 0x48424F)
            strengths, bins = sample_frame_strengths(
                b, ctx_frames, hb, generator=hb_gen, device=dev
            )
            out["history_bottleneck"] = hb
            out["history_strengths"] = strengths
            out["history_generator"] = torch.Generator(device=dev).manual_seed(
                gen_seed ^ 0x484247
            )
            out["target_frames"] = clean
            out["corruption_level"] = bins.max(dim=1).values
        # observation masking — drive from the PLAN, not the redundant signals.
        # Keep a CLEAN (pre-mask) reference as the diagnostic-CE target so the model
        # learns to DREAM a masked diagnostic, then set the (masked) model input.
        d_gen = torch.Generator(device=dev).manual_seed(gen_seed ^ 0x4F4253)
        out["signal_targets"] = (
            dict(batch["signals"]) if batch.get("signals") else batch.get("signals")
        )
        if config.per_modality_masking:
            # INDEPENDENT per-stream masking (full per-stream CE coverage).
            out["signals"] = _mask_observations_per_stream(
                batch.get("signals"),
                config.observation_dropout,
                generator=d_gen,
                device=dev,
            )
        else:
            # ALL-OR-NOTHING (the ablation path).
            obs_drop = (
                torch.rand(b, generator=d_gen, device=dev)
                < config.observation_dropout
            )
            out["signals"] = _drop_observations(batch.get("signals"), obs_drop)
        # control dropout (CFG): zero the WHOLE conditioning on a fraction.
        ctrl_drop = sample_control_dropout(b, drop_cfg, generator=d_gen, device=dev)
        if bool(ctrl_drop.any()):
            plan2, signals2 = apply_control_dropout(
                batch.get("plan"), out["signals"], ctrl_drop
            )
            if plan2 is not None:
                out["plan"] = plan2
            if signals2 is not None:
                out["signals"] = signals2
            act = batch.get("actuator")
            if act is not None:
                av = act["values"].clone()
                av[ctrl_drop] = 0.0
                out["actuator"] = {"values": av, "missing": act["missing"]}
        return out

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
                seed = (config.seed * 7919) ^ (env.rank * 104729) ^ step
                batch = _prepare_step_batch(batch, gen_seed=seed)
                # scheduled sampling — splice the model's own predictions into the
                # forecast context (rollout-in-the-loop).
                ss_prob = _scheduled_sampling_prob(step, config.steps, config)
                if ss_prob > 0.0:
                    ss_gen = torch.Generator(device=dev).manual_seed(seed ^ 0x53534D)
                    batch["frames"] = _scheduled_sampling_mix(
                        core,
                        batch,
                        batch.get("target_frames", batch["frames"]),
                        context_frames=ctx_frames,
                        prob=ss_prob,
                        chunk=config.chunk,
                        generator=ss_gen,
                    )
                    batch["target_frames"] = batch.get("target_frames", batch["frames"])
                is_boundary = (micro + 1) % accum == 0
                sync_ctx = (
                    model.no_sync()
                    if (env.enabled and not is_boundary)
                    else contextlib.nullcontext()
                )
                frame_w = None
                if config.excitation_weighting and coil_cols:
                    frame_w = _excitation_frame_weights(
                        batch.get("actuator"),
                        n_frames=int(batch["frames"].shape[1]),
                        coil_cols=coil_cols,
                        device=dev,
                        floor=config.excitation_weight_floor,
                        power=config.excitation_weight_power,
                    )
                # action-contrastive WRONG-plan negative (built from the prepared,
                # post-control-dropout actuator drive).
                if config.action_contrastive and perturbable_cols:
                    ac_gen = torch.Generator(device=dev).manual_seed(seed ^ 0x4143)
                    batch["actuator_random"] = _random_actuator_batch(
                        batch.get("actuator"),
                        perturbable_cols=perturbable_cols,
                        generator=ac_gen,
                    )
                with sync_ctx, _AutocastCtx(dev):
                    out = model(
                        batch,
                        loss_spec={
                            "chunk": config.chunk,
                            "context_frames": ctx_frames,
                            "inverse_dynamics_weight": config.inverse_dynamics_weight,
                            "diagnostic_weight": _diagnostic_weight(
                                step, config.steps, config
                            ),
                            "cross_modal_weight": config.cross_modal_weight,
                            "self_predictive_weight": config.self_predictive_weight,
                            "action_contrastive_weight": (
                                config.action_contrastive_weight
                            ),
                            "frame_weights": frame_w,
                            "return_components": True,
                        },
                    )
                    loss = out["loss"]
                    scaled = loss / accum
                cam_nll = float(out["camera_nll"])
                diag_ce = float(out["diagnostic_ce"])
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
                        "controllable-corpus step %d/%d loss=%.4f cam=%.4f "
                        "diag=%.4f lr=%.3e ss=%.2f (%.2f st/s world=%d)",
                        step,
                        config.steps,
                        losses[-1],
                        cam_nll,
                        diag_ce,
                        opt.param_groups[0]["lr"],
                        ss_prob,
                        rate,
                        env.world_size,
                    )

                if config.ckpt_every > 0 and step % config.ckpt_every == 0:
                    _barrier(env)
                    if env.is_main:
                        ckpt_path = save_controllable_checkpoint(
                            out_dir,
                            model=core,
                            optimizer=opt,
                            step=step,
                            window=config.window,
                            extra={"stream_names": stream_names},
                        )
                        logger.info("checkpoint @ step %d -> %s", step, ckpt_path)
                    _barrier(env)
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    # final checkpoint flush (covers a clean SIGTERM stop mid-run).
    _barrier(env)
    if env.is_main:
        ckpt_path = save_controllable_checkpoint(
            out_dir,
            model=core,
            optimizer=opt,
            step=step,
            window=config.window,
            extra={"stream_names": stream_names},
        )
        logger.info("final checkpoint @ step %d -> %s", step, ckpt_path)
    _barrier(env)

    return ControllableCorpusResult(
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
# Controllability gate (decoder-free token-space causal margin)
# ---------------------------------------------------------------------------
#
# The full M3 falsification scores DECODED pixel space (it loads the frozen
# Open-MAGVIT2 VQ).  For the cheap gate we measure the causal margin in TOKEN
# space first — it needs no decoder and is a strict lower bound on "does the
# plan move the camera": if the predicted TOKENS do not change when the plan
# changes, the decoded pixels cannot either.  The driver
# (controllable_gate_run) then OPTIONALLY decodes a couple of rollouts to confirm
# the token response shows up in pixel space (reusing M3's control_falsification
# metrics).


@torch.no_grad()
def _forward_with_actuator(
    model: ControllableSpacetimeTransformer,
    frames: torch.Tensor,
    plan: torch.Tensor | None,
    signals: dict[str, torch.Tensor] | None,
    actuator: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    return model._forward_tokens(frames, plan, signals, actuator=actuator)


@torch.no_grad()
def teacher_forced_token_mismatch(
    model: ControllableSpacetimeTransformer,
    batch: dict,
    *,
    actuator_a: dict[str, torch.Tensor] | None,
    actuator_b: dict[str, torch.Tensor] | None,
    chunk: int = 4096,
    frames_override: torch.Tensor | None = None,
) -> float:
    """Fraction of next-frame argmax tokens that DIFFER between two actuator plans.

    Both forwards are teacher-forced on the SAME frames + plan + signals and
    differ ONLY in the actuator plan.  A POSITIVE mismatch means the actuator
    plan causally moves the predicted camera tokens — the controllability signal
    in token space (a strict lower bound on the decoded-pixel response).

    ``frames_override`` swaps the teacher-forced camera history (e.g. a
    context-corrupted copy) so the test can probe how much the actuator plan
    moves the prediction WHEN the camera context cannot fully determine it — the
    fair controllability reading on an overfit model that has memorised the
    exact camera continuation (see :func:`controllability_gate`).
    """
    model.eval()
    frames = frames_override if frames_override is not None else batch["frames"]
    plan = batch.get("plan")
    signals = batch.get("signals")
    with _AutocastCtx(frames.device):
        ha = _forward_with_actuator(model, frames, plan, signals, actuator_a)
        hb = _forward_with_actuator(model, frames, plan, signals, actuator_b)
    t = ha.shape[1]
    diffs: list[float] = []
    for ti in range(t):
        pa = model.chunked_argmax_frame(ha[:, ti], chunk=chunk)
        pb = model.chunked_argmax_frame(hb[:, ti], chunk=chunk)
        diffs.append(float((pa != pb).float().mean()))
    model.train()
    return float(np.mean(diffs)) if diffs else float("nan")


def _actuator_batch_from_plan(plan, device: torch.device) -> dict:
    """One-sample ``{"values","missing"}`` batch from an :class:`ActuatorPlan`."""
    v = torch.as_tensor(plan.values[None], dtype=torch.float32, device=device)
    m = torch.as_tensor(plan.missing[None], dtype=torch.float32, device=device)
    return {"values": v, "missing": m}


def _set_plan_channels_physical(plan, channel_indices, level: float):
    """A copy of the plan with the given channels SET to a fixed physical command.

    Gate-fix (2b): the earlier gas/NBI lever was ``scale_plan_channels(idx, 3.0)``
    — a ×3 of a near-zero baseline, which on a flat-top window is a <1% input
    perturbation (gate_verdict_1220668 measured gas×3 = 0.72% of the drive
    L1-norm).  That is far too small to test whether the model RESPONDS to the
    gas/NBI command.  This SETS the selected channels to an absolute physical
    command instead — a real, in-distribution intervention ("fire the gas hard")
    — by writing the RAW value and re-normalising.  ``level`` is the absolute raw
    physical command (same units as ``raw_values``); a per-channel ``level`` of
    the window-MAX (see :func:`_physical_command_levels`) gives "the strongest
    level this actuator reached on this shot", a meaningful swing vs the silenced
    counterfactual.  ``missing`` is preserved.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ActuatorPlan,
        normalise_actuator_values,
    )

    raw = np.asarray(plan.raw_values, dtype=np.float64).copy()
    idx = [int(i) for i in channel_indices if 0 <= int(i) < raw.shape[1]]
    levels = np.atleast_1d(np.asarray(level, dtype=np.float64))
    for k, ch in enumerate(idx):
        raw[:, ch] = float(levels[k]) if levels.size == len(idx) else float(levels[0])
    return ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=plan.missing.copy(),
        channel_keys=list(plan.channel_keys),
        raw_values=raw.astype(np.float32),
    )


def _physical_command_levels(plan, channel_indices) -> list[float]:
    """Per-channel window-MAX raw physical command (the "fire it hard" level).

    For each selected channel, the maximum absolute raw command it reaches across
    the window (sign preserved).  Setting a channel to this level is a strong,
    in-distribution intervention; if the channel is flat at zero across the window
    the level is a small positive floor so the lever is still a real (if modest)
    command rather than a no-op.
    """
    raw = np.asarray(plan.raw_values, dtype=np.float64)
    levels: list[float] = []
    for ch in channel_indices:
        if not (0 <= int(ch) < raw.shape[1]):
            levels.append(0.0)
            continue
        col = raw[:, int(ch)]
        amax = float(col[np.argmax(np.abs(col))]) if col.size else 0.0
        if abs(amax) < 1e-9:
            # flat-at-zero channel: a modest positive floor so the lever fires.
            amax = 1.0
        levels.append(amax)
    return levels


def _corrupt_all_context(
    frames: torch.Tensor,
    *,
    context_frames: int,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    """Replace EVERY context-frame token with a random in-vocab id.

    On a model overfit to ~zero loss the camera CONTEXT alone reconstructs the
    memorised continuation, so no conditioning (actuator OR observation) is
    load-bearing on the clean-context teacher-forced test — every margin reads
    ~0 regardless of whether the controls work.  Destroying the context (rate
    1.0) removes that lookup shortcut: the only information left distinguishing
    the prediction is the conditioning prefix (plan + actuator + signals), so the
    actuator margin measured on a corrupted context is the FAIR controllability
    reading.  Uses the M2 history-corruption noiser at full rate.
    """
    rates = torch.ones(frames.shape[0], device=frames.device)
    gen = torch.Generator(device=frames.device).manual_seed(int(seed))
    return corrupt_context_tokens(
        frames,
        rates,
        context_frames=int(context_frames),
        vocab_size=int(vocab_size),
        generator=gen,
    )


@dataclass
class ControllabilityVerdict:
    """The gate's PASS/FAIL with numbers."""

    shot_id: int
    # CLEAN-context teacher-forced margins (fraction of next-frame tokens that
    # change).  On a fully-overfit model the camera context alone reconstructs
    # the memorised continuation, so these read ~0 even if the controls work —
    # they are reported for transparency but are NOT the gate's decision metric.
    true_vs_zeroed_mismatch: float  # full plan vs silenced plan (whole drive)
    gas_scale_mismatch: float  # full plan vs gas-puff command scaled up
    gas_zero_mismatch: float  # full plan vs gas-puff command silenced
    nbi_scale_mismatch: float  # full plan vs NBI command scaled up
    # observation ablation (control vs the redundant observations): how much the
    # measured signals move the prediction (should be SMALL relative to the plan
    # if the plan is the load-bearing surface).
    observation_mismatch: float
    plan_over_observation_ratio: float
    # CORRUPTED-context margins — the FAIR controllability reading.  With the
    # memorised camera context destroyed, the only information distinguishing the
    # prediction is the conditioning prefix, so a non-trivial true-vs-zeroed
    # margin here means the actuator plan is genuinely load-bearing.  This is the
    # gate's decision metric.
    cc_true_vs_zeroed_mismatch: float
    cc_gas_scale_mismatch: float
    cc_nbi_scale_mismatch: float
    cc_observation_mismatch: float
    n_gas_channels: int
    n_nbi_channels: int
    # how much the actuator PLAN itself varies across the window (summed
    # per-channel std of the normalised drive) + the camera's own frame-to-frame
    # change fraction — together they prove the gate ran on a TRANSIENT window
    # where there was control variation to respond to (not a flat-top).
    plan_variation: float
    camera_change_fraction: float
    is_transient: bool
    passed: bool

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "true_vs_zeroed_mismatch": self.true_vs_zeroed_mismatch,
            "gas_scale_mismatch": self.gas_scale_mismatch,
            "gas_zero_mismatch": self.gas_zero_mismatch,
            "nbi_scale_mismatch": self.nbi_scale_mismatch,
            "observation_mismatch": self.observation_mismatch,
            "plan_over_observation_ratio": self.plan_over_observation_ratio,
            "cc_true_vs_zeroed_mismatch": self.cc_true_vs_zeroed_mismatch,
            "cc_gas_scale_mismatch": self.cc_gas_scale_mismatch,
            "cc_nbi_scale_mismatch": self.cc_nbi_scale_mismatch,
            "cc_observation_mismatch": self.cc_observation_mismatch,
            "n_gas_channels": self.n_gas_channels,
            "n_nbi_channels": self.n_nbi_channels,
            "plan_variation": self.plan_variation,
            "camera_change_fraction": self.camera_change_fraction,
            "is_transient": self.is_transient,
            "passed": self.passed,
        }


def controllability_gate(
    model: ControllableSpacetimeTransformer,
    samples: Sequence[ControllableSpacetimeSample],
    stream_names: Sequence[str],
    *,
    device: str | None = None,
    chunk: int = 4096,
    gas_scale: float = 1.0,
    nbi_scale: float = 1.0,
    margin_threshold: float = 0.02,
    transient_threshold: float = 1e-3,
) -> tuple[list[ControllabilityVerdict], dict]:
    """Token-space controllability gate over the overfit samples.

    For each sample, teacher-force the model and compare the predicted next-frame
    tokens under the FULL actuator plan against:

    * the plan with the WHOLE drive silenced (:func:`zero_plan`) — the headline
      true-vs-zeroed causal margin;
    * the plan with the gas-puff command SET to a fixed physical level
      (``gas_scale`` × the window-max, default 1.0 = the window max) and SILENCED;
    * the plan with the NBI command SET to a fixed physical level
      (``nbi_scale`` × the window-max);
    * the measured observations ablated (signals zeroed) — the redundancy
      baseline the plan margin must beat (M3's failure mode).

    Each sample also records how much the actuator PLAN itself varies across the
    window (and the camera's own change fraction): the gate is only FAIR on a
    TRANSIENT window where the plan moves — a flat-top window has no control
    variation to respond to.  A sample with ``plan_variation <
    transient_threshold`` is flagged non-transient and excluded from the gate
    verdict's pass/fail denominator (it cannot fairly test controllability).

    A transient sample PASSES when the actuator plan causally moves the predicted
    tokens (true-vs-zeroed margin > ``margin_threshold``) AND a per-actuator edit
    (gas or NBI) also moves them.  Returns ``(per_sample_verdicts, summary)``.
    """
    dev = torch.device(device or next(model.parameters()).device)
    gas_idx = gas_puff_channel_indices()
    nbi_idx = nbi_channel_indices()

    verdicts: list[ControllabilityVerdict] = []
    for s in samples:
        batch = _batch_to(
            collate_controllable_windows([s], stream_names=list(stream_names)), dev
        )
        full_act = batch["actuator"]

        # how much the actuator plan itself moves across the window + the camera's
        # own frame-to-frame change — proves the window is transient.
        present = np.asarray(s.actuator.missing, dtype=np.float32).mean(axis=0) < 1.0
        plan_vals = np.asarray(s.actuator.values, dtype=np.float64)
        plan_var = (
            float(np.std(plan_vals[:, present], axis=0).sum())
            if bool(present.any()) and plan_vals.shape[0] > 1
            else 0.0
        )
        frames_np = np.asarray(s.frames, dtype=np.int64)
        cam_change = (
            float((frames_np[1:] != frames_np[:-1]).mean())
            if frames_np.shape[0] > 1
            else 0.0
        )
        is_transient = bool(plan_var >= transient_threshold)

        zero_act = _actuator_batch_from_plan(zero_plan(s.actuator), dev)
        # gas/NBI levers SET a fixed physical command (gate-fix 2b) — the window-
        # MAX level ("fire it hard"), a real in-distribution intervention, rather
        # than ×3 of a near-zero baseline (a <1% input nudge).  ``gas_scale`` /
        # ``nbi_scale`` now multiply that window-max level so a caller can push it
        # harder/softer while staying physical (default 1.0 = the window max).
        gas_levels = [
            lv * float(gas_scale)
            for lv in _physical_command_levels(s.actuator, gas_idx)
        ]
        nbi_levels = [
            lv * float(nbi_scale)
            for lv in _physical_command_levels(s.actuator, nbi_idx)
        ]
        gas_up_act = _actuator_batch_from_plan(
            _set_plan_channels_physical(s.actuator, gas_idx, gas_levels), dev
        )
        gas_zero_act = _actuator_batch_from_plan(
            scale_plan_channels(s.actuator, gas_idx, 0.0), dev
        )
        nbi_up_act = _actuator_batch_from_plan(
            _set_plan_channels_physical(s.actuator, nbi_idx, nbi_levels), dev
        )

        # CLEAN-context teacher-forced margins (transparency only — see below).
        m_true_zero = teacher_forced_token_mismatch(
            model, batch, actuator_a=full_act, actuator_b=zero_act, chunk=chunk
        )
        m_gas_scale = teacher_forced_token_mismatch(
            model, batch, actuator_a=full_act, actuator_b=gas_up_act, chunk=chunk
        )
        m_gas_zero = teacher_forced_token_mismatch(
            model, batch, actuator_a=full_act, actuator_b=gas_zero_act, chunk=chunk
        )
        m_nbi_scale = teacher_forced_token_mismatch(
            model, batch, actuator_a=full_act, actuator_b=nbi_up_act, chunk=chunk
        )

        # observation ablation: the redundant-observation effect — full-signals
        # vs zeroed-signals with the SAME (full) actuator plan.  This is the
        # baseline the actuator-plan margin must beat (M3's failure mode was that
        # the observations carry the response, not the controls).
        obs_zero_batch = dict(batch)
        obs_zero_batch["signals"] = {
            k: torch.zeros_like(v) for k, v in batch["signals"].items()
        }
        m_obs = _signal_token_mismatch(
            model, batch, obs_zero_batch, full_act, chunk=chunk
        )

        # CORRUPTED-context margins — the FAIR controllability reading.  The
        # memorised camera context is destroyed so the prediction can only differ
        # via the conditioning prefix; the actuator margin here is genuine.
        cc_frames = _corrupt_all_context(
            batch["frames"],
            context_frames=int(s.context_frames),
            vocab_size=int(model.config.vocab_size),
            seed=(int(s.shot_id) * 100003) ^ 0xC0,
        )
        cc_true_zero = teacher_forced_token_mismatch(
            model,
            batch,
            actuator_a=full_act,
            actuator_b=zero_act,
            chunk=chunk,
            frames_override=cc_frames,
        )
        cc_gas_scale = teacher_forced_token_mismatch(
            model,
            batch,
            actuator_a=full_act,
            actuator_b=gas_up_act,
            chunk=chunk,
            frames_override=cc_frames,
        )
        cc_nbi_scale = teacher_forced_token_mismatch(
            model,
            batch,
            actuator_a=full_act,
            actuator_b=nbi_up_act,
            chunk=chunk,
            frames_override=cc_frames,
        )
        cc_obs = _signal_token_mismatch(
            model,
            batch,
            obs_zero_batch,
            full_act,
            chunk=chunk,
            frames_override=cc_frames,
        )

        denom = max(cc_obs, 1e-6)
        ratio = float(cc_true_zero / denom)
        # The gate decides on the CORRUPTED-context margins (the clean-context
        # ones are ~0 on a memorised overfit regardless of controllability).  A
        # TRANSIENT sample PASSES when, with the camera lookup removed, the
        # actuator plan moves the prediction (cc true-vs-zeroed > threshold) AND a
        # per-actuator edit (gas or NBI) also moves it.
        passed = bool(
            is_transient
            and cc_true_zero > margin_threshold
            and (cc_gas_scale > margin_threshold or cc_nbi_scale > margin_threshold)
        )
        verdicts.append(
            ControllabilityVerdict(
                shot_id=int(s.shot_id),
                true_vs_zeroed_mismatch=m_true_zero,
                gas_scale_mismatch=m_gas_scale,
                gas_zero_mismatch=m_gas_zero,
                nbi_scale_mismatch=m_nbi_scale,
                observation_mismatch=m_obs,
                plan_over_observation_ratio=ratio,
                cc_true_vs_zeroed_mismatch=cc_true_zero,
                cc_gas_scale_mismatch=cc_gas_scale,
                cc_nbi_scale_mismatch=cc_nbi_scale,
                cc_observation_mismatch=cc_obs,
                n_gas_channels=len(gas_idx),
                n_nbi_channels=len(nbi_idx),
                plan_variation=plan_var,
                camera_change_fraction=cam_change,
                is_transient=is_transient,
                passed=passed,
            )
        )

    # The gate is scored ONLY over transient samples (the fair test set); a
    # flat-top window has no control variation and would falsely fail.
    transient = [v for v in verdicts if v.is_transient]
    score_set = transient or verdicts  # fall back if NONE were transient
    n_transient = len(transient)
    n_pass = sum(1 for v in score_set if v.passed)
    mean_true_zero = float(np.mean([v.true_vs_zeroed_mismatch for v in score_set]))
    mean_gas = float(np.mean([v.gas_scale_mismatch for v in score_set]))
    mean_nbi = float(np.mean([v.nbi_scale_mismatch for v in score_set]))
    mean_obs = float(np.mean([v.observation_mismatch for v in score_set]))
    mean_ratio = float(np.mean([v.plan_over_observation_ratio for v in score_set]))
    mean_plan_var = float(np.mean([v.plan_variation for v in score_set]))
    # corrupted-context means — the decision metrics.
    mean_cc_true_zero = float(
        np.mean([v.cc_true_vs_zeroed_mismatch for v in score_set])
    )
    mean_cc_gas = float(np.mean([v.cc_gas_scale_mismatch for v in score_set]))
    mean_cc_nbi = float(np.mean([v.cc_nbi_scale_mismatch for v in score_set]))
    mean_cc_obs = float(np.mean([v.cc_observation_mismatch for v in score_set]))
    # GATE PASS: on a MAJORITY of the TRANSIENT samples, with the memorised
    # camera context removed, the actuator plan moves the camera (cc margin) AND
    # the mean cc margin beats the threshold.  Requires at least one transient
    # window (else the gate was not fairly testable).  The clean-context margin
    # is NOT used — it reads ~0 on a memorised overfit regardless.
    gate_pass = bool(
        n_transient > 0
        and n_pass >= max(1, len(score_set) // 2 + 1)
        and mean_cc_true_zero > margin_threshold
    )
    summary = {
        "n_samples": len(verdicts),
        "n_transient": n_transient,
        "n_pass": n_pass,
        "decision_metric": "corrupted_context_true_vs_zeroed",
        "mean_true_vs_zeroed_mismatch": mean_true_zero,
        "mean_gas_scale_mismatch": mean_gas,
        "mean_nbi_scale_mismatch": mean_nbi,
        "mean_observation_mismatch": mean_obs,
        "mean_plan_over_observation_ratio": mean_ratio,
        "mean_cc_true_vs_zeroed_mismatch": mean_cc_true_zero,
        "mean_cc_gas_scale_mismatch": mean_cc_gas,
        "mean_cc_nbi_scale_mismatch": mean_cc_nbi,
        "mean_cc_observation_mismatch": mean_cc_obs,
        "mean_plan_variation": mean_plan_var,
        "margin_threshold": margin_threshold,
        "transient_threshold": transient_threshold,
        "gas_scale": gas_scale,
        "nbi_scale": nbi_scale,
        "scored_on_transient_only": bool(transient),
        "gate_pass": gate_pass,
        "gate_testable": bool(n_transient > 0),
        "verdict": "PASS" if gate_pass else "FAIL",
    }
    return verdicts, summary


@torch.no_grad()
def _signal_token_mismatch(
    model: ControllableSpacetimeTransformer,
    batch_full: dict,
    batch_zeroed: dict,
    actuator: dict[str, torch.Tensor] | None,
    *,
    chunk: int = 4096,
    frames_override: torch.Tensor | None = None,
) -> float:
    """Next-frame token mismatch between full-signals and zeroed-signals forwards.

    Same actuator + plan + frames; only the measured-signal blocks differ.  This
    is the redundant-observation effect the actuator-plan margin must beat.
    ``frames_override`` swaps the teacher-forced camera history (e.g. a
    corrupted-context copy) for the fair reading on a memorised overfit.
    """
    model.eval()
    frames = frames_override if frames_override is not None else batch_full["frames"]
    plan = batch_full.get("plan")
    with _AutocastCtx(frames.device):
        ha = _forward_with_actuator(
            model, frames, plan, batch_full.get("signals"), actuator
        )
        hb = _forward_with_actuator(
            model, frames, plan, batch_zeroed.get("signals"), actuator
        )
    t = ha.shape[1]
    diffs: list[float] = []
    for ti in range(t):
        pa = model.chunked_argmax_frame(ha[:, ti], chunk=chunk)
        pb = model.chunked_argmax_frame(hb[:, ti], chunk=chunk)
        diffs.append(float((pa != pb).float().mean()))
    model.train()
    return float(np.mean(diffs)) if diffs else float("nan")


# ---------------------------------------------------------------------------
# ΔN-M action-sensitivity gate (autoregressive rollout, decoder-free)
# ---------------------------------------------------------------------------
#
# This is the gate metric the M4 re-design adopts (Vid2World ΔN-M): the
# divergence of an AUTOREGRESSIVE rollout between the TRUE plan and a RANDOM
# plan, scored against a RANDOM-vs-RANDOM noise floor.  The teacher-forced
# token-mismatch gate above probes one-step sensitivity on a frozen history; the
# ΔN-M gate is closer to the PLAY use case — it lets the model consume its OWN
# predicted frames so a plan effect can COMPOUND over the rollout, which is how a
# load-bearing plan would actually steer the dream.  It is computed in TOKEN
# space (a strict lower bound on the decoded-pixel divergence — if the predicted
# tokens do not diverge, the decoded pixels cannot), so it needs no decoder; the
# driver optionally decodes a pair to confirm the token divergence shows in
# pixels.  Crucially it defines the NOISE FLOOR from random-vs-random rollouts:
# PASS requires true-vs-random to clear that floor by a clear margin.


#: Channel keys that are measured STATES / responses, NOT commands — DROPPED from
#: the actuator COMMAND vector (they remain available as v2 observation streams):
#: Ip is the response to the solenoid drive; density is the response to gas
#: fuelling.  You command the coils/solenoid/gas/NBI, not the state.
STATE_CHANNEL_KEYS: frozenset[str] = frozenset({"plasma_current", "ne_line_integrated"})

#: Quasi-static machine constant — kept IN the command vector (it is a real
#: actuator setting) but HELD at the true value in a counterfactual (you don't
#: dial the TF field shot-to-shot for control).
QUASI_STATIC_COMMAND_KEYS: frozenset[str] = frozenset({"tf_current"})


def command_channels(channels=None):
    """The actuator COMMAND channels — the full set minus the measured STATES.

    Drops ``plasma_current`` + ``ne_line_integrated`` (responses, not commands)
    and keeps the PF/CS coil currents + ``sol_current`` (PRIMARY actuators), the
    NBI powers, the gas flows, and ``tf_current``.  Retained for reference / a
    callers who want the filtered list; the corpus trainer instead keeps the FULL
    actuator vector and MASKS the states (+ tf) inside the model's
    :meth:`_plan_summary` — the single conditioning entry point — so train / gate
    / inference stay consistent (see :func:`masked_command_columns`).
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ACTUATOR_CHANNELS,
    )

    chans = list(ACTUATOR_CHANNELS if channels is None else channels)
    return [c for c in chans if c.key not in STATE_CHANNEL_KEYS]


def masked_command_columns(channel_keys: Sequence[str]) -> tuple[int, ...]:
    """Columns to ZERO from the model's conditioning — the NON-command channels.

    The measured STATES (``plasma_current``, ``ne_line_integrated``) PLUS the
    quasi-static ``tf_current`` — i.e. everything that is not a control command.
    Masking these at :meth:`ControllableSpacetimeTransformer._plan_summary` makes
    the model condition ONLY on the commands, so it cannot read the plasma state
    off an always-on Ip context (the cheat/persistence failure mode).  KEY-based
    against the plan's OWN ``channel_keys`` so it is correct for any channel order
    /subset.  Returned as a tuple for the model config (``masked_command_indices``).
    """
    masked = STATE_CHANNEL_KEYS | QUASI_STATIC_COMMAND_KEYS
    return tuple(i for i, key in enumerate(channel_keys) if key in masked)


def _perturbable_command_columns(channel_keys: Sequence[str]) -> list[int]:
    """Columns (into ``channel_keys``) of the actuators a counterfactual PERTURBS.

    KEY-based (robust to any channel subset — the command vector is filtered to 21
    channels, so positional indices into the full 23-channel set are WRONG): the
    perturbable commands are the COILS (PF + solenoid, ``amc`` minus Ip+tf), the
    NBI powers (``anb``), and the gas flows (``aga``).  ``tf_current`` (quasi-
    static) and any state key (Ip/density) are NOT perturbed.  Resolves against the
    plan's OWN ``channel_keys`` so it is correct whether the plan carries the full
    or the filtered command vector.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ACTUATOR_CHANNELS,
    )

    source_by_key = {c.key: c.source for c in ACTUATOR_CHANNELS}
    cols: list[int] = []
    for i, key in enumerate(channel_keys):
        if key in STATE_CHANNEL_KEYS or key in QUASI_STATIC_COMMAND_KEYS:
            continue
        src = source_by_key.get(key)
        # amc (coils+sol, Ip/tf already excluded above), anb (NBI), aga (gas).
        if src in ("amc", "anb", "aga"):
            cols.append(i)
    return cols


def _random_actuator_like(
    plan,
    *,
    rng: np.random.Generator,
    controllable_only: bool = True,
    perturb_scale: float = 0.3,
):
    """A realistic "different COMMANDS, same machine" counterfactual plan.

    The ΔN-M gate contrasts the TRUE plan against a DIFFERENT plan; the difference
    must be in the COMMANDS the operator could have dialled differently — and in a
    tokamak those are the COILS (+ solenoid) first, then NBI + gas
    (:func:`_perturbable_command_columns`, resolved by channel KEY so it is correct
    for the filtered 21-channel command vector too).  PERTURBING THE COILS is the
    point: the model already responds strongly to the coil channels, and that
    response IS the driveability we are testing.

    The perturbation is a BOUNDED, IN-DISTRIBUTION edit of the TRUE trajectory —
    NOT the earlier ~106% OOD re-randomisation (the gate-bug in
    gate_verdict_1220668: a "different machine", not a different actuation,
    inflating the random-vs-random noise floor so the true plan could never clear
    it).  Each perturbable command channel is transformed
    ``new = true*(1 + a) + b*range`` with per-channel ``a ~ U[-perturb_scale,
    +perturb_scale]`` (a bounded gain change) and ``b ~ U[-perturb_scale/2,
    +perturb_scale/2]`` of the channel's own window peak-to-peak range (a bounded
    level shift).  This PRESERVES the temporal SHAPE (a ramp stays a ramp at a
    different slope/offset) — a physically plausible alternative the operator could
    have programmed — rather than i.i.d. step noise.  The non-perturbed channels
    (the measured states Ip/density — if present — and tf, the quasi-static
    constant) are HELD at the true plan: a counterfactual must not fabricate a
    state independent of the commands that drive it.  ``controllable_only=False``
    restores a legacy all-channel re-randomisation (comparison only).  ``missing``
    is preserved so an absent actuator stays absent.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ActuatorPlan,
        normalise_actuator_values,
    )

    raw = np.asarray(plan.raw_values, dtype=np.float64)
    miss = np.asarray(plan.missing, dtype=np.float32)
    present = miss.mean(axis=0) < 1.0  # (C,)
    keys = list(plan.channel_keys)
    out = raw.copy()
    p, c = raw.shape
    s = float(perturb_scale)
    if controllable_only:
        targets = [
            i for i in _perturbable_command_columns(keys) if i < c and present[i]
        ]
        for ch in targets:
            col = raw[:, ch]
            rng_pp = float(col.max() - col.min())  # window peak-to-peak
            if rng_pp <= 0.0:
                rng_pp = abs(float(col.mean())) + 1.0  # flat channel: a floor band
            a = float(rng.uniform(-s, s))  # bounded gain change
            b = float(rng.uniform(-s / 2.0, s / 2.0))  # bounded level shift
            out[:, ch] = col * (1.0 + a) + b * rng_pp
    else:
        # legacy comparison path: i.i.d. Normal over every present channel.
        for ch in [i for i in range(c) if present[i]]:
            col = raw[:, ch]
            mu, sd = float(np.mean(col)), float(np.std(col))
            sd = max(sd, 0.1 * (abs(mu) + 1.0))
            out[:, ch] = rng.normal(mu, sd, size=p)
    return ActuatorPlan(
        values=normalise_actuator_values(out),
        missing=plan.missing.copy(),
        channel_keys=list(plan.channel_keys),
        raw_values=out.astype(np.float32),
    )


@torch.no_grad()
def _random_actuator_batch(
    actuator: dict[str, torch.Tensor] | None,
    *,
    perturbable_cols: Sequence[int],
    generator: torch.Generator,
    perturb_scale: float = 0.3,
) -> dict[str, torch.Tensor] | None:
    """A WRONG-plan actuator batch for the action-contrastive negative (tensor-level).

    Mirrors :func:`_random_actuator_like` on the COLLATED, already-NORMALISED batch
    tensors: each PERTURBABLE command column (coils + solenoid + NBI + gas; states
    Ip/density + the quasi-static tf are HELD) is transformed
    ``new = true*(1 + a) + b*range`` with per-(sample, channel) ``a ~
    U[-s, s]`` (a bounded gain change) and ``b ~ U[-s/2, s/2]`` of the channel's own
    per-sample window peak-to-peak range (a bounded level shift) — an
    IN-DISTRIBUTION counterfactual that preserves the temporal shape, NOT i.i.d.
    noise.  ``missing`` is preserved.  Returns ``None`` when there is no actuator /
    no perturbable column (the caller then skips the action-contrastive term).
    """
    if actuator is None:
        return None
    vals = actuator.get("values")
    miss = actuator.get("missing")
    if vals is None or vals.ndim != 3:
        return None
    b, p, c = vals.shape
    cols = [int(i) for i in perturbable_cols if 0 <= int(i) < c]
    if not cols:
        return None
    out = vals.clone()
    s = float(perturb_scale)
    col_idx = torch.as_tensor(cols, dtype=torch.long, device=vals.device)
    sub = vals[:, :, col_idx]  # (B, P, n_cols)
    rng_pp = (sub.amax(dim=1) - sub.amin(dim=1)).clamp_min(1e-6)  # (B, n_cols)
    a = (
        torch.rand(b, len(cols), generator=generator, device=vals.device) * 2.0 - 1.0
    ) * s  # (B, n_cols) in [-s, s]
    bsh = (
        torch.rand(b, len(cols), generator=generator, device=vals.device) - 0.5
    ) * s  # (B, n_cols) in [-s/2, s/2]
    new_sub = sub * (1.0 + a[:, None, :]) + (bsh * rng_pp)[:, None, :]
    out[:, :, col_idx] = new_sub
    return {"values": out, "missing": miss.clone() if miss is not None else None}


@torch.no_grad()
def _argmax_token_rollout(
    model: ControllableSpacetimeTransformer,
    sample: ControllableSpacetimeSample,
    stream_names: Sequence[str],
    actuator_batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    chunk: int = 4096,
) -> np.ndarray:
    """Autoregressive argmax token rollout under a FIXED actuator plan.

    Keeps the leading ``context_frames`` frames as truth, then rolls forward
    consuming its OWN predicted frames while conditioning on the plan + signals +
    the supplied actuator drive (via AdaLN) at every step.  Returns ``(T, S)``
    LOCAL token ids.  Decoder-free — the divergence between two such rollouts is a
    strict lower bound on the decoded-pixel divergence.
    """
    model.eval()
    ctx = int(sample.context_frames)
    t_total = int(np.asarray(sample.frames).shape[0])
    batch = _batch_to(
        collate_controllable_windows([sample], stream_names=list(stream_names)), device
    )
    plan = batch.get("plan")
    signals = batch.get("signals")
    gen = np.asarray(sample.frames, dtype=np.int64).copy()
    for ti in range(ctx, t_total):
        cur = torch.as_tensor(gen[:ti][None], dtype=torch.long, device=device)
        with _AutocastCtx(device):
            hidden = model._forward_tokens(cur, plan, signals, actuator=actuator_batch)
        pred = model.chunked_argmax_frame(hidden[:, ti - 1], chunk=chunk)
        gen[ti] = pred[0].cpu().numpy().astype(np.int64)
    return gen


@dataclass
class DeltaNMVerdict:
    """Per-sample ΔN-M action-sensitivity verdict (autoregressive token rollout)."""

    shot_id: int
    is_transient: bool
    plan_variation: float
    # mean over random draws of the forecast-window token-divergence between the
    # TRUE-plan rollout and a RANDOM-plan rollout (the action signal).
    true_vs_random: float
    # mean pairwise forecast-window token-divergence among RANDOM-plan rollouts
    # (the noise floor — how much the rollout moves under DIFFERENT wrong plans,
    # which bounds what a true-vs-random reading must beat to be real).
    random_vs_random: float
    # the margin (true_vs_random - random_vs_random) and the ratio.
    margin: float
    ratio: float
    n_random: int
    passed: bool

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "is_transient": self.is_transient,
            "plan_variation": self.plan_variation,
            "true_vs_random": self.true_vs_random,
            "random_vs_random": self.random_vs_random,
            "margin": self.margin,
            "ratio": self.ratio,
            "n_random": self.n_random,
            "passed": self.passed,
        }


def delta_nm_gate(
    model: ControllableSpacetimeTransformer,
    samples: Sequence[ControllableSpacetimeSample],
    stream_names: Sequence[str],
    *,
    device: str | None = None,
    chunk: int = 4096,
    n_random: int = 4,
    margin_threshold: float = 0.02,
    floor_ratio: float = 1.5,
    transient_threshold: float = 1e-3,
    seed: int = 0,
) -> tuple[list[DeltaNMVerdict], dict]:
    """Vid2World ΔN-M action-sensitivity gate over autoregressive token rollouts.

    For each TRANSIENT sample, roll out the model autoregressively under the TRUE
    actuator plan and under ``n_random`` RANDOM plans (:func:`_random_actuator_like`,
    matched marginal scale).  The forecast-window token-divergence between the
    true rollout and each random rollout is the ACTION signal (``true_vs_random``,
    averaged over draws); the mean PAIRWISE divergence among the random rollouts
    is the NOISE FLOOR (``random_vs_random`` — how much the rollout moves under
    DIFFERENT wrong plans).

    A sample PASSES when the action signal clears the floor by a clear margin:
    ``true_vs_random - random_vs_random > margin_threshold`` AND
    ``true_vs_random > floor_ratio * random_vs_random``.  The gate is scored ONLY
    over transient windows (a flat-top has no plan variation to respond to).
    Returns ``(per_sample_verdicts, summary)``.
    """
    dev = torch.device(device or next(model.parameters()).device)

    def _fc_divergence(a: np.ndarray, b: np.ndarray, ctx: int) -> float:
        # fraction of forecast-window tokens that differ between two rollouts.
        if a.shape[0] <= ctx:
            return 0.0
        return float((a[ctx:] != b[ctx:]).mean())

    verdicts: list[DeltaNMVerdict] = []
    for si, s in enumerate(samples):
        present = np.asarray(s.actuator.missing, dtype=np.float32).mean(axis=0) < 1.0
        plan_vals = np.asarray(s.actuator.values, dtype=np.float64)
        plan_var = (
            float(np.std(plan_vals[:, present], axis=0).sum())
            if bool(present.any()) and plan_vals.shape[0] > 1
            else 0.0
        )
        is_transient = bool(plan_var >= transient_threshold)
        ctx = int(s.context_frames)

        full_act = _actuator_batch_from_plan(s.actuator, dev)
        true_roll = _argmax_token_rollout(
            model, s, stream_names, full_act, dev, chunk=chunk
        )
        rng = np.random.default_rng((int(s.shot_id) * 1_000_003) ^ (seed * 31) ^ si)
        rand_rolls: list[np.ndarray] = []
        for _ in range(int(n_random)):
            rplan = _random_actuator_like(s.actuator, rng=rng)
            ract = _actuator_batch_from_plan(rplan, dev)
            rand_rolls.append(
                _argmax_token_rollout(model, s, stream_names, ract, dev, chunk=chunk)
            )
        tvr = float(np.mean([_fc_divergence(true_roll, r, ctx) for r in rand_rolls]))
        pair: list[float] = []
        for i in range(len(rand_rolls)):
            for j in range(i + 1, len(rand_rolls)):
                pair.append(_fc_divergence(rand_rolls[i], rand_rolls[j], ctx))
        rvr = float(np.mean(pair)) if pair else 0.0
        margin = tvr - rvr
        ratio = float("inf") if rvr == 0.0 else tvr / rvr
        passed = bool(
            is_transient
            and margin > margin_threshold
            and (rvr == 0.0 and tvr > margin_threshold or ratio > floor_ratio)
        )
        verdicts.append(
            DeltaNMVerdict(
                shot_id=int(s.shot_id),
                is_transient=is_transient,
                plan_variation=plan_var,
                true_vs_random=tvr,
                random_vs_random=rvr,
                margin=margin,
                ratio=ratio,
                n_random=int(n_random),
                passed=passed,
            )
        )

    transient = [v for v in verdicts if v.is_transient]
    score_set = transient or verdicts
    n_transient = len(transient)
    n_pass = sum(1 for v in score_set if v.passed)
    mean_tvr = float(np.mean([v.true_vs_random for v in score_set]))
    mean_rvr = float(np.mean([v.random_vs_random for v in score_set]))
    mean_margin = float(np.mean([v.margin for v in score_set]))
    finite_ratios = [v.ratio for v in score_set if np.isfinite(v.ratio)]
    mean_ratio = float(np.mean(finite_ratios)) if finite_ratios else float("inf")
    gate_pass = bool(
        n_transient > 0
        and n_pass >= max(1, len(score_set) // 2 + 1)
        and mean_margin > margin_threshold
    )
    summary = {
        "metric": "delta_nm_autoregressive_token_rollout",
        "n_samples": len(verdicts),
        "n_transient": n_transient,
        "n_pass": n_pass,
        "mean_true_vs_random": mean_tvr,
        "mean_random_vs_random_noise_floor": mean_rvr,
        "mean_margin": mean_margin,
        "mean_ratio": mean_ratio,
        "n_random": int(n_random),
        "margin_threshold": margin_threshold,
        "floor_ratio": floor_ratio,
        "transient_threshold": transient_threshold,
        "scored_on_transient_only": bool(transient),
        "gate_testable": bool(n_transient > 0),
        "gate_pass": gate_pass,
        "verdict": "PASS" if gate_pass else "FAIL",
    }
    return verdicts, summary


# ---------------------------------------------------------------------------
# CLI (the corpus re-train entrypoint — torchrun -m ... corpus)
# ---------------------------------------------------------------------------


def _train_shots_from_manifest(manifest_path: Path, held_out: set[int]) -> list[int]:
    """Unique train shot ids from a curated-window manifest, minus held-out.

    The curated manifest (built by the excitation-corpus selector) lists the
    excited windows; its distinct ``shot_id`` set is the train pool.  Held-out
    shots are excluded at SELECT time already, but we subtract them again here so
    the disjointness is enforced at BOTH ends (defence in depth).
    """
    import json  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    data = json.loads(_Path(manifest_path).read_text())
    windows = data.get("windows", [])
    seen: list[int] = []
    s: set[int] = set()
    for w in windows:
        sid = int(w["shot_id"])
        if sid in held_out or sid in s:
            continue
        s.add(sid)
        seen.append(sid)
    return seen


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    p = argparse.ArgumentParser(description="Controllable camera-model trainer.")
    sub = p.add_subparsers(dest="command")
    pc = sub.add_parser("corpus", help="DDP re-train on the curated excited corpus")
    pc.add_argument(
        "--manifest",
        default="/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_full.json",
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
        default="/work/projects/imas_gpu/worldmodel/ckpt/spacetime-corruption-1220391/best.pt",
        help="warm-start the backbone from the M2 forecaster (strict=False)",
    )
    # window
    pc.add_argument("--n-frames", type=int, default=24)
    pc.add_argument("--n-plan", type=int, default=8)
    pc.add_argument("--context-frames", type=int, default=8)
    pc.add_argument("--frame-stride", type=int, default=1)
    pc.add_argument(
        "--target-horizon-s",
        type=float,
        default=0.25,
        help="physical seconds the n_frames window spans (per-shot stride derived "
        "from each shot's fps); 0 = use the literal --frame-stride",
    )
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
    # model
    pc.add_argument("--d-model", type=int, default=None)
    pc.add_argument("--n-layers", type=int, default=None)
    pc.add_argument("--n-heads", type=int, default=None)
    pc.add_argument("--d-ff", type=int, default=None)
    pc.add_argument("--dropout", type=float, default=None)
    pc.add_argument("--adaln-hidden", type=int, default=256)
    # M4 controllability machinery
    pc.add_argument("--observation-dropout", type=float, default=0.8)
    pc.add_argument(
        "--no-per-modality-masking",
        action="store_true",
        help="DISABLE per-stream independent observation masking + clean diagnostic "
        "targets (the ablation): fall back to ALL-OR-NOTHING dropout that zeroes the "
        "WHOLE signal dict for a flagged sample (so the diagnostic CE is 0 on ~"
        "observation-dropout of steps).  Default ON — each stream is masked "
        "independently and the model dreams a masked stream from the others + "
        "commands + cameras against the clean target.",
    )
    pc.add_argument("--control-dropout", type=float, default=0.15)
    pc.add_argument("--hb-noise-std", type=float, default=1.0)
    pc.add_argument("--hb-mask-prob", type=float, default=0.5)
    pc.add_argument("--hb-max-strength", type=float, default=1.0)
    pc.add_argument("--hb-clean-fraction", type=float, default=0.2)
    pc.add_argument("--inverse-dynamics-weight", type=float, default=1.0)
    pc.add_argument(
        "--diagnostic-weight",
        type=float,
        default=0.5,
        help="weight of the next-step diagnostic cross-entropy (joint generation): "
        "the model dreams cameras AND diagnostics.  Warmed up linearly over "
        "--diagnostic-weight-warmup-frac of training.  0 keeps the heads built but "
        "the objective off (a weight-ablation).",
    )
    pc.add_argument(
        "--diagnostic-weight-warmup-frac",
        type=float,
        default=0.1,
        help="fraction of total steps over which the diagnostic CE weight ramps "
        "linearly from 0 to --diagnostic-weight (the CE can destabilise the camera "
        "loss early).",
    )
    pc.add_argument(
        "--no-generate-diagnostics",
        action="store_true",
        help="DISABLE joint generation: build no per-stream diagnostic heads and add "
        "no diagnostic objective (the ablation off-switch — byte-identical camera-"
        "only forecaster).  Default ON.",
    )
    # AUXILIARY contrastive objectives (all OFF by default — the §6 ablation grid
    # enables each independently to isolate its marginal contribution).
    pc.add_argument(
        "--cross-modal",
        action="store_true",
        help="ENABLE the cross-modal alignment auxiliary: a camera<->diagnostic "
        "InfoNCE over the batch (needs measured signals).  Default OFF.",
    )
    pc.add_argument("--cross-modal-weight", type=float, default=0.1)
    pc.add_argument(
        "--self-predictive",
        action="store_true",
        help="ENABLE the self-predictive (BYOL/SPR) latent auxiliary: an online "
        "predictor maps the current camera latent to a stop-grad projection of the "
        "future latent (no negatives).  Default OFF.",
    )
    pc.add_argument("--self-predictive-weight", type=float, default=0.1)
    pc.add_argument(
        "--action-contrastive",
        action="store_true",
        help="ENABLE the action-contrastive auxiliary: the true-plan next-state "
        "latent must sit CLOSER to the realised next state than a random-plan one "
        "(a second forward under a random plan; directly trains ΔN-M).  Default OFF.",
    )
    pc.add_argument("--action-contrastive-weight", type=float, default=0.1)
    pc.add_argument("--scheduled-sampling-max", type=float, default=0.25)
    pc.add_argument("--scheduled-sampling-ramp", type=float, default=0.5)
    pc.add_argument(
        "--no-excitation-weighting",
        action="store_true",
        help="DISABLE excitation weighting of the next-frame CE (the ablation "
        "baseline: uniform per-frame loss).  Default ON — up-weights forecast "
        "frames where the coils move so flat-top frames cannot drown the "
        "coil->plasma gradient on a full-shot window.",
    )
    pc.add_argument(
        "--excitation-weight-floor",
        type=float,
        default=0.1,
        help="baseline per-frame weight a NON-moving (flat-top) frame keeps "
        "(0..1); moving frames scale up to 1.0.  0 zeroes quiet frames entirely.",
    )
    pc.add_argument(
        "--excitation-weight-power",
        type=float,
        default=1.0,
        help="sharpen (>1) or soften (<1) the excitation-weight contrast.",
    )
    pc.add_argument(
        "--keep-state-channels",
        action="store_true",
        help="condition on the FULL actuator vector incl plasma_current + density "
        "+ tf (default MASKS those out of the model's conditioning — they are "
        "measured states/constant, not commands; they remain v2 observations)",
    )
    # multi-timescale / multi-camera / extended-signal conditioning (all OFF by
    # default so an unflagged run is byte-identical to the single-camera model).
    pc.add_argument(
        "--timescale-conditioning",
        action="store_true",
        help="condition on each window's per-frame log-Δt (cadence) — lets ONE "
        "model span the ~250x cadence range of the unified corpus (slow full-shot "
        "+ fast bursts).  Zero-init head, so a warm-started model starts as the "
        "forecaster.  Default OFF (cadence-blind, byte-identical to the prior model).",
    )
    pc.add_argument(
        "--camera-conditioning",
        action="store_true",
        help="add a per-camera embedding to the frame tokens so ONE model can "
        "ingest all 5 cameras (rbb/rco/rgb/rgc/rba).  Zero-init table, so a "
        "warm-started model starts view-blind.  Default OFF.",
    )
    pc.add_argument(
        "--signal-modalities",
        choices=("default", "extended"),
        default="default",
        help="'default' = the current measured-signal set; 'extended' adds the "
        "already-tokenised HF streams (xsx / xim / ait) the camera world model "
        "does not yet fuse.  Default 'default'.",
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
        win_list = None  # explicit shots -> legacy centred/random-window path
    else:
        # MANIFEST path (option i): read the curated EXCITED windows (B's
        # start_frame + per-shot stride spanning ~target_horizon_s) — the horizon
        # fix.  train_shots is derived from the windows (for the result count).
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
        "controllable re-train: %d %s (held-out %s DISJOINT) from %s "
        "[timescale_cond=%s camera_cond=%s signals=%s]",
        len(train_shots),
        "excited windows" if win_list is not None else "train shots",
        sorted(held_out),
        args.manifest if not args.shots.strip() else "--shots",
        args.timescale_conditioning,
        args.camera_conditioning,
        args.signal_modalities,
    )

    model_kwargs: dict = {"adaln_hidden": args.adaln_hidden}
    # When warm-starting from the forecaster, ADOPT its architecture dims so the
    # warm start actually loads (a size mismatch loads 0 tensors and trains a fresh
    # small model — the bug that wasted job 1220852).  An explicit --d-model/etc.
    # overrides the checkpoint (train-from-scratch at a chosen size).
    if args.init_checkpoint:
        ck_arch = architecture_from_checkpoint(_Path(args.init_checkpoint))
        ck_arch.pop("corruption_levels", None)  # the history-bottleneck sets levels
        for k, v in ck_arch.items():
            model_kwargs.setdefault(k, v)
        logger.info("adopting forecaster architecture from checkpoint: %s", ck_arch)
    for k in ("d_model", "n_layers", "n_heads", "d_ff", "dropout"):
        v = getattr(args, k, None)
        if v is not None:
            model_kwargs[k] = v  # explicit CLI override wins

    # Δt / camera conditioning reach the model through ControllableSpacetimeConfig
    # (built from model_kwargs); both default OFF so an unflagged run is unchanged.
    model_kwargs["timescale_conditioning"] = bool(args.timescale_conditioning)
    model_kwargs["camera_conditioning"] = bool(args.camera_conditioning)

    # extended signal modalities add the already-tokenised HF streams (xsx/xim/ait)
    # — the dataset reads them and the model embeds them; 'default' keeps today's set.
    modalities = (
        extended_signal_modalities()
        if args.signal_modalities == "extended"
        else default_signal_modalities()
    )

    cfg = ControllableCorpusConfig(
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
        observation_dropout=args.observation_dropout,
        per_modality_masking=not args.no_per_modality_masking,
        control_dropout=args.control_dropout,
        history_bottleneck=HistoryBottleneckConfig(
            noise_std=args.hb_noise_std,
            mask_prob=args.hb_mask_prob,
            max_strength=args.hb_max_strength,
            clean_fraction=args.hb_clean_fraction,
        ),
        inverse_dynamics_weight=args.inverse_dynamics_weight,
        generate_diagnostics=not args.no_generate_diagnostics,
        diagnostic_weight=args.diagnostic_weight,
        diagnostic_weight_warmup_frac=args.diagnostic_weight_warmup_frac,
        cross_modal=args.cross_modal,
        cross_modal_weight=args.cross_modal_weight,
        self_predictive=args.self_predictive,
        self_predictive_weight=args.self_predictive_weight,
        action_contrastive=args.action_contrastive,
        action_contrastive_weight=args.action_contrastive_weight,
        scheduled_sampling_max=args.scheduled_sampling_max,
        scheduled_sampling_ramp=args.scheduled_sampling_ramp,
        excitation_weighting=not args.no_excitation_weighting,
        excitation_weight_floor=args.excitation_weight_floor,
        excitation_weight_power=args.excitation_weight_power,
        drop_state_channels=not args.keep_state_channels,
        window=SpacetimeWindowConfig(
            n_frames=args.n_frames,
            n_plan=args.n_plan,
            context_frames=args.context_frames,
            frame_stride=args.frame_stride,
            target_horizon_s=args.target_horizon_s,
        ),
        modalities=modalities,
        model_kwargs=model_kwargs,
        init_checkpoint=_Path(args.init_checkpoint) if args.init_checkpoint else None,
    )
    result = train_controllable_corpus(
        train_shots,
        camera=args.camera,
        config=cfg,
        out_dir=_Path(args.out_dir) if args.out_dir else None,
        token_root=_Path(args.token_root) if args.token_root else None,
        manifest_windows=win_list,
    )
    logger.info(
        "controllable re-train done: %d steps, init=%.4f final=%.4f, ckpt=%s",
        result.steps_run,
        result.initial_loss,
        result.final_loss,
        result.checkpoint_path,
    )
    return 0


__all__ = [
    "ControllabilityVerdict",
    "ControllableCorpusConfig",
    "ControllableCorpusResult",
    "DeltaNMVerdict",
    "OverfitControllableConfig",
    "build_controllable_model",
    "collate_actuator",
    "collate_controllable_windows",
    "command_channels",
    "controllability_gate",
    "delta_nm_gate",
    "load_controllable_model_from_checkpoint",
    "main",
    "masked_command_columns",
    "overfit_controllable",
    "save_controllable_checkpoint",
    "teacher_forced_token_mismatch",
    "train_controllable_corpus",
]


if __name__ == "__main__":
    raise SystemExit(main())
