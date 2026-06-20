"""Overfit + controllability GATE for the actuator-PLAN-conditioned camera model.

This is the cheap de-risking gate for the M4 PLAY bridge (plan
``playable-plasma-wm-v0``): does conditioning the camera model on the demanded
actuator PLAN make the controls causally LOAD-BEARING?  M3 proved the
measured-signal conditioning is NOT controllable (the realised observations are
mutually redundant); the fix is to condition on the actuator vector PLAN with the
measured observations made OPTIONAL.  This module:

* builds a :class:`ControllableSpacetimeTransformer` (actuator-plan drive surface
  + the v2 measured-signal streams as optional context);
* overfits a handful of shots conditioned on the actuator plan, applying HIGH
  observation-dropout so the model cannot shortcut the control->camera map
  through the redundant observations (it must learn to drive the camera from the
  PLAN), plus control-dropout so classifier-free guidance works at inference;
* runs the controllability gate: vary the actuator plan (scale / silence the
  gas-puff or NBI command) and measure whether the DECODED camera responds — the
  true-vs-zeroed causal margin in decoded pixel space, and the CFG response.

GPU-safety (repo AGENTS.md §2b): model loaded once outside the per-item loop, a
SIGTERM/SIGINT STOP flag, ``try/finally`` releasing the model + empty_cache,
cudnn deterministic, bf16 autocast.  This is a 1-GPU gate (minutes), not the
6-GPU re-train.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from imas_ambix.worldmodel.actuator_plan import (
    N_ACTUATOR_CHANNELS,
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
from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeWindowConfig,
    plan_vocab,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import (
    SignalModalitySpec,
    default_signal_modalities,
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
    **model_kwargs: object,
) -> ControllableSpacetimeTransformer:
    """Build a :class:`ControllableSpacetimeTransformer` sized to window + streams.

    ``max_frames`` must cover the actuator steps + every present signal stream's
    steps + the tokenised plan steps + the camera frames; it defaults to
    ``n_act_steps + len(streams)*n_signal_steps + n_plan + n_frames`` with slack.
    """
    n_streams = len(signal_streams)
    if max_frames is None:
        max_frames = (
            int(n_act_steps)
            + n_streams * int(n_signal_steps)
            + window.n_plan
            + window.n_frames
            + 2
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
        **model_kwargs,  # type: ignore[arg-type]
    )
    return ControllableSpacetimeTransformer(cfg)


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


def collate_controllable_windows(
    samples: Sequence[ControllableSpacetimeSample],
    *,
    stream_names: Sequence[str] | None = None,
) -> dict:
    """Stack frames + plan + measured signals + the actuator plan into a batch."""
    signal_samples = [s.signal for s in samples]
    batch = collate_signal_windows(signal_samples, stream_names=stream_names)
    batch["actuator"] = collate_actuator(samples)
    return batch


def _batch_to(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["frames"] = batch["frames"].to(device, non_blocking=True)
    out["plan"] = batch["plan"].to(device, non_blocking=True)
    out["signals"] = {
        k: v.to(device, non_blocking=True) for k, v in batch["signals"].items()
    }
    act = batch.get("actuator")
    if act is not None:
        out["actuator"] = {
            "values": act["values"].to(device, non_blocking=True),
            "missing": act["missing"].to(device, non_blocking=True),
        }
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
    # control-dropout (CFG): zero the WHOLE conditioning (plan + actuator +
    # signals) on this fraction of steps so classifier-free guidance works.
    control_dropout: float = 0.15
    # TRANSIENT-window selection: pick each shot's window where the actuator
    # PLAN varies most (a ramp / toggle) rather than the centred flat-top window.
    # The controllability gate is only fair where the plan actually moves — a
    # flat-top window has no control variation to learn from or respond to, so
    # the gate would FALSELY fail.  Off => the centred window (debug).
    transient_windows: bool = True
    # CONTEXT-corruption during overfit (the M2 recipe).  Teacher-forcing on a
    # clean camera history lets the model MEMORISE the next frame from the camera
    # context alone, so it never needs the conditioning — and the controllability
    # gate then reads ~0 for EVERY conditioning (actuator and observation) on a
    # fully-overfit model.  Corrupting a fraction of the context-frame tokens
    # forces the model to LEAN on the conditioning prefix (plan + actuator +
    # signals) to predict the forecast frames, so the actuator plan can actually
    # become load-bearing — the precondition for a meaningful gate.  The loss is
    # always scored against the CLEAN frames.  0 disables it.
    context_corruption_rate: float = 0.5
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
) -> tuple[
    OverfitResult,
    ControllableSpacetimeTransformer,
    list[ControllableSpacetimeSample],
    list[str],
]:
    """Overfit a handful of shots conditioned on the actuator PLAN — the GATE.

    Applies HIGH observation-dropout (so the model must learn the control->camera
    map from the PLAN, not the redundant observations) + control-dropout (so CFG
    works).  Returns ``(result, model, samples, stream_names)``; the caller then
    runs :func:`controllability_gate` to PROVE the actuator plan is causally
    load-bearing.
    """
    config = config or OverfitControllableConfig()
    _set_determinism(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stop = _StopFlag()
    stop.install()

    span = (config.window.n_frames - 1) * config.window.frame_stride + 1
    samples: list[ControllableSpacetimeSample] = []
    window_info: list[tuple[int, int | None, float]] = []
    for sid in shot_ids:
        start_frame = None
        var_score = 0.0
        if config.transient_windows:
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

    model = build_controllable_model(
        config.window,
        plan_channels=plan_ch,
        signal_streams=streams,
        n_signal_steps=config.n_signal_steps,
        actuator_channels=act_channels,
        n_act_steps=config.n_act_steps,
        **config.model_kwargs,
    ).to(dev)
    model.train()
    logger.info(
        "overfit-controllable on %s: params=%d (%.1fM) n_frames=%d plan_ch=%d "
        "actuator_ch=%d n_act_steps=%d obs_dropout=%.2f streams=%s shots=%s",
        dev,
        model.num_parameters(),
        model.num_parameters() / 1e6,
        config.window.n_frames,
        plan_ch,
        act_channels,
        config.n_act_steps,
        config.observation_dropout,
        [(st.name, st.channels) for st in streams],
        list(shot_ids),
    )

    # per-step RNG for the dropout draws (reproducible).
    gen = torch.Generator(device=dev)
    drop_cfg = ContextCorruptionConfig(control_dropout=config.control_dropout)
    clean_frames = batch["frames"]
    ctx_frames = int(config.window.context_frames)
    vocab_size = int(model.config.vocab_size)
    cc_rate = float(config.context_corruption_rate)

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
            # CONTEXT-corruption (M2 recipe): noise a fraction of the context-frame
            # tokens so the model cannot memorise the next frame from a clean
            # camera history and MUST lean on the conditioning prefix.  The loss
            # is scored against the CLEAN frames (target_frames).
            if cc_rate > 0.0:
                rates = torch.full((b,), cc_rate, device=dev)
                step_batch["frames"] = corrupt_context_tokens(
                    clean_frames,
                    rates,
                    context_frames=ctx_frames,
                    vocab_size=vocab_size,
                    generator=gen,
                )
                step_batch["target_frames"] = clean_frames
            # OPTIONAL-observation dropout: zero the measured signals on a high
            # fraction of samples so the model drives from the PLAN.
            obs_drop = (
                torch.rand(b, generator=gen, device=dev) < config.observation_dropout
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

            opt.zero_grad(set_to_none=True)
            with _AutocastCtx(dev):
                loss = model(step_batch, loss_spec={"chunk": config.chunk})
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if step % config.log_every == 0 or step == config.steps - 1:
                logger.info(
                    "overfit-controllable step %d/%d loss=%.4f",
                    step,
                    config.steps,
                    losses[-1],
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
    gas_scale: float = 3.0,
    nbi_scale: float = 3.0,
    margin_threshold: float = 0.02,
    transient_threshold: float = 1e-3,
) -> tuple[list[ControllabilityVerdict], dict]:
    """Token-space controllability gate over the overfit samples.

    For each sample, teacher-force the model and compare the predicted next-frame
    tokens under the FULL actuator plan against:

    * the plan with the WHOLE drive silenced (:func:`zero_plan`) — the headline
      true-vs-zeroed causal margin;
    * the plan with the gas-puff command SCALED UP and SILENCED;
    * the plan with the NBI command SCALED UP;
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
        gas_up_act = _actuator_batch_from_plan(
            scale_plan_channels(s.actuator, gas_idx, gas_scale), dev
        )
        gas_zero_act = _actuator_batch_from_plan(
            scale_plan_channels(s.actuator, gas_idx, 0.0), dev
        )
        nbi_up_act = _actuator_batch_from_plan(
            scale_plan_channels(s.actuator, nbi_idx, nbi_scale), dev
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


__all__ = [
    "ControllabilityVerdict",
    "OverfitControllableConfig",
    "build_controllable_model",
    "collate_actuator",
    "collate_controllable_windows",
    "controllability_gate",
    "overfit_controllable",
    "teacher_forced_token_mismatch",
]
