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
    **model_kwargs: object,
) -> ControllableSpacetimeTransformer:
    """Build a :class:`ControllableSpacetimeTransformer` sized to window + streams.

    The actuator plan is injected via per-block AdaLN (NOT prepended as temporal
    frames), so ``max_frames`` only needs to cover the tokenised plan steps +
    every present signal stream's steps + the camera frames; it defaults to
    ``len(streams)*n_signal_steps + n_plan + n_frames`` with slack.
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
        )
    model.train()
    mixed = frames.clone()
    # predict each forecast frame t (>= ctx) from hidden[:, t-1] and splice it in
    # with per-(sample, frame) Bernoulli(prob).
    for ti in range(ctx, t):
        pred = model.chunked_argmax_frame(hidden[:, ti - 1], chunk=chunk)  # (b, s)
        take = (
            torch.rand(b, generator=generator, device=frames.device) < prob
        )  # (b,)
        if bool(take.any()):
            mixed[take, ti] = pred[take]
    return mixed


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

    # The history bottleneck quantises its per-frame strength to bins that feed
    # the model's history-corruption LEVEL embedding, so size the model's levels
    # to the bottleneck's (when the bottleneck is enabled).
    hb = config.history_bottleneck
    corruption_levels = int(hb.levels) if hb.enabled else 0
    model = build_controllable_model(
        config.window,
        plan_channels=plan_ch,
        signal_streams=streams,
        n_signal_steps=config.n_signal_steps,
        actuator_channels=act_channels,
        n_act_steps=config.n_act_steps,
        corruption_levels=corruption_levels,
        inverse_dynamics=config.inverse_dynamics_weight > 0.0,
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

            opt.zero_grad(set_to_none=True)
            with _AutocastCtx(dev):
                loss = model(
                    step_batch,
                    loss_spec={
                        "chunk": config.chunk,
                        "context_frames": ctx_frames,
                        "inverse_dynamics_weight": config.inverse_dynamics_weight,
                    },
                )
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
    control_dropout: float = 0.15
    history_bottleneck: HistoryBottleneckConfig = field(
        default_factory=HistoryBottleneckConfig
    )
    inverse_dynamics_weight: float = 1.0
    scheduled_sampling_max: float = 0.25
    scheduled_sampling_ramp: float = 0.5
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
) -> ControllableCorpusResult:
    """DDP re-train of the controllable model on a CORPUS of EXCITED shots.

    The 6-GPU re-train the de-risk gate cleared the way for: warm-start the M2
    forecaster, fine-tune on the curated excited corpus with the camera-history
    bottleneck + inverse-dynamics + scheduled-sampling + the actuator-PLAN drive,
    so the plan becomes load-bearing for a FREE-RUNNING rollout (not just the
    one-step prediction).  Reuses the v2 DDP primitives (``DistEnv``, the LR
    scheduler, checkpoint/resume, the distributed sampler) verbatim; the model,
    dataset, and per-step controllability machinery are the controllable ones.

    DDP-aware: each rank pins its card and trains a disjoint shot shard; the
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
    from imas_ambix.worldmodel.spacetime_train_v2 import (  # noqa: PLC0415
        save_checkpoint_v2,
    )

    config = config or ControllableCorpusConfig()
    if out_dir is None:
        import os  # noqa: PLC0415

        run_id = (
            os.environ.get("SLURM_JOB_ID")
            or os.environ.get("WM_RUN_ID")
            or "local"
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

    # Probe plan-channels + signal-stream widths + actuator-channel count.
    probe: list[ControllableSpacetimeSample] = []
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
                )
            )
        except (ValueError, FileNotFoundError, KeyError):
            continue
    if not probe:
        raise ValueError("no shot assembled in the probe — cannot size the model")
    plan_ch = _plan_channels_for([p.signal for p in probe])
    channels = probe_signal_channels(
        list(shot_ids)[:16],
        config.window,
        config.modalities,
        config.n_signal_steps,
        camera=camera,
        token_root=token_root,
    )
    act_channels = int(probe[0].actuator.n_channels)

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
        # observation dropout — drive from the PLAN, not the redundant signals.
        d_gen = torch.Generator(device=dev).manual_seed(gen_seed ^ 0x4F4253)
        obs_drop = (
            torch.rand(b, generator=d_gen, device=dev) < config.observation_dropout
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
                    batch["target_frames"] = batch.get(
                        "target_frames", batch["frames"]
                    )
                is_boundary = (micro + 1) % accum == 0
                sync_ctx = (
                    model.no_sync()
                    if (env.enabled and not is_boundary)
                    else contextlib.nullcontext()
                )
                with sync_ctx, _AutocastCtx(dev):
                    loss = model(
                        batch,
                        loss_spec={
                            "chunk": config.chunk,
                            "context_frames": ctx_frames,
                            "inverse_dynamics_weight": config.inverse_dynamics_weight,
                        },
                    )
                    scaled = loss / accum
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
                        "controllable-corpus step %d/%d loss=%.4f lr=%.3e "
                        "ss=%.2f (%.2f st/s world=%d)",
                        step,
                        config.steps,
                        losses[-1],
                        opt.param_groups[0]["lr"],
                        ss_prob,
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
        ckpt_path = save_checkpoint_v2(
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


def _controllable_actuator_indices() -> list[int]:
    """Columns of the CONTROLLABLE actuators — gas + NBI + density.

    These are the commands an operator dials in for a fixed machine geometry
    (gas-puff valves, NBI beam powers, the density target).  The complement —
    the PF/CS/TF coil currents + the plasma current (the ``amc`` source: the
    machine state / the response) — defines WHICH plasma this is and is held at
    the true plan when forming a "wrong actuation, right machine" counterfactual.
    Density (``ane``) is included as a controllable command alongside gas/NBI.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ACTUATOR_CHANNELS,
        gas_puff_channel_indices,
        nbi_channel_indices,
    )

    idx = set(gas_puff_channel_indices()) | set(nbi_channel_indices())
    idx |= {i for i, c in enumerate(ACTUATOR_CHANNELS) if c.source == "ane"}
    return sorted(idx)


def _random_actuator_like(
    plan,
    *,
    rng: np.random.Generator,
    controllable_only: bool = True,
):
    """A "wrong actuation, right machine" counterfactual plan.

    The ΔN-M gate contrasts the TRUE plan against a DIFFERENT plan; the
    difference must be in the CONTROLLABLE actuation (what the operator could
    have dialled differently), NOT in the machine state.  The earlier version
    re-randomised EVERY channel including the PF/CS/TF coil currents + the plasma
    current — a ~100% re-randomisation of the dominant drive (the coils are ~15
    of 23 channels), producing a non-physical "different machine" whose rollout
    scatters wildly and INFLATES the random-vs-random noise floor so the true
    plan can never clear it (the gate-bug noted in gate_verdict_1220668).

    With ``controllable_only`` (default), the machine-state channels (the ``amc``
    coil currents + plasma current) are held at the TRUE plan and ONLY the
    controllable actuators (gas / NBI / density) are resampled — each present
    controllable channel drawn from a Normal matched to its own window
    mean/std (a flat channel stays ~flat at a different level; a ramping one
    becomes a random walk of the same spread).  So "wrong plan" = "the operator
    dialled the gas/NBI/density differently on the SAME shot", which is exactly
    the controllability the gate should measure.  ``controllable_only=False``
    restores the legacy all-channel randomisation (kept for comparison only).
    ``missing`` is preserved so an absent actuator stays absent.
    """
    from imas_ambix.worldmodel.actuator_plan import (  # noqa: PLC0415
        ActuatorPlan,
        normalise_actuator_values,
    )

    raw = np.asarray(plan.raw_values, dtype=np.float64)
    miss = np.asarray(plan.missing, dtype=np.float32)
    present = miss.mean(axis=0) < 1.0  # (C,)
    out = raw.copy()
    p, c = raw.shape
    if controllable_only:
        targets = [i for i in _controllable_actuator_indices() if i < c and present[i]]
    else:
        targets = [i for i in range(c) if present[i]]
    for ch in targets:
        col = raw[:, ch]
        mu = float(np.mean(col))
        sd = float(np.std(col))
        # a non-degenerate spread even for a near-flat channel so the random plan
        # genuinely differs; scale the floor by the channel magnitude.
        sd = max(sd, 0.1 * (abs(mu) + 1.0))
        out[:, ch] = rng.normal(mu, sd, size=p)
    return ActuatorPlan(
        values=normalise_actuator_values(out),
        missing=plan.missing.copy(),
        channel_keys=list(plan.channel_keys),
        raw_values=out.astype(np.float32),
    )


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


__all__ = [
    "ControllabilityVerdict",
    "DeltaNMVerdict",
    "OverfitControllableConfig",
    "build_controllable_model",
    "collate_actuator",
    "collate_controllable_windows",
    "controllability_gate",
    "delta_nm_gate",
    "overfit_controllable",
    "teacher_forced_token_mismatch",
]
