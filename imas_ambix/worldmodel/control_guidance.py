"""Classifier-free guidance on the controls + the gas-puff falsification.

Why this exists (making the controls LOAD-BEARING)
--------------------------------------------------
The signal-conditioned camera transformer was fine-tuned with **control-dropout**
(:func:`imas_ambix.worldmodel.context_corruption.sample_control_dropout`): a
fraction of training steps zero the plan + measured-signal conditioning, so the
model has learnt BOTH a conditioned next-frame distribution and an unconditioned
one.  Classifier-free guidance (CFG) exploits that pair at inference: it
extrapolates the per-token logits AWAY from the unconditioned prediction toward
the conditioned one,

    logits_guided = logits_uncond + w * (logits_cond - logits_uncond),

so a guidance weight ``w > 1`` AMPLIFIES the controls' influence (``w = 1`` is the
plain conditioned model; ``w = 0`` the unconditioned one).  This is the lever
Genie-2 uses "to improve action controllability"; here it is what makes the pulse
schedule a steerable control rather than a weak hint.

The conditioning that CFG drops is the SAME one control-dropout dropped in
training — the plan + every measured-signal block, zeroed to PAD id 0 by
:func:`imas_ambix.worldmodel.context_corruption.apply_control_dropout` — so the
unconditioned pass at inference matches a regime the model actually trained on.

Why the rollout runs the backbone TWICE per frame
-------------------------------------------------
CFG combines LOGITS, not sampled tokens, so each generated frame needs the
per-token logits from two forward passes — one with the real conditioning, one
with it zeroed — over the SAME generated history.  The two passes share the
history (the previously-generated frames) and differ only in the prepended
conditioning prefix.  The combination + the temperature/nucleus draw is then done
on the guided logits, chunked over the (B*S) token axis exactly like
:meth:`SpacetimeTransformer.chunked_sample_frame` so head memory never exceeds
``chunk * vocab``.

The gas-puff falsification (W1's falsifiable instance)
------------------------------------------------------
The inboard gas-puff command lights a bright emission spot LEFT of the centre
column (camdyn ``puff_attribution``; token columns ``[2, 8)``).  The falsification
decodes the dream three ways on a held-out shot that fires the puff:

* (a) TRUE  — conditioned on the real ``gas_injection`` (+ the other streams);
* (b) ZEROED — the ``gas_injection`` stream zeroed (counterfactual "no puff");
* (c) CFG   — guidance amplifying the real conditioning (``w > 1``).

and measures the decoded emission in the inboard pixel band.  W1 wants the spot to
LIGHT in (a)/(c) when the puff fires and NOT in (b): a positive timing correlation
between the predicted inboard intensity and the puff command waveform, and a
positive counterfactual delta ``(a) - (b)``.  An honest null (no correlation, no
delta) is a reportable negative — the probe is built to be able to falsify.

Everything model-side here is decoder-FREE (token rollouts); the pixel scoring
lives in the driver (:mod:`imas_ambix.worldmodel.control_falsification`), which
decodes through the frozen Open-MAGVIT2 VQ exactly as the M1/M2 harness does.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.worldmodel.context_corruption import apply_control_dropout
from imas_ambix.worldmodel.spacetime_dataset import GRID_W

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from imas_ambix.worldmodel.spacetime_dataset_v2 import SignalSpacetimeSample
    from imas_ambix.worldmodel.spacetime_model_v2 import SignalSpacetimeTransformer

logger = logging.getLogger(__name__)

#: Inboard gas-puff token-column band (left-of-centre, full height).  The same
#: convention as camdyn ``puff_attribution``: the centre-stack sightline is cols
#: 6..10, and the inboard puff bright-spot sits left of it, so the inboard region
#: is the ``[2, 8)`` column band.  Pixel columns scale by 256/16 = 16.
INBOARD_COLS: tuple[int, int] = (2, 8)

#: Name of the measured gas-puff conditioning stream (dataset + model side).
GAS_STREAM: str = "gas_injection"


# ---------------------------------------------------------------------------
# CFG-guided token decode (model loaded once by the caller)
# ---------------------------------------------------------------------------


def _zeroed_conditioning(
    plan: torch.Tensor | None,
    signals: dict[str, torch.Tensor] | None,
    *,
    streams: Sequence[str] | None = None,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
    """Return the conditioning with the requested controls zeroed (PAD id 0).

    ``streams is None`` zeroes the WHOLE conditioning (plan + every signal block)
    — the unconditioned path CFG extrapolates from, matching training-time
    control-dropout.  A ``streams`` list zeroes ONLY those signal streams (and
    leaves the plan + other streams intact) — the per-channel counterfactual (e.g.
    zero just ``gas_injection``).  Built on
    :func:`imas_ambix.worldmodel.context_corruption.apply_control_dropout` so the
    "zeroed" semantics are byte-identical to what the model trained on.
    """
    import torch

    if streams is None:
        drop = torch.ones(1, dtype=torch.bool)
        return apply_control_dropout(plan, signals, drop)
    # zero only the named streams; keep plan + the rest.
    new_signals = signals
    if signals:
        wanted = set(streams)
        new_signals = {}
        for name, block in signals.items():
            nb = block.clone()
            if name in wanted and nb.numel() and nb.shape[1] > 0:
                nb[:] = 0
            new_signals[name] = nb
    return plan, new_signals


def _guided_sample_frame(
    model: SignalSpacetimeTransformer,
    hidden_cond_prev: torch.Tensor,
    hidden_uncond_prev: torch.Tensor,
    *,
    guidance: float,
    temperature: float,
    top_p: float,
    chunk: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample one frame from CFG-combined logits of two predecessor hiddens.

    ``hidden_cond_prev`` / ``hidden_uncond_prev`` are each ``(B, S, d)`` backbone
    hidden states at the frame BEFORE the one generated — from the conditioned and
    the unconditioned forward passes respectively.  The guided per-token logits are
    ``l_u + guidance * (l_c - l_u)``; ``guidance == 1`` reduces exactly to the
    conditioned logits (the standard sample), ``guidance == 0`` to the
    unconditioned ones.  ``temperature <= 0`` argmaxes the guided logits (a
    deterministic guided baseline through the same path); a positive temperature
    draws a nucleus sample.  Chunked over the ``(B*S)`` axis so head memory stays
    at ``chunk * vocab`` even though two heads are evaluated per chunk.
    """
    import torch

    from imas_ambix.worldmodel.spacetime_model import _nucleus_mask_logits

    b, s, d = hidden_cond_prev.shape
    flat_c = hidden_cond_prev.reshape(-1, d)
    flat_u = hidden_uncond_prev.reshape(-1, d)
    out = flat_c.new_zeros(flat_c.shape[0], dtype=torch.long)
    greedy = temperature is None or float(temperature) <= 0.0
    gen_dev = generator.device if generator is not None else None
    for start in range(0, flat_c.shape[0], chunk):
        stop = min(start + chunk, flat_c.shape[0])
        lc = model.head(flat_c[start:stop]).float()  # (rows, vocab)
        lu = model.head(flat_u[start:stop]).float()
        guided = lu + float(guidance) * (lc - lu)
        if greedy:
            out[start:stop] = guided.argmax(dim=-1)
            del lc, lu, guided
            continue
        guided = guided / float(temperature)
        if top_p < 1.0:
            guided = _nucleus_mask_logits(guided, float(top_p))
        probs = torch.softmax(guided, dim=-1)
        if gen_dev is not None and gen_dev != probs.device:
            idx = torch.multinomial(
                probs.to(gen_dev), num_samples=1, generator=generator
            ).to(out.device)
        else:
            idx = torch.multinomial(probs, num_samples=1, generator=generator)
        out[start:stop] = idx.squeeze(-1)
        del lc, lu, guided, probs, idx
    return out.reshape(b, s)


def cfg_guided_dream(
    model: SignalSpacetimeTransformer,
    sample: SignalSpacetimeSample,
    *,
    stream_names: Sequence[str] | None = None,
    guidance: float = 1.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    chunk: int = 4096,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    zero_streams: Sequence[str] | None = None,
) -> np.ndarray:
    """Autoregressive rollout with classifier-free guidance on the controls.

    The CFG counterpart of
    :func:`imas_ambix.worldmodel.spacetime_train_v2.autoregressive_signal_dream`:
    the model keeps the leading ``context_frames`` TRUE frames, then rolls forward
    consuming its own predicted frames while, on EVERY generated frame, running the
    backbone twice — once with the real conditioning (plan + the measured signals)
    and once with the conditioning zeroed to PAD — and sampling the next frame from
    the CFG-combined logits ``l_u + guidance*(l_c - l_u)``.

    Parameters
    ----------
    guidance:
        CFG weight ``w``.  ``1.0`` is the plain conditioned rollout (and skips the
        second forward for speed); ``> 1`` amplifies the controls; ``0`` is the
        unconditioned rollout.
    zero_streams:
        Optional list of signal streams to ZERO in the CONDITIONED pass too (the
        per-channel counterfactual, e.g. ``["gas_injection"]`` for the no-puff
        run).  ``None`` keeps the full conditioning.  The unconditioned pass always
        zeroes everything regardless.
    temperature / top_p / generator:
        Token-selection rule, identical to the non-CFG dream — ``temperature <= 0``
        argmaxes, a positive temperature draws a reproducible nucleus sample.

    Returns ``(T, S)`` LOCAL token ids (context frames = truth).
    """
    import torch

    from imas_ambix.worldmodel.spacetime_train_v2 import (
        _batch_to,
        _sample_stream_names,
        collate_signal_windows,
    )

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

    # The CONDITIONED conditioning: the real plan + signals, UNLESS a per-channel
    # counterfactual knocks specific streams out (``zero_streams``).  When
    # ``zero_streams`` is None/empty the conditioned pass uses the raw conditioning
    # (so w == 1 reproduces the plain conditioned rollout exactly).
    if zero_streams:
        plan_c, signals_c = _zeroed_conditioning(plan, signals, streams=zero_streams)
    else:
        plan_c, signals_c = plan, signals
    do_cfg = float(guidance) != 1.0
    if do_cfg:
        # the UNCONDITIONED conditioning CFG extrapolates from — all of it zeroed.
        plan_u, signals_u = _zeroed_conditioning(plan, signals, streams=None)

    gen = np.asarray(sample.frames, dtype=np.int64).copy()  # seed with truth
    with torch.no_grad():
        for ti in range(ctx, t_total):
            cur = torch.as_tensor(gen[:ti][None], dtype=torch.long, device=dev)
            hidden_c = model._forward_tokens(cur, plan_c, signals_c)  # (1, ti, S, d)
            if do_cfg:
                hidden_u = model._forward_tokens(cur, plan_u, signals_u)
                pred = _guided_sample_frame(
                    model,
                    hidden_c[:, ti - 1],
                    hidden_u[:, ti - 1],
                    guidance=guidance,
                    temperature=temperature,
                    top_p=top_p,
                    chunk=chunk,
                    generator=generator,
                )
            else:
                # w == 1 — the conditioned model; one forward, the shared decode.
                from imas_ambix.worldmodel.spacetime_train_v2 import _decode_frame

                pred = _decode_frame(
                    model,
                    hidden_c[:, ti - 1],
                    chunk=chunk,
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                )
            gen[ti] = pred[0].cpu().numpy().astype(np.int64)
    return gen


# ---------------------------------------------------------------------------
# Gas-puff command waveform (what the model conditions on)
# ---------------------------------------------------------------------------


def gas_command_per_frame(
    sample: SignalSpacetimeSample,
    *,
    stream: str = GAS_STREAM,
) -> np.ndarray:
    """Per-frame gas-puff command proxy from the assembled conditioning.

    The model conditions on ``n_signal_steps`` token "steps" of the
    ``gas_injection`` stream, sub-sampled across the camera window.  The uniform-
    quantiser token id is monotone in the measured inboard flow, so the per-step
    mean token id over the stream's channels is a faithful command proxy.  It is
    linearly interpolated onto the ``T`` camera frames so it aligns with the
    decoded emission series.  Returns ``(T,)`` float (zeros when the stream is
    absent).
    """
    t_total = int(sample.frames.shape[0])
    block = sample.signals.get(stream)
    if block is None or np.asarray(block).size == 0:
        return np.zeros(t_total, dtype=np.float64)
    arr = np.asarray(block, dtype=np.float64)  # (n_steps, n_ch)
    per_step = arr.mean(axis=1)  # (n_steps,)
    n_steps = per_step.shape[0]
    if n_steps == 1:
        return np.full(t_total, float(per_step[0]), dtype=np.float64)
    # map the n_steps positions (evenly spaced across the window span) onto frames.
    step_pos = np.linspace(0.0, t_total - 1.0, n_steps)
    frame_pos = np.arange(t_total, dtype=np.float64)
    return np.interp(frame_pos, step_pos, per_step).astype(np.float64)


# ---------------------------------------------------------------------------
# Inboard emission series + the falsification metrics (decoded pixel space)
# ---------------------------------------------------------------------------


def _to_gray_f64(stack: np.ndarray) -> np.ndarray:
    """``(F, H, W[, C]) -> (F, H, W)`` float64 luminance (mean over channels)."""
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected (F,H,W[,C]) image stack, got shape {arr.shape}")
    return arr


def inboard_pixel_cols(
    width: int,
    *,
    token_cols: tuple[int, int] = INBOARD_COLS,
    grid_w: int = GRID_W,
) -> tuple[int, int]:
    """Map the inboard TOKEN-column band to a decoded PIXEL-column band.

    A decoded frame is ``width`` px wide over ``grid_w`` token columns, so token
    column ``c`` spans pixels ``[c*scale, (c+1)*scale)`` with ``scale =
    width/grid_w``.  Returns the half-open pixel-column band for the inboard token
    band.
    """
    scale = float(width) / float(grid_w)
    c0, c1 = token_cols
    return int(round(c0 * scale)), int(round(c1 * scale))


def inboard_emission_series(
    frames: np.ndarray,
    *,
    token_cols: tuple[int, int] = INBOARD_COLS,
) -> np.ndarray:
    """Mean decoded luminance in the inboard pixel band, per frame.

    ``frames`` is a decoded ``(F, H, W[, C])`` stack.  Returns ``(F,)`` the mean
    luminance over the left-of-centre inboard column band (the region the inboard
    gas puff lights).  This is the "predicted-spot intensity" the W1 timing
    correlation scores against the puff command.
    """
    g = _to_gray_f64(frames)
    w = g.shape[2]
    p0, p1 = inboard_pixel_cols(w, token_cols=token_cols)
    return g[:, :, p0:p1].mean(axis=(1, 2))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def puff_timing_correlation(
    decoded_frames: np.ndarray,
    command_per_frame: np.ndarray,
    ctx: int,
    *,
    token_cols: tuple[int, int] = INBOARD_COLS,
) -> dict[str, float | int]:
    """Correlate inboard emission against the puff command over the forecast.

    Both inputs are on the camera-frame axis; only forecast frames (``f >= ctx``)
    are scored (the context frames are truth, identical across all conditionings).
    Returns the Pearson correlation of the inboard emission series vs the command,
    the number of forecast frames, and the mean inboard emission (a level the
    counterfactual delta is read against).
    """
    g = inboard_emission_series(decoded_frames, token_cols=token_cols)
    cmd = np.asarray(command_per_frame, dtype=np.float64)
    n = g.shape[0]
    if not 1 <= ctx < n:
        raise ValueError(f"ctx {ctx} out of range for {n} frames")
    if cmd.shape[0] != n:
        raise ValueError(f"command length {cmd.shape[0]} != frames {n}")
    fc = slice(ctx, n)
    return {
        "pearson_emission_vs_command": _pearson(g[fc], cmd[fc]),
        "n_forecast_frames": int(n - ctx),
        "mean_inboard_emission": float(g[fc].mean()),
    }


def counterfactual_delta(
    decoded_true: np.ndarray,
    decoded_zeroed: np.ndarray,
    ctx: int,
    *,
    token_cols: tuple[int, int] = INBOARD_COLS,
) -> dict[str, float | bool | int]:
    """Inboard-emission delta between TRUE-puff and ZEROED-puff rollouts.

    The counterfactual (a) − (b): mean inboard luminance with the real
    ``gas_injection`` conditioning minus the same with it zeroed, over the forecast
    window.  A POSITIVE delta means the puff command CAUSALLY raises the predicted
    inboard emission — the spot lights when the puff fires and not when it is
    removed.  Reported per-frame too so a caller can correlate the delta timing
    with the command.
    """
    a = inboard_emission_series(decoded_true, token_cols=token_cols)
    b = inboard_emission_series(decoded_zeroed, token_cols=token_cols)
    n = a.shape[0]
    if not 1 <= ctx < n:
        raise ValueError(f"ctx {ctx} out of range for {n} frames")
    if b.shape[0] != n:
        raise ValueError(f"zeroed length {b.shape[0]} != true {n}")
    fc = slice(ctx, n)
    per_frame = (a - b)[fc]
    return {
        "true_mean_inboard_emission": float(a[fc].mean()),
        "zeroed_mean_inboard_emission": float(b[fc].mean()),
        "counterfactual_delta": float(per_frame.mean()),
        "counterfactual_delta_positive": bool(per_frame.mean() > 0.0),
        "n_forecast_frames": int(n - ctx),
    }


def frame_l1(a: np.ndarray, b: np.ndarray, ctx: int) -> float:
    """Mean absolute decoded-pixel difference over the forecast window.

    A scale for "how much does the dream change" — used both for control-
    divergence (different conditionings) and same-conditioning sample spread.
    """
    ga = _to_gray_f64(a)
    gb = _to_gray_f64(b)
    n = ga.shape[0]
    if not 1 <= ctx < n:
        raise ValueError(f"ctx {ctx} out of range for {n} frames")
    fc = slice(ctx, n)
    return float(np.abs(ga[fc] - gb[fc]).mean())


def control_divergence(
    decoded_a: np.ndarray,
    decoded_b: np.ndarray,
    same_conditioning_members: Sequence[np.ndarray],
    ctx: int,
) -> dict[str, float | bool | int]:
    """Does changing the conditioning move the dream MORE than sampling noise?

    ``decoded_a`` / ``decoded_b`` are two rollouts from the SAME seed under
    DIFFERENT conditionings (e.g. true-puff vs zeroed-puff); their forecast-window
    pixel L1 is the CONTROL signal.  ``same_conditioning_members`` are >= 2
    rollouts under the SAME conditioning with DIFFERENT seeds; the mean pairwise L1
    among them is the sample-spread NOISE floor.  W1 wants control divergence to
    EXCEED the noise floor — the controls move the world more than the stochastic
    draw does.  Returns both magnitudes, their ratio, and the verdict flag.
    """
    control = frame_l1(decoded_a, decoded_b, ctx)
    members = list(same_conditioning_members)
    pair_l1: list[float] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            pair_l1.append(frame_l1(members[i], members[j], ctx))
    noise = float(np.mean(pair_l1)) if pair_l1 else float("nan")
    ratio = (
        float("inf") if (not np.isfinite(noise) or noise == 0.0) else control / noise
    )
    return {
        "control_divergence_l1": control,
        "same_conditioning_spread_l1": noise,
        "n_spread_pairs": len(pair_l1),
        "divergence_over_spread_ratio": ratio,
        "control_exceeds_spread": bool(np.isfinite(noise) and control > noise),
    }


__all__ = [
    "GAS_STREAM",
    "INBOARD_COLS",
    "cfg_guided_dream",
    "control_divergence",
    "counterfactual_delta",
    "frame_l1",
    "gas_command_per_frame",
    "inboard_emission_series",
    "inboard_pixel_cols",
    "puff_timing_correlation",
]
