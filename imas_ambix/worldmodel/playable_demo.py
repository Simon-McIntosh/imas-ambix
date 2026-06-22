"""Rendered "play the plasma with a coil" demos on the controllable camera model.

This is the headline demonstrable capability of the playable-plasma sprint: an
OFFLINE-RENDERED proof that the actuator commands steer the dreamed plasma.  It
runs on the best controllable checkpoint and reuses the established eval helpers
(:mod:`imas_ambix.worldmodel.controllable_eval` +
:mod:`imas_ambix.worldmodel.controllable_train`) for the rollout + decode — it
adds only the demo orchestration and a multi-panel renderer.

Three demos on one held-out exemplar shot (the model is driveable there; it is
not yet cross-shot robust — that honesty is carried into the artifacts):

1. **Faithful replay** — the model's dream under the shot's TRUE coil plan, beside
   the GROUND-TRUTH decode of the same window.  A coherence check: does the dream
   reproduce the real evolution?
2. **Counterfactual "same start, two plans"** — the SAME seed context rolled under
   the TRUE coil plan vs a DIFFERENT realistic plan (a bounded, in-distribution
   re-actuation — the same counterfactual the ΔN-M gate scores).  Shows the coils
   steer the outcome.
3. **Coil "knob" sweep** — sweep ONE position coil (P6-upper current, the strongest
   from the eval) across ``{-,0,+}`` of its trajectory, render the resulting dreams
   side-by-side, and plot the decoded plasma-centroid vs the knob value.  The
   literal "turn the knob, watch the plasma move" artifact.

Every demo decodes ALL of its rollouts (GT + dreams) in ONE Open-MAGVIT2 VQ pass
(:func:`imas_ambix.worldmodel.control_falsification.decode_roles`).

GPU-safety (AGENTS.md §2b): the transformer is loaded ONCE outside every loop; a
SIGTERM/SIGINT STOP flag makes a cancellation flush cleanly in < 5 s; ``try /
finally`` releases the model + ``torch.cuda.empty_cache()``; cudnn deterministic;
bf16 autocast inherited from the rollout helper.  The decode runs in the frozen-VQ
subprocess.  The transformer rollout is CPU-light (a handful of windows); the GPU
cost is the single VQ decode per demo — minutes total.

Run (single GPU; keep the neighbour's LLM serve + any retrain up)::

    .venv/bin/python -m imas_ambix.worldmodel.playable_demo \\
        --checkpoint <ckpt-dir>/controllable-1220940/latest.pt \\
        --token-root /work/projects/imas_gpu/worldmodel/curated-token-view \\
        --shot 18502 \\
        --out-dir docs/figures/playable-plasma-wm-v0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: The driveable held-out exemplar (ΔN-M 3.14x on the best checkpoint; 18503/4 are
#: not yet robust — the demos are honest that this is the exemplar, not cross-shot).
DEFAULT_SHOT = 18502

#: P6-upper current — the strongest position-coil knob measured by the eval's
#: per-coil centroid scan (13.6 px shift @ +0.3 on 1220940).  Resolved by KEY so
#: the column is correct for the filtered command vector; falls back to the eval's
#: best-coil pick if the key is absent.
DEFAULT_KNOB_KEY = "p6u_current"

#: The knob sweep values: a bounded gain edit of the coil's raw trajectory.  0.0
#: reproduces the TRUE plan exactly (the anchor), +/- push the coil harder/softer.
DEFAULT_SWEEP_FRACS = (-0.3, 0.0, 0.3)

#: Counterfactual re-actuation strength (bounded, in-distribution) — matches the
#: ΔN-M gate's perturb_scale so demo 2's "other plan" is the validated separator.
DEFAULT_COUNTERFACTUAL_SCALE = 0.3

#: Native rbb frame aspect (rows, cols) for display — mirrors dream_gifs._to_aspect.
ORIGINAL_HW = (112, 156)

#: GIF playback rate.
FPS = 8


# ---------------------------------------------------------------------------
# Clean-cancellation stop flag (repo §2b GPU-safety pattern)
# ---------------------------------------------------------------------------


class _StopFlag:
    """SIGTERM/SIGINT-set stop flag the rollout loops poll (clean cancel < 5 s)."""

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
# Display helpers (multi-panel; the eval's _panel_frame is 2-up only)
# ---------------------------------------------------------------------------


def _to_aspect(img: np.ndarray) -> np.ndarray:
    """Resize a decoded 256x256 (RGB or gray) frame to the native rbb aspect."""
    from PIL import Image  # noqa: PLC0415

    if img.ndim == 3:
        img = img[..., 0]
    im = Image.fromarray(img.astype(np.uint8)).resize(
        (ORIGINAL_HW[1], ORIGINAL_HW[0]), Image.BILINEAR
    )
    return np.asarray(im)


def _panel_row_frame(
    panels: list[np.ndarray],
    titles: list[str],
    *,
    banner: str,
    in_target: bool,
) -> np.ndarray:
    """Render one row of N decoded frames (with per-panel titles) to RGB uint8.

    Generalises :func:`imas_ambix.worldmodel.dream_gifs._panel_frame` to an
    arbitrary number of side-by-side panels (the knob sweep needs 3+).  A coloured
    title flags forecast frames (model is dreaming) vs the given context frames.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), dpi=100)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.80, bottom=0.02, left=0.02, right=0.98, wspace=0.05)
    for ax, img, title in zip(axes, panels, titles, strict=True):
        ax.imshow(img, cmap="inferno", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, color=("#d62728" if in_target else "#222222"))
    fig.suptitle(banner, fontsize=8, y=0.985)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _save_gif(frames: list[np.ndarray], out_path: Path, *, fps: int = FPS) -> tuple:
    """Write a looping GIF via PIL (imageio is absent on the offline node)."""
    from PIL import Image  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    pil[0].save(
        str(out_path),
        format="GIF",
        save_all=True,
        append_images=pil[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
    )
    h, w = frames[0].shape[:2]
    return int(h), int(w)


def _rel_time_ms(sample, ctx: int) -> np.ndarray:
    """Per-frame time in ms relative to the forecast frontier (ctx), or frame idx."""
    ft = np.asarray(getattr(sample, "frame_time", None))
    n = int(np.asarray(sample.frames).shape[0])
    if ft is None or ft.size < n or ctx >= ft.size:
        return np.arange(n, dtype=np.float64)
    return (ft[:n] - ft[ctx]) * 1e3


def _decode_work_dir(name: str) -> Path:
    """A scratch dir for the VQ decode bundle — under TMPDIR, never the figures dir.

    The decode round-trips multi-MB token/image ``.npz`` bundles; keeping them out
    of ``out_dir`` means a re-run leaves only the final GIFs/PNGs/JSON there (no
    scratch to clean up before committing).
    """
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    wd = Path(base) / "playable-demo-decode" / name
    wd.mkdir(parents=True, exist_ok=True)
    return wd


# ---------------------------------------------------------------------------
# Demo 1 — faithful replay (GT | TRUE-plan dream)
# ---------------------------------------------------------------------------


def faithful_replay(
    model,
    sample,
    *,
    device,
    out_dir: Path,
    chunk: int,
    fps: int = FPS,
    stop: _StopFlag | None = None,
) -> dict:
    """Render GT | TRUE-plan dream side-by-side + a centroid coherence panel.

    The GT is the assembled window's own tokens (``sample.frames``, local ids); the
    dream is the argmax rollout under the shot's TRUE actuator plan.  Both decode in
    one VQ pass.  Reports the decoded-pixel L1 between the dream and GT over the
    forecast window and their centroid traces (the coherence readout).
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.control_falsification import (
        decode_roles,  # noqa: PLC0415
    )
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _forecast_pixel_l1,
        decoded_centroid,
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
        _argmax_token_rollout,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        GRID_H,
        GRID_W,
        local_to_store,
    )

    dev = torch.device(device)
    ctx = int(sample.context_frames)
    stream_names = list(sample.signals.keys())
    n = int(np.asarray(sample.frames).shape[0])

    dream_tok = _argmax_token_rollout(
        model,
        sample,
        stream_names,
        _actuator_batch_from_plan(sample.actuator, dev),
        dev,
        chunk=chunk,
    )
    if stop and stop.stop:
        raise KeyboardInterrupt("stopped before decode")
    gt_local = np.asarray(sample.frames, dtype=np.int64)[:n]
    grids = {
        "gt": local_to_store(gt_local.reshape(-1, GRID_H, GRID_W)),
        "dream": local_to_store(dream_tok.reshape(-1, GRID_H, GRID_W)),
    }
    roles = [{"role": "gt"}, {"role": "dream"}]
    decoded = decode_roles(
        grids, roles, work_dir=_decode_work_dir("replay"), device=device
    )
    gt_px, dream_px = decoded["gt"], decoded["dream"]

    gt_cen = decoded_centroid(gt_px)
    dream_cen = decoded_centroid(dream_px)
    fc_l1 = _forecast_pixel_l1(gt_px, dream_px, ctx)

    t = _rel_time_ms(sample, ctx)
    nframes = min(gt_px.shape[0], dream_px.shape[0], n)
    frames = []
    for i in range(nframes):
        frames.append(
            _panel_row_frame(
                [_to_aspect(gt_px[i]), _to_aspect(dream_px[i])],
                [
                    f"ground truth   t={t[i]:+.0f} ms",
                    f"WM dream (true plan)   [{'FORECAST' if i >= ctx else 'context'}]",
                ],
                banner=(
                    f"faithful replay  |  shot {int(sample.shot_id)}  |  "
                    f"rbb full-res  |  exemplar (driveable held-out)"
                ),
                in_target=(i >= ctx),
            )
        )
    gif = out_dir / f"replay_shot{int(sample.shot_id)}.gif"
    _save_gif(frames, gif, fps=fps)
    png = _centroid_panel(
        out_dir / f"replay_shot{int(sample.shot_id)}_centroid.png",
        t,
        ctx,
        traces=[
            ("ground truth", gt_cen, "#1f77b4"),
            ("WM dream", dream_cen, "#d62728"),
        ],
        suptitle=(
            f"shot {int(sample.shot_id)}: dream-vs-GT plasma centroid "
            f"(forecast pixel-L1 = {fc_l1:.1f})"
        ),
    )
    logger.info("faithful replay: forecast dream-vs-GT pixel L1 = %.2f", fc_l1)
    return {
        "demo": "faithful_replay",
        "shot_id": int(sample.shot_id),
        "context_frames": ctx,
        "n_frames": int(nframes),
        "forecast_dream_vs_gt_pixel_l1": float(fc_l1),
        "gif_path": str(gif),
        "centroid_png_path": str(png),
    }


# ---------------------------------------------------------------------------
# Demo 2 — counterfactual "same start, two plans"
# ---------------------------------------------------------------------------


def counterfactual_two_plans(
    model,
    sample,
    *,
    device,
    out_dir: Path,
    chunk: int,
    scale: float = DEFAULT_COUNTERFACTUAL_SCALE,
    seed: int = 0,
    fps: int = FPS,
    stop: _StopFlag | None = None,
) -> dict:
    """Same seed context, TRUE coil plan vs a DIFFERENT realistic plan, side-by-side.

    The "other plan" is a bounded, in-distribution re-actuation of the commands
    (:func:`controllable_train._random_actuator_like` at ``scale`` — the same
    counterfactual the ΔN-M gate scores, which separated 3.14x on the exemplar).
    Both dreams share the identical context frames, so any divergence in the
    forecast window is the coils steering the outcome.  Reports the forecast
    pixel-L1 between the two dreams (the steer signal) and both centroid traces.
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.control_falsification import (
        decode_roles,  # noqa: PLC0415
    )
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _forecast_pixel_l1,
        decoded_centroid,
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
        _argmax_token_rollout,
        _random_actuator_like,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        GRID_H,
        GRID_W,
        local_to_store,
    )

    dev = torch.device(device)
    ctx = int(sample.context_frames)
    stream_names = list(sample.signals.keys())
    n = int(np.asarray(sample.frames).shape[0])
    rng = np.random.default_rng((int(sample.shot_id) * 1_000_003) ^ (seed * 31))

    true_tok = _argmax_token_rollout(
        model,
        sample,
        stream_names,
        _actuator_batch_from_plan(sample.actuator, dev),
        dev,
        chunk=chunk,
    )
    other_plan = _random_actuator_like(sample.actuator, rng=rng, perturb_scale=scale)
    other_tok = _argmax_token_rollout(
        model,
        sample,
        stream_names,
        _actuator_batch_from_plan(other_plan, dev),
        dev,
        chunk=chunk,
    )
    if stop and stop.stop:
        raise KeyboardInterrupt("stopped before decode")
    grids = {
        "true": local_to_store(true_tok.reshape(-1, GRID_H, GRID_W)),
        "other": local_to_store(other_tok.reshape(-1, GRID_H, GRID_W)),
    }
    roles = [{"role": "true"}, {"role": "other"}]
    decoded = decode_roles(grids, roles, work_dir=_decode_work_dir("cf"), device=device)
    true_px, other_px = decoded["true"], decoded["other"]

    true_cen = decoded_centroid(true_px)
    other_cen = decoded_centroid(other_px)
    fc_l1 = _forecast_pixel_l1(true_px, other_px, ctx)

    t = _rel_time_ms(sample, ctx)
    nframes = min(true_px.shape[0], other_px.shape[0], n)
    frames = []
    for i in range(nframes):
        frames.append(
            _panel_row_frame(
                [_to_aspect(true_px[i]), _to_aspect(other_px[i])],
                [
                    f"true coil plan   t={t[i]:+.0f} ms",
                    f"different plan   [{'FORECAST' if i >= ctx else 'shared ctx'}]",
                ],
                banner=(
                    f"counterfactual: same start, two plans  |  shot "
                    f"{int(sample.shot_id)}  |  re-actuation +/-{scale:.0%}"
                ),
                in_target=(i >= ctx),
            )
        )
    gif = out_dir / f"counterfactual_shot{int(sample.shot_id)}.gif"
    _save_gif(frames, gif, fps=fps)
    png = _centroid_panel(
        out_dir / f"counterfactual_shot{int(sample.shot_id)}_centroid.png",
        t,
        ctx,
        traces=[
            ("true coil plan", true_cen, "#1f77b4"),
            ("different plan", other_cen, "#d62728"),
        ],
        suptitle=(
            f"shot {int(sample.shot_id)}: two coil plans diverge from a shared "
            f"start (forecast pixel-L1 = {fc_l1:.1f})"
        ),
    )
    logger.info("counterfactual: forecast true-vs-other pixel L1 = %.2f", fc_l1)
    return {
        "demo": "counterfactual_two_plans",
        "shot_id": int(sample.shot_id),
        "context_frames": ctx,
        "n_frames": int(nframes),
        "counterfactual_scale": float(scale),
        "forecast_true_vs_other_pixel_l1": float(fc_l1),
        "gif_path": str(gif),
        "centroid_png_path": str(png),
    }


# ---------------------------------------------------------------------------
# Demo 3 — coil "knob" sweep (turn P6-upper, watch the plasma move)
# ---------------------------------------------------------------------------


def _resolve_knob_col(sample, knob_key: str) -> tuple[int, str]:
    """Resolve the sweep coil column by KEY; fall back to the eval's best position coil.

    Tries an exact channel-key match first (so ``p6u_current`` lands on the right
    column for the filtered command vector), then a substring match, then the
    eval's :func:`_position_coil_columns` best-present coil.  Returns
    ``(column, resolved_key)``.
    """
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _position_coil_columns,
    )

    keys = list(sample.actuator.channel_keys)
    present = np.asarray(sample.actuator.missing, dtype=np.float32).mean(axis=0) < 1.0
    for i, k in enumerate(keys):
        if k == knob_key and i < present.shape[0] and present[i]:
            return i, k
    for i, k in enumerate(keys):
        if knob_key in k and i < present.shape[0] and present[i]:
            return i, k
    cols = [
        c for c in _position_coil_columns(keys) if c < present.shape[0] and present[c]
    ]
    if not cols:
        raise ValueError(f"no present position coil on shot {int(sample.shot_id)}")
    return cols[0], keys[cols[0]]


def coil_knob_sweep(
    model,
    sample,
    *,
    device,
    out_dir: Path,
    chunk: int,
    knob_key: str = DEFAULT_KNOB_KEY,
    fracs=DEFAULT_SWEEP_FRACS,
    fps: int = FPS,
    stop: _StopFlag | None = None,
) -> dict:
    """Sweep ONE position coil across ``fracs`` and render the dreams + a knob plot.

    For each ``frac`` the coil's raw trajectory is scaled by ``(1+frac)``
    (:func:`controllable_eval._bounded_coil_edit`); ``frac=0`` reproduces the true
    plan.  All sweep dreams decode in ONE VQ pass.  Writes an N-up GIF (the plasma
    shifting as the knob turns) and a centroid-vs-knob PNG (forecast-mean centroid
    row & col vs the knob value — the literal turn-the-knob curve).
    """
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.control_falsification import (
        decode_roles,  # noqa: PLC0415
    )
    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        _bounded_coil_edit,
        decoded_centroid,
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        _actuator_batch_from_plan,
        _argmax_token_rollout,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        GRID_H,
        GRID_W,
        local_to_store,
    )

    dev = torch.device(device)
    ctx = int(sample.context_frames)
    stream_names = list(sample.signals.keys())
    n = int(np.asarray(sample.frames).shape[0])
    col, resolved_key = _resolve_knob_col(sample, knob_key)
    fracs = [float(f) for f in fracs]

    grids: dict[str, np.ndarray] = {}
    roles: list[dict] = []
    tok_by_frac: dict[float, np.ndarray] = {}
    for f in fracs:
        plan = (
            sample.actuator
            if abs(f) < 1e-9
            else _bounded_coil_edit(sample.actuator, col, frac=f)
        )
        tok = _argmax_token_rollout(
            model,
            sample,
            stream_names,
            _actuator_batch_from_plan(plan, dev),
            dev,
            chunk=chunk,
        )
        if stop and stop.stop:
            raise KeyboardInterrupt("stopped during sweep rollouts")
        tok_by_frac[f] = tok
        role = f"frac{f:+.2f}"
        grids[role] = local_to_store(tok.reshape(-1, GRID_H, GRID_W))
        roles.append({"role": role})

    decoded = decode_roles(
        grids, roles, work_dir=_decode_work_dir("sweep"), device=device
    )

    # forecast-mean centroid (row, col) per knob value — the turn-the-knob curve.
    centroids = {f: decoded_centroid(decoded[f"frac{f:+.2f}"]) for f in fracs}
    knob_curve = []
    for f in fracs:
        cen = centroids[f]
        fc = cen[ctx:] if cen.shape[0] > ctx else cen
        knob_curve.append((f, float(fc[:, 0].mean()), float(fc[:, 1].mean())))

    # N-up GIF over the full window.
    t = _rel_time_ms(sample, ctx)
    nframes = min(min(decoded[f"frac{f:+.2f}"].shape[0] for f in fracs), n)
    pretty = {f: (f"{knob_label(f)}  ({resolved_key} x{1 + f:.2f})") for f in fracs}
    frames = []
    for i in range(nframes):
        panels = [_to_aspect(decoded[f"frac{f:+.2f}"][i]) for f in fracs]
        titles = [
            f"{pretty[f]}   [{'FORECAST' if i >= ctx else 'context'}]" for f in fracs
        ]
        frames.append(
            _panel_row_frame(
                panels,
                titles,
                banner=(
                    f"coil knob sweep — turn {resolved_key}  |  shot "
                    f"{int(sample.shot_id)}  |  t={t[i]:+.0f} ms"
                ),
                in_target=(i >= ctx),
            )
        )
    gif = out_dir / f"knob_sweep_shot{int(sample.shot_id)}_{resolved_key}.gif"
    _save_gif(frames, gif, fps=fps)

    png = _knob_curve_panel(
        out_dir / f"knob_sweep_shot{int(sample.shot_id)}_{resolved_key}_curve.png",
        knob_curve,
        shot_id=int(sample.shot_id),
        coil_key=resolved_key,
    )
    logger.info(
        "knob sweep %s (col %d): centroid-vs-knob %s",
        resolved_key,
        col,
        [(round(f, 2), round(r, 1), round(c, 1)) for f, r, c in knob_curve],
    )
    return {
        "demo": "coil_knob_sweep",
        "shot_id": int(sample.shot_id),
        "context_frames": ctx,
        "n_frames": int(nframes),
        "knob_key": resolved_key,
        "knob_col": int(col),
        "fracs": fracs,
        "knob_curve_frac_row_col": knob_curve,
        "gif_path": str(gif),
        "curve_png_path": str(png),
    }


def knob_label(frac: float) -> str:
    """A short signed label for a sweep value (``-`` / ``0 (true)`` / ``+``)."""
    if abs(frac) < 1e-9:
        return "0 (true plan)"
    return f"{frac:+.0%}"


# ---------------------------------------------------------------------------
# Shared plotting
# ---------------------------------------------------------------------------


def _centroid_panel(
    png_path: Path,
    t: np.ndarray,
    ctx: int,
    *,
    traces: list[tuple[str, np.ndarray, str]],
    suptitle: str,
) -> Path:
    """Two-axis centroid trace (row / col vs time) for an arbitrary set of traces."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), dpi=110)
    for ax, dim, name in ((axes[0], 0, "row (vertical)"), (axes[1], 1, "col (radial)")):
        for label, cen, color in traces:
            m = min(len(t), cen.shape[0])
            ax.plot(t[:m], cen[:m, dim], "-o", ms=3, label=label, color=color)
        if 0 <= ctx < len(t):
            ax.axvline(t[ctx], ls="--", color="#888", lw=1)
        ax.set_title(f"centroid {name}", fontsize=10)
        ax.set_xlabel("t (ms, rel. forecast start)")
        ax.set_ylabel("pixel")
        ax.legend(fontsize=8)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def _knob_curve_panel(
    png_path: Path,
    knob_curve: list[tuple[float, float, float]],
    *,
    shot_id: int,
    coil_key: str,
) -> Path:
    """Plot forecast-mean plasma centroid (row & col) vs the knob value."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fr = np.asarray([k[0] for k in knob_curve], dtype=np.float64)
    row = np.asarray([k[1] for k in knob_curve], dtype=np.float64)
    col = np.asarray([k[2] for k in knob_curve], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=110)
    ax.plot(fr * 100.0, row, "-o", color="#1f77b4", label="centroid row (vertical)")
    ax.plot(fr * 100.0, col, "-s", color="#d62728", label="centroid col (radial)")
    ax.axvline(0.0, ls="--", color="#888", lw=1)
    ax.set_xlabel(f"{coil_key} knob (% of true trajectory)")
    ax.set_ylabel("forecast-mean centroid (pixel)")
    ax.set_title(f"shot {shot_id}: plasma centroid vs {coil_key} knob")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build(
    *,
    checkpoint: Path,
    shot_id: int,
    token_root: Path | None,
    out_dir: Path,
    device: str = "cuda",
    chunk: int = 8192,
    camera: str = "rbb",
    n_frames: int = 24,
    n_plan: int = 8,
    context_frames: int = 8,
    frame_stride: int = 1,
    target_horizon_s: float = 0.25,
    n_signal_steps: int = 4,
    n_act_steps: int = 8,
    knob_key: str = DEFAULT_KNOB_KEY,
    counterfactual_scale: float = DEFAULT_COUNTERFACTUAL_SCALE,
    fps: int = FPS,
) -> dict:
    """Load the model once, assemble the shot once, render all three demos."""
    import torch  # noqa: PLC0415

    from imas_ambix.worldmodel.controllable_eval import (  # noqa: PLC0415
        EvalConfig,
        _assemble_heldout,
    )
    from imas_ambix.worldmodel.controllable_train import (  # noqa: PLC0415
        load_controllable_model_from_checkpoint,
    )
    from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: PLC0415
        SpacetimeWindowConfig,
    )

    stop = _StopFlag()
    stop.install()

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — demos require the VQ decode; using CPU")
        device = "cpu"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(
        held_out=(int(shot_id),),
        chunk=chunk,
        n_signal_steps=n_signal_steps,
        n_act_steps=n_act_steps,
        window=SpacetimeWindowConfig(
            n_frames=n_frames,
            n_plan=n_plan,
            context_frames=context_frames,
            frame_stride=frame_stride,
            target_horizon_s=target_horizon_s,
        ),
    )

    model = None
    results: dict = {"checkpoint": str(checkpoint), "shot_id": int(shot_id)}
    try:
        model, payload = load_controllable_model_from_checkpoint(
            Path(checkpoint), map_location=device
        )
        model.eval()
        results["checkpoint_step"] = int(payload.get("step", -1))
        logger.info(
            "loaded controllable checkpoint step=%s on %s",
            results["checkpoint_step"],
            device,
        )

        sample = _assemble_heldout(
            int(shot_id), cfg, camera=camera, token_root=token_root
        )
        logger.info(
            "assembled shot %s: %d frames, context %d, start_frame %s",
            int(shot_id),
            int(np.asarray(sample.frames).shape[0]),
            int(sample.context_frames),
            getattr(sample, "start_frame", "?"),
        )

        results["faithful_replay"] = faithful_replay(
            model,
            sample,
            device=device,
            out_dir=out_dir,
            chunk=chunk,
            fps=fps,
            stop=stop,
        )
        results["counterfactual"] = counterfactual_two_plans(
            model,
            sample,
            device=device,
            out_dir=out_dir,
            chunk=chunk,
            scale=counterfactual_scale,
            fps=fps,
            stop=stop,
        )
        results["knob_sweep"] = coil_knob_sweep(
            model,
            sample,
            device=device,
            out_dir=out_dir,
            chunk=chunk,
            knob_key=knob_key,
            fps=fps,
            stop=stop,
        )
        (out_dir / f"playable_demo_shot{int(shot_id)}.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
    finally:
        try:
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model release note: %r", exc)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rendered playable-plasma coil demos.")
    p.add_argument(
        "--checkpoint", required=True, help="the controllable best.pt/latest.pt"
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--token-root", default=None)
    p.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    p.add_argument("--camera", default="rbb")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--n-plan", type=int, default=8)
    p.add_argument("--context-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--target-horizon-s", type=float, default=0.25)
    p.add_argument("--n-signal-steps", type=int, default=4)
    p.add_argument("--n-act-steps", type=int, default=8)
    p.add_argument("--knob-key", default=DEFAULT_KNOB_KEY)
    p.add_argument(
        "--counterfactual-scale", type=float, default=DEFAULT_COUNTERFACTUAL_SCALE
    )
    p.add_argument("--fps", type=int, default=FPS)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = build(
        checkpoint=Path(args.checkpoint),
        shot_id=args.shot,
        token_root=Path(args.token_root) if args.token_root else None,
        out_dir=Path(args.out_dir),
        device=args.device,
        chunk=args.chunk,
        camera=args.camera,
        n_frames=args.n_frames,
        n_plan=args.n_plan,
        context_frames=args.context_frames,
        frame_stride=args.frame_stride,
        target_horizon_s=args.target_horizon_s,
        n_signal_steps=args.n_signal_steps,
        n_act_steps=args.n_act_steps,
        knob_key=args.knob_key,
        counterfactual_scale=args.counterfactual_scale,
        fps=args.fps,
    )

    print("\n=== playable-plasma demos ===")
    print(
        f"checkpoint: {results['checkpoint']} (step {results.get('checkpoint_step')})"
    )
    print(f"shot: {results['shot_id']}")
    for key in ("faithful_replay", "counterfactual", "knob_sweep"):
        d = results.get(key, {})
        print(f"  {key}: {d.get('gif_path')}")
    return 0


__all__ = [
    "DEFAULT_SHOT",
    "build",
    "coil_knob_sweep",
    "counterfactual_two_plans",
    "faithful_replay",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
