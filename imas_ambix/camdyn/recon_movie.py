"""Headline demonstration GIFs + static panels + forward-forecast evidence.

This is the *clear-evidence* companion to :mod:`reconstruction_demo`.  Where
that module renders the dense per-scenario grids, this one produces the
side-by-side animations and clean three-row panels the lead asked for, all
comparing the trained **dynamics** arm against the trained per-frame
**baseline** arm and the ground truth.

Every comparison row set here is exactly three rows::

    ground truth        (raw level-1 camera frames)
    per-frame baseline  (cap_v1_baseline, temporal OFF — decoded full frame)
    dynamics            (cap_v1_dynamics, temporal ON  — decoded full frame)

The trivial zero-order-hold floor and the decoded-from-true-tokens round-trip
row are intentionally absent: the baseline arm is a real 202M model that emits
coherent full frames, so it is the honest visual yardstick.

Per-frame display normalisation
--------------------------------
The token/decode pipeline normalises per-SHOT, so dim ramp-up frames vanish
and flat-top brightness swamps structure.  For display each TIME column takes
robust (1st/99th percentile) limits from the *ground-truth* frame at that time
and applies the SAME vmin/vmax to all three rows (see
:func:`reconstruction_demo.display_limits`): structure is legible while genuine
brightness over/under-shoot in the reconstructions stays visible.

Deliverables (all under ``docs/figures/camera-dynamics-wm/``)
-------------------------------------------------------------
* ``recon-from-window.gif`` — clipped-view: the model sees only the
  ``fixed_section2`` sub-window stream (clip box outlined on the GT pane); the
  right pane is the dynamics full-frame reconstruction marching in time.
* ``recon-from-signals.gif`` — full mask: no camera input; the right pane is
  the dynamics reconstruction from the actuator signals alone.
* ``forecast-rollout.gif`` — observe the first half of a wide decimated
  window, then watch predicted vs true frames march forward (FRONTIER mask).
* ``fig-cdw-recon-window.png`` / ``fig-cdw-recon-signals.png`` — three-row
  static panels on a flat-top high-activity window.
* ``fig-cdw-recon-rampup.png`` — three-row panels on a discharge ramp-up
  window (rising Ip / brightness, low signal).
* ``fig-cdw-forecast-sweep.png`` + ``artifacts/forecast_sweep.json`` —
  reconstruction-quality-vs-horizon curve (dynamics vs persistence vs
  baseline) showing how far forward the dynamics arm beats persistence.

Decode architecture is shared with :mod:`reconstruction_demo`: the predict
phase (this venv, GPU) loads BOTH trained arms, runs them forward, and dumps
token grids; the decode phase (the Open-MAGVIT2 venv) decodes every grid to a
256² image through the SAME frozen VQModel.

Run (predict + decode + all deliverables, on a GPU node)::

    .venv/bin/python -m imas_ambix.camdyn.recon_movie \\
        --out docs/figures/camera-dynamics-wm/ \\
        --artifact imas_ambix/camdyn/artifacts/forecast_sweep.json
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.camdyn import reconstruction_demo as rd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

GRID_H, GRID_W = rd.GRID_H, rd.GRID_W
ORIGINAL_HW = rd.ORIGINAL_HW
ACCENT = rd.ACCENT

#: Finer physical horizons (ms) for the "how far forward" sweep — denser than
#: the locked W2 set so the persistence crossover is visible, but scored with
#: the SAME pre-registered horizon→offset machinery.
SWEEP_HORIZONS_MS: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)


# ===========================================================================
# Per-frame display normalisation → uint8 for GIF / array panels
# ===========================================================================


def normalise_for_display(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a float/gray image to uint8 ``[0,255]`` using fixed display limits.

    ``vmin``/``vmax`` are the per-column robust limits from
    :func:`reconstruction_demo.display_limits` (computed on the GT frame and
    shared across all three rows of that column).  Values below ``vmin`` clamp
    to black, above ``vmax`` clamp to white — so a reconstruction that
    over/under-shoots the GT brightness shows it honestly rather than being
    re-stretched into range.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    span = max(vmax - vmin, 1e-9)
    out = (a - vmin) / span
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def to_native_gray(img: np.ndarray) -> np.ndarray:
    """Resize a uint8 gray image to the canonical rbb aspect ``ORIGINAL_HW``.

    The raw level-1 rbb frames are NOT all the same native resolution across
    shots (older shots carry larger sensor frames), while the decoded model
    panes are always ``ORIGINAL_HW`` (256² → 112×156 via ``_to_aspect``).  The
    side-by-side GIF panes are concatenated as raw arrays, so the GT pane must
    be brought to the SAME aspect first — otherwise the heights mismatch.
    """
    from PIL import Image

    a = np.asarray(img, dtype=np.uint8)
    if a.shape[:2] == ORIGINAL_HW:
        return a
    im = Image.fromarray(a).resize((ORIGINAL_HW[1], ORIGINAL_HW[0]), Image.BILINEAR)
    return np.asarray(im)


def _draw_clip_box(img: np.ndarray, box, *, value: int = 255) -> np.ndarray:
    """Outline a token-grid clip box on a native-aspect uint8 image (in place)."""
    if box is None:
        return img
    r0, r1, c0, c1 = box
    h, w = img.shape[:2]
    rr0 = int(round(r0 / GRID_H * h))
    rr1 = int(round(r1 / GRID_H * h))
    cc0 = int(round(c0 / GRID_W * w))
    cc1 = int(round(c1 / GRID_W * w))
    rr0 = max(0, min(h - 1, rr0))
    rr1 = max(0, min(h - 1, rr1))
    cc0 = max(0, min(w - 1, cc0))
    cc1 = max(0, min(w - 1, cc1))
    img[rr0, cc0:cc1] = value
    img[rr1, cc0:cc1] = value
    img[rr0:rr1, cc0] = value
    img[rr0 : rr1 + 1, cc1] = value
    return img


def _label(img_rgb: np.ndarray, text: str) -> np.ndarray:
    """Stamp a small text label in the top-left corner of an RGB uint8 frame."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im)
    # a faint dark plate behind the text for legibility on bright frames
    draw.rectangle([0, 0, max(40, 7 * len(text)), 12], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 0))
    return np.asarray(im)


def side_by_side_frame(
    gt_gray: np.ndarray,
    model_gray: np.ndarray,
    *,
    scale: int,
    gt_label: str,
    model_label: str,
    counter: str,
    gap: int = 4,
) -> np.ndarray:
    """Assemble one GIF frame: GT (left) | model (right), labelled, scaled.

    Inputs are uint8 grayscale at native aspect; outputs are an RGB uint8
    array with the two panes side by side, a small label on each pane and a
    frame/time counter on the GT pane.  Pure numpy + PIL (no matplotlib) so it
    is cheap to call once per animation frame.
    """
    from PIL import Image

    def _up(gray):
        im = Image.fromarray(np.asarray(gray, dtype=np.uint8)).resize(
            (gray.shape[1] * scale, gray.shape[0] * scale), Image.NEAREST
        )
        return np.asarray(im.convert("RGB"))

    left = _up(gt_gray)
    right = _up(model_gray)
    left = _label(left, gt_label)  # row 1: "ground truth"
    left = _label_second_line(left, counter)  # row 2: "frame N  t=… ms"
    right = _label(right, model_label)
    h = max(left.shape[0], right.shape[0])
    sep = np.zeros((h, gap, 3), dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


def _label_second_line(img_rgb: np.ndarray, text: str) -> np.ndarray:
    """Stamp a second label line just under the top-left label."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 13, max(40, 7 * len(text)), 25], fill=(0, 0, 0))
    draw.text((2, 14), text, fill=(255, 255, 0))
    return np.asarray(im)


def panel_strip(
    panes_gray: list[np.ndarray],
    labels: list[str],
    *,
    scale: int,
    counter: str,
    gap: int = 4,
) -> np.ndarray:
    """Assemble one GIF/panel frame from N grayscale panes, left→right.

    Each pane is nearest-upscaled by ``scale``, labelled top-left, and joined
    by a thin black separator.  The frame/time ``counter`` is stamped as a
    second label line on the FIRST (ground-truth) pane.  Pure numpy + PIL so
    it is cheap to call once per animation frame.  The convention used by all
    deliverables is ``panes = [ground truth, static comparator, dynamics]``
    (baseline for reconstruction, persistence for forecasting).
    """
    from PIL import Image

    # Bring every pane to a common (H, W) BEFORE scaling so panes from
    # different native resolutions (e.g. a larger raw rbb sensor frame vs a
    # decoded 112×156 model frame) concatenate cleanly.
    h0 = max(np.asarray(p).shape[0] for p in panes_gray)
    w0 = max(np.asarray(p).shape[1] for p in panes_gray)

    def _up(gray):
        a = np.asarray(gray, dtype=np.uint8)
        im = Image.fromarray(a)
        if a.shape[:2] != (h0, w0):
            im = im.resize((w0, h0), Image.BILINEAR)
        im = im.resize((w0 * scale, h0 * scale), Image.NEAREST)
        return np.asarray(im.convert("RGB"))

    rgb = [_up(p) for p in panes_gray]
    for i, lab in enumerate(labels):
        rgb[i] = _label(rgb[i], lab)
    rgb[0] = _label_second_line(rgb[0], counter)  # counter on the GT pane

    h = rgb[0].shape[0]
    sep = np.zeros((h, gap, 3), dtype=np.uint8)
    out: list[np.ndarray] = []
    for i, p in enumerate(rgb):
        out.append(p)
        if i < len(rgb) - 1:
            out.append(sep)
    return np.concatenate(out, axis=1)


def write_gif(frames: list[np.ndarray], out_path: Path, *, duration_ms: int = 120):
    """Write an RGB-uint8 frame list to an animated GIF via PIL (no imagemagick)."""
    from PIL import Image

    pil = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil[0].save(
        str(out_path),
        save_all=True,
        append_images=pil[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    logger.info("[movie] wrote %s (%d frames)", out_path, len(frames))


# ===========================================================================
# Forecast horizon decimation (wide native window → n_frames spanning h ms)
# ===========================================================================


def forecast_stride_for(
    median_dt_s: float, n_frames: int, horizon_ms: float
) -> tuple[int, bool]:
    """Decimation stride so a ``n_frames``-window spans ``horizon_ms``.

    Mirrors :func:`horizon_eval.matched_stride_for` but takes the median Δt
    directly (so it is unit-testable without a corpus).  Returns
    ``(stride, reachable)``; ``reachable`` is False when even the fully
    decimated wide window cannot span the horizon (then the horizon is
    honestly reported as out of window, never faked).
    """
    if not np.isfinite(median_dt_s) or median_dt_s <= 0 or n_frames < 2:
        return 1, False
    frames_needed = (horizon_ms / 1000.0) / median_dt_s
    stride = max(1, int(np.ceil(frames_needed / (n_frames - 1))))
    reach_ms = (n_frames - 1) * stride * median_dt_s * 1000.0
    return stride, bool(reach_ms >= horizon_ms - 1e-9)


def decimated_indices(
    n_wide: int, n_target: int, median_dt_s: float, horizon_ms: float
) -> np.ndarray:
    """Indices into a wide native window picking ``n_target`` frames spanning h.

    The kept frames are ``arange(n_target) * stride`` clipped to the wide
    window, so the model (Δt-conditioned) sees the wider spacing through its
    dt input.  Falls back to a linspace when the cadence is unknown.
    """
    if n_wide <= n_target:
        return np.arange(n_wide, dtype=int)
    stride, _ = forecast_stride_for(median_dt_s, n_target, horizon_ms)
    stride = min(stride, max(1, (n_wide - 1) // max(1, n_target - 1)))
    idx = np.arange(n_target, dtype=int) * stride
    idx = idx[idx < n_wide]
    if idx.size < 2:
        idx = np.linspace(0, n_wide - 1, n_target).round().astype(int)
    return idx


# ===========================================================================
# Ramp-up window finder (rising plasma current / brightness, early in shot)
# ===========================================================================


def rampup_score(
    plasma_current: np.ndarray,
    brightness: np.ndarray,
    *,
    min_current_frac: float = 0.10,
) -> float:
    """Score a candidate window for "discharge ramp-up" character.

    A ramp-up window has a clearly RISING plasma current (positive slope, not
    yet at flat-top) and rising — but still sub-flat-top — brightness.  The
    score is the normalised mean forward slope of |Ip| over the window, gated
    to zero when the window is already near its own peak current (so flat-top
    windows do not score as ramp-up).  Higher = more ramp-up-like.
    """
    ip = np.abs(np.asarray(plasma_current, dtype=np.float64).reshape(-1))
    if ip.size < 2 or not np.isfinite(ip).all():
        return 0.0
    span = float(ip.max())
    if span <= 0:
        return 0.0
    rise = float(np.mean(np.diff(ip)))  # mean forward step
    # gate: the window must START well below its own end (genuinely rising)
    start_frac = float(ip[0] / span)
    end_frac = float(ip[-1] / span)
    if rise <= 0 or start_frac > 1.0 - min_current_frac:
        return 0.0
    # combine current rise with brightness rise (both normalised)
    b = np.asarray(brightness, dtype=np.float64).reshape(-1)
    b_rise = 0.0
    if b.size >= 2 and np.isfinite(b).all() and b.max() > 0:
        b_rise = float(np.mean(np.diff(b)) / b.max())
    return (rise / span) * 1000.0 + max(0.0, b_rise) * 10.0 + (end_frac - start_frac)


# ===========================================================================
# Persistence comparator (forecast mode): freeze the last OBSERVED frame
# ===========================================================================


def persistence_tokens(true_tokens: np.ndarray, frontier: int) -> np.ndarray:
    """Forecast persistence: every frame = the last OBSERVED frame's grid.

    The honest static comparator for the FRONTIER/forecast mode: copy the
    token grid at ``frontier - 1`` (the last frame the model actually saw)
    into every frame, so when it is decoded the middle GIF pane holds a frozen
    pre-frontier image while truth and the dynamics arm evolve.  Frames before
    the frontier keep their own (observed) tokens so the pre-forecast portion
    of the rollout is faithful.
    """
    tok = np.asarray(true_tokens, dtype=np.int64)
    n = tok.shape[0]
    last_obs = max(0, min(n - 1, frontier - 1))
    out = tok.copy()
    out[frontier:] = tok[last_obs]
    return out


# ===========================================================================
# ELM window finder (transient Dα spikes — edge-localized-mode signature)
# ===========================================================================


def elm_spike_score(dalpha: np.ndarray) -> tuple[float, int]:
    """Score a window for ELM character from its Dα trace + locate the peak.

    ELMs are fast, bright, quasi-periodic Dα bursts: a sharp transient spike
    well above the local baseline.  The score is the height of the strongest
    in-window spike above the window's median, normalised by the robust
    spread (MAD) so it is a "how many sigma is the burst" measure — high only
    when there is a genuine sharp transient, not a slow ramp.  Returns
    ``(score, peak_frame)``; ``score`` is 0 for a flat / monotone trace.

    Dα is used for WINDOW SELECTION ONLY — it is never a model input, so this
    is not leakage (the model still ingests only the clipped camera +
    actuators).
    """
    d = np.asarray(dalpha, dtype=np.float64).reshape(-1)
    if d.size < 3 or not np.isfinite(d).all():
        return 0.0, 0
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) or (float(d.std()) or 1.0)
    peak_frame = int(np.argmax(d))
    peak_height = (float(d.max()) - med) / (mad + 1e-9)
    # require the peak to be transient: it must fall off on at least one side
    # (a monotone rise to the last frame is a ramp, not a burst)
    rises_in = peak_frame > 0 and d[peak_frame] > d[peak_frame - 1]
    falls_out = peak_frame < d.size - 1 and d[peak_frame] > d[peak_frame + 1]
    if not (rises_in and falls_out):
        peak_height *= 0.2  # heavily discount edge-of-window / monotone peaks
    return float(max(0.0, peak_height)), peak_frame


def camera_brightness_trace(frames: np.ndarray, *, edge_rows: int = 30) -> np.ndarray:
    """Per-frame brightness proxy for ELM detection from RAW rbb frames.

    ELMs show as fast transient brightening of the plasma EDGE / divertor
    region.  We combine the whole-frame mean with the bottom-``edge_rows``
    (lower divertor) mean so an edge-localised burst is not washed out by the
    bulk.  ``frames`` is ``(F, H, W)``; returns ``(F,)``.  Detecting the burst
    directly in the camera is alignment-free: the burst frame index maps 1:1
    to the token-window frame (temporal_compression = 1), no cross-diagnostic
    time-base juggling.
    """
    f = np.asarray(frames, dtype=np.float64)
    if f.ndim != 3:
        f = f.reshape(f.shape[0], -1)[:, None, :]
    whole = f.reshape(f.shape[0], -1).mean(axis=1)
    er = min(edge_rows, f.shape[1])
    edge = f[:, -er:, :].reshape(f.shape[0], -1).mean(axis=1)
    return 0.5 * whole + 0.5 * edge


def camera_elm_score(frames: np.ndarray, *, edge_rows: int = 30) -> tuple[float, int]:
    """Score a window for ELM character directly from RAW rbb camera frames.

    High-passes the per-frame edge/divertor brightness trace (subtract a
    3-frame moving baseline) and measures the strongest transient burst above
    the robust spread — exactly the sub-ms edge brightening an ELM produces.
    Returns ``(score, peak_frame)``; the score is 0 for a flat or monotone
    window.  This needs NO Dα / cross-diagnostic alignment.
    """
    b = camera_brightness_trace(frames, edge_rows=edge_rows)
    if b.size < 3 or not np.isfinite(b).all():
        return 0.0, 0
    # high-pass: subtract a centred 3-frame moving mean to isolate transients
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(b, kernel, mode="same")
    hp = b - smooth
    # reuse the transient-spike scorer on the high-passed trace
    return elm_spike_score(hp)


# ===========================================================================
# Static three-row panel (GT / baseline / dynamics), per-frame normalised
# ===========================================================================


def assemble_three_row_panel(
    scenario: str,
    meta_entry: dict,
    images: np.ndarray,
    slot: dict,
    window_index: int,
    raw_frames: np.ndarray | None,
    *,
    out_path: Path,
    title_extra: str = "",
    middle_role: str = "baseline",
    middle_name: str = "per-frame baseline",
    highlight_frame: int | None = None,
):
    """Three-row GT / static-comparator / dynamics panel for one window.

    ``images`` ``(N,F,256,256,3)`` decoded bundle; ``slot`` maps
    ``(window, scenario, role)`` → image index.  The MIDDLE row is the best
    static comparator for the mode: ``"baseline"`` (the trained per-frame arm)
    for reconstruction, ``"persistence"`` (the frozen last-observed frame) for
    forecasting.  Columns are spread across the window (or anchored on
    ``highlight_frame``, e.g. an ELM Dα peak); each column shares the GT
    frame's robust display limits across all three rows so structure is
    legible while reconstruction over/under-shoot stays honest.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    ft = np.asarray(meta_entry["frame_time"], dtype=float)
    n_frames = ft.shape[0]
    cols = list(np.linspace(0, n_frames - 1, 5).round().astype(int))
    if highlight_frame is not None and 0 <= highlight_frame < n_frames:
        # ensure the burst frame (and a couple around it) are among the columns
        anchor = [
            max(0, highlight_frame - 2),
            highlight_frame,
            min(n_frames - 1, highlight_frame + 2),
        ]
        cols = sorted(set(cols) | set(anchor))
    mid = images[slot[(window_index, scenario, middle_role)]]
    dyn = images[slot[(window_index, scenario, "dynamics")]]
    box = rd.clip_box(scenario)

    row_names = ["truth (raw)", middle_name, "dynamics"]
    n_rows, n_cols = 3, len(cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.7 * n_cols + 1.4, 1.55 * n_rows + 0.7),
        squeeze=False,
        constrained_layout=True,
    )

    for ci, fi in enumerate(cols):
        # per-column display limits from the GT frame
        if raw_frames is not None and fi < raw_frames.shape[0]:
            gt = raw_frames[fi].astype(np.float64)
        else:
            gt = rd._to_aspect(mid[fi]).astype(np.float64)
        vmin, vmax = rd.display_limits(gt)
        dt_ms = (ft[fi] - ft[0]) * 1e3

        triples = [gt, rd._to_aspect(mid[fi]), rd._to_aspect(dyn[fi])]
        for ri in range(3):
            ax = axes[ri][ci]
            rd._imshow_cam(ax, triples[ri], vmin=vmin, vmax=vmax)
            if box is not None:
                r0, r1, c0, c1 = box
                ax.add_patch(
                    mpatches.Rectangle(
                        (
                            c0 / GRID_W * ORIGINAL_HW[1] - 0.5,
                            r0 / GRID_H * ORIGINAL_HW[0] - 0.5,
                        ),
                        (c1 - c0) / GRID_W * ORIGINAL_HW[1],
                        (r1 - r0) / GRID_H * ORIGINAL_HW[0],
                        fill=False,
                        edgecolor=ACCENT,
                        linewidth=1.3,
                    )
                )
            if ri == 0:
                burst = (
                    "  (ELM)"
                    if (highlight_frame is not None and fi == highlight_frame)
                    else ""
                )
                ax.set_title(f"{dt_ms:+.1f} ms{burst}", fontsize=8)
            if ci == 0:
                ax.set_ylabel(row_names[ri], fontsize=8)

    bits = [
        rd.SCENARIO_TITLE.get(scenario, scenario),
        f"shot {meta_entry['shot_id']}",
        f"{ft[0] * 1e3:.0f}-{ft[-1] * 1e3:.0f} ms",
        "per-column GT 1/99-pct norm",
        "frozen Open-MAGVIT2 decoder",
    ]
    if title_extra:
        bits.insert(1, title_extra)
    fig.suptitle("camera-dynamics-wm — " + " | ".join(bits), fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[movie] wrote %s", out_path)


if __name__ == "__main__":  # pragma: no cover - GPU orchestration entry point
    from imas_ambix.camdyn.recon_movie_run import main

    raise SystemExit(main())
