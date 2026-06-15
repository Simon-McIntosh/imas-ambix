"""Does mode-decode average away the persistent edge filaments the model knows?

A diagnostic probe that separates a DECODE-MODE artifact from a HEAD/OBJECTIVE
failure for the camera-dynamics world model.

Mechanism under test
---------------------
The model head emits 18 INDEPENDENT bit-logits (a bitwise-factorised LFQ
likelihood).  Both scoring and rendering currently decode each cell by per-bit
MAP::

    pred_id = Σ_b (z_b > 0) << b

(``reconstruction_demo._bit_map_tokens`` / ``model.score_window_bits``).  Under
genuine aleatoric uncertainty about the exact position of a bright edge/SOL
filament, the per-bit MAP collapses to a smeared "mean" token grid — so a
predicted region blurs out the striations the ground-truth frames clearly show.

Hypothesis: the model's per-bit distribution still CONTAINS the filament
structure; taking the mode destroys it.  Test: decode by SAMPLING instead of
MAP and measure whether the high-spatial-frequency power and edge contrast that
MAP suppresses come back.

Three token-grid decode paths from the SAME logits ``z_b`` (the tensor BEFORE
the ``>0`` threshold), for the genuinely-PREDICTED region of a real scenario:

* **MAP** — the current decode, ``bit = (z_b > 0)``.
* **per-bit sample** — independent Bernoulli per bit, ``bit ~ σ(z_b / T)`` at
  temperatures ``T ∈ {0.7, 1.0}``.  Tests the bit-independence hypothesis.
* **joint sample** — a restricted-vocabulary categorical sample over a
  candidate set (the true token + its single-bit / two-bit neighbours) using
  the exact bit-factorised token log-likelihood
  (``model.bit_logits_to_token_logits``).  Tests whether a COHERENT joint
  sample (vs the independent-bit sample) is needed to recover plausible
  texture.

Diagnosis
---------
Two distinguishable outcomes, both informative:

* The model is HEDGING — high per-bit predictive entropy in the edge-filament
  region (it represents uncertainty over filament position).  Then sampling
  restores sharp plausible texture and HF power toward ground truth → the blur
  is a MAP-decode ARTIFACT, fixable with sampled / temperature decode (and
  ancestral sampling in rollout) and NO retraining.
* The model is COLLAPSED / confidently-wrong — low per-bit entropy yet the
  decode is blurred, and per-bit sampling produces incoherent salt-and-pepper
  noise rather than filaments.  Then the bit-independence head cannot represent
  the coherent joint and the fix is a richer head / structure-aware objective,
  not a decode change.

Reuses the existing predict→decode infrastructure verbatim: window selection
and decimation from :mod:`recon_movie_run` / :mod:`reconstruction_demo`, the
``BundleBuilder`` token→image handoff, and the frozen Open-MAGVIT2 decoder
subprocess.  Runs on a compute node with GPFS + the OMAG2 decode venv + a GPU
(no network needed — checkpoint, tokens, raw frames and the venv are on GPFS).

Run (predict + decode + figure + metrics, on a GPU node)::

    .venv/bin/python -m imas_ambix.camdyn.structure_fidelity
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.camdyn import recon_movie as mv
from imas_ambix.camdyn import recon_movie_run as mvr
from imas_ambix.camdyn import reconstruction_demo as rd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Output figure (committed).  Rows = GT | MAP | sample T=0.7 | sample T=1.0 |
#: joint, columns = time across the predicted window.
DEFAULT_FIGURE = Path("docs/figures/camera-dynamics-wm/fig-cdw-filament-decode.png")

#: Held-out flat-top shots scanned for the strongest persistent edge structure.
#: 24446 first — the tokenizer round-trip retains ~95 % of fine edge energy
#: there (``tokenizer_fidelity``), so any blur in the PREDICTION cannot be
#: blamed on the tokenizer: the structure is representable in the token grid.
FLATTOP_SHOTS = (24446, 24065, 23937)

#: Forecast scenario: the model sees full frames up to ``FRONTIER`` and must
#: PREDICT the rest — the edge filaments in the post-frontier frames are
#: genuinely forecast, not copied.  Matches the existing GIF scenarios.
SCENARIO = "frontier"
FRONTIER = 8
N_FRAMES = 16

#: Decimate the wide native window so the predicted span covers a few ms of
#: real plasma evolution (a contiguous 16-frame rbb window spans < 1 ms).
SPAN_MS = 6.0

#: Sampling temperatures for the per-bit Bernoulli decode.
TEMPERATURES = (0.7, 1.0)

#: Random seed for the stochastic decodes (reproducible).
SEED = 12345


# ---------------------------------------------------------------------------
# Window selection — strongest persistent edge structure on a flat-top shot
# ---------------------------------------------------------------------------


def _edge_rows(grid_h: int = mv.GRID_H, n: int = 4) -> slice:
    """Token-grid rows that map to the lower/divertor edge of the rbb frame.

    The persistent bright edge/SOL striations live in the lower portion of the
    rbb field of view, so the bottom ``n`` token rows are the edge-filament
    region used for the entropy / contrast diagnostics.
    """
    return slice(grid_h - n, grid_h)


def _persistent_edge_power(raw_frames: np.ndarray, frontier: int) -> float:
    """Persistent high-spatial-frequency power in the POST-FRONTIER edge region.

    We want a window whose ground truth shows clear, MULTI-FRAME (persistent,
    not a single-frame flash) bright edge striations in the part the model must
    PREDICT.  For each post-frontier frame we high-pass the lower-edge band of
    the raw frame (subtract a box-blurred copy) and take the variance of the
    residual — the fine edge-striation power.  The window score is the MINIMUM
    across post-frontier frames (so a single bright frame cannot win; the
    structure must persist) times the mean (so brighter is preferred).
    """
    from PIL import Image, ImageFilter

    f = np.asarray(raw_frames, dtype=np.float64)
    if f.ndim != 3 or f.shape[0] <= frontier:
        return 0.0
    er = max(8, f.shape[1] // 3)
    powers = []
    for fi in range(frontier, f.shape[0]):
        band = f[fi, -er:, :]
        if band.size == 0 or not np.isfinite(band).all():
            powers.append(0.0)
            continue
        bmax = float(band.max()) or 1.0
        u8 = np.clip(band / bmax * 255.0, 0, 255).astype(np.uint8)
        blur = np.asarray(
            Image.fromarray(u8).filter(ImageFilter.BoxBlur(3)), dtype=np.float64
        )
        hp = u8.astype(np.float64) - blur
        powers.append(float(np.var(hp)))
    if not powers:
        return 0.0
    return float(np.min(powers)) * float(np.mean(powers)) ** 0.5


@dataclass
class StructureWindow:
    """The selected forecast window + why it carries persistent edge structure."""

    window: rd.DemoWindow
    edge_power: float


def select_structure_window(
    *,
    shots=FLATTOP_SHOTS,
    n_frames: int = N_FRAMES,
    frontier: int = FRONTIER,
    span_ms: float = SPAN_MS,
    wide_factor: int = 16,
) -> StructureWindow | None:
    """Pick the flat-top window with the strongest PERSISTENT post-frontier edge.

    Scans bright flat-top windows of the candidate held-out shots (the same
    brightness-ranked selection the demo uses), decimates each to ``n_frames``
    spanning ``span_ms`` so real plasma evolution is visible, and ranks by the
    persistent post-frontier lower-edge high-frequency power
    (:func:`_persistent_edge_power`).  The winner has clear, multi-frame bright
    edge striations in exactly the region the model must forecast.
    """
    from imas_ambix.camdyn.dataset import (
        FrameTokenDataset,
        FrameWindowConfig,
        discover_token_shots,
    )

    best: StructureWindow | None = None
    best_score = 0.0
    wide_n = n_frames * wide_factor
    for sid in shots:
        specs = discover_token_shots(shot_ids=[sid], read_n_frames=True)
        if not specs or specs[0].level1_path is None:
            continue
        ds = FrameTokenDataset(
            specs, FrameWindowConfig(n_frames=wide_n, stride=wide_n, seed=0)
        )
        if len(ds) == 0:
            continue
        starts = [ds._windows[i][1] for i in range(len(ds))]
        bright = rd._window_brightness(int(sid), starts, wide_n)
        order = (
            list(np.argsort(-bright)) if bright is not None else list(range(len(ds)))
        )
        # consider the few brightest wide windows per shot
        for pick in order[:4]:
            win = ds[int(pick)]
            base = rd.DemoWindow(
                shot_id=int(win.shot_id),
                start=int(win.start),
                frame_time=np.asarray(win.frame_time, dtype=np.float64),
                dt=np.asarray(win.dt, dtype=np.float64),
                valid=np.asarray(win.valid_frames, dtype=bool),
                true_tokens=np.asarray(win.tokens, dtype=np.int64),
                motion_fraction=0.0,
            )
            dwin = _decimate(base, span_ms=span_ms, n_frames=n_frames)
            if dwin is None:
                continue
            raw = rd.load_raw_frames(
                dwin.shot_id, dwin.start, dwin.true_tokens.shape[0]
            )
            if raw is None or raw.shape[0] < n_frames:
                continue
            # restrict to canonical-aspect rbb so the edge region is the
            # production geometry the world model trains on.
            if abs(raw.shape[1] - mv.ORIGINAL_HW[0]) > 24:
                continue
            score = _persistent_edge_power(raw.astype(np.float64), frontier)
            logger.info(
                "[structfid] candidate shot %d start %d span %.1f ms edge-power=%.3e",
                dwin.shot_id,
                dwin.start,
                (dwin.frame_time[-1] - dwin.frame_time[0]) * 1e3,
                score,
            )
            if score > best_score:
                best_score = score
                best = StructureWindow(window=dwin, edge_power=float(score))
    if best is not None:
        w = best.window
        ft = np.asarray(w.frame_time, dtype=float)
        logger.info(
            "[structfid] SELECTED shot %d start %d window %.1f-%.1f ms "
            "(frontier@f%d t=%.1f ms) persistent-edge-power=%.3e",
            w.shot_id,
            w.start,
            ft[0] * 1e3,
            ft[-1] * 1e3,
            FRONTIER,
            ft[FRONTIER] * 1e3,
            best.edge_power,
        )
    else:
        logger.warning("[structfid] no flat-top edge window found")
    return best


def _decimate(
    base: rd.DemoWindow, *, span_ms: float, n_frames: int
) -> rd.DemoWindow | None:
    """Decimate a wide native window to ``n_frames`` spanning ``span_ms``.

    The wide window is read straight off disk, so reuse the movie driver's
    cadence-aware decimator (Δt-conditioned model sees the wider spacing).
    """
    ft0 = np.asarray(base.frame_time, dtype=np.float64)
    if ft0.size < 2:
        return None
    dt_med = float(np.median(np.diff(ft0)))
    idx = mv.decimated_indices(ft0.shape[0], n_frames, dt_med, span_ms)
    if idx.size < n_frames:
        # pad with the last frame so every grid shares the frame axis
        idx = np.concatenate([idx, np.repeat(idx[-1:], n_frames - idx.size)])
    idx = idx[:n_frames]
    tok = np.asarray(base.true_tokens, dtype=np.int64)[idx]
    ftd = ft0[idx]
    dt = (
        np.concatenate([np.diff(ftd), np.diff(ftd)[-1:]])
        if ftd.size > 1
        else np.zeros_like(ftd)
    )
    valid = np.asarray(base.valid, dtype=bool)[idx]
    return rd.DemoWindow(
        shot_id=int(base.shot_id),
        start=int(base.start),
        frame_time=ftd,
        dt=dt.astype(np.float64),
        valid=valid,
        true_tokens=tok,
        motion_fraction=0.0,
    )


# ---------------------------------------------------------------------------
# Forward — per-bit LOGITS (the tensor BEFORE the >0 MAP)
# ---------------------------------------------------------------------------


def forward_bit_logits(
    model, torch, device, win: rd.DemoWindow, cond_stats, scenario: str, frontier: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run one arm forward → ``(visible, bit_logits)``.

    ``bit_logits`` ``(F,H,W,bits)`` are the RAW per-bit logits z_b — the exact
    tensor before :func:`reconstruction_demo._bit_map_tokens` thresholds them at
    zero.  Mirrors :func:`reconstruction_demo.predict_window_arm` but returns
    the logits instead of the MAP token ids, so MAP / sampled / joint decodes
    all derive from the SAME forward pass.
    """
    from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS, load_conditioning
    from imas_ambix.camdyn.dataset import discover_token_shots

    n_frames = win.true_tokens.shape[0]
    visible = rd.scenario_mask(scenario, n_frames, frontier)

    specs = discover_token_shots(shot_ids=[win.shot_id], read_n_frames=False)
    level1_path = specs[0].level1_path if specs else None
    cond = load_conditioning(
        level1_path, win.frame_time, win.shot_id, channels=CONDITIONING_CHANNELS
    )
    cv = rd._zscore(cond.values, cond_stats)[None]
    cm = cond.missing[None].astype(np.float32)
    dt = win.dt[None].astype(np.float32)

    tokens_t = torch.from_numpy(win.true_tokens[None]).to(device)
    vis_t = torch.from_numpy(visible[None]).to(device)
    cv_t = torch.from_numpy(cv).to(device)
    cm_t = torch.from_numpy(cm).to(device)
    dt_t = torch.from_numpy(dt).to(device)

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model.module(tokens_t, vis_t, cv_t, cm_t, dt_t)
        bl = logits.float().cpu().numpy()[0]  # (F,H,W,bits)
    return visible, bl


# ---------------------------------------------------------------------------
# Decode strategies — bit logits → token grid (three ways)
# ---------------------------------------------------------------------------


def decode_map(bit_logits: np.ndarray) -> np.ndarray:
    """MAP token grid (the CURRENT decode): ``id = Σ_b (z_b > 0) << b``."""
    return rd._bit_map_tokens(bit_logits)


def decode_per_bit_sample(
    bit_logits: np.ndarray, *, temperature: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-bit Bernoulli sample: bit ``b`` is 1 with prob ``σ(z_b / T)``.

    Each of the 18 bits is sampled INDEPENDENTLY (the head's factorisation),
    then packed to the token id.  ``T < 1`` sharpens toward the MAP; ``T = 1``
    samples the head's native marginal; ``T > 1`` flattens it.
    """
    z = np.asarray(bit_logits, dtype=np.float64) / max(temperature, 1e-6)
    p = 1.0 / (1.0 + np.exp(-z))  # σ(z/T), (F,H,W,bits)
    bits = (rng.random(p.shape) < p).astype(np.int64)
    shifts = np.arange(bits.shape[-1], dtype=np.int64)
    return (bits << shifts).sum(axis=-1)


def decode_joint_sample(
    bit_logits: np.ndarray,
    true_tokens: np.ndarray,
    *,
    temperature: float,
    rng: np.random.Generator,
    n_neighbours: int = 2,
) -> np.ndarray:
    """Restricted-vocab JOINT (categorical) sample per cell.

    For each cell the candidate set is the TRUE token plus its single- and
    (optionally) two-bit-flip neighbours — the ids most plausible under a
    coherent joint.  The exact bit-factorised log-likelihood of candidate id
    ``v`` under the head is ``log p(v) = Σ_b log σ(s_b(v) · z_b)`` with
    ``s_b(v) = +1`` if bit ``b`` of ``v`` is set else ``-1`` — the same scoring
    :func:`model.bit_logits_to_token_logits` uses.  We softmax over the
    candidates (with temperature) and sample one.  This tests whether a sample
    from the JOINT (restricted) distribution is more coherent than the
    independent-bit sample — i.e. whether bit-independence is the limitation.

    The candidate set is PER-CELL (true id XOR a fixed offset table), so the
    log-likelihood is computed directly per-cell-per-candidate (no quadratic
    candidate×cell matrix).
    """
    from imas_ambix.camdyn.model import LFQ_BITS

    z = np.asarray(bit_logits, dtype=np.float64)  # (F,H,W,bits)
    tgt = np.asarray(true_tokens, dtype=np.int64)
    fhw = tgt.shape  # (F,H,W)
    nbits = z.shape[-1]

    # candidate offsets: 0 (true id) + every single-bit flip (+ optional pairs)
    single = [1 << b for b in range(nbits)]
    offsets = [0, *single]
    if n_neighbours >= 2:
        # a bounded set of two-bit flips (adjacent bit pairs) keeps K small
        offsets += [(1 << b) | (1 << (b + 1)) for b in range(nbits - 1)]
    offsets = np.asarray(sorted(set(offsets)), dtype=np.int64)  # (K,)
    k = offsets.shape[0]

    # per-cell candidate ids = true_id XOR offset (clamped to the 18-bit vocab)
    cand = (tgt[..., None] ^ offsets[None, None, None, :]) & (
        (1 << LFQ_BITS) - 1
    )  # (F,H,W,K)

    # signed-bit table per candidate: (F,H,W,K,bits) ∈ {-1,+1}
    cand_bits = ((cand[..., None] >> np.arange(nbits)) & 1).astype(np.float64)
    signs = 2.0 * cand_bits - 1.0  # (F,H,W,K,bits)
    # log σ(s·z) = -softplus(-s·z); sum over bits → log-likelihood per candidate
    signed = signs * z[..., None, :]  # broadcast z over the K axis
    log_sig = -np.logaddexp(0.0, -signed)
    scores = log_sig.sum(axis=-1)  # (F,H,W,K)

    scores /= max(temperature, 1e-6)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)  # (F,H,W,K)
    # inverse-CDF categorical sample over K candidates per cell
    cdf = np.cumsum(p, axis=-1)
    u = rng.random(fhw + (1,))
    choice = (u > cdf).sum(axis=-1)  # index in [0,K)
    choice = np.clip(choice, 0, k - 1)
    out = np.take_along_axis(cand, choice[..., None], axis=-1)[..., 0]
    return out.astype(np.int64)


# ---------------------------------------------------------------------------
# Predictive entropy — hedging vs collapsed
# ---------------------------------------------------------------------------


def per_bit_entropy(bit_logits: np.ndarray) -> np.ndarray:
    """Per-cell predictive entropy (nats), summed over the 18 independent bits.

    Each bit is a Bernoulli with ``p = σ(z_b)``; its entropy is
    ``H = -p log p - (1-p) log(1-p)``.  The cell entropy is the sum over bits
    (bit-independence → joint entropy = Σ bit entropies).  High entropy ⇒ the
    head is HEDGING (uncertain over the exact token, e.g. filament position);
    low entropy ⇒ the head is CONFIDENT (collapsed onto one id).  Returns
    ``(F,H,W)`` nats.
    """
    z = np.asarray(bit_logits, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    h = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))  # (F,H,W,bits)
    return h.sum(axis=-1)


# ---------------------------------------------------------------------------
# Image-space structure metrics (GT vs each decode)
# ---------------------------------------------------------------------------


def _to_gray256(img: np.ndarray) -> np.ndarray:
    """A frame → float64 grayscale 256² (like-for-like image-space metrics)."""
    from PIL import Image

    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape[:2] != (256, 256):
        a = np.asarray(
            Image.fromarray(a.astype(np.uint8)).resize((256, 256), Image.BILINEAR)
        )
    return a.astype(np.float64)


def radial_hf_spectrum(
    img: np.ndarray, *, hf_frac: float = 0.5
) -> tuple[np.ndarray, float]:
    """Radially-averaged 2-D power spectrum + integrated high-frequency power.

    Windowed 2-D FFT (Hann window to suppress edge leakage), power = |F|²,
    radially binned by spatial frequency.  Returns ``(radial_power, hf_power)``
    where ``hf_power`` is the power integrated over the upper ``hf_frac`` of the
    radial-frequency band — the fine-structure / striation content MAP is
    suspected to suppress.  The image is contrast-normalised first so the
    spectrum is about STRUCTURE, not absolute brightness.
    """
    a = np.asarray(img, dtype=np.float64)
    a = a - a.mean()
    sd = a.std() or 1.0
    a = a / sd
    h, w = a.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    aw = a * (wy * wx)
    f = np.fft.fftshift(np.fft.fft2(aw))
    power = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    nbins = min(cy, cx)
    radial = np.zeros(nbins, dtype=np.float64)
    for k in range(nbins):
        sel = r == k
        if sel.any():
            radial[k] = float(power[sel].mean())
    cut = int(round((1.0 - hf_frac) * nbins))
    hf_power = float(radial[cut:].sum())
    return radial, hf_power


def total_variation(img: np.ndarray) -> float:
    """Mean anisotropic total variation — texture / sharpness density."""
    a = np.asarray(img, dtype=np.float64)
    dx = np.abs(np.diff(a, axis=1))
    dy = np.abs(np.diff(a, axis=0))
    return float(dx.mean() + dy.mean())


def edge_rms_contrast(img: np.ndarray, *, n_rows: int = 96) -> float:
    """RMS of the high-passed lower-edge band — edge fluctuation contrast.

    Subtracts a smoothed copy of the lower ``n_rows`` of the 256² frame and
    reports the RMS of the residual: the fine edge-striation fluctuation
    amplitude.  MAP-blur lowers it; restored texture raises it toward GT.
    """
    from PIL import Image, ImageFilter

    a = np.asarray(img, dtype=np.float64)
    band = a[-n_rows:, :]
    bmax = float(band.max()) or 1.0
    u8 = np.clip(band / bmax * 255.0, 0, 255).astype(np.uint8)
    blur = np.asarray(
        Image.fromarray(u8).filter(ImageFilter.BoxBlur(4)), dtype=np.float64
    )
    hp = u8.astype(np.float64) - blur
    return float(np.sqrt(np.mean(hp**2)))


def structure_metrics(gray256: np.ndarray) -> dict:
    """All image-space structure numbers for one 256² grayscale frame."""
    radial, hf = radial_hf_spectrum(gray256)
    return {
        "hf_power": hf,
        "total_power": float(radial.sum()),
        "total_variation": total_variation(gray256),
        "edge_rms_contrast": edge_rms_contrast(gray256),
        "_radial": radial,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

ROW_ORDER = ("gt", "map", "samp0.7", "samp1.0", "joint")
ROW_LABEL = {
    "gt": "ground truth",
    "map": "MAP decode (current)",
    "samp0.7": "per-bit sample T=0.7",
    "samp1.0": "per-bit sample T=1.0",
    "joint": "joint restricted-vocab sample",
}


def render_figure(
    win: rd.DemoWindow,
    decoded: dict,
    raw_frames: np.ndarray | None,
    frontier: int,
    metrics_by_frame: dict,
    out_path: Path,
) -> None:
    """Rows = GT | MAP | sample T=0.7 | sample T=1.0 | joint, columns = time.

    Only POST-FRONTIER (predicted) columns are shown — that is where the edge
    filaments are genuinely forecast.  Each column shares the GT frame's robust
    display limits across all rows so structure is legible and reconstruction
    over/under-shoot stays honest (the ``reconstruction_demo`` convention).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ft = np.asarray(win.frame_time, dtype=float)
    n_frames = ft.shape[0]
    post = list(range(frontier, n_frames))
    n_cols = min(6, len(post))
    cols = sorted(set(np.linspace(post[0], post[-1], n_cols).round().astype(int)))
    n_cols = len(cols)

    rows = [r for r in ROW_ORDER if r in decoded]
    n_rows = len(rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.9 * n_cols + 1.9, 1.6 * n_rows + 1.3),
        squeeze=False,
        constrained_layout=True,
    )

    for ci, fi in enumerate(cols):
        # per-column display limits from the GT frame (raw if available)
        if raw_frames is not None and fi < raw_frames.shape[0]:
            gt_disp = raw_frames[fi].astype(np.float64)
        else:
            gt_disp = rd._to_aspect(decoded["gt"][fi]).astype(np.float64)
        vmin, vmax = rd.display_limits(gt_disp)
        dt_ms = (ft[fi] - ft[frontier]) * 1e3
        for ri, role in enumerate(rows):
            ax = axes[ri][ci]
            if role == "gt" and raw_frames is not None and fi < raw_frames.shape[0]:
                img = raw_frames[fi].astype(np.float64)
            else:
                img = rd._to_aspect(decoded[role][fi]).astype(np.float64)
            ax.imshow(
                img, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"+{dt_ms:.1f} ms", fontsize=9)
            if ci == 0:
                ax.set_ylabel(ROW_LABEL[role], fontsize=8)

    # caption with the headline HF-power retention numbers
    def _retain(role):
        g = metrics_by_frame["gt"]["hf_power"]
        return metrics_by_frame[role]["hf_power"] / g if g > 0 else float("nan")

    bits = [
        f"shot {win.shot_id}",
        f"forecast {ft[frontier] * 1e3:.0f}-{ft[-1] * 1e3:.0f} ms "
        f"(frontier @ {ft[frontier] * 1e3:.0f} ms)",
        f"HF-power vs GT: MAP {_retain('map'):.2f}",
    ]
    if "samp1.0" in metrics_by_frame:
        bits[-1] += f", sampT1.0 {_retain('samp1.0'):.2f}"
    if "joint" in metrics_by_frame:
        bits[-1] += f", joint {_retain('joint'):.2f}"
    caption = (
        "Edge-filament decode: MAP vs sampled — camera-dynamics forecast.  "
        + " | ".join(bits)
        + ".\nAll rows decode the SAME per-bit logits z_b through the frozen "
        "Open-MAGVIT2 tokenizer; only the token-grid decode rule differs "
        "(MAP = z_b>0; sample = bit~σ(z_b/T)).  Columns = predicted frames; "
        "per-column GT 1/99-pct display norm."
    )
    fig.suptitle(caption, fontsize=10)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[structfid] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _decode_grids(grids_by_role: dict, work_dir: Path, device: str) -> dict:
    """Decode a dict of role→(F,16,16) token grids → role→(F,256,256,3) images.

    One batched OMAG2 decode pass over all roles (reuses the BundleBuilder
    token→image handoff).  Token grids are GLOBAL ids; the decode subprocess
    subtracts the registry offset.
    """
    bb = mvr.BundleBuilder()
    wi = bb.add_window({"frame_time": []})
    for role, grid in grids_by_role.items():
        bb.add_grid(grid, wi, "_window", role)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"
    bb.save(token_bundle)
    rd.run_decode_subprocess(token_bundle, image_bundle, device)
    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)  # (N,F,256,256,3)
    index = json.loads(str(data["index"]))
    slot = {e["role"]: e["slot"] for e in index}
    return {role: images[slot[role]] for role in grids_by_role}


def run(out_path: Path = DEFAULT_FIGURE, *, device: str = "cuda") -> dict:
    """Full probe: select → forward → MAP/sample/joint decode → metrics → figure."""
    import torch

    from imas_ambix.camdyn.arm_compare import _load_arm

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("[structfid] device = %s", dev)

    sel = select_structure_window()
    if sel is None:
        raise RuntimeError("no flat-top edge window could be selected")
    win = sel.window

    logger.info("[structfid] loading dynamics arm on %s", dev)
    model, _cfg, cond_stats = _load_arm(rd.DYNAMICS_CKPT, torch, dev)
    try:
        visible, bit_logits = forward_bit_logits(
            model, torch, dev, win, cond_stats, SCENARIO, FRONTIER
        )
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    rng = np.random.default_rng(SEED)
    grids = {
        "gt": win.true_tokens,
        "map": decode_map(bit_logits),
        "samp0.7": decode_per_bit_sample(bit_logits, temperature=0.7, rng=rng),
        "samp1.0": decode_per_bit_sample(bit_logits, temperature=1.0, rng=rng),
    }
    # joint restricted-vocab sample (cheap — K candidates per cell)
    try:
        grids["joint"] = decode_joint_sample(
            bit_logits, win.true_tokens, temperature=1.0, rng=rng
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[structfid] joint sample skipped: %s", exc)

    work_dir = Path(
        tempfile.mkdtemp(prefix="structfid-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    dev_str = "cuda" if dev.type == "cuda" else "cpu"
    decoded = _decode_grids(grids, work_dir, dev_str)

    raw_frames = rd.load_raw_frames(win.shot_id, win.start, win.true_tokens.shape[0])

    # ---- per-decode structure metrics, averaged over PREDICTED frames -------
    post = list(range(FRONTIER, win.true_tokens.shape[0]))
    metrics_by_frame: dict[str, dict] = {}
    radial_by_role: dict[str, np.ndarray] = {}
    for role, imgs in decoded.items():
        accum = {
            "hf_power": [],
            "total_power": [],
            "total_variation": [],
            "edge_rms_contrast": [],
        }
        radials = []
        for fi in post:
            g = _to_gray256(imgs[fi])
            m = structure_metrics(g)
            for k in accum:
                accum[k].append(m[k])
            radials.append(m["_radial"])
        metrics_by_frame[role] = {k: float(np.mean(v)) for k, v in accum.items()}
        radial_by_role[role] = np.mean(np.stack(radials), axis=0)

    # ---- per-bit predictive entropy: edge region vs bulk --------------------
    ent = per_bit_entropy(bit_logits)  # (F,H,W) nats
    er = _edge_rows()
    post_arr = np.asarray(post, dtype=int)
    ent_post = ent[post_arr]  # only predicted frames
    edge_ent = float(ent_post[:, er, :].mean())
    bulk_rows = slice(0, mv.GRID_H - 4)
    bulk_ent = float(ent_post[:, bulk_rows, :].mean())
    from imas_ambix.camdyn.model import LFQ_BITS

    max_cell_ent = LFQ_BITS * np.log(2.0)

    # ---- verdict ------------------------------------------------------------
    gt_hf = metrics_by_frame["gt"]["hf_power"]
    map_ret = metrics_by_frame["map"]["hf_power"] / gt_hf if gt_hf > 0 else float("nan")
    s10_ret = (
        metrics_by_frame["samp1.0"]["hf_power"] / gt_hf if gt_hf > 0 else float("nan")
    )
    joint_ret = (
        metrics_by_frame["joint"]["hf_power"] / gt_hf
        if ("joint" in metrics_by_frame and gt_hf > 0)
        else float("nan")
    )
    map_suppresses = map_ret < 0.7
    sampling_restores = np.isfinite(s10_ret) and s10_ret > map_ret * 1.25
    hedging = edge_ent > 0.25 * max_cell_ent  # > a quarter of the max cell entropy

    if map_suppresses and sampling_restores and hedging:
        verdict = (
            "decode-mode artifact (sampling restores structure; head is "
            "hedging) — cheap no-retrain fix"
        )
    elif map_suppresses and not sampling_restores and not hedging:
        verdict = (
            "head/objective failure (low entropy, blur survives sampling) — "
            "needs richer head / structure-aware loss"
        )
    elif map_suppresses and sampling_restores and not hedging:
        verdict = (
            "partial: sampling restores HF power but entropy is low — "
            "sample-decode helps yet head expressiveness is the deeper limit"
        )
    elif not map_suppresses:
        verdict = (
            "MAP does NOT suppress HF power on this window — averaging "
            "hypothesis not reproduced here"
        )
    else:
        verdict = "mixed: see numbers (sampling does not clearly restore structure)"

    summary = {
        "shot_id": int(win.shot_id),
        "window_start": int(win.start),
        "window_ms": [float(win.frame_time[0] * 1e3), float(win.frame_time[-1] * 1e3)],
        "frontier_frame": int(FRONTIER),
        "frontier_ms": float(win.frame_time[FRONTIER] * 1e3),
        "scenario": SCENARIO,
        "persistent_edge_power": float(sel.edge_power),
        "metrics": {
            role: {k: v for k, v in m.items()} for role, m in metrics_by_frame.items()
        },
        "hf_power_retention_vs_gt": {
            "map": float(map_ret),
            "samp0.7": float(metrics_by_frame["samp0.7"]["hf_power"] / gt_hf)
            if gt_hf > 0
            else float("nan"),
            "samp1.0": float(s10_ret),
            "joint": float(joint_ret),
        },
        "per_bit_entropy_nats": {
            "edge_region": edge_ent,
            "bulk_region": bulk_ent,
            "max_cell_entropy": float(max_cell_ent),
            "edge_frac_of_max": float(edge_ent / max_cell_ent),
        },
        "verdict": verdict,
        "figure": str(out_path),
    }

    render_figure(win, decoded, raw_frames, FRONTIER, metrics_by_frame, out_path)

    # ---- print every number -------------------------------------------------
    ft = np.asarray(win.frame_time, dtype=float)
    print("\n" + "=" * 76)
    print("STRUCTURE FIDELITY — SAMPLED vs MAP DECODE OF EDGE FILAMENTS")
    print("=" * 76)
    print(f"shot_id            : {win.shot_id}")
    print(
        f"window             : {ft[0] * 1e3:.1f} - {ft[-1] * 1e3:.1f} ms "
        f"({(ft[-1] - ft[0]) * 1e3:.1f} ms, {len(ft)} frames)"
    )
    print(
        f"scenario           : {SCENARIO} "
        f"(frontier @ f{FRONTIER}, t={ft[FRONTIER] * 1e3:.1f} ms)"
    )
    print(f"persistent edge pwr: {sel.edge_power:.3e}  (selection score)")
    print("-" * 76)
    print("IMAGE-SPACE STRUCTURE (mean over predicted frames):")
    print(f"  {'role':<22}{'HF power':>12}{'HF/GT':>8}{'TV':>9}{'edgeRMS':>9}")
    for role in ROW_ORDER:
        if role not in metrics_by_frame:
            continue
        m = metrics_by_frame[role]
        ret = m["hf_power"] / gt_hf if gt_hf > 0 else float("nan")
        print(
            f"  {ROW_LABEL[role]:<22}{m['hf_power']:>12.3e}{ret:>8.2f}"
            f"{m['total_variation']:>9.3f}{m['edge_rms_contrast']:>9.3f}"
        )
    print("-" * 76)
    print("PER-BIT PREDICTIVE ENTROPY (nats/cell, predicted frames):")
    edge_pct = 100 * edge_ent / max_cell_ent
    bulk_pct = 100 * bulk_ent / max_cell_ent
    print(
        f"  edge region        : {edge_ent:.3f}  "
        f"({edge_pct:.0f}% of max {max_cell_ent:.2f})"
    )
    print(f"  bulk region        : {bulk_ent:.3f}  ({bulk_pct:.0f}% of max)")
    print(f"  edge/bulk ratio    : {edge_ent / max(bulk_ent, 1e-9):.2f}")
    print("-" * 76)
    print(
        f"MAP suppresses HF GT has : {map_suppresses} "
        f"(MAP retains {map_ret:.2f} of GT HF power)"
    )
    print(
        f"sampling restores HF     : {sampling_restores} "
        f"(sampT1.0 retains {s10_ret:.2f})"
    )
    print(
        f"head is hedging          : {hedging} "
        f"(edge entropy {edge_pct:.0f}% of max)"
    )
    print("-" * 76)
    print(f"VERDICT            : {verdict}")
    print(f"figure             : {out_path}")
    print("=" * 76 + "\n")
    print("SUMMARY_JSON " + json.dumps(summary))
    return summary


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_FIGURE))
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(Path(args.out), device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
