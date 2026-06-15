"""Does sampled decode restore the CORRECT persistent edge filaments, or only texture?

A hardened diagnostic probe that separates a DECODE-MODE artifact from a
HEAD/OBJECTIVE failure for the camera-dynamics world model — and, critically,
measures whether sampling restores structure in the RIGHT PLACE and with the
RIGHT TEMPORAL COHERENCE, not merely the right texture statistics.

Mechanism under test
---------------------
The model head emits 18 INDEPENDENT bit-logits (a bitwise-factorised LFQ
likelihood).  Both scoring and rendering currently decode each cell by per-bit
MAP (``pred_id = Σ_b (z_b > 0) << b``).  Under genuine aleatoric uncertainty
about the exact position of a bright edge/SOL filament, the per-bit MAP
collapses to a smeared "mean" token grid — so a predicted region blurs out the
striations the ground-truth frames show.  Hypothesis: the per-bit distribution
still CONTAINS the structure; the mode destroys it; SAMPLING recovers it.

Why the FIRST version of this probe was gameable (adversarial critique)
-----------------------------------------------------------------------
The original verdict rode on a radially-averaged high-frequency power spectrum.
That number is:

* **phase-blind** — it ignores WHERE the structure is, so spectrum-matched
  coloured noise scores like real filaments;
* **time-collapsed** — it averages frames, so a model with the right texture
  but FROZEN (non-evolving) filaments scores perfectly even though it does not
  do its dynamics job;
* scored against the DECODED GT TOKENS, not the raw camera frames, so it could
  not see decoder-introduced blur and flattered every decode;
* the headline "joint" decode that matched GT best was an ORACLE (its candidate
  ids were ``true_tokens XOR offsets``) — unshippable and upward-biased.

Hardened metric (this version)
-------------------------------
The CORRECTNESS verdict is now driven by two truth-grounded terms, both scored
against the RAW level-1 camera frames (resized to 256², the camera truth — see
:func:`reconstruction_demo.load_raw_frames`), NOT the decoded GT tokens:

* **location** — masked edge-band SSIM of each predicted frame vs the GT frame
  (structural similarity in the lower-edge / divertor band where the filaments
  live).  A decode whose bright structure is in the wrong place scores low even
  if its texture statistics match.
* **temporal** — frame-to-frame coherence: the edge-band SSIM of the
  INTER-FRAME DIFFERENCE (pred Δframe vs GT Δframe).  A frozen-filament decode
  has a near-zero Δframe and fails this even with perfect per-frame texture, so
  a dynamics model is only eligible to win on its actual job.

The radial HF-power spectrum is retained as a SECONDARY reporting number only
(it is the texture-magnitude term, kept for continuity, never the verdict).

Decisive falsification control: a SPECTRUM-MATCHED COLOURED-NOISE role is
injected — noise whitened to the GT radial spectrum.  It MUST pass the
(gameable) HF-power term and FAIL the location + temporal correctness terms.
If it passes the correctness terms, the metric is still gameable and we report
that honestly.

Decode roles scored (all from the SAME forward pass, both arms)
---------------------------------------------------------------
* **map** — the current decode (deterministic mode), the scoring default.
* **bernoulli-T** — truth-free per-bit Bernoulli sample
  (:func:`token_sampling.bernoulli_sample`) at the temperature that maximises
  the correctness score.
* **beam** — truth-free BIT-BEAM joint sampler
  (:func:`token_sampling.bit_beam_sample`) — coherent (real codebook ids ranked
  by the head's factorised likelihood) without truth access; the SHIPPABLE
  coherent decode.
* **persistence** — the last observed frame frozen across the forecast window,
  decoded the SAME way.  If persistence's correctness score ≈ the model's, the
  model adds no forecast structure.
* **joint-oracle** — the restricted-vocab joint sample whose candidates are
  ``true_tokens XOR offsets``.  PROBE-ONLY UPPER BOUND (labelled oracle-biased
  in the JSON); never shipped.
* **coloured-noise** — the falsification control.

Each role reports BOTH the correctness score AND masked top-1 / NLL on the SAME
decoded token grid, so the accuracy-vs-sharpness Pareto (physical-mean MAP vs
plausible-sample) is explicit, not hidden.

Seed-averaging + arms + windows
--------------------------------
The stochastic decodes are averaged over ``N_SEEDS`` (≥ 8) seeds, with a
bootstrap CI (:func:`metrics.bootstrap_ci`) over the per-seed × per-frame
scores.  Both arms (``cap_v1_baseline`` and ``cap_v1_dynamics``) are run across
all three :data:`FLATTOP_SHOTS` windows.

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
from imas_ambix.camdyn import token_sampling as ts
from imas_ambix.camdyn.metrics import bootstrap_ci

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Output figures (committed).
DEFAULT_FIGURE = Path("docs/figures/camera-dynamics-wm/fig-cdw-filament-sampled.png")
DEFAULT_JSON = Path("imas_ambix/camdyn/artifacts/structure_fidelity.json")

#: Held-out flat-top shots scanned for the strongest persistent edge structure.
FLATTOP_SHOTS = (24446, 24065, 23937)

#: Forecast scenario: the model sees full frames up to FRONTIER and PREDICTS
#: the rest — the post-frontier edge filaments are genuinely forecast.
SCENARIO = "frontier"
FRONTIER = 8
N_FRAMES = 16
SPAN_MS = 6.0

#: Sampling temperatures swept for the correctness-maximising decode.
TEMPERATURES = (0.6, 0.7, 0.8, 0.9, 1.0)

#: MaskGIT iterative-decode config (the cross-cell coherence lever).
MASKGIT_ROUNDS = 8
MASKGIT_TOP_K = 6  # least-confident bits resampled per committed cell
MASKGIT_CONF_NOISE = 1.0  # annealed Gumbel noise on the commit ordering

#: Seeds the stochastic decodes are averaged over (≥ 8 per critique).
N_SEEDS = 8
SEED0 = 12345

#: The two matched arms.
ARMS = {"dynamics": rd.DYNAMICS_CKPT, "baseline": rd.BASELINE_CKPT}


# ---------------------------------------------------------------------------
# Window selection — strongest persistent edge structure on a flat-top shot
# ---------------------------------------------------------------------------


def _edge_rows(grid_h: int = mv.GRID_H, n: int = 4) -> slice:
    """Token-grid rows that map to the lower/divertor edge of the rbb frame."""
    return slice(grid_h - n, grid_h)


def _persistent_edge_power(raw_frames: np.ndarray, frontier: int) -> float:
    """Persistent high-spatial-frequency power in the POST-FRONTIER edge region.

    For each post-frontier frame we high-pass the lower-edge band (subtract a
    box-blurred copy) and take the residual variance — the fine edge-striation
    power.  The window score is MIN across post-frontier frames (structure must
    persist) times √mean (brighter preferred).
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


def select_structure_windows(
    *,
    shots=FLATTOP_SHOTS,
    n_frames: int = N_FRAMES,
    frontier: int = FRONTIER,
    span_ms: float = SPAN_MS,
    wide_factor: int = 16,
) -> list[StructureWindow]:
    """Pick the strongest PERSISTENT post-frontier edge window for EACH shot.

    Returns one window per shot (the critique requires running on all 3
    FLATTOP_SHOTS), each the brightest persistent-edge window of that shot.
    """
    from imas_ambix.camdyn.dataset import (
        FrameTokenDataset,
        FrameWindowConfig,
        discover_token_shots,
    )

    wide_n = n_frames * wide_factor
    out: list[StructureWindow] = []
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
        best: StructureWindow | None = None
        best_score = 0.0
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
            if abs(raw.shape[1] - mv.ORIGINAL_HW[0]) > 24:
                continue
            score = _persistent_edge_power(raw.astype(np.float64), frontier)
            if score > best_score:
                best_score = score
                best = StructureWindow(window=dwin, edge_power=float(score))
        if best is not None:
            w = best.window
            ft = np.asarray(w.frame_time, dtype=float)
            logger.info(
                "[structfid] SELECTED shot %d start %d %.1f-%.1f ms "
                "(frontier@f%d t=%.1f ms) edge-power=%.3e",
                w.shot_id,
                w.start,
                ft[0] * 1e3,
                ft[-1] * 1e3,
                frontier,
                ft[frontier] * 1e3,
                best.edge_power,
            )
            out.append(best)
    if not out:
        logger.warning("[structfid] no flat-top edge window found on any shot")
    return out


def _decimate(
    base: rd.DemoWindow, *, span_ms: float, n_frames: int
) -> rd.DemoWindow | None:
    """Decimate a wide native window to ``n_frames`` spanning ``span_ms``."""
    ft0 = np.asarray(base.frame_time, dtype=np.float64)
    if ft0.size < 2:
        return None
    dt_med = float(np.median(np.diff(ft0)))
    idx = mv.decimated_indices(ft0.shape[0], n_frames, dt_med, span_ms)
    if idx.size < n_frames:
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


def _conditioning_tensors(torch, device, win: rd.DemoWindow, cond_stats):
    """Frozen per-window conditioning tensors ``(cv_t, cm_t, dt_t)`` (batch 1).

    The conditioning (z-scored actuator values, missing-flags, Δt) does NOT
    depend on the token grid or visibility, so it is built ONCE and reused
    across the MaskGIT forward rounds (re-running ``load_conditioning`` every
    round would dominate the wall-clock and is pure waste).
    """
    from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS, load_conditioning
    from imas_ambix.camdyn.dataset import discover_token_shots

    specs = discover_token_shots(shot_ids=[win.shot_id], read_n_frames=False)
    level1_path = specs[0].level1_path if specs else None
    cond = load_conditioning(
        level1_path, win.frame_time, win.shot_id, channels=CONDITIONING_CHANNELS
    )
    cv = rd._zscore(cond.values, cond_stats)[None]
    cm = cond.missing[None].astype(np.float32)
    dt = win.dt[None].astype(np.float32)
    cv_t = torch.from_numpy(cv).to(device)
    cm_t = torch.from_numpy(cm).to(device)
    dt_t = torch.from_numpy(dt).to(device)
    return cv_t, cm_t, dt_t


def _run_forward(model, torch, device, tokens, visible, cv_t, cm_t, dt_t) -> np.ndarray:
    """Single model forward on the given token/visibility state → bit-logits.

    ``tokens`` / ``visible`` are ``(F,H,W)`` numpy arrays (one window); returns
    ``(F,H,W,bits)`` float numpy logits.  Shared by the single-shot probe and
    the iterative MaskGIT closure so every role decodes through the IDENTICAL
    forward.
    """
    tokens_t = torch.from_numpy(np.asarray(tokens, dtype=np.int64)[None]).to(device)
    vis_t = torch.from_numpy(np.asarray(visible, dtype=bool)[None]).to(device)
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model.module(tokens_t, vis_t, cv_t, cm_t, dt_t)
        return logits.float().cpu().numpy()[0]  # (F,H,W,bits)


def make_forward_fn(model, torch, device, win: rd.DemoWindow, cond_stats):
    """Build the ``(tokens, visible) -> bit_logits`` closure MaskGIT re-runs.

    Closes over the frozen conditioning so each MaskGIT round only pays the
    transformer forward, not the conditioning load.  This is the EXACT
    conditioning mechanism the head was trained with — committed cells fold
    into the visible context via the model's own ``[MASK]``-embedding path.
    """
    cv_t, cm_t, dt_t = _conditioning_tensors(torch, device, win, cond_stats)

    def forward_fn(tokens: np.ndarray, visible: np.ndarray) -> np.ndarray:
        return _run_forward(model, torch, device, tokens, visible, cv_t, cm_t, dt_t)

    return forward_fn


def forward_bit_logits(
    model, torch, device, win: rd.DemoWindow, cond_stats, scenario: str, frontier: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run one arm forward → ``(visible, bit_logits)`` ``(F,H,W,bits)``.

    The RAW per-bit logits before the ``>0`` threshold, so MAP / bernoulli /
    beam / oracle decodes all derive from the SAME forward pass.
    """
    n_frames = win.true_tokens.shape[0]
    visible = rd.scenario_mask(scenario, n_frames, frontier)
    cv_t, cm_t, dt_t = _conditioning_tensors(torch, device, win, cond_stats)
    bl = _run_forward(
        model, torch, device, win.true_tokens, visible, cv_t, cm_t, dt_t
    )
    return visible, bl


# ---------------------------------------------------------------------------
# Decode strategies — bit logits → token grid
# ---------------------------------------------------------------------------


def decode_map(bit_logits: np.ndarray) -> np.ndarray:
    """MAP token grid (the current decode)."""
    return ts.map_decode(bit_logits)


def decode_oracle_joint(
    bit_logits: np.ndarray,
    true_tokens: np.ndarray,
    *,
    temperature: float,
    rng: np.random.Generator,
    n_neighbours: int = 2,
) -> np.ndarray:
    """ORACLE restricted-vocab joint sample — candidates = true id XOR offsets.

    This SEES the truth (its candidate set is the true token plus its single-
    and two-bit-flip neighbours) so it can only sample NEAR the truth.  It is an
    UNSHIPPABLE upper bound, retained as a probe-only ceiling and labelled
    ``oracle_biased`` in the JSON.  Never wired into a renderer.
    """
    from imas_ambix.camdyn.model import LFQ_BITS

    z = np.asarray(bit_logits, dtype=np.float64)
    tgt = np.asarray(true_tokens, dtype=np.int64)
    fhw = tgt.shape
    nbits = z.shape[-1]

    single = [1 << b for b in range(nbits)]
    offsets = [0, *single]
    if n_neighbours >= 2:
        offsets += [(1 << b) | (1 << (b + 1)) for b in range(nbits - 1)]
    offsets = np.asarray(sorted(set(offsets)), dtype=np.int64)
    k = offsets.shape[0]

    cand = (tgt[..., None] ^ offsets[None, None, None, :]) & ((1 << LFQ_BITS) - 1)
    cand_bits = ((cand[..., None] >> np.arange(nbits)) & 1).astype(np.float64)
    signs = 2.0 * cand_bits - 1.0
    signed = signs * z[..., None, :]
    log_sig = -np.logaddexp(0.0, -signed)
    scores = log_sig.sum(axis=-1)
    scores /= max(temperature, 1e-6)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(p, axis=-1)
    u = rng.random(fhw + (1,))
    choice = np.clip((u > cdf).sum(axis=-1), 0, k - 1)
    return np.take_along_axis(cand, choice[..., None], axis=-1)[..., 0].astype(np.int64)


# ---------------------------------------------------------------------------
# Predictive entropy + the recalibrated hedging threshold
# ---------------------------------------------------------------------------


def per_bit_entropy(bit_logits: np.ndarray) -> np.ndarray:
    """Per-cell predictive entropy (nats), summed over the 18 independent bits."""
    z = np.asarray(bit_logits, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    h = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
    return h.sum(axis=-1)


def calibrate_hedging_threshold(
    entropy: np.ndarray,
    raw_frames: np.ndarray,
    frontier: int,
) -> dict:
    """Recalibrate "hedging" against a held-out CELL-LEVEL entropy distribution.

    The original threshold (0.25 × 18·ln2) was a uniform-over-2^18 ceiling,
    biased toward "decode fix".  Instead we split the post-frontier cells into
    KNOWN-STATIC-BACKGROUND vs KNOWN-FILAMENT/EDGE using the RAW GT frames:

    * a cell is FILAMENT/EDGE if its GT raw-frame patch is in the lower-edge
      band AND its local high-frequency (striation) energy is in the top
      tercile of post-frontier edge cells;
    * a cell is STATIC-BACKGROUND if its GT raw patch barely changes across the
      post-frontier window (bottom tercile of per-cell temporal variance) and
      is not in the edge band.

    "Hedging" then means the head's per-bit entropy on filament/edge cells is
    materially ABOVE its entropy on confidently-correct static-background cells
    — i.e. the head is more uncertain exactly where the filaments are.  The
    threshold is the static-background entropy mean + 1σ; ``hedging`` is True
    when the filament-cell entropy mean exceeds it.
    """
    gh, gw = mv.GRID_H, mv.GRID_W
    post = list(range(frontier, entropy.shape[0]))
    if raw_frames is None or raw_frames.shape[0] <= frontier:
        # cannot calibrate without GT frames — fall back to a neutral split
        return {
            "calibrated": False,
            "reason": "no raw frames",
            "hedging": False,
        }
    # resize GT raw frames to the token grid so each cell maps to a GT patch
    from PIL import Image

    gt = np.stack(
        [
            np.asarray(
                Image.fromarray(
                    np.clip(raw_frames[fi].astype(np.float64), 0, None).astype(np.uint8)
                    if raw_frames[fi].dtype != np.uint8
                    else raw_frames[fi]
                ).resize((gw, gh), Image.BILINEAR),
                dtype=np.float64,
            )
            for fi in post
        ]
    )  # (P, gh, gw)

    # per-cell temporal variance (static = low) over the post-frontier window
    temporal_var = gt.var(axis=0)  # (gh, gw)
    # per-cell edge-band membership + local HF energy (filament proxy)
    edge_band = np.zeros((gh, gw), dtype=bool)
    edge_band[gh - 4 :, :] = True
    # local HF energy: |cell - mean of 3x3 neighbourhood| averaged over frames
    hf = np.zeros((gh, gw), dtype=np.float64)
    for fi in range(gt.shape[0]):
        fr = gt[fi]
        pad = np.pad(fr, 1, mode="edge")
        local_mean = (
            pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] + fr
        ) / 5.0
        hf += np.abs(fr - local_mean)
    hf /= gt.shape[0]

    edge_hf = hf[edge_band]
    if edge_hf.size == 0:
        return {"calibrated": False, "reason": "empty edge band", "hedging": False}
    hf_hi = np.percentile(edge_hf, 66.0)
    var_lo = np.percentile(temporal_var, 33.0)

    filament = edge_band & (hf >= hf_hi)
    static_bg = (~edge_band) & (temporal_var <= var_lo)

    ent_post = entropy[np.asarray(post, dtype=int)]  # (P, gh, gw)
    ent_mean_cell = ent_post.mean(axis=0)  # (gh, gw) per-cell mean entropy

    fil_ent = ent_mean_cell[filament]
    bg_ent = ent_mean_cell[static_bg]
    if fil_ent.size == 0 or bg_ent.size == 0:
        return {
            "calibrated": False,
            "reason": "empty filament or background cell set",
            "hedging": False,
        }
    bg_mean = float(bg_ent.mean())
    bg_sd = float(bg_ent.std()) or 1e-6
    fil_mean = float(fil_ent.mean())
    threshold = bg_mean + bg_sd
    return {
        "calibrated": True,
        "n_filament_cells": int(filament.sum()),
        "n_static_bg_cells": int(static_bg.sum()),
        "filament_entropy_mean": fil_mean,
        "static_bg_entropy_mean": bg_mean,
        "static_bg_entropy_sd": bg_sd,
        "threshold": float(threshold),
        # head hedges where the filaments are: filament entropy clears the
        # confidently-correct-background mean + 1σ.
        "hedging": bool(fil_mean > threshold),
        "filament_over_background_ratio": float(fil_mean / max(bg_mean, 1e-9)),
    }


# ---------------------------------------------------------------------------
# Image-space metrics — TRUTH-GROUNDED location + temporal correctness
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


def _edge_band_256(n_rows: int = 96) -> slice:
    """Lower-edge / divertor band of a 256² frame (where the filaments live)."""
    return slice(256 - n_rows, 256)


def _uniform_box(x: np.ndarray, win: int) -> np.ndarray:
    """Separable uniform (box) mean filter via cumulative sums — pure numpy.

    Works directly on float arrays (PIL's BoxBlur rejects float "F" mode), so
    SSIM has no PIL/skimage dependency and runs anywhere numpy does.  Edges use
    reflect padding so the window count is consistent.
    """
    r = max(1, win) // 2
    if r == 0:
        return x.astype(np.float64)
    xp = np.pad(x.astype(np.float64), r, mode="reflect")
    # rows then cols via cumulative-sum sliding window
    cs = np.cumsum(xp, axis=0)
    cs = np.vstack([np.zeros((1, cs.shape[1])), cs])
    win_len = 2 * r + 1
    row = cs[win_len:, :] - cs[:-win_len, :]
    cs2 = np.cumsum(row, axis=1)
    cs2 = np.hstack([np.zeros((cs2.shape[0], 1)), cs2])
    out = cs2[:, win_len:] - cs2[:, :-win_len]
    return out / (win_len * win_len)


def ssim_map(a: np.ndarray, b: np.ndarray, *, win: int = 7) -> float:
    """Mean SSIM between two float images (Wang et al.), windowed by a box mean.

    A self-contained SSIM (no skimage dependency): local means / variances /
    covariance via a numpy uniform box filter, then the standard SSIM formula
    with the usual stabilisers.  Inputs are scaled to a common dynamic range
    first so SSIM compares STRUCTURE, not absolute brightness.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # common dynamic range (per-pair robust scaling) so SSIM is about structure
    def _norm(x):
        lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    a = _norm(a)
    b = _norm(b)
    L = 1.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    def _box(x):
        return _uniform_box(x, win)

    mu_a = _box(a)
    mu_b = _box(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = _box(a * a) - mu_a2
    var_b = _box(b * b) - mu_b2
    cov_ab = _box(a * b) - mu_ab
    ssim = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return float(np.clip(ssim, -1.0, 1.0).mean())


def radial_hf_spectrum(
    img: np.ndarray, *, hf_frac: float = 0.5
) -> tuple[np.ndarray, float]:
    """Radially-averaged 2-D power spectrum + integrated high-frequency power.

    SECONDARY reporting only — phase-blind and time-collapsed, so it is a
    texture-MAGNITUDE number, never the correctness verdict.
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


def correctness_scores(
    decoded_imgs: np.ndarray,
    gt_imgs: np.ndarray,
    post: list[int],
) -> dict:
    """LOCATION + TEMPORAL correctness of a decode vs RAW GT, + secondary HF.

    Parameters
    ----------
    decoded_imgs:
        ``(F,256,256,3)`` or ``(F,256,256)`` decoded frames (the role under test).
    gt_imgs:
        ``(F,256,256)`` RAW GT frames resized to 256² (camera truth).
    post:
        Post-frontier frame indices to score.

    Returns
    -------
    ``{location, temporal, hf_power, hf_ratio_self}`` where

    * ``location`` — mean masked edge-band SSIM(pred, GT) over post frames
      (HIGH = bright structure in the right place);
    * ``temporal`` — mean edge-band SSIM(Δpred, ΔGT) over consecutive post
      frames (HIGH = the predicted structure EVOLVES like GT — frozen filaments
      score ~0);
    * ``hf_power`` — secondary radial HF power (texture magnitude, mean over
      post frames).
    """
    eb = _edge_band_256()
    loc, temp, hf = [], [], []
    prev_pred = prev_gt = None
    for fi in post:
        pred = _to_gray256(decoded_imgs[fi])
        gt = _to_gray256(gt_imgs[fi]) if fi < gt_imgs.shape[0] else None
        if gt is None:
            continue
        # location: edge-band SSIM vs GT
        loc.append(ssim_map(pred[eb, :], gt[eb, :]))
        # secondary HF power on the predicted frame
        _, hfp = radial_hf_spectrum(pred)
        hf.append(hfp)
        # temporal: SSIM of the inter-frame difference vs GT's
        if prev_pred is not None:
            dp = pred - prev_pred
            dg = gt - prev_gt
            temp.append(ssim_map(dp[eb, :], dg[eb, :]))
        prev_pred, prev_gt = pred, gt
    return {
        "location": float(np.mean(loc)) if loc else float("nan"),
        "temporal": float(np.mean(temp)) if temp else float("nan"),
        "hf_power": float(np.mean(hf)) if hf else float("nan"),
        "_location_per_frame": np.asarray(loc, dtype=np.float64),
        "_temporal_per_frame": np.asarray(temp, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Coloured-noise falsification control
# ---------------------------------------------------------------------------


def coloured_noise_like(
    gt_imgs: np.ndarray, post: list[int], rng: np.random.Generator
) -> np.ndarray:
    """Spectrum-matched coloured noise: per-frame noise whitened to GT's spectrum.

    For each post frame, draw white Gaussian noise, take its FFT, and impose the
    GT frame's radial amplitude spectrum (so |Noise(f)| ≈ |GT(f)| but the PHASE
    is random).  The result has the SAME radial HF power as GT but its structure
    is in random places — the decisive control: it MUST pass the (gameable) HF
    term and FAIL the location + temporal correctness terms.
    """
    out = np.zeros((gt_imgs.shape[0], 256, 256), dtype=np.float64)
    for fi in post:
        gt = _to_gray256(gt_imgs[fi]) if fi < gt_imgs.shape[0] else None
        if gt is None:
            continue
        G = np.fft.fft2(gt - gt.mean())
        amp = np.abs(G)
        white = rng.standard_normal(gt.shape)
        W = np.fft.fft2(white)
        phase = np.exp(1j * np.angle(W))
        coloured = np.fft.ifft2(amp * phase).real
        # match GT brightness range so the decode-display path treats it like a frame
        c = coloured
        c = (c - c.min()) / (np.ptp(c) or 1.0)
        out[fi] = np.clip(c * 255.0, 0, 255)
    return out


# ---------------------------------------------------------------------------
# Accuracy term — masked top-1 / NLL on the SAME decoded token grid
# ---------------------------------------------------------------------------


def token_accuracy(
    decoded_tokens: np.ndarray,
    true_tokens: np.ndarray,
    bit_logits: np.ndarray,
    post: list[int],
) -> dict:
    """Masked top-1 + bitwise NLL of a DECODED token grid over post-frontier cells.

    The accuracy-vs-sharpness Pareto axis: the SAME grid that produced the
    structure score is scored for correctness against the true tokens.  Top-1 is
    exact id match; NLL is the head's bitwise NLL of the DECODED id (how (un)likely
    the sampled id is under the head — a sample trades NLL for sharpness).
    """
    from imas_ambix.camdyn.model import bitwise_nll

    p = np.asarray(post, dtype=int)
    dec = np.asarray(decoded_tokens)[p]
    tru = np.asarray(true_tokens)[p]
    top1 = float((dec == tru).mean())
    # NLL of the decoded id under the head (per the bit-factorised likelihood)
    nll = float(bitwise_nll(np.asarray(bit_logits)[p], dec).mean())
    # NLL of the TRUE id under the head (reference — what MAP-correctness costs)
    nll_true = float(bitwise_nll(np.asarray(bit_logits)[p], tru).mean())
    return {"top1_vs_true": top1, "nll_decoded_id": nll, "nll_true_id": nll_true}


# ---------------------------------------------------------------------------
# Decode bundle (token grids → images via the frozen OMAG2 subprocess)
# ---------------------------------------------------------------------------


def _decode_grids(grids_by_role: dict, work_dir: Path, device: str) -> dict:
    """Decode role→(F,16,16) token grids → role→(F,256,256,3) images (one pass)."""
    bb = mvr.BundleBuilder()
    wi = bb.add_window({"frame_time": []})
    for role, grid in grids_by_role.items():
        bb.add_grid(grid, wi, "_window", role)
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"
    bb.save(token_bundle)
    rd.run_decode_subprocess(token_bundle, image_bundle, device)
    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)
    index = json.loads(str(data["index"]))
    slot = {e["role"]: e["slot"] for e in index}
    return {role: images[slot[role]] for role in grids_by_role}


def _raw_gt_256(win: rd.DemoWindow) -> np.ndarray | None:
    """RAW level-1 frames for the window, resized to 256² — the camera truth."""
    raw = rd.load_raw_frames(win.shot_id, win.start, win.true_tokens.shape[0])
    if raw is None:
        return None
    out = np.zeros((raw.shape[0], 256, 256), dtype=np.float64)
    for fi in range(raw.shape[0]):
        out[fi] = _to_gray256(raw[fi])
    return out


# ---------------------------------------------------------------------------
# Per-arm, per-window evaluation
# ---------------------------------------------------------------------------


def evaluate_window(
    arm_name: str,
    bit_logits: np.ndarray,
    win: rd.DemoWindow,
    temperatures,
    n_seeds: int,
    seed0: int,
    work_dir: Path,
    device: str,
    *,
    visible: np.ndarray | None = None,
    forward_fn=None,
    maskgit_rounds: int = MASKGIT_ROUNDS,
) -> dict:
    """Score MAP / bernoulli / beam / maskgit / persistence / oracle / control.

    All stochastic roles are seed-averaged (``n_seeds``) with a bootstrap CI over
    the per-seed × per-frame correctness scores.  Returns a dict with the
    correctness terms (mean ± CI), HF secondary, and the accuracy Pareto numbers.

    ``forward_fn`` / ``visible`` enable the iterative MaskGIT role: ``forward_fn``
    is the ``(tokens, visible) -> bit_logits`` closure and ``visible`` is the
    scenario visibility mask (True = observed context MaskGIT never re-decodes).
    When either is None the MaskGIT role is skipped.
    """
    from imas_ambix.camdyn.model import LFQ_BITS

    post = list(range(FRONTIER, win.true_tokens.shape[0]))
    true_tokens = win.true_tokens
    gt256 = _raw_gt_256(win)
    if gt256 is None:
        return {"arm": arm_name, "error": "no raw GT frames"}

    do_maskgit = forward_fn is not None and visible is not None

    # --- deterministic token grids (MAP, persistence) ----------------------
    map_tok = decode_map(bit_logits)
    persist_tok = mv.persistence_tokens(true_tokens, FRONTIER)

    # --- build EVERY token grid up front, decode in ONE OMAG2 pass ----------
    # (repo §2b in-process rule: model load is the dominant cost — never decode
    # per-seed.  We materialise all sweep × seed grids and decode them together,
    # so the frozen VQModel is loaded ONCE per (arm, window).)
    grids: dict = {"map": map_tok, "persistence": persist_tok}
    grids["oracle_joint"] = decode_oracle_joint(
        bit_logits, true_tokens, temperature=1.0, rng=np.random.default_rng(seed0)
    )
    # per (decoder, T, seed) grids — seed s uses a deterministic per-seed RNG so
    # the sweep (seed 0) and the seed-average (seeds 0..n-1) are reproducible.
    sample_kinds = ["bernoulli", "beam"]
    if do_maskgit:
        sample_kinds.append("maskgit")
    seed_grid_tok: dict[tuple[str, float, int], np.ndarray] = {}
    for kind in sample_kinds:
        for T in temperatures:
            for s in range(n_seeds):
                rng = np.random.default_rng(seed0 + 1000 * s + int(round(T * 100)))
                if kind == "bernoulli":
                    tok = ts.bernoulli_sample(bit_logits, temperature=T, rng=rng)
                elif kind == "beam":
                    tok = ts.bit_beam_sample(
                        bit_logits, temperature=T, n_expand_bits=8, rng=rng
                    )
                else:  # maskgit — iterative, re-runs the model forward per round
                    tok = ts.maskgit_decode(
                        forward_fn,
                        true_tokens,
                        visible,
                        n_rounds=maskgit_rounds,
                        temperature=T,
                        top_k=MASKGIT_TOP_K,
                        confidence_noise=MASKGIT_CONF_NOISE,
                        rng=rng,
                    )
                key = (kind, float(T), s)
                seed_grid_tok[key] = tok
                grids[f"{kind}_T{T}_s{s}"] = tok

    decoded = _decode_grids(grids, work_dir, device)

    # coloured-noise control (image-space, not a token grid)
    cn = coloured_noise_like(gt256, post, np.random.default_rng(seed0))

    def _score_role(imgs, tok=None):
        sc = correctness_scores(imgs, gt256, post)
        out = {
            "location": sc["location"],
            "temporal": sc["temporal"],
            "hf_power": sc["hf_power"],
        }
        if tok is not None:
            out.update(token_accuracy(tok, true_tokens, bit_logits, post))
        return out, sc

    roles_out: dict = {}
    roles_out["map"], _ = _score_role(decoded["map"], map_tok)
    roles_out["persistence"], _ = _score_role(decoded["persistence"], persist_tok)
    roles_out["oracle_joint"], _ = _score_role(
        decoded["oracle_joint"], grids["oracle_joint"]
    )
    roles_out["oracle_joint"]["oracle_biased"] = True
    roles_out["oracle_joint"]["note"] = (
        "candidates = true_tokens XOR offsets — sees truth; UPPER BOUND only, "
        "never shipped"
    )
    cn_sc, _ = _score_role(cn)
    roles_out["coloured_noise"] = cn_sc

    # HF retention vs GT-raw for the secondary number
    gt_hf = np.mean([radial_hf_spectrum(gt256[fi])[1] for fi in post])
    for r in roles_out.values():
        if np.isfinite(r.get("hf_power", np.nan)) and gt_hf > 0:
            r["hf_ratio_vs_gt"] = float(r["hf_power"] / gt_hf)

    # --- temperature sweep: seed-0 correctness per T, pick the best T -------
    best = {k: (None, -np.inf) for k in sample_kinds}
    sweep = {k: {} for k in sample_kinds}
    for kind in sample_kinds:
        for T in temperatures:
            grid_key = f"{kind}_T{T}_s0"
            sc, _ = _score_role(decoded[grid_key], seed_grid_tok[(kind, float(T), 0)])
            c = _corr_objective(sc)
            sweep[kind][str(T)] = {**sc, "correctness": c}
            if c > best[kind][1]:
                best[kind] = (T, c)

    # --- seed-average the winning T for each sampled decoder (decoded above) --
    seed_avg = {}
    for kind in sample_kinds:
        T = best[kind][0]
        loc_samples, temp_samples = [], []
        hf_samples, top1_samples, nll_samples = [], [], []
        for s in range(n_seeds):
            dd = decoded[f"{kind}_T{T}_s{s}"]
            tok = seed_grid_tok[(kind, float(T), s)]
            sc = correctness_scores(dd, gt256, post)
            loc_samples.append(sc["_location_per_frame"])
            temp_samples.append(sc["_temporal_per_frame"])
            hf_samples.append(sc["hf_power"])
            acc = token_accuracy(tok, true_tokens, bit_logits, post)
            top1_samples.append(acc["top1_vs_true"])
            nll_samples.append(acc["nll_decoded_id"])
        loc_all = np.concatenate(loc_samples) if loc_samples else np.array([])
        temp_all = np.concatenate(temp_samples) if temp_samples else np.array([])
        loc_ci = bootstrap_ci(loc_all - 0.0) if loc_all.size else None
        temp_ci = bootstrap_ci(temp_all - 0.0) if temp_all.size else None
        seed_avg[kind] = {
            "temperature": float(T),
            "n_seeds": n_seeds,
            "location_mean": float(np.mean(loc_all)) if loc_all.size else float("nan"),
            "location_ci": [loc_ci["lo"], loc_ci["hi"]] if loc_ci else None,
            "temporal_mean": float(np.mean(temp_all))
            if temp_all.size
            else float("nan"),
            "temporal_ci": [temp_ci["lo"], temp_ci["hi"]] if temp_ci else None,
            "hf_power_mean": float(np.mean(hf_samples)) if hf_samples else float("nan"),
            "top1_vs_true_mean": float(np.mean(top1_samples)),
            "nll_decoded_id_mean": float(np.mean(nll_samples)),
        }
        if np.isfinite(seed_avg[kind]["hf_power_mean"]) and gt_hf > 0:
            seed_avg[kind]["hf_ratio_vs_gt"] = float(
                seed_avg[kind]["hf_power_mean"] / gt_hf
            )

    # --- hedging calibration (held-out cell-level entropy distribution) -----
    ent = per_bit_entropy(bit_logits)
    raw_native = rd.load_raw_frames(win.shot_id, win.start, win.true_tokens.shape[0])
    hedging = calibrate_hedging_threshold(ent, raw_native, FRONTIER)
    max_cell_ent = LFQ_BITS * np.log(2.0)

    return {
        "arm": arm_name,
        "shot_id": int(win.shot_id),
        "window_ms": [float(win.frame_time[0] * 1e3), float(win.frame_time[-1] * 1e3)],
        "gt_hf_power": float(gt_hf),
        "roles": roles_out,
        "temperature_sweep": sweep,
        "best_temperature": {k: best[k][0] for k in best},
        "seed_averaged": seed_avg,
        "hedging": hedging,
        "max_cell_entropy": float(max_cell_ent),
        "_decoded_for_figure": {
            "map": decoded["map"],
            f"bern_T{best['bernoulli'][0]}": decoded[
                f"bernoulli_T{best['bernoulli'][0]}_s0"
            ],
            f"beam_T{best['beam'][0]}": decoded[f"beam_T{best['beam'][0]}_s0"],
            **(
                {f"maskgit_T{best['maskgit'][0]}": decoded[
                    f"maskgit_T{best['maskgit'][0]}_s0"
                ]}
                if do_maskgit
                else {}
            ),
            "oracle_joint": decoded["oracle_joint"],
            "coloured_noise": cn,
        },
        "_gt256": gt256,
    }


def _corr_objective(scores: dict) -> float:
    """Correctness objective = location + temporal (NaN-safe).  HF is excluded."""
    loc = scores.get("location", np.nan)
    temp = scores.get("temporal", np.nan)
    parts = [v for v in (loc, temp) if np.isfinite(v)]
    return float(np.mean(parts)) if parts else -np.inf


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_figure(results: list[dict], out_path: Path) -> None:
    """Sampled-decode demo: per (arm, window) rows GT | MAP | beam | bernoulli |
    oracle | coloured-noise, columns = predicted frames; one block per arm.

    Renders the dynamics arm's first window as the headline block and overlays
    the correctness numbers so the restored filament structure is visible
    against MAP (blurred), the oracle ceiling, and the rejected coloured noise.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # headline: dynamics arm, the window with the strongest GT edge structure
    dyn = [
        r for r in results if r.get("arm") == "dynamics" and "_decoded_for_figure" in r
    ]
    if not dyn:
        logger.warning("[structfid] no dynamics result to render")
        return
    res = max(dyn, key=lambda r: r.get("gt_hf_power", 0.0))
    dec = res["_decoded_for_figure"]
    gt256 = res["_gt256"]
    post = list(range(FRONTIER, gt256.shape[0]))
    n_cols = min(6, len(post))
    cols = sorted(set(np.linspace(post[0], post[-1], n_cols).round().astype(int)))
    n_cols = len(cols)

    bern_key = next(k for k in dec if k.startswith("bern_T"))
    beam_key = next(k for k in dec if k.startswith("beam_T"))
    maskgit_key = next((k for k in dec if k.startswith("maskgit_T")), None)
    rows = [
        ("gt", "ground truth (raw)"),
        ("map", "MAP decode (bar)"),
    ]
    if maskgit_key is not None:
        rows.append(
            (maskgit_key, f"MaskGIT {maskgit_key.split('T')[1]} (coherent)")
        )
    rows += [
        (beam_key, f"beam {beam_key.split('T')[1]}"),
        (bern_key, f"bernoulli {bern_key.split('T')[1]}"),
        ("oracle_joint", "oracle joint (upper bound)"),
        ("coloured_noise", "coloured noise (control)"),
    ]
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.9 * n_cols + 2.2, 1.55 * n_rows + 1.3),
        squeeze=False,
        constrained_layout=True,
    )

    for ci, fi in enumerate(cols):
        gt_disp = gt256[fi]
        vmin, vmax = rd.display_limits(gt_disp)
        for ri, (rk, _lab) in enumerate(rows):
            ax = axes[ri][ci]
            if rk == "gt":
                img = gt256[fi]
            elif rk == "coloured_noise":
                img = dec["coloured_noise"][fi]
            else:
                img = _to_gray256(dec[rk][fi])
            ax.imshow(
                img, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"+{fi - FRONTIER} fr", fontsize=9)
            if ci == 0:
                ax.set_ylabel(rows[ri][1], fontsize=8)

    roles = res["roles"]
    sa = res["seed_averaged"]

    def _fmt(role_key, src):
        if src == "roles":
            r = roles.get(role_key, {})
            return f"loc={r.get('location', float('nan')):.3f} tmp={r.get('temporal', float('nan')):.3f}"
        r = sa.get(role_key, {})
        return f"loc={r.get('location_mean', float('nan')):.3f} tmp={r.get('temporal_mean', float('nan')):.3f}"

    mg_txt = (
        f" | MaskGIT {_fmt('maskgit', 'seed')}" if maskgit_key is not None else ""
    )
    caption = (
        f"Coherent vs per-bit decode — dynamics arm, shot {res['shot_id']} forecast.  "
        f"Correctness = edge-band SSIM vs RAW GT (location) + Δframe SSIM (temporal).\n"
        f"MAP {_fmt('map', 'roles')} (bar){mg_txt} | beam {_fmt('beam', 'seed')} | "
        f"bern {_fmt('bernoulli', 'seed')} | oracle {_fmt('oracle_joint', 'roles')} | "
        f"coloured-noise {_fmt('coloured_noise', 'roles')} (control — must FAIL location+temporal).  "
        f"MaskGIT re-forwards the model, conditioning each committed cell on its decided "
        f"neighbours; the others decode one frozen forward."
    )
    fig.suptitle(caption, fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[structfid] wrote %s", out_path)


def render_pareto(results: list[dict], out_path: Path) -> None:
    """Accuracy-vs-sharpness Pareto: structure (correctness) vs token top-1 / NLL.

    For each (arm, role) plot the correctness (location+temporal mean) against
    masked top-1 vs true tokens — the physical-mean MAP (high accuracy, low
    structure) vs the plausible-sample (lower accuracy, restored structure)
    trade-off the critique demands be made explicit.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colours = {"dynamics": "#2ca02c", "baseline": "#1f77b4"}
    markers = {
        "map": "o",
        "maskgit": "P",
        "beam": "^",
        "bernoulli": "s",
        "persistence": "x",
        "oracle_joint": "*",
        "coloured_noise": "D",
    }
    for res in results:
        if "roles" not in res:
            continue
        arm = res["arm"]
        col = colours.get(arm, "#888")
        roles = res["roles"]
        sa = res.get("seed_averaged", {})
        pts = []
        # MAP / persistence / oracle / coloured-noise from single decode
        for rk in ("map", "persistence", "oracle_joint", "coloured_noise"):
            r = roles.get(rk, {})
            corr = _corr_objective(r)
            top1 = r.get("top1_vs_true", np.nan)
            nll = r.get("nll_decoded_id", np.nan)
            pts.append((rk, corr, top1, nll))
        # seed-averaged sampled decodes (maskgit only present if it was run)
        for rk in ("maskgit", "beam", "bernoulli"):
            if rk not in sa:
                continue
            r = sa.get(rk, {})
            corr = _corr_objective(
                {"location": r.get("location_mean"), "temporal": r.get("temporal_mean")}
            )
            pts.append(
                (
                    rk,
                    corr,
                    r.get("top1_vs_true_mean", np.nan),
                    r.get("nll_decoded_id_mean", np.nan),
                )
            )
        for rk, corr, top1, nll in pts:
            if not np.isfinite(corr):
                continue
            ax1.scatter(
                top1,
                corr,
                c=col,
                marker=markers.get(rk, "."),
                s=70,
                edgecolors="k",
                linewidths=0.4,
            )
            if np.isfinite(nll):
                ax2.scatter(
                    nll,
                    corr,
                    c=col,
                    marker=markers.get(rk, "."),
                    s=70,
                    edgecolors="k",
                    linewidths=0.4,
                )
    ax1.set_xlabel("masked top-1 vs true tokens (accuracy)")
    ax1.set_ylabel("correctness (location + temporal SSIM)")
    ax1.set_title("accuracy vs restored structure")
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("bitwise NLL of decoded id (sharpness cost, lower=closer to MAP)")
    ax2.set_ylabel("correctness (location + temporal SSIM)")
    ax2.set_title("sharpness cost vs restored structure")
    ax2.grid(alpha=0.3)
    # legend
    import matplotlib.lines as mlines

    handles = [
        mlines.Line2D([], [], color="k", marker=m, ls="", label=rk)
        for rk, m in markers.items()
    ]
    handles += [
        mlines.Line2D([], [], color=c, marker="o", ls="", label=a)
        for a, c in colours.items()
    ]
    ax1.legend(handles=handles, fontsize=7, ncol=2)
    fig.suptitle(
        "camera-dynamics-wm — accuracy-vs-sharpness Pareto (MAP = physical mean, "
        "sample = plausible structure)",
        fontsize=11,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[structfid] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def build_verdict(results: list[dict]) -> dict:
    """Aggregate the per-arm/window results into the CORRECTNESS verdict.

    Decisive tests (per the critique):

    1. The coloured-noise control MUST be REJECTED — it must score near-zero on
       location + temporal while passing HF magnitude.  If it passes correctness,
       the metric is still gameable (reported honestly).
    2. Sampling restores CORRECT structure iff a sampled decode (beam/bernoulli)
       beats MAP on the correctness terms AND beats PERSISTENCE (else the model
       adds no forecast structure) — for the DYNAMICS arm.
    3. Otherwise sampling restores only plausible TEXTURE (HF up, location/temporal
       not).
    """

    def _agg(arm, role_src):
        loc, temp, hf = [], [], []
        for res in results:
            if res.get("arm") != arm or "roles" not in res:
                continue
            if role_src in ("map", "persistence", "oracle_joint", "coloured_noise"):
                r = res["roles"].get(role_src, {})
                loc.append(r.get("location", np.nan))
                temp.append(r.get("temporal", np.nan))
                hf.append(r.get("hf_ratio_vs_gt", np.nan))
            else:
                r = res.get("seed_averaged", {}).get(role_src, {})
                loc.append(r.get("location_mean", np.nan))
                temp.append(r.get("temporal_mean", np.nan))
                hf.append(r.get("hf_ratio_vs_gt", np.nan))
        return {
            "location": float(np.nanmean(loc)) if loc else float("nan"),
            "temporal": float(np.nanmean(temp)) if temp else float("nan"),
            "hf_ratio_vs_gt": float(np.nanmean(hf)) if hf else float("nan"),
        }

    # MaskGIT is present iff any result seed-averaged it.
    has_maskgit = any(
        "maskgit" in r.get("seed_averaged", {}) for r in results if "roles" in r
    )
    sample_roles = ["beam", "bernoulli"] + (["maskgit"] if has_maskgit else [])

    summary = {}
    for arm in ("dynamics", "baseline"):
        summary[arm] = {
            role: _agg(arm, role)
            for role in (
                ["map", *sample_roles, "persistence", "oracle_joint", "coloured_noise"]
            )
        }

    cn = summary["dynamics"]["coloured_noise"]
    mapd = summary["dynamics"]["map"]
    # control rejected: low location & temporal (well below MAP) but HF ~ passes
    cn_loc_fails = np.isfinite(cn["location"]) and cn["location"] < 0.5 * max(
        mapd["location"], 1e-6
    )
    cn_temp_fails = (not np.isfinite(cn["temporal"])) or cn["temporal"] < 0.2
    cn_hf_ok = np.isfinite(cn["hf_ratio_vs_gt"]) and cn["hf_ratio_vs_gt"] > 0.5
    control_rejected = bool(cn_loc_fails and cn_temp_fails)

    persist = summary["dynamics"]["persistence"]
    # MaskGIT is the coherent-decode lever under primary test; track it both
    # within the best-of-all-samplers verdict AND on its own vs MAP.
    maskgit = summary["dynamics"].get("maskgit")
    sample_candidates = [summary["dynamics"][r] for r in sample_roles]
    best_sample = max(sample_candidates, key=lambda r: _corr_objective(r))

    sample_beats_map = _corr_objective(best_sample) > _corr_objective(mapd)
    sample_beats_persist = _corr_objective(best_sample) > _corr_objective(persist)
    hf_restored = (
        np.isfinite(best_sample["hf_ratio_vs_gt"])
        and np.isfinite(mapd["hf_ratio_vs_gt"])
        and best_sample["hf_ratio_vs_gt"] > mapd["hf_ratio_vs_gt"] * 1.25
    )
    # MaskGIT-specific flags (the coherence lever, location + temporal vs MAP)
    maskgit_beats_map_located = bool(
        maskgit is not None
        and np.isfinite(maskgit["location"])
        and maskgit["location"] > mapd["location"]
    )
    maskgit_beats_map_temporal = bool(
        maskgit is not None
        and np.isfinite(maskgit["temporal"])
        and maskgit["temporal"] > mapd["temporal"]
    )
    maskgit_beats_map = bool(
        maskgit is not None
        and _corr_objective(maskgit) > _corr_objective(mapd)
    )

    if sample_beats_map and sample_beats_persist:
        verdict = (
            "RESTORES CORRECT STRUCTURE — sampled decode beats MAP and persistence "
            "on located + temporally-coherent structure (truth-grounded vs raw GT)"
        )
    elif hf_restored and not sample_beats_map:
        verdict = (
            "PLAUSIBLE TEXTURE ONLY — sampling raises HF magnitude but NOT the "
            "located/temporal correctness; the restored texture is not in the right "
            "place / does not evolve like GT"
        )
    elif sample_beats_map and not sample_beats_persist:
        verdict = (
            "AMBIGUOUS — sampled decode beats MAP on correctness but not persistence; "
            "the restored structure is no better than copy-the-last-frame"
        )
    else:
        verdict = "NO RESTORATION — sampling does not improve located/temporal correctness over MAP"

    if not control_rejected:
        verdict += (
            "  [WARNING: coloured-noise control NOT cleanly rejected — the metric "
            "may still be partly gameable; see control numbers]"
        )

    # MaskGIT-specific verdict (the coherence lever this run was built to test):
    # the question is whether iterative COHERENT decode — conditioning each
    # committed cell on its decided neighbours — beats the per-bit MAP on
    # LOCATED and/or TEMPORAL correctness where the single-pass samplers did not.
    if maskgit is not None:
        if maskgit_beats_map_located and maskgit_beats_map_temporal:
            maskgit_verdict = (
                "MASKGIT BEATS MAP — iterative coherent decode beats the per-bit MAP "
                "on BOTH located and temporal correctness; coherence (not just "
                "sampling) restores filaments in the right place"
            )
        elif maskgit_beats_map_located:
            maskgit_verdict = (
                "MASKGIT BEATS MAP ON LOCATION ONLY — coherent decode improves where "
                "the structure is, but not how it evolves"
            )
        elif maskgit_beats_map_temporal:
            maskgit_verdict = (
                "MASKGIT BEATS MAP ON TEMPORAL ONLY — coherent decode improves the "
                "Δframe coherence, but not the per-frame location"
            )
        else:
            maskgit_verdict = (
                "MASKGIT DOES NOT BEAT MAP — iterative coherent decode does not improve "
                "located/temporal correctness over the per-bit MAP; the lever is "
                "uncertainty-reduction (conditioning/temporal), not decode — MaskGIT "
                "is at best a visualisation option"
            )
    else:
        maskgit_verdict = "MASKGIT NOT RUN (no forward closure supplied)"

    return {
        "summary_by_arm": summary,
        "control": {
            "coloured_noise_dynamics": cn,
            "location_fails": bool(cn_loc_fails),
            "temporal_fails": bool(cn_temp_fails),
            "hf_passes": bool(cn_hf_ok),
            "rejected": control_rejected,
            "interpretation": (
                "spectrum-matched coloured noise must PASS hf and FAIL location+temporal; "
                "rejected=True means the correctness metric is NOT fooled by texture"
            ),
        },
        "dynamics_sample_beats_map": bool(sample_beats_map),
        "dynamics_sample_beats_persistence": bool(sample_beats_persist),
        "dynamics_hf_restored_but_not_correct": bool(
            hf_restored and not sample_beats_map
        ),
        "dynamics_maskgit_beats_map": maskgit_beats_map,
        "dynamics_maskgit_beats_map_located": maskgit_beats_map_located,
        "dynamics_maskgit_beats_map_temporal": maskgit_beats_map_temporal,
        "maskgit_verdict": maskgit_verdict,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    out_path: Path = DEFAULT_FIGURE,
    json_path: Path = DEFAULT_JSON,
    *,
    device: str = "cuda",
    n_seeds: int = N_SEEDS,
) -> dict:
    """Full hardened probe: select 3 windows → both arms → MAP/sample/oracle/control."""
    import contextlib

    import torch

    from imas_ambix.camdyn.arm_compare import _load_arm

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("[structfid] device = %s", dev)

    windows = select_structure_windows()
    if not windows:
        raise RuntimeError("no flat-top edge windows could be selected")
    logger.info(
        "[structfid] %d windows; %d seeds; arms=%s", len(windows), n_seeds, list(ARMS)
    )

    work_dir = Path(
        tempfile.mkdtemp(prefix="structfid-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    dev_str = "cuda" if dev.type == "cuda" else "cpu"

    results: list[dict] = []
    for arm_name, ckpt in ARMS.items():
        logger.info("[structfid] loading %s arm on %s", arm_name, dev)
        model, _cfg, cond_stats = _load_arm(ckpt, torch, dev)
        try:
            forward_fn = None
            for sw in windows:
                win = sw.window
                visible, bit_logits = forward_bit_logits(
                    model, torch, dev, win, cond_stats, SCENARIO, FRONTIER
                )
                # closure MaskGIT re-runs each round (frozen conditioning); the
                # same model/window the single-shot bit_logits came from.
                forward_fn = make_forward_fn(model, torch, dev, win, cond_stats)
                logger.info("[structfid] scoring %s arm shot %d", arm_name, win.shot_id)
                res = evaluate_window(
                    arm_name,
                    bit_logits,
                    win,
                    TEMPERATURES,
                    n_seeds,
                    SEED0,
                    work_dir,
                    dev_str,
                    visible=visible,
                    forward_fn=forward_fn,
                )
                results.append(res)
        finally:
            with contextlib.suppress(Exception):
                del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    verdict = build_verdict(results)

    # figures
    render_figure(results, out_path)
    render_pareto(results, out_path.parent / "fig-cdw-filament-pareto.png")

    # strip the heavy image arrays from the JSON
    json_results = []
    for r in results:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        json_results.append(rr)
    summary = {
        "task": "sampled-decode structure fidelity — does sampling restore CORRECT filaments?",
        "metric": (
            "location = edge-band SSIM(pred, RAW GT 256); temporal = edge-band "
            "SSIM(Δpred, ΔGT); HF radial power = SECONDARY texture-magnitude only. "
            "Scored vs load_raw_frames (camera truth), NOT decoded GT tokens."
        ),
        "scenario": SCENARIO,
        "frontier_frame": FRONTIER,
        "n_seeds": n_seeds,
        "temperatures_swept": list(TEMPERATURES),
        "maskgit_config": {
            "rounds": MASKGIT_ROUNDS,
            "schedule": "cosine masking ratio",
            "confidence_rule": (
                "joint bit-factorised log-likelihood of the chosen id; commit "
                "highest-confidence still-masked cells per round"
            ),
            "top_k_resampled_bits": MASKGIT_TOP_K,
            "confidence_noise": MASKGIT_CONF_NOISE,
            "temperatures_swept": list(TEMPERATURES),
            "note": (
                "iterative coherent decode — re-forwards the model each round so a "
                "committed cell conditions its still-masked neighbours via the "
                "model's own [MASK]-embedding path (no retrain, no model.py edit)"
            ),
        },
        "shots": [int(sw.window.shot_id) for sw in windows],
        "arms": list(ARMS),
        "oracle_note": (
            "oracle_joint is an UPPER BOUND (candidates = true_tokens XOR offsets); "
            "labelled oracle_biased; NOT a shippable decoder"
        ),
        "shippable_decoders": [
            "map (default)",
            "bernoulli (truth-free, single-pass)",
            "beam (truth-free coherent, single-pass)",
            "maskgit (truth-free coherent, iterative — re-forwards the model)",
        ],
        "results": json_results,
        "verdict": verdict,
        "figures": {
            "demo": str(out_path),
            "pareto": str(out_path.parent / "fig-cdw-filament-pareto.png"),
        },
    }

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_report(summary)
    return summary


def _accuracy_table(summary: dict) -> list[tuple[str, float, float]]:
    """Per-role mean top-1 / decoded-id NLL on the DYNAMICS arm's windows.

    MAP / oracle come from the single-decode ``roles`` block; the sampled
    decoders (maskgit / beam / bernoulli) from the seed-averaged block.  Skipped
    if no dynamics result carries the role.  Returns ``[(role, top1, nll), ...]``
    — the accuracy-vs-sharpness axis the critique requires be explicit.
    """
    res = [r for r in summary["results"] if r.get("arm") == "dynamics"]
    out: list[tuple[str, float, float]] = []
    for role in ("map", "maskgit", "beam", "bernoulli", "oracle_joint"):
        top1, nll = [], []
        for r in res:
            if role in ("map", "oracle_joint"):
                rr = r.get("roles", {}).get(role, {})
                t = rr.get("top1_vs_true", np.nan)
                n = rr.get("nll_decoded_id", np.nan)
            else:
                rr = r.get("seed_averaged", {}).get(role, {})
                t = rr.get("top1_vs_true_mean", np.nan)
                n = rr.get("nll_decoded_id_mean", np.nan)
            if np.isfinite(t):
                top1.append(t)
            if np.isfinite(n):
                nll.append(n)
        if top1 or nll:
            out.append(
                (
                    role,
                    float(np.mean(top1)) if top1 else float("nan"),
                    float(np.mean(nll)) if nll else float("nan"),
                )
            )
    return out


def _print_report(summary: dict) -> None:
    v = summary["verdict"]
    print("\n" + "=" * 80)
    print("STRUCTURE FIDELITY (HARDENED) — does sampling restore CORRECT filaments?")
    print("=" * 80)
    print(f"shots   : {summary['shots']}")
    print(
        f"arms    : {summary['arms']}   seeds: {summary['n_seeds']}   T swept: {summary['temperatures_swept']}"
    )
    print("-" * 80)
    role_order = [
        r
        for r in (
            "map",
            "maskgit",
            "beam",
            "bernoulli",
            "persistence",
            "oracle_joint",
            "coloured_noise",
        )
        if r in v["summary_by_arm"]["dynamics"]
    ]
    for arm in summary["arms"]:
        s = v["summary_by_arm"][arm]
        print(f"[{arm}]  (mean over {len(summary['shots'])} windows)")
        print(f"  {'role':<16}{'location':>10}{'temporal':>10}{'HF/GT':>9}")
        for role in role_order:
            r = s[role]
            print(
                f"  {role:<16}{r['location']:>10.3f}{r['temporal']:>10.3f}"
                f"{r['hf_ratio_vs_gt']:>9.2f}"
            )
    # accuracy-vs-sharpness: top-1 / NLL of the decoded ids (dynamics arm, mean)
    print("-" * 80)
    print("[dynamics] decoded-token accuracy (mean over windows)")
    print(f"  {'role':<16}{'top1':>10}{'NLL_dec':>10}")
    for role, top1, nll in _accuracy_table(summary):
        t = f"{top1:.4f}" if np.isfinite(top1) else "   --"
        n = f"{nll:.3f}" if np.isfinite(nll) else "   --"
        print(f"  {role:<16}{t:>10}{n:>10}")
    print("-" * 80)
    c = v["control"]
    print(
        f"COLOURED-NOISE CONTROL: location_fails={c['location_fails']} "
        f"temporal_fails={c['temporal_fails']} hf_passes={c['hf_passes']} "
        f"-> REJECTED={c['rejected']}"
    )
    print(
        f"dynamics: sample_beats_MAP={v['dynamics_sample_beats_map']} "
        f"sample_beats_PERSISTENCE={v['dynamics_sample_beats_persistence']} "
        f"hf_restored_but_not_correct={v['dynamics_hf_restored_but_not_correct']}"
    )
    if "maskgit_verdict" in v:
        print(
            f"dynamics MaskGIT: beats_MAP={v.get('dynamics_maskgit_beats_map')} "
            f"located={v.get('dynamics_maskgit_beats_map_located')} "
            f"temporal={v.get('dynamics_maskgit_beats_map_temporal')}"
        )
    print("-" * 80)
    if "maskgit_verdict" in v:
        print(f"MASKGIT : {v['maskgit_verdict']}")
    print(f"VERDICT : {v['verdict']}")
    print(f"figures : {summary['figures']['demo']}")
    print(f"          {summary['figures']['pareto']}")
    print("=" * 80 + "\n")
    print(
        "SUMMARY_JSON "
        + json.dumps({"verdict": v["verdict"], "shots": summary["shots"]})
    )


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DEFAULT_FIGURE))
    p.add_argument("--json", default=str(DEFAULT_JSON))
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(Path(args.out), Path(args.json), device=args.device, n_seeds=args.n_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
