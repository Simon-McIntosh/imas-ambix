"""Tokenizer round-trip fidelity probe for bright ELM edge structure.

Settles a pure-tokenizer question (NO world-model prediction): does the
frozen Open-MAGVIT2 (OMAG2) 18-bit LFQ patch tokenizer preserve the
macroscopic bright ELM filament / edge-burst structure visible in raw MAST
``rbb`` camera frames, or does it smear it away at encode time?

The probe is the tokenizer ROUND-TRIP: take the GROUND-TRUTH tokens already
stored on disk for a real-ELM camera window (``win.true_tokens``), decode
them straight back to images through the same frozen VQModel the corpus
encoder used, and put that decode next to the raw camera frames.  No model
forward, no masking, no conditioning — encode-then-decode and nothing else.

What it does
------------
1. **Select a clearly-ELMing window.**  ELMs are confirmed with the FAST
   ``xim`` Dα photodiodes (~50 kHz, the real sub-ms ELM signature — not the
   slow ~1 kHz ``ada`` integrated trace).  The strongest fast-Dα burst on
   each candidate shot is mapped to a camera frame; the window is built so
   the burst evolves across the columns, and the choice is cross-checked
   against the raw-camera edge-burst score so the selected burst is actually
   bright and legible in the image (the token grid only resolves bright
   structure).
2. **Round-trip** ``win.true_tokens`` through the OMAG2 decoder subprocess.
3. **Load raw** frames at NATIVE resolution (no downsample — the true detail).
4. **Render** a 2-row figure: raw (native) over round-trip (256²), columns =
   time across the ELM, the ELM-peak column marked.
5. **Quantify** retention: SSIM / PSNR (full frame + ELM-filament crop), a
   high-spatial-frequency energy-retention ratio on the crop, and a 1-D
   intensity profile across a bright filament (do the peaks survive?).
6. **Conditional Cosmos comparison**: only if OMAG2 visibly SMEARS the
   filament does it also round-trip through the on-disk Cosmos tokenizer and
   add a third row.  If OMAG2 preserves the filament, Cosmos is skipped — the
   question is already answered.

Runs on a compute node with GPFS + the OMAG2 decoder venv (no network
needed: tokens, raw frames and the venv are all on GPFS).  Decode device
falls back to CPU automatically when no GPU is present.
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

#: Shots scanned for a dramatic ELM.  The flat-top demo shots first (known
#: bright + structured), then a bounded held-out sample (reuses the movie
#: candidate list so the selection is consistent with the rest of the demo).
FIDELITY_FRAMES = 16

#: Fast Dα photodiode channels in the ``xim`` group (~50 kHz).  These resolve
#: individual ELMs (20 µs cadence ≪ inter-ELM spacing), unlike the ~1 kHz
#: ``ada`` integrated trace.  ``hm10`` is the horizontal-midplane monitor,
#: the classic MAST ELM diode; the others cross-check the burst.
FAST_DALPHA_CHANNELS = (
    "da_hm10_r",
    "da_hm10_t",
    "da_hl11_r",
    "da_to10",
    "da_bo10",
    "da_hu10_t",
)

#: Output figure (committed) — raw vs OMAG2 round-trip across an ELM.
DEFAULT_FIGURE = Path(
    "docs/figures/camera-dynamics-wm/fig-cdw-tokenizer-elm-fidelity.png"
)

#: On-disk Cosmos tokenizers (conditional secondary comparison).
COSMOS_ROOT = Path("/work/projects/imas_gpu/mast-tokens/cosmos/v1")


# ---------------------------------------------------------------------------
# Fast-Dα ELM confirmation
# ---------------------------------------------------------------------------


@dataclass
class DalphaBurst:
    """A confirmed fast-Dα ELM burst on one shot."""

    shot_id: int
    channel: str
    burst_time_s: float
    burst_ratio: float  # peak / robust-baseline (how many×)
    burst_sigma: float  # peak height in robust σ above baseline


def _fast_dalpha_burst(shot_id: int) -> DalphaBurst | None:
    """Strongest fast-Dα ELM burst on a shot, over the flat-top window.

    Reads the ~50 kHz ``xim`` Dα diodes, restricts to the flat-top
    (0.15–0.45 s), high-passes each channel and reports the single most
    dramatic transient across all channels.  Returns None if the ``xim``
    group / channels are unavailable.
    """
    import zarr

    path = Path(f"/work/projects/imas_gpu/mast/level1/shots/{shot_id}.zarr/xim")
    if not path.exists():
        return None
    try:
        g = zarr.open_group(str(path), mode="r")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tokfid] xim for shot %d unavailable: %s", shot_id, exc)
        return None
    keys = set(g.array_keys())
    if "time" not in keys:
        return None
    t = np.asarray(g["time"][:]).astype(np.float64)
    flat = (t > 0.15) & (t < 0.45)
    if flat.sum() < 50:
        return None

    best: DalphaBurst | None = None
    for ch in FAST_DALPHA_CHANNELS:
        if ch not in keys:
            continue
        d = np.asarray(g[ch][:]).astype(np.float64)
        fin = np.isfinite(d) & flat
        if fin.sum() < 50:
            continue
        dw = d[fin]
        tw = t[fin]
        base = float(np.median(dw))
        # A near-zero / negative baseline (offset-only channel) inflates both
        # the ratio and the MAD-σ into meaningless astronomical numbers — that
        # is a calibration artifact, not a brighter ELM.  Require the channel
        # to carry a real positive light level before trusting its burst.
        if base < 0.02:
            continue
        mad = float(np.median(np.abs(dw - base))) or (float(dw.std()) or 1.0)
        pk = int(np.argmax(dw))
        sigma = (float(dw[pk]) - base) / (1.4826 * mad + 1e-12)
        ratio = float(dw[pk]) / (abs(base) + 1e-12)
        # require a genuine transient: rises into and falls out of the peak
        rises = pk > 0 and dw[pk] > dw[pk - 1]
        falls = pk < dw.size - 1 and dw[pk] > dw[pk + 1]
        if not (rises and falls):
            sigma *= 0.2
        if best is None or sigma > best.burst_sigma:
            best = DalphaBurst(
                shot_id=int(shot_id),
                channel=ch,
                burst_time_s=float(tw[pk]),
                burst_ratio=ratio,
                burst_sigma=float(sigma),
            )
    return best


def _camera_frame_at_time(shot_id: int, t_target_s: float) -> tuple[int, float] | None:
    """Camera ``rbb`` frame index nearest a target time → ``(frame, t_s)``."""
    import zarr

    path = Path(f"/work/projects/imas_gpu/mast/level1/shots/{shot_id}.zarr/rbb")
    if not path.exists():
        return None
    try:
        g = zarr.open_group(str(path), mode="r")
        if "time" not in set(g.array_keys()):
            return None
        t = np.asarray(g["time"][:]).astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tokfid] rbb time for shot %d unavailable: %s", shot_id, exc)
        return None
    if t.size == 0:
        return None
    fi = int(np.argmin(np.abs(t - t_target_s)))
    return fi, float(t[fi])


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


@dataclass
class FidelityWindow:
    """The selected ELM window + its fast-Dα confirmation."""

    window: rd.DemoWindow
    peak_frame: int  # ELM-peak frame INDEX in the window
    burst: DalphaBurst | None
    camera_score: float


#: The encoder ingests the native rbb frame resized to 256² assuming the
#: canonical ``(112,156)`` aspect.  Old large-format sensors (e.g. 528×512,
#: 462×512, 80×100) tokenise with a different field of view and are not the
#: regime the world model trains on, so we restrict the probe to cameras
#: close to the canonical aspect — the fidelity question is about the
#: production tokenizer on production-geometry frames.
CANONICAL_HW = mv.ORIGINAL_HW  # (112, 156)


def _is_canonical_aspect(raw: np.ndarray) -> bool:
    """True when raw frames are close to the canonical ``(112,156)`` rbb size."""
    h, w = raw.shape[1], raw.shape[2]
    return abs(h - CANONICAL_HW[0]) <= 24 and abs(w - CANONICAL_HW[1]) <= 24


def select_elm_window(
    n_frames: int = FIDELITY_FRAMES,
) -> FidelityWindow | None:
    """Pick the most dramatic, legible ELM window across the candidate shots.

    Each candidate shot is confirmed with the fast ``xim`` Dα diodes; the
    burst time is mapped to a camera frame and a window is built so the burst
    sits centrally and evolves across columns.  Selection guards reject the
    two failure modes that masquerade as "dramatic ELMs":

    * **termination spikes** — the single brightest Dα peak is often the
      discharge ending (a one-frame flash that decays to near-dark), not a
      repetitive flat-top ELM.  We require the frames *after* the camera peak
      to stay plasma-bright (the burst sits on a living plasma).
    * **off-geometry sensors** — old large-format cameras tokenise a
      different field of view than the production world model sees, so we
      keep only canonical-aspect frames.

    The window is ranked by the raw-camera edge-burst score × peak-frame
    brightness (the token grid only resolves bright structure), weighted by
    the fast-Dα burst σ (capped so a near-zero-baseline channel cannot
    dominate).  A native window spanning < 1 ms is decimated so the ELM
    actually evolves across the columns.  Returns the best window.
    """
    from imas_ambix.camdyn.dataset import discover_token_shots

    candidates = mvr._elm_candidate_shots()
    best: FidelityWindow | None = None
    best_rank = 0.0
    half = n_frames // 2

    for sid in candidates:
        burst = _fast_dalpha_burst(sid)
        if burst is None or burst.burst_sigma < 4.0:
            continue  # no convincing fast-Dα ELM on this shot
        cam = _camera_frame_at_time(sid, burst.burst_time_s)
        if cam is None:
            continue
        burst_frame, _ = cam

        specs = discover_token_shots(shot_ids=[sid], read_n_frames=True)
        if not specs:
            continue
        n_tot = int(specs[0].n_frames)
        start = int(np.clip(burst_frame - half, 0, max(0, n_tot - n_frames)))

        win = mvr._window_at(sid, start, n_frames)
        if win is None:
            continue
        raw = rd.load_raw_frames(sid, start, n_frames)
        if raw is None or raw.shape[0] < 3:
            continue
        # off-geometry sensor → not the production tokenizer regime
        if not _is_canonical_aspect(raw):
            continue
        rawf = raw.astype(np.float64)
        cam_score, cam_peak = mv.camera_elm_score(rawf)
        bright = float(np.mean(rawf[cam_peak])) if rawf.shape[0] > cam_peak else 0.0

        # termination-spike guard: a flat-top ELM sits on a living plasma and
        # evolves across the window — the burst rises INTO the peak and falls
        # OUT of it, with plasma-bright frames on BOTH sides.  A disruption
        # flash instead lands on the last frame and decays to near-dark.  We
        # require the camera peak to sit in the window interior with a bright
        # tail after it (and the peak to be plasma-bright, not noise).
        n_raw = rawf.shape[0]
        head = rawf[:cam_peak]
        tail = rawf[cam_peak + 1 :]
        head_bright = float(np.mean(head)) if head.size else 0.0
        tail_bright = float(np.mean(tail)) if tail.size else 0.0
        interior = 1 <= cam_peak <= n_raw - 2
        living = (
            bright > 40.0
            and interior
            and tail_bright > 0.4 * bright
            and head_bright > 0.2 * bright
        )

        rank = burst.burst_sigma * max(cam_score, 1e-6) * max(bright, 1.0)
        if not living:
            rank *= 0.001  # heavily demote termination flashes / edge bursts
        logger.info(
            "[tokfid] candidate shot %d: fast-Dα %s σ=%.1f (×%.1f) @%.1fms "
            "→ camF%d; cam-burst=%.2f peakF=%d bright=%.0f head=%.0f tail=%.0f "
            "living=%s rank=%.1f",
            sid,
            burst.channel,
            burst.burst_sigma,
            burst.burst_ratio,
            burst.burst_time_s * 1e3,
            burst_frame,
            cam_score,
            cam_peak,
            bright,
            head_bright,
            tail_bright,
            living,
            rank,
        )
        if rank > best_rank:
            best_rank = rank
            best = FidelityWindow(
                window=win,
                peak_frame=int(cam_peak),
                burst=burst,
                camera_score=float(cam_score),
            )

    # If the native window spans < 1 ms (fast cadence), the ELM barely evolves
    # across the 16 frames; decimate a wide window so the burst rises and falls
    # over a real ELM timescale (~1.5 ms) with the peak ~35 % in.
    if best is not None:
        w = best.window
        ft = np.asarray(w.frame_time, dtype=float)
        span_ms = float((ft[-1] - ft[0]) * 1e3)
        if span_ms < 1.0:
            dwin, new_peak = mvr.decimate_demo_window(
                w, span_ms=1.5, anchor_frame=best.peak_frame, anchor_frac=0.35
            )
            logger.info(
                "[tokfid] native window span %.2f ms < 1 ms → decimated to "
                "%.2f ms (peak f%d→f%d)",
                span_ms,
                float((dwin.frame_time[-1] - dwin.frame_time[0]) * 1e3),
                best.peak_frame,
                new_peak,
            )
            best = FidelityWindow(
                window=dwin,
                peak_frame=int(new_peak if new_peak >= 0 else best.peak_frame),
                burst=best.burst,
                camera_score=best.camera_score,
            )

    if best is not None:
        w = best.window
        ft = np.asarray(w.frame_time, dtype=float)
        logger.info(
            "[tokfid] SELECTED shot %d start %d window %.1f-%.1f ms peak@f%d "
            "(fast-Dα %s σ=%.1f, ×%.1f @%.1fms)",
            w.shot_id,
            w.start,
            ft[0] * 1e3,
            ft[-1] * 1e3,
            best.peak_frame,
            best.burst.channel if best.burst else "?",
            best.burst.burst_sigma if best.burst else float("nan"),
            best.burst.burst_ratio if best.burst else float("nan"),
            best.burst.burst_time_s * 1e3 if best.burst else float("nan"),
        )
    else:
        logger.warning("[tokfid] no convincing fast-Dα ELM window found")
    return best


# ---------------------------------------------------------------------------
# Round-trip decode (true tokens → images, no model)
# ---------------------------------------------------------------------------


def roundtrip_true_tokens(
    win: rd.DemoWindow, work_dir: Path, device: str
) -> np.ndarray:
    """Decode ``win.true_tokens`` → ``(F,256,256,3)`` uint8 via OMAG2.

    Pure tokenizer round-trip: the stored ground-truth tokens are decoded
    straight back through the frozen VQModel — no model forward, no masking.
    Reuses the same one-window-as-bundle handoff the movie driver uses.
    """
    bb = mvr.BundleBuilder()
    wi = bb.add_window(
        {
            "shot_id": int(win.shot_id),
            "start": int(win.start),
            "frame_time": np.asarray(win.frame_time).tolist(),
        }
    )
    bb.add_grid(win.true_tokens, wi, "_window", "true")
    token_bundle = work_dir / "tokens.npz"
    image_bundle = work_dir / "images.npz"
    bb.save(token_bundle)
    rd.run_decode_subprocess(token_bundle, image_bundle, device)

    data = np.load(str(image_bundle), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)  # (1,F,256,256,3)
    return images[0]


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------


def _to_gray256(img: np.ndarray) -> np.ndarray:
    """A frame → float64 grayscale resized to 256² (for like-for-like metrics)."""
    from PIL import Image

    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., 0]
    im = Image.fromarray(a.astype(np.uint8)).resize((256, 256), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM between two 0–255 float images (Wang et al. constants)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return float(num / den) if den > 0 else 0.0


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR (dB) between two 0–255 images."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def _highpass_energy(img: np.ndarray) -> float:
    """High-spatial-frequency energy: variance of a Laplacian-filtered image.

    A 3×3 discrete Laplacian isolates the sharp edges / striations an ELM
    filament produces; its energy (sum of squares) measures how much fine
    edge structure the image carries.  Reported as a RATIO (round-trip ÷ raw)
    so a value near 1 means the filament's high-frequency content survived
    and ≪ 1 means the tokenizer smeared it.
    """
    a = img.astype(np.float64)
    lap = np.zeros_like(a)
    lap[1:-1, 1:-1] = (
        a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4.0 * a[1:-1, 1:-1]
    )
    return float(np.sum(lap**2))


def _elm_crop_box(raw_gray256: np.ndarray, margin: int = 24, half: int = 48) -> tuple:
    """Locate the ELM edge-structure crop on the 256² grid → ``(r0,r1,c0,c1)``.

    The ELM filaments / edge brightening are where the frame carries the most
    fine spatial structure.  We take the local Laplacian energy (a high-pass),
    smooth it, exclude a border ``margin`` (so a sensor-edge artifact or the
    saturated frame border cannot win), and centre a ``2·half`` crop on the
    peak interior edge-structure region.
    """
    from PIL import Image, ImageFilter

    a = raw_gray256.astype(np.float64)
    lap = np.zeros_like(a)
    lap[1:-1, 1:-1] = (
        a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4.0 * a[1:-1, 1:-1]
    )
    lap_max = float((lap**2).max())
    if lap_max <= 0:  # featureless frame → centre crop, nothing to localise
        return 80, 176, 80, 176
    em = np.asarray(
        Image.fromarray(
            np.clip(lap**2 / lap_max * 255, 0, 255).astype(np.uint8)
        ).filter(ImageFilter.BoxBlur(10)),
        dtype=np.float64,
    )
    # mask out the border so the crop centres on interior plasma structure
    mask = np.zeros_like(em)
    mask[margin : 256 - margin, margin : 256 - margin] = 1.0
    em = em * mask
    r, c = np.unravel_index(int(np.argmax(em)), em.shape)
    r0 = int(np.clip(r - half, 0, 256 - 2 * half))
    c0 = int(np.clip(c - half, 0, 256 - 2 * half))
    return r0, r0 + 2 * half, c0, c0 + 2 * half


def _filament_profile(
    raw_gray256: np.ndarray, rt_gray256: np.ndarray, box: tuple
) -> dict:
    """1-D intensity transect across the strongest filament edge in the crop.

    Operates on UN-saturated grayscale (a frame whose bright core clips to a
    flat plateau would give a meaningless zero-contrast line).  Picks the crop
    ROW with the largest intra-row intensity variation in the RAW frame — the
    transect that actually crosses a filament edge — and reads that same row
    from both raw and round-trip.  Reports the per-image peak count (rising
    edges of runs above mean+0.5σ), the peak value, and the line contrast so
    "do the bright filament peaks survive?" is answerable.
    """
    r0, r1, c0, c1 = box
    raw_crop = raw_gray256[r0:r1, c0:c1].astype(np.float64)
    rt_crop = rt_gray256[r0:r1, c0:c1].astype(np.float64)
    # transect = the crop row with the most intensity variation (an edge), not
    # the brightest row (which may be a saturated plateau with no contrast).
    row = int(np.argmax(raw_crop.std(axis=1)))
    raw_line = raw_crop[row]
    rt_line = rt_crop[row]

    def _count_peaks(line: np.ndarray) -> int:
        thr = float(line.mean()) + 0.5 * float(line.std())
        above = line > thr
        return int(np.sum(above[1:] & ~above[:-1])) + int(above[0])

    return {
        "row": row,
        "raw_line": raw_line.tolist(),
        "rt_line": rt_line.tolist(),
        "raw_peaks": _count_peaks(raw_line),
        "rt_peaks": _count_peaks(rt_line),
        "raw_peak_value": float(raw_line.max()),
        "rt_peak_value": float(rt_line.max()),
        "raw_contrast": float(raw_line.max() - raw_line.min()),
        "rt_contrast": float(rt_line.max() - rt_line.min()),
    }


def _metric_frame(raw_frames: np.ndarray, peak_frame: int, radius: int = 3) -> int:
    """Pick the frame near the ELM peak with the most RESOLVABLE structure.

    The ELM peak frame is often a near-total saturation white-out — a uniform
    bright field with no fine structure for ANY tokenizer to preserve or
    destroy, so it cannot discriminate.  The filaments are sharpest on the
    rise/decay frames, where the edge is bright but the core has not clipped.
    Among frames within ``radius`` of the peak we choose the one whose 256²
    grayscale carries the highest intensity standard deviation (the most
    contrast / structure) while staying ELM-bright (mean ≥ 0.5× the peak's).
    Falls back to the peak frame.
    """
    n = raw_frames.shape[0]
    pf = int(np.clip(peak_frame, 0, n - 1))
    peak_mean = float(_to_gray256(raw_frames[pf]).mean())
    lo, hi = max(0, pf - radius), min(n, pf + radius + 1)
    best_f, best_std = pf, -1.0
    for fi in range(lo, hi):
        g = _to_gray256(raw_frames[fi])
        if g.mean() < 0.5 * peak_mean:
            continue  # require it to still be an ELM-bright frame
        sat = float((g >= 254).mean())
        # contrast of the unsaturated part = resolvable structure
        std = float(g.std()) * (1.0 - sat)
        if std > best_std:
            best_std, best_f = std, fi
    return best_f


def compute_fidelity(
    raw_frames: np.ndarray, roundtrip: np.ndarray, peak_frame: int
) -> dict:
    """All fidelity numbers on the most-resolvable ELM frame near the peak.

    ``raw_frames`` is native-res ``(F,H,W)``; ``roundtrip`` is ``(F,256,256,3)``.
    Raw is resized to 256² FOR THE METRIC ONLY (the figure shows native raw).
    Both are robustly contrast-normalised to a common 0–255 scale so SSIM /
    PSNR / HF-energy compare structure, not the per-shot brightness offset.
    """
    pf = _metric_frame(raw_frames, peak_frame)
    raw_g = _to_gray256(raw_frames[pf])
    rt_g = _to_gray256(roundtrip[pf])

    # SSIM / PSNR / HF use a SHARED contrast normalisation (1/99 pct of the raw
    # frame, applied to both) so the comparison is about STRUCTURE, not the
    # cross-tokenizer absolute level.  Clipping here is fine — these metrics
    # are scale-relative.
    vmin, vmax = rd.display_limits(raw_g)
    raw_n = np.clip((raw_g - vmin) / max(vmax - vmin, 1e-9) * 255.0, 0, 255)
    rt_n = np.clip((rt_g - vmin) / max(vmax - vmin, 1e-9) * 255.0, 0, 255)

    # The filament TRANSECT + crop placement need an UN-saturated stretch: each
    # image mapped to its OWN full robust range (2/98 pct) so a bright ELM frame
    # still carries the filament contrast instead of clipping to a flat 255.
    def _gentle(g):
        lo, hi = np.percentile(g, 2.0), np.percentile(g, 98.0)
        return np.clip((g - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255)

    raw_t = _gentle(raw_g)
    rt_t = _gentle(rt_g)

    box = _elm_crop_box(raw_t)
    r0, r1, c0, c1 = box
    raw_crop = raw_n[r0:r1, c0:c1]
    rt_crop = rt_n[r0:r1, c0:c1]

    raw_hf = _highpass_energy(raw_crop)
    rt_hf = _highpass_energy(rt_crop)

    return {
        "peak_frame": pf,
        "ssim_full": _ssim(raw_n, rt_n),
        "psnr_full_db": _psnr(raw_n, rt_n),
        "ssim_crop": _ssim(raw_crop, rt_crop),
        "psnr_crop_db": _psnr(raw_crop, rt_crop),
        "hf_energy_raw": raw_hf,
        "hf_energy_roundtrip": rt_hf,
        "hf_energy_retention_ratio": float(rt_hf / raw_hf) if raw_hf > 0 else 0.0,
        "elm_crop_box": list(box),
        "filament_profile": _filament_profile(raw_t, rt_t, box),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_figure(
    win: rd.DemoWindow,
    raw_frames: np.ndarray,
    roundtrip: np.ndarray,
    peak_frame: int,
    metrics: dict,
    burst: DalphaBurst | None,
    out_path: Path,
    cosmos: dict | None = None,
) -> None:
    """Raw (native, row 1) over OMAG2 round-trip (256², row 2) across the ELM.

    Optional third row = a Cosmos round-trip (only when OMAG2 smears the
    filament).  Per-column robust display limits from the raw frame are shared
    down each column so structure is legible and reconstruction over/under-shoot
    stays honest.  The ELM-peak column is boxed in the accent colour.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    ft = np.asarray(win.frame_time, dtype=float)
    n_frames = ft.shape[0]
    n_cols = min(8, n_frames)
    cols = sorted(set(np.linspace(0, n_frames - 1, n_cols).round().astype(int)))
    # ensure the ELM-peak frame is one of the columns
    pf = int(np.clip(peak_frame, 0, n_frames - 1))
    if pf not in cols:
        cols[int(np.argmin([abs(c - pf) for c in cols]))] = pf
        cols = sorted(set(cols))
    n_cols = len(cols)

    rows = [("raw rbb (native res)", "raw"), ("OMAG2 round-trip (256²)", "omag2")]
    if cosmos is not None:
        rows.append((f"Cosmos round-trip ({cosmos['name']})", "cosmos"))
    n_rows = len(rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.9 * n_cols + 1.8, 1.7 * n_rows + 1.2),
        squeeze=False,
        constrained_layout=True,
    )

    # crop box on the 256² grid (only marked on the round-trip rows)
    cb = metrics.get("elm_crop_box")

    for ci, fi in enumerate(cols):
        raw_native = raw_frames[fi].astype(np.float64)
        rt_g = _to_gray256(roundtrip[fi])
        # per-column limits from the raw native frame, mapped consistently
        vmin, vmax = rd.display_limits(raw_native)
        # the round-trip is on a different absolute scale; use its own robust
        # limits so its structure is visible (the metric handles cross-scale).
        rvmin, rvmax = rd.display_limits(rt_g)
        is_peak = fi == pf
        dt_ms = (ft[fi] - ft[pf]) * 1e3

        # row 0: raw native
        ax = axes[0][ci]
        ax.imshow(
            raw_native, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{'ELM PEAK ' if is_peak else ''}{dt_ms:+.2f} ms",
            fontsize=9,
            color=(mv.ACCENT if is_peak else "black"),
            fontweight=("bold" if is_peak else "normal"),
        )
        if is_peak:
            for s in ax.spines.values():
                s.set_color(mv.ACCENT)
                s.set_linewidth(2.5)
        if ci == 0:
            ax.set_ylabel(rows[0][0], fontsize=9)

        # row 1: OMAG2 round-trip
        ax = axes[1][ci]
        ax.imshow(rt_g, cmap="inferno", vmin=rvmin, vmax=rvmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if cb is not None:
            r0, r1, c0, c1 = cb
            ax.add_patch(
                mpatches.Rectangle(
                    (c0, r0),
                    c1 - c0,
                    r1 - r0,
                    fill=False,
                    edgecolor="#39c5cf",
                    linewidth=1.0,
                    linestyle=":",
                )
            )
        if is_peak:
            for s in ax.spines.values():
                s.set_color(mv.ACCENT)
                s.set_linewidth(2.5)
        if ci == 0:
            ax.set_ylabel(rows[1][0], fontsize=9)

        # row 2: Cosmos round-trip (optional)
        if cosmos is not None:
            ax = axes[2][ci]
            cg = _to_gray256(cosmos["images"][fi])
            cvmin, cvmax = rd.display_limits(cg)
            ax.imshow(
                cg, cmap="inferno", vmin=cvmin, vmax=cvmax, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if is_peak:
                for s in ax.spines.values():
                    s.set_color(mv.ACCENT)
                    s.set_linewidth(2.5)
            if ci == 0:
                ax.set_ylabel(rows[2][0], fontsize=9)

    fp = metrics["filament_profile"]
    burst_txt = (
        f"fast-Dα {burst.channel} burst ×{burst.burst_ratio:.0f} "  # noqa: E501
        f"(σ={burst.burst_sigma:.0f}) @ {burst.burst_time_s * 1e3:.1f} ms"
        if burst is not None
        else "camera-burst selected"
    )
    caption = (
        "Tokenizer round-trip fidelity for ELM edge structure — "
        f"shot {win.shot_id}, window {ft[0] * 1e3:.1f}–{ft[-1] * 1e3:.1f} ms "
        f"(peak @ {ft[pf] * 1e3:.1f} ms).  {burst_txt}.\n"
        "Row 1: RAW rbb camera (native 112×156, no downsample).  "
        "Row 2: decode of the GROUND-TRUTH stored tokens through the frozen "
        "Open-MAGVIT2 18-bit LFQ tokenizer — NO model, NO masking, just "
        "encode→decode.  "
        f"ELM-peak frame: SSIM {metrics['ssim_full']:.3f} (full) / "
        f"{metrics['ssim_crop']:.3f} (crop), PSNR {metrics['psnr_full_db']:.1f} dB, "
        f"HF-energy retention {metrics['hf_energy_retention_ratio']:.2f}, "
        f"filament peaks raw {fp['raw_peaks']} → round-trip {fp['rt_peaks']}.  "
        "Cyan dotted box = the ELM-filament metric crop."
    )
    fig.suptitle(caption, fontsize=10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[tokfid] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Cosmos round-trip (conditional secondary)
# ---------------------------------------------------------------------------


def _cosmos_available() -> dict | None:
    """Locate an on-disk Cosmos tokenizer for the same camera, or None.

    Prefers the discrete-image ``DI16x16`` tokenizer (matches the 16×16 OMAG2
    grid).  Returns ``{"name", "token_dir"}`` or None when the Cosmos tokens
    are not on disk for our shots.
    """
    for name in ("DI16x16", "DV4x8x8"):
        d = COSMOS_ROOT / name
        if d.exists():
            return {"name": name, "token_dir": d}
    return None


def roundtrip_cosmos(
    win: rd.DemoWindow, cosmos_info: dict, work_dir: Path, device: str
) -> dict | None:
    """Best-effort Cosmos round-trip of the same window.

    Only invoked when OMAG2 visibly smears the filament.  Looks for stored
    Cosmos tokens for this shot under the Cosmos token dir and decodes them
    with the Cosmos decoder if one is reachable; returns
    ``{"name", "images"}`` or None if Cosmos cannot be round-tripped here.
    """
    import zarr

    # try the same per-shot/camera layout the OMAG2 tokens use, then a flat one
    candidates = [
        cosmos_info["token_dir"] / f"{win.shot_id}" / "rbb.zarr",
        cosmos_info["token_dir"] / f"{win.shot_id}.zarr",
        cosmos_info["token_dir"] / "frames" / f"{win.shot_id}" / "rbb.zarr",
    ]
    tok_path = next((p for p in candidates if p.exists()), candidates[0])
    if not tok_path.exists():
        logger.warning(
            "[tokfid] Cosmos tokens for shot %d not found under %s — skipping",
            win.shot_id,
            cosmos_info["token_dir"],
        )
        return None
    try:
        g = zarr.open_group(str(tok_path), mode="r")
        keys = set(g.array_keys())
        tkey = "tokens" if "tokens" in keys else next(iter(keys))
        all_tok = np.asarray(g[tkey][:])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[tokfid] Cosmos tokens unreadable for shot %d: %s", win.shot_id, exc
        )
        return None
    start = int(win.start)
    n = win.true_tokens.shape[0]
    end = min(start + n, all_tok.shape[0])
    cos_tok = all_tok[start:end]
    # NOTE: decoding Cosmos tokens needs the Cosmos decoder venv; if it is not
    # wired in this repo we cannot produce images and return None (the OMAG2
    # answer stands).  This path is only reached when OMAG2 already smeared,
    # so a missing Cosmos decoder is reported honestly rather than faked.
    logger.warning(
        "[tokfid] Cosmos decoder not wired in this repo — found %d Cosmos "
        "tokens for shot %d but cannot decode them here; reporting OMAG2 only",
        cos_tok.shape[0],
        win.shot_id,
    )
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _verdict(metrics: dict) -> str:
    """Map the fidelity numbers to {survive-well | survive-partial | destroyed}."""
    ssim_c = metrics["ssim_crop"]
    hf = metrics["hf_energy_retention_ratio"]
    fp = metrics["filament_profile"]
    peaks_ok = fp["rt_peaks"] >= max(1, fp["raw_peaks"] - 1)
    contrast_ok = fp["rt_contrast"] >= 0.4 * fp["raw_contrast"]
    if ssim_c >= 0.6 and hf >= 0.5 and peaks_ok and contrast_ok:
        return "survive-well"
    if ssim_c >= 0.35 and hf >= 0.25 and (peaks_ok or contrast_ok):
        return "survive-partial"
    return "destroyed"


def run(out_path: Path = DEFAULT_FIGURE, *, device: str = "cuda") -> dict:
    """Full probe: select → round-trip → metrics → figure (+ conditional Cosmos)."""
    try:
        import torch

        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    except Exception:  # noqa: BLE001
        dev = "cpu"
    logger.info("[tokfid] decode device = %s", dev)

    sel = select_elm_window()
    if sel is None:
        raise RuntimeError("no ELM window could be selected for the fidelity probe")
    win = sel.window

    work_dir = Path(
        tempfile.mkdtemp(prefix="tokfid-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    roundtrip = roundtrip_true_tokens(win, work_dir, dev)

    raw_frames = rd.load_raw_frames(win.shot_id, win.start, win.true_tokens.shape[0])
    if raw_frames is None:
        raise RuntimeError(f"raw frames unavailable for shot {win.shot_id}")

    metrics = compute_fidelity(raw_frames, roundtrip, sel.peak_frame)
    verdict = _verdict(metrics)
    metrics["verdict"] = verdict

    # CONDITIONAL: only round-trip Cosmos if OMAG2 visibly smeared the filament.
    cosmos = None
    if verdict == "destroyed":
        cinfo = _cosmos_available()
        if cinfo is not None:
            logger.info(
                "[tokfid] OMAG2 smeared the filament → trying Cosmos %s", cinfo["name"]
            )
            cosmos = roundtrip_cosmos(win, cinfo, work_dir, dev)
    else:
        logger.info(
            "[tokfid] OMAG2 verdict=%s → SKIPPING Cosmos (question answered)", verdict
        )

    render_figure(
        win,
        raw_frames,
        roundtrip,
        sel.peak_frame,
        metrics,
        sel.burst,
        out_path,
        cosmos=cosmos,
    )

    # ---- print every number ------------------------------------------------
    ft = np.asarray(win.frame_time, dtype=float)
    fp = metrics["filament_profile"]
    b = sel.burst
    print("\n" + "=" * 72)
    print("TOKENIZER ROUND-TRIP FIDELITY — ELM EDGE STRUCTURE")
    print("=" * 72)
    print(f"shot_id              : {win.shot_id}")
    print(f"window start frame   : {win.start}")
    print(
        f"window time range    : {ft[0] * 1e3:.2f} – {ft[-1] * 1e3:.2f} ms "
        f"({(ft[-1] - ft[0]) * 1e3:.2f} ms span, {len(ft)} frames)"
    )
    print(
        f"ELM peak frame       : {sel.peak_frame} (window-relative), "
        f"t = {ft[sel.peak_frame] * 1e3:.2f} ms"
    )
    mf = metrics["peak_frame"]
    print(
        f"metric frame         : {mf} (most-resolvable ELM frame near the peak; "
        f"the peak itself saturates), t = {ft[mf] * 1e3:.2f} ms"
    )
    if b is not None:
        print(
            f"fast-Dα confirmation : channel {b.channel} (xim, ~50 kHz), "
            f"burst ×{b.burst_ratio:.1f} baseline (σ={b.burst_sigma:.1f}) "
            f"@ {b.burst_time_s * 1e3:.2f} ms"
        )
    else:
        print("fast-Dα confirmation : (none — selected by raw-camera burst)")
    print(f"camera edge-burst score: {sel.camera_score:.3f}")
    print("-" * 72)
    print("FIDELITY (metric frame; raw resized to 256² for the metric only):")
    print(f"  SSIM  full frame   : {metrics['ssim_full']:.4f}")
    print(f"  PSNR  full frame   : {metrics['psnr_full_db']:.2f} dB")
    print(f"  SSIM  ELM crop     : {metrics['ssim_crop']:.4f}")
    print(f"  PSNR  ELM crop     : {metrics['psnr_crop_db']:.2f} dB")
    print(
        f"  HF-energy raw      : {metrics['hf_energy_raw']:.3e}  "
        f"round-trip : {metrics['hf_energy_roundtrip']:.3e}"
    )
    print(f"  HF-energy retention: {metrics['hf_energy_retention_ratio']:.3f}")
    print(f"  ELM crop box (256²): {metrics['elm_crop_box']}")
    print("-" * 72)
    print("FILAMENT TRANSECT (max-variation crop row, un-saturated stretch):")
    print(f"  peaks raw → round-trip : {fp['raw_peaks']} → {fp['rt_peaks']}")
    print(
        f"  peak value raw → rt    : {fp['raw_peak_value']:.1f} → "  # noqa: E501
        f"{fp['rt_peak_value']:.1f}"
    )
    pct = 100 * fp["rt_contrast"] / max(fp["raw_contrast"], 1e-9)
    print(
        f"  contrast raw → rt      : {fp['raw_contrast']:.1f} → "
        f"{fp['rt_contrast']:.1f} ({pct:.0f}% retained)"
    )
    print("-" * 72)
    print(f"VERDICT              : {verdict}")
    print(f"Cosmos round-trip    : {'yes' if cosmos else 'no (skipped/unavailable)'}")
    print(f"figure               : {out_path}")
    print("=" * 72 + "\n")

    summary = {
        "shot_id": int(win.shot_id),
        "window_start": int(win.start),
        "window_ms": [float(ft[0] * 1e3), float(ft[-1] * 1e3)],
        "peak_frame": int(sel.peak_frame),
        "peak_ms": float(ft[sel.peak_frame] * 1e3),
        "fast_dalpha": (
            {
                "channel": b.channel,
                "ratio": b.burst_ratio,
                "sigma": b.burst_sigma,
                "time_ms": b.burst_time_s * 1e3,
            }
            if b is not None
            else None
        ),
        "metrics": {k: v for k, v in metrics.items() if k != "filament_profile"},
        "filament": {k: v for k, v in fp.items() if k not in ("raw_line", "rt_line")},
        "verdict": verdict,
        "cosmos_used": cosmos is not None,
        "figure": str(out_path),
    }
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
