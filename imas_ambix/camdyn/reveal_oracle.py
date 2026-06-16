"""Is the camera-dynamics filament-placement floor INPUT-LIMITED or ALEATORIC?

The cheap, decisive eval-only gate that answers, BEFORE anyone spends a
GPU-day on a second-camera corpus + multi-view retrain: *would richer
observation (a second view) lower the single-view filament-placement floor?*

No training, no encode — the model forward is the SAME one already run for the
placement-trajectory probe, so this is effectively free (minutes of GPU).

Context
-------
The decontaminated placement metric
(:func:`placement_trajectory.filament_located_ssim` — per-pixel SSIM vs RAW
GT on FILAMENT cells only, residual over persistence) shows the dynamics MAP
decode beats persistence by a small, PLATEAUED margin (≈ +0.094 SSIM).  The
in-view per-cell calibration oracle (:func:`structure_fidelity.decode_oracle_joint`,
candidates = true-token XOR per-bit offsets) CANNOT relocate a filament — it
is a per-cell logit ceiling, and it scored BELOW MAP, hinting the floor is
*which-cell* uncertainty (a spatial / placement question), not per-cell
calibration.

This gate measures the UPPER BOUND on what more OBSERVATION could do, by
exploiting the fact that the model forward
(:func:`structure_fidelity._run_forward`) accepts an ARBITRARY ``(F,H,W)`` bool
``visible`` mask.

Two arms
--------
(A) REVEAL-k oracle.  For k ∈ {2,4,8}, flip that many post-frontier FILAMENT
    cells from masked→visible (filling in the in-view rbb TRUTH ids), re-run the
    SAME forward, and re-measure the ``(MAP − persistence)`` filament-residual +
    the per-cell predictive entropy on the STILL-masked filament cells.  This is
    the MOST favourable possible extra input — perfect pixels of the very cells
    we score — so it UPPER-BOUNDS any second view: a real second camera is
    occluded, coarser and mistimed, so if a PERFECT in-view reveal does not move
    the residual on the neighbouring cells, a real second view cannot.
    Decisive-to-KILL.

(B) SECOND-CROP proxy.  Make a second DISJOINT rbb sub-window visible (an
    upper-frame crop that does NOT overlap the scored lower-edge filament band)
    across all frames, alongside the frontier context; re-measure whether
    cross-region in-view fusion lowers primary-region (filament) placement
    uncertainty.  This is the optimistic co-registered, co-temporal bound for a
    second view (a real second camera adds misalignment and cadence loss this
    crop does not).

Pre-registered threshold (printed BEFORE any number)
----------------------------------------------------
INPUT-LIMITED iff revealing k=4 in-view filament cells raises the
``(MAP − persistence)`` filament-residual by > +0.02 SSIM AND drops the
remaining still-masked filament-cell entropy by > 10 % relative.  Otherwise →
ALEATORIC / model floor (input enrichment will not help).

Kill / license asymmetry (stated honestly in the verdict)
---------------------------------------------------------
A NULL here KILLS the second-camera idea (a perfect in-view reveal is a strict
upper bound on a real, degraded second view).  A POSITIVE only LICENSES SCOPING
it — it does not prove a real ``rba`` would deliver the gain, because the real
camera loses cadence and adds occlusion / misalignment this in-view reveal does
not.

Run (eval-only, GPU node)::

    .venv/bin/python -m imas_ambix.camdyn.reveal_oracle

Smoke (1 window, k=4 only) before the full sweep::

    .venv/bin/python -m imas_ambix.camdyn.reveal_oracle --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from imas_ambix.camdyn import placement_trajectory as pt
from imas_ambix.camdyn import recon_movie as mv
from imas_ambix.camdyn import reconstruction_demo as rd
from imas_ambix.camdyn import structure_fidelity as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: The placement-probe windows the trajectory gate ran on.
SHOTS = (24446, 24065)
FRONTIER = sf.FRONTIER

#: The single checkpoint this gate reads (the dynamics arm headline).
CKPT = Path("/work/projects/imas_gpu/mast-checkpoints/camdyn/cap_v1_dynamics/final.pt")

#: Reveal counts swept.  k=4 is the PRE-REGISTERED decision point.
REVEAL_KS = (2, 4, 8)

DEFAULT_FIGURE = Path("docs/figures/camera-dynamics-wm/fig-cdw-reveal-oracle.png")
DEFAULT_JSON = Path("imas_ambix/camdyn/artifacts/reveal_oracle.json")

# --- pre-registered threshold (stated verbatim, printed before the numbers) --
RESIDUAL_GAIN_THRESHOLD = 0.02  # +SSIM on the (MAP-persistence) filament residual
ENTROPY_DROP_THRESHOLD = 0.10  # >10% relative fall in still-masked filament entropy
DECISION_K = 4  # the pre-registered reveal count the verdict is read at

PREREG_THRESHOLD = (
    "PRE-REGISTERED THRESHOLD: INPUT-LIMITED iff revealing k=4 in-view filament "
    "cells raises the (MAP - persistence) filament-residual by > +0.02 SSIM AND "
    "drops the remaining still-masked filament-cell entropy by > 10% relative. "
    "Otherwise -> ALEATORIC/model floor (input enrichment will not help)."
)


# ---------------------------------------------------------------------------
# Second-crop geometry — a DISJOINT upper-frame window
# ---------------------------------------------------------------------------


def second_crop_mask(n_frames: int) -> np.ndarray:
    """A second DISJOINT rbb sub-window, visible across all frames.

    The scored filament cells live in the lower-edge band (token rows 12-15).
    This crop is a central UPPER-frame window (rows ~2-8) that does NOT touch
    that band, so making it visible adds genuinely cross-region in-view
    information rather than trivially revealing the cells we score.  It is the
    optimistic co-registered / co-temporal stand-in for a second camera view.
    """
    from imas_ambix.camdyn.masking import _window_mask

    win = _window_mask(
        (mv.GRID_H, mv.GRID_W), centre=(5.0, 8.0), half_h=3.0, half_w=4.0
    )
    # hard guard: never overlap the lower-edge filament band (rows GRID_H-4..)
    win[mv.GRID_H - 4 :, :] = False
    return np.broadcast_to(win, (n_frames, mv.GRID_H, mv.GRID_W)).copy()


# ---------------------------------------------------------------------------
# Filament-cell ranking — pick the k strongest cells to reveal
# ---------------------------------------------------------------------------


def _ranked_filament_cells(raw_native: np.ndarray, frontier: int) -> np.ndarray:
    """``(r, c)`` filament cells, ordered by GT high-frequency striation energy.

    Same edge-band / top-HF-tercile split the placement metric uses, but with the
    cells ordered (strongest first) so REVEAL-k flips the k brightest filaments —
    the most-informative cells a real second view would most plausibly resolve.
    """
    from PIL import Image

    gh, gw = mv.GRID_H, mv.GRID_W
    post = list(range(frontier, raw_native.shape[0]))
    gt = np.stack(
        [
            np.asarray(
                Image.fromarray(
                    np.clip(raw_native[fi].astype(np.float64), 0, None).astype(np.uint8)
                    if raw_native[fi].dtype != np.uint8
                    else raw_native[fi]
                ).resize((gw, gh), Image.BILINEAR),
                dtype=np.float64,
            )
            for fi in post
        ]
    )
    edge_band = np.zeros((gh, gw), dtype=bool)
    edge_band[gh - 4 :, :] = True
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
    hf_hi = np.percentile(edge_hf, 66.0)
    filament = edge_band & (hf >= hf_hi)
    rc = np.argwhere(filament)
    order = np.argsort(-hf[filament])  # strongest first
    return rc[order]


# ---------------------------------------------------------------------------
# Per-window reveal-k + second-crop evaluation
# ---------------------------------------------------------------------------


def evaluate_reveal(
    model,
    torch,
    device,
    win: rd.DemoWindow,
    cond_stats,
    work_dir: Path,
    *,
    frontier: int = FRONTIER,
    reveal_ks=REVEAL_KS,
    seed0: int = sf.SEED0,
) -> dict:
    """Reveal-k + second-crop placement read for ONE (model, window).

    Baseline forward = the frontier scenario (the placement-trajectory state).
    For each k we flip the k brightest post-frontier filament cells from
    masked->visible (filling the TRUTH ids), re-run the SAME forward, and
    re-measure the (MAP-persistence) filament residual + the predictive entropy
    on the STILL-masked filament cells.  The second-crop arm reveals a disjoint
    upper-frame window instead.
    """
    cond_t = sf._conditioning_tensors(torch, device, win, cond_stats)
    cv_t, cm_t, dt_t = cond_t
    true_tokens = win.true_tokens
    n_frames = true_tokens.shape[0]
    post = list(range(frontier, n_frames))

    gt256 = sf._raw_gt_256(win)
    raw_native = rd.load_raw_frames(win.shot_id, win.start, n_frames)
    if gt256 is None or raw_native is None:
        return {"error": "no raw GT frames", "shot_id": int(win.shot_id)}

    cells = pt.filament_cell_mask(raw_native, frontier)
    if not cells.get("calibrated"):
        return {
            "error": cells.get("reason", "uncalibrated"),
            "shot_id": int(win.shot_id),
        }
    filament_mask = cells["filament"]  # (gh, gw) bool
    ranked = _ranked_filament_cells(raw_native, frontier)
    n_fil = int(filament_mask.sum())

    base_visible = rd.scenario_mask(sf.SCENARIO, n_frames, frontier)

    persist_tok = mv.persistence_tokens(true_tokens, frontier)

    def _residual_and_entropy(visible, scored_cell_mask):
        """(MAP-persistence) filament-residual + filament entropy on scored cells.

        ``scored_cell_mask`` selects the (gh,gw) cells the residual/entropy are
        read on — for reveal-k this EXCLUDES the revealed cells so we measure the
        information transferred to the neighbours, not the trivially-correct
        revealed pixels.
        """
        bit_logits = sf._run_forward(
            model, torch, device, true_tokens, visible, cv_t, cm_t, dt_t
        )
        map_tok = sf.decode_map(bit_logits)
        if not scored_cell_mask.any():
            return {
                "residual_map_vs_persistence": float("nan"),
                "filament_cell_entropy": float("nan"),
                "n_scored_cells": 0,
            }
        scored_px = pt._cell_mask_to_pixels(scored_cell_mask)
        grids = {"map": map_tok, "persistence": persist_tok}
        dev_str = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
        decoded = sf._decode_grids(grids, work_dir, dev_str)
        ssim_map = pt.filament_located_ssim(decoded["map"], gt256, post, scored_px)
        ssim_persist = pt.filament_located_ssim(
            decoded["persistence"], gt256, post, scored_px
        )
        ent = sf.per_bit_entropy(bit_logits)
        ent_post = ent[np.asarray(post, dtype=int)].mean(axis=0)  # (gh,gw)
        return {
            "residual_map_vs_persistence": float(ssim_map - ssim_persist),
            "map_filament_ssim": float(ssim_map),
            "persistence_filament_ssim": float(ssim_persist),
            "filament_cell_entropy": float(ent_post[scored_cell_mask].mean()),
            "n_scored_cells": int(scored_cell_mask.sum()),
        }

    # --- per-cell oracle ceiling on the SAME (full) filament cells -----------
    # read against the baseline forward so the reveal-k spatial gain is set
    # against the known per-cell logit ceiling.
    base_logits = sf._run_forward(
        model, torch, device, true_tokens, base_visible, cv_t, cm_t, dt_t
    )
    base_map = sf.decode_map(base_logits)
    oracle_tok = sf.decode_oracle_joint(
        base_logits, true_tokens, temperature=1.0, rng=np.random.default_rng(seed0)
    )
    fil_px_full = pt._cell_mask_to_pixels(filament_mask)
    grids0 = {"map": base_map, "persistence": persist_tok, "oracle_joint": oracle_tok}
    dev_str = "cuda" if getattr(device, "type", str(device)) == "cuda" else "cpu"
    decoded0 = sf._decode_grids(grids0, work_dir, dev_str)
    ssim_map0 = pt.filament_located_ssim(decoded0["map"], gt256, post, fil_px_full)
    ssim_persist0 = pt.filament_located_ssim(
        decoded0["persistence"], gt256, post, fil_px_full
    )
    ssim_oracle0 = pt.filament_located_ssim(
        decoded0["oracle_joint"], gt256, post, fil_px_full
    )

    # baseline residual + entropy on the FULL filament set (reference, k=0)
    base_read = _residual_and_entropy(base_visible, filament_mask)
    base_residual = base_read["residual_map_vs_persistence"]
    base_entropy = base_read["filament_cell_entropy"]

    # --- REVEAL-k arm --------------------------------------------------------
    reveal: dict = {}
    for k in reveal_ks:
        k_eff = min(int(k), max(0, ranked.shape[0] - 1))  # leave >=1 cell still masked
        revealed = np.zeros_like(filament_mask)
        for r, c in ranked[:k_eff]:
            revealed[int(r), int(c)] = True
        still_masked = filament_mask & (~revealed)
        vis_k = base_visible.copy()
        # flip the revealed filament cells to visible across ALL post-frontier
        # frames (the model then conditions on the TRUE tokens at those cells).
        for fi in post:
            vis_k[fi][revealed] = True
        rd_k = _residual_and_entropy(vis_k, still_masked)
        # baseline read on the SAME still-masked cells, for an honest k-vs-0 diff
        base_on_still = _residual_and_entropy(base_visible, still_masked)
        rel_drop = (
            float(
                (base_on_still["filament_cell_entropy"] - rd_k["filament_cell_entropy"])
                / max(base_on_still["filament_cell_entropy"], 1e-9)
            )
            if np.isfinite(base_on_still["filament_cell_entropy"])
            else float("nan")
        )
        reveal[str(k)] = {
            "k_requested": int(k),
            "k_revealed": int(k_eff),
            "n_still_masked_cells": int(still_masked.sum()),
            "reveal": rd_k,
            "baseline_on_same_cells": base_on_still,
            "residual_gain_vs_baseline": float(
                rd_k["residual_map_vs_persistence"]
                - base_on_still["residual_map_vs_persistence"]
            ),
            "entropy_relative_drop": rel_drop,
        }

    # --- SECOND-CROP arm -----------------------------------------------------
    crop = second_crop_mask(n_frames)
    vis_crop = base_visible.copy()
    for fi in post:
        vis_crop[fi][crop[fi]] = True
    # the crop reveals TRUE tokens at a disjoint upper-frame window; the scored
    # filament cells stay fully masked, so any gain is cross-region fusion.
    crop_read = _residual_and_entropy(vis_crop, filament_mask)
    second_crop = {
        "n_crop_cells": int(crop[frontier].sum()),
        "read": crop_read,
        "baseline_full_filament": {
            "residual_map_vs_persistence": base_residual,
            "filament_cell_entropy": base_entropy,
        },
        "residual_gain_vs_baseline": float(
            crop_read["residual_map_vs_persistence"] - base_residual
        ),
        "entropy_relative_drop": float(
            (base_entropy - crop_read["filament_cell_entropy"])
            / max(base_entropy, 1e-9)
        )
        if np.isfinite(base_entropy)
        else float("nan"),
    }

    return {
        "shot_id": int(win.shot_id),
        "window_ms": [
            float(win.frame_time[0] * 1e3),
            float(win.frame_time[-1] * 1e3),
        ],
        "n_filament_cells": n_fil,
        "baseline_full_filament": {
            "map_filament_ssim": float(ssim_map0),
            "persistence_filament_ssim": float(ssim_persist0),
            "oracle_filament_ssim": float(ssim_oracle0),
            "residual_map_vs_persistence": float(ssim_map0 - ssim_persist0),
            "residual_oracle_vs_persistence": float(ssim_oracle0 - ssim_persist0),
            "filament_cell_entropy": base_entropy,
        },
        "reveal_k": reveal,
        "second_crop": second_crop,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_reveal(
    *,
    device: str = "cuda",
    shots=SHOTS,
    reveal_ks=REVEAL_KS,
    smoke: bool = False,
) -> dict:
    """Reveal-k + second-crop gate across the placement-probe windows."""
    import contextlib

    import torch

    from imas_ambix.camdyn.arm_compare import _load_arm

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("[reveal] device = %s", dev)

    if smoke:
        shots = (shots[0],)
        reveal_ks = (DECISION_K,)

    windows = pt._select_windows(shots)
    if not windows:
        raise RuntimeError("no flat-top edge windows could be selected")
    logger.info(
        "[reveal] %d window(s) on shots %s; reveal-k=%s",
        len(windows),
        [w.shot_id for w in windows],
        list(reveal_ks),
    )

    work_dir = Path(
        tempfile.mkdtemp(prefix="reveal-", dir=os.environ.get("TMPDIR", "/tmp"))
    )

    logger.info("[reveal] loading dynamics arm final.pt")
    model, _cfg, cond_stats = _load_arm(CKPT, torch, dev)
    per_window: list[dict] = []
    try:
        for win in windows:
            res = evaluate_reveal(
                model, torch, dev, win, cond_stats, work_dir, reveal_ks=reveal_ks
            )
            per_window.append(res)
            if "error" not in res:
                rk = res["reveal_k"].get(str(DECISION_K), {})
                logger.info(
                    "[reveal] shot %d  base-resid=%+.4f  k=%d resid-gain=%+.4f "
                    "ent-drop=%.1f%%  crop resid-gain=%+.4f",
                    res["shot_id"],
                    res["baseline_full_filament"]["residual_map_vs_persistence"],
                    DECISION_K,
                    rk.get("residual_gain_vs_baseline", float("nan")),
                    100.0 * rk.get("entropy_relative_drop", float("nan")),
                    res["second_crop"]["residual_gain_vs_baseline"],
                )
    finally:
        with contextlib.suppress(Exception):
            del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    ok = [r for r in per_window if "error" not in r]
    verdict = reveal_verdict(ok, reveal_ks=reveal_ks)
    return {
        "task": (
            "in-view reveal-k oracle + second-crop proxy gate — upper-bounds the "
            "second-view payoff on the decontaminated filament-placement floor"
        ),
        "preregistered_threshold": PREREG_THRESHOLD,
        "residual_gain_threshold": RESIDUAL_GAIN_THRESHOLD,
        "entropy_drop_threshold": ENTROPY_DROP_THRESHOLD,
        "decision_k": DECISION_K,
        "checkpoint": str(CKPT),
        "scenario": sf.SCENARIO,
        "frontier_frame": FRONTIER,
        "shots": [int(r["shot_id"]) for r in ok],
        "reveal_ks": list(reveal_ks),
        "kill_license_asymmetry": (
            "A NULL here KILLS the second-camera idea (a perfect in-view reveal is "
            "a strict upper bound on a real, degraded second view). A POSITIVE only "
            "LICENSES scoping it — a real rba loses cadence and adds occlusion / "
            "misalignment this in-view reveal does not."
        ),
        "per_window": per_window,
        "verdict": verdict,
    }


def reveal_verdict(window_results: list[dict], *, reveal_ks=REVEAL_KS) -> dict:
    """INPUT-LIMITED vs ALEATORIC-FLOOR, read at the pre-registered k=4 point."""
    if not window_results:
        return {"verdict": "NO-DATA", "detail": "no valid windows"}

    def _mean_k(k, path):
        vals = []
        for r in window_results:
            cur = r["reveal_k"].get(str(k), {})
            for key in path:
                cur = cur.get(key, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, (int, float)) and np.isfinite(cur):
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else float("nan")

    k = DECISION_K
    resid_gain = _mean_k(k, ["residual_gain_vs_baseline"])
    ent_drop = _mean_k(k, ["entropy_relative_drop"])

    resid_pass = bool(np.isfinite(resid_gain) and resid_gain > RESIDUAL_GAIN_THRESHOLD)
    ent_pass = bool(np.isfinite(ent_drop) and ent_drop > ENTROPY_DROP_THRESHOLD)
    input_limited = resid_pass and ent_pass

    if input_limited:
        verdict = "INPUT-LIMITED"
        detail = (
            f"revealing k={k} in-view filament cells raised the (MAP-persistence) "
            f"residual by {resid_gain:+.4f} SSIM (> {RESIDUAL_GAIN_THRESHOLD:+.3f}) "
            f"AND dropped still-masked filament entropy by {100 * ent_drop:.1f}% "
            f"(> {100 * ENTROPY_DROP_THRESHOLD:.0f}%). The floor is which-cell "
            "uncertainty more observation can resolve -> a second view / "
            "mode-structure conditioning is worth SCOPING (license only; a real "
            "rba loses cadence + adds occlusion this reveal does not)."
        )
    else:
        verdict = "ALEATORIC-FLOOR"
        detail = (
            f"revealing k={k} in-view filament cells moved the (MAP-persistence) "
            f"residual by only {resid_gain:+.4f} SSIM "
            f"({'>' if resid_pass else 'NOT >'} {RESIDUAL_GAIN_THRESHOLD:+.3f}) and "
            f"the still-masked filament entropy by {100 * ent_drop:.1f}% "
            f"({'>' if ent_pass else 'NOT >'} {100 * ENTROPY_DROP_THRESHOLD:.0f}%). "
            "A PERFECT in-view reveal — the strict upper bound on any real second "
            "view — does not move the placement floor: it is ALEATORIC / "
            "model-limited within this view. This KILLS the second-camera idea; "
            "the lever is the temporal-evolution objective."
        )

    return {
        "verdict": verdict,
        "detail": detail,
        "decision_k": k,
        "mean_residual_gain_at_k": resid_gain,
        "mean_entropy_relative_drop_at_k": ent_drop,
        "residual_threshold_passed": resid_pass,
        "entropy_threshold_passed": ent_pass,
        "n_windows": len(window_results),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_figure(summary: dict, out_path: Path) -> None:
    """Residual-gain + remaining-entropy-drop vs k, both arms, per window.

    Left: the (MAP-persistence) filament-residual GAIN vs k for the reveal arm
    and the second-crop proxy, with the pre-registered +0.02 SSIM line.  Right:
    the relative drop in still-masked filament-cell entropy vs k, with the 10%
    line.  The verdict is in the suptitle.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in summary["per_window"] if "error" not in r]
    reveal_ks = summary["reveal_ks"]
    verdict = summary["verdict"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    cols = [cmap(0.15 + 0.6 * i / max(1, len(ok) - 1)) for i in range(len(ok))]

    for wi, r in enumerate(ok):
        sid = r["shot_id"]
        col = cols[wi]
        ks = [int(k) for k in reveal_ks]
        rg = [r["reveal_k"][str(k)]["residual_gain_vs_baseline"] for k in reveal_ks]
        ed = [100.0 * r["reveal_k"][str(k)]["entropy_relative_drop"] for k in reveal_ks]
        ax1.plot(ks, rg, "-o", color=col, ms=6, label=f"shot {sid} reveal-k")
        ax2.plot(ks, ed, "-o", color=col, ms=6, label=f"shot {sid} reveal-k")
        # second-crop proxy (a single disjoint-window reveal — plot as a level)
        sc_rg = r["second_crop"]["residual_gain_vs_baseline"]
        sc_ed = 100.0 * r["second_crop"]["entropy_relative_drop"]
        ax1.axhline(
            sc_rg,
            color=col,
            ls="--",
            alpha=0.55,
            lw=1.3,
            label=f"shot {sid} second-crop",
        )
        ax2.axhline(sc_ed, color=col, ls="--", alpha=0.55, lw=1.3)

    ax1.axhline(
        summary["residual_gain_threshold"],
        color="k",
        ls=":",
        lw=1.5,
        label=f"threshold +{summary['residual_gain_threshold']:.2f}",
    )
    ax1.axhline(0.0, color="grey", lw=0.7, alpha=0.6)
    ax1.set_xlabel("k filament cells revealed (perfect in-view pixels)")
    ax1.set_ylabel("(MAP - persistence) residual GAIN vs baseline")
    ax1.set_title(
        "placement-residual gain from revealing k cells\n"
        "(measured on the STILL-masked filament cells)"
    )
    ax1.set_xticks([int(k) for k in reveal_ks])
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)

    ax2.axhline(
        100.0 * summary["entropy_drop_threshold"],
        color="k",
        ls=":",
        lw=1.5,
        label=f"threshold {100 * summary['entropy_drop_threshold']:.0f}%",
    )
    ax2.axhline(0.0, color="grey", lw=0.7, alpha=0.6)
    ax2.set_xlabel("k filament cells revealed (perfect in-view pixels)")
    ax2.set_ylabel("still-masked filament entropy DROP (% relative)")
    ax2.set_title("remaining filament-cell uncertainty\nresolved by the reveal")
    ax2.set_xticks([int(k) for k in reveal_ks])
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    fig.suptitle(
        "camera-dynamics-wm second-view payoff gate — VERDICT: "
        f"{verdict['verdict']}\n{verdict['detail']}",
        fontsize=9.5,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[reveal] wrote %s", out_path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(summary: dict) -> None:
    print("\n" + "=" * 84)
    print(summary["preregistered_threshold"])
    print("=" * 84)
    print("SECOND-VIEW PAYOFF GATE — in-view reveal-k oracle + second-crop proxy")
    print("=" * 84)
    print(f"checkpoint : {summary['checkpoint']}")
    print(f"shots      : {summary['shots']}  (frontier @ f{summary['frontier_frame']})")
    for r in summary["per_window"]:
        if "error" in r:
            print(f"\n[shot {r.get('shot_id')}] ERROR: {r['error']}")
            continue
        b = r["baseline_full_filament"]
        print(f"\n[shot {r['shot_id']}]  n_filament_cells={r['n_filament_cells']}")
        print(
            f"  baseline  MAP-persist={b['residual_map_vs_persistence']:+.4f}  "
            f"oracle-persist={b['residual_oracle_vs_persistence']:+.4f}  "
            f"fil_entropy={b['filament_cell_entropy']:.3f}"
        )
        print(f"  {'k':>3}{'k_rev':>6}{'still':>6}{'resid-gain':>12}{'ent-drop%':>11}")
        for k in summary["reveal_ks"]:
            rk = r["reveal_k"][str(k)]
            print(
                f"  {k:>3}{rk['k_revealed']:>6}{rk['n_still_masked_cells']:>6}"
                f"{rk['residual_gain_vs_baseline']:>+12.4f}"
                f"{100.0 * rk['entropy_relative_drop']:>+11.1f}"
            )
        sc = r["second_crop"]
        print(
            f"  second-crop ({sc['n_crop_cells']} cells)  "
            f"resid-gain={sc['residual_gain_vs_baseline']:+.4f}  "
            f"ent-drop={100.0 * sc['entropy_relative_drop']:+.1f}%"
        )
    v = summary["verdict"]
    print("-" * 84)
    print(f"VERDICT: {v['verdict']}  (read at pre-registered k={v.get('decision_k')})")
    print(
        f"  mean residual gain @k={v.get('decision_k')}: "
        f"{v.get('mean_residual_gain_at_k'):+.4f} "
        f"(threshold +{summary['residual_gain_threshold']:.2f}, "
        f"passed={v.get('residual_threshold_passed')})"
    )
    print(
        f"  mean entropy drop  @k={v.get('decision_k')}: "
        f"{100.0 * v.get('mean_entropy_relative_drop_at_k'):.1f}% "
        f"(threshold {100 * summary['entropy_drop_threshold']:.0f}%, "
        f"passed={v.get('entropy_threshold_passed')})"
    )
    print(f"  {v['detail']}")
    print("=" * 84 + "\n")


def run(
    *,
    device: str = "cuda",
    figure: Path = DEFAULT_FIGURE,
    json_path: Path = DEFAULT_JSON,
    smoke: bool = False,
) -> dict:
    # Pre-registered threshold printed BEFORE any number (critic requirement).
    print("\n" + PREREG_THRESHOLD + "\n")

    summary = run_reveal(device=device, smoke=smoke)

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("[reveal] written %s", json_path)

    if not smoke:
        render_figure(summary, figure)
        summary["figure"] = str(figure)

    _print_report(summary)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--figure", default=str(DEFAULT_FIGURE))
    p.add_argument("--json", default=str(DEFAULT_JSON))
    p.add_argument("--smoke", action="store_true", help="1 window, k=4 only, no figure")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        device=args.device,
        figure=Path(args.figure),
        json_path=Path(args.json),
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
