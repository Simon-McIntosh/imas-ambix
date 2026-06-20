"""Figures for the signal-conditioned camera world model (held-out shots).

Two phases, run as two subcommands so the GPU is touched ONCE:

``gpu``    — Phase A (WM venv, betelgeuse, 1 GPU).  Load the signal-conditioned
             checkpoint ONCE (AGENTS.md §2b), and for each held-out shot build
             three role grids — ground truth, teacher-forced reconstruction, and
             the autoregressive DREAM at that shot's best sampling setting (read
             from the rescore verdict, falling back to T=0.9/top_p=0.95).  Decode
             all roles for a shot in ONE frozen Open-MAGVIT2 VQ pass and save the
             decoded ``(F, 256, 256, 3)`` image stacks to a per-shot ``.npz`` on
             GPFS.  No matplotlib here — just the GPU-bound generation + decode.

``render`` — Phase B (no GPU; login or compute).  Read the saved image bundles +
             the rescore verdict JSONs and write the deliverables to
             ``docs/figures/playable-plasma-wm-v0/``:
               * per-shot GT-vs-dream and GT-vs-reconstruction GIFs;
               * a CRPS-ratio-vs-persistence bar chart (M1 vs M2, per shot);
               * a horizon-drift plot (24f vs 26f CRPS ratio per shot);
               * a motion/collapse plot (argmax vs sampling per shot).

The two phases share the bundle format (a per-shot ``.npz`` with one image stack
per role + a meta JSON), so the GPU phase can run on betelgeuse and the render
phase anywhere with the bundles on the shared filesystem.

The dream is scored elsewhere (``sampling_rescore``); the verdict that the model
BEATS persistence is the CRPS-ratio < 1 in those JSONs, NOT the look of the GIF
(a model can look coherent while losing — the v1 camera-only lesson).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed locations (the converged M2 run + the M1 baseline + the verdicts).
# ---------------------------------------------------------------------------

#: The converged signal-conditioned checkpoint (step 12000).
M2_CKPT = Path(
    "/work/projects/imas_gpu/worldmodel/ckpt/spacetime-corruption-1220391/best.pt"
)

#: The M2 rescore verdicts — per-shot best sampling setting + CRPS + motion.
M2_VERDICT_DIR = Path(
    "/work/projects/imas_gpu/worldmodel/ckpt/"
    "spacetime-corruption-1220391/corruption_rescore_s1000"
)
M2_VERDICT_H16 = M2_VERDICT_DIR / "verdict_h16.json"  # matched 24f window
M2_VERDICT_H26 = M2_VERDICT_DIR / "verdict_h26.json"  # longer 26f window

#: The M1 baseline rescore verdict (signal-conditioned, step 7000).
M1_VERDICT = Path(
    "/work/projects/imas_gpu/worldmodel/ckpt/"
    "spacetime-v2-1219852/sampling_rescore/verdict.json"
)

#: Where the GPU phase writes the decoded image bundles (GPFS, NOT git-tracked).
DECODE_DIR = Path("/work/projects/imas_gpu/worldmodel/playable_plasma_figs")

#: The held-out shots (bright 18502/18505, dim 18503/18504).
HELDOUT_SHOTS = (18502, 18503, 18504, 18505)
BRIGHT_SHOTS = {18502, 18505}

#: The matched evaluation window — MUST match the rescore + training window.
WINDOW = dict(n_frames=24, n_plan=8, context_frames=8, frame_stride=1, n_signal_steps=4)

#: Default sampling setting if a shot has no verdict entry.
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.95

#: GIF playback rate.
FPS = 8

#: Where the rendered assets land (git-tracked).
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "figures" / "playable-plasma-wm-v0"

DPI = 150

# ---------------------------------------------------------------------------
# Tufte palette (light bg, dark text, thin rules, label directly).
# ---------------------------------------------------------------------------
DARK = "#1a1a1a"  # body text / axes
RULE = "#999999"  # thin rules / tick marks
ACCENT_M2 = "#2980b9"  # M2 (the new model) — cool blue
ACCENT_M1 = "#bdc3c7"  # M1 baseline — muted grey
ACCENT_PERS = "#7f8c8d"  # persistence reference
ACCENT_WIN = "#27ae60"  # beats / GT-like motion — green
ACCENT_LOSS = "#c0392b"  # loses / collapsed — red
WIN_FILL = "#f0f4f8"  # context-region shading


# ===========================================================================
# Verdict helpers (shared by both phases)
# ===========================================================================
def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _per_shot_rows(verdict_path: Path) -> dict[int, dict]:
    """Map ``shot_id -> summary per-shot row`` for a rescore verdict JSON."""
    rows = _load_json(verdict_path)["summary"]["per_shot"]
    return {int(r["shot_id"]): r for r in rows}


def best_setting_for_shot(verdict: dict, shot_id: int) -> tuple[float, float, str]:
    """Return ``(temperature, top_p, setting_name)`` for a shot's best sampling.

    Reads the summary's per-shot ``best_sampling_setting`` (chosen by lowest CRPS
    ratio) and parses its ``T=<t>_p=<p>`` name.  Falls back to the project default
    if the shot is absent or the name does not parse.
    """
    per_shot = verdict.get("summary", {}).get("per_shot", [])
    rows = {int(r["shot_id"]): r for r in per_shot}
    row = rows.get(int(shot_id))
    name = row.get("best_sampling_setting") if row else None
    if name and name.startswith("T=") and "_p=" in name:
        try:
            t_str, p_str = name[2:].split("_p=")
            return float(t_str), float(p_str), name
        except ValueError:
            pass
    default_name = f"T={DEFAULT_TEMPERATURE}_p={DEFAULT_TOP_P}"
    return DEFAULT_TEMPERATURE, DEFAULT_TOP_P, default_name


# ===========================================================================
# PHASE A — generate + decode (GPU)
# ===========================================================================
def _generate_and_decode_shot(
    *,
    model,
    payload: dict,
    shot_id: int,
    temperature: float,
    top_p: float,
    setting_name: str,
    work_dir: Path,
    token_root: Path | None,
    device: str,
    chunk: int,
) -> dict:
    """Build GT + teacher-forced + dream grids for one shot and decode them once.

    Reuses the v2 rollout primitives (signal-conditioned) and the single-pass VQ
    decode from :mod:`imas_ambix.worldmodel.sampling_rescore`.  The dream uses the
    shot's best ``(temperature, top_p)`` with the SAME per-shot reproducible seed
    convention as the rescore so the rendered dream matches the scored object.
    """
    import torch

    from imas_ambix.worldmodel.sampling_rescore import decode_roles
    from imas_ambix.worldmodel.spacetime_dataset import (
        GRID_H,
        GRID_W,
        REFERENCE_CAMERA,
        local_to_store,
    )
    from imas_ambix.worldmodel.spacetime_dataset_v2 import (
        SpacetimeWindowConfig,
        assemble_signal_window,
        default_signal_modalities,
    )
    from imas_ambix.worldmodel.spacetime_train_v2 import (
        autoregressive_signal_dream,
        teacher_forced_signal_frames,
    )

    window = SpacetimeWindowConfig(
        n_frames=WINDOW["n_frames"],
        n_plan=WINDOW["n_plan"],
        context_frames=WINDOW["context_frames"],
        frame_stride=WINDOW["frame_stride"],
    )
    modalities = default_signal_modalities()
    model_streams = [st.name for st in model.config.signal_streams]
    dev = torch.device(device)

    sample = assemble_signal_window(
        int(shot_id),
        window,
        modalities,
        int(WINDOW["n_signal_steps"]),
        camera=REFERENCE_CAMERA,
        token_root=token_root,
    )
    present = sorted(sample.signals.keys())
    ctx = int(sample.context_frames)
    n_frames = int(sample.frames.shape[0])

    gt_local = np.asarray(sample.frames, dtype=np.int64).reshape(
        n_frames, GRID_H, GRID_W
    )

    # teacher-forced reconstruction (deterministic).
    tf_local = teacher_forced_signal_frames(
        model, sample, stream_names=model_streams, device=dev
    ).reshape(n_frames, GRID_H, GRID_W)

    # the dream at the shot's best (temperature, top_p) — reproducible seed.
    # Mirror sampling_rescore.generate_rollouts: best setting == grid index of the
    # winning (T, p); member 0.  We reproduce its seed for the winning grid point
    # so the rendered dream is one of the scored ensemble members, not a new draw.
    from imas_ambix.worldmodel.sampling_rescore import default_grid

    grid = default_grid()
    try:
        ti = grid.index((float(temperature), float(top_p)))
    except ValueError:
        ti = 0
    seed = (int(shot_id) * 100003) ^ (ti * 1009) ^ (0 * 31) ^ 0x5A5A
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    dream_local = autoregressive_signal_dream(
        model,
        sample,
        stream_names=model_streams,
        device=dev,
        temperature=float(temperature),
        top_p=float(top_p),
        generator=gen,
        chunk=int(chunk),
    ).reshape(n_frames, GRID_H, GRID_W)

    # honesty token-space readouts over the forecast window.
    tf_mismatch = float((tf_local[ctx:] != gt_local[ctx:]).mean())
    dream_mismatch = float((dream_local[ctx:] != gt_local[ctx:]).mean())
    last_ctx = dream_local[ctx - 1]
    dream_change = float((dream_local[ctx:] != last_ctx[None]).mean())

    # decode all three roles in ONE VQ pass (store-id space; decode subtracts
    # REGISTRY_OFFSET itself).
    grids_by_role = {
        "gt": local_to_store(gt_local),
        "teacher_forced": local_to_store(tf_local),
        "dream": local_to_store(dream_local),
    }
    roles_index = [{"role": "gt"}, {"role": "teacher_forced"}, {"role": "dream"}]
    decoded = decode_roles(grids_by_role, roles_index, work_dir=work_dir, device=device)

    meta = {
        "shot_id": int(shot_id),
        "is_bright": bool(shot_id in BRIGHT_SHOTS),
        "checkpoint": str(M2_CKPT),
        "checkpoint_step": int((payload or {}).get("step", -1)),
        "camera": REFERENCE_CAMERA,
        "n_frames": int(n_frames),
        "context_frames": int(ctx),
        "frame_time": np.asarray(sample.frame_time, dtype=np.float64).tolist(),
        "dream_setting": setting_name,
        "dream_temperature": float(temperature),
        "dream_top_p": float(top_p),
        "dream_seed": int(seed),
        "present_streams": present,
        "signal_streams": model_streams,
        "teacher_forced_token_mismatch": tf_mismatch,
        "dream_token_mismatch": dream_mismatch,
        "dream_change_fraction": dream_change,
        "grid_hw": [GRID_H, GRID_W],
    }
    return {
        "gt": decoded["gt"].astype(np.uint8),
        "teacher_forced": decoded["teacher_forced"].astype(np.uint8),
        "dream": decoded["dream"].astype(np.uint8),
        "meta": meta,
    }


def run_gpu_phase(
    *,
    shots: tuple[int, ...],
    decode_dir: Path,
    token_root: Path | None,
    device: str,
    chunk: int,
) -> list[Path]:
    """Load the M2 model ONCE, build + decode every held-out shot, save bundles."""
    import torch

    from imas_ambix.worldmodel.spacetime_train_v2 import (
        load_signal_model_from_checkpoint,
    )

    decode_dir.mkdir(parents=True, exist_ok=True)
    verdict = _load_json(M2_VERDICT_H16)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU (slow)")
        device = "cpu"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    logger.info("loading M2 checkpoint ONCE on %s: %s", device, M2_CKPT)
    model, payload = load_signal_model_from_checkpoint(M2_CKPT, map_location=device)
    model.eval()
    logger.info(
        "checkpoint step=%d params=%d",
        int(payload.get("step", -1)),
        int(model.num_parameters()),
    )

    base_work = Path(
        tempfile.mkdtemp(prefix="ppfigs-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    written: list[Path] = []
    try:
        for sid in shots:
            temp, top_p, name = best_setting_for_shot(verdict, sid)
            logger.info("==== shot %s — dream setting %s ====", sid, name)
            out = _generate_and_decode_shot(
                model=model,
                payload=payload,
                shot_id=int(sid),
                temperature=temp,
                top_p=top_p,
                setting_name=name,
                work_dir=base_work / f"shot-{sid}",
                token_root=token_root,
                device=device,
                chunk=chunk,
            )
            bundle = decode_dir / f"shot-{sid}.npz"
            np.savez_compressed(
                bundle,
                gt=out["gt"],
                teacher_forced=out["teacher_forced"],
                dream=out["dream"],
                meta=json.dumps(out["meta"]),
            )
            m = out["meta"]
            logger.info(
                "wrote %s | dream_token_mismatch=%.4f dream_change=%.4f "
                "(present streams=%d)",
                bundle,
                m["dream_token_mismatch"],
                m["dream_change_fraction"],
                len(m["present_streams"]),
            )
            written.append(bundle)
    finally:
        try:
            del model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model release note: %r", exc)
    return written


# ===========================================================================
# PHASE B — render GIFs (no GPU)
# ===========================================================================
ORIGINAL_HW = (112, 156)  # native rbb aspect for display


def _to_aspect(img_square: np.ndarray) -> np.ndarray:
    """Resize a square decoded frame to the native rbb aspect for display."""
    from PIL import Image

    if img_square.ndim == 3:
        img_square = img_square[..., 0]
    im = Image.fromarray(img_square.astype(np.uint8)).resize(
        (ORIGINAL_HW[1], ORIGINAL_HW[0]), Image.BILINEAR
    )
    return np.asarray(im)


def _panel_frame(gt_img, pred_img, *, left_title, right_title, banner, in_target):
    """One side-by-side GT | prediction panel rendered to an RGB array."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), dpi=100)
    fig.subplots_adjust(top=0.80, bottom=0.02, left=0.02, right=0.98, wspace=0.04)
    for ax, img, title in (
        (axes[0], gt_img, left_title),
        (axes[1], pred_img, right_title),
    ):
        ax.imshow(img, cmap="inferno", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, color=(ACCENT_LOSS if in_target else DARK))
    fig.suptitle(banner, fontsize=8, y=0.985)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _save_gif(frames, out_path: Path, *, fps: int = FPS) -> None:
    from PIL import Image

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


def _render_side_by_side_gif(
    *,
    gt: np.ndarray,
    pred: np.ndarray,
    meta: dict,
    role_label: str,
    right_tag: str,
    out_path: Path,
) -> None:
    """GT | prediction GIF over the whole window; forecast frames flagged red."""
    ctx = int(meta["context_frames"])
    n_frames = int(meta["n_frames"])
    step = int(meta["checkpoint_step"])
    shot_id = int(meta["shot_id"])
    ftime = np.asarray(meta["frame_time"], dtype=np.float64)
    n = min(gt.shape[0], pred.shape[0], n_frames)

    frames = []
    for fi in range(n):
        in_target = fi >= ctx
        t_ms = float(ftime[fi] - ftime[ctx]) * 1e3 if ftime.size > ctx else 0.0
        banner = (
            f"M2 {role_label} | shot {shot_id}"
            f"{' (bright)' if meta.get('is_bright') else ' (dim)'} | "
            f"rbb full-res | step {step:,}"
        )
        frames.append(
            _panel_frame(
                _to_aspect(gt[fi]),
                _to_aspect(pred[fi]),
                left_title=f"ground truth   t={t_ms:+.0f} ms",
                right_title=f"{right_tag}   [{'FORECAST' if in_target else 'context'}]",
                banner=banner,
                in_target=in_target,
            )
        )
    _save_gif(frames, out_path)
    logger.info("  wrote %s (%d frames @ %d fps)", out_path, n, FPS)


def render_gifs(decode_dir: Path, out_dir: Path) -> list[Path]:
    """Write per-shot GT|dream and GT|reconstruction GIFs from the bundles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sid in HELDOUT_SHOTS:
        bundle = decode_dir / f"shot-{sid}.npz"
        if not bundle.exists():
            logger.warning("bundle missing for shot %s: %s — skipping", sid, bundle)
            continue
        data = np.load(str(bundle), allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        gt = np.asarray(data["gt"], dtype=np.uint8)
        tf = np.asarray(data["teacher_forced"], dtype=np.uint8)
        dream = np.asarray(data["dream"], dtype=np.uint8)
        logger.info(
            "rendering shot %s (dream setting %s)", sid, meta.get("dream_setting")
        )

        dream_gif = out_dir / f"m2-heldout-{sid}-dream.gif"
        _render_side_by_side_gif(
            gt=gt,
            pred=dream,
            meta=meta,
            role_label="dream (autoregressive, sampled)",
            right_tag=f"model dream  {meta.get('dream_setting', '')}",
            out_path=dream_gif,
        )
        written.append(dream_gif)

        recon_gif = out_dir / f"m2-heldout-{sid}-recon.gif"
        _render_side_by_side_gif(
            gt=gt,
            pred=tf,
            meta=meta,
            role_label="reconstruction (teacher-forced)",
            right_tag="model reconstruction",
            out_path=recon_gif,
        )
        written.append(recon_gif)
    return written


# ===========================================================================
# PHASE B — metric plots (no GPU; from the verdict JSONs alone)
# ===========================================================================
def _style_axis(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_edgecolor(RULE)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=RULE, labelcolor=DARK)
    ax.set_axisbelow(True)


def fig_crps_bars(out_dir: Path) -> Path:
    """CRPS-ratio-vs-persistence bars, per shot, M1 vs M2 (best sampling)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m1_by = _per_shot_rows(M1_VERDICT)
    m2_by = _per_shot_rows(M2_VERDICT_H16)

    shots = list(HELDOUT_SHOTS)
    labels = [f"{s}\n{'bright' if s in BRIGHT_SHOTS else 'dim'}" for s in shots]
    m1_vals = [float(m1_by[s]["best_sampling_crps_ratio"]) for s in shots]
    m2_vals = [float(m2_by[s]["best_sampling_crps_ratio"]) for s in shots]

    x = np.arange(len(shots))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    b1 = ax.bar(x - w / 2, m1_vals, w, color=ACCENT_M1, label="M1 (signal-cond.)")
    b2 = ax.bar(x + w / 2, m2_vals, w, color=ACCENT_M2, label="M2 (corruption-trained)")

    # persistence reference line at ratio = 1.0 (below = beats persistence).
    ax.axhline(1.0, color=ACCENT_LOSS, linewidth=1.3, linestyle="--", zorder=1)
    ax.text(
        len(shots) - 0.45,
        1.01,
        "persistence",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=ACCENT_LOSS,
        style="italic",
    )

    for bars, vals in ((b1, m1_vals), (b2, m2_vals)):
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=DARK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=DARK)
    ax.set_ylabel(
        "CRPS ratio vs persistence  (lower is better)", fontsize=11, color=DARK
    )
    ax.set_ylim(0, max(max(m1_vals), max(m2_vals), 1.0) * 1.18)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    _style_axis(ax)
    ax.yaxis.grid(True, color=RULE, linewidth=0.4, linestyle="--", alpha=0.5)

    ax.set_title(
        "Forecast skill vs persistence — ensemble CRPS ratio, held-out shots",
        fontsize=14,
        color=DARK,
        pad=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Each bar is the best sampling setting's ensemble CRPS divided by the "
        "persistence baseline. Below the dashed line beats persistence. "
        "M2 beats persistence on all four shots and improves on M1 on both bright "
        "shots (18502, 18505).",
        ha="center",
        va="top",
        fontsize=9.5,
        color=RULE,
        wrap=True,
    )
    fig.tight_layout()
    out = out_dir / "crps-ratio-vs-persistence.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  wrote %s", out)
    return out


def fig_horizon_drift(out_dir: Path) -> Path:
    """M2 CRPS ratio at the matched (24f) vs longer (26f) horizon, per shot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h16 = _per_shot_rows(M2_VERDICT_H16)
    h26 = _per_shot_rows(M2_VERDICT_H26)

    ctx = WINDOW["context_frames"]
    h_short = WINDOW["n_frames"] - ctx  # 16 forecast frames
    h_long = 26 - ctx  # 18 forecast frames

    shots = list(HELDOUT_SHOTS)
    short_vals = [float(h16[s]["best_sampling_crps_ratio"]) for s in shots]
    long_vals = [float(h26[s]["best_sampling_crps_ratio"]) for s in shots]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = [h_short, h_long]
    for s, sv, lv in zip(shots, short_vals, long_vals, strict=True):
        bright = s in BRIGHT_SHOTS
        color = ACCENT_M2 if bright else ACCENT_PERS
        ax.plot(
            xs,
            [sv, lv],
            "-o",
            color=color,
            lw=1.8,
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            zorder=3,
            alpha=0.95,
        )
        # label directly at the right endpoint.
        ax.text(
            h_long + 0.25,
            lv,
            f"{s} ({'bright' if bright else 'dim'})",
            va="center",
            ha="left",
            fontsize=9.5,
            color=color,
        )

    ax.axhline(1.0, color=ACCENT_LOSS, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(
        h_short - 0.05,
        1.01,
        "persistence",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=ACCENT_LOSS,
        style="italic",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{h_short}-frame\nforecast", f"{h_long}-frame\nforecast"],
        fontsize=11,
        color=DARK,
    )
    ax.set_xlim(h_short - 0.8, h_long + 2.6)
    ax.set_ylim(0, max(max(short_vals), max(long_vals), 1.0) * 1.12)
    ax.set_ylabel(
        "CRPS ratio vs persistence  (lower is better)", fontsize=11, color=DARK
    )
    _style_axis(ax)
    ax.yaxis.grid(True, color=RULE, linewidth=0.4, linestyle="--", alpha=0.5)

    ax.set_title(
        "Skill holds at a longer horizon — M2 CRPS ratio, matched vs extended",
        fontsize=13.5,
        color=DARK,
        pad=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Extending the forecast from 16 to 18 frames does not erode the skill: "
        "every shot stays below persistence, and the ratio holds or improves at "
        "the longer horizon.",
        ha="center",
        va="top",
        fontsize=9.5,
        color=RULE,
        wrap=True,
    )
    fig.tight_layout()
    out = out_dir / "horizon-drift.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  wrote %s", out)
    return out


def fig_motion_collapse(out_dir: Path) -> Path:
    """Argmax (collapsed ~0.1) vs sampling (~1.0 = GT-like) motion, per shot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _per_shot_rows(M2_VERDICT_H16)

    shots = list(HELDOUT_SHOTS)
    labels = [f"{s}\n{'bright' if s in BRIGHT_SHOTS else 'dim'}" for s in shots]
    argmax_vals = [float(rows[s]["argmax_collapse_ratio"]) for s in shots]
    samp_vals = [float(rows[s]["best_sampling_collapse_ratio"]) for s in shots]

    x = np.arange(len(shots))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ba = ax.bar(x - w / 2, argmax_vals, w, color=ACCENT_LOSS, label="argmax (greedy)")
    bs = ax.bar(x + w / 2, samp_vals, w, color=ACCENT_WIN, label="best sampling")

    # GT-like motion reference at collapse ratio = 1.0.
    ax.axhline(1.0, color=DARK, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(
        len(shots) - 0.45,
        1.02,
        "ground-truth motion",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=DARK,
        style="italic",
    )

    for bars, vals in ((ba, argmax_vals), (bs, samp_vals)):
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=DARK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=DARK)
    ax.set_ylabel(
        "motion ratio vs ground truth  (1.0 = GT-like)", fontsize=11, color=DARK
    )
    ax.set_ylim(0, max(max(argmax_vals), max(samp_vals), 1.0) * 1.22)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    _style_axis(ax)
    ax.yaxis.grid(True, color=RULE, linewidth=0.4, linestyle="--", alpha=0.5)

    ax.set_title(
        "Sampling cures the collapse — frame-to-frame motion vs ground truth",
        fontsize=14,
        color=DARK,
        pad=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Greedy argmax freezes the forecast (motion ~0.1 of ground truth — a "
        "static collapse). Sampling restores GT-like motion (ratio near 1.0), the "
        "mechanism behind the CRPS win.",
        ha="center",
        va="top",
        fontsize=9.5,
        color=RULE,
        wrap=True,
    )
    fig.tight_layout()
    out = out_dir / "motion-collapse.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  wrote %s", out)
    return out


def render_plots(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        fig_crps_bars(out_dir),
        fig_horizon_drift(out_dir),
        fig_motion_collapse(out_dir),
    ]


# ===========================================================================
# CLI
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gpu", help="Phase A: generate + decode (GPU, betelgeuse)")
    g.add_argument("--decode-dir", default=str(DECODE_DIR))
    g.add_argument("--shots", default=",".join(str(s) for s in HELDOUT_SHOTS))
    g.add_argument("--token-root", default=None)
    g.add_argument("--device", default="cuda")
    g.add_argument("--chunk", type=int, default=8192)

    r = sub.add_parser("render", help="Phase B: GIFs + metric PNGs (no GPU)")
    r.add_argument("--decode-dir", default=str(DECODE_DIR))
    r.add_argument("--out-dir", default=str(OUT_DIR))
    r.add_argument("--gifs-only", action="store_true")
    r.add_argument("--plots-only", action="store_true")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "gpu":
        shots = tuple(int(s) for s in args.shots.split(",") if s.strip())
        written = run_gpu_phase(
            shots=shots,
            decode_dir=Path(args.decode_dir),
            token_root=Path(args.token_root) if args.token_root else None,
            device=args.device,
            chunk=args.chunk,
        )
        print("\n=== GPU phase wrote image bundles ===")
        for w in written:
            print(f"  {w}")
        return 0

    if args.cmd == "render":
        out_dir = Path(args.out_dir)
        produced: list[Path] = []
        if not args.plots_only:
            produced += render_gifs(Path(args.decode_dir), out_dir)
        if not args.gifs_only:
            produced += render_plots(out_dir)
        print("\n=== render phase wrote assets ===")
        for w in produced:
            print(f"  {w}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
