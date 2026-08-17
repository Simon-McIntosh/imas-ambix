"""Render all figures for the world-model demonstration research document.

Loads eval_bundle.npz and gt_camera_images.npz from the GPFS artifact
directory and writes PNGs + GIFs to docs/figures/world-model-demonstration/.

Re-runnable; overwrites existing files.

GIF encoding uses Pillow (imageio not available in the project venv).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BUNDLE_DIR = Path("/work/projects/imas_gpu/worldmodel/demo")
EVAL_BUNDLE = BUNDLE_DIR / "eval_bundle.npz"
GT_CAM_BUNDLE = BUNDLE_DIR / "gt_camera_images.npz"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "figures" / "world-model-demonstration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150

# ---------------------------------------------------------------------------
# Tufte palette
# ---------------------------------------------------------------------------
DARK = "#1a1a1a"  # body text / axes
RULE = "#999999"  # thin rules / tick marks
WIN_FILL = "#f0f4f8"  # context-region shading (cool blue-grey)
ACCENT_NEG = "#c0392b"  # losses (bars below 0)
ACCENT_POS = "#2980b9"  # wins  (bars above 0)
ACCENT_TRUTH = "#1a1a1a"
ACCENT_PRED = "#e67e22"  # warm orange
ACCENT_PERS = "#7f8c8d"  # muted grey
MATCH_HIT = "#27ae60"  # green
MATCH_MISS = "#e74c3c"  # red


# ---------------------------------------------------------------------------
# Load bundles once
# ---------------------------------------------------------------------------
def load_bundles():
    print(f"Loading {EVAL_BUNDLE} …", flush=True)
    bundle = np.load(EVAL_BUNDLE, allow_pickle=True)
    print(f"Loading {GT_CAM_BUNDLE} …", flush=True)
    cam = np.load(GT_CAM_BUNDLE, allow_pickle=True)
    summary = json.loads(str(bundle["summary"]))
    index = json.loads(str(bundle["index"]))
    cam_index = json.loads(str(cam["index"]))
    return bundle, cam, summary, index, cam_index


# ===========================================================================
# Fig 1: params-breakdown.png
# ===========================================================================
def fig_params_breakdown(summary: dict) -> Path:
    """Horizontal bar — where the 1.034 B parameters live."""
    pb = summary["param_breakdown"]
    groups = pb["groups"]
    total = pb["sum"]

    # Fixed labels in the order specified
    labels = [
        "camera codebook tables",
        "transformer backbone",
        "other-modality embed/head",
        "positional/segment",
    ]
    keys = [
        "camera_embed_head",
        "transformer_backbone",
        "other_modality_embed_head",
        "other",
    ]
    values = [groups[k] for k in keys]
    pcts = [v / total * 100 for v in values]

    # Sanity-check against spec values
    expected = [1_007_976_320, 14_196_480, 11_804_313, 49_920]
    for k, v, e in zip(keys, values, expected):
        if v != e:
            print(f"  [WARN] {k}: bundle has {v}, spec says {e}", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # log-x so small bars are visible
    y_pos = np.arange(len(labels))
    colors = ["#2c3e50", "#2980b9", "#27ae60", "#7f8c8d"]
    bars = ax.barh(y_pos, values, color=colors, height=0.55, edgecolor="none")

    ax.set_xscale("log")
    ax.set_xlim(left=1e4, right=3e9)

    # Direct value labels on each bar
    for i, (bar, v, pct) in enumerate(zip(bars, values, pcts)):
        label = f"{v:,.0f}  ({pct:.1f}%)"
        x_end = bar.get_width()
        ax.text(x_end * 1.06, i, label, va="center", ha="left", fontsize=11, color=DARK)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, color=DARK)
    ax.invert_yaxis()
    ax.set_xlabel("Parameter count (log scale)", fontsize=12, color=DARK)
    ax.tick_params(colors=RULE, labelcolor=DARK)
    for spine in ax.spines.values():
        spine.set_edgecolor(RULE)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(
        True, which="both", color=RULE, linewidth=0.5, linestyle="--", alpha=0.6
    )
    ax.set_axisbelow(True)

    ax.set_title(
        "Where the 1.034 billion parameters live",
        fontsize=14,
        color=DARK,
        pad=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.04,
        "97.5% of parameters are the five cameras' lookup tables (2¸¹⁸-entry codebooks). "
        "The actual sequence-modelling transformer is ~14 M parameters.",
        ha="center",
        va="top",
        fontsize=10,
        color=RULE,
        wrap=True,
    )

    fig.tight_layout()
    out = OUT_DIR / "params-breakdown.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


# ===========================================================================
# Fig 2: data-coverage.png
# ===========================================================================
def fig_data_coverage(summary: dict) -> Path:
    """Two small labeled bars showing corpus and token coverage."""
    _train_limit = summary["train_discovery_limit"]  # 4000
    total_shots = 15_361  # spec
    trained_shots = 3_996  # spec (derived from 4000 limit → ~3996 unique shots)
    tokens_used = 16
    tokens_total = 256

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    fig.patch.set_facecolor("white")

    ax_a, ax_b = axes

    def two_bar(ax, used, total, label_used, label_rest, title, unit=""):
        colors = [ACCENT_POS, "#dde6ed"]
        vals = [used, total - used]
        ax.barh([1, 0], vals, color=colors, height=0.5, edgecolor="none")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["rest", label_used], fontsize=12, color=DARK)
        ax.set_xlim(0, total * 1.35)
        pct = used / total * 100
        ax.text(
            used + total * 0.02,
            1,
            f"{used:,} / {total:,}  ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=11,
            color=DARK,
        )
        ax.text(
            total - used + total * 0.02,
            0,
            f"{total - used:,}",
            va="center",
            ha="left",
            fontsize=11,
            color=RULE,
        )
        ax.set_title(title, fontsize=13, color=DARK, pad=8, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False, colors=RULE)
        ax.set_facecolor("white")

    two_bar(
        ax_a,
        trained_shots,
        total_shots,
        "trained",
        "not used",
        "(a) Shots trained / available",
    )
    two_bar(
        ax_b,
        tokens_used,
        tokens_total,
        "used tokens",
        "not used",
        "(b) Camera tokens per frame",
    )

    fig.suptitle(
        "How much of the corpus this demo used",
        fontsize=14,
        color=DARK,
        y=1.02,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.06,
        "All 14 signal types are wired, but only ~a quarter of shots "
        "and 1/16 of each camera frame’s spatial tokens.",
        ha="center",
        va="top",
        fontsize=10,
        color=RULE,
    )
    fig.tight_layout()
    out = OUT_DIR / "data-coverage.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


# ===========================================================================
# Fig 3: skill-by-modality.png
# ===========================================================================
def fig_skill_by_modality(summary: dict) -> Path:
    """Horizontal signed-bar of per-modality skill vs persistence."""
    skill_data = summary["averaged_skill"]

    # Exclude __overall__ from bars but keep for reference line
    overall_skill = skill_data["__overall__"]["skill"]

    rows = [(mod, d["skill"]) for mod, d in skill_data.items() if mod != "__overall__"]
    rows.sort(key=lambda x: x[1])  # sort ascending

    labels = [r[0] for r in rows]
    skills = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [ACCENT_NEG if s < 0 else ACCENT_POS for s in skills]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, skills, color=colors, height=0.6, edgecolor="none")

    # Annotate each bar with its skill value
    for i, (bar, s) in enumerate(zip(bars, skills)):
        x_off = 0.004 if s >= 0 else -0.004
        ha = "left" if s >= 0 else "right"
        ax.text(s + x_off, i, f"{s:+.4f}", va="center", ha=ha, fontsize=9.5, color=DARK)

    # Overall reference line
    ax.axvline(overall_skill, color=DARK, linewidth=1.5, linestyle="--", alpha=0.85)
    ax.text(
        overall_skill + 0.002,
        len(labels) - 0.3,
        f"overall {overall_skill:+.4f}",
        fontsize=9.5,
        color=DARK,
        va="top",
    )

    # Zero line
    ax.axvline(0, color=RULE, linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, color=DARK)
    ax.set_xlabel(
        "Skill = 1 − model_error / persistence_error", fontsize=11, color=DARK
    )
    ax.tick_params(colors=RULE, labelcolor=DARK)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor(RULE)
        ax.spines[spine].set_linewidth(0.8)
    ax.xaxis.grid(True, color=RULE, linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    win_patch = mpatches.Patch(color=ACCENT_POS, label="beats persistence")
    loss_patch = mpatches.Patch(color=ACCENT_NEG, label="loses to persistence")
    ax.legend(
        handles=[win_patch, loss_patch], fontsize=10, frameon=False, loc="lower right"
    )

    ax.set_title(
        "Forward-prediction skill vs persistence — held-out shots, 48-step horizon",
        fontsize=14,
        color=DARK,
        pad=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.03,
        "Skill = 1 − model_error/persistence_error. Positive beats persistence; "
        "the model loses on average (overall −0.043). "
        "Persistence is a very strong baseline on quasi-stationary signals. "
        "(xma is a degenerate size-1 codebook: both errors 0.)",
        ha="center",
        va="top",
        fontsize=9.5,
        color=RULE,
        wrap=True,
    )
    fig.tight_layout()
    out = OUT_DIR / "skill-by-modality.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out


# ===========================================================================
# Fig 4: signal-traces.png
# ===========================================================================
def fig_signal_traces(bundle: np.lib.npyio.NpzFile) -> tuple[Path, list[dict]]:
    """3 stacked panels for held-out shot 23735."""
    SHOT = 23735
    CONTEXT = 16

    # Channel selection rationale:
    #   - gas_injection ch 0: 48/48 valid, wide token range [113,236] = quasi-continuous
    #     gas_injection is the spec's "worst-case" modality (skill -0.39)
    #   - pf_active ch 2: 48/48 valid, widest token range [91,169] in pf_active (34 unique)
    #     pf_active = core electromagnetic signal
    #   - xsx ch 0: 48/48 valid, broadest range [66,852] = 8 unique tokens
    #     soft_x_rays = plasma thermal emission, different physics from the above
    channels = [
        dict(
            modality="gas_injection",
            ch=0,
            note="gas injection (worst-case modality, skill −0.39)",
        ),
        dict(
            modality="pf_active",
            ch=2,
            note="poloidal field coil (electromagnetic, 34 unique tokens)",
        ),
        dict(modality="xsx", ch=0, note="soft x-ray line (plasma thermal emission)"),
    ]

    steps = np.arange(64)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, ch_info in zip(axes, channels):
        mod = ch_info["modality"]
        c = ch_info["ch"]
        truth = bundle[f"{SHOT}::{mod}::truth"][:, c]
        pred = bundle[f"{SHOT}::{mod}::pred"][:, c]
        persist = bundle[f"{SHOT}::{mod}::persist"][:, c]
        valid = bundle[f"{SHOT}::{mod}::valid"][:, c]

        ax.set_facecolor("white")
        # Context shading
        ax.axvspan(0, CONTEXT - 0.5, color=WIN_FILL, alpha=0.75, zorder=0)
        ax.axvline(CONTEXT - 0.5, color=RULE, linewidth=1, linestyle="--", zorder=1)

        # Mask invalid steps
        t_masked = np.where(valid, truth, np.nan).astype(float)
        p_masked = np.where(valid, pred, np.nan).astype(float)
        r_masked = np.where(valid, persist, np.nan).astype(float)

        ax.plot(
            steps,
            t_masked,
            color=ACCENT_TRUTH,
            lw=1.5,
            zorder=3,
            label="truth",
            solid_capstyle="round",
        )
        ax.plot(
            steps,
            p_masked,
            color=ACCENT_PRED,
            lw=1.5,
            zorder=4,
            linestyle="--",
            label="model pred",
        )
        ax.plot(
            steps,
            r_masked,
            color=ACCENT_PERS,
            lw=1.0,
            zorder=2,
            linestyle=":",
            label="persistence",
        )

        ax.set_ylabel("token id", fontsize=10, color=DARK)
        ax.set_title(
            f"{mod}  ch {c}", fontsize=12, color=DARK, pad=4, fontweight="bold"
        )
        ax.tick_params(colors=RULE, labelcolor=DARK)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_edgecolor(RULE)
            ax.spines[spine].set_linewidth(0.8)
        ax.yaxis.grid(True, color=RULE, linewidth=0.3, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=9.5, frameon=False, ncol=3)

    axes[-1].set_xlabel("grid step", fontsize=11, color=DARK)
    axes[0].text(
        CONTEXT / 2,
        axes[0].get_ylim()[1],
        "context",
        ha="center",
        va="bottom",
        fontsize=9,
        color=RULE,
        style="italic",
    )

    fig.suptitle(
        f"Predicted vs actual token trajectories (held-out shot {SHOT})",
        fontsize=14,
        color=DARK,
        y=1.01,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Context steps 0–15 are given; steps 16–63 are generated autoregressively. "
        "Discrete codebook ids; the model rarely diverges far from persistence here.",
        ha="center",
        va="top",
        fontsize=10,
        color=RULE,
    )
    fig.tight_layout()
    out = OUT_DIR / "signal-traces.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return out, channels


# ===========================================================================
# Fig 5: camera-gt.gif
# ===========================================================================
def fig_camera_gt_gif(cam: np.lib.npyio.NpzFile) -> Path:
    """Animate the first shot's 48 GT rbb frames at 8 fps via Pillow."""
    images = cam["images"]  # (8, 48, 256, 256, 3) uint8
    cam_index = json.loads(str(cam["index"]))

    # slot 0 = shot 23735 (first pick)
    slot = 0
    shot_id = cam_index[slot]["shot_id"]
    frames_np = images[slot]  # (48, 256, 256, 3)

    def _add_title_strip(
        frame_np: np.ndarray, title: str, step: int, n_frames: int
    ) -> Image.Image:
        """Add a thin title strip and step counter onto a frame."""
        img = Image.fromarray(frame_np)
        # Paste into a slightly taller canvas
        strip_h = 20
        canvas = Image.new("RGB", (img.width, img.height + strip_h), color=(30, 30, 30))
        canvas.paste(img, (0, strip_h))
        # Draw title text using a simple bitmap approach (PIL ImageDraw)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(canvas)
        draw.text(
            (4, 2), f"{title}  [frame {step + 1}/{n_frames}]", fill=(220, 220, 220)
        )
        return canvas

    pil_frames = []
    n_frames = frames_np.shape[0]
    title = "Ground truth rbb (decoded full-resolution)"
    for i in range(n_frames):
        pil_frames.append(_add_title_strip(frames_np[i], title, i, n_frames))

    out = OUT_DIR / "camera-gt.gif"
    duration_ms = int(1000 / 8)  # 8 fps
    pil_frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  wrote {out}  (shot {shot_id}, {n_frames} frames @ 8 fps)", flush=True)
    return out


# ===========================================================================
# Fig 6: camera-pred-vs-truth.gif
# ===========================================================================
def fig_camera_pred_vs_truth_gif(bundle: np.lib.npyio.NpzFile) -> Path:
    """
    Animated 3-panel 4x4 heatmap: truth | model-pred | match
    for shot 23735 rbb, target steps 16..63 (48 steps).
    """
    SHOT = 23735
    CONTEXT = 16
    GRID_H, GRID_W = 4, 4

    rbb_truth = bundle[f"{SHOT}::rbb::truth"]  # (64, 16)
    rbb_pred = bundle[f"{SHOT}::rbb::pred"]  # (64, 16)
    rbb_valid = bundle[f"{SHOT}::rbb::valid"]  # (64, 16)

    target_truth = rbb_truth[CONTEXT:, :].reshape(-1, GRID_H, GRID_W)  # (48, 4, 4)
    target_pred = rbb_pred[CONTEXT:, :].reshape(-1, GRID_H, GRID_W)  # (48, 4, 4)
    target_valid = rbb_valid[CONTEXT:, :].reshape(-1, GRID_H, GRID_W)  # (48, 4, 4)
    n_steps = target_truth.shape[0]  # 48

    # Shared color scale across both truth and pred panels
    vmin = min(target_truth.min(), target_pred.min())
    vmax = max(target_truth.max(), target_pred.max())

    pil_frames = []
    cmap_signal = "viridis"

    for step in range(n_steps):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"rbb coarse 16-token view: truth | model | match"
            f"  [target step {step + CONTEXT}/{CONTEXT + n_steps - 1}]",
            fontsize=11,
            color=DARK,
            fontweight="bold",
        )

        t_grid = target_truth[step].astype(float)
        p_grid = target_pred[step].astype(float)
        match = (target_truth[step] == target_pred[step]).astype(float)

        # Mask invalid tokens
        inv_t = ~target_valid[step]
        t_grid[inv_t] = np.nan
        p_grid[inv_t] = np.nan
        match[inv_t] = np.nan

        # Panel 0: truth
        axes[0].imshow(
            t_grid, vmin=vmin, vmax=vmax, cmap=cmap_signal, interpolation="nearest"
        )
        axes[0].set_title("truth", fontsize=11, color=DARK)

        # Panel 1: model prediction
        axes[1].imshow(
            p_grid, vmin=vmin, vmax=vmax, cmap=cmap_signal, interpolation="nearest"
        )
        axes[1].set_title("model pred", fontsize=11, color=DARK)

        # Panel 2: match map
        # green=1 (hit), red=0 (miss), grey=invalid
        cmap_match = mcolors.ListedColormap([MATCH_MISS, MATCH_HIT])
        axes[2].imshow(match, vmin=0, vmax=1, cmap=cmap_match, interpolation="nearest")
        axes[2].set_title("match", fontsize=11, color=DARK)

        for ax in axes:
            ax.set_facecolor("white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(RULE)
                spine.set_linewidth(0.8)

        # match fraction text
        valid_mask = ~inv_t
        if valid_mask.any():
            frac = match[valid_mask].mean()
            axes[2].text(
                0.5,
                -0.12,
                f"{frac:.0%} match",
                ha="center",
                va="top",
                transform=axes[2].transAxes,
                fontsize=10,
                color=DARK,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        # Render to PIL
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())

    out = OUT_DIR / "camera-pred-vs-truth.gif"
    duration_ms = int(1000 / 8)  # 8 fps
    pil_frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  wrote {out}  ({n_steps} frames @ 8 fps)", flush=True)
    return out


# ===========================================================================
# Main
# ===========================================================================
def main():
    bundle, cam, summary, index, cam_index = load_bundles()

    print("\n[1/6] params-breakdown.png")
    fig_params_breakdown(summary)

    print("\n[2/6] data-coverage.png")
    fig_data_coverage(summary)

    print("\n[3/6] skill-by-modality.png")
    fig_skill_by_modality(summary)

    print("\n[4/6] signal-traces.png")
    out4, chosen_channels = fig_signal_traces(bundle)
    print("  channels used:")
    for ch in chosen_channels:
        print(f"    {ch['modality']} ch {ch['ch']} — {ch['note']}")

    print("\n[5/6] camera-gt.gif")
    fig_camera_gt_gif(cam)

    print("\n[6/6] camera-pred-vs-truth.gif")
    fig_camera_pred_vs_truth_gif(bundle)

    print("\nDone. Files written to:", OUT_DIR)


if __name__ == "__main__":
    main()
