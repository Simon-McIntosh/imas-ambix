"""Figures for the accelerator-choice study: memory budget, concurrency, envelope.

Every quantity plotted here is either measured on 98dci4-gpu-0003 or a vendor
specification cited in the study text. Projected bars are drawn in outline
rather than fill so a reader can see which is which without the caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIGDIR = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "figures"
    / ("open-weight-serving-accelerator-choice")
)

INK = "#1a1a1a"
MUTED = "#5a5a5a"
ACCENT = "#1f5fa8"
WARM = "#b4531b"
GRID = "#d8d8d8"

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.size": 13,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 0.8,
    }
)


def strip(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


# ── Figure 1 — per-card memory budget ────────────────────────────────────────
# Measured on four H200 NVL serving DeepSeek-V4.1-Flash at TP4/EP4: the engine
# ledger read 137.85 GB available, 65.24 after weights, 55.15 after the
# 4,000,000-token pool, 52.62 at ready; the fused-MoE workspace asked 10.11 GiB
# at a ~500k prefill and the 16,384-token chunk carries ~3 GB of activation.
WEIGHTS = 75.2
POOL_4M = 10.1
WORKSPACE = 13.1


def memory_budget():
    cards = [
        ("H200 NVL\n141 GB · ours", 140.4, False),
        ("B200\n180 GB", 180.0, True),
        ("B300\n262 GB usable", 262.0, True),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    height = 0.5
    for row, (label, total, projected) in enumerate(cards):
        y = len(cards) - 1 - row
        free = total - WEIGHTS - POOL_4M - WORKSPACE
        segments = [
            (WEIGHTS, "#c9d6e8"),
            (POOL_4M, ACCENT),
            (WORKSPACE, WARM),
            (free, "white"),
        ]
        left = 0.0
        for width, colour in segments:
            ax.barh(
                y,
                width,
                left=left,
                height=height,
                color=colour,
                edgecolor=MUTED if colour == "white" else "none",
                linewidth=0.8,
                linestyle=(0, (3, 2)) if projected and colour == "white" else "solid",
            )
            left += width
        ax.text(
            total + 5,
            y,
            f"{free:,.0f} GB free",
            va="center",
            fontsize=13,
            color=INK if not projected else MUTED,
        )
        ax.text(-5, y, label, va="center", ha="right", fontsize=13)

    ax.axvline(WEIGHTS + POOL_4M + WORKSPACE + 12.0, color=WARM, lw=0.9, ls=":")
    ax.text(
        WEIGHTS + POOL_4M + WORKSPACE + 14.0,
        2.52,
        "+12 GiB/card — what DSpark asked for and could not get",
        fontsize=12,
        color=WARM,
        va="center",
    )
    ax.set_xlim(-48, 340)
    ax.set_ylim(-0.5, 2.8)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_xlabel("GB per card — one DeepSeek-V4.1-Flash rank at TP4")
    strip(ax, keep=("bottom",))
    fig.legend(
        handles=[
            Patch(facecolor="#c9d6e8", label="weights + engine"),
            Patch(facecolor=ACCENT, label="KV pool (4M tokens)"),
            Patch(facecolor=WARM, label="MoE workspace + chunk activation"),
            Patch(
                facecolor="white",
                edgecolor=MUTED,
                label="free (dashed = projected)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.54, -0.30),
        frameon=False,
        fontsize=12,
        ncol=2,
    )
    fig.savefig(
        FIGDIR / "memory-budget-per-card.svg", bbox_inches="tight", pad_inches=0.25
    )
    plt.close(fig)


# ── Figure 2 — measured concurrency curves ───────────────────────────────────
# Left: per-stream decode. Right: aggregate. All four H200 NVL cards, ours.
SERIES = {
    "DeepSeek-V4-Flash, vLLM": (
        [1, 2, 4, 8, 16, 32],
        [113.9, 109.0, 100.5, 87.7, 68.5, 53.0],
        [113.9, 218.0, 402.1, 701.3, 1096.5, 1696.0],
        ACCENT,
        "-o",
    ),
    "GLM-5.2 INT4, vLLM": (
        [1, 2, 4, 8, 16, 32],
        [83.8, 76.1, 67.0, 54.7, 44.7, 34.6],
        [83.8, 152.2, 268.0, 437.9, 714.4, 1108.7],
        "#6b8f3a",
        "-s",
    ),
    "DeepSeek-V4.1-Flash, SGLang": (
        [1, 4, 16, 20, 32],
        [36.4, 40.6, 30.9, 26.7, 24.8],
        [36.0, 162.0, 493.0, 533.0, 794.0],
        WARM,
        "-^",
    ),
}


def concurrency():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for name, (x, per, agg, colour, style) in SERIES.items():
        a.plot(x, per, style, color=colour, lw=1.4, ms=4.5, label=name)
        b.plot(x, agg, style, color=colour, lw=1.4, ms=4.5, label=name)

    a.axhspan(6.9, 10.1, color="#d8d8d8", alpha=0.75, lw=0)
    a.text(
        1.05,
        11.4,
        "6.9–10.1 tok/s — V4-Flash on REAL agent traffic",
        fontsize=12,
        color=INK,
    )
    a.annotate(
        "knee at 4",
        xy=(4, 40.6),
        xytext=(5.4, 58),
        fontsize=12,
        color=WARM,
        arrowprops=dict(arrowstyle="-", color=WARM, lw=0.8),
    )
    a.set_xscale("log", base=2)
    a.set_xticks([1, 2, 4, 8, 16, 32])
    a.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    a.set_xlabel("concurrent streams")
    a.set_ylabel("per-stream decode (tok/s)")
    a.set_ylim(0, 125)
    strip(a)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)

    b.set_xscale("log", base=2)
    b.set_xticks([1, 2, 4, 8, 16, 32])
    b.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    b.set_xlabel("concurrent streams")
    b.set_ylabel("aggregate decode (tok/s)")
    strip(b)
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "measured-concurrency.svg", bbox_inches="tight")
    plt.close(fig)


# ── Figure 3 — power, capacity and the air-cooling frontier ──────────────────
PARTS = [
    ("H200 NVL (ours)", 600, 141, "air", True, (14, 12)),
    ("RTX PRO 6000 SE", 600, 96, "air", False, (14, -22)),
    ("H200 SXM", 700, 141, "air", False, (14, -22)),
    ("B200", 1000, 180, "air", False, (14, 10)),
    ("B300 air bin", 1100, 288, "air", False, (-8, 14)),
    # The 1,400 W bin is one point on the plane occupied by two parts with
    # different cooling stories, so the markers are concentric and share a
    # label rather than being nudged apart.
    ("B300 DLC bin", 1400, 288, "liquid", False, (-104, -30)),
    ("MI355X", 1400, 288, "either", False, (14, 10)),
    ("Rubin VR200 Max Q", 1800, 288, "liquid", False, (-96, -30)),
]


def envelope():
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.axvspan(430, 1150, color="#eef3f8", lw=0)
    ax.text(455, 332, "air-coolable in a standard rack", fontsize=12, color=ACCENT)
    for name, watts, gb, cooling, ours, (dx, dy) in PARTS:
        face = {"air": ACCENT, "liquid": WARM, "either": "#6b8f3a"}[cooling]
        ax.plot(
            watts,
            gb,
            "o",
            ms=15 if cooling == "either" else (11 if ours else 8),
            mfc="white" if cooling == "either" else face,
            mec=face,
            mew=1.6,
            zorder=3 if cooling == "either" else 4,
        )
        ax.annotate(
            name + ("  \u2190  today" if ours else ""),
            xy=(watts, gb),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            color=WARM if ours else INK,
        )
    ax.set_xlabel("per-GPU power envelope (W)")
    ax.set_ylabel("HBM per GPU (GB)")
    ax.set_xlim(430, 1990)
    ax.set_ylim(60, 350)
    strip(ax)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=ACCENT, label="air-cooled SKU exists"),
            Patch(facecolor="white", edgecolor="#6b8f3a", label="air or liquid"),
            Patch(facecolor=WARM, label="liquid required"),
        ],
        frameon=False,
        fontsize=12,
        loc="center right",
    )
    fig.savefig(
        FIGDIR / "power-capacity-envelope.svg", bbox_inches="tight", pad_inches=0.25
    )
    plt.close(fig)


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    memory_budget()
    concurrency()
    envelope()
    print("wrote", *(p.name for p in sorted(FIGDIR.glob("*.svg"))))
