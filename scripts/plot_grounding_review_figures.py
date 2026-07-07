"""Figures for the project grounding review doc (from existing gate artifacts).

F1: annotated copy of the thread-1 current-distribution figure (what is
    physically wrong in each panel).
F2: current-moment order sweep — tune-cohort vs frozen eval skills (selection
    fragility / cohort variance).
F3: per-angle LCFS offsets — free vs moment vs origin-corrected moment.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/ITER/mcintos/Code/imas-ambix")
OUT = REPO / "docs" / "figures" / "project-grounding-review"
OUT.mkdir(parents=True, exist_ok=True)

# Paul Tol bright — colourblind-safe
BLUE = "#4477AA"  # free-current
ORANGE = "#EE7733"  # current-moment (eval / headline)
ORANGE_LT = "#F5B389"
GRAY = "#777777"
INK = "#1a1a1a"

# ---------------------------------------------------------------- F1: annotate
src = (
    REPO
    / "docs/figures/equilibrium-topology-fidelity"
    / "thread1-current-distribution.png"
)
img = mpimg.imread(src)
h, w = img.shape[:2]
rgb = img[..., :3]
redness = rgb[..., 0] - rgb[..., 2]  # red blobs
blueness = rgb[..., 2] - rgb[..., 0]  # blue halo


def hotspot(field, x0, x1, y0, y1):
    """(x, y) of the strongest pixel of `field` inside the fractional window."""
    ys, xs = slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w))
    sub = field[ys, xs]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return int(x0 * w) + ix, int(y0 * h) + iy


upper_lobe = hotspot(redness, 0.05, 0.45, 0.02, 0.30)
lower_lobe = hotspot(redness, 0.05, 0.45, 0.70, 0.99)
halo = hotspot(blueness, 0.05, 0.50, 0.30, 0.75)

fig, ax = plt.subplots(figsize=(w / 100, (h + 60) / 100), dpi=100)
ax.imshow(img)
ax.set_axis_off()


def note(xy_px, txt_px, text, color, extra_xy=None):
    ann = ax.annotate(
        text,
        xy=xy_px,
        xytext=txt_px,
        fontsize=11,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.2),
        ha="left",
        va="center",
        annotation_clip=False,
    )
    if extra_xy is not None:
        ax.annotate(
            "",
            xy=extra_xy,
            xytext=txt_px,
            arrowprops=dict(arrowstyle="->", color=color, lw=2.2),
            annotation_clip=False,
        )
    return ann


note(
    halo,
    (0.02 * w, 0.97 * h),
    "negative-current halo hugging the wall:\nsign-indefinite null-space fill"
    " — no jφ·sign(Ip) ≥ 0 prior",
    "#994455",
)
note(
    upper_lobe,
    (0.30 * w, 0.10 * h),
    "current lobes at the upper/lower inboard\ncorners (divertor-coil region)",
    "#994455",
    extra_xy=lower_lobe,
)
note(
    (0.72 * w, 0.45 * h),
    (0.52 * w, 0.99 * h),
    "monopole = Ip spread UNIFORMLY over every\nin-limiter cell: jφ ≠ 0 in the"
    " vacuum region,\nso Δ*ψ = 0 is violated exactly where the\n"
    "boundary/X-point is read",
    "#8a5a00",
)
fig.tight_layout()
fig.savefig(OUT / "current-distribution-annotated.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- F2: tune vs eval bars
orders = [2, 3, 4, 5]
tune_xpt = [-0.503, -2.310, -1.196, -3.199]
tune_lcfs = [-15.244, -7.021, -10.450, -11.245]
eval_o3_xpt, eval_o3_lcfs = -0.151, -4.470
free_eval_xpt, free_eval_lcfs = -0.614, -6.274

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=False)
for ax, tune, ev, free, name in [
    (axes[0], tune_xpt, eval_o3_xpt, free_eval_xpt, "X-point-set skill"),
    (axes[1], tune_lcfs, eval_o3_lcfs, free_eval_lcfs, "LCFS skill"),
]:
    xs = np.arange(len(orders), dtype=float)
    ax.bar(xs, tune, width=0.55, color=ORANGE_LT, label="tune cohort (order sweep)")
    ax.bar(
        [len(orders) + 0.4],
        [ev],
        width=0.55,
        color=ORANGE,
        label="frozen order 3 — held-out eval",
    )
    ax.axhline(
        free, color=BLUE, lw=2, ls="--", label="free-current baseline (held-out eval)"
    )
    ax.axhline(0.0, color=GRAY, lw=1)
    ax.set_xticks(list(xs) + [len(orders) + 0.4])
    ax.set_xticklabels([f"o{o}" for o in orders] + ["o3\neval"])
    ax.set_title(name, fontsize=11.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e3e3", lw=0.7)
    ax.set_axisbelow(True)
    for x, v in zip(xs, tune, strict=True):
        ax.annotate(
            f"{v:.2f}",
            (x, v),
            textcoords="offset points",
            xytext=(0, -11),
            ha="center",
            fontsize=8.5,
            color=INK,
        )
    ax.annotate(
        f"{ev:.2f}",
        (len(orders) + 0.4, ev),
        textcoords="offset points",
        xytext=(0, -11),
        ha="center",
        fontsize=8.5,
        color=INK,
        fontweight="bold",
    )
axes[0].set_ylabel("skill vs train-mean baseline\n(0 = baseline parity)")
axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
fig.suptitle(
    "Tune cohort and frozen eval disagree — no CI on any of these skills",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "moment-tune-vs-eval-instability.png", dpi=140)
plt.close(fig)

# ------------------------------------------------ F3: per-angle LCFS residual
ang_lbl = [
    "0°\noutboard",
    "45°",
    "90°\ntop",
    "135°\ninboard\ncorner",
    "180°\ninboard",
    "225°\ninboard\ncorner",
    "270°\nbottom",
    "315°",
]
free_raw = [24.3, 22.8, 40.4, 39.8, 29.8, 39.5, 42.2, 23.1]
mom_raw = [19.8, 19.2, 26.1, 39.8, 31.9, 38.8, 20.5, 16.0]
mom_corr = [22.0, 18.5, 24.7, 36.9, 30.7, 38.6, 20.5, 18.4]

xs = np.arange(8, dtype=float)
fig, ax = plt.subplots(figsize=(10.5, 4.2))
wdt = 0.27
ax.bar(xs - wdt, free_raw, width=wdt, color=BLUE, label="free-current inverse")
ax.bar(xs, mom_raw, width=wdt, color=ORANGE, label="current-moment (order 3)")
ax.bar(
    xs + wdt,
    mom_corr,
    width=wdt,
    color=ORANGE_LT,
    label="current-moment, ray-origin corrected",
)
for k in (3, 5):
    ax.axvspan(xs[k] - 0.5, xs[k] + 0.5, color="#994455", alpha=0.07, zorder=0)
ax.set_xticks(xs)
ax.set_xticklabels(ang_lbl, fontsize=9)
ax.set_ylabel("median |LCFS radius error| [cm]\n(held-out, n=160)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e3e3e3", lw=0.7)
ax.set_axisbelow(True)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=9, frameon=False
)
ax.set_title(
    "Per-angle LCFS residual — the inboard-corner deficit (shaded) survives "
    "the origin correction",
    fontsize=11.5,
    pad=28,
)
fig.tight_layout()
fig.savefig(OUT / "lcfs-per-angle-deconfound.png", dpi=140)
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))], sep="\n  ")
