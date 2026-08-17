#!/usr/bin/env python
"""Machine-configuration timeline figures for the shot-map plan.

``fig-residual-vs-shot.png`` — census flat-top boundary-shape and axis
residuals binned by shot number, with the documented machine events
overlaid (campaign boundaries, in-vessel RMP coil install/upgrade, the
passive-geometry coarsening, and the sensor-set erosion), so the
degradation can be attributed to hardware history rather than an ad-hoc
shot threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_DIR = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURE_DIR = Path("docs/figures/mast-machine-configuration")

CLASSES = ("connected-dn", "marginal-dn", "limited", "sn-upper", "sn-lower")

# documented machine events (FAIR-MAST campaign fields + literature; see the
# machine-configuration-map plan for citations)
EVENTS = [
    (14708, "M6", "#888"),
    (19031, "M7 — 12 in-vessel RMP coils installed", "#c33"),
    (22163, "amm passive group intermittent in zarr", "#c93"),
    (25404, "M8 — RMP set 12→18", "#c33"),
    (28390, "M9", "#888"),
]


def _rows() -> list[dict]:
    rows: list[dict] = []
    for cls in CLASSES:
        p = ARTIFACT_DIR / f"topology_full_engine_crossval-plain-{cls}.json"
        rows += json.loads(p.read_text()).get("rows", [])
    seen, out = set(), []
    for r in rows:
        key = (r["shot"], r["k"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def residual_vs_shot() -> None:
    rows = [r for r in _rows() if r["phase"] == "flattop"]
    shots = np.array([r["shot"] for r in rows])
    shape = np.array([r.get("shape_dmed_cm", np.nan) for r in rows])
    axis = np.array([r["axis_d_cm"] for r in rows])
    bins = np.arange(11500, 31000, 750)
    mid, med_s, med_a, n = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (shots >= lo) & (shots < hi) & np.isfinite(shape)
        if m.sum() < 8:
            continue
        mid.append(0.5 * (lo + hi))
        med_s.append(float(np.median(shape[m])))
        med_a.append(float(np.median(axis[m])))
        n.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(
        mid,
        med_s,
        "o-",
        color="#268",
        lw=1.4,
        ms=4,
        label="boundary shape (own-axis) median",
    )
    ax.plot(mid, med_a, "s--", color="#c66", lw=1.2, ms=4, label="axis distance median")
    for s, lab, col in EVENTS:
        ax.axvline(s, color=col, lw=1.0, ls=":" if col == "#888" else "-", alpha=0.85)
        ax.text(
            s,
            ax.get_ylim()[1] * 0.98,
            f" {lab}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=6.5,
            color=col,
        )
    ax.set_xlabel("shot number")
    ax.set_ylabel("flat-top median residual vs EFIT [cm]")
    ax.set_title(
        "census residuals vs shot number, machine events overlaid "
        "(750-shot bins, all classes pooled)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIGURE_DIR / "fig-residual-vs-shot.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out} ({len(mid)} bins, {sum(n)} slices)")


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    residual_vs_shot()
