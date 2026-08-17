#!/usr/bin/env python
"""Architecture map of the GS forward / physics-spine EM infrastructure.

A read-only diagnostic figure for the consolidation review: it lays out the
electromagnetic stack in three tiers (kernels -> coupling assembly ->
machine-description + consumers) and annotates the duplicated wrappers, the
shared primitives, the superseded boundary reads, and the readers that are
implemented but carry no product caller.  No project code is imported — the
layout is hand-encoded from the survey so the figure never drifts if a module
moves, which also means a box only disappears when someone edits this file.

Output: docs/figures/physics-spine-consolidation/fig-em-stack-map.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/figures/physics-spine-consolidation/fig-em-stack-map.png"

LIVE = "#2f7f4f"      # green — core/live
SHARED = "#1565c0"    # blue — shared primitive
DUP = "#c77800"       # amber — duplicated wrapper
DEAD = "#b0392b"      # red — dead / test-only / superseded
UNWIRED = "#6a3d9a"   # purple — implemented but no product caller
BG = "#f4f4f2"


def box(ax, x, y, w, h, title, sub, color, *, alpha=0.14):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.6, edgecolor=color, facecolor=color, alpha=alpha,
        )
    )
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top",
            fontsize=8.4, fontweight="bold", color="#1a1a1a")
    if sub:
        ax.text(x + w / 2, y + h - 0.062, sub, ha="center", va="top",
                fontsize=6.7, color="#333", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#666", style="-|>", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=color, linestyle=ls,
        shrinkA=2, shrinkB=2, alpha=0.8))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.add_patch(FancyBboxPatch((0.006, 0.006), 0.988, 0.988,
                 boxstyle="round,pad=0.004", facecolor=BG, edgecolor="none"))

    ax.text(0.5, 0.972, "GS forward / physics-spine — electromagnetic stack map",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(0.5, 0.945,
            "one shared finite-area kernel · duplicated assembly wrappers · "
            "layered boundary reads · two machine readers, one of them uncalled",
            ha="center", fontsize=8.6, color="#444", style="italic")

    # ---- Tier labels ----
    for ty, lab in ((0.80, "CONSUMERS / SOLVE"),
                    (0.55, "COUPLING ASSEMBLY  (gs/operator · latent/{boundary_disc,temporal_operator})"),
                    (0.28, "MACHINE DESCRIPTION"),
                    (0.06, "EM KERNELS  (gs/)")):
        ax.text(0.012, ty, lab, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color="#888", rotation=0)

    # ---- Tier 4: consumers ----
    box(ax, 0.05, 0.83, 0.16, 0.10, "gs_solve.py", "interior force-balance\nsolve (2659 LOC)", LIVE)
    box(ax, 0.23, 0.83, 0.14, 0.10, "engine.py", "learned operator\nloss terms", LIVE)
    box(ax, 0.39, 0.83, 0.19, 0.10, "spine_bench/runner", "frozen-metric\nharness", LIVE)
    box(ax, 0.60, 0.87, 0.17, 0.055, "connectivity_boundary", "in-solve LCFS (LIVE)", LIVE, alpha=0.10)
    box(ax, 0.60, 0.805, 0.17, 0.055, "boundary_disc.disc_read", "harness read (LIVE)", LIVE, alpha=0.10)
    box(ax, 0.79, 0.87, 0.175, 0.055, "boundary_moment", "read SUPERSEDED\n(basis reused)", DEAD, alpha=0.10)
    box(ax, 0.79, 0.805, 0.175, 0.055, "boundary_harmonic", "read SUPERSEDED\n(anchor reused)", DEAD, alpha=0.10)

    # ---- Tier 3: coupling assembly (the duplication tier) ----
    box(ax, 0.05, 0.44, 0.20, 0.135,
        "build_operator", "G_pf / G_passive / G_plasma\npoint-observer, finite-area src", DUP)
    box(ax, 0.27, 0.44, 0.21, 0.135,
        "passive_coupling_matrices", "a_sens / g_grid\npoint-observer  (delegated to\nby build_passive_circuit_system)", DUP)
    box(ax, 0.50, 0.44, 0.22, 0.135,
        "build_passive_circuit_system", "L (P×P) + m_coil_circ\ntwo-section OBSERVER  (distinct,\nintentional physics)", DUP)
    box(ax, 0.74, 0.44, 0.21, 0.135,
        "build_drive_linkage", "lam (C×C)\ntwo-section observer +\n/n_merge reciprocal (distinct)", DUP)

    # duplication badge
    ax.text(0.50, 0.615,
            "DUPLICATED across these wrappers → "
            "classify_circuits+by_circ (×8) · channel-merge+solenoid-scale (×3) · "
            "sensor projection ψ|B·n̂ (×5) · PolygonSection override (×3)",
            ha="center", va="center", fontsize=6.9, color=DUP, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff6e8", ec=DUP, lw=1.0))

    # ---- Tier 2: machine description ----
    box(ax, 0.05, 0.15, 0.20, 0.10, "geometry.py", "GeometryTable records\nand geometry utilities", LIVE)
    box(ax, 0.27, 0.15, 0.15, 0.10, "circuits.py", "pfSystems.xml table\n(Active/Case)", LIVE)
    box(ax, 0.44, 0.15, 0.17, 0.10, "operator.classify_circuits", "role assignment\n(only assigner)", LIVE)
    box(ax, 0.63, 0.21, 0.20, 0.075, "description_reader",
        "declared map — every product caller", LIVE, alpha=0.10)
    box(ax, 0.63, 0.115, 0.20, 0.075, "artifact_geometry.py",
        "artifact read — no product caller", UNWIRED, alpha=0.12)
    ax.text(0.78, 0.335,
            "parallel coil→channel map:\n_PF_COIL_AMC ↔ ActiveCircuit (pinned)",
            ha="center", va="center", fontsize=6.4, color=DUP,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff6e8", ec=DUP, lw=0.8))

    # ---- Tier 1: kernels ----
    box(ax, 0.05, 0.02, 0.14, 0.085, "greens_psi / bz_br", "point ring\n(zero-size limit)", SHARED)
    box(ax, 0.21, 0.02, 0.17, 0.085, "cylinder_greens\n→ hybrid_greens", "finite-area RECT\nPRIMARY shared", SHARED, alpha=0.20)
    box(ax, 0.40, 0.02, 0.15, 0.085, "polygon_greens", "finite-area POLYGON\n(rect = 4-vertex)", SHARED)
    box(ax, 0.57, 0.02, 0.15, 0.085, "filaments3d", "3D line/arc\n(EFCC; rdp test-only)", LIVE, alpha=0.10)

    # ---- flows ----
    arrow(ax, 0.15, 0.44, 0.15, 0.25, DUP)          # assembly -> machine desc
    arrow(ax, 0.37, 0.44, 0.34, 0.25, DUP)
    arrow(ax, 0.60, 0.44, 0.52, 0.25, DUP)
    arrow(ax, 0.15, 0.15, 0.27, 0.105, SHARED)      # machine desc -> kernels
    arrow(ax, 0.15, 0.83, 0.15, 0.575, LIVE)        # consumers -> assembly
    arrow(ax, 0.30, 0.83, 0.34, 0.575, LIVE)
    arrow(ax, 0.47, 0.83, 0.58, 0.575, LIVE)
    # kernels used by assembly (shared primitive emphasis)
    arrow(ax, 0.29, 0.105, 0.13, 0.44, SHARED, lw=1.6)
    arrow(ax, 0.29, 0.105, 0.36, 0.44, SHARED, lw=1.6)
    arrow(ax, 0.29, 0.105, 0.60, 0.44, SHARED, lw=1.6)

    # ---- legend ----
    handles = [
        ("core / live", LIVE),
        ("shared primitive", SHARED),
        ("duplicated wrapper", DUP),
        ("superseded read", DEAD),
        ("no product caller", UNWIRED),
    ]
    # place legend swatches along the top
    lx = 0.055
    for lab, c in handles:
        ax.add_patch(FancyBboxPatch((lx, 0.917), 0.016, 0.014,
                     boxstyle="round,pad=0.002", fc=c, ec=c, alpha=0.5))
        ax.text(lx + 0.02, 0.924, lab, fontsize=6.6, va="center", color="#222")
        lx += 0.028 + 0.006 * len(lab)

    fig.savefig(OUT, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
