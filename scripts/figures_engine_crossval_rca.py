#!/usr/bin/env python
"""Evidence figures for the census-crossval scoring RCA.

Two figures:

* ``fig-frame-artifact-scatter.png`` — per-slice boundary residual (engine-
  frame) against the engine-vs-EFIT axis distance.  The rows come from the
  re-scored artifacts, whose ``radii_dmed_cm`` is the same engine-frame
  quantity the pre-fix harness reported (a ``--prefix-snapshot`` directory
  of the original single-frame artifacts is accepted as an alternate
  source).  A ray-fan rendered about a displaced origin projects the
  displacement into the radii at ~|cos θ| ≈ 0.7 on average, so
  residual ≈ 0.7·axis-distance is the signature that the metric was
  measuring axis placement, not boundary shape.

* ``fig-heldout-gate-repro.png`` — the 112-shot held-out MSE gate re-run:
  the archived result against fresh runs of the current tree and of the
  §4-era tree, engine vs persistence pitch RMSE (identical bars = the solve
  chain is unchanged; the census verdict shift is the ruler, not the engine).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path("docs/figures/connectivity-topology-reader")
ARTIFACT_DIR = Path("imas_ambix/latent/artifacts/patch_gate")
CLASSES = ("connected-dn", "marginal-dn", "limited", "sn-upper", "sn-lower")
CLASS_COLORS = {
    "connected-dn": "#268",
    "marginal-dn": "#89b",
    "limited": "#9b6",
    "sn-upper": "#e90",
    "sn-lower": "#c66",
}


def _plain_rows(source_dir: Path, cname: str) -> list[dict]:
    path = source_dir / f"topology_full_engine_crossval-plain-{cname}.json"
    return json.loads(path.read_text()).get("rows", [])


def frame_artifact_scatter(source_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    all_ax, all_lc = [], []
    for cname in CLASSES:
        rows = _plain_rows(source_dir, cname)
        a = np.array([r["axis_d_cm"] for r in rows])
        m = np.array([r["radii_dmed_cm"] for r in rows])
        fin = np.isfinite(a) & np.isfinite(m)
        ax.scatter(a[fin], m[fin], s=5, alpha=0.35,
                   color=CLASS_COLORS[cname], label=cname, linewidths=0)
        all_ax.append(a[fin])
        all_lc.append(m[fin])
    a = np.concatenate(all_ax)
    m = np.concatenate(all_lc)
    r = float(np.corrcoef(a, m)[0, 1])
    xs = np.linspace(0, np.percentile(a, 99), 50)
    ax.plot(xs, 0.7 * xs, "k--", lw=1.2,
            label="0.7 × axis distance (|cos θ| ray-fan projection)")
    ax.set_xlim(0, np.percentile(a, 99))
    ax.set_ylim(0, np.percentile(m, 99))
    ax.set_xlabel("engine axis distance to EFIT axis [cm]")
    ax.set_ylabel("engine-frame boundary residual (the pre-fix G-E2 metric) [cm]")
    ax.set_title(
        f"the pre-fix boundary residual tracks the axis error (r = {r:.2f})\n"
        "— the metric measured axis placement, not boundary shape",
        fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = FIGURE_DIR / "fig-frame-artifact-scatter.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out} (n={a.size}, r={r:.3f})")


def _combine_quarters(paths: list[Path]) -> tuple[float, float, int] | None:
    """n-weighted mean of the per-quarter shot-mean RMSEs (the harness
    aggregates as the mean over shots, so the recombination is exact)."""
    tot = 0
    acc_e = acc_p = 0.0
    for p in paths:
        if not p.exists():
            return None
        s = json.loads(p.read_text())["summary"]
        n = int(s["n_shots_scored"])
        tot += n
        acc_e += float(s["engine_pitch_rmse"]) * n
        acc_p += float(s["persistence_pitch_rmse_live"]) * n
    return acc_e / tot, acc_p / tot, tot


def heldout_gate_repro() -> None:
    runs: list[tuple[str, object]] = [
        ("archived §4 result", ARTIFACT_DIR / "heldout_mse_gate-v0.json"),
        ("re-run, current tree",
         [ARTIFACT_DIR / f"heldout_mse_gate-repro-head-q{q}.json"
          for q in (1, 2, 3, 4)]),
        ("re-run, gate-landing-commit tree",
         [ARTIFACT_DIR / f"heldout_mse_gate-repro-s4-q{q}.json"
          for q in (1, 2, 3, 4)]),
    ]
    labels, eng, per, n_scored = [], [], [], []
    for label, src in runs:
        if isinstance(src, list):
            got = _combine_quarters(src)
            if got is None:
                print(f"skip {label}: quarter artifacts missing")
                continue
            e, p, n = got
        else:
            if not src.exists():
                print(f"skip {label}: {src} missing")
                continue
            s = json.loads(src.read_text())["summary"]
            e = float(s["engine_pitch_rmse"])
            p = float(s["persistence_pitch_rmse_live"])
            n = int(s["n_shots_scored"])
        labels.append(label)
        eng.append(e)
        per.append(p)
        n_scored.append(n)
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - 0.18, per, width=0.34, color="#bbb", label="persistence (live)")
    ax.bar(x + 0.18, eng, width=0.34, color="#268", label="engine")
    for xi, (e, n) in zip(x, zip(eng, n_scored, strict=True), strict=True):
        ax.text(xi + 0.18, e, f"{e:.3f}\nn={n}", ha="center", va="bottom",
                fontsize=7)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("held-out MSE pitch RMSE [rad]")
    ax.set_title("the 112-shot held-out gate reproduces on the current tree —"
                 " the solve chain is unchanged", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURE_DIR / "fig-heldout-gate-repro.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rows-dir", type=Path, default=ARTIFACT_DIR,
        help="directory holding the plain-arm per-class artifacts "
             "(pass a snapshot of the pre-fix single-frame artifacts to "
             "reproduce the RCA figure from the original scoring)")
    args = ap.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame_artifact_scatter(args.rows_dir)
    heldout_gate_repro()
