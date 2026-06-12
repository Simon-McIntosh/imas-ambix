"""Render the forecasting / ablation / attribution evaluation figures.

Three figures, one per evaluation artifact, written under
``docs/figures/camera-dynamics-wm/``:

``fig-cdw-horizon-table.png``
    Per-horizon (10/50/200 ms) top-1 for dynamics / persistence / per-frame
    baseline with paired-CI error bars, plus the per-horizon valid-window
    counts — the W2 forward-horizon table.

``fig-cdw-cond-ablation.png``
    none / ip_ne / full held-out + frontier masked-token NLL and top-1 at the
    matched 8000-step budget — does conditioning matter, and where.

``fig-cdw-puff-attribution.png``
    Pooled command-vs-predicted-activity scatter + per-shot correlation
    histogram + the zeroed-puff counterfactual delta with its CI — the
    falsification probe (positive attribution or honest null).

Run::

    uv run python -m imas_ambix.camdyn.make_eval_figures horizon
    uv run python -m imas_ambix.camdyn.make_eval_figures ablation
    uv run python -m imas_ambix.camdyn.make_eval_figures puff
    uv run python -m imas_ambix.camdyn.make_eval_figures all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FIG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "figures"
    / "camera-dynamics-wm"
)


# ---------------------------------------------------------------------------
# Horizon forecasting table
# ---------------------------------------------------------------------------


def make_horizon_figure(artifact: Path | None = None, out: Path | None = None) -> Path:
    artifact = artifact or (ARTIFACT_DIR / "horizon_table.json")
    art = json.loads(Path(artifact).read_text())
    horizons = [float(h) for h in art["horizons_ms"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # show the matched (populated) regime; native is reported in the artifact
    regime = "matched"
    table = art[regime]["table"]
    x = np.arange(len(horizons))
    width = 0.26

    ax = axes[0]
    dyn_y, dyn_lo, dyn_hi, per_y, base_y, valid_n = [], [], [], [], [], []
    for h in horizons:
        cell = table[str(h)]
        if cell.get("valid_windows"):
            dyn_y.append(cell["dynamics"]["top1"])
            per_y.append(cell["persistence"]["top1"])
            base_y.append(cell["baseline"]["top1"])
            ci = cell["dynamics_vs_persistence_top1"]
            # error bar around the dynamics bar: half-width of the paired CI
            half = 0.5 * (ci["hi"] - ci["lo"])
            dyn_lo.append(half)
            dyn_hi.append(half)
            valid_n.append(cell["valid_windows"])
        else:
            dyn_y.append(np.nan)
            per_y.append(np.nan)
            base_y.append(np.nan)
            dyn_lo.append(0.0)
            dyn_hi.append(0.0)
            valid_n.append(0)

    ax.bar(
        x - width,
        dyn_y,
        width,
        yerr=[dyn_lo, dyn_hi],
        capsize=4,
        label="dynamics",
        color="C0",
    )
    ax.bar(x, per_y, width, label="persistence", color="C1")
    ax.bar(x + width, base_y, width, label="per-frame baseline", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:.0f} ms" for h in horizons])
    ax.set_ylabel("masked-token top-1 accuracy")
    ax.set_title(f"forward-horizon top-1 ({regime} cadence)")
    ax.legend(fontsize=8)
    for xi, n in zip(x, valid_n, strict=False):
        ax.annotate(
            f"n={n}",
            (xi, 0.01),
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.3",
        )

    # right panel: per-horizon paired-CI deltas (dynamics minus reference)
    ax2 = axes[1]
    for key, color, lbl in (
        ("dynamics_vs_persistence_top1", "C1", "dyn − persistence"),
        ("dynamics_vs_baseline_top1", "C2", "dyn − baseline"),
    ):
        ys, los, his, xs = [], [], [], []
        for xi, h in zip(x, horizons, strict=False):
            cell = table[str(h)]
            if cell.get("valid_windows") and key in cell:
                ci = cell[key]
                ys.append(ci["mean"])
                los.append(ci["mean"] - ci["lo"])
                his.append(ci["hi"] - ci["mean"])
                xs.append(xi)
        if xs:
            ax2.errorbar(
                np.array(xs) + (0.08 if "baseline" in key else -0.08),
                ys,
                yerr=[los, his],
                fmt="o",
                capsize=4,
                color=color,
                label=lbl,
            )
    ax2.axhline(0.0, color="0.5", lw=1, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h:.0f} ms" for h in horizons])
    ax2.set_ylabel("paired top-1 delta (positive = dynamics better)")
    ax2.set_title("dynamics advantage with 95% paired CI")
    ax2.legend(fontsize=8)

    fig.suptitle(
        "camera-dynamics-wm — forward-horizon forecasting (W2): "
        "dynamics vs persistence vs per-frame baseline",
        fontsize=12,
    )
    out = out or (FIG_DIR / "fig-cdw-horizon-table.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Conditioning ablation
# ---------------------------------------------------------------------------


def make_ablation_figure(artifact: Path | None = None, out: Path | None = None) -> Path:
    artifact = artifact or (ARTIFACT_DIR / "cond_ablation.json")
    art = json.loads(Path(artifact).read_text())
    arms = ["none", "ip_ne", "full"]
    colors = {"none": "C3", "ip_ne": "C1", "full": "C0"}

    sections = [("held_out", "held-out (mixture mask)")]
    if "frontier_half" in art.get("named_geometry", {}):
        sections.append(("frontier_half", "frontier (forecasting mode)"))

    fig, axes = plt.subplots(
        2, len(sections), figsize=(5.5 * len(sections), 8), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T if len(sections) == 1 else axes

    for col, (sec_key, sec_title) in enumerate(sections):
        sec = (
            art["held_out"] if sec_key == "held_out" else art["named_geometry"][sec_key]
        )
        x = np.arange(len(arms))
        nll = [sec[a]["masked_nll"]["mean"] for a in arms]
        top1 = [sec[a]["masked_top1"]["mean"] for a in arms]

        ax_nll = axes[0, col] if axes.ndim == 2 else axes[0]
        ax_nll.bar(x, nll, color=[colors[a] for a in arms])
        ax_nll.set_xticks(x)
        ax_nll.set_xticklabels(arms)
        ax_nll.set_ylabel("masked-token NLL (nats, lower better)")
        ax_nll.set_title(f"{sec_title} — NLL")

        ax_acc = axes[1, col] if axes.ndim == 2 else axes[1]
        ax_acc.bar(x, top1, color=[colors[a] for a in arms])
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(arms)
        ax_acc.set_ylabel("masked-token top-1 (higher better)")
        ax_acc.set_title(f"{sec_title} — top-1")

    # annotate the held-out verdict
    v = art.get("verdict", {})
    txt = (
        f"full beats none (NLL): {v.get('full_beats_none_nll')}\n"
        f"ip_ne beats none (NLL): {v.get('ip_ne_beats_none_nll')}\n"
        f"frontier full beats none (NLL): {v.get('frontier_full_beats_none_nll')}"
    )
    fig.suptitle(
        "camera-dynamics-wm — conditioning ablation (matched 8000-step budget)\n" + txt,
        fontsize=11,
    )
    out = out or (FIG_DIR / "fig-cdw-cond-ablation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Gas-puff attribution
# ---------------------------------------------------------------------------


def make_puff_figure(artifact: Path | None = None, out: Path | None = None) -> Path:
    artifact = artifact or (ARTIFACT_DIR / "puff_attribution.json")
    art = json.loads(Path(artifact).read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    # panel 1: per-shot correlation histogram
    ax = axes[0]
    per_shot = art.get("per_shot", {})
    corrs = [
        v["pearson_cmd_vs_activity"]
        for v in per_shot.values()
        if np.isfinite(v.get("pearson_cmd_vs_activity", np.nan))
    ]
    if corrs:
        ax.hist(corrs, bins=15, color="C0", alpha=0.8)
        ax.axvline(0.0, color="0.4", ls="--", lw=1)
        psc = art.get("per_shot_correlation", {})
        ax.axvline(
            psc.get("median_pearson", np.nan),
            color="C3",
            lw=2,
            label=f"median r={psc.get('median_pearson', float('nan')):.3f}",
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no puff shots", ha="center", va="center")
    ax.set_xlabel("per-shot Pearson r (puff command vs predicted activity)")
    ax.set_ylabel("shots")
    ax.set_title("timing correlation by shot")

    # panel 2: pooled scatter (command vs predicted activity) is not stored;
    # show the pooled correlation + n as text, plus per-shot means scatter
    ax2 = axes[1]
    means_act = [v.get("mean_counterfactual_delta", np.nan) for v in per_shot.values()]
    ns = [v.get("n_frames", 0) for v in per_shot.values()]
    if per_shot:
        ax2.scatter(ns, means_act, color="C2", alpha=0.7)
        ax2.axhline(0.0, color="0.4", ls="--", lw=1)
    ax2.set_xlabel("frames per shot")
    ax2.set_ylabel("mean counterfactual delta (with-aga − zeroed)")
    ax2.set_title("per-shot counterfactual effect")

    # panel 3: pooled counterfactual delta with CI + verdict
    ax3 = axes[2]
    pooled = art.get("pooled", {})
    if pooled:
        d = pooled["counterfactual_delta_mean"]
        lo, hi = pooled["counterfactual_delta_ci"]
        ax3.errorbar(
            [0],
            [d],
            yerr=[[d - lo], [hi - d]],
            fmt="o",
            capsize=6,
            color="C0",
            markersize=9,
        )
        ax3.axhline(0.0, color="0.4", ls="--", lw=1)
        ax3.set_xlim(-0.5, 0.5)
        ax3.set_xticks([])
        ax3.set_ylabel("pooled counterfactual delta")
        pooled_r = pooled.get("pearson_cmd_vs_activity", float("nan"))
        ax3.set_title(
            f"pooled: r={pooled_r:.3f}  Δ={d:+.4f}\n"
            f"clear of zero: {pooled.get('counterfactual_clear_of_zero')}"
        )
    else:
        ax3.text(0.5, 0.5, "untested", ha="center", va="center")
        ax3.set_xticks([])

    verdict = art.get("verdict", {})
    fig.suptitle(
        "camera-dynamics-wm — gas-puff attribution probe "
        f"(attribution: {verdict.get('attribution', 'n/a')}; "
        f"n_puff_shots={art.get('n_puff_shots', 0)}, "
        f"n_windows={art.get('n_windows_used', 0)})",
        fontsize=12,
    )
    out = out or (FIG_DIR / "fig-cdw-puff-attribution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="camdyn evaluation figures")
    p.add_argument(
        "which",
        choices=["horizon", "ablation", "puff", "all"],
        help="which figure(s) to render",
    )
    p.add_argument("--artifact", default=None, help="override artifact JSON path")
    p.add_argument("--out", default=None, help="override output PNG path")
    args = p.parse_args(argv)
    art = Path(args.artifact) if args.artifact else None
    out = Path(args.out) if args.out else None
    if args.which in ("horizon", "all"):
        make_horizon_figure(art if args.which == "horizon" else None, out)
    if args.which in ("ablation", "all"):
        make_ablation_figure(art if args.which == "ablation" else None, out)
    if args.which in ("puff", "all"):
        make_puff_figure(art if args.which == "puff" else None, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
