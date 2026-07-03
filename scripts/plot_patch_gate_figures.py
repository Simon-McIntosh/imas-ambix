#!/usr/bin/env python
r"""Figures for the patch-current force-balance gate-2 rematch record.

Two figures, both driven from ``imas_ambix/latent/artifacts/patch_gate/``:

``fig-gate-rematch.png`` — the policy/λ selection story: (a) the fixed-λ
sweep on TRAIN shots showing physics weight is monotonically load-bearing,
with the best warm-start / discrepancy arms as reference lines; (b) the
held-out gate (for ``--tag``) per-arm axis-error distribution against the
corrected-Picard and train-mean-baseline reference lines; (c) the winning
arm's per-slice axis error over shot time, colour-coded by shot, showing
early-ramp vs flat-top regime behaviour.

``fig-recovered-closures.png`` — the recovered force-balance closures for
the winning arm at ``--tag``: (a) ``a_k = p'(ψ)``, (b) ``b_k = FF'/μ₀``,
both masked to well-populated ψ-bins, and (c) the integrated ``F²(ψ)``
(via ``structure_residual.integrate_closures``, the SAME routine the
solver uses) for three high-weight-mass slices from distinct shots,
against the vacuum reference and ``F²=0``.

Usage::

    uv run python scripts/plot_patch_gate_figures.py --tag _gate2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from imas_ambix.latent.structure_residual import ClosureFit, integrate_closures

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "imas_ambix" / "latent" / "artifacts" / "patch_gate"
FIG_DIR = REPO / "docs" / "figures" / "patch-current-force-balance"
FIG_DIR.mkdir(parents=True, exist_ok=True)

F_VAC = 0.85 * 0.55  # R0 [m] * Bt0 [T] -> vacuum F = R*Bt reference [T*m]

ARM_COLOR = {"fixed": "#2166ac", "warm-start": "#1b7837", "discrepancy": "#d95f02"}
BASELINE_COLOR = "#636363"
PICARD_COLOR = "#8a3324"
SLICE_PALETTE = ["#08519c", "#238b45", "#cb181d"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def arm_family(arm: str) -> str:
    return arm.split(":")[0]


def npz_slug(arm: str) -> str:
    return arm.replace(":", "-")


def load_json(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text())


def load_arm_npz(arm: str, tag: str):
    return np.load(ARTIFACT_DIR / f"gate_arrays_{npz_slug(arm)}{tag}.npz")


def axis_error_cm(npz) -> np.ndarray:
    return np.asarray(npz["axis_errors"], dtype=np.float64) * 100.0


def baseline_error_cm(npz) -> np.ndarray:
    baseline = np.asarray(npz["baseline"], dtype=np.float64)
    ref = np.asarray(npz["ref"], dtype=np.float64)
    return np.hypot(baseline[:, 0] - ref[:, 0], baseline[:, 1] - ref[:, 1]) * 100.0


# --------------------------------------------------------------------------
# figure 1 — gate rematch
# --------------------------------------------------------------------------


def panel_tune_sweep(ax, ax2, tune: dict) -> None:
    pol = tune["per_policy"]
    lambdas = [0, 1, 3, 10, 30]
    arms = [f"fixed:{lam}" for lam in lambdas]
    skills = [pol[a]["axis_skill"] for a in arms]
    medians_cm = [pol[a]["axis_error_median_m"] * 100.0 for a in arms]

    col = ARM_COLOR["fixed"]
    ax.plot(
        lambdas, skills, color=col, marker="o", lw=1.8, label="axis skill (fixed λ)"
    )
    ax2.plot(
        lambdas,
        medians_cm,
        color=col,
        marker="s",
        lw=1.4,
        ls="--",
        label="median axis error (fixed λ)",
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(v) for v in lambdas])
    ax.set_xlabel("physics weight  λ  (0 = pure data misfit)")
    ax.set_ylabel("axis skill  (0 = train-mean baseline)", color=col)
    ax2.set_ylabel("median axis error  [cm]", color=col)
    ax.tick_params(axis="y", labelcolor=col)
    ax2.tick_params(axis="y", labelcolor=col)

    text_anchors = {"warm-start": (lambdas[1], -14), "discrepancy": (lambdas[3], 6)}
    for family, ref_col in (
        ("warm-start", ARM_COLOR["warm-start"]),
        ("discrepancy", ARM_COLOR["discrepancy"]),
    ):
        cand = {k: v for k, v in pol.items() if arm_family(k) == family}
        best_arm = max(cand, key=lambda k: cand[k]["axis_skill"])
        best = cand[best_arm]
        ax.axhline(best["axis_skill"], color=ref_col, lw=1.2, ls=":", alpha=0.85)
        ax2.axhline(
            best["axis_error_median_m"] * 100.0,
            color=ref_col,
            lw=1.0,
            ls="-.",
            alpha=0.6,
        )
        anchor_x, dy = text_anchors[family]
        ax.annotate(
            f"best {family} ({best_arm})",
            xy=(anchor_x, best["axis_skill"]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=ref_col,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )

    ax.set_title("policy/λ selection on TRAIN shots  (n=100 slices)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")


def panel_heldout_gate(ax, tag: str, eval_json: dict) -> None:
    arms = list(eval_json["per_policy"].keys())
    errs = {a: axis_error_cm(load_arm_npz(a, tag)) for a in arms}
    box_data = [errs[a] for a in arms]
    colors = [ARM_COLOR[arm_family(a)] for a in arms]

    bp = ax.boxplot(
        box_data,
        positions=range(len(arms)),
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "ms": 3, "alpha": 0.4},
    )
    for patch, c in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    for key in ("whiskers", "caps", "medians"):
        for artist, c in zip(
            bp[key], np.repeat(colors, 2 if key != "medians" else 1), strict=True
        ):
            artist.set_color(c)

    ax.set_yscale("log")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_ylabel("axis error  [cm]  (log scale)")

    picard = eval_json["picard_reference"]
    picard_cm = picard["axis_error_median_m"] * 100.0
    ax.axhline(picard_cm, color=PICARD_COLOR, lw=1.4, ls="--")

    base_cm = np.median(baseline_error_cm(load_arm_npz(arms[0], tag)))
    ax.axhline(base_cm, color=BASELINE_COLOR, lw=1.4, ls=":")

    ax.text(
        0.02,
        0.03,
        f"corrected Picard median {picard_cm:.1f} cm "
        f"({picard['n_scored']}/{picard['n_candidate']})\n"
        f"train-mean baseline median {base_cm:.1f} cm",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": "#999999",
            "alpha": 0.9,
        },
    )
    top = max(np.max(e) for e in box_data)
    for i, a in enumerate(arms):
        pp = eval_json["per_policy"][a]
        ax.annotate(
            f"skill {pp['axis_skill']:.2f}\n{pp['n_scored']}/{pp['n_candidate']}",
            xy=(i, top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=colors[i],
        )
    ax.set_ylim(top=top * 2.4)

    run_label = f"tag={tag.lstrip('_')}" if tag else "first run"
    ax.set_title(f"held-out gate ({run_label})  vs corrected-Picard / baseline")


def panel_winner_regime(ax, tag: str, winner: str, diag_json: dict) -> None:
    npz = load_arm_npz(winner, tag)
    errs = axis_error_cm(npz)
    diag = diag_json[winner]["diag"]
    shots = sorted({d["shot"] for d in diag})
    cmap = plt.get_cmap("tab10")
    shot_color = {s: cmap(i % 10) for i, s in enumerate(shots)}

    for s in shots:
        idx = [i for i, d in enumerate(diag) if d["shot"] == s]
        t = [diag[i]["time_s"] for i in idx]
        e = [errs[i] for i in idx]
        order = np.argsort(t)
        t = np.asarray(t)[order]
        e = np.asarray(e)[order]
        ax.plot(t, e, marker="o", ms=3, lw=1.0, color=shot_color[s], label=str(s))

    ax.set_yscale("log")
    ax.set_xlabel("shot time  t  [s]")
    ax.set_ylabel("axis error  [cm]  (log scale)")
    ax.set_title(f"winner arm '{winner}': per-slice error over shot time")
    ax.legend(title="shot", fontsize=7, ncol=2, framealpha=0.85)


def fig_gate_rematch(tag: str) -> None:
    tune = load_json("patch_gate_eval_tune.json")
    eval_json = load_json(f"patch_gate_eval{tag}.json")
    diag_json = load_json(f"patch_gate_diag{tag}.json")

    winner = max(
        eval_json["per_policy"], key=lambda k: eval_json["per_policy"][k]["axis_skill"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    ax0, ax1, ax2 = axes
    ax0b = ax0.twinx()

    panel_tune_sweep(ax0, ax0b, tune)
    panel_heldout_gate(ax1, tag, eval_json)
    panel_winner_regime(ax2, tag, winner, diag_json)

    fig.savefig(FIG_DIR / "fig-gate-rematch.png")
    plt.close(fig)
    print(f"[fig-gate-rematch] winner arm = {winner!r}")


# --------------------------------------------------------------------------
# figure 2 — recovered closures
# --------------------------------------------------------------------------


def pick_representative_slices(closures: list[dict], n: int = 3) -> list[int]:
    totals = [(i, c["shot"], sum(c["weight_mass"])) for i, c in enumerate(closures)]
    totals.sort(key=lambda x: -x[2])
    seen: set[int] = set()
    picked: list[int] = []
    for i, shot, _tot in totals:
        if shot in seen:
            continue
        seen.add(shot)
        picked.append(i)
        if len(picked) == n:
            break
    return picked


def closure_fit_from_json(c: dict) -> ClosureFit:
    return ClosureFit(
        psi_centers=torch.tensor(c["psi_bins"], dtype=torch.float64),
        a_k=torch.tensor(c["a_k"], dtype=torch.float64),
        b_k=torch.tensor(c["b_k"], dtype=torch.float64),
        a_err=torch.tensor(c["a_err"], dtype=torch.float64),
        b_err=torch.tensor(c["b_err"], dtype=torch.float64),
        weight_mass=torch.tensor(c["weight_mass"], dtype=torch.float64),
    )


def mass_mask(weight_mass: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    m = np.asarray(weight_mass, dtype=np.float64)
    return m >= threshold * m.max()


def fig_recovered_closures(tag: str) -> None:
    eval_json = load_json(f"patch_gate_eval{tag}.json")
    diag_json = load_json(f"patch_gate_diag{tag}.json")
    winner = max(
        eval_json["per_policy"], key=lambda k: eval_json["per_policy"][k]["axis_skill"]
    )
    closures = diag_json[winner]["closures"]
    picks = pick_representative_slices(closures, n=3)

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(13.5, 4.6), constrained_layout=True
    )

    f2_mins = np.asarray([c["f2_min"] for c in closures], dtype=np.float64)

    for k, idx in enumerate(picks):
        c = closures[idx]
        color = SLICE_PALETTE[k % len(SLICE_PALETTE)]
        label = f"shot {c['shot']}  t_idx={c['t_index']}"
        psi = np.asarray(c["psi_bins"], dtype=np.float64)
        wm = np.asarray(c["weight_mass"], dtype=np.float64)
        mask = mass_mask(wm)

        a_k = np.asarray(c["a_k"], dtype=np.float64)
        a_err = np.asarray(c["a_err"], dtype=np.float64)
        axa.errorbar(
            psi[mask],
            a_k[mask],
            yerr=a_err[mask],
            color=color,
            marker="o",
            ms=3,
            lw=1.2,
            capsize=2,
            label=label,
        )

        b_k = np.asarray(c["b_k"], dtype=np.float64)
        b_err = np.asarray(c["b_err"], dtype=np.float64)
        axb.errorbar(
            psi[mask],
            b_k[mask],
            yerr=b_err[mask],
            color=color,
            marker="o",
            ms=3,
            lw=1.2,
            capsize=2,
            label=label,
        )

        fit = closure_fit_from_json(c)
        out = integrate_closures(
            fit, psi_axis=c["psi_axis"], psi_boundary=c["psi_boundary"], f_vac=F_VAC
        )
        axc.plot(
            out["psi"],
            out["f_squared"],
            color=color,
            marker="o",
            ms=3,
            lw=1.4,
            label=f"{label}  (min={c['f2_min']:.2f})",
        )

    axa.axhline(0.0, color=BASELINE_COLOR, lw=0.8, alpha=0.6)
    axa.set_xlabel("ψ  [Wb]")
    axa.set_ylabel("recovered  a_k = p′(ψ)  [Pa/Wb]")
    axa.set_title("recovered pressure-gradient closure")
    axa.legend(fontsize=6.5, framealpha=0.85)

    axb.axhline(0.0, color=BASELINE_COLOR, lw=0.8, alpha=0.6)
    axb.set_xlabel("ψ  [Wb]")
    axb.set_ylabel("recovered  b_k = FF′/μ₀  [A/m]")
    axb.set_title("recovered toroidal-field closure")
    axb.legend(fontsize=6.5, framealpha=0.85)

    axc.axhline(
        F_VAC**2,
        color=BASELINE_COLOR,
        lw=1.2,
        ls="--",
        label=f"F²_vac = {F_VAC**2:.3f}",
    )
    axc.axhline(0.0, color=PICARD_COLOR, lw=1.2, ls=":", label="F² = 0")
    axc.set_xlabel("ψ  [Wb]  (boundary → axis)")
    axc.set_ylabel("F²(ψ)  [T²m²]")
    n_pos = int((f2_mins >= 0).sum())
    axc.set_title(
        f"integrated F²(ψ) — MAST paramagnetic\n"
        f"F²_min ≥ 0 in {n_pos}/{len(f2_mins)} slices "
        f"(min={f2_mins.min():.3f}, median={np.median(f2_mins):.3f})"
    )
    axc.legend(fontsize=6.5, framealpha=0.85)

    fig.savefig(FIG_DIR / "fig-recovered-closures.png")
    plt.close(fig)
    print(f"[fig-recovered-closures] winner arm = {winner!r}, picks = {picks}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="_gate2", help="artifact tag suffix, e.g. _gate2")
    args = ap.parse_args()

    fig_gate_rematch(args.tag)
    fig_recovered_closures(args.tag)
    print(f"[done] figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
