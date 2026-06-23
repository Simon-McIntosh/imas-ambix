"""Measure the controllability gate's STATISTICAL POWER as a function of the
number of random rollouts per shot.

The robust cohort gate scores each held-out shot by a noise-floor-NORMALISED
ratio (true-vs-random pixel-L1 / random-vs-random floor) and aggregates a cohort
mean ratio + bootstrap CI.  At n_random=3 that CI spans roughly [0.8, 3.0] —
too wide to tell two checkpoints apart (baseline ~1.1 vs joint-gen ~1.7 have
overlapping CIs).  This driver re-evals the SAME fixed cohort at n_random=3 AND
n_random=10 for the baseline + joint-gen checkpoints and asks:

  * do the cohort CIs TIGHTEN when n_random rises?
  * does the baseline-vs-joint-gen gap become RESOLVABLE (non-overlapping CIs)?
  * which variance dominates — WITHIN-shot sampling noise (more rollouts help)
    or ACROSS-shot heterogeneity (more rollouts do not; the pass-FRACTION over a
    driveable-enriched cohort is the lever)?

It loads each model ONCE and decodes its rollouts in one-VQ-pass batches
(AGENTS.md §2b).  scancel + ordinary --time are fine (AGENTS.md §2a).  NOT
production code — a self-contained analysis driver staged on the shared FS for
the gate-power investigation.  Run inside a betelgeuse srun, venv active,
HF offline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from imas_ambix.worldmodel.controllable_eval import (  # noqa: E402
    EvalConfig,
    _resolve_eval_modalities,
    multi_shot_delta_nm,
)
from imas_ambix.worldmodel.controllable_train import (  # noqa: E402
    load_controllable_model_from_checkpoint,
)
from imas_ambix.worldmodel.gate_cohort import load_cohort  # noqa: E402
from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: E402
    SpacetimeWindowConfig,
)

TOKEN_ROOT = Path("/work/projects/imas_gpu/mast-tokens")
CAMERA = "rbb"
DEVICE = "cuda"
HORIZON_S = 0.25
COHORT_JSON = Path("/work/projects/imas_gpu/worldmodel/gate_cohort.json")
OUT_ROOT = Path("/work/projects/imas_gpu/worldmodel/gate_power")
FIG_DIR = Path(
    "/home/ITER/mcintos/Code/imas-ambix/docs/figures/joint-multimodal-plasma-wm"
)

# the two n_random settings to compare (low = current default, high = the raise).
N_RANDOM = [int(x) for x in os.environ.get("WM_N_RANDOM_LIST", "3,10").split(",")]

# checkpoints to re-eval — baseline + joint-gen are the core comparison; the
# extra two run if WM_FULL is set (time permitting).
CKPT = {
    "1221741": Path(
        "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221741/latest.pt"
    ),
    "1221834": Path(
        "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221834/latest.pt"
    ),
}
if os.environ.get("WM_FULL"):
    CKPT["1221883"] = Path(
        "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221883/latest.pt"
    )
    CKPT["1222038"] = Path(
        "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1222038/latest.pt"
    )
LABEL = {
    "1221741": "baseline",
    "1221834": "joint-gen",
    "1221883": "diag-off",
    "1222038": "13-stream",
}


def window_cfg() -> SpacetimeWindowConfig:
    return SpacetimeWindowConfig(
        n_frames=24,
        n_plan=8,
        context_frames=8,
        frame_stride=1,
        target_horizon_s=HORIZON_S,
    )


def make_cfg(modalities, held_out, n_random) -> EvalConfig:
    return EvalConfig(
        held_out=tuple(held_out),
        n_random=int(n_random),
        perturb_scale=0.3,
        n_signal_steps=4,
        n_act_steps=8,
        modalities=modalities,
        robust_gate=True,
        ratio_threshold=1.5,
        n_bootstrap=2000,
        reject_collapsed=True,
        window=window_cfg(),
    )


def setup_torch():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def eval_checkpoint(ckpt_key: str, cohort: list[int]) -> dict:
    """Eval one checkpoint at every N_RANDOM; model loaded once for all settings."""
    print(f"=== eval {ckpt_key} ({LABEL[ckpt_key]}) ===", flush=True)
    model, payload = load_controllable_model_from_checkpoint(
        CKPT[ckpt_key], map_location=DEVICE
    )
    model.eval()
    by_n: dict[str, dict] = {}
    try:
        modalities = _resolve_eval_modalities("auto", payload)
        for n in N_RANDOM:
            cfg = make_cfg(modalities, cohort, n)
            out_dir = OUT_ROOT / ckpt_key / f"nr{n}"
            summary = multi_shot_delta_nm(
                model,
                config=cfg,
                camera=CAMERA,
                token_root=TOKEN_ROOT,
                device=DEVICE,
                out_json=out_dir / "heldout_delta_nm.json",
                work_dir=out_dir / "_dnm",
                decode=True,
            )
            vd = summary.get("variance_decomposition", {})
            print(
                f"{ckpt_key} n_random={n}: verdict={summary['verdict']} "
                f"pass_frac={summary['pass_fraction']:.2f} "
                f"mean_ratio={summary['mean_normalised_ratio']:.2f} "
                f"median={summary.get('median_normalised_ratio', 0.0):.2f} "
                f"CI=[{summary['ratio_ci_lo']:.2f},{summary['ratio_ci_hi']:.2f}] "
                f"CIwidth={summary['ratio_ci_hi'] - summary['ratio_ci_lo']:.2f} "
                f"within_std={summary.get('mean_within_shot_ratio_std')} "
                f"across/within={vd.get('across_over_within')} "
                f"collapsed={summary['n_random_collapsed_total']}",
                flush=True,
            )
            print(f"    {vd.get('interpretation', '')}", flush=True)
            by_n[str(n)] = summary
        return by_n
    finally:
        del model
        torch.cuda.empty_cache()


def make_figure(per_ckpt: dict, cohort: list[int]):
    keys = list(CKPT.keys())
    nr_lo, nr_hi = N_RANDOM[0], N_RANDOM[-1]
    fig = plt.figure(figsize=(14.5, 9.0), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    palette = {
        "1221741": "#4c78a8",
        "1221834": "#e45756",
        "1221883": "#54a24b",
        "1222038": "#b279a2",
    }

    # --- panel 0: mean ratio + CI, low vs high n_random, side by side ---
    ax0 = fig.add_subplot(gs[0, 0])
    width = 0.34
    x = np.arange(len(keys))
    for off, n in ((-0.5, nr_lo), (0.5, nr_hi)):
        mr = [per_ckpt[k][str(n)]["mean_normalised_ratio"] for k in keys]
        lo = [per_ckpt[k][str(n)]["ratio_ci_lo"] for k in keys]
        hi = [per_ckpt[k][str(n)]["ratio_ci_hi"] for k in keys]
        yerr = np.array(
            [
                [m - lo_i for m, lo_i in zip(mr, lo, strict=True)],
                [hi_i - m for m, hi_i in zip(mr, hi, strict=True)],
            ]
        )
        ax0.errorbar(
            x + off * width,
            mr,
            yerr=yerr,
            fmt="o",
            ms=8,
            capsize=6,
            elinewidth=2,
            label=f"n_random={n}",
            color="#333" if n == nr_lo else "#c44",
        )
    ax0.axhline(1.0, ls="--", color="#888", lw=1.2, label="noise floor (1.0)")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{k}\n{LABEL[k]}" for k in keys], fontsize=8)
    ax0.set_ylabel("cohort mean normalised ratio (tvr / floor)")
    ax0.set_title(
        f"mean ratio + 95% bootstrap CI: n_random={nr_lo} vs {nr_hi}\n"
        "(do the CIs tighten? do baseline & joint-gen separate?)",
        fontsize=10,
    )
    ax0.legend(fontsize=8)

    # --- panel 1: CI WIDTH low vs high n_random ---
    ax1 = fig.add_subplot(gs[0, 1])
    bw = 0.34
    for off, n in ((-0.5, nr_lo), (0.5, nr_hi)):
        w = [
            per_ckpt[k][str(n)]["ratio_ci_hi"] - per_ckpt[k][str(n)]["ratio_ci_lo"]
            for k in keys
        ]
        bars = ax1.bar(
            x + off * bw,
            w,
            bw,
            label=f"n_random={n}",
            color="#aab" if n == nr_lo else "#e8a",
            edgecolor="#333",
            linewidth=0.4,
        )
        for bi, v in zip(bars, w, strict=True):
            ax1.text(
                bi.get_x() + bi.get_width() / 2,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{k}\n{LABEL[k]}" for k in keys], fontsize=8)
    ax1.set_ylabel("bootstrap CI width (hi - lo)")
    ax1.set_title("CI width shrinks with more random rollouts?", fontsize=10)
    ax1.legend(fontsize=8)

    # --- panel 2: variance decomposition (within vs across) at high n_random ---
    ax2 = fig.add_subplot(gs[1, 0])
    within = []
    across = []
    for k in keys:
        vd = per_ckpt[k][str(nr_hi)].get("variance_decomposition", {})
        within.append(float(vd.get("mean_within_shot_variance", np.nan)))
        across.append(float(vd.get("across_shot_variance", np.nan)))
    vw = 0.34
    ax2.bar(
        x - vw / 2,
        within,
        vw,
        label="within-shot (sampling noise)",
        color="#7aa",
        edgecolor="#333",
        linewidth=0.4,
    )
    ax2.bar(
        x + vw / 2,
        across,
        vw,
        label="across-shot (heterogeneity)",
        color="#d88",
        edgecolor="#333",
        linewidth=0.4,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{k}\n{LABEL[k]}" for k in keys], fontsize=8)
    ax2.set_ylabel("variance of per-shot ratio")
    ax2.set_title(
        f"variance decomposition @ n_random={nr_hi}\n"
        "(across >> within -> heterogeneity dominates, more rollouts won't help)",
        fontsize=10,
    )
    ax2.legend(fontsize=8)

    # --- panel 3: per-shot ratio DISTRIBUTION (the stable signal) ---
    ax3 = fig.add_subplot(gs[1, 1])
    for k in keys:
        rs = per_ckpt[k][str(nr_hi)].get("per_shot_ratios_sorted", [])
        ax3.plot(
            np.arange(len(rs)),
            rs,
            "-o",
            ms=3,
            label=f"{LABEL[k]} (pass {per_ckpt[k][str(nr_hi)]['pass_fraction']:.2f})",
            color=palette.get(k, "#333"),
        )
    ax3.axhline(1.5, ls="--", color="#888", lw=1, label="ratio_threshold (1.5)")
    ax3.axhline(1.0, ls=":", color="#bbb", lw=1)
    ax3.set_xlabel("shot rank (sorted by ratio)")
    ax3.set_ylabel("per-shot normalised ratio")
    ax3.set_yscale("log")
    ax3.set_title(
        f"per-shot ratio distribution @ n_random={nr_hi}\n"
        "(bimodal/heavy-tailed = a few driveable shots among many flat ones)",
        fontsize=10,
    )
    ax3.legend(fontsize=8)

    cohort_n = len(cohort)
    caption = (
        f"Gate power on a fixed {cohort_n}-shot train-disjoint cohort. The wide "
        f"n_random={nr_lo} CI cannot resolve checkpoints; this asks whether "
        f"raising to {nr_hi} tightens it. The variance decomposition splits the "
        "per-shot ratio variance into WITHIN-shot sampling noise (shrinks with "
        "more rollouts) vs ACROSS-shot heterogeneity (does not). The pass-FRACTION "
        "+ sorted-ratio distribution are the stable signal."
    )
    fig.suptitle(
        "Controllability gate — statistical power vs number of random rollouts",
        fontsize=13,
    )
    fig.text(0.02, 0.005, caption, fontsize=8.5, wrap=True, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = FIG_DIR / "gate_power.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}", flush=True)


def main() -> int:
    setup_torch()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(COHORT_JSON)
    print(f"COHORT ({len(cohort)} shots): {cohort}", flush=True)
    print(f"N_RANDOM settings: {N_RANDOM}", flush=True)
    if not cohort:
        print("EMPTY COHORT — aborting", flush=True)
        return 1

    per_ckpt: dict[str, dict] = {}
    for k in CKPT:
        per_ckpt[k] = eval_checkpoint(k, cohort)

    (OUT_ROOT / "gate_power_compare.json").write_text(
        json.dumps(
            {"cohort": cohort, "n_random": N_RANDOM, "per_checkpoint": per_ckpt},
            indent=2,
            default=str,
        )
    )
    make_figure(per_ckpt, cohort)
    print("DONE=1", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
