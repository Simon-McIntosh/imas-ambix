"""Build the screened eval-only cohort, then run the ROBUST controllability gate
on three checkpoints against it, and write the comparison figure.

NOT production code — a self-contained analysis driver staged on the shared FS
for the gate-robustness investigation.  Decoder-only GPU work; loads each model
once, decodes rollouts in one-VQ-pass batches.

Run inside a betelgeuse srun with the venv active (HF offline).
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
    multi_shot_delta_nm,
)
from imas_ambix.worldmodel.controllable_train import (  # noqa: E402
    load_controllable_model_from_checkpoint,
)
from imas_ambix.worldmodel.gate_cohort import (  # noqa: E402
    build_screened_cohort,
    load_cohort,
)
from imas_ambix.worldmodel.spacetime_dataset import (  # noqa: E402
    SpacetimeWindowConfig,
)

TOKEN_ROOT = Path("/work/projects/imas_gpu/mast-tokens")
MANIFEST = (
    "/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_unified_6cam.json"
)
CAMERA = "rbb"
DEVICE = "cuda"
HORIZON_S = 0.25
COHORT_OUT = Path("/work/projects/imas_gpu/worldmodel/gate_cohort.json")
# Expanded cohort: scan up to a few hundred train-disjoint candidates and target
# ~25 passing shots so the gate's bootstrap CI is no longer 7-shot-wide.  The
# builder loads the VQ once for the whole scan (AGENTS.md §2b) and relaxes the
# brightness/motion gates toward their floors if too few pass (recorded in the
# cohort JSON), so a 400-cap scan is fast and reliably reaches the target.
CANDIDATE_CAP = int(os.environ.get("WM_COHORT_CAP", "400"))
COHORT_TARGET = int(os.environ.get("WM_COHORT_TARGET", "25"))

CKPT = {
    "1221741": Path(
        "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221741/latest.pt"
    ),
    "1221834": Path(
        "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221834/latest.pt"
    ),
    "1221883": Path(
        "/work/projects/imas_gpu/worldmodel/ckpt/controllable-1221883/latest.pt"
    ),
}
LABEL = {
    "1221741": "baseline",
    "1221834": "joint-gen",
    "1221883": "diag-off",
}
FIG_DIR = Path(
    "/home/ITER/mcintos/Code/imas-ambix/docs/figures/joint-multimodal-plasma-wm"
)
OUT_ROOT = Path("/work/projects/imas_gpu/worldmodel/robust_gate")


def window_cfg() -> SpacetimeWindowConfig:
    return SpacetimeWindowConfig(
        n_frames=24,
        n_plan=8,
        context_frames=8,
        frame_stride=1,
        target_horizon_s=HORIZON_S,
    )


def make_cfg(modalities, held_out) -> EvalConfig:
    return EvalConfig(
        held_out=tuple(held_out),
        n_random=3,
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


def build_cohort_once() -> list[int]:
    """Screen the cohort once (uses a fresh model only for modality resolution)."""
    from imas_ambix.worldmodel.controllable_eval import _resolve_eval_modalities

    # the screen does not need the model — only the modality list (for stream
    # presence counting) + the window cfg.  Use the baseline's trained streams.
    _model, payload = load_controllable_model_from_checkpoint(
        CKPT["1221741"], map_location=DEVICE
    )
    del _model
    torch.cuda.empty_cache()
    modalities = _resolve_eval_modalities("auto", payload)
    screen_cfg = EvalConfig(
        n_signal_steps=4,
        n_act_steps=8,
        modalities=modalities,
        window=window_cfg(),
    )
    summary = build_screened_cohort(
        screen_cfg,
        camera=CAMERA,
        token_root=TOKEN_ROOT,
        manifest_path=MANIFEST,
        device=DEVICE,
        out_json=COHORT_OUT,
        candidate_cap=CANDIDATE_CAP,
        target_size=COHORT_TARGET,
        work_dir=OUT_ROOT / "_cohort_screen",
    )
    print("COHORT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    return load_cohort(COHORT_OUT)


def eval_checkpoint(ckpt_key: str, cohort: list[int]) -> dict:
    from imas_ambix.worldmodel.controllable_eval import _resolve_eval_modalities

    print(f"=== eval {ckpt_key} ({LABEL[ckpt_key]}) ===", flush=True)
    model, payload = load_controllable_model_from_checkpoint(
        CKPT[ckpt_key], map_location=DEVICE
    )
    model.eval()
    try:
        modalities = _resolve_eval_modalities("auto", payload)
        cfg = make_cfg(modalities, cohort)
        out_dir = OUT_ROOT / ckpt_key
        summary = multi_shot_delta_nm(
            model,
            config=cfg,
            camera=CAMERA,
            token_root=TOKEN_ROOT,
            device=DEVICE,
            out_json=out_dir / "robust_gate.json",
            work_dir=out_dir / "_dnm",
            decode=True,
        )
        print(
            f"{ckpt_key}: verdict={summary['verdict']} "
            f"pass_frac={summary['pass_fraction']:.2f} "
            f"mean_ratio={summary['mean_normalised_ratio']:.2f} "
            f"CI=[{summary['ratio_ci_lo']:.2f},{summary['ratio_ci_hi']:.2f}] "
            f"collapsed={summary['n_random_collapsed_total']}",
            flush=True,
        )
        return summary
    finally:
        del model
        torch.cuda.empty_cache()


def make_figure(cohort_summary: dict, per_ckpt: dict, cohort_json: dict):
    keys = list(CKPT.keys())
    fig = plt.figure(figsize=(13.5, 8.5), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    # --- cohort screen panel: which candidates passed + why ---
    per_shot = cohort_json["per_shot"]
    assemblable = [r for r in per_shot if r["assemblable"]]
    ax0 = fig.add_subplot(gs[0, 0])
    bri = [r["mean_brightness"] for r in assemblable]
    mot = [r["transient_motion"] for r in assemblable]
    col = ["#54a24b" if r["passed"] else "#e45756" for r in assemblable]
    ax0.scatter(bri, mot, c=col, s=28, edgecolor="#333", linewidth=0.4)
    th = cohort_json["thresholds"]
    ax0.axvline(th["min_brightness"], ls="--", color="#888", lw=1)
    ax0.axhline(th["min_transient_motion"], ls="--", color="#888", lw=1)
    ax0.set_xlabel("GT forecast mean brightness (0-255)")
    ax0.set_ylabel("GT transient motion (frame-to-frame pixel-L1)")
    ax0.set_title(
        f"cohort screen: {cohort_summary['n_passed']} / "
        f"{cohort_summary['n_assemblable']} assemblable candidates pass\n"
        f"(green=kept, red=rejected; dashed=thresholds)",
        fontsize=10,
    )

    # --- per-checkpoint pass fraction ---
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(keys))
    pf = [per_ckpt[k]["pass_fraction"] for k in keys]
    b = ax1.bar(x, pf, color=["#4c78a8", "#e45756", "#54a24b"])
    ax1.axhline(0.5, ls="--", color="#888", lw=1, label="majority (0.5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{k}\n{LABEL[k]}" for k in keys], fontsize=8)
    ax1.set_ylabel("cohort pass-fraction")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("per-checkpoint cohort pass-fraction", fontsize=10)
    for bi, v in zip(b, pf):
        ax1.text(
            bi.get_x() + bi.get_width() / 2,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.legend(fontsize=8)

    # --- mean normalised ratio + bootstrap CI ---
    ax2 = fig.add_subplot(gs[1, 0])
    mr = [per_ckpt[k]["mean_normalised_ratio"] for k in keys]
    lo = [per_ckpt[k]["ratio_ci_lo"] for k in keys]
    hi = [per_ckpt[k]["ratio_ci_hi"] for k in keys]
    yerr = np.array([[m - l for m, l in zip(mr, lo)], [h - m for m, h in zip(mr, hi)]])
    ax2.errorbar(
        x,
        mr,
        yerr=yerr,
        fmt="o",
        ms=8,
        capsize=6,
        color="#333",
        ecolor="#4c78a8",
        elinewidth=2,
    )
    ax2.axhline(1.0, ls="--", color="#e45756", lw=1.2, label="noise floor (ratio=1.0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{k}\n{LABEL[k]}" for k in keys], fontsize=8)
    ax2.set_ylabel("mean normalised ratio (tvr / floor)")
    ax2.set_title(
        "cohort mean ratio + 95% bootstrap CI\n"
        "(CI lower bound clear of 1.0 = controllability not a 1-shot artifact)",
        fontsize=10,
    )
    for xi, m, l in zip(x, mr, lo):
        ax2.text(xi + 0.06, m, f"{m:.2f}\n[lo {l:.2f}]", fontsize=8, va="center")
    ax2.legend(fontsize=8)

    # --- verdict table ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    cols = [
        "ckpt",
        "role",
        "verdict",
        "n_shots",
        "pass_frac",
        "mean_ratio",
        "CI_lo",
        "collapsed",
    ]
    cells = []
    for k in keys:
        s = per_ckpt[k]
        cells.append(
            [
                k,
                LABEL[k],
                s["verdict"],
                str(s["n_transient"]),
                f"{s['pass_fraction']:.2f}",
                f"{s['mean_normalised_ratio']:.2f}",
                f"{s['ratio_ci_lo']:.2f}",
                str(s["n_random_collapsed_total"]),
            ]
        )
    tbl = ax3.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.7)
    for j, k in enumerate(keys):
        color = "#d6f5d6" if per_ckpt[k]["verdict"] == "PASS" else "#f5e3d6"
        for c in range(len(cols)):
            tbl[(j + 1, c)].set_facecolor(color)

    cohort_n = cohort_summary["cohort_size"]
    caption = (
        f"ROBUST gate: a {cohort_n}-shot TRAIN-DISJOINT cohort screened on GT "
        "brightness + transient motion + plan variation. The floor EXCLUDES "
        "collapsed random dreams; the gate passes only when a MAJORITY of cohort "
        "shots clear ratio>1.5 AND the bootstrap-CI lower bound is clear of 1.0 — "
        "so no single GOOD shot can carry the gate (the old 18502-05 failure mode)."
    )
    fig.suptitle(
        "Controllability gate — robust cohort + normalised, collapse-rejecting metric",
        fontsize=13,
    )
    fig.text(0.02, 0.005, caption, fontsize=8.5, wrap=True, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = FIG_DIR / "gate_robust_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"WROTE {out}", flush=True)


def main() -> int:
    setup_torch()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cohort = build_cohort_once()
    cohort_json = json.loads(COHORT_OUT.read_text())
    cohort_summary = cohort_json["summary"]
    print(f"COHORT ({len(cohort)} shots): {cohort}", flush=True)
    if not cohort:
        print("EMPTY COHORT — aborting", flush=True)
        return 1

    per_ckpt: dict[str, dict] = {}
    for k in CKPT:
        per_ckpt[k] = eval_checkpoint(k, cohort)

    (OUT_ROOT / "robust_gate_compare.json").write_text(
        json.dumps(
            {"cohort_summary": cohort_summary, "per_checkpoint": per_ckpt},
            indent=2,
            default=str,
        )
    )
    make_figure(cohort_summary, per_ckpt, cohort_json)
    print("DONE=1", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
