#!/usr/bin/env python
"""Offline closure-arm report: profile-DOF ladder curve + shard merge/score.

Consumes the per-config gate artifacts the closure gate writes
(``closure_gate_eval-calibrated-dofN*.json`` + ``_arrays.npz``) and produces

* a merged, re-scored artifact for shot-sharded runs (the continuous arm is
  sharded two shots per SLURM task; per-shard skills are meaningless because
  the paired bootstrap needs the full 8-shot cohort);
* the ladder evidence curve — held-out skill and coverage vs profile DOF at
  the standard cost thresholds, with paired-bootstrap CIs — as JSON + figure;
* a comparison table against the recorded 2-DOF grid baselines.

Pure post-processing: everything here re-reads gate outputs; no solver call,
no EFIT access beyond what the gate already recorded in its arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.closure_gate_eval import cost_sweep
from scripts.patch_gate_eval import lcfs_offset_cm_stats, score, train_mean_baseline

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/force-balance-spine")


def _load(tag: str):
    j = ARTIFACTS / f"closure_gate_eval{tag}.json"
    z = ARTIFACTS / f"closure_gate_eval{tag}_arrays.npz"
    if not (j.exists() and z.exists()):
        return None
    return json.loads(j.read_text()), np.load(z)


def rescore(arr_list: list, baseline_vec: np.ndarray) -> dict:
    """Concatenate shard arrays and score once on the full cohort."""
    model = np.concatenate([a["model"] for a in arr_list])
    ref = np.concatenate([a["ref"] for a in arr_list])
    shot_ids = np.concatenate([a["shot_ids"] for a in arr_list])
    cost = np.concatenate([a["cost"] for a in arr_list])
    flattop = np.concatenate([a["flattop_mask"] for a in arr_list])
    sc = score(model, ref, baseline_vec, shot_ids=shot_ids)
    sc.pop("axis_errors")
    out = {
        "n_scored": int(model.shape[0]),
        "cost_median": float(np.median(cost)),
        **sc,
        **lcfs_offset_cm_stats(model, ref, flattop),
        "cost_sweep": cost_sweep(model, ref, baseline_vec, cost, shot_ids),
    }
    for k in ("z0",):
        if all(k in a.files for a in arr_list):
            v = np.concatenate([a[k] for a in arr_list])
            if np.isfinite(v).any():
                out[f"{k}_median"] = float(np.nanmedian(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dof-tags",
        type=str,
        default="dof2:2,dof3:3,dof4:4,dof6:6,dof8:8,dof10:10",
        help="comma list of <suffix>:<dof> ladder artifacts (calibrated)",
    )
    ap.add_argument(
        "--cont-shards",
        type=str,
        default="cont-s1,cont-s2,cont-s3,cont-s4",
        help="continuous-arm shard suffixes to merge (calibrated)",
    )
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )

    report: dict = {"ladder": [], "continuous": None, "references": {}}

    # references: the recorded grid-mode runs + the sign-free/passive arms
    for name, tag in (
        ("grid", ""),
        ("grid-calibrated", "-calibrated"),
        ("ladder-free-k2", "-calibrated-dof2"),
        ("ladder-free-k2-passive8", "-calibrated-dof2-pk8"),
    ):
        loaded = _load(tag)
        if loaded is None:
            continue
        j, a = loaded
        report["references"][name] = {
            "cost_median": float(np.median(a["cost"])),
            "axis_skill": j.get("axis_skill"),
            "axis_skill_ci": j.get("axis_skill_ci"),
            "lcfs_skill": j.get("lcfs_skill"),
            "cost_sweep": cost_sweep(
                a["model"], a["ref"], baseline_vec, a["cost"], a["shot_ids"]
            ),
        }

    # ladder curve
    for item in args.dof_tags.split(","):
        suffix, dof = item.split(":")
        loaded = _load(f"-calibrated-{suffix.strip()}")
        if loaded is None:
            continue
        j, a = loaded
        row = {
            "dof": int(dof),
            "suffix": suffix.strip(),
            "n_scored": j["n_scored"],
            "n_candidate": j["n_candidate"],
            "converged_fraction": j.get("strict_converged_fraction_of_scored"),
            "cost_median": float(np.median(a["cost"])),
            "axis_skill": j.get("axis_skill"),
            "axis_skill_ci": j.get("axis_skill_ci"),
            "lcfs_skill": j.get("lcfs_skill"),
            "lcfs_skill_ci": j.get("lcfs_skill_ci"),
            "xpoint_set_skill": j.get("xpoint_set_skill"),
            "xpoint_set_skill_ci": j.get("xpoint_set_skill_ci"),
            "axis_error_median_m": j.get("axis_error_median_m"),
            "lcfs_offset_median_cm_flattop": j.get("lcfs_offset_median_cm_flattop"),
            "cost_sweep": j.get("cost_sweep"),
        }
        report["ladder"].append(row)
    report["ladder"].sort(key=lambda r: r["dof"])

    # continuous arm: merge shards, score once
    arrs = []
    for suffix in args.cont_shards.split(","):
        loaded = _load(f"-calibrated-{suffix.strip()}")
        if loaded is not None:
            arrs.append(loaded[1])
    if arrs:
        report["continuous"] = rescore(arrs, baseline_vec)

    out = ARTIFACTS / "closure_ladder_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

    # ---- the evidence curve figure ----
    rows = report["ladder"]
    if rows:
        dofs = [r["dof"] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

        ax = axes[0]
        ax.plot(dofs, [r["cost_median"] for r in rows], "o-", color="#1565c0")
        for name, colour in (
            ("grid-calibrated", "#444444"),
            ("ladder-free-k2", "#8a3324"),
            ("ladder-free-k2-passive8", "#1b7837"),
        ):
            ref = report["references"].get(name)
            if ref:
                ax.axhline(
                    ref["cost_median"], color=colour, ls="--", lw=1.0, label=name
                )
        ax.set_xlabel("profile DOF (n_p + n_f)")
        ax.set_ylabel("whitened misfit, median")
        ax.set_yscale("log")
        ax.set_title("Cost floor vs profile DOF")
        ax.legend(fontsize=8)

        ax = axes[1]
        for key, colour, label in (
            ("axis_skill", "#1b7837", "axis"),
            ("lcfs_skill", "#d95f02", "LCFS"),
            ("xpoint_set_skill", "#7570b3", "X-point set"),
        ):
            vals = [r[key] for r in rows]
            ax.plot(dofs, vals, "o-", color=colour, label=label)
            ci_key = f"{key}_ci"
            if all(r.get(ci_key) for r in rows):
                lo = [r[ci_key][0] for r in rows]
                hi = [r[ci_key][1] for r in rows]
                ax.fill_between(dofs, lo, hi, color=colour, alpha=0.15)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("profile DOF (n_p + n_f)")
        ax.set_ylabel("held-out skill vs train-mean (CI band)")
        ax.set_title("Topology skill vs profile DOF")
        ax.legend(fontsize=8)

        ax = axes[2]
        for thr, colour in ((1.0, "#1565c0"), (3.0, "#5e97d1"), (10.0, "#a8c6e8")):
            cov = []
            for r in rows:
                sweep = {s.get("cost_le"): s for s in (r.get("cost_sweep") or [])}
                cov.append(sweep.get(thr, {}).get("coverage"))
            ax.plot(dofs, cov, "o-", color=colour, label=f"cost ≤ {thr:g}")
        ax.set_xlabel("profile DOF (n_p + n_f)")
        ax.set_ylabel("coverage (fraction of 160)")
        ax.set_ylim(0, 1.02)
        ax.set_title("Self-consistent coverage vs profile DOF")
        ax.legend(fontsize=8)

        fig.suptitle(
            "Profile-DOF ladder — held-out, calibrated payloads "
            "(LSQ-per-sweep closure through the free-boundary GS fixed point)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig_path = FIGURES / "fig-ladder-curve.png"
        fig.savefig(fig_path, dpi=140)
        print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
