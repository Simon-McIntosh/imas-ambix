#!/usr/bin/env python
"""Gate-2 evaluation with the variational patch-current inverse (no Picard).

The powered rematch of the training-free gate-2 protocol: per held-out slice,
invert the RAW measured magnetics for the patch-current vector (whitened sensor
misfit + Rogowski Ip anchor + λ·structure-residual, per-slice Adam under one of
the weight-policy arms), read axis / X-point set / LCFS radii from the
assembled ψ at evaluation time, and score against the firewalled EFIT referee
with the same skill formulas, train-mean baseline, shot list, and slice
selection as ``gs_solve_gate_eval.py`` — the numbers are directly comparable.

There is no inner solve and hence no convergence masking: every candidate
slice is scored (the corrected Picard chain scores 12/160).

Additionally recovers the closures the fit implies: per-slice p′(ψ) and
FF′(ψ)/μ0 from the per-bin regression coefficients with their uncertainties,
and the integrated p(ψ), F²(ψ) with the F² ≥ 0 integrability check
(F_vac from the measured TF current where available).

Artifacts:  imas_ambix/latent/artifacts/patch_gate/
Figures:    docs/figures/patch-current-force-balance/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.evaluate import (
    headline_skill,
    matched_xpoint_error,
    per_quantity_skill,
)
from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    _read_axis,
    _read_boundary_psi,
)
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_inverse import (
    InverseConfig,
    SlicePayload,
    invert_slices,
)
from imas_ambix.latent.structure_residual import (
    fit_flux_functions,
    integrate_closures,
)
from imas_ambix.latent.topology import (
    _inside_polygon,
    find_critical_points,
    lcfs_radii,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/patch-current-force-balance")

TARGET_NAMES = [
    "axis_R",
    "axis_Z",
    "xpt0_R",
    "xpt0_Z",
    "xpt1_R",
    "xpt1_Z",
    *[f"lcfs_r_{k}" for k in range(8)],
]

POLICY_COLOR = {
    "fixed": "#2166ac",
    "warm-start": "#1b7837",
    "discrepancy": "#d95f02",
}


def geometry_target(
    psi2d: np.ndarray, grid: EquilibriumGrid
) -> tuple[np.ndarray, float, float]:
    """Oracle-shaped 14-D geometry read of an assembled ψ field.

    Mirrors ``gs_solve_gate_eval.equilibrium_target`` but reads everything from
    the ψ field alone (no EquilibriumResult): sign-aware conductor-clear axis,
    innermost in-polygon X-point / limiter-contact boundary flux, LCFS radii.
    Returns (target, psi_axis, psi_boundary).
    """
    target = np.full(14, np.nan)
    # plasma current here is positive-Ip MAST convention: axis = max of ψ; the
    # sign-aware read picks the sign from the field itself via both attempts
    ax_pos, psi_pos = _read_axis(psi2d, grid, +1.0)
    ax_neg, psi_neg = _read_axis(psi2d, grid, -1.0)
    # choose the sign whose extremum deviates more from the field median —
    # the plasma well dominates the interior either way
    med = float(np.median(psi2d))
    if abs(psi_pos - med) >= abs(psi_neg - med):
        axis, axis_psi = ax_pos, psi_pos
    else:
        axis, axis_psi = ax_neg, psi_neg
    target[0], target[1] = axis
    boundary_psi = _read_boundary_psi(psi2d, grid, axis_psi)
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    if cp.x_points.shape[0]:
        ins = _inside_polygon(
            cp.x_points[:, 0], cp.x_points[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(cp.x_points[:, 0], cp.x_points[:, 1])
        pts = cp.x_points[ins]
        xpsi = cp.x_psi[ins]
        if pts.shape[0]:
            order = np.argsort(np.abs(xpsi - boundary_psi))
            for slot in range(min(2, pts.shape[0])):
                target[2 + 2 * slot] = pts[order[slot], 0]
                target[3 + 2 * slot] = pts[order[slot], 1]
    target[6:] = lcfs_radii(psi2d, grid.rg, grid.zg, tuple(axis), boundary_psi)
    return target, float(axis_psi), float(boundary_psi)


def shot_payloads(shot: int, *, nr, nz, max_slices, min_ip_ka, split="eval"):
    """Per-shot geometry + slice payloads, identical selection to the Picard gate."""
    table = build_table_for_shot(int(shot))
    fwd = build_operator(table)
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    basis = PatchBasis.from_table(table, nr=nr, nz=nz)
    g_sens, channels = grid.sensor_greens(table)

    w = load_shot_windows(int(shot), fwd, split, feature_schema(), with_referee=True)
    if w is None or w.ref_target is None:
        return None
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    ch_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = ch_rows >= 0
    valid = [
        t
        for t in range(w.times.size)
        if np.isfinite(w.ref_target[t, :2]).all() and abs(w.anchored[t, 0]) > min_ip_ka
    ]
    if len(valid) > max_slices:
        valid = valid[:: max(1, len(valid) // max_slices)][:max_slices]
    if not valid:
        return None
    scale = np.nanstd(w.raw_mag, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    scale_ch = np.where(present, scale[np.clip(ch_rows, 0, None)], 1.0)

    payloads, refs = [], []
    for t in valid:
        vac = fwd.vacuum_prediction(w.i_pf[t])
        payloads.append(
            SlicePayload(
                measured=np.where(
                    present, w.raw_mag[t][np.clip(ch_rows, 0, None)], np.nan
                ),
                vacuum=np.where(present, vac[np.clip(ch_rows, 0, None)], 0.0),
                mask=present & w.mag_mask[t][np.clip(ch_rows, 0, None)],
                scale=scale_ch,
                i_pf=w.i_pf[t],
                ip_amperes=float(abs(w.anchored[t, 0])) * 1e3,
                shot=int(shot),
                t_index=int(t),
                time_s=float(w.times[t]),
            )
        )
        refs.append(w.ref_target[t])
    return {
        "table": table,
        "grid": grid,
        "basis": basis,
        "payloads": payloads,
        "refs": np.array(refs),
    }


def train_mean_baseline(n_train, n_baseline_shots, min_ip_ka):
    schema = feature_schema()
    train_shots, _ = read_split_shot_lists(n_train, 8)
    rows = []
    for s in train_shots[:n_baseline_shots]:
        try:
            fwd = build_operator(build_table_for_shot(int(s)))
        except Exception:  # noqa: BLE001
            continue
        wtr = load_shot_windows(int(s), fwd, "train", schema, with_referee=True)
        if wtr is not None and wtr.ref_target is not None:
            on = np.abs(wtr.anchored[:, 0]) > min_ip_ka
            rows.append(wtr.ref_target[on])
    return (
        np.nanmean(np.concatenate(rows, axis=0), axis=0)
        if rows
        else np.full(14, np.nan)
    )


def score(model, ref, baseline_vec):
    """Same skill computation as the Picard gate (per-quantity RMSE skill)."""
    baseline = np.tile(baseline_vec, (len(model), 1))
    skill = per_quantity_skill(model, ref, baseline, TARGET_NAMES)
    xm = np.array(
        [
            matched_xpoint_error(model[i, 2:6].reshape(2, 2), ref[i, 2:6].reshape(2, 2))
            for i in range(len(model))
        ]
    )
    xb = np.array(
        [
            matched_xpoint_error(
                baseline[i, 2:6].reshape(2, 2), ref[i, 2:6].reshape(2, 2)
            )
            for i in range(len(model))
        ]
    )
    finite = np.isfinite(xm) & np.isfinite(xb)
    xpt_skill = (
        1.0
        - np.sqrt(np.nanmean(xm[finite] ** 2)) / np.sqrt(np.nanmean(xb[finite] ** 2))
        if finite.any()
        else np.nan
    )
    axis_err = np.hypot(model[:, 0] - ref[:, 0], model[:, 1] - ref[:, 1])
    return {
        "per_quantity_skill": {
            k: (None if not np.isfinite(v) else float(v)) for k, v in skill.items()
        },
        "axis_skill": headline_skill(skill, ["axis_R", "axis_Z"]),
        "xpoint_set_skill": None if not np.isfinite(xpt_skill) else float(xpt_skill),
        "lcfs_skill": headline_skill(skill, [f"lcfs_r_{k}" for k in range(8)]),
        "axis_error_mean_m": float(np.nanmean(axis_err)),
        "axis_error_median_m": float(np.nanmedian(axis_err)),
        "axis_errors": axis_err,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--policies", type=str, default="fixed,warm-start,discrepancy")
    ap.add_argument("--lambda-fb", type=float, default=10.0)
    ap.add_argument(
        "--arms",
        type=str,
        default="",
        help=(
            "explicit arm spec overriding --policies/--lambda-fb: comma list of "
            "policy:lambda[:misfit_ratio[:lambda_max]] tokens, e.g. "
            "'fixed:0,fixed:3,fixed:10,warm-start:10,discrepancy:10:1.3:30'"
        ),
    )
    ap.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=("eval", "train"),
        help=(
            "eval = the held-out gate; train = TRAIN-shot slices for "
            "leakage-free policy/lambda selection (shots after the baseline block)"
        ),
    )
    ap.add_argument("--n-tune-shots", type=int, default=4)
    ap.add_argument("--out-tag", type=str, default="")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--form", type=str, default="affine-r2")
    ap.add_argument("--connectivity", type=str, default="locality")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--throughput-bench", action="store_true")
    args = ap.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    connectivity = None if args.connectivity in ("", "none") else args.connectivity
    logger.info("device=%s connectivity=%s", device, connectivity)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )

    if args.split == "train":
        # tuning cohort: TRAIN shots after the baseline block — selection on
        # these referee labels never touches the held-out gate
        eval_shots = train_shots[
            args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots
        ]
    else:
        eval_shots = held_shots
    shots = []
    for s in eval_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split=args.split,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is not None:
            shots.append(payload)
            logger.info(
                "shot %d: %d candidate slices",
                payload["payloads"][0].shot,
                len(payload["payloads"]),
            )

    # ---- P1-gate throughput bench (batched forward on this device) ----------
    if args.throughput_bench and shots:
        basis = shots[0]["basis"]
        for batch in (64, 1024, 4096):
            rate = basis.throughput(batch=batch, n_iter=20, device=device)
            logger.info(
                "throughput: %d-slice batched forward on %s → %.0f slices/s",
                batch,
                device,
                rate,
            )
            (ARTIFACTS / "throughput.json").write_text(
                json.dumps({"device": device, "batch": batch, "slices_per_s": rate})
            )

    # ---- inverse per policy arm ---------------------------------------------
    per_policy: dict[str, dict] = {}
    tag = args.out_tag or ("_tune" if args.split == "train" else "")
    if args.arms:
        arm_specs = []
        for tok in args.arms.split(","):
            parts = tok.split(":")
            kw: dict = {"policy": parts[0]}
            if len(parts) > 1:
                kw["lambda_fb"] = float(parts[1])
            if len(parts) > 2:
                kw["misfit_ratio"] = float(parts[2])
            if len(parts) > 3:
                kw["lambda_max"] = float(parts[3])
            arm_specs.append((tok, kw))
    else:
        arm_specs = [
            (p, {"policy": p, "lambda_fb": args.lambda_fb})
            for p in args.policies.split(",")
        ]
    for policy, arm_kw in arm_specs:
        cfg = InverseConfig(
            iters=args.iters,
            lr=args.lr,
            n_bins=args.n_bins,
            connectivity=connectivity,
            **arm_kw,
        )
        model_rows, ref_rows, diag_rows = [], [], []
        psi_reads = []  # (psi_axis, psi_boundary) per scored slice, for closures
        inversions_all = []
        t0 = time.perf_counter()
        for payload in shots:
            grid, basis = payload["grid"], payload["basis"]
            inv = invert_slices(basis, payload["payloads"], cfg, device=device)
            inversions_all.append((payload, inv))
            for k, r in enumerate(inv):
                psi2d = basis.psi_grid_2d_np(r.i_cell, payload["payloads"][k].i_pf)
                target, psi_ax, psi_b = geometry_target(psi2d, grid)
                model_rows.append(target)
                ref_rows.append(payload["refs"][k])
                psi_reads.append((psi_ax, psi_b))
                diag_rows.append(
                    {
                        "shot": r.shot,
                        "t_index": r.t_index,
                        "time_s": r.time_s,
                        "misfit": r.misfit,
                        "structure": r.structure,
                        "lambda_final": r.lambda_final,
                        "ip_rel_err": r.ip_rel_err,
                    }
                )
        dt = time.perf_counter() - t0
        model = np.array(model_rows)
        ref = np.array(ref_rows)
        sc = score(model, ref, baseline_vec)
        axis_errors = sc.pop("axis_errors")
        per_policy[policy] = {
            **sc,
            "n_scored": int(len(model)),
            "n_candidate": int(len(model)),
            "scored_fraction": 1.0,
            "wall_s": dt,
            "diag": diag_rows,
        }
        np.savez(
            ARTIFACTS / f"gate_arrays_{policy.replace(':', '-')}{tag}.npz",
            model=model,
            ref=ref,
            baseline=np.tile(baseline_vec, (len(model), 1)),
            axis_errors=axis_errors,
        )
        logger.info(
            "[%s] scored %d/%d axis_skill=%.3f lcfs_skill=%s median %.3f m (%.0f s)",
            policy,
            len(model),
            len(model),
            sc["axis_skill"],
            sc["lcfs_skill"],
            sc["axis_error_median_m"],
            dt,
        )

        # ---- closures for this arm (recovered p', FF'/mu0 per slice) --------
        if args.split == "train":
            per_policy[policy]["closures"] = []
            continue  # tuning run: skills only, closures belong to the gate
        closure_rows = []
        idx = 0
        for payload, inv in inversions_all:
            basis = payload["basis"]
            r_c = basis.r_cells.to(torch.float64)
            for k, r in enumerate(inv):
                p = payload["payloads"][k]
                psi_c = basis.psi_cells_np(r.i_cell, p.i_pf)
                jphi = r.i_cell / float(basis.cell_area)
                fit = fit_flux_functions(
                    torch.as_tensor(psi_c, dtype=torch.float64),
                    r_c,
                    torch.as_tensor(jphi, dtype=torch.float64),
                    n_bins=args.n_bins,
                    form=args.form,
                )
                psi_ax, psi_b = psi_reads[idx]
                integ = integrate_closures(
                    fit, psi_axis=psi_ax, psi_boundary=psi_b, f_vac=0.85 * 0.55
                )
                closure_rows.append(
                    {
                        "shot": r.shot,
                        "t_index": r.t_index,
                        "psi_bins": np.asarray(fit.psi_centers).tolist(),
                        "a_k": np.asarray(fit.a_k).tolist(),
                        "b_k": np.asarray(fit.b_k).tolist(),
                        "a_err": np.asarray(fit.a_err).tolist(),
                        "b_err": np.asarray(fit.b_err).tolist(),
                        "weight_mass": np.asarray(fit.weight_mass).tolist(),
                        "psi_axis": psi_ax,
                        "psi_boundary": psi_b,
                        "f2_min": float(np.min(integ["f_squared"])),
                        "p_axis": float(np.max(np.abs(integ["p"]))),
                    }
                )
                idx += 1
        per_policy[policy]["closures"] = closure_rows

    # ---- artifacts + report --------------------------------------------------
    picard_ref = None
    picard_path = Path("imas_ambix/latent/artifacts/gs_solve_gate_eval.json")
    if picard_path.exists():
        picard_ref = json.loads(picard_path.read_text())
    result = {
        "config": {k: v for k, v in vars(args).items()},
        "device": device,
        "baseline_axis": [float(baseline_vec[0]), float(baseline_vec[1])],
        "picard_reference": picard_ref
        and {
            "n_scored": picard_ref["n_scored"],
            "n_candidate": picard_ref["n_candidate"],
            "axis_error_median_m": picard_ref["axis_error_median_m"],
            "axis_skill": picard_ref["axis_skill"],
            "lcfs_skill": picard_ref["lcfs_skill"],
        },
        "per_policy": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("diag", "closures")}
            for k, v in per_policy.items()
        },
    }
    (ARTIFACTS / f"patch_gate_eval{tag}.json").write_text(json.dumps(result, indent=2))
    (ARTIFACTS / f"patch_gate_diag{tag}.json").write_text(
        json.dumps(
            {
                k: {"diag": v["diag"], "closures": v["closures"]}
                for k, v in per_policy.items()
            },
            indent=2,
        )
    )
    logger.info("gate artifacts written to %s", ARTIFACTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
