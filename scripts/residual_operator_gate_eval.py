#!/usr/bin/env python
"""Non-inferiority gate: trained residual operator vs the frozen classical spine.

Held-out paired evaluation on the closure gate cohort (n≥128 slices, 8 shots):
for every slice the frozen classical spine (staged-disc soft prior + profile
ladder solve) is fit exactly as in the frozen configuration, then the trained
residual operator corrects the profile DOF about that solution through the
exact Green's layer.  Both arms are read with the SAME push-out geometry
reader and scored against the firewalled EFIT referee with the SAME
paired-bootstrap harness — bootstrap draws share the RNG seed, so the
per-draw skill difference IS the paired Δskill distribution.

Gate (non-inferiority): the paired Δskill (operator − spine) 95% CI must not
extend below the margin (default −0.05) on BOTH the LCFS and the axis
headline skills, with n_scored ≥ 128.  The verdict is recorded either way.

Firewall: EFIT enters referee/scoring only; the operator saw only spine-
manufactured labels from disjoint train-split shots.

Artifacts: imas_ambix/latent/artifacts/patch_gate/residual_operator_gate_eval.json
Figures:   docs/figures/learned-equilibrium-operator/
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.boundary_disc import sensor_signature_arrays
from imas_ambix.latent.data import (
    feature_schema,
    load_shot_windows,
    read_split_shot_lists,
)
from imas_ambix.latent.profile_greens_decoder import ProfileGreensDecoder
from imas_ambix.latent.residual_operator import (
    load_checkpoint,
    slice_globals,
    slice_tokens,
)
from scripts.closure_gate_eval import (
    _shot_passive_sidecar,
    fit_and_read_slice,
    geometry_target_pushout,
)
from scripts.patch_gate_eval import (
    _bootstrap_skill_draws,
    _percentile_ci,
    lcfs_offset_cm_stats,
    score,
    shot_payloads,
    train_mean_baseline,
)
from scripts.spine_label_factory import frozen_spine_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("residual_operator_gate_eval")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/learned-equilibrium-operator")


def eval_shot(job: tuple) -> dict | None:
    """One shot's paired spine + operator evaluation (runs in a fork worker)."""
    shot, args_d = job
    spine, config_sha = frozen_spine_config()
    isolve = spine["interior_solve"]
    spc = dict(spine["soft_priors"])

    payload = shot_payloads(
        shot,
        nr=args_d["nr"],
        nz=args_d["nz"],
        max_slices=args_d["max_slices_per_shot"],
        min_ip_ka=args_d["min_ip_ka"],
        split="eval",
    )
    if payload is None:
        return None
    grid, table, basis = payload["grid"], payload["table"], payload["basis"]
    sidecar = _shot_passive_sidecar(payload, int(isolve["passive_k"]))
    modes = np.asarray(sidecar["modes"], dtype=np.float64)
    cell_area = grid.dr * grid.dz

    # firewall-safe n_e per slice (referee-free reload, aligned by t_index)
    fwd = build_operator(table)
    w = load_shot_windows(
        shot, fwd, table.signature.key, feature_schema(), with_referee=False
    )
    # t_index in the payloads indexes the plasma-on window rows of this SAME
    # loader, so the anchored n_e row aligns directly
    ne_rows = None if w is None else w.anchored[:, 1]

    model, ckpt = load_checkpoint(args_d["checkpoint"])
    # the decoder gets its OWN fp64 basis — converting the payload's basis in
    # place would perturb the spine chain's disc read (fp32 in the frozen
    # configuration) and break the paired comparison
    from imas_ambix.latent.patch_basis import PatchBasis  # noqa: PLC0415

    basis64 = PatchBasis.from_table(
        table, nr=args_d["nr"], nz=args_d["nz"], dtype=torch.float64
    )
    dec = ProfileGreensDecoder(
        basis64,
        n_p=int(isolve["n_p"]),
        n_f=int(isolve["n_f"]),
        kind=str(isolve["profile_kind"]),
    )
    sr, sz, sang, is_flux = sensor_signature_arrays(table)

    rows: list[dict] = []
    warm_jphi = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    t0 = time.perf_counter()
    for k in order:
        p = payload["payloads"][int(k)]
        f = fit_and_read_slice(
            grid,
            table,
            p,
            beta0_grid=(0.5,),
            alpha_grid=(1.0,),
            cost_limit=float("inf"),
            convergence_limit=args_d["convergence_limit"],
            retry_max_iterations=args_d["retry_max_iterations"],
            fit_mode="ladder",
            n_p=int(isolve["n_p"]),
            n_f=int(isolve["n_f"]),
            smoothness=float(isolve["smoothness"]),
            nonneg=isolve["profile_kind"] == "monomial-nonneg",
            passive=sidecar,
            passive_ridge=1.0,
            warm_jphi=warm_jphi,
            reseed_axis_r_max=float(isolve["reseed_axis_r_max"]),
            keep_psi=True,
            keep_jphi=True,
            basis=basis,
            meta={},
            soft_prior_cfg=spc,
            boundary_read=isolve["boundary_read_scoring"],
        )
        if not f.scored:
            rows.append({"scored": False, "reason": f.reason, "k": int(k)})
            continue
        warm_jphi = f.jphi_flat
        i_cell0 = f.jphi_flat[grid.cells] * cell_area
        _t, psi_ax, psi_b = geometry_target_pushout(f.psi, grid)
        psi_n = np.clip(
            (f.psi.ravel()[grid.cells] - psi_ax) / (psi_b - psi_ax), 0.0, 1.5
        )
        if f.passive_amp is not None:
            a_pass, *_ = np.linalg.lstsq(modes, np.asarray(f.passive_amp), rcond=None)
            sens_passive = np.asarray(sidecar["g_cols"], dtype=np.float64) @ a_pass
        else:
            sens_passive = np.zeros(p.measured.size)

        # operator arm: tokens -> dc -> exact Green's decode
        m_sens_np = np.asarray(dec.basis.m_sens.numpy(), dtype=np.float64)
        spine_pred = m_sens_np @ i_cell0 + p.vacuum + sens_passive
        tokens, tmask = slice_tokens(
            p.measured, p.vacuum, spine_pred, p.scale, p.mask, sr, sz, sang, is_flux
        )
        n_e = float(ne_rows[p.t_index]) if ne_rows is not None else float("nan")
        gl = slice_globals(p.ip_amperes, n_e)
        with torch.no_grad():
            dc = model(
                torch.tensor(tokens).unsqueeze(0),
                torch.tensor(tmask).unsqueeze(0),
                torch.tensor(gl).unsqueeze(0),
            ).double()
            cols = dec.profile_columns(
                torch.tensor(psi_n, dtype=torch.float64).unsqueeze(0),
                torch.tensor([p.ip_amperes], dtype=torch.float64),
            )
            i_new = dec.cell_currents(
                torch.tensor(i_cell0, dtype=torch.float64).unsqueeze(0),
                dc,
                cols,
                torch.tensor([p.ip_amperes], dtype=torch.float64),
            )[0].numpy()
        dpsi = (dec.basis._g_pg_np @ (i_new - i_cell0)).reshape(grid.nz, grid.nr)
        target_op, _, _ = geometry_target_pushout(f.psi + dpsi, grid)

        whit = np.where(
            p.mask & np.isfinite(p.measured),
            (spine_pred - np.nan_to_num(p.measured)) / np.clip(p.scale, 1e-12, None),
            0.0,
        )
        n_keep = max(int((p.mask & np.isfinite(p.measured)).sum()), 1)
        op_pred = spine_pred + m_sens_np @ (i_new - i_cell0)
        whit_op = np.where(
            p.mask & np.isfinite(p.measured),
            (op_pred - np.nan_to_num(p.measured)) / np.clip(p.scale, 1e-12, None),
            0.0,
        )
        rows.append(
            {
                "scored": True,
                "k": int(k),
                "t_index": p.t_index,
                "time_s": p.time_s,
                "ip": p.ip_amperes,
                "target_spine": f.target,
                "target_op": target_op,
                "ref": payload["refs"][int(k)],
                "cost_spine": float(np.sum(whit**2) / n_keep),
                "cost_op": float(np.sum(whit_op**2) / n_keep),
                "cost_solver": f.cost,
                "dc": dc[0].numpy(),
                "reseeded": f.reason == "scored-reseeded",
                "dpsi_max": float(np.abs(dpsi).max()),
            }
        )
    return {
        "shot": int(shot),
        "wall_s": time.perf_counter() - t0,
        "rows": rows,
        "config_sha": config_sha,
        "checkpoint_meta": {k: ckpt.get(k) for k in ("n_dof", "dc_scale") if k in ckpt},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="imas_ambix/latent/artifacts/residual_operator/residual_operator.pt",
    )
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=16)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--convergence-limit", type=float, default=5e-3)
    ap.add_argument("--retry-max-iterations", type=int, default=160)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    _train, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    args_d = vars(args)
    jobs = [(int(s), args_d) for s in held_shots]
    logger.info("gate eval: shots %s", held_shots)
    ctx = multiprocessing.get_context("fork")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            shot_results = [r for r in pool.map(eval_shot, jobs) if r is not None]
    else:
        shot_results = [r for r in map(eval_shot, jobs) if r is not None]

    scored = [
        r | {"shot": sr["shot"]}
        for sr in shot_results
        for r in sr["rows"]
        if r.get("scored")
    ]
    n_candidate = sum(len(sr["rows"]) for sr in shot_results)
    n_scored = len(scored)
    logger.info("scored %d/%d", n_scored, n_candidate)
    if n_scored == 0:
        raise SystemExit("no slices scored — cannot gate")

    model_spine = np.array([r["target_spine"] for r in scored])
    model_op = np.array([r["target_op"] for r in scored])
    ref = np.array([r["ref"] for r in scored])
    shot_ids = np.array([r["shot"] for r in scored])
    flattop = np.zeros(n_scored, dtype=bool)
    for s in np.unique(shot_ids):
        idx = np.flatnonzero(shot_ids == s)
        flattop[idx[int(np.argmax([scored[i]["ip"] for i in idx]))]] = True

    sc_spine = score(model_spine, ref, baseline_vec, shot_ids=shot_ids)
    sc_op = score(model_op, ref, baseline_vec, shot_ids=shot_ids)
    for sc in (sc_spine, sc_op):
        sc.pop("axis_errors", None)

    # paired Δskill: identical seed -> identical shot draws -> paired by draw
    seed, n_boot = 0, 2000
    ax_s, lc_s, xp_s, _ = _bootstrap_skill_draws(
        model_spine,
        ref,
        np.tile(baseline_vec, (n_scored, 1)),
        shot_ids,
        n_boot=n_boot,
        seed=seed,
    )
    ax_o, lc_o, xp_o, _ = _bootstrap_skill_draws(
        model_op,
        ref,
        np.tile(baseline_vec, (n_scored, 1)),
        shot_ids,
        n_boot=n_boot,
        seed=seed,
    )
    d_axis, d_lcfs, d_xpt = ax_o - ax_s, lc_o - lc_s, xp_o - xp_s
    delta = {
        "axis_skill_delta": float(sc_op["axis_skill"] - sc_spine["axis_skill"]),
        "axis_skill_delta_ci": _percentile_ci(d_axis),
        "lcfs_skill_delta": float(sc_op["lcfs_skill"] - sc_spine["lcfs_skill"]),
        "lcfs_skill_delta_ci": _percentile_ci(d_lcfs),
        "xpoint_set_skill_delta": (
            None
            if sc_op["xpoint_set_skill"] is None
            else float(sc_op["xpoint_set_skill"] - sc_spine["xpoint_set_skill"])
        ),
        "xpoint_set_skill_delta_ci": _percentile_ci(d_xpt),
    }
    non_inferior = (
        n_scored >= 128
        and delta["axis_skill_delta_ci"][0] is not None
        and delta["axis_skill_delta_ci"][0] >= -args.margin
        and delta["lcfs_skill_delta_ci"][0] >= -args.margin
    )

    lcfs_cm_spine = lcfs_offset_cm_stats(model_spine, ref, flattop)
    lcfs_cm_op = lcfs_offset_cm_stats(model_op, ref, flattop)
    dc_all = np.array([r["dc"] for r in scored])
    result = {
        "arm": "residual-operator-vs-spine",
        "checkpoint": args.checkpoint,
        "spine_config_sha256": shot_results[0]["config_sha"],
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_scored": n_scored,
        "n_candidate": n_candidate,
        "non_inferiority_margin": args.margin,
        "gate_pass": bool(non_inferior),
        "delta": delta,
        "spine": {**sc_spine, **{f"lcfs_{k}": v for k, v in lcfs_cm_spine.items()}},
        "operator": {**sc_op, **{f"lcfs_{k}": v for k, v in lcfs_cm_op.items()}},
        "cost_median_spine": float(np.median([r["cost_spine"] for r in scored])),
        "cost_median_operator": float(np.median([r["cost_op"] for r in scored])),
        "dc_abs_median": float(np.median(np.abs(dc_all))),
        "dc_abs_max": float(np.max(np.abs(dc_all))),
        "reseed_fraction": float(np.mean([r["reseeded"] for r in scored])),
        "wall_s_total": float(sum(sr["wall_s"] for sr in shot_results)),
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"residual_operator_gate_eval{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    np.savez(
        ARTIFACTS / f"residual_operator_gate_eval{tag}_arrays.npz",
        model_spine=model_spine,
        model_op=model_op,
        ref=ref,
        shot_ids=shot_ids,
        flattop_mask=flattop,
        dc=dc_all,
        cost_spine=np.array([r["cost_spine"] for r in scored]),
        cost_op=np.array([r["cost_op"] for r in scored]),
        delta_axis_draws=d_axis,
        delta_lcfs_draws=d_lcfs,
    )

    # ---- figures ----
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, draws, d0, name in (
        (axes[0], d_axis, delta["axis_skill_delta"], "axis skill"),
        (axes[1], d_lcfs, delta["lcfs_skill_delta"], "LCFS skill"),
    ):
        ax.hist(draws[np.isfinite(draws)], bins=60, color="#4477aa", alpha=0.85)
        ax.axvline(
            -args.margin,
            color="#bb5566",
            ls="--",
            lw=1.6,
            label=f"non-inferiority margin −{args.margin}",
        )
        ax.axvline(0.0, color="#555", lw=1.0)
        ax.axvline(d0, color="#222", lw=1.8, label=f"Δ = {d0:+.3f}")
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        ax.set_title(f"paired Δ{name} (op − spine)\n95% CI [{lo:+.3f}, {hi:+.3f}]")
        ax.set_xlabel("Δ skill (bootstrap over shots)")
        ax.legend(fontsize=8)
    err_s = np.linalg.norm(model_spine[:, 6:] - ref[:, 6:], axis=1) / np.sqrt(8) * 100
    err_o = np.linalg.norm(model_op[:, 6:] - ref[:, 6:], axis=1) / np.sqrt(8) * 100
    colours = np.where(flattop, "#bb5566", "#4477aa")
    axes[2].scatter(err_s, err_o, s=18, alpha=0.6, c=colours)
    lim = np.nanpercentile(np.concatenate([err_s, err_o]), 98)
    axes[2].plot([0, lim], [0, lim], color="#555", lw=1.0)
    axes[2].set_xlabel("spine LCFS RMS offset [cm]")
    axes[2].set_ylabel("operator LCFS RMS offset [cm]")
    axes[2].set_title("per-slice LCFS error (red = flat-top proxy)")
    fig.suptitle(
        f"Residual operator vs frozen classical spine — held-out n={n_scored}, "
        f"gate {'PASS' if non_inferior else 'FAIL'}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGURES / f"fig-residual-operator-gate{tag}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(
        "GATE %s: Δaxis %+0.3f CI %s | Δlcfs %+0.3f CI %s | n=%d | fig %s",
        "PASS" if non_inferior else "FAIL",
        delta["axis_skill_delta"],
        delta["axis_skill_delta_ci"],
        delta["lcfs_skill_delta"],
        delta["lcfs_skill_delta_ci"],
        n_scored,
        out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
