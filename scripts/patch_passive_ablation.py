#!/usr/bin/env python
"""Passive/eddy-block ablation for the training-free variational patch inverse.

NOT production code -- a self-contained MEASURED experiment.  It answers one
evidence-gated question: is the inferred passive-conductor (eddy-current) block
LOAD-BEARING for the patch-current inverse on the held-out gate, or does the
variational plasma-only fit already explain the trustworthy magnetics at
flat-top (where eddy currents could plausibly matter) without it?

Design
------
The production inverse (:func:`imas_ambix.latent.patch_inverse.invert_slices`)
fits held-out magnetics for a per-cell plasma current vector with a whitened
sensor misfit + a Rogowski Ip anchor + a lambda*structure-residual force-balance
term.  It carries NO passive/eddy conductors.  This script builds an EXTENDED
arm: the SAME plasma inverse PLUS a rank-4 passive-amplitude vector on the
truncated-SVD low-rank passive basis (:func:`imas_ambix.gs.residual
.passive_lowrank_basis`, sized from the measured passive effective rank).  The
passive amplitudes enter ONLY the sensor prediction (the data term) -- they are
structural eddy currents, carry no Grad-Shafranov flux-function structure, and
so are deliberately EXCLUDED from the jphi^2-weighted structure residual and
from the plasma-current Ip anchor.  They are penalised by a light L2 ridge on
the whitened sensor-field contribution (equivalently ``sum_k s_k^2 a_k^2`` on
the orthonormal SVD amplitudes -- an L2 scaled by the rank-4 singular values),
which breaks the near-collinear plasma<->passive cancellation the residual
module warns about while still letting genuinely-needed eddy currents in.

Because the augmented sensor model cannot be expressed through the PUBLIC
``invert_slices`` API without polluting the structure residual with fake
zero-area passive "cells", the small Adam loop is REIMPLEMENTED locally
(:func:`invert_slices_passive`).  With ``use_passives=False`` it is byte-for-byte
the production loop (same seed, same schedule via the imported
``_lambda_schedule``, same loss composition), so an EQUIVALENCE CHECK against
``invert_slices`` validates the reimplementation before the passive arm is
trusted.

The geometry (axis / X-points / LCFS) is read from the plasma+coil flux in BOTH
arms (:meth:`PatchBasis.psi_grid_2d_np` -- passives do not contribute a grid
Green's here; they enter the DATA term only, as documented).  The comparison
therefore isolates whether letting passives absorb sensor residual changes the
recovered PLASMA current -- and hence the misfit and the geometry.

Verdict
-------
"load-bearing" iff the flat-top misfit improves >= 20% AND the axis skill does
not degrade; otherwise "not load-bearing at the current sensor set".  The plan
keeps the passive block ONLY on this evidence.

Artifacts:  imas_ambix/latent/artifacts/patch_gate/passive_ablation{tag}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from imas_ambix.gs.operator import build_operator
from imas_ambix.gs.residual import passive_lowrank_basis
from imas_ambix.latent.data import read_split_shot_lists  # noqa: E402
from imas_ambix.latent.patch_inverse import (
    InverseConfig,
    SliceInversion,
    SlicePayload,
    _lambda_schedule,
    invert_slices,
)
from imas_ambix.latent.structure_residual import structure_residual

# Reuse the gate machinery verbatim so the numbers are directly comparable.
from scripts.patch_gate_eval import (  # noqa: E402
    geometry_target,
    score,
    shot_payloads,
    train_mean_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_passive_ablation")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")


# --------------------------------------------------------------------------
# passive low-rank sensor columns aligned to the patch-basis sensor rows
# --------------------------------------------------------------------------


def passive_sensor_columns(table, basis, scale: np.ndarray, rank: int):
    """Rank-``rank`` passive sensor columns aligned to ``basis.sensor_channels``.

    Rebuilds the campaign forward operator to read its inferred-passive Green's
    block ``g_passive`` (n_sensor_fwd x n_passive_circuit), aligns its rows to
    the patch-basis sensor order (the order the payloads use), and returns the
    leading ``rank`` truncated-SVD modes of the ROW-WHITENED design.

    Returns ``(passive_cols, v_basis, s_vals, present)`` where
    ``passive_cols`` (S_basis x rank) are the RAW-unit [Wb/T per unit amplitude]
    sensor columns, ``v_basis`` (n_passive x rank) maps an amplitude vector back
    to physical passive circuit currents [A], ``s_vals`` (rank,) are the whitened
    singular values (the L2 metric), and ``present`` (S_basis,) flags the
    patch-basis channels that map to a forward-operator row.  Returns ``None``
    if the campaign carries no passive block.
    """
    fwd = build_operator(table)
    g_passive = np.asarray(fwd.g_passive, dtype=np.float64)  # (S_fwd, n_passive)
    if g_passive.shape[1] == 0:
        return None
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    ch_rows = np.array([row_of.get(ch, -1) for ch in basis.sensor_channels])
    present = ch_rows >= 0
    s_basis = len(basis.sensor_channels)
    g_aligned = np.zeros((s_basis, g_passive.shape[1]), dtype=np.float64)
    g_aligned[present] = g_passive[ch_rows[present]]
    # row-whiten (1/scale) so the SVD ranks the passive modes by their
    # trustworthy-sensor field energy, exactly as gs.residual does; absent rows
    # stay zero and contribute nothing to the SVD.
    w = np.zeros(s_basis, dtype=np.float64)
    w[present] = 1.0 / np.asarray(scale, dtype=np.float64)[present]
    g_white = w[:, None] * g_aligned
    v_basis = passive_lowrank_basis(g_white, rank)  # (n_passive, r)
    r = v_basis.shape[1]
    if r == 0:
        return None
    # whitened singular values = column norms of the whitened design projected
    # onto the retained right-singular vectors.
    s_vals = np.linalg.norm(g_white @ v_basis, axis=0)  # (r,)
    passive_cols = g_aligned @ v_basis  # (S_basis, r) raw units per unit amp
    return passive_cols, v_basis, s_vals, present


# --------------------------------------------------------------------------
# local Adam loop -- production inverse + optional passive amplitude vector
# --------------------------------------------------------------------------


def invert_slices_passive(
    basis,
    payloads: list[SlicePayload],
    cfg: InverseConfig,
    *,
    passive_cols: np.ndarray | None,
    s_vals: np.ndarray | None,
    use_passives: bool,
    passive_l2: float,
    device: str | torch.device = "cpu",
) -> tuple[list[SliceInversion], np.ndarray | None]:
    """Mirror of ``invert_slices`` with an optional passive amplitude vector.

    With ``use_passives=False`` the loop is identical to the production inverse
    (validated by the equivalence check).  With ``use_passives=True`` a per-slice
    rank-r passive amplitude vector ``a_pass`` is co-optimised, entering ONLY the
    sensor prediction and penalised by ``passive_l2 * sum_k s_k^2 a_k^2`` (an L2
    ridge scaled by the rank-r whitened singular values).  Returns the per-slice
    inversions and the final ``a_pass`` array (B x r) or ``None``.
    """
    dev = torch.device(device)
    dt = cfg.dtype
    n = int(basis.r_cells.shape[0])
    b = len(payloads)

    m_sens = basis.m_sens.to(device=dev, dtype=dt)
    g_cc = basis.g_cc.to(device=dev, dtype=dt)
    r_c = basis.r_cells.to(device=dev, dtype=dt)
    z_c = basis.z_cells.to(device=dev, dtype=dt)
    candidate = basis.candidate_mask.to(device=dev, dtype=dt)
    cell_area = float(basis.cell_area)

    meas = torch.stack(
        [torch.as_tensor(np.nan_to_num(p.measured), dtype=dt) for p in payloads]
    ).to(dev)
    vac = torch.stack([torch.as_tensor(p.vacuum, dtype=dt) for p in payloads]).to(dev)
    mask = torch.stack(
        [torch.as_tensor(p.mask.astype(np.float64), dtype=dt) for p in payloads]
    ).to(dev)
    scale = torch.stack([torch.as_tensor(p.scale, dtype=dt) for p in payloads]).to(dev)
    ip = torch.tensor([p.ip_amperes for p in payloads], dtype=dt, device=dev)
    psi_coil = torch.stack(
        [
            basis.psi_coil_cells_for(np.asarray(p.i_pf, dtype=np.float64))
            for p in payloads
        ]
    ).to(device=dev, dtype=dt)

    seed = torch.exp(
        -(((r_c - basis.r0) / cfg.seed_width_r) ** 2 + (z_c / cfg.seed_width_z) ** 2)
    )
    seed = seed / seed.sum() * n
    x = seed.expand(b, n).clone().requires_grad_(True)

    active_passive = (
        use_passives and passive_cols is not None and passive_cols.shape[1] > 0
    )
    if active_passive:
        pcols = torch.as_tensor(passive_cols, dtype=dt, device=dev)  # (S, r)
        sv = torch.as_tensor(s_vals, dtype=dt, device=dev)  # (r,)
        a_pass = torch.zeros(b, pcols.shape[1], dtype=dt, device=dev).requires_grad_(
            True
        )
        opt = torch.optim.Adam([x, a_pass], lr=cfg.lr)
    else:
        a_pass = None
        opt = torch.optim.Adam([x], lr=cfg.lr)

    lam = torch.zeros(b, dtype=dt, device=dev)
    target = torch.full((b,), float("inf"), dtype=dt, device=dev)
    warmup_end = int(cfg.warmup_fraction * cfg.iters)

    misfit = torch.zeros(b, dtype=dt, device=dev)
    fb = torch.zeros(b, dtype=dt, device=dev)
    for step in range(cfg.iters):
        with torch.no_grad():
            lam = _lambda_schedule(cfg, step, lam, misfit.detach(), target)
        opt.zero_grad()
        i_eff = x * candidate * (ip[:, None] / n)
        pred = vac + i_eff @ m_sens.T
        if a_pass is not None:
            pred = pred + a_pass @ pcols.T
        misfit = (mask * ((pred - meas) / scale) ** 2).sum(-1) / mask.sum(-1).clamp_min(
            1.0
        )
        ip_pen = ((i_eff.sum(-1) - ip) / ip) ** 2
        psi_c = i_eff @ g_cc.T + psi_coil
        fb_rows = []
        for k in range(b):
            fb_rows.append(
                structure_residual(
                    psi_c[k],
                    r_c,
                    i_eff[k] / cell_area,
                    n_bins=cfg.n_bins,
                    form=cfg.form,
                    z_c=z_c,
                    connectivity=cfg.connectivity,
                    locality_scale=cfg.locality_scale,
                )
            )
        fb = torch.stack(fb_rows)
        loss = misfit + cfg.ip_weight * ip_pen + lam * fb
        if a_pass is not None:
            loss = loss + passive_l2 * (sv[None, :] ** 2 * a_pass**2).sum(-1)
        loss = loss.sum()
        loss.backward()
        opt.step()
        if cfg.policy == "discrepancy" and step == max(warmup_end - 1, 0):
            target = cfg.misfit_ratio * misfit.detach().clone()

    out: list[SliceInversion] = []
    with torch.no_grad():
        i_fin = (x * candidate * (ip[:, None] / n)).cpu().numpy()
        for k, p in enumerate(payloads):
            out.append(
                SliceInversion(
                    i_cell=i_fin[k],
                    misfit=float(misfit[k]),
                    structure=float(fb[k]),
                    lambda_final=float(lam[k]),
                    ip_rel_err=float(abs(i_fin[k].sum() - p.ip_amperes) / p.ip_amperes),
                    shot=p.shot,
                    t_index=p.t_index,
                    time_s=p.time_s,
                )
            )
        a_pass_np = a_pass.detach().cpu().numpy() if a_pass is not None else None
    return out, a_pass_np


# --------------------------------------------------------------------------
# experiment driver
# --------------------------------------------------------------------------


def load_flattop_shots(eval_shots, args):
    """Per-shot flat-top payloads + aligned passive columns."""
    shots = []
    for s in eval_shots:
        try:
            pay = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if pay is None:
            continue
        keep = [i for i, p in enumerate(pay["payloads"]) if p.time_s >= args.flat_top_s]
        if not keep:
            continue
        pay["payloads"] = [pay["payloads"][i] for i in keep]
        pay["refs"] = pay["refs"][keep]
        scale = pay["payloads"][0].scale
        pc = passive_sensor_columns(
            pay["table"], pay["basis"], scale, rank=args.passive_rank
        )
        pay["passive"] = pc
        shots.append(pay)
        logger.info(
            "shot %d: %d flat-top slices, passive rank=%s",
            pay["payloads"][0].shot,
            len(pay["payloads"]),
            0 if pc is None else pc[0].shape[1],
        )
    return shots


def passive_slice_stats(pay, a_pass) -> list[dict]:
    """Per-slice passive-usage stats: physical current, Ip fraction, field share."""
    if a_pass is None or pay["passive"] is None:
        return []
    passive_cols, v_basis, _s, present = pay["passive"]
    rows = []
    for k, p in enumerate(pay["payloads"]):
        c_passive = v_basis @ a_pass[k]  # (n_passive,) [A]
        contrib = passive_cols @ a_pass[k]  # (S,) raw sensor units
        m = p.mask & present
        w = np.zeros_like(p.scale)
        w[m] = 1.0 / p.scale[m]
        field_w = np.linalg.norm((contrib * w)[m])
        meas_w = np.linalg.norm((np.nan_to_num(p.measured) * w)[m])
        rows.append(
            {
                "c_passive_norm_A": float(np.linalg.norm(c_passive)),
                "c_passive_frac_ip": float(
                    np.linalg.norm(c_passive) / max(p.ip_amperes, 1.0)
                ),
                "passive_field_frac": float(field_w / meas_w) if meas_w > 0 else 0.0,
            }
        )
    return rows


def run_arm(shots, cfg, baseline_vec, *, use_passives, passive_l2, device):
    """Invert every flat-top slice under one arm; score geometry + misfit."""
    model_rows, ref_rows, misfits = [], [], []
    passive_rows: list[dict] = []
    t0 = time.perf_counter()
    for pay in shots:
        grid, basis = pay["grid"], pay["basis"]
        pc = pay["passive"]
        inv, a_pass = invert_slices_passive(
            basis,
            pay["payloads"],
            cfg,
            passive_cols=None if pc is None else pc[0],
            s_vals=None if pc is None else pc[2],
            use_passives=use_passives,
            passive_l2=passive_l2,
            device=device,
        )
        for k, r in enumerate(inv):
            psi2d = basis.psi_grid_2d_np(r.i_cell, pay["payloads"][k].i_pf)
            target, _pa, _pb = geometry_target(psi2d, grid)
            model_rows.append(target)
            ref_rows.append(pay["refs"][k])
            misfits.append(r.misfit)
        if use_passives:
            passive_rows.extend(passive_slice_stats(pay, a_pass))
    dt = time.perf_counter() - t0
    model = np.array(model_rows)
    ref = np.array(ref_rows)
    sc = score(model, ref, baseline_vec)
    sc.pop("axis_errors")
    mis = np.array(misfits)
    out = {
        "n_scored": int(len(model)),
        "misfit_median": float(np.median(mis)),
        "misfit_p90": float(np.percentile(mis, 90)),
        "axis_skill": sc["axis_skill"],
        "lcfs_skill": sc["lcfs_skill"],
        "axis_error_median_m": sc["axis_error_median_m"],
        "wall_s": dt,
    }
    if use_passives and passive_rows:
        out["passive_c_norm_A_median"] = float(
            np.median([r["c_passive_norm_A"] for r in passive_rows])
        )
        out["passive_frac_ip_median"] = float(
            np.median([r["c_passive_frac_ip"] for r in passive_rows])
        )
        out["passive_field_frac_median"] = float(
            np.median([r["passive_field_frac"] for r in passive_rows])
        )
        out["passive_field_frac_p90"] = float(
            np.percentile([r["passive_field_frac"] for r in passive_rows], 90)
        )
    return out, model


def equivalence_check(shots, cfg, baseline_vec, device):
    """Compare the local no-passive loop against the production ``invert_slices``.

    Both must reproduce the same axis-error median on the flat-top subset -- the
    validation that the reimplementation is faithful before the passive arm is
    trusted.  Returns the two medians and their difference [cm].
    """
    local_model, ref_rows = [], []
    prod_model = []
    for pay in shots:
        grid, basis = pay["grid"], pay["basis"]
        loc, _ = invert_slices_passive(
            basis,
            pay["payloads"],
            cfg,
            passive_cols=None,
            s_vals=None,
            use_passives=False,
            passive_l2=0.0,
            device=device,
        )
        prod = invert_slices(basis, pay["payloads"], cfg, device=device)
        for k in range(len(loc)):
            lm, _, _ = geometry_target(
                basis.psi_grid_2d_np(loc[k].i_cell, pay["payloads"][k].i_pf), grid
            )
            pm, _, _ = geometry_target(
                basis.psi_grid_2d_np(prod[k].i_cell, pay["payloads"][k].i_pf), grid
            )
            local_model.append(lm)
            prod_model.append(pm)
            ref_rows.append(pay["refs"][k])
    local_model = np.array(local_model)
    prod_model = np.array(prod_model)
    ref = np.array(ref_rows)
    la = np.nanmedian(
        np.hypot(local_model[:, 0] - ref[:, 0], local_model[:, 1] - ref[:, 1])
    )
    pa = np.nanmedian(
        np.hypot(prod_model[:, 0] - ref[:, 0], prod_model[:, 1] - ref[:, 1])
    )
    return {
        "local_axis_median_m": float(la),
        "prod_axis_median_m": float(pa),
        "diff_cm": float(abs(la - pa) * 100.0),
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
    ap.add_argument("--flat-top-s", type=float, default=0.15)
    ap.add_argument("--passive-rank", type=int, default=4)
    ap.add_argument("--passive-l2", type=float, default=1e-2)
    # winner gate config: discrepancy lambda0=3, ratio 1.5, lambda_max 100, 800 it
    ap.add_argument("--lambda-fb", type=float, default=3.0)
    ap.add_argument("--misfit-ratio", type=float, default=1.5)
    ap.add_argument("--lambda-max", type=float, default=100.0)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--form", type=str, default="affine-r2")
    ap.add_argument("--connectivity", type=str, default="locality")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--probe",
        type=int,
        default=2,
        help="fast CPU pass over the first N held-out shots (0 = full 8-shot run)",
    )
    ap.add_argument("--out-tag", type=str, default="")
    args = ap.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    connectivity = None if args.connectivity in ("", "none") else args.connectivity
    logger.info("device=%s connectivity=%s probe=%d", device, connectivity, args.probe)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    _train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    eval_shots = held_shots[: args.probe] if args.probe else held_shots
    tag = args.out_tag or (f"_probe{args.probe}" if args.probe else "_gate")

    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )

    cfg = InverseConfig(
        iters=args.iters,
        lr=args.lr,
        lambda_fb=args.lambda_fb,
        policy="discrepancy",
        misfit_ratio=args.misfit_ratio,
        lambda_max=args.lambda_max,
        n_bins=args.n_bins,
        form=args.form,
        connectivity=connectivity,
    )

    shots = load_flattop_shots(eval_shots, args)
    if not shots:
        logger.error("no flat-top slices loaded -- nothing to ablate")
        return 1

    logger.info("equivalence check: local no-passive loop vs invert_slices ...")
    equiv = equivalence_check(shots, cfg, baseline_vec, device)
    logger.info(
        "equivalence: local %.4f m vs prod %.4f m -> %.3f cm",
        equiv["local_axis_median_m"],
        equiv["prod_axis_median_m"],
        equiv["diff_cm"],
    )

    logger.info("arm A: plasma-only (no passives) ...")
    arm_off, _ = run_arm(
        shots, cfg, baseline_vec, use_passives=False, passive_l2=0.0, device=device
    )
    logger.info(
        "  misfit median=%.4f p90=%.4f axis_skill=%.3f axis_median=%.4f m",
        arm_off["misfit_median"],
        arm_off["misfit_p90"],
        arm_off["axis_skill"],
        arm_off["axis_error_median_m"],
    )

    logger.info("arm B: plasma + rank-%d passives ...", args.passive_rank)
    arm_on, _ = run_arm(
        shots,
        cfg,
        baseline_vec,
        use_passives=True,
        passive_l2=args.passive_l2,
        device=device,
    )
    logger.info(
        "  misfit median=%.4f p90=%.4f axis_skill=%.3f axis_median=%.4f m "
        "passive |c|=%.0f A (%.3f Ip) field_frac=%.3f",
        arm_on["misfit_median"],
        arm_on["misfit_p90"],
        arm_on["axis_skill"],
        arm_on["axis_error_median_m"],
        arm_on.get("passive_c_norm_A_median", float("nan")),
        arm_on.get("passive_frac_ip_median", float("nan")),
        arm_on.get("passive_field_frac_median", float("nan")),
    )

    # verdict: load-bearing iff flat-top misfit improves >= 20% AND axis skill
    # does not degrade.
    m_off = arm_off["misfit_median"]
    m_on = arm_on["misfit_median"]
    misfit_improvement = (m_off - m_on) / m_off if m_off > 0 else 0.0
    axis_ok = arm_on["axis_skill"] >= arm_off["axis_skill"]
    load_bearing = misfit_improvement >= 0.20 and axis_ok
    verdict = (
        "load-bearing" if load_bearing else "not load-bearing at the current sensor set"
    )
    logger.info(
        "VERDICT: %s (misfit improvement %.1f%%, axis skill %s: %.3f -> %.3f)",
        verdict,
        100.0 * misfit_improvement,
        "held" if axis_ok else "DEGRADED",
        arm_off["axis_skill"],
        arm_on["axis_skill"],
    )

    result = {
        "schema": "patch-passive-ablation-v0",
        "note": (
            "measured evidence-gated experiment; passives enter the sensor DATA "
            "term only (excluded from the structure residual and Ip anchor); "
            "geometry read from plasma+coil flux in BOTH arms"
        ),
        "config": {k: v for k, v in vars(args).items()},
        "device": device,
        "n_shots": len(shots),
        "equivalence_check": equiv,
        "arm_no_passives": arm_off,
        "arm_with_passives": arm_on,
        "verdict": {
            "load_bearing": bool(load_bearing),
            "verdict": verdict,
            "misfit_improvement": float(misfit_improvement),
            "axis_skill_held": bool(axis_ok),
            "criterion": (
                "load-bearing iff flat-top misfit improves >=20% AND axis skill "
                "does not degrade"
            ),
        },
    }
    out_path = ARTIFACTS / f"passive_ablation{tag}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("artifact written to %s", out_path)

    sbatch = (
        "sbatch --partition=betelgeuse_debug --gres=gpu:1 --cpus-per-task=8 "
        "--mem=32G --time=00:30:00 --wrap='cd "
        + str(Path.cwd())
        + " && export TMPDIR=/tmp && uv run python scripts/patch_passive_ablation.py "
        "--probe 0'"
    )
    logger.info("FULL 8-shot GPU run (submit yourself):\n%s", sbatch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
