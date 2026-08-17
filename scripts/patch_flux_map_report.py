#!/usr/bin/env python
"""Flux-map comparison: patch-inverse ψ(R,Z) contours vs the EFIT ground truth.

For the eight held-out shots this renders, per shot, one rampup and one
flattop slice: the winner-config training-free patch-current inverse's
assembled poloidal flux ψ(R,Z) overlaid (imas-ink) on the firewalled EFIT
``efm`` reconstruction on the same machine cross-section, plus a quantitative
table (axis / LCFS / ψ_N=0.5 radial offsets).

Firewall (evaluator-only).  The EFIT ``psirz`` flux map, ``psi_axis`` /
``psi_boundary``, magnetic-axis and LCFS contour are read here for SCORING and
PLOTTING ONLY — nothing on this path feeds back into any fit.  The geometry
library readers deliberately EXCLUDE ``psirz`` (``imas_ambix/gs/geometry.py``);
this script reads the zarr ``efm`` group directly, inside the referee's
:func:`~imas_ambix.eval.efit_referee.evaluator_context` gate (the same
CODE-OUTPUTS-ONLY contract the referee enforces).

EFIT flux-map facts resolved empirically (recorded in the JSON):

* ``efm/psirz`` is stored ``(time, profile_z=65, profile_r=129)`` — its own
  ``shape`` attr ``[t,65,65]`` is wrong.  Only 65 of the 129 R columns are
  finite, and those columns' ``profile_r`` values are bit-identical to
  ``efm/gridr`` (65).  So the real map is ``(gridz=65, gridr=65)`` = (Z, R),
  recovered by selecting the finite R columns.  Orientation confirmed by the
  flux extremum landing on the reference magnetic axis.
* UNITS: EFIT ψ is per-radian [Wb/rad]; our patch ψ is TOTAL Φ = 2πR·A_φ [Wb].
  EFIT is multiplied by 2π for absolute-level comparison.  SIGN: MAST here has
  ``psi_axis > psi_boundary`` (axis is the flux maximum), matching our
  sign-aware axis read; imas-ink falls back to ψ_norm-matched contours if the
  absolute frames ever disagree.

Artifacts:  imas_ambix/latent/artifacts/patch_gate/flux_map_report.json
Figures:    docs/figures/patch-current-force-balance/fig-flux-maps-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from matplotlib.path import Path as MplPath

from imas_ambix.eval.efit_referee import evaluator_context
from imas_ambix.latent.data import CHANNEL_SCALE_KIND_FLOOR_REL, read_split_shot_lists
from imas_ambix.latent.patch_inverse import InverseConfig, invert_slices

# geometry_target + shot_payloads: reuse the gate's exact selection/read path.
from scripts.patch_gate_eval import geometry_target, shot_payloads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patch_flux_map_report")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/patch-current-force-balance")
L1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")

# THE P3 winner config (used verbatim, per the gate).  connectivity="locality"
# matches patch_gate_eval's --connectivity default — an exhibit must show the
# same regularisation the gates score, not InverseConfig's bare class default
# (connectivity=None), which would use a different structure-residual term.
WINNER = InverseConfig(
    policy="discrepancy",
    lambda_fb=3.0,
    misfit_ratio=1.5,
    lambda_max=100.0,
    iters=800,
    connectivity="locality",
)

SNAP_TOL_S = 0.015  # snap a slice to the nearest efm all_times within 15 ms
N_ANGLES = 240  # poloidal angles for radial-offset sampling


# --------------------------------------------------------------------------
# EFIT ground-truth read (evaluator-only firewall)
# --------------------------------------------------------------------------
def read_efit_slice(shot: int, time_s: float) -> dict | None:
    """Read the firewalled EFIT flux map + boundary nearest ``time_s``.

    Returns a dict with the 2π-scaled absolute flux map on (gridr, gridz), the
    axis/boundary flux, magnetic axis, LCFS contour, |Δt| to the snapped efm
    time, and the snapped time — or ``None`` if no efm slice is within tolerance.
    Must be called inside :func:`evaluator_context`.
    """
    z = zarr.open(str(L1_ROOT / f"{shot}.zarr"), mode="r")
    efm = z["efm"]
    all_times = np.asarray(efm["all_times"])
    eidx = int(np.argmin(np.abs(all_times - time_s)))
    dt = float(abs(all_times[eidx] - time_s))
    if dt > SNAP_TOL_S:
        return None

    gridr = np.asarray(efm["gridr"], dtype=np.float64).ravel()  # (65,) R [m]
    gridz = np.asarray(efm["gridz"], dtype=np.float64).ravel()  # (65,) Z [m]
    prof_r = np.asarray(efm["profile_r"], dtype=np.float64).ravel()  # (129,)
    raw = np.asarray(efm["psirz"][eidx], dtype=np.float64)  # (profile_z=65, 129)
    # select the finite R columns — they equal gridr (verified empirically)
    fin_cols = np.where(np.isfinite(raw).any(axis=0))[0]
    assert np.allclose(prof_r[fin_cols], gridr, atol=1e-4), "profile_r cols != gridr"
    psi_zr = raw[:, fin_cols]  # (Z=65, R=65) [Wb/rad]

    twopi = 2.0 * np.pi
    lcfs_r = np.asarray(efm["lcfs_r"][eidx], dtype=np.float64)
    lcfs_z = np.asarray(efm["lcfs_z"][eidx], dtype=np.float64)
    fin = np.isfinite(lcfs_r) & np.isfinite(lcfs_z) & (lcfs_r > 0)
    return {
        "eidx": eidx,
        "dt_s": dt,
        "time_efm_s": float(all_times[eidx]),
        "rg": gridr,
        "zg": gridz,
        "psi_zr": psi_zr * twopi,  # (Z, R) absolute total flux [Wb]
        "psi_axis": float(np.asarray(efm["psi_axis"])[eidx]) * twopi,
        "psi_boundary": float(np.asarray(efm["psi_boundary"])[eidx]) * twopi,
        "axis_r": float(np.asarray(efm["magnetic_axis_r"])[eidx]),
        "axis_z": float(np.asarray(efm["magnetic_axis_z"])[eidx]),
        "lcfs_r": lcfs_r[fin],
        "lcfs_z": lcfs_z[fin],
        "ip": float(np.asarray(efm["plasma_current_c"])[eidx]),
    }


# --------------------------------------------------------------------------
# contour geometry helpers
# --------------------------------------------------------------------------
def _closed_contour_about(rg, zg, psi_zr, level, axis_r, axis_z):
    """Largest closed ψ=level contour (nz,nr grid) enclosing the axis, (M,2)."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(rg, zg, psi_zr, levels=[level])
    segs = list(cs.allsegs[0]) if cs.allsegs else []
    plt.close(fig)
    best = None
    for s in segs:
        if len(s) < 8:
            continue
        if MplPath(s).contains_point((axis_r, axis_z)) and (
            best is None or len(s) > len(best)
        ):
            best = s
    return best


def _r_of_theta(contour_r, contour_z, axis_r, axis_z, angles):
    """Radius from (axis_r, axis_z) at each requested poloidal angle [m]."""
    th = np.arctan2(contour_z - axis_z, contour_r - axis_r)
    rad = np.hypot(contour_r - axis_r, contour_z - axis_z)
    order = np.argsort(th)
    th, rad = th[order], rad[order]
    # triplicate for wrap-around interpolation across ±π
    th_ext = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    rad_ext = np.concatenate([rad, rad, rad])
    return np.interp(angles, th_ext, rad_ext)


def radial_offset_cm(c_ours, c_efit, axis_r, axis_z):
    """|Δr(θ)| [cm] between two contours sampled about a common axis.

    Returns (median, p90, mean) or (nan, nan, nan) if either contour is missing.
    """
    if c_ours is None or c_efit is None or len(c_ours) < 8 or len(c_efit) < 8:
        return float("nan"), float("nan"), float("nan")
    angles = np.linspace(-np.pi, np.pi, N_ANGLES, endpoint=False)
    r1 = _r_of_theta(c_ours[:, 0], c_ours[:, 1], axis_r, axis_z, angles)
    r2 = _r_of_theta(c_efit[:, 0], c_efit[:, 1], axis_r, axis_z, angles)
    d = np.abs(r1 - r2) * 100.0
    return float(np.median(d)), float(np.percentile(d, 90)), float(np.mean(d))


# --------------------------------------------------------------------------
# imas-ink slice/geometry builders
# --------------------------------------------------------------------------
def _machine_geometry(grid, table):
    """MachineGeometry (limiter + PF coil boxes) from the campaign table."""
    from imas_ink._types import CoilRect, MachineGeometry

    lr = np.asarray(grid.limiter_r, dtype=np.float64)
    lz = np.asarray(grid.limiter_z, dtype=np.float64)
    clip = np.column_stack([np.append(lr, lr[0]), np.append(lz, lz[0])])
    # one bounding box per PF circuit
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    rects = []
    for circ, fils in sorted(by_circ.items()):
        r0 = min(f.r - abs(f.width) / 2 for f in fils)
        r1 = max(f.r + abs(f.width) / 2 for f in fils)
        z0 = min(f.z - abs(f.height) / 2 for f in fils)
        z1 = max(f.z + abs(f.height) / 2 for f in fils)
        rects.append(
            CoilRect(r=r0, z=z0, width=r1 - r0, height=z1 - z0, name=str(circ))
        )
    return MachineGeometry(
        wall_r=lr,
        wall_z=lz,
        coil_rects=rects,
        wall_clip_vertices=clip,
        wall_units=[(lr, lz)],
    )


def _our_slice(psi2d, grid, target, psi_ax, psi_b, ip, time_s, lcfs):
    """Our EquilibriumSlice (psi_2d transposed to indexing='ij' (nR, nZ))."""
    from imas_ink._types import EquilibriumSlice

    xpts = []
    for slot in range(2):
        rr, zz = target[2 + 2 * slot], target[3 + 2 * slot]
        if np.isfinite(rr) and np.isfinite(zz):
            xpts.append((float(rr), float(zz)))
    return EquilibriumSlice(
        psi_2d=np.ascontiguousarray(psi2d.T),  # (nz,nr) -> (nR,nZ)
        r_grid=np.asarray(grid.rg, dtype=np.float64),
        z_grid=np.asarray(grid.zg, dtype=np.float64),
        psi_axis=float(psi_ax),
        psi_boundary=float(psi_b),
        r_axis=float(target[0]),
        z_axis=float(target[1]),
        ip=float(ip),
        time=float(time_s),
        converged=True,
        x_points=xpts,
        boundary_r=None if lcfs is None else lcfs[:, 0],
        boundary_z=None if lcfs is None else lcfs[:, 1],
    )


def _efit_slice(efit):
    """EFIT reference EquilibriumSlice from a read_efit_slice dict."""
    from imas_ink._types import EquilibriumSlice

    return EquilibriumSlice(
        psi_2d=np.ascontiguousarray(efit["psi_zr"].T),  # (Z,R) -> (nR,nZ)
        r_grid=efit["rg"],
        z_grid=efit["zg"],
        psi_axis=efit["psi_axis"],
        psi_boundary=efit["psi_boundary"],
        r_axis=efit["axis_r"],
        z_axis=efit["axis_z"],
        ip=efit["ip"],
        time=efit["time_efm_s"],
        converged=True,
        boundary_r=efit["lcfs_r"],
        boundary_z=efit["lcfs_z"],
    )


def _fig_to_rgba(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


# --------------------------------------------------------------------------
# per-shot slice selection
# --------------------------------------------------------------------------
def select_slices(payloads, shot):
    """Pick (rampup, flattop) payload indices, each snapped to an efm slice.

    rampup = earliest slice whose time snaps within tolerance; flattop = the
    highest-|Ip| slice that snaps.  Returns [(kind, k, efit_dict), ...].
    """
    ip = np.array([p.ip_amperes for p in payloads])
    picks = []
    used = set()
    with evaluator_context():
        # rampup: earliest snappable
        for k in range(len(payloads)):
            efit = read_efit_slice(shot, payloads[k].time_s)
            if efit is not None:
                picks.append(("rampup", k, efit))
                used.add(k)
                break
        # flattop: highest |Ip| snappable, not the rampup slice
        for k in np.argsort(-ip):
            k = int(k)
            if k in used:
                continue
            efit = read_efit_slice(shot, payloads[k].time_s)
            if efit is not None:
                picks.append(("flattop", k, efit))
                break
    return picks


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scale-floor-rel",
        type=float,
        default=CHANNEL_SCALE_KIND_FLOOR_REL,
        help=(
            "rel_floor passed to shot_payloads' robust_channel_scale (F "
            "floor-sensitivity sweep; default = the training convention, "
            "0.05). Recorded in flux_map_report.json."
        ),
    )
    return ap.parse_args()


def main() -> int:
    from imas_ink.figures import equilibrium_figure_mpl

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "device=%s  winner=%s  scale_floor_rel=%s",
        device,
        WINNER,
        args.scale_floor_rel,
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    _, held_shots = read_split_shot_lists(40, 8)
    logger.info("held-out shots: %s", held_shots)

    panels = {"rampup": {}, "flattop": {}}  # regime -> shot -> rgba image
    slice_metrics = []

    for shot in held_shots:
        try:
            payload = shot_payloads(
                shot,
                nr=65,
                nz=97,
                max_slices=20,
                min_ip_ka=300.0,
                split="eval",
                scale_floor_rel=args.scale_floor_rel,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", shot, exc)
            continue
        if payload is None:
            logger.warning("shot %s: no gate slices", shot)
            continue
        grid, basis, payloads = payload["grid"], payload["basis"], payload["payloads"]
        table, refs = payload["table"], payload["refs"]

        picks = select_slices(payloads, shot)
        if not picks:
            logger.warning(
                "shot %s: no slice snaps within %.0f ms", shot, SNAP_TOL_S * 1e3
            )
            continue
        sel_k = [k for _, k, _ in picks]
        sel_payloads = [payloads[k] for k in sel_k]
        inv = invert_slices(basis, sel_payloads, WINNER, device=device)
        geom = _machine_geometry(grid, table)

        for (kind, k, efit), r in zip(picks, inv, strict=True):
            psi2d = basis.psi_grid_2d_np(r.i_cell, payloads[k].i_pf)  # (nz,nr) [Wb]
            target, psi_ax, psi_b = geometry_target(psi2d, grid)
            our_axis = (float(target[0]), float(target[1]))
            ref_axis = (float(refs[k][0]), float(refs[k][1]))  # referee target axis

            # contours (our LCFS + ψ_N=0.5; EFIT LCFS from efm + ψ_N=0.5)
            our_lcfs = _closed_contour_about(grid.rg, grid.zg, psi2d, psi_b, *our_axis)
            level_half_ours = psi_ax + 0.5 * (psi_b - psi_ax)
            our_half = _closed_contour_about(
                grid.rg, grid.zg, psi2d, level_half_ours, *our_axis
            )
            efit_axis = (efit["axis_r"], efit["axis_z"])
            efit_lcfs = np.column_stack([efit["lcfs_r"], efit["lcfs_z"]])
            level_half_efit = efit["psi_axis"] + 0.5 * (
                efit["psi_boundary"] - efit["psi_axis"]
            )
            efit_half = _closed_contour_about(
                efit["rg"], efit["zg"], efit["psi_zr"], level_half_efit, *efit_axis
            )

            axis_off = 100.0 * float(np.hypot(*(np.subtract(our_axis, ref_axis))))
            lcfs_med, lcfs_p90, _ = radial_offset_cm(our_lcfs, efit_lcfs, *efit_axis)
            _, _, half_mean = radial_offset_cm(our_half, efit_half, *efit_axis)

            slice_metrics.append(
                {
                    "shot": int(shot),
                    "regime": kind,
                    "time_s": round(float(payloads[k].time_s), 4),
                    "time_efm_s": round(efit["time_efm_s"], 4),
                    "snap_dt_ms": round(efit["dt_s"] * 1e3, 2),
                    "our_axis_rz": [round(our_axis[0], 4), round(our_axis[1], 4)],
                    "referee_axis_rz": [round(ref_axis[0], 4), round(ref_axis[1], 4)],
                    "efit_map_axis_rz": [
                        round(efit_axis[0], 4),
                        round(efit_axis[1], 4),
                    ],
                    "axis_offset_cm": round(axis_off, 2),
                    "lcfs_offset_median_cm": round(lcfs_med, 2),
                    "lcfs_offset_p90_cm": round(lcfs_p90, 2),
                    "psi_n_half_offset_cm": round(half_mean, 2),
                    "misfit": round(float(r.misfit), 4),
                    "structure": round(float(r.structure), 6),
                    "lambda_final": round(float(r.lambda_final), 3),
                    "ip_rel_err": round(float(r.ip_rel_err), 4),
                }
            )

            our_sl = _our_slice(
                psi2d,
                grid,
                target,
                psi_ax,
                psi_b,
                payloads[k].ip_amperes,
                payloads[k].time_s,
                our_lcfs,
            )
            fig, _ax = equilibrium_figure_mpl(
                our_sl,
                geom,
                reference_slice=_efit_slice(efit),
                reference_name="EFIT",
                figsize=(5.0, 6.4),
                show_probes=False,
                show_flux_loops=False,
            )
            fig.suptitle(
                f"{shot}  t={payloads[k].time_s:.3f}s  ({kind})",
                fontsize=10,
            )
            panels[kind][int(shot)] = _fig_to_rgba(fig)
            plt.close(fig)
            logger.info(
                "%d %-7s t=%.3f axis_off=%.1fcm lcfs_med=%.1fcm psiN0.5=%.1fcm",
                shot,
                kind,
                payloads[k].time_s,
                axis_off,
                lcfs_med,
                half_mean,
            )

    # ---- compose per-regime 2x4 grids -------------------------------------
    fig_paths = []
    for regime in ("rampup", "flattop"):
        shots = sorted(panels[regime])
        if not shots:
            continue
        ncol = 4
        nrow = int(np.ceil(len(shots) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 5.2 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, s in zip(axes, shots, strict=False):
            ax.imshow(panels[regime][s])
        fig.suptitle(
            f"Patch-inverse ψ(R,Z) vs EFIT — {regime} "
            f"(primary: patch-inverse, faint sienna: EFIT)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = FIGURES / f"fig-flux-maps-{regime}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        fig_paths.append(str(out))
        logger.info("wrote %s (%d panels)", out, len(shots))

    # ---- summary + JSON ---------------------------------------------------
    def _agg(key):
        vals = [m[key] for m in slice_metrics if np.isfinite(m[key])]
        return round(float(np.median(vals)), 2) if vals else None

    report = {
        "winner_config": {
            "policy": WINNER.policy,
            "lambda_fb": WINNER.lambda_fb,
            "misfit_ratio": WINNER.misfit_ratio,
            "lambda_max": WINNER.lambda_max,
            "iters": WINNER.iters,
        },
        "device": device,
        "scale_floor_rel": args.scale_floor_rel,
        "held_shots": [int(s) for s in held_shots],
        "n_slices": len(slice_metrics),
        "flux_map_findings": {
            "psirz_stored_shape": "(time, profile_z=65, profile_r=129)",
            "psirz_own_shape_attr": "[t, 65, 65] (WRONG)",
            "real_map_shape": "(gridz=65, gridr=65) via finite profile_r columns",
            "finite_R_columns_equal_gridr": True,
            "orientation": "axis 0 = Z (gridz), axis 1 = R (gridr); extremum lands on ref axis",
            "units": "EFIT psi is Wb/rad; multiplied by 2*pi to match our total flux Phi=2*pi*R*A_phi [Wb]",
            "sign": "psi_axis > psi_boundary (axis is flux maximum) — matches our sign-aware read; imas-ink falls back to psi_norm if frames disagree",
        },
        "aggregate_medians": {
            "axis_offset_cm": _agg("axis_offset_cm"),
            "lcfs_offset_median_cm": _agg("lcfs_offset_median_cm"),
            "psi_n_half_offset_cm": _agg("psi_n_half_offset_cm"),
        },
        "slices": slice_metrics,
        "figures": fig_paths,
    }
    out_json = ARTIFACTS / "flux_map_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s (%d slices)", out_json, len(slice_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
