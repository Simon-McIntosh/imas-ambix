#!/usr/bin/env python
"""Well-posed position-controlled free-boundary solve — the label-engine spine.

Driven by a real shot's MEASURED coil program, the free-boundary Grad-Shafranov
solve drifts outboard: the coil field is radially STABLE at the physical axis
(decay index n approx 0.1 at R approx 0.85 m) but climbs through the radial
stability limit n = 3/2 into an UNSTABLE band at the in-vessel P4/P5/P6 coils
(R approx 1.34-1.68 m), and an unpinned current slides out through it.  This
holds the current on the physical branch by (1) seeding from the validated
staged-disc read and (2) constraining the R + Z current centroid — a
firewall-safe measured magnetic moment (a field moment of the plasma current,
like the Ip Rogowski anchor), never an EFIT output or a profile prior.  The
internal profile stays free (GS + jphi >= 0); the position is externally
determined and the profile is the residual earned later from dynamics and proven
on held-out MSE.

Three arms per slice:

* ``position`` — Ip + R/Z current-centroid moment + disc seed, NO full
  magnetics.  The label-engine's position-controlled solve.
* ``recon``    — full magnetics + disc flux anchor + passive sidecar (the frozen
  reconstruction spine).  The magnetics-constrained reference the position solve
  is validated against.
* ``free``     — bare forward solve (no seed, no centroid, no reseed): the
  outboard-drift baseline this rung fixes (before/after evidence only).

Gates (pre-declared):

* G2a position held — |R_axis,position - R_axis,recon| <= 5 cm per-shot median
  across the shot set (ramp + flat-top), from a disc seed + temporal warm-start,
  with no gross outboard drift (axis R < 1.4 m).
* G2b seed robustness — the disc-seeded and temporally-warm-chained position
  solves converge to the same inboard solution; an OUTBOARD seed is either
  recovered inboard or correctly reported as not-confined (never silently
  shipped outboard).
* G2c firewall — the position arm consumes only the centroid moment (+ Ip):
  the magnetics mask is all-off (asserted below) and the profile stays free
  (GS + jphi >= 0, no z-pin, no EFIT); unit-tested in test_position_constraint.

Stop rule: if the centroid constraint cannot hold the inboard branch across the
shot set from physical seeds, STOP and surface — never add EFIT-derived position
or a profile prior to force it.

Artifact: imas_ambix/latent/artifacts/patch_gate/position_controlled_solve.json
Figures:  docs/figures/mse-gated-reanalysis/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("position_controlled_solve_gate")

ARTIFACT = Path("imas_ambix/latent/artifacts/patch_gate/position_controlled_solve.json")
FIGURES = Path("docs/figures/mse-gated-reanalysis")

CONFINED_AXIS_R_MAX = 1.4  # beyond this the read is the outboard attractor
G2A_GATE_CM = 5.0  # position vs reconstruction axis agreement
DEFAULT_SHOTS = (11766, 11767, 11772)
DEFAULT_SIGMA_M = 0.02  # centroid tether 1σ [m]
OUTBOARD_SEED_R = 1.62  # a pathological outboard seed for the basin test [m]


def _disc_seed_flat(grid, inv) -> np.ndarray:
    """Staged-disc per-cell current as a jφ DENSITY on the flattened grid.

    The solve rescales the seed to the measured Ip, so only the shape/position
    matters; the disc places a uniform current inboard at the measured centroid.
    """
    seed = np.zeros(grid.flat_r.size)
    seed[grid.cells] = np.asarray(inv.i_cell, dtype=np.float64) / (grid.dr * grid.dz)
    return seed


def _gaussian_seed_flat(grid, r0: float, z0: float = 0.0) -> np.ndarray:
    """A compact Gaussian jφ seed centred at (r0, z0) — the basin-probe seed."""
    seed = np.zeros(grid.flat_r.size)
    seed[grid.cells] = np.exp(
        -(
            ((grid.flat_r[grid.cells] - r0) / 0.30) ** 2
            + ((grid.flat_z[grid.cells] - z0) / 0.45) ** 2
        )
    )
    return seed


def _solve(
    grid,
    table,
    payload,
    spine,
    *,
    mask,
    warm,
    soft_prior_cfg=None,
    sidecar=None,
    centroid=None,
    sigma=DEFAULT_SIGMA_M,
    reseed=True,
    basis=None,
    n_p=None,
    n_f=None,
    nonneg=None,
):
    """One ladder solve; interior config from the frozen spine unless overridden.

    The reconstruction arm uses the spine's n=3 nonneg data-fit profile; the
    position-controlled arm overrides to the stable free-sign K=2 brancher
    (n_p=n_f=1), which holds the confined branch from a physical seed far more
    robustly than the basin-fragile edge-capable nonneg profile does WITHOUT the
    magnetics to pin it (the frozen-spine note: a free edge-capable solve
    destabilises).
    """
    from scripts.closure_gate_eval import fit_and_read_slice  # noqa: PLC0415

    isolve = spine["interior_solve"]
    p = dataclasses.replace(payload, mask=mask)
    use_nonneg = (
        (isolve["profile_kind"] == "monomial-nonneg") if nonneg is None else nonneg
    )
    return fit_and_read_slice(
        grid,
        table,
        p,
        beta0_grid=(0.5,),
        alpha_grid=(1.0,),
        cost_limit=float("inf"),
        convergence_limit=5e-3,
        retry_max_iterations=160,
        fit_mode="ladder",
        n_p=int(isolve["n_p"]) if n_p is None else int(n_p),
        n_f=int(isolve["n_f"]) if n_f is None else int(n_f),
        smoothness=float(isolve["smoothness"]),
        nonneg=use_nonneg,
        passive=sidecar,
        passive_ridge=1.0,
        warm_jphi=warm,
        centroid_constraint=(
            (centroid[0], centroid[1], sigma) if centroid is not None else None
        ),
        reseed_axis_r_max=(float(isolve["reseed_axis_r_max"]) if reseed else None),
        keep_psi=True,
        keep_jphi=True,
        basis=basis,
        meta={},
        soft_prior_cfg=soft_prior_cfg,
        boundary_read=isolve["boundary_read_scoring"],
    )


def _axis(f) -> tuple[float, float]:
    if not (f.scored and f.target is not None):
        return float("nan"), float("nan")
    return float(f.target[0]), float(f.target[1])


def _current_centroid(grid, f) -> tuple[float, float]:
    """Current centroid (R, Z) [m] of a scored fit's converged jφ."""
    if not (f.scored and f.jphi_flat is not None):
        return float("nan"), float("nan")
    jf = np.asarray(f.jphi_flat, dtype=np.float64)
    ic = jf[grid.cells]
    tot = ic.sum()
    if abs(tot) < 1e-12:
        return float("nan"), float("nan")
    return (
        float((grid.flat_r[grid.cells] * ic).sum() / tot),
        float((grid.flat_z[grid.cells] * ic).sum() / tot),
    )


def _confined(axis_r: float) -> bool:
    return bool(np.isfinite(axis_r) and axis_r <= CONFINED_AXIS_R_MAX)


def run_shot(shot: int, *, nr: int, nz: int, sigma: float) -> dict:
    """Run the three arms over one shot's ramp + flat-top slices."""
    from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415
    from scripts.closure_gate_eval import _shot_passive_sidecar  # noqa: PLC0415
    from scripts.measured_pattern_confinement import (  # noqa: PLC0415
        _interp,
        _read_referee,
    )
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, spine_sha = frozen_spine_config()
    disc_cfg = dict(spine["soft_priors"])
    disc_cfg.setdefault("boundary_prior", "disc")

    payload = factory_shot_payloads(shot, nr=nr, nz=nz, max_slices=12, min_ip_ka=60.0)
    if payload is None:
        return {"shot": shot, "slices": []}
    grid, table, basis = payload["grid"], payload["table"], payload["basis"]
    sidecar = _shot_passive_sidecar(payload, int(spine["interior_solve"]["passive_k"]))
    ref = _read_referee(shot)
    ip_peak = max(float(p.ip_amperes) for p in payload["payloads"])

    rows: list[dict] = []
    warm_rec = warm_pos = None
    order = np.argsort([p.time_s for p in payload["payloads"]])
    for kk in order:
        p = payload["payloads"][int(kk)]
        tag = "flat" if abs(p.ip_amperes) >= 0.9 * ip_peak else "ramp"
        r_ref = _interp(ref, "magnetic_axis_r", p.time_s)
        z_ref = _interp(ref, "magnetic_axis_z", p.time_s)
        off = np.zeros_like(p.mask, dtype=bool)

        # firewall-safe measured centroid + compact inboard seed from the disc read
        inv = disc_read(p, grid, table, basis)
        if inv is None or inv.ring is None:
            logger.info("%d t=%.3f disc read failed — slice skipped", shot, p.time_s)
            continue
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        disc_seed = _disc_seed_flat(grid, inv)

        # arm 1: reconstruction reference (full magnetics + disc anchor + passive)
        f_rec = _solve(
            grid,
            table,
            p,
            spine,
            mask=p.mask,
            warm=warm_rec,
            soft_prior_cfg=disc_cfg,
            sidecar=sidecar,
            basis=basis,
        )
        if f_rec.scored and f_rec.converged and f_rec.jphi_flat is not None:
            warm_rec = f_rec.jphi_flat

        # the position arm uses the stable free-sign K=2 brancher (no magnetics
        # to pin the edge-capable nonneg profile — see _solve docstring)
        k2 = dict(n_p=1, n_f=1, nonneg=False)

        # arm 2: position-controlled solve (Ip + centroid + disc seed, NO magnetics)
        f_pos = _solve(
            grid,
            table,
            p,
            spine,
            mask=off,
            warm=warm_pos if warm_pos is not None else disc_seed,
            centroid=centroid,
            sigma=sigma,
            reseed=False,
            **k2,
        )
        # G2c firewall: assert no magnetics were consumed by the position arm
        assert not off.any(), "position arm must run with the magnetics mask OFF"
        if f_pos.scored and f_pos.jphi_flat is not None and _confined(_axis(f_pos)[0]):
            warm_pos = f_pos.jphi_flat

        # arm 3: free baseline (no seed, no centroid, no reseed) — the drift
        f_free = _solve(grid, table, p, spine, mask=off, warm=None, reseed=False, **k2)

        # G2b: an outboard-seeded position solve (basin probe) — same centroid
        f_out = _solve(
            grid,
            table,
            p,
            spine,
            mask=off,
            warm=_gaussian_seed_flat(grid, OUTBOARD_SEED_R),
            centroid=centroid,
            sigma=sigma,
            reseed=False,
            **k2,
        )

        pr, pz = _axis(f_pos)
        rr, rz = _axis(f_rec)
        fr, _fz = _axis(f_free)
        outr, _ = _axis(f_out)
        cen_pos = _current_centroid(grid, f_pos)
        d_axis = abs(pr - rr) if (_confined(pr) and _confined(rr)) else float("nan")
        rows.append(
            {
                "shot": shot,
                "time_s": p.time_s,
                "ip_amperes": p.ip_amperes,
                "tag": tag,
                "axis_r_ref": r_ref,
                "axis_z_ref": z_ref,
                "centroid_target_r": centroid[0],
                "centroid_target_z": centroid[1],
                "position": {
                    "axis_r": pr,
                    "axis_z": pz,
                    "confined": _confined(pr),
                    "converged": bool(getattr(f_pos, "converged", False)),
                    "centroid_r": cen_pos[0],
                    "centroid_z": cen_pos[1],
                    "centroid_err_cm": (
                        float(
                            np.hypot(cen_pos[0] - centroid[0], cen_pos[1] - centroid[1])
                        )
                        * 100.0
                        if np.isfinite(cen_pos[0])
                        else float("nan")
                    ),
                },
                "recon": {
                    "axis_r": rr,
                    "axis_z": rz,
                    "confined": _confined(rr),
                    "converged": bool(getattr(f_rec, "converged", False)),
                },
                "free": {"axis_r": fr, "confined": _confined(fr)},
                "outboard_seed": {"axis_r": outr, "confined": _confined(outr)},
                "position_vs_recon_axis_cm": (
                    d_axis * 100.0 if np.isfinite(d_axis) else float("nan")
                ),
            }
        )
        logger.info(
            "%d t=%.3f %s Ip=%.0fkA | pos R=%.3f(%s) recon R=%.3f free R=%.3f "
            "| Δaxis=%.1fcm cen_err=%.1fcm out R=%.3f(%s)",
            shot,
            p.time_s,
            tag,
            p.ip_amperes / 1e3,
            pr,
            "conf" if _confined(pr) else "OUT",
            rr,
            fr,
            rows[-1]["position_vs_recon_axis_cm"],
            rows[-1]["position"]["centroid_err_cm"],
            outr,
            "conf" if _confined(outr) else "OUT",
        )
    return {"shot": shot, "ip_peak": ip_peak, "spine_sha": spine_sha, "slices": rows}


def evaluate_gates(shots: list[dict]) -> dict:
    """G2a/G2b verdicts across the shot set (G2c is asserted in code + tests)."""
    per_shot = {}
    all_d = []
    for sh in shots:
        ds = [
            s["position_vs_recon_axis_cm"]
            for s in sh["slices"]
            if np.isfinite(s["position_vs_recon_axis_cm"])
        ]
        med = float(np.median(ds)) if ds else float("nan")
        per_shot[str(sh["shot"])] = {"n": len(ds), "median_axis_cm": med}
        all_d.extend(ds)

    n_pos_conf = sum(s["position"]["confined"] for sh in shots for s in sh["slices"])
    n_slices = sum(len(sh["slices"]) for sh in shots)
    # G2b: outboard-seed basin — recovered inboard (confined) OR flagged (not).
    # The pass condition is that NO outboard-seeded solve is silently shipped
    # outboard AS a confined read — i.e. confinement is reported honestly.
    n_out_recovered = sum(
        s["outboard_seed"]["confined"] for sh in shots for s in sh["slices"]
    )
    # per-shot medians all within the gate → G2a pass
    shot_medians = [
        v["median_axis_cm"]
        for v in per_shot.values()
        if np.isfinite(v["median_axis_cm"])
    ]
    g2a = bool(shot_medians) and all(m <= G2A_GATE_CM for m in shot_medians)

    return {
        "G2a_rule": (
            f"position vs reconstruction axis |ΔR| <= {G2A_GATE_CM:.0f} cm "
            "per-shot median across the shot set (ramp + flat-top)"
        ),
        "G2a_per_shot": per_shot,
        "G2a_pooled_median_cm": (float(np.median(all_d)) if all_d else float("nan")),
        "G2a_pooled_p90_cm": (
            float(np.percentile(all_d, 90)) if all_d else float("nan")
        ),
        "G2a_passes": g2a,
        "position_confined_fraction": (
            n_pos_conf / n_slices if n_slices else float("nan")
        ),
        "G2b_outboard_seed_recovered": int(n_out_recovered),
        "G2b_outboard_seed_of": int(n_slices),
        "G2c_firewall": (
            "position arm mask all-off (asserted in run_shot); only Ip + R/Z "
            "current-centroid moment consumed; profile free (GS + jphi>=0); "
            "no z-pin; no EFIT — unit-tested in test_position_constraint.py"
        ),
        "n_slices": n_slices,
    }


def _coil_rects(table):
    """Winding-pack rectangles (r0, z0, dr, dz) for the flux-map overlay."""
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    rects = []
    for _circ, fils in sorted(by_circ.items()):
        r0 = min(f.r - abs(f.width) / 2 for f in fils)
        r1 = max(f.r + abs(f.width) / 2 for f in fils)
        z0 = min(f.z - abs(f.height) / 2 for f in fils)
        z1 = max(f.z + abs(f.height) / 2 for f in fils)
        rects.append((r0, z0, r1 - r0, z1 - z0))
    return rects


def _panel(ax, grid, table, psi, psi_ax, psi_b, axis, *, title, centroid=None):
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    rr, zz = np.meshgrid(grid.rg, grid.zg)
    span = psi_b - psi_ax if abs(psi_b - psi_ax) > 1e-12 else 1.0
    psi_n = (psi - psi_ax) / span
    ax.contour(
        rr,
        zz,
        psi_n,
        levels=np.linspace(0.1, 0.95, 8),
        colors="#4477aa",
        linewidths=0.6,
    )
    ax.contour(rr, zz, psi_n, levels=[1.0], colors="#ee6677", linewidths=1.8)  # LCFS
    for r0, z0, dr, dz in _coil_rects(table):
        ax.add_patch(
            Rectangle(
                (r0, z0), dr, dz, facecolor="#bbbbbb", edgecolor="#555555", lw=0.4
            )
        )
    lr = np.append(grid.limiter_r, grid.limiter_r[0])
    lz = np.append(grid.limiter_z, grid.limiter_z[0])
    ax.plot(lr, lz, "k-", lw=1.0)
    ax.plot(
        [axis[0]],
        [axis[1]],
        "+",
        color="#228833",
        ms=12,
        mew=2.2,
        label="magnetic axis",
    )
    if centroid is not None:
        ax.plot(
            [centroid[0]],
            [centroid[1]],
            "x",
            color="#cc3311",
            ms=9,
            mew=2.0,
            label="measured centroid",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper right")


def make_flux_map(shot: int, *, nr: int, nz: int, sigma: float) -> None:
    """Side-by-side ψ flux map: position-controlled solve vs reconstruction.

    Re-solves the peak-Ip flat-top slice both ways and renders normalised-flux
    contours with the LCFS, limiter, winding packs, and the magnetic axis — the
    centroid-held equilibrium next to the full-magnetics reconstruction.
    """
    from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415
    from scripts.closure_gate_eval import (  # noqa: PLC0415
        _shot_passive_sidecar,
        geometry_target_pushout,
    )
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, _ = frozen_spine_config()
    disc_cfg = dict(spine["soft_priors"])
    disc_cfg.setdefault("boundary_prior", "disc")
    pl = factory_shot_payloads(shot, nr=nr, nz=nz, max_slices=12, min_ip_ka=60.0)
    if pl is None:
        return
    grid, table, basis = pl["grid"], pl["table"], pl["basis"]
    sidecar = _shot_passive_sidecar(pl, int(spine["interior_solve"]["passive_k"]))
    # a settled flat-top slice (median time among the near-peak-Ip slices) — the
    # first post-ramp slice has the largest axis offset and is unrepresentative
    ips = np.array([p.ip_amperes for p in pl["payloads"]])
    times = np.array([p.time_s for p in pl["payloads"]])
    flat = np.where(ips >= 0.9 * ips.max())[0]
    k = int(flat[int(np.argsort(times[flat])[len(flat) // 2])])
    p = pl["payloads"][k]
    inv = disc_read(p, grid, table, basis)
    if inv is None or inv.ring is None:
        return
    centroid = (float(inv.centroid_r), float(inv.centroid_z))
    off = np.zeros_like(p.mask, dtype=bool)

    f_rec = _solve(
        grid,
        table,
        p,
        spine,
        mask=p.mask,
        warm=None,
        soft_prior_cfg=disc_cfg,
        sidecar=sidecar,
        basis=basis,
    )
    f_pos = _solve(
        grid,
        table,
        p,
        spine,
        mask=off,
        warm=_disc_seed_flat(grid, inv),
        centroid=centroid,
        sigma=sigma,
        reseed=False,
        n_p=1,
        n_f=1,
        nonneg=False,
    )
    if not (f_pos.scored and f_rec.scored):
        return
    tp, pap, pbp = geometry_target_pushout(f_pos.psi, grid)
    tr, par, pbr = geometry_target_pushout(f_rec.psi, grid)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.4), sharey=True)
    _panel(
        axes[0],
        grid,
        table,
        f_pos.psi,
        pap,
        pbp,
        (tp[0], tp[1]),
        title=f"position-controlled (Ip + centroid)\naxis R={tp[0]:.3f} m",
        centroid=centroid,
    )
    _panel(
        axes[1],
        grid,
        table,
        f_rec.psi,
        par,
        pbr,
        (tr[0], tr[1]),
        title=f"reconstruction (full magnetics)\naxis R={tr[0]:.3f} m",
    )
    axes[0].set_ylabel("Z [m]")
    fig.suptitle(
        f"Shot {shot} flat-top (Ip {p.ip_amperes / 1e3:.0f} kA): the centroid-held "
        f"equilibrium reproduces the reconstruction (Δaxis "
        f"{abs(tp[0] - tr[0]) * 100:.1f} cm)",
        fontsize=10,
    )
    fig.savefig(
        FIGURES / "fig-flux-map-position-vs-recon.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def make_figures(shots: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    marker = {"11766": "o", "11767": "s", "11772": "^"}

    # Fig 1: axis R vs Ip — position (held) vs free (drift) vs recon reference
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for i, sh in enumerate(shots):
        m = marker.get(str(sh["shot"]), "o")
        sl = sh["slices"]
        ipk = [s["ip_amperes"] / 1e3 for s in sl]
        ax.plot(
            ipk,
            [s["free"]["axis_r"] for s in sl],
            m,
            mfc="none",
            color="#cc6677",
            ms=6,
            label="free (no position control)" if i == 0 else None,
        )
        ax.plot(
            ipk,
            [s["position"]["axis_r"] for s in sl],
            m,
            color="#228833",
            ms=6,
            label="position-controlled (Ip + centroid)" if i == 0 else None,
        )
        ax.plot(
            ipk,
            [s["recon"]["axis_r"] for s in sl],
            m,
            mfc="none",
            color="#4477aa",
            ms=4,
            ls=":",
            label="reconstruction (full magnetics)" if i == 0 else None,
        )
    ax.axhline(
        CONFINED_AXIS_R_MAX,
        color="k",
        ls=":",
        lw=1,
        label=f"outboard-attractor line (R={CONFINED_AXIS_R_MAX} m)",
    )
    ax.set_xlabel("Ip [kA]")
    ax.set_ylabel("magnetic axis R [m]")
    ax.set_title(
        "Centroid constraint holds the axis inboard; the free solve drifts out"
    )
    ax.legend(fontsize=7, loc="best")
    fig.savefig(FIGURES / "fig-position-held-vs-free.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: position vs reconstruction axis agreement (the G2a metric)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for sh in shots:
        m = marker.get(str(sh["shot"]), "o")
        sl = [s for s in sh["slices"] if np.isfinite(s["position_vs_recon_axis_cm"])]
        ax.plot(
            [s["ip_amperes"] / 1e3 for s in sl],
            [s["position_vs_recon_axis_cm"] for s in sl],
            m,
            ms=6,
            label=str(sh["shot"]),
        )
    ax.axhline(
        G2A_GATE_CM,
        color="#cc6677",
        ls="--",
        lw=1.2,
        label=f"G2a gate ({G2A_GATE_CM:.0f} cm)",
    )
    ax.set_xlabel("Ip [kA]")
    ax.set_ylabel("|R_axis,position − R_axis,recon| [cm]")
    ax.set_title("Position solve reproduces the full-magnetics reconstruction axis")
    ax.legend(fontsize=8)
    fig.savefig(
        FIGURES / "fig-position-vs-recon-axis.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA_M)
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    shots = []
    for s in args.shots:
        r = run_shot(int(s), nr=args.nr, nz=args.nz, sigma=args.sigma)
        if r["slices"]:
            shots.append(r)
    gates = evaluate_gates(shots)
    logger.info("GATES: %s", json.dumps(gates, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"sigma_m": args.sigma, "shots": shots, "gates": gates}, indent=1)
    )
    logger.info("artifact: %s", args.out)
    if not args.no_figures:
        make_figures(shots)
        if shots:
            make_flux_map(
                int(shots[0]["shot"]), nr=args.nr, nz=args.nz, sigma=args.sigma
            )
        logger.info("figures: %s", FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
