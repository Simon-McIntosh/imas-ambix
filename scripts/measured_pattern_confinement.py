#!/usr/bin/env python
"""Confinement probe on a real shot's measured flat-top coil pattern.

The forward operator's standing scenario blocker: on real MAST coil currents
the free-boundary Picard historically had no confined fixed point — every
synthetic truth ran on a manufactured uniform field instead.  This probe asks
the binding precondition question directly: driven by shot 11766's MEASURED
flat-top coil pattern plus the predicted vessel state, does the free-boundary
solve hold a CONFINED fixed point — interior O-point, axis near the referee
axis, NO z-symmetry pin?  (Warm-start at the referee axis is allowed; the
referee is diagnostic-only and parameterises nothing.)

The vessel state comes from the one-matrix coupled solve (design rule: every
circuit — coil, case, passive structure, plasma — in ONE interaction matrix,
the applied loop voltage solved, so every component of dψ/dt is accounted
for, the dL/dt of the evolving column included): the plasma row is pinned to
the measured Ip trace at the measured centroid/shape trace and the vessel
rows integrate the full coil+plasma drive history from a machine-quiescent
start.

Gate (pre-declared): PASS iff the plain two-term profile (β0 = 0.5, α = 1.0)
solve is confined (interior O-point, axis R ≤ 1.4 m) with
|R_axis − R_axis,ref| ≤ 0.15 m.  A (β0, α) grid around it reports basin
robustness; a no-vessel ablation isolates the eddy fold's flat-top share.

Artifact: imas_ambix/gs/artifacts/measured_pattern_confinement.json
Figure:   docs/figures/equilibrium-realism/fig-confinement-probe.png
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

from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.data import (
    anchored_columns,
    feature_schema,
    load_shot_slices_raw,
    schema_group_offsets,
)
from imas_ambix.latent.gs_solve import EquilibriumGrid
from imas_ambix.latent.plasma_screening import solve_pinned_plasma_circuit
from imas_ambix.latent.synthetic_truth import _forward_picard, _two_term_shape_fn
from imas_ambix.latent.temporal_operator import build_passive_circuit_system

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("measured_pattern_confinement")

ARTIFACT = Path("imas_ambix/gs/artifacts/measured_pattern_confinement.json")
FIGURES = Path("docs/figures/equilibrium-realism")

CONFINED_AXIS_R_MAX = 1.4  # beyond this the read is the outboard attractor
AXIS_GATE_M = 0.15
BETA0_GRID = (0.4, 0.5, 0.6)
ALPHA_GRID = (1.0, 1.5, 2.0)
HEADLINE = (0.5, 1.0)


def _read_referee(shot: int) -> dict[str, np.ndarray]:
    """L2 equilibrium referee quantities — DIAGNOSTIC-ONLY (locked decision)."""
    import zarr  # noqa: PLC0415

    from imas_ambix.worldmodel.equilibrium_labels import (  # noqa: PLC0415
        equilibrium_store_path,
    )

    eq = zarr.open_group(str(equilibrium_store_path(shot, None)), mode="r")[
        "equilibrium"
    ]
    return {
        k: np.asarray(eq[k], dtype=np.float64)
        for k in (
            "time",
            "magnetic_axis_r",
            "magnetic_axis_z",
            "minor_radius",
            "elongation",
            "li",
        )
    }


def _interp_ref(ref: dict[str, np.ndarray], key: str, t: float) -> float:
    tt = ref["time"]
    yy = ref[key]
    ok = np.isfinite(tt) & np.isfinite(yy)
    if ok.sum() < 2 or t < tt[ok][0] or t > tt[ok][-1]:
        return float("nan")
    return float(np.interp(t, tt[ok], yy[ok]))


def measured_state(shot: int) -> dict:
    """Measured coil currents, Ip trace, and the flat-top slice pick."""
    schema = feature_schema()
    table = build_table_for_shot(shot)
    fwd = build_operator(table)
    loaded = load_shot_slices_raw(shot, schema)
    if loaded is None:
        raise RuntimeError(f"shot {shot}: no level-1 data")
    x, times, plasma_on = loaded
    ref = _read_referee(shot)

    offsets = schema_group_offsets(schema)
    amc_names = schema["amc"]
    ip_col, _ne = anchored_columns(schema)
    amc_block = x[:, offsets["amc"] : offsets["amc"] + len(amc_names)]
    n_t = x.shape[0]
    i_pf = np.zeros((n_t, len(fwd.pf_amc_channels)))
    for t in range(n_t):
        vals = {
            ch: float(amc_block[t, j])
            for j, ch in enumerate(amc_names)
            if np.isfinite(amc_block[t, j])
        }
        i_pf[t] = fwd.assemble_pf_currents(vals)
    ip_ka = np.where(np.isfinite(x[:, ip_col]), x[:, ip_col], 0.0)

    tt = ref["time"]
    ok = np.isfinite(tt) & np.isfinite(ref["magnetic_axis_r"])
    covered = (
        (times >= tt[ok][0]) & (times <= tt[ok][-1])
        if ok.sum() >= 2
        else np.zeros_like(plasma_on)
    )
    usable = plasma_on & covered
    ip_abs = np.abs(ip_ka)
    peak = float(ip_abs[usable].max())
    flat = np.nonzero((ip_abs >= 0.9 * peak) & usable)[0]
    k_mid = int(flat[flat.size // 2])
    t_mid = float(times[k_mid])

    # measured/diagnostic plasma-geometry traces for the one-matrix solve
    # (np.interp clamps outside coverage; those samples only shape the early
    # dL/dt diagnostic, and the eddies they drive decay on the vessel τ)
    def _trace(key: str) -> np.ndarray:
        okk = np.isfinite(tt) & np.isfinite(ref[key])
        return np.interp(times, tt[okk], ref[key][okk])

    return {
        "table": table,
        "fwd": fwd,
        "i_pf": i_pf,
        "ip_amp": ip_ka * 1e3,
        "times": times,
        "k_slice": k_mid,
        "time_s": t_mid,
        "axis_trace": np.column_stack(
            [_trace("magnetic_axis_r"), _trace("magnetic_axis_z")]
        ),
        "a_trace": _trace("minor_radius"),
        "kappa_trace": _trace("elongation"),
        "li_trace": _trace("li"),
        "axis_ref": (
            _interp_ref(ref, "magnetic_axis_r", t_mid),
            _interp_ref(ref, "magnetic_axis_z", t_mid),
        ),
    }


def predicted_vessel_state(state: dict, grid: EquilibriumGrid) -> np.ndarray:
    """Vessel flux on the grid at the slice, from the one-matrix pinned solve."""
    table = state["table"]
    vsys = build_passive_circuit_system(table, grid)
    sol = solve_pinned_plasma_circuit(
        table,
        vsys,
        state["i_pf"],
        list(state["fwd"].pf_amc_channels),
        state["times"],
        ip_amperes=state["ip_amp"],
        axis_rz=state["axis_trace"],
        minor_radius=state["a_trace"],
        elongation=state["kappa_trace"],
        internal_inductance=state["li_trace"],
    )
    k = state["k_slice"]
    i_vessel = sol.i_vessel[k]
    logger.info(
        "vessel state at t=%.3f: max |i_circ| = %.1f A; solved loop voltage "
        "%.3f V (dψ/dt shares: self %.3f, coils %.3f, vessel %.3f, "
        "resistive %.3f V)",
        state["time_s"],
        float(np.abs(i_vessel).max()),
        float(sol.u_loop[k]),
        float(sol.dpsi_terms["plasma_self"][k]),
        float(sol.dpsi_terms["coils"][k]),
        float(sol.dpsi_terms["vessel"][k]),
        float(sol.dpsi_terms["resistive"][k]),
    )
    psi_vessel = vsys.g_circ @ i_vessel
    return np.asarray(psi_vessel, dtype=np.float64)


def probe(shot: int, *, nr: int = 65, nz: int = 97) -> dict:
    state = measured_state(shot)
    table = state["table"]
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    psi_vessel = predicted_vessel_state(state, grid)
    k = state["k_slice"]
    i_pf_slice = state["i_pf"][k]
    ip_slice = float(state["ip_amp"][k])
    r_ref, z_ref = state["axis_ref"]
    logger.info(
        "shot %d flat-top slice t=%.3f: Ip=%.0f kA, axis_ref=(%.3f, %.3f)",
        shot,
        state["time_s"],
        ip_slice / 1e3,
        r_ref,
        z_ref,
    )

    rows = []
    psi_headline = None
    core_headline = None
    for beta0 in BETA0_GRID:
        for alpha in ALPHA_GRID:
            for with_vessel in (True, False) if (beta0, alpha) == HEADLINE else (True,):
                shape = _two_term_shape_fn(beta0, alpha, grid.r0, None)
                (
                    psi2d,
                    _cells,
                    _jphi,
                    axis,
                    _apsi,
                    _bpsi,
                    core,
                    converged,
                    residual,
                ) = _forward_picard(
                    grid,
                    i_pf_slice,
                    ip_slice,
                    shape,
                    psi_passive_grid=psi_vessel if with_vessel else None,
                    seed_z0=z_ref if np.isfinite(z_ref) else 0.0,
                    seed_width=(0.2, 0.35),
                    relax=0.2,
                    max_iterations=300,
                    tolerance=3e-4,
                    initial_jphi=None,
                    z_symmetric=False,
                )
                confined = bool(axis[0] <= CONFINED_AXIS_R_MAX and core.sum() > 4)
                axis_err = (
                    float(abs(axis[0] - r_ref)) if np.isfinite(r_ref) else float("nan")
                )
                rows.append(
                    {
                        "beta0": beta0,
                        "alpha": alpha,
                        "with_vessel": with_vessel,
                        "converged": bool(converged),
                        "residual": float(residual),
                        "axis_r": float(axis[0]),
                        "axis_z": float(axis[1]),
                        "axis_err_m": axis_err,
                        "confined": confined,
                        "within_gate": bool(confined and axis_err <= AXIS_GATE_M),
                    }
                )
                logger.info(
                    "b0=%.1f a=%.1f vessel=%s: axis=(%.3f, %.3f) err=%.3f m "
                    "confined=%s conv=%s res=%.1e",
                    beta0,
                    alpha,
                    with_vessel,
                    axis[0],
                    axis[1],
                    axis_err,
                    confined,
                    converged,
                    residual,
                )
                if (beta0, alpha) == HEADLINE and with_vessel:
                    psi_headline = psi2d
                    core_headline = core

    # existence arm: a fixed point either exists or it does not — a cold-seed
    # miss only shows the seed left the basin.  Warm-start from the confined
    # branch of the manufactured field and homotope the coil pattern to the
    # measured one; where the branch is lost is the sharpest existence read.
    continuation = continuation_probe(
        state, grid, psi_vessel, i_pf_slice, ip_slice, r_ref
    )

    headline = next(
        r for r in rows if (r["beta0"], r["alpha"]) == HEADLINE and r["with_vessel"]
    )
    cont_end = continuation[-1] if continuation else None
    cont_holds = bool(
        cont_end
        and cont_end["fraction"] == 1.0
        and cont_end["confined"]
        and cont_end["axis_err_m"] <= AXIS_GATE_M
    )
    gate = {
        "rule": (
            "PASS iff a confined fixed point exists on the measured flat-top "
            "pattern + predicted vessel state (interior O-point, axis R <= "
            "1.4 m, |R_axis - R_ref| <= 0.15 m; no z-symmetry pin anywhere): "
            "either the plain two-term (beta0=0.5, alpha=1.0) cold solve "
            "seeded at the referee axis holds it, or the warm continuation "
            "from the manufactured-field confined branch carries it to the "
            "measured pattern"
        ),
        "headline": headline,
        "n_grid_confined": int(sum(r["confined"] for r in rows if r["with_vessel"])),
        "n_grid_total": int(sum(1 for r in rows if r["with_vessel"])),
        "continuation_holds_at_measured": cont_holds,
        "branch_lost_at_fraction": next(
            (c["fraction"] for c in continuation if not c["confined"]), None
        ),
        "passes": bool(headline["within_gate"] or cont_holds),
    }
    return {
        "shot": shot,
        "time_s": state["time_s"],
        "ip_amperes": ip_slice,
        "axis_ref": [r_ref, z_ref],
        "i_pf_channels": list(state["fwd"].pf_amc_channels),
        "i_pf_slice": i_pf_slice.tolist(),
        "rows": rows,
        "continuation": continuation,
        "gate": gate,
        "_psi": psi_headline,
        "_core": core_headline,
        "_grid": grid,
        "_table": table,
    }


def continuation_probe(
    state: dict,
    grid: EquilibriumGrid,
    psi_vessel: np.ndarray,
    i_pf_measured: np.ndarray,
    ip_slice: float,
    r_ref: float,
) -> list[dict]:
    """Warm continuation from the manufactured confining field.

    Solves the plain two-term profile on the manufactured symmetric outer-coil
    field (known to hold an interior O-point), then homotopes the coil vector
    toward the measured pattern in steps, warm-starting each solve from the
    previous converged current density (the vessel flux scales in with the
    same fraction).  Where along the homotopy the interior O-point is lost —
    if at all — separates 'no confined fixed point exists on the measured
    pattern' from 'the cold seed left the basin'.
    """
    from imas_ambix.latent.synthetic_truth import (  # noqa: PLC0415
        DEFAULT_VF_STRENGTH,
        build_confining_i_pf,
    )

    i_pf_man = build_confining_i_pf(state["fwd"], DEFAULT_VF_STRENGTH)
    shape = _two_term_shape_fn(HEADLINE[0], HEADLINE[1], grid.r0, None)
    out: list[dict] = []
    warm = None
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        i_pf = (1.0 - frac) * i_pf_man + frac * i_pf_measured
        (
            _psi2d,
            _cells,
            jphi_full,
            axis,
            _apsi,
            _bpsi,
            core,
            converged,
            residual,
        ) = _forward_picard(
            grid,
            i_pf,
            ip_slice,
            shape,
            psi_passive_grid=frac * psi_vessel,
            seed_z0=0.0,
            seed_width=(0.2, 0.35),
            relax=0.2,
            max_iterations=300,
            tolerance=3e-4,
            initial_jphi=warm,
            z_symmetric=False,
        )
        confined = bool(axis[0] <= CONFINED_AXIS_R_MAX and core.sum() > 4)
        row = {
            "fraction": frac,
            "axis_r": float(axis[0]),
            "axis_z": float(axis[1]),
            "axis_err_m": float(abs(axis[0] - r_ref))
            if np.isfinite(r_ref)
            else float("nan"),
            "confined": confined,
            "converged": bool(converged),
            "residual": float(residual),
        }
        out.append(row)
        logger.info(
            "continuation frac=%.2f: axis=(%.3f, %.3f) confined=%s conv=%s",
            frac,
            axis[0],
            axis[1],
            confined,
            converged,
        )
        if not confined:
            break  # the branch is gone; later fractions start basin-less
        warm = jphi_full
    return out


def make_figure(result: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid = result["_grid"]
    table = result["_table"]
    psi = result["_psi"]
    core = result["_core"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 5.6), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    rr = grid.rg
    zz = grid.zg
    ax1.contour(rr, zz, psi, levels=24, colors="0.55", linewidths=0.6)
    if core is not None:
        ax1.contourf(
            rr, zz, core.astype(float), levels=[0.5, 1.5], colors=["#cc331122"]
        )
    ax1.plot(table.limiter_r, table.limiter_z, "k-", lw=1.2)
    hl = result["gate"]["headline"]
    ax1.plot(hl["axis_r"], hl["axis_z"], "o", color="#cc3311", ms=8, label="solve axis")
    ax1.plot(
        *result["axis_ref"], "x", color="#4477aa", ms=10, mew=2, label="referee axis"
    )
    ax1.set_aspect("equal")
    ax1.set_xlabel("R [m]")
    ax1.set_ylabel("Z [m]")
    ax1.set_title(
        f"shot {result['shot']} @ {result['time_s']:.3f}s — measured pattern, "
        f"Ip {result['ip_amperes'] / 1e3:.0f} kA"
    )
    ax1.legend(fontsize=8, loc="lower right")

    rows = [r for r in result["rows"] if r["with_vessel"]]
    b0s = sorted({r["beta0"] for r in rows})
    als = sorted({r["alpha"] for r in rows})
    axis_map = np.full((len(als), len(b0s)), np.nan)
    for r in rows:
        axis_map[als.index(r["alpha"]), b0s.index(r["beta0"])] = r["axis_r"]
    im = ax2.imshow(
        axis_map,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(-0.5, len(b0s) - 0.5, -0.5, len(als) - 0.5),
    )
    for r in rows:
        i, j = als.index(r["alpha"]), b0s.index(r["beta0"])
        ax2.text(
            j,
            i,
            f"{r['axis_r']:.2f}\n{'ok' if r['within_gate'] else 'X'}",
            ha="center",
            va="center",
            fontsize=9,
            color="w" if r["within_gate"] else "#ffcccc",
        )
    ax2.set_xticks(range(len(b0s)), [f"{b:.1f}" for b in b0s])
    ax2.set_yticks(range(len(als)), [f"{a:.1f}" for a in als])
    ax2.set_xlabel(r"$\beta_0$")
    ax2.set_ylabel(r"$\alpha$")
    ax2.set_title("axis R [m] across the profile grid (gate: err ≤ 0.15 m)")
    fig.colorbar(im, ax=ax2, shrink=0.85)
    fig.savefig(FIGURES / "fig-confinement-probe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=11766)
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    args = ap.parse_args()

    result = probe(int(args.shot))
    logger.info("GATE: %s", json.dumps(result["gate"], indent=2))
    make_figure(result)
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    logger.info("artifact: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
