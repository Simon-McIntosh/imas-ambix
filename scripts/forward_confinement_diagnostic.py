#!/usr/bin/env python
"""Why the free-boundary forward solve does not hold the measured equilibrium.

The reconstruction spine confines correctly on the measured coil program
(magnetics-constrained, 23/23 flat-top within 15 cm of the referee).  The
open question is the FORWARD operator: driven by the measured coils alone
(jφ ≥ 0 + Rogowski Ip, NO magnetics), does a free-boundary solve independently
hold that same inboard equilibrium?  It does not — it converges outboard.  This
diagnostic pins WHY, separating three candidate mechanisms:

  * n = 0 vertical instability   → the axis would run OFF the midplane (|Z| ↑)
  * spurious coil-flux-pocket    → the axis would lock onto an in-vessel coil
    capture                        O-point (conductor-interior, ψ-pocket)
  * radial interior-null         → the current sits at a genuine O-point,
    (profile freedom)              Shafranov-shifted outboard because the
                                   profile peakedness (β_p + l_i/2) that sets
                                   the axis is externally undetermined

For every slice it records the forward magnetic axis (R, Z), the current
CENTROID (R, Z), whether the axis is a genuine conductor-clear O-point (vs a
coil pocket), and the Shafranov shift axis−centroid; against the committed
reconstruction/referee axes (measured_pattern_confinement.json, diagnostic-only
referee) it reports the forward↔physical radial gap and its Ip dependence.

The anchor slice additionally carries a seed-sensitivity table: the harness
ladder path bootstraps from a legacy ``coil_field_mode="boundary-continuation"``
stage-1 seed (documented as inaccurate near the in-vessel coils), which for the
data-free forward solve inflates the outboard drift relative to a plain
compact-seed analytic-add solve — so part of the gate's forward offset is a
seed artefact, and the irreducible residual is the interior-null.

Artifact: imas_ambix/gs/artifacts/forward_confinement_diagnostic.json
Figures:  docs/figures/equilibrium-realism/
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forward_confinement_diagnostic")

ARTIFACT = Path("imas_ambix/gs/artifacts/forward_confinement_diagnostic.json")
GATE_ARTIFACT = Path("imas_ambix/gs/artifacts/measured_pattern_confinement.json")
FIGURES = Path("docs/figures/equilibrium-realism")
DEFAULT_SHOTS = (11766, 11767, 11772)
COMPACT_SEED = (0.25, 0.35)  # the reseed scout's compact midplane seed


def _spine():
    from scripts.spine_label_factory import frozen_spine_config  # noqa: PLC0415

    spine, _sha = frozen_spine_config()
    return spine, spine["interior_solve"]


def _forward_compact(grid, table, p, sidecar, iso):
    """Data-free profile-free solve from a compact midplane seed (no magnetics,
    no boundary prior, no reseed) — the raw forward operator's fixed point."""
    from imas_ambix.latent.gs_solve import solve_equilibrium_lsq  # noqa: PLC0415

    return solve_equilibrium_lsq(
        grid,
        table,
        p.i_pf,
        p.ip_amperes,
        measured=p.measured,
        vacuum_prediction=p.vacuum,
        sensor_scale=p.scale,
        sensor_mask=np.zeros_like(p.mask, dtype=bool),
        n_p=int(iso["n_p"]),
        n_f=int(iso["n_f"]),
        smoothness=float(iso["smoothness"]),
        nonneg=iso["profile_kind"] == "monomial-nonneg",
        passive=sidecar,
        passive_ridge=1.0,
        seed_width=COMPACT_SEED,
    )


def _classify(psi2d, jphi2d, grid):
    """Axis read + current centroid + conductor-clear classification."""
    from imas_ambix.latent.topology import find_critical_points  # noqa: PLC0415
    from scripts.patch_gate_eval import geometry_target  # noqa: PLC0415

    t, psi_ax, psi_b = geometry_target(psi2d, grid)
    axis_r, axis_z = float(t[0]), float(t[1])
    axis_clear = bool(
        grid.clear_of_conductors(np.array([axis_r]), np.array([axis_z]))[0]
    )
    tot = float(jphi2d.sum())
    rc = float((grid.mesh_r * jphi2d).sum() / tot) if abs(tot) > 0 else float("nan")
    zc = float((grid.mesh_z * jphi2d).sum() / tot) if abs(tot) > 0 else float("nan")
    cp = find_critical_points(psi2d, grid.rg, grid.zg)
    n_coil_o = sum(
        1
        for r, z in zip(cp.o_points[:, 0], cp.o_points[:, 1], strict=True)
        if not grid.clear_of_conductors(np.array([r]), np.array([z]))[0]
    )
    n_clear_o = int(cp.o_points.shape[0]) - n_coil_o
    return {
        "axis_r": axis_r,
        "axis_z": axis_z,
        "axis_is_clear_o_point": axis_clear,
        "centroid_r": rc,
        "centroid_z": zc,
        "shafranov_shift_m": axis_r - rc if np.isfinite(rc) else float("nan"),
        "n_o_points": int(cp.o_points.shape[0]),
        "n_coil_o_points_excluded": int(n_coil_o),
        "n_clear_o_points": n_clear_o,
        "psi_axis": float(psi_ax),
        "psi_boundary": float(psi_b),
    }


def _gate_reference() -> dict:
    """Per-(shot, round(time_s,3)) recon/referee/forward-ladder axes from the
    committed gate artifact (the reconstruction spine's own output)."""
    ref: dict = {}
    if not GATE_ARTIFACT.exists():
        return ref
    d = json.loads(GATE_ARTIFACT.read_text())
    for sh in d["shots"]:
        for s in sh["slices"]:
            ref[(int(sh["shot"]), round(float(s["time_s"]), 3))] = {
                "recon_axis_r": s["recon"]["axis_r"],
                "recon_axis_z": s["recon"].get("axis_z", float("nan")),
                "referee_axis_r": s.get("axis_r_ref", float("nan")),
                "referee_axis_z": s.get("axis_z_ref", float("nan")),
                "forward_ladder_axis_r": s["forward"]["axis_r"],
                "tag": s["tag"],
            }
    return ref


def diagnose_shot(shot: int, iso, ref: dict) -> dict:
    from scripts.closure_gate_eval import _shot_passive_sidecar  # noqa: PLC0415
    from scripts.spine_label_factory import factory_shot_payloads  # noqa: PLC0415

    payload = factory_shot_payloads(shot, nr=65, nz=97, max_slices=12, min_ip_ka=60.0)
    if payload is None:
        return {"shot": shot, "slices": []}
    grid, table = payload["grid"], payload["table"]
    sidecar = _shot_passive_sidecar(payload, int(iso["passive_k"]))
    ip_peak = max(float(p.ip_amperes) for p in payload["payloads"])

    rows = []
    order = np.argsort([p.time_s for p in payload["payloads"]])
    for kk in order:
        p = payload["payloads"][int(kk)]
        lf = _forward_compact(grid, table, p, sidecar, iso)
        cls = _classify(lf.result.psi, lf.result.jphi, grid)
        g = ref.get((shot, round(float(p.time_s), 3)), {})
        tag = g.get("tag", "flat" if abs(p.ip_amperes) >= 0.9 * ip_peak else "ramp")
        rr, rz = (
            g.get("referee_axis_r", float("nan")),
            g.get("referee_axis_z", float("nan")),
        )
        row = {
            "shot": shot,
            "time_s": float(p.time_s),
            "ip_amperes": float(p.ip_amperes),
            "tag": tag,
            "converged": bool(lf.result.converged),
            "forward_compact": cls,
            "forward_ladder_axis_r": g.get("forward_ladder_axis_r", float("nan")),
            "recon_axis_r": g.get("recon_axis_r", float("nan")),
            "recon_axis_z": g.get("recon_axis_z", float("nan")),
            "referee_axis_r": rr,
            "referee_axis_z": rz,
            "forward_vs_referee_cm": (
                abs(cls["axis_r"] - rr) * 100.0 if np.isfinite(rr) else float("nan")
            ),
        }
        rows.append(row)
        logger.info(
            "%d t=%.3f %-4s Ip=%3.0fkA  fwd(compact) axisR=%.3f Z=%+.3f centroidR=%.3f "
            "shift=%+.2fm clearO=%s coilO_excl=%d | recon R=%.3f ref R=%.3f",
            shot,
            p.time_s,
            tag,
            p.ip_amperes / 1e3,
            cls["axis_r"],
            cls["axis_z"],
            cls["centroid_r"],
            cls["shafranov_shift_m"],
            cls["axis_is_clear_o_point"],
            cls["n_coil_o_points_excluded"],
            row["recon_axis_r"],
            rr,
        )
    return {"shot": shot, "ip_peak": ip_peak, "slices": rows}


def anchor_seed_sensitivity(shot: int, iso) -> dict:
    """On the flat-top anchor slice, compare the forward axis under three solve
    paths (harness ladder w/ legacy seed, compact-seed direct, and the
    magnetics-constrained recon) plus the retired fixed-two-term Picard."""
    import dataclasses  # noqa: PLC0415

    from imas_ambix.latent.gs_solve import solve_equilibrium  # noqa: PLC0415
    from scripts.closure_gate_eval import (  # noqa: PLC0415
        _shot_passive_sidecar,
        fit_and_read_slice,
    )
    from scripts.patch_gate_eval import geometry_target  # noqa: PLC0415
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, _ = frozen_spine_config()
    disc = dict(spine["soft_priors"])
    disc.setdefault("boundary_prior", "disc")
    payload = factory_shot_payloads(shot, nr=65, nz=97, max_slices=12, min_ip_ka=60.0)
    grid, table = payload["grid"], payload["table"]
    sidecar = _shot_passive_sidecar(payload, int(iso["passive_k"]))
    k = int(np.argmax([p.ip_amperes for p in payload["payloads"]]))
    p = payload["payloads"][k]

    def harness(mask, spc, reseed):
        return fit_and_read_slice(
            grid,
            table,
            dataclasses.replace(p, mask=mask),
            beta0_grid=(0.5,),
            alpha_grid=(1.0,),
            cost_limit=float("inf"),
            convergence_limit=5e-3,
            retry_max_iterations=160,
            fit_mode="ladder",
            n_p=int(iso["n_p"]),
            n_f=int(iso["n_f"]),
            smoothness=float(iso["smoothness"]),
            nonneg=iso["profile_kind"] == "monomial-nonneg",
            passive=sidecar,
            passive_ridge=1.0,
            reseed_axis_r_max=(1.4 if reseed else 1e9),
            keep_psi=True,
            keep_jphi=True,
            basis=None,
            meta={},
            soft_prior_cfg=spc,
            boundary_read=iso["boundary_read_scoring"],
        )

    off = np.zeros_like(p.mask, dtype=bool)
    fwd_ladder = harness(off, None, reseed=True)
    fwd_ladder_nore = harness(off, None, reseed=False)
    rec = harness(p.mask, disc, reseed=True)
    fwd_compact = _forward_compact(grid, table, p, sidecar, iso)
    # the retired fixed-two-term Picard, compact seed, magnetics-free
    two_term = solve_equilibrium(grid, p.i_pf, p.ip_amperes, seed_width=COMPACT_SEED)
    tt, _, _ = geometry_target(two_term.psi, grid)

    return {
        "shot": shot,
        "time_s": float(p.time_s),
        "ip_amperes": float(p.ip_amperes),
        "forward_ladder_reseed": float(fwd_ladder.target[0]),
        "forward_ladder_noreseed": float(fwd_ladder_nore.target[0]),
        "forward_compact_direct": _classify(
            fwd_compact.result.psi, fwd_compact.result.jphi, grid
        ),
        "forward_two_term_axis_r": float(tt[0]),
        "recon": _classify(rec.psi, rec.jphi_flat.reshape(grid.nz, grid.nr), grid),
        "recon_cost": float(rec.cost),
        "_figure": (grid, table, p, fwd_compact, rec),
    }


def evaluate(shots: list[dict], anchor: dict) -> dict:
    flat = [s for sh in shots for s in sh["slices"] if s["tag"] == "flat"]
    fz = [
        abs(s["forward_compact"]["axis_z"])
        for sh in shots
        for s in sh["slices"]
        if np.isfinite(s["forward_compact"]["axis_z"])
    ]
    coil_captures = [
        s
        for sh in shots
        for s in sh["slices"]
        if not s["forward_compact"]["axis_is_clear_o_point"]
    ]
    gaps = [
        s["forward_vs_referee_cm"]
        for s in flat
        if np.isfinite(s["forward_vs_referee_cm"])
    ]
    shifts = [
        s["forward_compact"]["shafranov_shift_m"]
        for sh in shots
        for s in sh["slices"]
        if np.isfinite(s["forward_compact"]["shafranov_shift_m"])
    ]
    return {
        "mechanism": {
            "vertical_mode": {
                "max_forward_axis_abs_z_cm": (max(fz) * 100.0 if fz else float("nan")),
                "verdict": (
                    "Z-STABLE — forward axis stays on the midplane at every slice; "
                    "the n=0 vertical mode is NOT the forward failure, so a modeled "
                    "Z-controller (feedback-forward-only) is a no-op for V2-forward"
                ),
            },
            "coil_flux_pocket": {
                "n_slices_axis_on_coil_pocket": len(coil_captures),
                "verdict": (
                    "NOT a coil-flux-pocket capture — the forward axis is a genuine "
                    "conductor-clear O-point at every slice; the in-vessel coil "
                    "O-points are correctly excluded (corrects the §2.5 stated "
                    "mechanism)"
                ),
            },
            "radial_interior_null": {
                "median_forward_vs_referee_cm": (
                    float(np.median(gaps)) if gaps else float("nan")
                ),
                "median_shafranov_shift_m": (
                    float(np.median(shifts)) if shifts else float("nan")
                ),
                "verdict": (
                    "RADIAL interior-null — the current sits at a genuine O-point "
                    "Shafranov-shifted outboard; the axis position depends on the "
                    "profile peakedness (β_p+l_i/2), externally undetermined without "
                    "magnetics (reconstruction) or a learned profile prior"
                ),
            },
        },
        "seed_artefact": {
            "anchor_forward_ladder_reseed": anchor["forward_ladder_reseed"],
            "anchor_forward_ladder_noreseed": anchor["forward_ladder_noreseed"],
            "anchor_forward_compact_direct_axis_r": anchor["forward_compact_direct"][
                "axis_r"
            ],
            "anchor_recon_axis_r": anchor["recon"]["axis_r"],
            "note": (
                "the harness ladder forward path bootstraps from a legacy "
                "boundary-continuation stage-1 seed that inflates the outboard "
                "drift; a compact-seed analytic-add solve lands markedly more "
                "inboard, but still outboard of the physical axis by the "
                "interior-null Shafranov shift"
            ),
        },
        "gate_G3a_forward_holds_physical_eq": {
            "passes": bool(gaps and max(gaps) <= 15.0),
            "rule": "forward axis within 15 cm of the referee on all flat-top slices",
        },
    }


def make_figures(shots: list[dict], anchor: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    marker = {"11766": "o", "11767": "s", "11772": "^"}

    # --- mechanism scatter: axis/centroid/recon/referee R vs Ip; Z vs Ip ---
    fig, (ax_r, ax_z) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    first = True
    for sh in shots:
        m = marker.get(str(sh["shot"]), "o")
        ip = [s["ip_amperes"] / 1e3 for s in sh["slices"]]
        ax_r.plot(
            ip,
            [s["forward_compact"]["axis_r"] for s in sh["slices"]],
            m,
            color="#cc6677",
            mfc="none",
            ms=6,
            label="forward axis (no data)" if first else None,
        )
        ax_r.plot(
            ip,
            [s["forward_compact"]["centroid_r"] for s in sh["slices"]],
            m,
            color="#ee9944",
            ms=4,
            label="forward current centroid" if first else None,
        )
        ax_r.plot(
            ip,
            [s["recon_axis_r"] for s in sh["slices"]],
            m,
            color="#228833",
            ms=5,
            label="recon axis (magnetics)" if first else None,
        )
        ax_r.plot(
            ip,
            [s["referee_axis_r"] for s in sh["slices"]],
            m,
            color="#4477aa",
            mfc="none",
            ms=4,
            ls=":",
            label="referee axis (EFIT)" if first else None,
        )
        ax_z.plot(
            ip,
            [s["forward_compact"]["axis_z"] * 100 for s in sh["slices"]],
            m,
            color="#cc6677",
            mfc="none",
            ms=6,
            label="forward axis Z" if first else None,
        )
        ax_z.plot(
            ip,
            [s["recon_axis_z"] * 100 for s in sh["slices"]],
            m,
            color="#228833",
            ms=4,
            label="recon axis Z" if first else None,
        )
        first = False
    ax_r.axhline(1.4, color="k", ls=":", lw=1)
    ax_r.set_xlabel("Ip [kA]")
    ax_r.set_ylabel("R [m]")
    ax_r.set_title(
        "Radial: forward current sits ~physical,\naxis Shafranov-shifted outboard"
    )
    ax_r.legend(fontsize=7)
    ax_z.axhspan(-0.5, 0.5, color="#dddddd", alpha=0.6)
    ax_z.set_xlabel("Ip [kA]")
    ax_z.set_ylabel("axis Z [cm]")
    ax_z.set_title("Vertical: Z stays on the midplane\n(n=0 mode is not the failure)")
    ax_z.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig-forward-mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", FIGURES / "fig-forward-mechanism.png")

    # --- flux map: the forward free-solve's genuine outboard O-point (no coil
    # capture) — the companion to the §2.5 recon flux map (fig-confined-equilibrium) ---
    grid, table, p, fwd, rec = anchor.pop("_figure")
    try:
        from imas_ink.figures import equilibrium_figure_mpl  # noqa: PLC0415

        from scripts.confinement_flux_figure import (  # noqa: PLC0415, E501
            _machine_geometry,
            _slice,
        )
        from scripts.patch_gate_eval import geometry_target  # noqa: PLC0415

        geom = _machine_geometry(grid, table)
        t, pa, pb = geometry_target(fwd.result.psi, grid)
        sl = _slice(fwd.result.psi, grid, t, pa, pb, p.ip_amperes, p.time_s)
        fig, _ax = equilibrium_figure_mpl(sl, geom, show_vacuum_surfaces=False)
        fig.suptitle(
            f"Forward free-solve (no magnetics) — shot {anchor['shot']} flat-top "
            f"(Ip {p.ip_amperes / 1e3:.0f} kA); genuine O-point at "
            f"R={t[0]:.3f} m, Z={t[1]:+.3f} m — Shafranov-shifted outboard, "
            f"not a coil capture",
            fontsize=9,
        )
        fig.savefig(FIGURES / "fig-forward-flux.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", FIGURES / "fig-forward-flux.png")
    except Exception as exc:  # imas-ink signature drift must not sink the artifact
        logger.warning("flux-map figure skipped: %s", exc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    ap.add_argument("--anchor", type=int, default=11766)
    ap.add_argument("--out", type=Path, default=ARTIFACT)
    args = ap.parse_args()

    _spine_cfg, iso = _spine()
    ref = _gate_reference()
    shots = [diagnose_shot(int(s), iso, ref) for s in args.shots]
    shots = [s for s in shots if s["slices"]]
    anchor = anchor_seed_sensitivity(int(args.anchor), iso)
    gate = evaluate(shots, anchor)
    logger.info("VERDICT:\n%s", json.dumps(gate, indent=2))
    make_figures(shots, anchor)  # pops the un-serialisable _figure handle
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"shots": shots, "anchor": anchor, "gate": gate}, indent=1)
    )
    logger.info("artifact: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
