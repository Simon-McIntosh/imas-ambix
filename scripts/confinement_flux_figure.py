#!/usr/bin/env python
"""imas-ink flux-surface map of the profile-free confined equilibrium.

Renders the magnetics-matched recon equilibrium the regularized profile-free
solve holds on shot 11766's measured flat-top coil program: a centred,
up/down-symmetric diverted plasma at the measured axis (R~0.82, near the
real ~0.9).  This is the V2 result — NOT the pure-forward free-solve, which
without the magnetics constraint drifts to the interior-null (outboard) and
is a diagnostic only (see scripts/ip_sign_divertor_audit.py for the
forward-vs-recon-vs-Ip-flip comparison that establishes this).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("confinement_flux_figure")

FIGURES = Path("docs/figures/equilibrium-realism")


def _machine_geometry(grid, table):
    from imas_ink._types import CoilRect, MachineGeometry  # noqa: PLC0415

    lr = np.asarray(grid.limiter_r, dtype=np.float64)
    lz = np.asarray(grid.limiter_z, dtype=np.float64)
    clip = np.column_stack([np.append(lr, lr[0]), np.append(lz, lz[0])])
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


def _slice(psi2d, grid, target, psi_ax, psi_b, ip, time_s):
    from imas_ink._types import EquilibriumSlice  # noqa: PLC0415

    return EquilibriumSlice(
        psi_2d=np.ascontiguousarray(psi2d.T),
        r_grid=np.asarray(grid.rg, dtype=np.float64),
        z_grid=np.asarray(grid.zg, dtype=np.float64),
        psi_axis=float(psi_ax),
        psi_boundary=float(psi_b),
        r_axis=float(target[0]),
        z_axis=float(target[1]),
        ip=float(ip),
        time=float(time_s),
        converged=True,
    )


def _solve_anchor(shot: int):
    from scripts.closure_gate_eval import (  # noqa: PLC0415, E501
        _shot_passive_sidecar,
        fit_and_read_slice,
    )
    from scripts.spine_label_factory import (  # noqa: PLC0415
        factory_shot_payloads,
        frozen_spine_config,
    )

    spine, _ = frozen_spine_config()
    isolve = spine["interior_solve"]
    disc_cfg = dict(spine["soft_priors"])
    disc_cfg.setdefault("boundary_prior", "disc")
    pl = factory_shot_payloads(shot, nr=65, nz=97, max_slices=12, min_ip_ka=60.0)
    grid, table = pl["grid"], pl["table"]
    sidecar = _shot_passive_sidecar(pl, int(isolve["passive_k"]))
    k = int(np.argmax([p.ip_amperes for p in pl["payloads"]]))
    p = pl["payloads"][k]

    def run(mask, spc):
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
            n_p=int(isolve["n_p"]),
            n_f=int(isolve["n_f"]),
            smoothness=float(isolve["smoothness"]),
            nonneg=isolve["profile_kind"] == "monomial-nonneg",
            passive=sidecar,
            passive_ridge=1.0,
            reseed_axis_r_max=1.4,
            keep_psi=True,
            keep_jphi=True,
            basis=None,
            meta={},
            soft_prior_cfg=spc,
            boundary_read=isolve["boundary_read_scoring"],
        )

    rec = run(p.mask, disc_cfg)
    fwd = run(np.zeros_like(p.mask, dtype=bool), None)
    return grid, table, p, rec, fwd


def main() -> int:
    from imas_ink.figures import equilibrium_figure_mpl  # noqa: PLC0415

    from scripts.patch_gate_eval import geometry_target  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=11766)
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    grid, table, p, rec, fwd = _solve_anchor(args.shot)
    geom = _machine_geometry(grid, table)
    logger.info(
        "recon axis=(%.3f,%.3f)  forward axis=(%.3f,%.3f)  Ip=%.0f kA",
        rec.target[0],
        rec.target[1],
        fwd.target[0],
        fwd.target[1],
        p.ip_amperes / 1e3,
    )

    t_rec, pa_rec, pb_rec = geometry_target(rec.psi, grid)
    rec_sl = _slice(rec.psi, grid, t_rec, pa_rec, pb_rec, p.ip_amperes, p.time_s)

    # the magnetics-matched recon equilibrium — the V2 result: a centred,
    # up/down-symmetric diverted plasma at the measured axis.  The pure-forward
    # free-solve is deliberately NOT shown here (it drifts to the interior-null
    # outboard and would misrepresent the result); it lives in the sign-audit
    # figure as a labelled diagnostic.
    fig, _ax = equilibrium_figure_mpl(rec_sl, geom, show_vacuum_surfaces=False)
    fig.suptitle(
        f"Profile-free recon equilibrium — shot {args.shot} flat-top "
        f"(Ip {p.ip_amperes / 1e3:.0f} kA); axis R={t_rec[0]:.3f} m, "
        f"up/down-symmetric",
        fontsize=10,
    )
    fig.savefig(FIGURES / "fig-confined-equilibrium.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", FIGURES / "fig-confined-equilibrium.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
