#!/usr/bin/env python
"""Generate the TORAX reference case for the current-diffusion cross-check.

RUNS UNDER THE TORAX ENVIRONMENT (``uv run --project ~/Code/torax``), never
inside the imas-ambix pipeline (JAX stays out).  Pure ohmic current
diffusion on the built-in circular geometry: heat/particle evolution OFF,
prescribed temperature/density (fixing the Sauter conductivity), no
external current sources, bootstrap off, a ramping prescribed Ip — exactly
the equation the in-house torch operator implements.

The output NPZ carries TORAX's own geometry metrics (g2·g3/ρ̂, vpr, F,
Φ_b), the parallel conductivity profile, the Ip trace, and ψ(t, ρ̂) — the
imas-ambix test ``test_current_diffusion.py::test_torax_crosscheck`` builds
a :class:`FluxSurfaceGeometry` from these arrays, evolves the same initial
ψ with the same σ and Ip through ``diffuse_psi``, and pins the final state.

Usage:
    cd ~/Code/torax && uv run python \\
        ~/Code/imas-ambix/scripts/torax_crosscheck_reference.py \\
        --out ~/Code/imas-ambix/tests/data/torax_circular_psi_reference.npz
"""

from __future__ import annotations

import argparse

import numpy as np

CONFIG = {
    "plasma_composition": {},
    "profile_conditions": {
        "Ip": {0.0: 1.0e6, 0.3: 1.4e6},  # ramping total current [A]
        "T_i": {0.0: {0.0: 2.0, 1.0: 0.2}},
        "T_i_right_bc": 0.2,
        "T_e": {0.0: {0.0: 2.0, 1.0: 0.2}},
        "T_e_right_bc": 0.2,
        "n_e": {0: {0.0: 1.2e20, 1.0: 0.6e20}},
        "n_e_right_bc": 0.6e20,
        "n_e_nbar_is_fGW": False,
        "n_e_right_bc_is_fGW": False,
        "normalize_n_e_to_nbar": False,
    },
    "numerics": {
        "t_final": 0.3,
        "fixed_dt": 0.002,
        "evolve_ion_heat": False,
        "evolve_electron_heat": False,
        "evolve_current": True,
        "evolve_density": False,
    },
    "geometry": {
        "geometry_type": "circular",
        "R_major": 1.65,
        "a_minor": 0.5,
        "B_0": 2.0,
        "n_rho": 48,
    },
    "neoclassical": {
        "bootstrap_current": {"bootstrap_multiplier": 0.0},
    },
    "sources": {},  # pure ohmic — no external current drive
    "pedestal": {},
    "transport": {"model_name": "constant"},
    "solver": {"solver_type": "linear"},
    "time_step_calculator": {"calculator_type": "fixed"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    import torax

    cfg = torax.ToraxConfig.from_dict(CONFIG)
    data_tree, history = torax.run_simulation(cfg, progress_bar=False)

    prof = data_tree["profiles"]
    times = np.asarray(data_tree["time"].values, dtype=np.float64)
    psi = np.asarray(prof["psi"].values, dtype=np.float64)  # (t, rho_cell)
    sigma = np.asarray(prof["sigma_parallel"].values, dtype=np.float64)
    ip = np.asarray(prof["Ip_profile"].values, dtype=np.float64)[:, -1]

    geo = history.geometries[0]  # static for the circular case
    take = {
        "rho_cell_norm": np.asarray(data_tree["rho_cell_norm"].values),
        "rho_face_norm": np.asarray(data_tree["rho_face_norm"].values),
        "g2_face": np.asarray(geo.g2_face),
        "g3_face": np.asarray(geo.g3_face),
        "g2g3_over_rhon_face": np.asarray(geo.g2g3_over_rhon_face),
        "vpr_face": np.asarray(geo.vpr_face),
        "F_face": np.asarray(geo.F_face),
        "Phi_b": float(np.asarray(geo.Phi_b)),
        "R_major": float(CONFIG["geometry"]["R_major"]),
        "B_0": float(CONFIG["geometry"]["B_0"]),
        "a_minor": float(CONFIG["geometry"]["a_minor"]),
    }
    np.savez_compressed(
        args.out,
        times=times,
        psi=psi,
        sigma_parallel=sigma,
        ip=ip,
        **take,
    )
    print(
        f"wrote {args.out}: {times.size} steps, psi {psi.shape}, "
        f"Ip {ip[0]:.3g}->{ip[-1]:.3g} A, Phi_b={take['Phi_b']:.4g} Wb"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
