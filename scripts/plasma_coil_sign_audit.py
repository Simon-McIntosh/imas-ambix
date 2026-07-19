"""Decisive test of the plasma-vs-coil sign in the GS solver.

Does the solver confine a +Ip plasma with an OPPOSITE-sign vertical-field coil
pair inboard (correct), or push it onto the coils (same-sign attraction = a
sign/COCOS bug)?  Anchors the interpretation of the 11766 forward O-point at
R=1.51.
"""

import numpy as np

from imas_ambix.gs.geometry import build_table_for_shot
from imas_ambix.gs.operator import build_operator
from imas_ambix.latent.gs_solve import EquilibriumGrid, solve_equilibrium_lsq
from imas_ambix.latent.synthetic_truth import build_confining_i_pf

SHOT = 11766
table = build_table_for_shot(SHOT)
fwd = build_operator(table)
grid = EquilibriumGrid.from_table(table, nr=65, nz=97)
S = len(fwd.sensor_channels)
scale = np.ones(S)
meas = np.zeros(S)
vac = np.zeros(S)
mask_off = np.zeros(S, dtype=bool)  # forward: no magnetics

IP = 4.66e5  # +Ip, positive


def fwd_axis(i_pf, ip):
    lf = solve_equilibrium_lsq(
        grid,
        table,
        i_pf,
        ip,
        measured=meas,
        vacuum_prediction=vac,
        sensor_scale=scale,
        sensor_mask=mask_off,
        n_p=3,
        n_f=3,
        smoothness=0.01,
        nonneg=True,
        max_iterations=160,
        relax=0.5,
        tolerance=5e-3,
    )
    r = lf.result
    core = int(np.sum(r.core_mask))
    return float(r.axis[0]), float(r.axis[1]), core, bool(r.converged)


# --- solver convention: +Ip plasma alone, is the axis a flux MAX? ---
# (drive a compact plasma with zero coils, read the axis sign)
lf0 = solve_equilibrium_lsq(
    grid,
    table,
    np.zeros(len(fwd.pf_amc_channels)),
    IP,
    measured=meas,
    vacuum_prediction=vac,
    sensor_scale=scale,
    sensor_mask=mask_off,
    n_p=3,
    n_f=3,
    smoothness=0.01,
    nonneg=True,
    max_iterations=120,
    relax=0.5,
    tolerance=5e-3,
)
psi = lf0.result.psi
ax = lf0.result.axis
ia = int(np.argmin(np.abs(grid.zg - ax[1])))
ja = int(np.argmin(np.abs(grid.rg - ax[0])))
psi_ax = psi[ia, ja]
psi_med = float(np.median(psi))
print("=== solver convention (+Ip, no coils) ===")
print(
    f"  axis=({ax[0]:.3f},{ax[1]:.3f})  psi_axis={psi_ax:.4g}  psi_median={psi_med:.4g}"
)
print(
    f"  axis is a flux {'MAX' if psi_ax > psi_med else 'MIN'} for +Ip "
    f"(COCOS sigma_Bp {'-1 (like 3/13/17)' if psi_ax > psi_med else '+1 (like 1/11)'})"
)

# --- clean confinement: OPPOSITE-sign vs SAME-sign outer-coil VF pair ---
i_conf = build_confining_i_pf(fwd, 6.0e4)  # P4/P5/P6 at -60 kA (opposite +Ip)
print("\n=== clean VF-pair confinement (forward, no magnetics) ===")
r_opp = fwd_axis(i_conf, IP)
print(
    f"  OPPOSITE-sign coils (-60kA), +Ip: axis R={r_opp[0]:.3f} Z={r_opp[1]:.3f} "
    f"core={r_opp[2]} conv={r_opp[3]}  -> {'CONFINED INBOARD' if r_opp[0] < 1.1 else 'RAN OUT'}"
)
r_same = fwd_axis(-i_conf, IP)  # flip: +60 kA, SAME sign as +Ip
print(
    f"  SAME-sign coils (+60kA), +Ip:     axis R={r_same[0]:.3f} Z={r_same[1]:.3f} "
    f"core={r_same[2]} conv={r_same[3]}  -> {'pulled to coil' if r_same[0] > 1.3 else 'inboard'}"
)

# --- 11766 measured pattern: vacuum psi_coil extrema (is there an LFS max?) ---
schema_ok = True
try:
    from imas_ambix.latent.data import (
        anchored_columns,
        feature_schema,
        load_shot_slices_raw,
        schema_group_offsets,
    )

    schema = feature_schema()
    x, times, plasma_on = load_shot_slices_raw(SHOT, schema)
    off = schema_group_offsets(schema)
    amc_names = schema["amc"]
    ip_col, _ = anchored_columns(schema)
    amc_block = x[:, off["amc"] : off["amc"] + len(amc_names)]
    ip_ka = np.where(np.isfinite(x[:, ip_col]), x[:, ip_col], 0.0)
    kf = int(np.argmax(np.abs(ip_ka)))
    vals = {
        ch: float(amc_block[kf, j])
        for j, ch in enumerate(amc_names)
        if np.isfinite(amc_block[kf, j])
    }
    i_pf_11766 = fwd.assemble_pf_currents(vals)
except Exception as e:  # noqa: BLE001
    schema_ok = False
    print("11766 load failed:", e)

if schema_ok:
    psi_coil = grid.coil_psi(i_pf_11766).reshape(grid.nz, grid.nr)
    # restrict to inside limiter for the extremum search
    from imas_ambix.gs.operator import _inside_polygon

    inside = _inside_polygon(
        grid.flat_r, grid.flat_z, np.asarray(grid.limiter_r), np.asarray(grid.limiter_z)
    )
    pc = np.where(inside.reshape(grid.nz, grid.nr), psi_coil, np.nan)
    kmax = np.unravel_index(np.nanargmax(pc), pc.shape)
    kmin = np.unravel_index(np.nanargmin(pc), pc.shape)
    print("\n=== 11766 measured pattern: vacuum coil psi structure (in-limiter) ===")
    print(
        f"  psi_coil MAX at R={grid.rg[kmax[1]]:.3f} Z={grid.zg[kmax[0]]:.3f} "
        f"= {pc[kmax]:.4g} Wb"
    )
    print(
        f"  psi_coil MIN at R={grid.rg[kmin[1]]:.3f} Z={grid.zg[kmin[0]]:.3f} "
        f"= {pc[kmin]:.4g} Wb"
    )
    print(
        "  (a +Ip plasma seeks a psi MAX; if the in-limiter coil-psi MAX is "
        "outboard, the free profile is pulled there)"
    )
    r11 = fwd_axis(i_pf_11766, abs(float(ip_ka[kf])) * 1e3)
    print(f"  11766 forward axis R={r11[0]:.3f} Z={r11[1]:.3f} core={r11[2]}")
    # which coils are same-sign as +Ip?
    print("  coil signs (measured, + = same as +Ip):")
    for j, ch in enumerate(fwd.pf_amc_channels):
        if (
            ch.startswith(("p4", "p5", "p6"))
            and "case" not in ch
            and abs(i_pf_11766[j]) > 1e3
        ):
            print(
                f"    {ch:20s} {i_pf_11766[j] / 1e3:+7.1f} kA  "
                f"{'SAME as Ip' if i_pf_11766[j] > 0 else 'opposite'}"
            )
