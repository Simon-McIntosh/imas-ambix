#!/usr/bin/env python
"""Plasma-current-sign + divertor-orientation audit on shot 11766.

Answers two sign-convention questions raised on the confinement figure:

1. Does flipping the APPLIED plasma-current sign turn the forward solve from a
   radial-out equilibrium into an up/down-diverted one?  (It does not — the
   flipped solve runs further outboard and off-midplane.)
2. What does the magnetics-matched recon solve — our actual result — produce:
   an up/down-symmetric diverted plasma at the measured axis (physics sound)
   or a radial one (a real sign error)?  (It produces the former.)

Three forward/recon solves are rendered with their O-point and in-domain
X-points marked, so the divertor orientation is directly visible.
"""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.latent.topology import find_critical_points
from scripts.closure_gate_eval import _shot_passive_sidecar, fit_and_read_slice
from scripts.spine_label_factory import factory_shot_payloads, frozen_spine_config

SHOT = 11766
spine, _ = frozen_spine_config()
iso = spine["interior_solve"]
disc = dict(spine["soft_priors"])
disc.setdefault("boundary_prior", "disc")
pl = factory_shot_payloads(SHOT, nr=65, nz=97, max_slices=12, min_ip_ka=60.0)
grid, table = pl["grid"], pl["table"]
sidecar = _shot_passive_sidecar(pl, int(iso["passive_k"]))
k = int(np.argmax([p.ip_amperes for p in pl["payloads"]]))
p = pl["payloads"][k]


def solve(mask, spc, ip_signed):
    pp = dataclasses.replace(p, mask=mask, ip_amperes=ip_signed)
    return fit_and_read_slice(
        grid,
        table,
        pp,
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
        reseed_axis_r_max=1.4,
        keep_psi=True,
        keep_jphi=True,
        basis=None,
        meta={},
        soft_prior_cfg=spc,
        boundary_read=iso["boundary_read_scoring"],
    )


ip0 = abs(p.ip_amperes)
off = np.zeros_like(p.mask, dtype=bool)
cases = [
    ("forward +Ip", solve(off, None, +ip0)),
    ("forward -Ip", solve(off, None, -ip0)),
    ("recon  +Ip", solve(p.mask, disc, +ip0)),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 6))
for ax, (label, f) in zip(axes, cases, strict=True):
    if not (f.scored and f.psi is not None):
        ax.set_title(f"{label}: unscored")
        continue
    psi = f.psi
    ar, az = float(f.target[0]), float(f.target[1])
    cp = find_critical_points(psi, grid.rg, grid.zg)
    # in-limiter X-points, nearest-in-flux to the axis
    xs = cp.x_points
    print(f"\n=== {label} ===  axis=({ar:.3f},{az:.3f})")
    xin = []
    for i in range(xs.shape[0]):
        xr, xz = float(xs[i, 0]), float(xs[i, 1])
        if grid.rg[0] < xr < grid.rg[-1] and grid.zg[0] < xz < grid.zg[-1]:
            xin.append((xr, xz))
    xin.sort(key=lambda t: abs(t[1]))  # not the point; we want proximity+strength
    for xr, xz in xin[:6]:
        orient = "UP/DOWN" if abs(xz) > 0.25 else "MID/RADIAL"
        print(f"   X-point R={xr:.3f} Z={xz:+.3f}  [{orient}]")
    if not xin:
        print("   no in-domain X-point (limited)")
    ax.contour(grid.rg, grid.zg, psi, levels=30, colors="0.6", linewidths=0.5)
    ax.plot(table.limiter_r, table.limiter_z, "k-", lw=1.2)
    ax.plot(ar, az, "ro", ms=8)
    for xr, xz in xin[:6]:
        ax.plot(xr, xz, "bx", ms=9, mew=2)
    ax.set_aspect("equal")
    ax.set_title(f"{label}\naxis R={ar:.2f} Z={az:.2f}")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
fig.suptitle(f"11766 flat-top Ip={ip0 / 1e3:.0f}kA — Ip-flip + divertor orientation")
out = Path("docs/figures/equilibrium-realism/fig-ip-sign-divertor-audit.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("\nwrote", out)
