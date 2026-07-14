"""Figures for the outermost-closed-contour LCFS boundary read.

Two figures written under
``docs/figures/connectivity-boundary-classification/``:

1. ``flux-surfaces-across-phases.png`` — for one held-out shot, the masked
   TOTAL psi with its nested flux contours and the pushed LCFS ring at four
   phases (breakdown/ramp → flat-top → termination), the emergent X-point, the
   limiter, and the firewalled EFIT reference boundary for comparison.  Shows
   the read behaving sensibly across the whole discharge with ONE code path.
2. ``scoring-and-timing.png`` — held-out (n=160) LCFS + X-point skill, old
   ray-cast read vs the new contour push (with bootstrap CIs), and the per-slice
   timing of the two chains.

Run: ``uv run python scripts/figures_connectivity_boundary.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from boundary_harmonic_gate_eval import (  # noqa: E402
    _adaptive_radii,
    _origin_and_pole,
    hybrid_target_harmonic,
    sensor_arrays,
)
from boundary_moment_gate_eval import load_cohort  # noqa: E402

from imas_ambix.latent.boundary_harmonic import (  # noqa: E402
    HarmonicFitConfig,
    _fit_one,
    harmonic_columns,
    harmonic_sensor_matrix,
)
from imas_ambix.latent.boundary_moment import (  # noqa: E402
    MomentFitConfig,
    fit_moment_currents,
)
from imas_ambix.latent.topology import lcfs_contour  # noqa: E402
from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES  # noqa: E402

OUT = Path("docs/figures/connectivity-boundary-classification")
ORDER, RIDGE, FRACTION = 3, 1e-8, 0.41


def _read_slice(payload, k, order, ridge, fraction):
    """Replicate the gate's per-slice read; return the plot-relevant fields."""
    grid, basis, table = payload["grid"], payload["basis"], payload["table"]
    n_cells = int(basis.r_cells.shape[0])
    sr, sz, sang, is_flux = sensor_arrays(table)
    rr, zz = np.meshgrid(grid.rg, grid.zg)
    gr, gz = rr.ravel(), zz.ravel()
    p = payload["payloads"][k]

    class _A:  # minimal args shim for _origin_and_pole / _adaptive_radii
        pole_source = "track"
        pole_r = None
        pole_z = 0.0
        mask_frac = 0.5
        exclude_frac = 1.1
        mask_radius = 0.25
        exclude_radius = 0.55

    mom = fit_moment_currents(basis, p, MomentFitConfig(order=3))
    origin = (mom.centroid_r, mom.centroid_z)
    origin, pole = _origin_and_pole(origin, grid, _A, fraction)
    mask_r, excl_r = _adaptive_radii(origin, pole, _A)
    cfg = HarmonicFitConfig(pole_r=pole[0], pole_z=pole[1], order=order, ridge=ridge)
    a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
    coeffs, _, _ = _fit_one(a_sens, p.measured, p.vacuum, p.mask, p.scale, cfg.ridge)
    grid_cols, _ = harmonic_columns(gr, gz, cfg)
    psi_plasma = (grid_cols @ coeffs).reshape(grid.nz, grid.nr)
    psi_coil = basis.psi_grid_2d_np(np.zeros(n_cells), p.i_pf)
    psi_tot = psi_plasma + psi_coil
    target, axis_psi, boundary_psi, field, diverted = hybrid_target_harmonic(
        psi_tot, grid, origin, pole, mask_r, excl_r
    )
    lcfs = lcfs_contour(
        field,
        grid.rg,
        grid.zg,
        origin,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    return {
        "grid": grid,
        "field": field,
        "lcfs": lcfs,
        "origin": origin,
        "target": target,
        "ref": payload["refs"][k],
        "diverted": diverted,
        "ip": abs(p.ip_amperes),
    }


def _panel(ax, rec, title):
    grid = rec["grid"]
    field = rec["field"]
    rg, zg = grid.rg, grid.zg
    # clip the deep masked plateau for a readable colour range
    finite = np.isfinite(field)
    vmax = np.nanpercentile(field[finite], 98)
    vmin = np.nanpercentile(field[finite], 2)
    ax.contourf(
        rg, zg, np.clip(field, vmin, vmax), levels=24, cmap="viridis", alpha=0.7
    )
    ax.contour(rg, zg, field, levels=18, colors="white", linewidths=0.35, alpha=0.6)
    ax.plot(grid.limiter_r, grid.limiter_z, "-", color="0.2", lw=1.3, label="limiter")
    lcfs = rec["lcfs"]
    if lcfs.found and lcfs.ring.shape[0]:
        ax.plot(
            lcfs.ring[:, 0],
            lcfs.ring[:, 1],
            "-",
            color="red",
            lw=2.2,
            label="LCFS (push)",
        )
    ax.plot(*rec["origin"], "o", color="yellow", mec="k", ms=7, label="axis")
    tgt = rec["target"]
    for s in range(2):
        xr, xz = tgt[2 + 2 * s], tgt[3 + 2 * s]
        if np.isfinite(xr) and np.isfinite(xz):
            ax.plot(
                xr, xz, "X", color="magenta", mec="k", ms=11, label="X-point (emergent)"
            )
    # firewalled EFIT reference LCFS from its 8 radii about the ref axis
    ref = rec["ref"]
    ax_r, ax_z = ref[0], ref[1]
    rr = ref[6:]
    if np.isfinite(ax_r) and np.isfinite(rr).any():
        rr_c = np.concatenate([rr, rr[:1]])
        ang_c = np.concatenate([LCFS_ANGLES, LCFS_ANGLES[:1]])
        ax.plot(
            ax_r + rr_c * np.cos(ang_c),
            ax_z + rr_c * np.sin(ang_c),
            "--",
            color="cyan",
            lw=1.6,
            label="EFIT ref",
        )
    cls = "diverted" if rec["diverted"] else "limited"
    ax.set_title(f"{title}\nIp={rec['ip'] / 1e3:.0f} kA · {cls}", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]", fontsize=8)


def figure_phases(shots):
    """Pick the shot with the widest Ip span; plot 4 phases."""
    best = max(
        shots,
        key=lambda p: (
            (
                max(abs(x.ip_amperes) for x in p["payloads"])
                - min(abs(x.ip_amperes) for x in p["payloads"])
            )
            if len(p["payloads"]) >= 4
            else -1
        ),
    )
    ips = np.array([abs(x.ip_amperes) for x in best["payloads"]])
    n = len(best["payloads"])
    order = np.argsort(ips)  # low Ip (breakdown/termination) → high (flat-top)
    # pick spread: lowest, ~33%, flat-top (max), and a late (near-min after max)
    idx_lo = int(order[0])
    idx_mid = int(order[n // 3])
    idx_ft = int(order[-1])
    # a termination-side slice: the last slice in time with Ip below flat-top
    idx_term = (
        int(np.argmin(ips[max(idx_ft, 1) :]) + max(idx_ft, 1))
        if idx_ft < n - 1
        else int(order[1])
    )
    picks = [
        (idx_lo, "breakdown / ramp"),
        (idx_mid, "rising"),
        (idx_ft, "flat-top"),
        (idx_term, "termination"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (k, name) in zip(axes, picks, strict=True):
        rec = _read_slice(best, k, ORDER, RIDGE, FRACTION)
        _panel(ax, rec, name)
    axes[0].set_ylabel("Z [m]", fontsize=8)
    # single dedup legend
    h, lb = axes[0].get_legend_handles_labels()
    seen = dict(zip(lb, h, strict=True))
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    shot_id = int(best["payloads"][0].shot)
    fig.suptitle(
        f"LCFS by outermost closed axis-enclosing flux contour — shot {shot_id}, "
        "one code path across the discharge (red=push LCFS, cyan=EFIT ref)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "flux-surfaces-across-phases.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "flux-surfaces-across-phases.png")


def figure_scoring():
    """Old ray-cast vs new contour push: held-out skill (CIs) + per-slice timing."""
    art = Path(
        "imas_ambix/latent/artifacts/patch_gate/"
        "boundary_read_harmonic-o3-centroidorigin-frac0.41.json"
    )
    new = json.loads(art.read_text())
    old_path = Path(
        os.path.expandvars(
            "$SP/baseline/boundary_read_harmonic-o3-centroidorigin-frac0.41.json"
        )
    )
    old = json.loads(old_path.read_text()) if old_path.exists() else None
    # timing from the profiler numbers (measured)
    timing = {
        "OLD\ncrit + ray-cast": 12.3,
        "NEW\nLCFS push": 5.5,
        "NEW + emergent\npush+crit+prox": 9.7,
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if old is not None:
        labels = ["LCFS skill\n(primary)", "X-point skill\n(emergent)"]
        old_v = [old["lcfs_skill"], old["xpoint_set_skill"]]
        new_v = [new["lcfs_skill"], new["xpoint_set_skill"]]
        old_ci = [old["lcfs_skill_ci"], old["xpoint_set_skill_ci"]]
        new_ci = [new["lcfs_skill_ci"], new["xpoint_set_skill_ci"]]
        x = np.arange(len(labels))
        w = 0.36
        for v, ci, off, c, lab in [
            (old_v, old_ci, -w / 2, "0.5", "old ray-cast"),
            (new_v, new_ci, w / 2, "tab:red", "new contour push"),
        ]:
            err = np.array(
                [[v[j] - ci[j][0], ci[j][1] - v[j]] for j in range(len(v))]
            ).T
            a1.bar(x + off, v, w, color=c, label=lab)
            a1.errorbar(x + off, v, yerr=err, fmt="none", ecolor="k", capsize=4, lw=1)
        a1.axhline(-2.48, ls=":", color="green", lw=1.2, label="plan bar (−2.48)")
        a1.set_xticks(x)
        a1.set_xticklabels(labels, fontsize=9)
        a1.set_ylabel("skill vs persistence")
        a1.set_title("Held-out (n=160) skill — 2000-boot CIs", fontsize=10)
        a1.legend(fontsize=8, loc="lower right")
    a2.bar(
        list(timing.keys()),
        list(timing.values()),
        color=["0.5", "tab:red", "tab:orange"],
    )
    a2.set_ylabel("ms / slice (65×97, 1-thread CPU)")
    a2.set_title("Per-slice boundary-read cost (measured)", fontsize=10)
    for i, v in enumerate(timing.values()):
        a2.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "scoring-and-timing.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "scoring-and-timing.png")


def main() -> int:
    class Args:
        split = "eval"
        nr, nz = 65, 97
        n_train, n_heldout = 40, 8
        n_baseline_shots, n_tune_shots = 20, 8
        max_slices_per_shot = 20
        min_ip_ka = 50.0
        origin_source = "centroid"

    shots, _ = load_cohort("eval", Args)
    print(f"loaded {len(shots)} held-out shots")
    figure_phases(shots)
    figure_scoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
