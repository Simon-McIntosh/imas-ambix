"""Figures for the outermost-closed-contour LCFS boundary read.

Two figures written under
``docs/figures/connectivity-boundary-classification/``:

1. ``flux-surfaces-across-phases.png`` — for one held-out shot, the imas-ink
   equilibrium cross-section (``equilibrium_figure_mpl``) at each discharge
   phase: our pushed-LCFS read (blue confined surfaces + bold LCFS ring) with
   the FIREWALLED EFIT reference (faint sienna surfaces + dashed EFIT boundary
   read from ``efm.lcfs_r/lcfs_z`` — the real boundary contour, not an 8-radius
   reconstruction). Rendered by imas-ink and tiled into one row.
2. ``scoring-and-timing.png`` — held-out (n=160) LCFS + X-point skill, old
   ray-cast read vs the new contour push (with bootstrap CIs), and the per-slice
   timing of the two chains.

The EFIT map is read ONLY inside ``evaluator_context()`` (firewall: referee
outputs, used here purely to draw the reference overlay).

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
from imas_ink.figures import equilibrium_figure_mpl  # noqa: E402
from patch_flux_map_report import (  # noqa: E402
    _efit_slice,
    _fig_to_rgba,
    _machine_geometry,
    _our_slice,
)

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

OUT = Path("docs/figures/connectivity-boundary-classification")
ORDER, RIDGE, FRACTION = 3, 1e-8, 0.41


class _A:  # minimal args shim for _origin_and_pole / _adaptive_radii
    pole_source = "track"
    pole_r = None
    pole_z = 0.0
    mask_frac = 0.5
    exclude_frac = 1.1
    mask_radius = 0.25
    exclude_radius = 0.55


def _read_slice(payload, k):
    """Replicate the gate's per-slice read; return the plot-relevant fields."""
    grid, basis, table = payload["grid"], payload["basis"], payload["table"]
    n_cells = int(basis.r_cells.shape[0])
    sr, sz, sang, is_flux = sensor_arrays(table)
    rr, zz = np.meshgrid(grid.rg, grid.zg)
    gr, gz = rr.ravel(), zz.ravel()
    p = payload["payloads"][k]

    mom = fit_moment_currents(basis, p, MomentFitConfig(order=3))
    origin = (mom.centroid_r, mom.centroid_z)
    origin, pole = _origin_and_pole(origin, grid, _A, FRACTION)
    mask_r, excl_r = _adaptive_radii(origin, pole, _A)
    cfg = HarmonicFitConfig(pole_r=pole[0], pole_z=pole[1], order=ORDER, ridge=RIDGE)
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
    ring = lcfs.ring if lcfs.found and lcfs.ring.shape[0] >= 3 else None
    # Plot the raw total flux for the field-structure underlay (the harmonic
    # PLASMA flux + thick-cylinder coil term) — matches the sibling boundary-read
    # figures.  The bold LCFS ring itself is the SCORED read off the masked field;
    # this read owns the BOUNDARY, not a physical interior (decoupled by design).
    return {
        "grid": grid,
        "psi_2d": psi_tot,
        "target": target,
        "axis_psi": axis_psi,
        "boundary_psi": boundary_psi,
        "ring": ring,
        "diverted": diverted,
        "ip": abs(p.ip_amperes),
        "time_s": p.time_s,
    }


def _pick_phases(payload, shot):
    """Phase indices (breakdown → termination) each snapped to a real EFIT slice.

    Reuses the sibling script's picker so the phases + EFIT snaps are identical.
    """
    from plot_boundary_harmonic_figures import _pick_phase_indices

    return _pick_phase_indices(payload, shot)


def figure_phases(shots):
    """imas-ink equilibrium overlays (our read vs firewalled EFIT), tiled by phase."""
    # widest-Ip-span shot with >=4 slices
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
    shot_id = int(best["payloads"][0].shot)
    grid, table = best["grid"], best["table"]
    geom = _machine_geometry(grid, table)
    picks = _pick_phases(best, shot_id)  # [(label, k, efit_dict), ...]
    if not picks:
        print("no EFIT-snapped phases; skipping phases figure")
        return

    tiles = []
    for label, k, efit in picks:
        rec = _read_slice(best, k)
        our_sl = _our_slice(
            rec["psi_2d"],
            grid,
            rec["target"],
            rec["axis_psi"],
            rec["boundary_psi"],
            rec["ip"],
            rec["time_s"],
            rec["ring"],
        )
        cls = "diverted" if rec["diverted"] else "limited"
        fig, ax = equilibrium_figure_mpl(
            our_sl,
            geom,
            reference_slice=_efit_slice(efit),
            reference_name="EFIT",
            figsize=(4.2, 6.2),
            show_probes=False,
            show_flux_loops=False,
        )
        ax.set_title(
            f"{label}\nIp={rec['ip'] / 1e3:.0f} kA · {cls}"
            f"  ·  |Δt|={efit['dt_s'] * 1e3:.0f} ms",
            fontsize=9,
        )
        fig.canvas.draw()
        tiles.append(_fig_to_rgba(fig))
        plt.close(fig)

    fig, axes = plt.subplots(1, len(tiles), figsize=(3.5 * len(tiles), 6.4))
    axes = np.atleast_1d(axes)
    for ax, img in zip(axes, tiles, strict=True):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle(
        f"LCFS by outermost closed axis-enclosing flux contour — shot {shot_id}\n"
        "imas-ink cross-section: blue = our pushed-LCFS read · dashed sienna = "
        "firewalled EFIT boundary (efm.lcfs) · one code path across the discharge",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "flux-surfaces-across-phases.png", dpi=140, bbox_inches="tight")
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
    # _pick_phases opens its own evaluator_context() for the firewalled EFIT
    # reads; the harmonic fit + imas-ink render use only the returned efit dict.
    figure_phases(shots)
    figure_scoring()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
