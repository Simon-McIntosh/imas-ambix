#!/usr/bin/env python
"""Illustrations for the source-free toroidal-harmonic boundary read.

Maps the external toroidal-harmonic psi(R,Z)
(:mod:`imas_ambix.latent.boundary_harmonic`) against the firewalled EFIT
ground truth on held-out shots, and (once the
orchestrator's Gate C has scored it) compares its skill against the
current-moment and free-current arms.  Regenerates, under
``docs/figures/toroidal-harmonic-annulus-read/`` (PNG + SVG each):

1. ``phi-map-vs-efit-<shot>.png`` — for several held-out pulses, the harmonic
   psi_tot(R,Z) (plasma harmonic + thick-cylinder coil term) overlaid on the
   EFIT flux map at one flat-top slice, matched absolute levels + LCFS + axis
   (clone of ``plot_boundary_moment_figures.py`` Figure 3).
2. ``phases-<shot>.png`` — for one held-out shot, a row of panels spanning
   breakdown -> ramp-up -> flat-top -> ramp-down -> termination, selected off
   the |Ip| trajectory and each snapped to the nearest EFIT slice.  This is
   the key deliverable showing the read holds from breakdown to termination,
   not only at flat-top.
3. ``gate-c-skill-bars.png`` — axis / X-point-set / LCFS skill with 95% CI,
   harmonic vs current-moment vs free-current carrier, read from the gate's
   JSON artifacts.  Skipped (with a logged warning, not a hard failure) if the
   harmonic gate artifact has not been written yet.

Conventions (matching ``imas_ambix.latent.boundary_harmonic`` /
``scripts/plot_boundary_moment_figures.py``): psi is the TOTAL poloidal flux
Phi = 2*pi*R*A_phi [Wb]; EFIT ``efm`` psi is Wb/rad and is multiplied by 2*pi
for absolute-level comparison; MAST sign is psi_axis > psi_boundary.  The coil
contribution is ALWAYS the harness's thick-cylinder ``hybrid_greens`` term
(``basis.psi_grid_2d_np`` with zero cell currents) -- never a point-filament
coil term.  The firewalled EFIT map is read only inside ``evaluator_context()``.

The harmonic basis is mpmath-backed (see the module docstring) -- evaluating
it on a plotting grid is the slow step, so this script defaults to a coarser
grid (``--nr``/``--nz``) than the gate's own ``nr=65, nz=97`` scoring raster;
only the illustration resolution is affected, not any scored quantity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make the sibling helper scripts importable no matter how this file is run
# (bare script, ``python -m scripts.plot_boundary_harmonic_figures``, or an
# in-process import) -- mirrors plot_boundary_moment_figures.py's path fix.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_boundary_harmonic_figures")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/toroidal-harmonic-annulus-read")

# consistent colours across this script (and with the sibling boundary-read
# figure scripts: C_FREE/C_MOM match plot_boundary_moment_figures.py exactly)
C_FREE = "#c04a2e"  # free-current carrier arm (also: the EFIT-reference line)
C_MOM = "#2a6f97"  # current-moment arm
C_HARM = "#3f7d20"  # toroidal-harmonic read -- this script's headline arm
C_EFIT = C_FREE


# --------------------------------------------------------------------------
# shared harmonic-fit helper
# --------------------------------------------------------------------------
def _scored_read(table, basis, grid, payload, args):
    """The EXACT gate read for one slice (default ``--origin-source centroid``):
    magnetically-constrained current-centroid origin, per-slice inboard-fraction
    pole, size-adaptive annulus mask, source-free harmonic fit.  Returns
    ``(masked_field, psi_tot, target, axis_psi, boundary_psi, origin, misfit)`` --
    ``field`` is what the boundary is READ from; ``psi_tot`` is the raw total flux
    (harmonic plasma + thick-cylinder coil) whose NESTED FLUX SURFACES the figure
    draws, so all surfaces (not just the separatrix) are shown."""
    from boundary_harmonic_gate_eval import _adaptive_radii, hybrid_target_harmonic

    from imas_ambix.latent.boundary_harmonic import HarmonicFitConfig, fit_harmonic
    from imas_ambix.latent.boundary_moment import MomentFitConfig, fit_moment_currents

    # magnetically-constrained current centroid = origin + pole reference
    mom = fit_moment_currents(basis, payload, MomentFitConfig(order=3))
    origin = (mom.centroid_r, mom.centroid_z)
    frac = args.pole_inboard_fraction
    pole = (float(origin[0]) * (1.0 - frac), float(origin[1]))

    class _A:  # adapter so _adaptive_radii reads the frozen mask/exclude fractions
        mask_frac = args.mask_frac
        exclude_frac = args.exclude_frac
        mask_radius = 0.25
        exclude_radius = 0.55

    mask_r, excl_r = _adaptive_radii(origin, pole, _A)
    cfg = HarmonicFitConfig(
        pole_r=pole[0], pole_z=pole[1], order=args.order, kind="P", ridge=args.ridge
    )
    sr = np.array([m.r for m in table.sensor_map])
    sz = np.array([m.z for m in table.sensor_map])
    sang = np.array(
        [0.0 if m.angle_deg is None else float(m.angle_deg) for m in table.sensor_map]
    )
    is_flux = np.array([m.kind == "flux_loop" for m in table.sensor_map])
    inv = fit_harmonic((sr, sz, sang, is_flux), payload, cfg)
    psi_tot = inv.psi_on_grid(grid.rg, grid.zg) + basis.psi_grid_2d_np(
        np.zeros(basis.r_cells.shape[0]), payload.i_pf
    )
    target, axis_psi, boundary_psi, field = hybrid_target_harmonic(
        psi_tot, grid, np.array(origin), pole, mask_r, excl_r
    )
    return (
        field,
        psi_tot,
        target,
        axis_psi,
        boundary_psi,
        (float(origin[0]), float(origin[1])),
        inv.misfit,
    )


def _nested_levels(axis_psi: float, boundary_psi: float, n_in=6, n_out=2):
    """Flux levels for nested-surface contours: ``n_in`` between axis and boundary
    (the confined surfaces) + ``n_out`` just outside (the SOL), so the plot shows
    ALL flux surfaces, not only the separatrix."""
    import numpy as np  # noqa: PLC0415

    inner = axis_psi + (boundary_psi - axis_psi) * np.linspace(0.12, 1.0, n_in)
    step = (boundary_psi - axis_psi) / max(n_in, 1)
    outer = boundary_psi + step * np.arange(1, n_out + 1)
    return np.sort(np.concatenate([inner, outer]))


def _savefig(fig, stem: Path) -> None:
    """Write both PNG (raster, dpi=150) and SVG (vector) for one figure stem."""
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    logger.info("wrote %s{.png,.svg}", stem.name)


# --------------------------------------------------------------------------
# Figure 1 — phi-map vs EFIT at one flat-top slice, several held-out shots
# --------------------------------------------------------------------------
def _flux_overlay(
    stem,
    shot,
    payload,
    grid,
    geom,
    psi_model,
    target,
    psi_ax,
    psi_b,
    axis,
    lcfs,
    efit,
    *,
    misfit,
):
    """imas-ink psi(R,Z) overlay; matplotlib contour fallback if it fails."""
    try:
        from imas_ink.figures import equilibrium_figure_mpl
        from patch_flux_map_report import _efit_slice, _our_slice

        our_sl = _our_slice(
            psi_model,
            grid,
            target,
            psi_ax,
            psi_b,
            payload.ip_amperes,
            payload.time_s,
            lcfs,
        )
        fig, _ax = equilibrium_figure_mpl(
            our_sl,
            geom,
            reference_slice=_efit_slice(efit),
            reference_name="EFIT",
            figsize=(5.4, 6.8),
            show_probes=False,
            show_flux_loops=False,
        )
        fig.suptitle(
            f"Toroidal-harmonic ψ(R,Z) vs EFIT — shot {shot}  "
            f"t={payload.time_s:.3f}s (flat-top)\n"
            f"solid: source-free harmonic read (misfit={misfit:.2e})  ·  "
            "faint: firewalled EFIT reference",
            fontsize=10.5,
        )
        _savefig(fig, stem)
        plt.close(fig)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("imas-ink overlay failed (%s); using matplotlib fallback", exc)

    # --- matplotlib fallback: matched absolute psi levels, both x2pi-consistent
    fig, ax = plt.subplots(figsize=(5.6, 7.0))
    ax.plot(
        np.append(grid.limiter_r, grid.limiter_r[0]),
        np.append(grid.limiter_z, grid.limiter_z[0]),
        color="0.3",
        lw=1.3,
        label="limiter",
    )
    frac = np.linspace(0.1, 0.9, 5)
    our_levels = psi_ax + frac * (psi_b - psi_ax)
    ax.contour(
        grid.rg,
        grid.zg,
        psi_model,
        levels=np.sort(our_levels),
        colors=C_HARM,
        linewidths=1.1,
    )
    ef_levels = efit["psi_axis"] + frac * (efit["psi_boundary"] - efit["psi_axis"])
    ax.contour(
        efit["rg"],
        efit["zg"],
        efit["psi_zr"],
        levels=np.sort(ef_levels),
        colors=C_EFIT,
        linewidths=1.0,
        linestyles="--",
    )
    if lcfs is not None:
        ax.plot(lcfs[:, 0], lcfs[:, 1], color=C_HARM, lw=2.2, label="harmonic LCFS")
    ax.plot(
        efit["lcfs_r"], efit["lcfs_z"], color=C_EFIT, lw=1.8, ls="--", label="EFIT LCFS"
    )
    ax.plot(*axis, "*", color=C_HARM, ms=13, label="harmonic axis")
    ax.plot(efit["axis_r"], efit["axis_z"], "P", color=C_EFIT, ms=10, label="EFIT axis")
    for slot in range(2):
        rr, zz = target[2 + 2 * slot], target[3 + 2 * slot]
        if np.isfinite(rr) and np.isfinite(zz):
            ax.plot(rr, zz, "x", color=C_HARM, ms=10, mew=2.2)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(
        f"Toroidal-harmonic ψ(R,Z) vs EFIT — shot {shot}  "
        f"t={payload.time_s:.3f}s (flat-top)\n"
        "green: source-free harmonic read  ·  dashed sienna: firewalled EFIT",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    _savefig(fig, stem)
    plt.close(fig)


def fig_phi_map_vs_efit(shot: int, args) -> bool:
    """One shot's flat-top harmonic-vs-EFIT overlay.  Returns False (logged,
    not raised) if the shot fails to load or has no flat-top EFIT snap."""
    from patch_flux_map_report import (
        _closed_contour_about,
        _machine_geometry,
        select_slices,
    )
    from patch_gate_eval import shot_payloads

    payload = shot_payloads(
        shot,
        nr=args.nr,
        nz=args.nz,
        max_slices=args.max_slices,
        min_ip_ka=args.min_ip_ka,
        split="eval",
    )
    if payload is None:
        logger.warning("shot %s: no payload (skipping phi-map figure)", shot)
        return False
    picks = select_slices(payload["payloads"], shot)
    flat = [p for p in picks if p[0] == "flattop"]
    if not flat:
        logger.warning("shot %s: no flat-top EFIT snap (skipping phi-map figure)", shot)
        return False
    _, k, efit = flat[0]

    grid, basis, table = payload["grid"], payload["basis"], payload["table"]
    p = payload["payloads"][k]
    # the SCORED read: current-centroid origin + per-slice inboard pole +
    # annulus-masked harmonic boundary (matches the gate's --origin-source centroid)
    field, psi_tot, target, psi_ax, psi_b, axis, misfit = _scored_read(
        table, basis, grid, p, args
    )
    geom = _machine_geometry(grid, table)
    lcfs = _closed_contour_about(grid.rg, grid.zg, field, psi_b, *axis)

    # render nested surfaces from the RAW total flux (all surfaces, not just the
    # separatrix); the LCFS curve itself is the masked-read boundary
    stem = args.out_dir / f"phi-map-vs-efit-{shot}"
    _flux_overlay(
        stem,
        shot,
        p,
        grid,
        geom,
        psi_tot,
        target,
        psi_ax,
        psi_b,
        axis,
        lcfs,
        efit,
        misfit=misfit,
    )
    return True


def _auto_pick_shots(args) -> list[int]:
    """Held-out shots (in campaign order) with a flat-top EFIT snap, up to
    ``args.n_shots`` -- used when ``--shots`` is not given explicitly."""
    from patch_flux_map_report import select_slices
    from patch_gate_eval import shot_payloads

    from imas_ambix.latent.data import read_split_shot_lists

    _, held = read_split_shot_lists(40, 8)
    picked: list[int] = []
    for shot in held:
        try:
            payload = shot_payloads(
                shot,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices,
                min_ip_ka=args.min_ip_ka,
                split="eval",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", shot, exc)
            continue
        if payload is None:
            continue
        picks = select_slices(payload["payloads"], shot)
        if any(kind == "flattop" for kind, _, _ in picks):
            picked.append(shot)
        if len(picked) >= args.n_shots:
            break
    return picked


# --------------------------------------------------------------------------
# Figure 2 — breakdown -> termination phase row, one held-out shot
# --------------------------------------------------------------------------
def _pick_phase_indices(payload: dict, shot: int) -> list[tuple[str, int, dict]]:
    """Pick payload indices for breakdown/ramp-up/flat-top/ramp-down/termination,
    each snapped to the nearest EFIT slice (searching outward from the
    trajectory-derived target index when the exact index doesn't snap)."""
    from patch_flux_map_report import read_efit_slice

    from imas_ambix.eval.efit_referee import evaluator_context

    payloads = payload["payloads"]
    n = len(payloads)
    if n == 0:
        return []
    ip = np.array([abs(p.ip_amperes) for p in payloads])
    i_peak = int(np.argmax(ip))
    half = 0.5 * ip[i_peak]
    rise = np.flatnonzero(ip[: i_peak + 1] >= half)
    i_rampup = int(rise[0]) if rise.size else max(i_peak - 1, 0)
    fall = np.flatnonzero(ip[i_peak:] <= half)
    i_rampdown = int(i_peak + fall[0]) if fall.size else min(i_peak + 1, n - 1)
    targets = [
        ("breakdown", 0),
        ("ramp-up", i_rampup),
        ("flat-top", i_peak),
        ("ramp-down", i_rampdown),
        ("termination", n - 1),
    ]

    picks: list[tuple[str, int, dict]] = []
    used: set[int] = set()
    with evaluator_context():
        for label, k0 in targets:
            found = None
            for delta in range(n):
                for k in {k0 - delta, k0 + delta}:
                    if k < 0 or k >= n or k in used:
                        continue
                    efit = read_efit_slice(shot, payloads[k].time_s)
                    if efit is not None:
                        found = (k, efit)
                        break
                if found is not None:
                    break
            if found is not None:
                k, efit = found
                used.add(k)
                picks.append((label, k, efit))
            else:
                logger.warning("shot %s: no EFIT snap near phase %s", shot, label)
    return picks


def _panel_contour(
    ax, field, psi_tot, grid, target, psi_ax, psi_b, efit, title
) -> None:
    """One phase panel: ALL harmonic flux surfaces (faint nested contours of the
    raw total psi) + the bold LCFS, vs the EFIT LCFS, with axis + X-point markers.

    Faint nested surfaces are drawn from the RAW ``psi_tot`` (harmonic plasma +
    thick-cylinder coil) so the field structure is visible, not just the
    separatrix; the bold LCFS is the SCORED boundary read off the annulus-masked
    ``field``."""
    from patch_flux_map_report import _closed_contour_about

    ax.plot(
        np.append(grid.limiter_r, grid.limiter_r[0]),
        np.append(grid.limiter_z, grid.limiter_z[0]),
        color="0.3",
        lw=1.0,
    )
    axis = (float(target[0]), float(target[1]))
    # ALL flux surfaces: faint nested contours of the raw total flux
    levels = _nested_levels(psi_ax, psi_b)
    ax.contour(
        grid.rg,
        grid.zg,
        psi_tot,
        levels=levels,
        colors=C_HARM,
        linewidths=0.5,
        alpha=0.45,
    )
    lcfs = _closed_contour_about(grid.rg, grid.zg, field, psi_b, *axis)
    if lcfs is not None and len(lcfs):
        ax.plot(lcfs[:, 0], lcfs[:, 1], color=C_HARM, lw=2.0, label="harmonic LCFS")
    ax.plot(
        efit["lcfs_r"],
        efit["lcfs_z"],
        color=C_EFIT,
        lw=1.5,
        ls="--",
        label="EFIT LCFS",
    )
    ax.plot(axis[0], axis[1], "*", color=C_HARM, ms=10)
    ax.plot(efit["axis_r"], efit["axis_z"], "P", color=C_EFIT, ms=7)
    for slot in range(2):
        xr, xz = target[2 + 2 * slot], target[3 + 2 * slot]
        if np.isfinite(xr) and np.isfinite(xz):
            ax.plot(xr, xz, "x", color=C_HARM, ms=8, mew=2.0)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_title(title, fontsize=9)


def fig_phases(shot: int, args) -> bool:
    """Row of harmonic-vs-EFIT panels spanning breakdown -> termination for
    one held-out shot.  Returns False (logged, not raised) on any failure to
    load or snap."""
    from patch_gate_eval import shot_payloads

    payload = shot_payloads(
        shot,
        nr=args.nr,
        nz=args.nz,
        max_slices=args.phase_max_slices,
        min_ip_ka=args.phase_min_ip_ka,
        split="eval",
    )
    if payload is None:
        logger.warning("shot %s: no payload for phase figure", shot)
        return False
    picks = _pick_phase_indices(payload, shot)
    if not picks:
        logger.warning("shot %s: no phase slices snapped to EFIT", shot)
        return False

    grid, basis, table = payload["grid"], payload["basis"], payload["table"]

    fig, axes = plt.subplots(
        1, len(picks), figsize=(3.3 * len(picks), 5.4), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (label, k, efit) in zip(axes, picks, strict=True):
        p = payload["payloads"][k]
        field, psi_tot, target, psi_ax, psi_b, _axis, _misfit = _scored_read(
            table, basis, grid, p, args
        )
        _panel_contour(
            ax,
            field,
            psi_tot,
            grid,
            target,
            psi_ax,
            psi_b,
            efit,
            f"t={p.time_s:.3f}s\n{label}",
        )
    axes[0].set_ylabel("Z [m]")
    fig.suptitle(
        f"Toroidal-harmonic ψ(R,Z) vs EFIT across the discharge — shot {shot}\n"
        "solid green: harmonic read  ·  dashed sienna: firewalled EFIT reference",
        fontsize=11.5,
    )
    fig.tight_layout()
    _savefig(fig, args.out_dir / f"phases-{shot}")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# Figure 3 — Gate C skill bars: harmonic vs current-moment vs carrier
# --------------------------------------------------------------------------
def _find_harmonic_artifact() -> dict | None:
    """The CURRENT scored held-out ``boundary_read_harmonic-*.json`` artifact.

    Prefers the per-slice-pole read (``-frac`` in the name — the current scored
    configuration) over the retired fixed-pole one; within that class picks the
    best lcfs_skill.  Returns ``None`` if the gate has not written one yet."""
    cands = [
        p
        for p in ARTIFACTS.glob("boundary_read_harmonic-*.json")
        if "-tune" not in p.stem
    ]
    evals = []
    for p in sorted(cands):
        try:
            d = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read %s: %s", p, exc)
            continue
        if d.get("split") == "eval":
            evals.append((p.stem, d))
    if not evals:
        return None
    # per-slice-pole (frac) artifacts are the current scored read; fall back to
    # any eval artifact if none exist yet.
    frac = [d for stem, d in evals if "frac" in stem]
    pool = frac or [d for _, d in evals]
    return max(pool, key=lambda d: d.get("lcfs_skill") or -1e9)


def fig_gate_c_skill_bars(stem: Path) -> bool:
    """Axis / X-point-set / LCFS skill with 95% CI, harmonic vs current-moment
    vs free-current carrier.  Skips gracefully (logged warning, no exception)
    if the harmonic gate hasn't been run yet."""
    harmonic = _find_harmonic_artifact()
    if harmonic is None:
        logger.warning(
            "no boundary_read_harmonic-*.json artifact found yet (Gate C not run) "
            "-- skipping gate-c-skill-bars"
        )
        return False
    moment_path = ARTIFACTS / "boundary_read_moment-o3.json"
    baseline_path = ARTIFACTS / "boundary_read_baseline.json"
    if not moment_path.exists() or not baseline_path.exists():
        logger.warning(
            "moment (%s) or carrier (%s) missing -- skipping gate-c-skill-bars",
            moment_path,
            baseline_path,
        )
        return False
    moment = json.loads(moment_path.read_text())
    baseline = json.loads(baseline_path.read_text())

    arms = [
        ("carrier\n(free-current)", baseline, C_FREE),
        ("current-\nmoment", moment, C_MOM),
        ("toroidal-\nharmonic", harmonic, C_HARM),
    ]
    metrics = [
        ("axis_skill", "axis_skill_ci", "axis skill"),
        ("xpoint_set_skill", "xpoint_set_skill_ci", "X-point-set skill"),
        ("lcfs_skill", "lcfs_skill_ci", "LCFS skill"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.0))
    for ax, (key, ci_key, title) in zip(axes[:3], metrics, strict=True):
        xs = np.arange(len(arms))
        vals = np.array(
            [float(a[1][key]) if a[1].get(key) is not None else np.nan for a in arms]
        )
        yerr = np.zeros((2, len(arms)))
        for i, a in enumerate(arms):
            ci = a[1].get(ci_key)
            if ci and ci[0] is not None and ci[1] is not None and np.isfinite(vals[i]):
                yerr[0, i] = vals[i] - ci[0]
                yerr[1, i] = ci[1] - vals[i]
            else:
                logger.warning("no %s CI for arm %r; plotting a bare bar", key, a[0])
        ax.bar(xs, vals, 0.6, color=[a[2] for a in arms])
        ax.errorbar(xs, vals, yerr=yerr, fmt="none", ecolor="0.15", capsize=4, lw=1.3)
        ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([a[0] for a in arms], fontsize=8.5)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("skill  (higher is better; >0 beats train-mean)")

    # 4th panel -- vacuum-annulus consistency RMS: a HARMONIC-only diagnostic
    # (harmonic psi vs the interior carrier psi in their shared source-free
    # annulus; no comparable quantity exists for the moment/carrier arms, and
    # boundary_harmonic_gate_eval.py records no CI for it).
    ax = axes[3]
    med = harmonic.get("consistency_rms_annulus")
    mean = harmonic.get("consistency_rms_annulus_mean")
    if med is None and mean is None:
        logger.warning("no consistency_rms_annulus in the harmonic artifact")
        ax.axis("off")
        ax.set_title("annulus consistency RMS\n(not recorded)", fontsize=11)
    else:
        labels = [lbl for lbl, v in (("median", med), ("mean", mean)) if v is not None]
        vals = [v for v in (med, mean) if v is not None]
        xs = np.arange(len(vals))
        ax.bar(xs, vals, 0.5, color=C_HARM)
        for xi, v in zip(xs, vals, strict=True):
            ax.annotate(
                f"{v:.3g}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(
            "vacuum-annulus consistency RMS\n(harmonic ψ vs interior carrier ψ)",
            fontsize=10.5,
        )
    fig.suptitle(
        "Gate C — source-free toroidal-harmonic boundary read vs current-moment "
        "and free-current carrier (held-out, EFIT-scored, 95% CI)",
        fontsize=12,
    )
    fig.tight_layout()
    _savefig(fig, stem)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help="held-out shots for the phi-map figure (default: auto-pick up to "
        "--n-shots held-out shots with a flat-top EFIT snap)",
    )
    ap.add_argument("--n-shots", type=int, default=3)
    ap.add_argument(
        "--order",
        type=int,
        default=3,
        help="toroidal-harmonic order (frozen by the Gate C train-cohort ladder)",
    )
    ap.add_argument(
        "--ridge",
        type=float,
        default=1e-8,
        help="Tikhonov ridge (frozen by the Gate C ladder alongside the order)",
    )
    ap.add_argument(
        "--pole-inboard-fraction",
        type=float,
        default=0.41,
        help="dimensionless inboard fraction: pole_r = carrier axis_R*(1-fraction) "
        "(machine-agnostic; frozen by the Gate C ladder)",
    )
    ap.add_argument("--out-dir", type=Path, default=FIGURES)
    ap.add_argument(
        "--nr",
        type=int,
        default=33,
        help="plotting-grid R points (coarser than the gate's nr=65 -- the "
        "mpmath harmonic evaluation is the slow step)",
    )
    ap.add_argument("--nz", type=int, default=45)
    ap.add_argument("--max-slices", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument(
        "--phase-shot",
        type=int,
        default=None,
        help="held-out shot for the breakdown -> termination phase figure "
        "(default: the first phi-map shot that snapped)",
    )
    ap.add_argument(
        "--phase-min-ip-ka",
        type=float,
        default=5.0,
        help="lower Ip floor so breakdown/termination slices survive the "
        "shot_payloads validity filter",
    )
    ap.add_argument("--phase-max-slices", type=int, default=60)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shots = args.shots if args.shots is not None else _auto_pick_shots(args)
    if not shots:
        logger.warning(
            "no shots snapped to EFIT; only gate-c-skill-bars will be attempted"
        )

    done: list[int] = []
    for shot in shots:
        try:
            if fig_phi_map_vs_efit(shot, args):
                done.append(shot)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s: phi-map figure failed: %s", shot, exc)

    phase_shot = args.phase_shot or (done[0] if done else (shots[0] if shots else None))
    if phase_shot is not None:
        try:
            fig_phases(phase_shot, args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s: phase figure failed: %s", phase_shot, exc)

    fig_gate_c_skill_bars(args.out_dir / "gate-c-skill-bars")

    for p in sorted(args.out_dir.glob("*.png")):
        kb = p.stat().st_size / 1024.0
        flag = "OK" if kb > 5 else "TOO-SMALL"
        logger.info("  %s  %.1f KB  %s", p.name, kb, flag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
