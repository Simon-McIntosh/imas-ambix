#!/usr/bin/env python
"""Low-DOF closure arm: EFIT-style profile-parametrised free-boundary inverse.

The free ~4,787-DOF patch inverse leaves a ~4,700-dimensional null space that
only soft priors span, so nothing shapes how current distributes within its
support.  EFIT's boundary is "trivial" instead because it parametrises
jφ(R,ψ_N) = R·p′(ψ_N) + FF′(ψ_N)/(μ₀R) with ~10 DOF, solves the free-boundary
fixed point each Picard sweep, and confines support by construction.  This
arm scores that fixed point on the hardened gate harness: per held-out slice
fit the two-parameter jφ(ψ_N; β0, α) profile against the RAW magnetics
(measured − vacuum, whitened, masked) through the in-tree free-boundary GS
solve (:mod:`imas_ambix.latent.gs_solve`), read axis / X-point set / LCFS from
the force-balanced ψ with the SAME readout the other arms use
(:func:`scripts.patch_gate_eval.geometry_target`), and score against the
firewalled EFIT referee with the paired-bootstrap CIs, saddle-excess metric,
cohort, and split the patch arm uses.

Coverage is reported honestly (n_scored / n_candidate): a slice whose best fit
does not meet the relaxed-Picard convergence criterion, or whose whitened
magnetics cost exceeds the honesty ceiling, is REPORTED as masked with its
reason — never scored with a fabricated readout, never silently dropped.

Protocol: leakage-free.  The profile grid, cost ceiling, and Picard knobs are
the frozen physics-motivated defaults, CONFIRMED on ``--split tune`` (the same
4 train-shot tune cohort the patch arm uses) and then applied UNCHANGED to the
single ``--split eval`` held-out run (160 slices).

Firewall: EFIT enters only the referee/scoring/plotting path (inside the
referee's evaluator context), never the fit.  Raw magnetics only; conventions
total flux Φ = 2πR·A_φ [Wb], μ0 explicit, MAST psi_axis > psi_boundary.

Artifacts:  imas_ambix/latent/artifacts/patch_gate/closure_gate_eval[-tune].json
Figures:    docs/figures/plasma-current-priors-hardening/fig-closure-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import numpy as np

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION
from imas_ambix.gs.operator import COIL_MODEL_VERSION
from imas_ambix.latent.data import read_split_shot_lists

if TYPE_CHECKING:
    from imas_ambix.latent.patch_inverse import SlicePayload

from imas_ambix.latent.gs_solve import (
    EquilibriumGrid,
    SoftPriors,
    build_passive_sidecar,
    fit_profile,
    fit_profile_continuous,
    fit_profile_ladder,
    profile_jphi_shape,
    solve_equilibrium_lsq,
)

# reuse the hardened harness verbatim — identical cohort, readout, and scoring
from scripts.patch_gate_eval import (
    count_saddles,
    geometry_target,
    lcfs_offset_cm_stats,
    saddle_excess_stats,
    score,
    shot_payloads,
    train_mean_baseline,
)


def geometry_target_pushout(psi2d, grid, clip_legs=True):
    """14-D geometry read of the interior ψ using the LANDED push-out LCFS
    reader (:func:`imas_ambix.latent.topology.lcfs_contour`) — the outermost
    closed axis-enclosing flux ring — instead of the innermost-X-point /
    limiter-contact read of :func:`geometry_target`.

    The interior free-boundary ψ carries the correct large closed-flux region
    (it matches the §2 source-free read), but the innermost-X read under-sizes
    the LCFS by tens of cm when it locks onto a discretisation-scale saddle.
    This read is the SAME one the §2 harmonic gate uses, so the interior and
    boundary arms become directly comparable.  Axis (target[0:2]) + confined-
    side flux come from :func:`geometry_target`; the LCFS radii (target[6:]) and
    emergent X-points/class (target[2:6]) are re-read off the push-out ring.
    Returns ``(target, psi_axis, psi_boundary)``.
    """
    from imas_ambix.latent.topology import (  # noqa: PLC0415
        _inside_polygon as _inpoly,
    )
    from imas_ambix.latent.topology import (  # noqa: PLC0415
        emergent_xpoints,
        find_critical_points,
        lcfs_contour,
    )

    target, psi_ax, psi_b = geometry_target(psi2d, grid)
    axis = (float(target[0]), float(target[1]))
    lc = lcfs_contour(
        np.asarray(psi2d),
        grid.rg,
        grid.zg,
        axis,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
        clip_legs=clip_legs,
    )
    if not lc.found:
        return target, psi_ax, psi_b
    target = target.copy()
    target[6:] = lc.radii
    boundary_psi = float(lc.psi_bnd)
    # emergent X-points read AFTER the boundary (in-limiter, conductor-clear)
    cp = find_critical_points(np.asarray(psi2d), grid.rg, grid.zg)
    xpts = cp.x_points
    if xpts.shape[0]:
        ins = _inpoly(
            xpts[:, 0], xpts[:, 1], grid.limiter_r, grid.limiter_z
        ) & grid.clear_of_conductors(xpts[:, 0], xpts[:, 1])
        xpts = xpts[ins]
    xset, _diverted = emergent_xpoints(xpts, lc.ring, tol=0.05)
    target[2:6] = xset.reshape(-1)
    return target, psi_ax, boundary_psi


def load_frozen_lookup(path: str):
    """Load the frozen harmonic prior into a ``(shot, t_index) -> slice`` map.

    Returns ``(lookup, meta)`` where ``meta`` carries the frozen config (order /
    ridge / kind) and ``lookup`` keys each per-slice dict (coeffs / coeff_cov /
    origin / pole / misfit / dyn_range) by ``(int(shot), int(t_index))``.
    """
    from imas_ambix.latent.boundary_harmonic import load_frozen_harmonic_prior

    frozen = load_frozen_harmonic_prior(path)
    lookup = {(int(s["shot"]), int(s["t_index"])): s for s in frozen["slices"]}
    return lookup, frozen.get("meta", {})


def _read_harmonic_at_origin(
    payload, grid, table, sensors, origin, *, pole_ref, frac, order, ridge, sobolev_p
):
    """One source-free harmonic read: pole at ``pole_ref``, ray-cast at ``origin``.

    The focal ring is the plasma-CURRENT source, so the pole is placed inboard of
    ``pole_ref`` (the current/moment centroid — physical and stable) by ``frac``,
    NOT inboard of the ray-cast ``origin``.  Keeping the two separate lets the
    ray-cast origin re-site to the LCFS centroid for small-plasma robustness while
    the pole (and the near-pole invalid-interior mask that follows it) stays a
    small central disk — so the mask never drifts onto the inboard/limited edge
    and pinches the boundary ring.  Fits the plasma harmonic (moderate ``order`` +
    graded Sobolev ridge ``sobolev_p`` — the field only needs to carry the smooth
    O-point 'lump'; the X-point cusp is a level-set feature of the KNOWN coil
    saddle extracted by the leg-clip reader), adds the coil field, and reads the
    ψ_N=1 leg-clipped LCFS.  Returns
    ``(cfg, coeffs, misfit, psi_tot, axis_psi, boundary_psi, ring)`` or None.
    """
    from boundary_harmonic_gate_eval import (  # noqa: PLC0415
        _adaptive_radii,
        hybrid_target_harmonic,
    )

    from imas_ambix.latent.boundary_harmonic import (  # noqa: PLC0415
        HarmonicFitConfig,
        _fit_one,
        harmonic_columns,
        harmonic_mode_penalty,
        harmonic_sensor_matrix,
    )
    from imas_ambix.latent.topology import lcfs_contour  # noqa: PLC0415

    sr, sz, sang, is_flux = sensors
    pole = (pole_ref[0] * (1.0 - frac), pole_ref[1])
    cfg = HarmonicFitConfig(
        pole_r=pole[0],
        pole_z=pole[1],
        order=int(order),
        ridge=float(ridge),
        sobolev_p=float(sobolev_p),
    )
    a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
    pen = harmonic_mode_penalty(cfg.order, sobolev_p) if sobolev_p > 0 else None
    coeffs, misfit, _ = _fit_one(
        a_sens,
        payload.measured,
        payload.vacuum,
        payload.mask,
        payload.scale,
        cfg.ridge,
        mode_penalty=pen,
    )
    rr, zz = np.meshgrid(grid.rg, grid.zg)
    cols, _ = harmonic_columns(rr.ravel(), zz.ravel(), cfg)
    psi_tot = (cols @ coeffs).reshape(grid.nz, grid.nr) + grid.coil_psi(
        payload.i_pf
    ).reshape(grid.nz, grid.nr)
    mask_r, excl_r = _adaptive_radii(origin, pole, _RadiiArgs())
    _t, axis_psi, boundary_psi, field, _d = hybrid_target_harmonic(
        psi_tot, grid, origin, pole, mask_r, excl_r, clip_legs=True
    )
    lc = lcfs_contour(
        field,
        grid.rg,
        grid.zg,
        origin,
        clip_legs=True,
        limiter_r=grid.limiter_r,
        limiter_z=grid.limiter_z,
    )
    ring = lc.ring if lc.found else None
    return (
        cfg,
        coeffs,
        float(misfit),
        psi_tot,
        float(axis_psi),
        float(boundary_psi),
        ring,
    )


def _harmonic_read_for_slice(
    payload, grid, table, basis, meta, *, order1=2, sobolev_p=1.0, repass_frac=0.05
):
    """Two-pass source-free harmonic read for one slice (leakage-free, EFIT-off).

    The ray-cast origin is the most sensitive input: the Ip-weighted moment
    centroid ≠ the geometric plasma center for small / moving plasmas, so a
    mis-sited origin skews the whole expansion.  PASS 1 uses a REDUCED order
    (``order1`` — fast, just enough to place the centroid); the LCFS geometric
    centroid it returns re-sites the origin.  PASS 2 refits at the MODERATE
    frozen ``order`` + light graded Sobolev ridge (``sobolev_p``) at the
    re-sited origin.  Pass 2 only fires when the centroid moved more than
    ``repass_frac`` of the plasma minor radius (well-centred flat-tops pay ~0).
    Returns ``(cfg, coeffs, misfit, psi_tot, axis_psi, boundary_psi)`` or None.
    """
    from boundary_harmonic_gate_eval import sensor_arrays  # noqa: PLC0415

    from imas_ambix.latent.boundary_moment import (  # noqa: PLC0415
        MomentFitConfig,
        fit_moment_currents,
    )

    frac = float(meta.get("pole_inboard_fraction", 0.41))
    order2 = int(meta.get("order", 3))
    ridge = float(meta.get("ridge", 1e-8))
    sensors = sensor_arrays(table)
    mom = fit_moment_currents(basis, payload, MomentFitConfig(order=3))
    # the current (moment) centroid anchors the POLE for BOTH passes — it is the
    # physical plasma-current source location and is stable across the re-siting.
    pole_ref = (float(mom.centroid_r), float(mom.centroid_z))
    origin = pole_ref  # ray-cast origin; re-sited below, pole_ref does not move

    # pass 1: reduced order, moment-centroid origin — just locate the LCFS centre
    p1 = _read_harmonic_at_origin(
        payload,
        grid,
        table,
        sensors,
        origin,
        pole_ref=pole_ref,
        frac=frac,
        order=min(order1, order2),
        ridge=ridge,
        sobolev_p=0.0,
    )
    if p1 is not None and p1[6] is not None and p1[6].shape[0] >= 6:
        ring = p1[6]
        cx, cz = float(ring[:, 0].mean()), float(ring[:, 1].mean())
        minor = float(np.hypot(ring[:, 0] - cx, ring[:, 1] - cz).mean())
        if np.hypot(cx - origin[0], cz - origin[1]) > repass_frac * max(minor, 1e-3):
            origin = (cx, cz)  # re-site the RAY-CAST on the LCFS geometric centroid

    # pass 2: moderate order + light graded ridge at the (re-sited) ray-cast
    # origin, but the pole stays at the current centroid (mask stays central)
    p2 = _read_harmonic_at_origin(
        payload,
        grid,
        table,
        sensors,
        origin,
        pole_ref=pole_ref,
        frac=frac,
        order=order2,
        ridge=ridge,
        sobolev_p=sobolev_p,
    )
    # a tiny plasma can have its LCFS swallowed by the (size-scaled) near-pole
    # mask after re-siting; fall back to the un-re-sited pass-1 read rather than
    # dropping the slice — a coarser valid boundary beats none for the soft prior
    if p2 is None or p2[6] is None:
        return p1 if (p1 is not None and p1[6] is not None) else None
    return p2  # (cfg, coeffs, misfit, psi_tot, axis_psi, boundary_psi, ring)


def build_slice_soft_priors(payload, grid, table, basis, meta, spc):
    """Per-slice :class:`SoftPriors` for the anchored interior solve, or None.

    The annulus boundary anchor recomputes the §2 boundary read INLINE for this
    slice, reads its own boundary/axis flux (frozen per slice), fixes the
    annulus point set there, and targets the read's plasma ψ at those points
    (abs-ψ + rank-1 offset — the gauge-keeping form the measurement selected).
    ``spc["boundary_prior"]`` selects the read: ``"disc"`` (default — the
    staged-disc read, :mod:`imas_ambix.latent.boundary_disc`) or ``"th"`` (the
    frozen toroidal-harmonic read, retained for ablation).  The soft SOL edge,
    the q ≥ 1 sawtooth bound, and the Ip-soft prior are added from ``spc``.
    Returns ``(SoftPriors|None, anchored_bool)``.
    """
    from imas_ambix.latent.boundary_prior import (
        annulus_point_set,
        harmonic_annulus_target,
    )
    from imas_ambix.latent.profile_regularization import q_axis_linear_bound

    sp_kwargs: dict = {}
    anchored = False
    weight = float(spc.get("anchor_weight", 0.0))
    prior_kind = str(spc.get("boundary_prior", "disc"))
    if weight > 0.0 and basis is not None:
        # normalise both reads to (misfit, psi_tot, axis_psi, boundary_psi,
        # target_fn) where target_fn(flat_idx, form) -> plasma-psi target rows
        read_norm = None
        if prior_kind == "disc":
            from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415

            inv = disc_read(payload, grid, table, basis)
            if inv is not None and inv.ring is not None:
                psi_plasma_flat = inv.psi_plasma.ravel()

                def _disc_target(flat_idx, form):
                    if form != "abs-psi":  # pragma: no cover - guarded upstream
                        raise ValueError(
                            "disc boundary prior supplies abs-psi targets only"
                        )
                    return psi_plasma_flat[flat_idx]

                read_norm = (
                    inv.misfit,
                    inv.psi_tot,
                    inv.axis_psi,
                    inv.boundary_psi,
                    _disc_target,
                )
        else:
            read = _harmonic_read_for_slice(payload, grid, table, basis, meta or {})
            if read is not None:
                cfg, coeffs, misfit, psi_tot, axis_psi, boundary_psi, _ring = read
                frozen = {"cfg": cfg, "coeffs": coeffs}

                def _th_target(flat_idx, form):
                    return harmonic_annulus_target(frozen, grid, flat_idx, form)

                read_norm = (misfit, psi_tot, axis_psi, boundary_psi, _th_target)

        if read_norm is not None:
            misfit, psi_tot, axis_psi, boundary_psi, target_fn = read_norm
            ann_idx = annulus_point_set(
                grid,
                psi_carrier=psi_tot,
                axis_psi=axis_psi,
                boundary_psi=boundary_psi,
            )
            ann_rows = np.searchsorted(grid.cells, ann_idx)
            valid = (ann_rows < grid.cells.size) & (
                grid.cells[np.clip(ann_rows, 0, grid.cells.size - 1)] == ann_idx
            )
            ann_rows = ann_rows[valid]
            if ann_rows.size >= 4:
                form = spc.get("anchor_form", "abs-psi")
                # per-slice read uncertainty (heavy-tailed): use the fit misfit,
                # floored, so noisier slices weigh the anchor less
                unc = float(np.sqrt(max(misfit, 1e-6)))
                common = dict(
                    anchor_form=form,
                    anchor_weight=weight,
                    anchor_ann_rows=ann_rows,
                    anchor_robust_clip=spc.get("anchor_robust_clip", 3.0),
                    anchor_uncertainty=unc,
                )
                if form == "grad-psi":
                    # field-matched "virtual magnetics": the read's flux GRADIENT
                    # [dΦ/dR ; dΦ/dZ] at the annulus points (gauge-free; TH only)
                    target = target_fn(grid.cells[ann_rows], "grad-psi")
                    sp_kwargs.update(
                        common, anchor_grad_target=np.asarray(target, dtype=np.float64)
                    )
                else:
                    target = target_fn(grid.cells[ann_rows], "abs-psi")
                    # the rank-1 gauge offset exists for the TH read's genuine
                    # gauge ambiguity (7.9% level bias).  The disc read's psi is
                    # ABSOLUTELY gauged by construction (real currents + coil),
                    # so a free offset would discard exactly the psi-LEVEL
                    # information that sets the boundary size — pin it there.
                    gauge_default = prior_kind != "disc"
                    sp_kwargs.update(
                        common,
                        anchor_psi_target=np.asarray(target, dtype=np.float64),
                        anchor_gauge_offset=bool(
                            spc.get("anchor_gauge_offset", gauge_default)
                        ),
                    )
                anchored = True

    if spc.get("sol_cap", 1.0) > 1.0:
        sp_kwargs.update(
            sol_cap=float(spc["sol_cap"]),
            sol_foot_w=float(spc.get("sol_foot_w", 0.05)),
        )
    if spc.get("q_bound"):
        b_phi0 = float(spc.get("b_phi0", 0.55))
        sp_kwargs.update(
            q_axis_max=q_axis_linear_bound(b_phi0=b_phi0, r0=grid.r0),
            q_weight=float(spc.get("q_weight", 1.0)),
        )
    if spc.get("ip_soft_sigma"):
        sp_kwargs.update(ip_soft_sigma=float(spc["ip_soft_sigma"]))

    if not sp_kwargs:
        return None, anchored
    return SoftPriors(**sp_kwargs), anchored


class _RadiiArgs:
    """Minimal stand-in exposing the fixed adaptive-radii fractions the §2 read
    uses (its argparse defaults), so :func:`_adaptive_radii` needs no full args."""

    mask_frac = 0.5
    exclude_frac = 1.1
    mask_radius = 0.25
    exclude_radius = 0.55


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("closure_gate_eval")


def _apply_calibration(payload: dict, calibration: str) -> None:
    """Correct raw payloads with a frozen static calibration JSON (name-mapped).

    Same contract as ``patch_gate_eval --calibration``: measured' =
    (measured − offset)/gain, scale' = scale/|gain|; channels absent from the
    calibration stay identity.  No-op when ``calibration`` is empty.
    """
    if not calibration:
        return
    cal = json.loads(Path(calibration).read_text())
    by_name = {
        c: (g, o)
        for c, g, o in zip(cal["channels"], cal["gain"], cal["offset"], strict=True)
    }
    names = list(payload["basis"].sensor_channels)
    gain = np.array([by_name.get(c, (1.0, 0.0))[0] for c in names])
    offset = np.array([by_name.get(c, (1.0, 0.0))[1] for c in names])
    for p in payload["payloads"]:
        p.measured[:] = (p.measured - offset) / gain
        p.scale[:] = p.scale / np.abs(gain)


def _shot_passive_sidecar(payload: dict, k: int) -> dict:
    """Build the per-shot rank-k passive eigenmode sidecar.

    Maps the forward operator's ``g_passive`` rows (fwd channel order) onto
    the grid's sensor-channel order by NAME — the same alignment
    :func:`scripts.patch_gate_eval.shot_payloads` applies to the measured
    payloads — then reduces to the top-k whitened sensor-space modes.
    """
    from imas_ambix.gs.operator import build_operator

    table, grid = payload["table"], payload["grid"]
    fwd = build_operator(table)
    _g_sens, channels = grid.sensor_greens(table)
    row_of = {ch: i for i, ch in enumerate(fwd.sensor_channels)}
    g_pass = np.zeros((len(channels), fwd.g_passive.shape[1]))
    for i, ch in enumerate(channels):
        j = row_of.get(ch, -1)
        if j >= 0:
            g_pass[i] = fwd.g_passive[j]
    return build_passive_sidecar(
        table,
        grid,
        g_passive=g_pass,
        sensor_scale=payload["payloads"][0].scale,
        k=k,
    )


ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/plasma-current-priors-hardening")
# non-grid fit modes belong to the force-balance-spine plan — their figures
# land there and never clobber the recorded grid-mode exhibits
FIGURES_SPINE = Path("docs/figures/force-balance-spine")


@dataclass
class ClosureSliceFit:
    """One slice's closure fit: geometry read + fit provenance, or an honest
    masked record (``scored=False``, ``target=None``, ``reason`` set)."""

    shot: int
    t_index: int
    time_s: float
    ip_amperes: float
    scored: bool
    reason: str  # "scored" | "no-converged-candidate" | "cost-exceeds-limit"
    beta0: float | None = None
    alpha: float | None = None
    cost: float | None = None
    converged: bool | None = None
    residual: float | None = None
    target: np.ndarray | None = None  # 14-D geometry read (None if masked)
    saddles: int | None = None
    fit_mode: str = "grid"  # "grid" | "continuous" | "ladder"
    z0: float | None = None  # continuous-mode fitted seed vertical centre [m]
    dof: int | None = None  # ladder-mode profile DOF (n_p + n_f)
    coeffs: list | None = None  # ladder-mode normalised basis coefficients
    psi: np.ndarray | None = None  # retained only when keep_psi=True (figures)
    jphi_flat: np.ndarray | None = None  # converged jφ for temporal warm-starting
    passive_amp: list | None = None  # rank-k passive eigenmode amplitudes [A]


def fit_and_read_slice(
    grid: EquilibriumGrid,
    table,
    payload: SlicePayload,
    *,
    beta0_grid: tuple[float, ...],
    alpha_grid: tuple[float, ...],
    cost_limit: float,
    convergence_limit: float,
    retry_max_iterations: int | None = None,
    fit_mode: str = "grid",
    fit_z0: bool = False,
    n_p: int = 1,
    n_f: int = 1,
    smoothness: float = 0.0,
    nonneg: bool = False,
    passive: dict | None = None,
    passive_ridge: float = 1.0,
    passive_prior: tuple[np.ndarray, float | np.ndarray] | None = None,
    coeff_prior: tuple[np.ndarray, float] | None = None,
    beta_sep_prior: tuple[float, np.ndarray, float] | None = None,
    maxfev: int = 60,
    warm_x0: tuple[float, ...] | None = None,
    warm_jphi: np.ndarray | None = None,
    reseed_axis_r_max: float | None = None,
    keep_psi: bool = False,
    keep_jphi: bool = False,
    basis=None,
    meta: dict | None = None,
    soft_prior_cfg: dict | None = None,
    boundary_read: str = "innermost",
) -> ClosureSliceFit:
    """Fit the profile against ``payload``'s raw magnetics through the GS fixed
    point and read the hardened 14-D geometry from the force-balanced ψ.

    ``fit_mode`` selects the fit machinery: ``"grid"`` (the frozen 5×3
    candidate enumeration — historical default), ``"continuous"`` (bounded
    Nelder–Mead over (β0, α[, z0]) warm-started from ``warm_x0``), or
    ``"ladder"`` (K = n_p + n_f coefficient LSQ-per-Picard-sweep solve).
    ``warm_jphi`` (the previous time slice's converged current) warm-starts
    the Picard chain in the continuous/ladder modes — the first, cheap use of
    temporal coherence.

    ``retry_max_iterations``: if the default-Picard fit masks the slice (no
    candidate meets ``convergence_limit``), retry ONCE with a longer Picard
    budget — a bounded, honest coverage lever (no fallback current is ever
    fabricated).  Returns a :class:`ClosureSliceFit`; masked slices carry
    ``scored=False`` and a ``reason``, never a fabricated readout.

    ``cost_limit`` is an OPTIONAL honesty ceiling on the whitened magnetics
    misfit; the default (``inf``) scores every CONVERGED equilibrium and
    carries the cost as a diagnostic.  This mirrors the free patch arm's
    contract (score all, report misfit) so the two arms' coverage is
    comparable, and it does not conflate "no force-balanced equilibrium"
    (the only thing that masks) with "converged but doesn't fit the magnetics"
    (the expected static coil/calibration misfit — a finding, not a mask).
    """
    payload_kw = dict(
        i_pf=payload.i_pf,
        ip_amperes=payload.ip_amperes,
        measured=payload.measured,
        vacuum_prediction=payload.vacuum,
        sensor_scale=payload.scale,
        sensor_mask=payload.mask,
    )
    base = dict(
        shot=payload.shot,
        t_index=payload.t_index,
        time_s=payload.time_s,
        ip_amperes=payload.ip_amperes,
    )
    z0 = None
    dof = None
    coeffs = None
    sp_slice = None  # per-slice soft priors (ladder branch); reseed reuses it

    if fit_mode == "grid":
        kw = payload_kw | dict(
            beta0_grid=beta0_grid,
            alpha_grid=alpha_grid,
            convergence_limit=convergence_limit,
        )
        fit = fit_profile(grid, table, **kw)
        if fit is None and retry_max_iterations:
            fit = fit_profile(grid, table, max_iterations=retry_max_iterations, **kw)
        beta0 = fit.beta0 if fit else None
        alpha = fit.alpha if fit else None
    elif fit_mode == "continuous":
        kw = payload_kw | dict(
            x0=warm_x0 if warm_x0 is not None else (0.5, 1.5),
            fit_z0=fit_z0,
            convergence_limit=convergence_limit,
            maxfev=maxfev,
        )
        if warm_jphi is not None:
            kw["initial_jphi"] = warm_jphi
        fit = fit_profile_continuous(grid, table, **kw)
        if fit is None and retry_max_iterations:
            fit = fit_profile_continuous(
                grid, table, max_iterations=retry_max_iterations, **kw
            )
        beta0 = fit.beta0 if fit else None
        alpha = fit.alpha if fit else None
        z0 = fit.z0 if fit else None
    elif fit_mode == "ladder":
        kw = payload_kw | dict(n_p=n_p, n_f=n_f, smoothness=smoothness, nonneg=nonneg)
        if passive is not None:
            kw["passive"] = passive
            kw["passive_ridge"] = passive_ridge
        if warm_jphi is not None:
            kw["initial_jphi"] = warm_jphi
        # per-slice soft priors (annulus anchor / SOL foot / q≥1 / Ip-soft):
        # the boundary read enters HERE as a soft prior on the free-boundary solve
        sp_slice = None
        if soft_prior_cfg:
            sp_slice, _anchored = build_slice_soft_priors(
                payload, grid, table, basis, meta or {}, soft_prior_cfg
            )
        # circuit-integrated eddy trajectory as a soft prior on the passive
        # amplitudes (sidecar whitened coordinates); weight 0 leaves the solve
        # byte-identical to the frozen spine
        if passive_prior is not None and passive is not None:
            center, prior_weight = passive_prior
            if bool(np.any(np.asarray(prior_weight) > 0.0)):
                if sp_slice is None:
                    sp_slice = SoftPriors()
                sp_slice.passive_prior_center = np.asarray(center, dtype=np.float64)
                # a per-mode weight VECTOR lets heterogeneous sidecar blocks
                # (vessel modes at 0 + screening modes at w) share one solve
                sp_slice.passive_prior_weight = (
                    float(prior_weight)
                    if np.isscalar(prior_weight)
                    else np.asarray(prior_weight, dtype=np.float64)
                )
        # diffusion-evolved profile-coefficient prediction as a soft temporal-
        # consistency prior on the ladder coefficients; weight 0 leaves the
        # solve byte-identical to the frozen spine
        if coeff_prior is not None:
            c_center, c_weight = coeff_prior
            if c_center is not None and float(c_weight) > 0.0:
                if sp_slice is None:
                    sp_slice = SoftPriors()
                sp_slice.coeff_prior_center = np.asarray(c_center, dtype=np.float64)
                sp_slice.coeff_prior_weight = float(c_weight)
        # ledger-backed βp separation: one whitened moment row pinning the
        # p′-family amplitude (sensitivity ∂βp/∂coeffs, FF′ entries zero) to
        # the external target βp = (βp+li/2)_moment − li_ledger/2; an infinite
        # or non-positive sigma leaves the solve byte-identical
        if beta_sep_prior is not None:
            b_target, b_sens, b_sigma = beta_sep_prior
            if b_sens is not None and np.isfinite(b_target) and float(b_sigma) > 0.0:
                if sp_slice is None:
                    sp_slice = SoftPriors()
                sp_slice.beta_li_target = float(b_target)
                sp_slice.beta_li_sensitivity = np.asarray(b_sens, dtype=np.float64)
                sp_slice.beta_li_sigma = float(b_sigma)
        if sp_slice is not None:
            kw["soft_priors"] = sp_slice
        lf = fit_profile_ladder(grid, table, **kw)
        if (
            not lf.result.converged
            and lf.result.residual > convergence_limit
            and retry_max_iterations
        ):
            lf = fit_profile_ladder(
                grid, table, max_iterations=retry_max_iterations, **kw
            )
        fit = (
            lf
            if (lf.result.converged or lf.result.residual <= convergence_limit)
            else None
        )
        beta0 = alpha = None
        if fit is not None:
            dof = int(lf.dof)
            coeffs = [float(c) for c in lf.coeffs]
    else:  # pragma: no cover — argparse restricts the choices
        raise ValueError(f"unknown fit_mode {fit_mode!r}")

    if fit is None:
        return ClosureSliceFit(
            **base, scored=False, reason="no-converged-candidate", fit_mode=fit_mode
        )
    if fit.cost > cost_limit:
        return ClosureSliceFit(
            **base,
            scored=False,
            reason="cost-exceeds-limit",
            beta0=beta0,
            alpha=alpha,
            cost=fit.cost,
            converged=bool(fit.result.converged),
            residual=float(fit.result.residual),
            fit_mode=fit_mode,
            z0=z0,
            dof=dof,
            coeffs=coeffs,
        )
    _geom = geometry_target_pushout if boundary_read == "pushout" else geometry_target
    target, _, _ = _geom(fit.result.psi, grid)
    reseeded = False
    if (
        fit_mode == "ladder"
        and reseed_axis_r_max is not None
        and np.isfinite(target[0])
        and float(target[0]) > reseed_axis_r_max
        and fit.cost > convergence_limit
    ):
        # Outboard corner attractor: the free-boundary Picard converged onto the
        # non-physical outboard branch (axis R far outboard + high misfit).  A
        # physical confined fixed point exists — re-scout it with the stable
        # free-sign K=2 LSQ from a compact midplane seed, then re-certify the
        # winner ladder there (Tier-3 instrument guides; Tier-2 carries).  Kept
        # only if it lands the axis inboard without materially worsening cost —
        # never a fabricated readout (an honest coverage lever).
        scout = solve_equilibrium_lsq(
            grid,
            table,
            payload.i_pf,
            payload.ip_amperes,
            measured=payload.measured,
            vacuum_prediction=payload.vacuum,
            sensor_scale=payload.scale,
            sensor_mask=payload.mask,
            n_p=1,
            n_f=1,
            seed_width=(0.25, 0.35),
        )
        reseed_kw = dict(
            n_p=n_p,
            n_f=n_f,
            smoothness=smoothness,
            nonneg=nonneg,
            initial_jphi=scout.result.jphi.ravel(),
        )
        if sp_slice is not None:
            reseed_kw["soft_priors"] = sp_slice
        if passive is not None:
            reseed_kw["passive"] = passive
            reseed_kw["passive_ridge"] = passive_ridge
        lf2 = fit_profile_ladder(grid, table, **payload_kw, **reseed_kw)
        if lf2.result.converged or lf2.result.residual <= convergence_limit:
            t2, _, _ = _geom(lf2.result.psi, grid)
            if (
                np.isfinite(t2[0])
                and float(t2[0]) < float(target[0])
                and lf2.cost <= fit.cost * 1.05
            ):
                fit, target, reseeded = lf2, t2, True
                dof = int(lf2.dof)
                coeffs = [float(c) for c in lf2.coeffs]
    return ClosureSliceFit(
        **base,
        scored=True,
        reason="scored-reseeded" if reseeded else "scored",
        beta0=beta0,
        alpha=alpha,
        cost=fit.cost,
        converged=bool(fit.result.converged),
        residual=float(fit.result.residual),
        target=target,
        saddles=count_saddles(fit.result.psi, grid),
        fit_mode=fit_mode,
        z0=z0,
        dof=dof,
        coeffs=coeffs,
        psi=fit.result.psi if keep_psi else None,
        jphi_flat=fit.result.jphi.ravel() if keep_jphi else None,
        passive_amp=(
            [float(a) for a in fit.passive_amplitudes]
            if getattr(fit, "passive_amplitudes", None) is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# per-shot fork pool: the SuperLU-holding grid is inherited copy-on-write by
# the children (unpicklable), so worker state is populated in the PARENT and
# the pool only ships the tiny per-slice index + config (mirrors
# gs_solve_gate_eval's fork pattern; the default start method is fork here).
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(state: dict) -> None:
    _WORKER.update(state)


def _fit_slice_worker(k: int) -> ClosureSliceFit:
    return fit_and_read_slice(
        _WORKER["grid"],
        _WORKER["table"],
        _WORKER["payloads"][k],
        beta0_grid=_WORKER["beta0_grid"],
        alpha_grid=_WORKER["alpha_grid"],
        cost_limit=_WORKER["cost_limit"],
        convergence_limit=_WORKER["convergence_limit"],
        retry_max_iterations=_WORKER["retry_max_iterations"],
        fit_mode=_WORKER.get("fit_mode", "grid"),
        fit_z0=_WORKER.get("fit_z0", False),
        n_p=_WORKER.get("n_p", 1),
        n_f=_WORKER.get("n_f", 1),
        smoothness=_WORKER.get("smoothness", 0.0),
        maxfev=_WORKER.get("maxfev", 60),
    )


def fit_shot(payload: dict, cfg: dict, workers: int) -> list[ClosureSliceFit]:
    """Fit every candidate slice of one shot.

    Grid mode keeps the historical fork pool over slices (candidates are
    independent).  The continuous and ladder modes run the slices SEQUENTIALLY
    in time order instead, warm-starting each slice's search point and Picard
    current from the previous scored slice (temporal coherence as a free
    regularizer); wall-time parallelism then comes from sharding shots across
    SLURM array tasks (``--shots``).
    """
    payloads = payload["payloads"]
    mode = cfg.get("fit_mode", "grid")
    if mode == "grid":
        state = {
            "grid": payload["grid"],
            "table": payload["table"],
            "payloads": payloads,
            **cfg,
        }
        _init_worker(state)
        idx = list(range(len(payloads)))
        if workers > 1 and len(idx) > 1:
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                return list(pool.map(_fit_slice_worker, idx))
        return [_fit_slice_worker(k) for k in idx]

    fits: list[ClosureSliceFit] = []
    warm_x0: tuple[float, ...] | None = None
    warm_jphi: np.ndarray | None = None
    order = np.argsort([p.time_s for p in payloads])
    by_index: dict[int, ClosureSliceFit] = {}
    for k in order:
        f = fit_and_read_slice(
            payload["grid"],
            payload["table"],
            payloads[int(k)],
            beta0_grid=cfg["beta0_grid"],
            alpha_grid=cfg["alpha_grid"],
            cost_limit=cfg["cost_limit"],
            convergence_limit=cfg["convergence_limit"],
            retry_max_iterations=cfg["retry_max_iterations"],
            fit_mode=mode,
            fit_z0=cfg.get("fit_z0", False),
            n_p=cfg.get("n_p", 1),
            n_f=cfg.get("n_f", 1),
            smoothness=cfg.get("smoothness", 0.0),
            nonneg=cfg.get("nonneg", False),
            passive=payload.get("passive"),
            passive_ridge=cfg.get("passive_ridge", 1.0),
            maxfev=cfg.get("maxfev", 60),
            warm_x0=warm_x0,
            warm_jphi=warm_jphi,
            reseed_axis_r_max=cfg.get("reseed_axis_r_max"),
            keep_jphi=True,
            basis=payload.get("basis"),
            meta=cfg.get("frozen_meta"),
            soft_prior_cfg=cfg.get("soft_prior_cfg"),
            boundary_read=cfg.get("boundary_read", "innermost"),
        )
        if f.scored and f.converged:
            warm_jphi = f.jphi_flat
            if mode == "continuous":
                warm_x0 = (
                    (f.beta0, f.alpha, f.z0 or 0.0)
                    if cfg.get("fit_z0", False)
                    else (f.beta0, f.alpha)
                )
        f.jphi_flat = None  # warm-chain only — never serialised
        by_index[int(k)] = f
    fits = [by_index[k] for k in range(len(payloads))]
    return fits


def _grids_for(split: str, args) -> tuple[tuple[float, ...], tuple[float, ...]]:
    b = tuple(float(v) for v in args.beta0_grid.split(",") if v.strip())
    a = tuple(float(v) for v in args.alpha_grid.split(",") if v.strip())
    return b, a


COST_SWEEP_THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf"))


def cost_sweep(
    model: np.ndarray,
    ref: np.ndarray,
    baseline_vec: np.ndarray,
    cost: np.ndarray,
    shot_ids: np.ndarray,
    thresholds: tuple[float, ...] = COST_SWEEP_THRESHOLDS,
) -> list[dict]:
    """The standard accuracy AND coverage report: per cost threshold, the
    subset size and the skill metrics (with CIs when ≥ 2 shots survive).

    Reported in every closure artifact so no headline can hide behind a
    convenient threshold (the Gate-A tail-domination lesson).
    """
    rows: list[dict] = []
    for thr in thresholds:
        m = np.asarray(cost) <= thr
        row: dict = {
            "cost_le": None if np.isinf(thr) else float(thr),
            "n": int(m.sum()),
            "coverage": float(m.mean()) if m.size else 0.0,
        }
        if m.sum() >= 3 and np.unique(shot_ids[m]).size >= 2:
            sc = score(model[m], ref[m], baseline_vec, shot_ids=shot_ids[m])
            sc.pop("axis_errors")
            sc.pop("per_quantity_skill", None)
            sc.pop("per_quantity_skill_ci", None)
            row |= sc
        rows.append(row)
    return rows


def run_gate(args) -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    beta0_grid, alpha_grid = _grids_for(args.split, args)
    cfg = {
        "beta0_grid": beta0_grid,
        "alpha_grid": alpha_grid,
        "cost_limit": args.cost_limit,
        "convergence_limit": args.convergence_limit,
        "retry_max_iterations": args.retry_max_iterations,
        "fit_mode": args.fit_mode,
        "fit_z0": args.fit_z0,
        "n_p": args.n_p,
        "n_f": args.n_f,
        "smoothness": args.smoothness,
        "nonneg": args.ladder_nonneg,
        "passive_k": args.passive_k,
        "passive_ridge": args.passive_ridge,
        "maxfev": args.maxfev,
        "boundary_read": args.boundary_read,
        "reseed_axis_r_max": (
            args.reseed_axis_r
            if (args.reseed_axis_r is not None and args.reseed_axis_r > 0)
            else None
        ),
    }
    # soft-prior config (the §3 ablation ladder): the annulus boundary anchor,
    # the soft SOL edge, the q≥1 sawtooth clamp, and the Ip-soft prior.  The
    # frozen harmonic prior is loaded ONCE and shared across shots (pinned
    # against editable-install drift).
    soft_prior_cfg = {
        "anchor_weight": args.anchor_weight,
        "anchor_form": args.anchor_form,
        "boundary_prior": args.boundary_prior,
        "anchor_robust_clip": (
            args.anchor_robust_clip if args.anchor_robust_clip > 0 else None
        ),
        "sol_cap": args.sol_cap,
        "sol_foot_w": args.sol_foot_w,
        "q_bound": args.q_bound,
        "q_weight": args.q_weight,
        "b_phi0": args.b_phi0,
        "ip_soft_sigma": args.ip_soft_sigma,
    }
    active = (
        args.anchor_weight > 0.0
        or args.sol_cap > 1.0
        or args.q_bound
        or args.ip_soft_sigma
    )
    if active:
        cfg["soft_prior_cfg"] = soft_prior_cfg
        # the frozen prior supplies only the FROZEN CONFIG (order / ridge / kind /
        # pole-inboard fraction); the per-slice harmonic read is recomputed inline
        # from each slice's own magnetics (cohort-independent, leakage-free).
        meta = {}
        if args.frozen_prior:
            _lookup, meta = load_frozen_lookup(args.frozen_prior)
            logger.info("loaded frozen harmonic config: meta=%s", meta)
        cfg["frozen_meta"] = meta
        cfg["config_soft_priors"] = soft_prior_cfg
    cfg_json = {k: v for k, v in cfg.items() if k not in ("frozen_meta",)}
    logger.info("closure gate split=%s cfg=%s", args.split, cfg_json)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    baseline_vec = train_mean_baseline(
        args.n_train, args.n_baseline_shots, args.min_ip_ka
    )
    logger.info(
        "baseline (train-mean) axis: (%.3f, %.3f)", baseline_vec[0], baseline_vec[1]
    )
    if args.split == "train":
        eval_shots = train_shots[
            args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots
        ]
    else:
        eval_shots = held_shots
    if args.shots:
        want = {int(s) for s in args.shots.split(",") if s.strip()}
        eval_shots = [s for s in eval_shots if int(s) in want]
    logger.info("shots: %s", list(eval_shots))

    all_fits: list[ClosureSliceFit] = []
    refs_by_shot: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for s in eval_shots:
        try:
            payload = shot_payloads(
                s,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split=args.split,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", s, exc)
            continue
        if payload is None:
            continue
        _apply_calibration(payload, args.calibration)
        if args.passive_k > 0 and args.fit_mode == "ladder":
            payload["passive"] = _shot_passive_sidecar(payload, args.passive_k)
        refs_by_shot[int(s)] = payload["refs"]
        fits = fit_shot(payload, cfg, args.workers)
        # attach the per-shot referee row index for scored slices
        for k, f in enumerate(fits):
            f._ref = payload["refs"][k]  # type: ignore[attr-defined]
        n_scored = sum(f.scored for f in fits)
        logger.info(
            "shot %s: %d/%d scored (%d masked)",
            s,
            n_scored,
            len(fits),
            len(fits) - n_scored,
        )
        all_fits.extend(fits)
    wall_s = time.perf_counter() - t0

    n_candidate = len(all_fits)
    scored = [f for f in all_fits if f.scored]
    n_scored = len(scored)
    reasons: dict[str, int] = {}
    for f in all_fits:
        reasons[f.reason] = reasons.get(f.reason, 0) + 1
    tag = (
        ("-calibrated" if args.calibration else "")
        + ("-tune" if args.split == "train" else "")
        + (f"-{args.out_suffix}" if args.out_suffix else "")
    )
    if n_scored == 0:
        # write the diagnostic record anyway (coverage + reasons) — a zero-
        # coverage cohort is a reportable finding, not a silent abort
        logger.error(
            "no slices scored on split=%s (coverage 0/%d, reasons=%s)",
            args.split,
            n_candidate,
            reasons,
        )
        (ARTIFACTS / f"closure_gate_eval{tag}.json").write_text(
            json.dumps(
                {
                    "arm": "closure",
                    "split": args.split,
                    "config": cfg_json,
                    "n_scored": 0,
                    "n_candidate": n_candidate,
                    "scored_fraction": 0.0,
                    "mask_reasons": reasons,
                },
                indent=2,
            )
        )
        return 0

    model = np.array([f.target for f in scored])
    ref = np.array([f._ref for f in scored])  # type: ignore[attr-defined]
    shot_ids = np.array([f.shot for f in scored])
    saddles = np.array([f.saddles for f in scored])
    # flat-top proxy: per shot, the single scored slice with the largest |Ip|
    flattop_mask = np.zeros(n_scored, dtype=bool)
    for s in np.unique(shot_ids):
        idx = np.flatnonzero(shot_ids == s)
        ips = np.array([scored[i].ip_amperes for i in idx])
        flattop_mask[idx[int(np.argmax(ips))]] = True

    sc = score(model, ref, baseline_vec, shot_ids=shot_ids)
    axis_errors = sc.pop("axis_errors")
    lcfs_cm = lcfs_offset_cm_stats(model, ref, flattop_mask)
    saddle_stats = saddle_excess_stats(saddles, ref)

    beta0_fit = np.array(
        [np.nan if f.beta0 is None else f.beta0 for f in scored], dtype=np.float64
    )
    alpha_fit = np.array(
        [np.nan if f.alpha is None else f.alpha for f in scored], dtype=np.float64
    )
    z0_fit = np.array(
        [np.nan if f.z0 is None else f.z0 for f in scored], dtype=np.float64
    )
    cost_fit = np.array([f.cost for f in scored], dtype=np.float64)
    conv_frac = float(np.mean([bool(f.converged) for f in scored]))
    sweep = cost_sweep(model, ref, baseline_vec, cost_fit, shot_ids)
    # per-slice GROSS passive current [A] — the circuit-space vectors are
    # ragged across campaigns (passive circuit counts differ per geometry)
    pass_amp = None
    if args.passive_k > 0 and all(f.passive_amp is not None for f in scored):
        pass_amp = np.array(
            [np.abs(f.passive_amp).sum() for f in scored], dtype=np.float64
        )

    result = {
        "arm": "closure",
        "split": args.split,
        "config": cfg_json
        | {
            "n_train": args.n_train,
            "n_heldout": args.n_heldout,
            "nr": args.nr,
            "nz": args.nz,
            "min_ip_ka": args.min_ip_ka,
        },
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "n_scored": n_scored,
        "n_candidate": n_candidate,
        "scored_fraction": float(n_scored / max(n_candidate, 1)),
        "mask_reasons": reasons,
        "strict_converged_fraction_of_scored": conv_frac,
        # attractor instrument (§3.2 gate): the reseed-lever trigger rate and the
        # outboard-axis rate — the metric to watch collapse to ≈0 once the soft
        # anchor supplies the missing boundary constraint (reseed/scout retire).
        "reseed_trigger_fraction": float(
            np.mean([f.reason == "scored-reseeded" for f in scored])
        ),
        "axis_outboard_fraction": float(
            np.mean(
                [bool(np.isfinite(f.target[0]) and f.target[0] > 1.1) for f in scored]
            )
        ),
        "beta0_median": (
            float(np.nanmedian(beta0_fit)) if np.isfinite(beta0_fit).any() else None
        ),
        "alpha_median": (
            float(np.nanmedian(alpha_fit)) if np.isfinite(alpha_fit).any() else None
        ),
        "z0_median": (
            float(np.nanmedian(z0_fit)) if np.isfinite(z0_fit).any() else None
        ),
        "cost_median_scored": float(np.median(cost_fit)),
        "cost_le_3_fraction": float(np.mean(cost_fit <= 3.0)),
        "cost_sweep": sweep,
        "passive_gross_current_over_ip_median": (
            float(np.median(pass_amp / np.abs([f.ip_amperes for f in scored])))
            if pass_amp is not None
            else None
        ),
        "wall_s": wall_s,
        **sc,
        **lcfs_cm,
        **saddle_stats,
    }
    tag = (
        ("-calibrated" if args.calibration else "")
        + ("-tune" if args.split == "train" else "")
        + (f"-{args.out_suffix}" if args.out_suffix else "")
    )
    (ARTIFACTS / f"closure_gate_eval{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    extra_arrays: dict[str, np.ndarray] = {}
    if args.fit_mode == "ladder":
        extra_arrays["coeffs"] = np.array([f.coeffs for f in scored], dtype=np.float64)
    if pass_amp is not None:
        extra_arrays["passive_gross_current"] = pass_amp
    np.savez(
        ARTIFACTS / f"closure_gate_eval{tag}_arrays.npz",
        model=model,
        ref=ref,
        baseline=np.tile(baseline_vec, (n_scored, 1)),
        axis_errors=axis_errors,
        shot_ids=shot_ids,
        flattop_mask=flattop_mask,
        saddles=saddles,
        beta0=beta0_fit,
        alpha=alpha_fit,
        z0=z0_fit,
        cost=np.array([f.cost for f in scored], dtype=np.float64),
        **extra_arrays,
    )
    logger.info(
        "[closure %s] scored %d/%d (%.0f%%) axis_skill=%.3f ci=%s lcfs_skill=%s "
        "xpt_skill=%s axis_median=%.3f m lcfs_cm(all/flat)=%s/%s saddle_excess_med=%s "
        "(%.0f s)",
        args.split,
        n_scored,
        n_candidate,
        100.0 * n_scored / max(n_candidate, 1),
        sc["axis_skill"],
        sc.get("axis_skill_ci"),
        sc["lcfs_skill"],
        sc["xpoint_set_skill"],
        sc["axis_error_median_m"],
        lcfs_cm["lcfs_offset_median_cm_all"],
        lcfs_cm["lcfs_offset_median_cm_flattop"],
        saddle_stats["saddle_excess_median"],
        wall_s,
    )
    return 0


# ---------------------------------------------------------------------------
# Figures (imas-ink): closure-arm ψ vs EFIT for the held-out shots + the
# fitted jφ(ψ_N) profile family.
# ---------------------------------------------------------------------------
def run_figures(args) -> int:
    import matplotlib.pyplot as plt
    from imas_ink.figures import equilibrium_figure_mpl

    from scripts.patch_flux_map_report import (
        _closed_contour_about,
        _efit_slice,
        _fig_to_rgba,
        _machine_geometry,
        _our_slice,
        select_slices,
    )

    fig_dir = FIGURES if args.fit_mode == "grid" else FIGURES_SPINE
    fig_suffix = "" if args.fit_mode == "grid" else f"-{args.fit_mode}"
    if args.out_suffix:
        fig_suffix += f"-{args.out_suffix}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    beta0_grid, alpha_grid = _grids_for("eval", args)
    _, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    logger.info("figures: held-out shots %s", list(held_shots))

    flux_panels = {"rampup": {}, "flattop": {}}
    profile_rows = []  # (regime, beta0, alpha, psi_n grid, shape)
    for shot in held_shots:
        try:
            payload = shot_payloads(
                shot,
                nr=args.nr,
                nz=args.nz,
                max_slices=args.max_slices_per_shot,
                min_ip_ka=args.min_ip_ka,
                split="eval",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed to load: %s", shot, exc)
            continue
        if payload is None:
            continue
        _apply_calibration(payload, args.calibration)
        grid, table, payloads = payload["grid"], payload["table"], payload["payloads"]
        picks = select_slices(payloads, shot)
        if not picks:
            continue
        geom = _machine_geometry(grid, table)
        for kind, k, efit in picks:
            p = payloads[k]
            fit = fit_and_read_slice(
                grid,
                table,
                p,
                beta0_grid=beta0_grid,
                alpha_grid=alpha_grid,
                cost_limit=args.cost_limit,
                convergence_limit=args.convergence_limit,
                retry_max_iterations=args.retry_max_iterations,
                fit_mode=args.fit_mode,
                fit_z0=args.fit_z0,
                n_p=args.n_p,
                n_f=args.n_f,
                smoothness=args.smoothness,
                nonneg=args.ladder_nonneg,
                maxfev=args.maxfev,
                keep_psi=True,
            )
            if not fit.scored:
                logger.info("%d %-7s masked (%s) — skip panel", shot, kind, fit.reason)
                continue
            psi2d = fit.psi
            target, psi_ax, psi_b = geometry_target(psi2d, grid)
            axis_rz = (float(target[0]), float(target[1]))
            our_lcfs = _closed_contour_about(grid.rg, grid.zg, psi2d, psi_b, *axis_rz)
            sl = _our_slice(
                psi2d, grid, target, psi_ax, psi_b, p.ip_amperes, p.time_s, our_lcfs
            )
            fig, _ax = equilibrium_figure_mpl(
                sl,
                geom,
                reference_slice=_efit_slice(efit),
                reference_name="EFIT",
                figsize=(4.6, 6.0),
                show_probes=False,
                show_flux_loops=False,
            )
            if fit.beta0 is not None:
                fit_label = f"β0={fit.beta0:.2f} α={fit.alpha:.1f}"
            else:
                fit_label = f"K={fit.dof} LSQ  cost={fit.cost:.2f}"
            fig.suptitle(
                f"{shot} t={p.time_s:.3f}s ({kind})  {fit_label}",
                fontsize=9,
            )
            flux_panels[kind][int(shot)] = _fig_to_rgba(fig)
            plt.close(fig)
            psi_n = np.linspace(0.0, 1.0, 60)
            if fit.beta0 is not None:
                shape = profile_jphi_shape(
                    psi_n,
                    np.full_like(psi_n, grid.r0),
                    r0=grid.r0,
                    beta0=fit.beta0,
                    alpha=fit.alpha,
                )
            else:
                from imas_ambix.latent.gs_solve import profile_basis

                shape = profile_basis(
                    psi_n,
                    np.full_like(psi_n, grid.r0),
                    r0=grid.r0,
                    n_p=args.n_p,
                    n_f=args.n_f,
                ) @ np.asarray(fit.coeffs)
            profile_rows.append((kind, fit.beta0, fit.alpha, psi_n, shape))
            logger.info("%d %-7s %s rendered", shot, kind, fit_label)

    fig_paths = []
    for regime in ("rampup", "flattop"):
        shots = sorted(flux_panels[regime])
        if not shots:
            continue
        ncol = min(4, len(shots))
        nrow = int(np.ceil(len(shots) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 5.4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, s in zip(axes, shots, strict=False):
            ax.imshow(flux_panels[regime][s])
        fig.suptitle(
            f"Closure-arm ψ(R,Z) vs EFIT — {regime} "
            f"(primary: profile-parametrised GS solve, faint: EFIT)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = fig_dir / f"fig-closure-flux-maps-{regime}{fig_suffix}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        fig_paths.append(str(out))
        logger.info("wrote %s (%d panels)", out, len(shots))

    # fitted jφ(ψ_N) profile family + β0/α distribution
    if profile_rows:
        fig, (axc, axd) = plt.subplots(1, 2, figsize=(11.0, 4.6))
        cmap = {"rampup": "#1b7837", "flattop": "#d95f02"}
        for kind, _b0, _al, psi_n, shape in profile_rows:
            axc.plot(psi_n, shape, color=cmap[kind], alpha=0.5, lw=1.2)
        axc.set_xlabel("ψ_N")
        axc.set_ylabel("jφ shape at R=R0  (β0 + (1−β0))·(1−ψ_N)^α")
        axc.set_title("Fitted jφ(ψ_N) profiles (per held-out slice)")
        for kind in ("rampup", "flattop"):
            axc.plot([], [], color=cmap[kind], label=kind)
        axc.legend(fontsize=8)
        for kind in ("rampup", "flattop"):
            b0 = [r[1] for r in profile_rows if r[0] == kind and r[1] is not None]
            al = [r[2] for r in profile_rows if r[0] == kind and r[2] is not None]
            axd.scatter(b0, al, color=cmap[kind], label=kind, s=40, alpha=0.7)
        axd.set_xlabel("β0 (pressure/FF′ split)")
        axd.set_ylabel("α (peakedness)")
        axd.set_title("Fitted (β0, α) distribution")
        axd.legend(fontsize=8)
        fig.tight_layout()
        out = fig_dir / f"fig-closure-profile-fits{fig_suffix}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        fig_paths.append(str(out))
        logger.info("wrote %s (%d profiles)", out, len(profile_rows))

    (ARTIFACTS / f"closure_figures{fig_suffix}.json").write_text(
        json.dumps({"figures": fig_paths, "n_profiles": len(profile_rows)}, indent=2)
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices-per-shot", type=int, default=20)
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--n-tune-shots", type=int, default=4)
    ap.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=("eval", "train"),
        help="eval = held-out gate; train = the 4-shot leakage-free tune cohort",
    )
    ap.add_argument(
        "--shots",
        type=str,
        default="",
        help="comma list to restrict the run (sbatch-array sharding); '' = all",
    )
    # frozen physics-motivated config (confirmed on --split train, applied
    # unchanged to --split eval)
    ap.add_argument("--beta0-grid", type=str, default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--alpha-grid", type=str, default="1.0,1.5,2.0")
    ap.add_argument(
        "--cost-limit",
        type=float,
        default=float("inf"),
        help="optional whitened-misfit ceiling; default inf scores every "
        "converged equilibrium (cost carried as a diagnostic)",
    )
    ap.add_argument("--convergence-limit", type=float, default=5e-3)
    ap.add_argument("--retry-max-iterations", type=int, default=160)
    ap.add_argument(
        "--fit-mode",
        type=str,
        default="grid",
        choices=("grid", "continuous", "ladder"),
        help="grid = frozen 5x3 candidate enumeration (historical default); "
        "continuous = bounded Nelder-Mead over (beta0, alpha[, z0]), "
        "warm-started along the time axis; ladder = K = n_p + n_f coefficient "
        "LSQ-per-Picard-sweep solve (the profile-DOF ladder)",
    )
    ap.add_argument(
        "--fit-z0",
        action="store_true",
        help="continuous mode: add the vertical seed-centre DOF",
    )
    ap.add_argument("--n-p", type=int, default=1, help="ladder: p' basis size")
    ap.add_argument("--n-f", type=int, default=1, help="ladder: FF' basis size")
    ap.add_argument(
        "--smoothness",
        type=float,
        default=0.0,
        help="ladder: second-difference coefficient smoothness ridge weight",
    )
    ap.add_argument(
        "--ladder-nonneg",
        action="store_true",
        help="ladder: non-negative monomial basis + bounded solve — "
        "jphi*sign(Ip) >= 0 by construction (R1 at the profile level)",
    )
    ap.add_argument(
        "--passive-k",
        type=int,
        default=0,
        help="ladder: rank of the g_passive eigenmode sidecar (0 = OFF, the "
        "default — P5 discipline: measured before load-bearing)",
    )
    ap.add_argument(
        "--passive-ridge",
        type=float,
        default=1.0,
        help="ladder: relative ridge weight on the passive mode amplitudes",
    )
    ap.add_argument(
        "--maxfev",
        type=int,
        default=60,
        help="continuous: objective-evaluation budget per slice",
    )
    ap.add_argument(
        "--reseed-axis-r",
        type=float,
        default=None,
        help="ladder: if a converged slice reads its axis outboard of this R [m] "
        "with cost above the convergence limit, re-scout the confined branch "
        "from a compact seed and re-certify there (outboard-attractor coverage "
        "lever; None = OFF, the default). Physical MAST confined axes sit "
        "R < 1.1 m, so ~1.4 is a safe outboard-corner trigger.",
    )
    # --- §3 soft-prior interior solve: the boundary read as a SOFT prior ---
    ap.add_argument(
        "--frozen-prior",
        type=str,
        default="",
        help="path to the frozen harmonic prior NPZ/JSON "
        "(imas_ambix/latent/artifacts/patch_gate/harmonic_prior_frozen[-tune]); "
        "the annulus anchor loads its per-slice coeffs, pinned against drift",
    )
    ap.add_argument(
        "--anchor-weight",
        type=float,
        default=0.0,
        help="annulus ψ-consistency anchor weight (0 = OFF = free-boundary A0); "
        "the boundary read enters as a SOFT prior at this weight (abs-ψ + rank-1 "
        "gauge offset — the gauge-keeping form the measurement selected)",
    )
    ap.add_argument(
        "--boundary-prior",
        type=str,
        default="disc",
        choices=("disc", "th"),
        help="§2 boundary read feeding the annulus anchor: 'disc' (staged "
        "uniform-disc + gated quadrupole — the cohort-validated default) or "
        "'th' (frozen toroidal-harmonic read, retained for ablation)",
    )
    ap.add_argument(
        "--anchor-form",
        type=str,
        default="abs-psi",
        choices=("abs-psi", "grad-psi"),
        help="annulus anchor form: 'abs-psi' (ψ + rank-1 gauge offset, the "
        "measurement-selected form) or 'grad-psi' (gauge-free flux-gradient / "
        "field matching — the densified near-plasma 'virtual magnetics' arm)",
    )
    ap.add_argument(
        "--anchor-robust-clip",
        type=float,
        default=3.0,
        help="Huber clip (robust-σ) down-weighting outlier annulus points "
        "(heavy-tailed consistency); None-like <=0 disables",
    )
    ap.add_argument(
        "--sol-cap",
        type=float,
        default=1.0,
        help="soft SOL edge cap ψ_N (1.0 = hard ψ_N<1 mask = A1; >1 e.g. 1.1 "
        "admits public-SOL current through a C¹ decay foot = A2)",
    )
    ap.add_argument(
        "--sol-foot-w",
        type=float,
        default=0.05,
        help="SOL decay-foot width (dimensionless)",
    )
    ap.add_argument(
        "--q-bound",
        action="store_true",
        help="add the q≥1 sawtooth clamp (soft upper bound on on-axis jφ)",
    )
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument(
        "--b-phi0",
        type=float,
        default=0.55,
        help="vacuum toroidal field [T] at R0 for the q≥1 bound (MAST nominal)",
    )
    ap.add_argument(
        "--ip-soft-sigma",
        type=float,
        default=0.0,
        help="if >0, use the SOFT Ip prior at this fractional σ instead of the "
        "hard Rogowski KKT (default 0 = hard anchor)",
    )
    ap.add_argument(
        "--boundary-read",
        type=str,
        default="innermost",
        choices=("innermost", "pushout"),
        help="interior LCFS scoring read: 'innermost' (X-point/limiter-contact, "
        "historical) or 'pushout' (outermost axis-enclosing ring, lcfs_contour — "
        "the landed reader §2 uses; fixes the saddle-lock under-sizing)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--figures", action="store_true", help="render figures only")
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="artifact-name suffix (shot-sharded runs merge offline)",
    )
    ap.add_argument(
        "--calibration",
        type=str,
        default="",
        help="frozen static calibration JSON applied to raw payloads",
    )
    args = ap.parse_args()

    if args.figures:
        return run_figures(args)
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
