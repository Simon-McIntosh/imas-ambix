"""Standalone GS force-balance RESIDUAL monitor.

This is the regularised INVERSE of the forward operator
(:mod:`imas_ambix.gs.operator`), which gives the geometry-only forward map

    pred = G_pf · I_pf  +  G_plasma · c_plasma  +  G_passive · c_passive

with ``I_pf`` KNOWN (raw amc coil currents).  This module **inverts** it: given
the raw ``amb`` magnetics at one slice and the known PF term, solve for the
INFERRED ``c_plasma`` / ``c_passive`` against the TRUSTWORTHY sensor target, then
form the GS force-balance residual ``r`` = the fractional reconstruction misfit
that remains after the best GS-constrained current model.  A small ``r`` means
the slice is consistent with a smooth, low-DOF axisymmetric current distribution
in force balance; a large ``r`` means force balance has BROKEN — the transient /
off-normal signature a departure monitor reads off this residual.

The residual is computed directly from raw signals plus the forward operator,
with no ``engine.py`` retrain, so it stands on its own before any joint
grounding head (:mod:`imas_ambix.gs.grounding`) is trained against it.

Why a regulariser is MANDATORY
------------------------------
The forward map is intentionally OVER-COMPLETE — ~84 plasma + 78..144 passive
columns vs ~77 trustworthy sensors — so a pointwise-free / unconstrained solve
drives ``r → 0`` and measures NOTHING.  But the nominal column count is
MISLEADING: Green's functions from clustered filaments are highly correlated
(smooth kernels), so the *effective* rank of the row-scaled design is tiny —
measured ~5 (plasma), ~2 (passive), ~6 (combined) at 99 % energy with condition
numbers up to 1e19.  The instrument resolution therefore comes from TWO explicit
restrictions, swept together and reported as a 2-D frontier:

1. **profile-DOF on the plasma block** — ``c_plasma = B_poly · θ`` with a 2-D
   polynomial basis in dimensionless ``(R−R0)/a, Z/a`` over the plasma nodes.
   This is the current-space translation of the ``p′/FF′`` profile-DOF under a
   current-distribution Green's representation:
   order-1 = {1, ρ_R, ρ_Z} (3 DOF), order-2 = +{ρ_R², ρ_Rρ_Z, ρ_Z²} (6 DOF),
   order-4 ≈ 15 DOF.  The representation is not re-decided here; this is its
   faithful current-space mapping.
2. **a low-rank passive basis + λ ridge** — the passive block (78 ≈ 77 sensors)
   is the REAL ``r → 0`` driver: a free 78-column passive basis alone can nearly
   fit 77 sensors, so restricting plasma while passive runs free still gives
   trivial ``r``.  The passive amplitudes are therefore restricted to a
   truncated-SVD low-rank basis (``passive_rank``) AND penalised by λ.

λ MUST act on the COLUMN-NORMALISED design — each block's columns whitened to
unit norm on the trustworthy rows, λ applied, then rescaled back.  Against
un-normalised columns a single scalar λ is incoherent, because it then penalises
each block in that block's own arbitrary units.  The frontier ``λ × profile-DOF``
defines the instrument's resolution: the min plasma-DOF that gives a NON-trivial
quiescent ``r`` (the floor) and the max DOF before ``r → 0`` (the ceiling).

The residual is a RECONSTRUCTION MISFIT under the GS-constrained (smooth,
low-DOF) current model — it is NOT an explicit ``Δ*ψ`` grid residual.

SI and framing
--------------
The operator works in raw SI (Wb / T / A).  The residual is reported in a
**fractional / dimensionless** form ``||W(pred−raw)|| / ||W·raw||`` with ``W`` a
per-sensor robust scale.  Fractional is required twice over: for dimensional
coherence, because one norm mixes 76 B-probe Tesla rows with a flux-loop Wb row;
and to keep the residual from reconstructing the very labels a deconfounding
study holds out, since a raw-Tesla residual carries the Ip²/ne axes directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from imas_ambix.gs.operator import ForwardOperator

# --- the swept frontier grid --------------------------------------------

LAMBDA_GRID: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1)
"""The λ axis of the 2-D frontier."""

PROFILE_DOF_GRID: tuple[int, ...] = (1, 2, 4)
"""The profile-DOF axis: polynomial order 1 / 2 / 4 (3 / 6 / 15 plasma basis
DOF).  These are the current-space translation of the p′/FF′ DOF."""

_PASSIVE_RANK_DEFAULT = 4
"""Default truncated-SVD rank for the passive nuisance basis.

Sized from the measured passive effective rank (~2 at 99 % energy, ~4 with
headroom).  A free 78-column passive block trivially fits 77 sensors; the
low-rank restriction is what keeps the residual non-trivial.  Reported in the
frontier artifact so the choice is inspectable."""

_RIDGE_FLOOR = 1e-9
"""Numerical ridge floor so the λ=0 corner is still solvable (the design is
near-singular — condition numbers up to 1e19 — so an exact λ=0 is ill-posed;
this floor makes λ=0 the *minimally* regularised corner, not a crash)."""


# --- profile-DOF plasma basis -----------------------------------------


def plasma_poly_basis(
    plasma_rz: np.ndarray, order: int, r0: float, minor_radius: float
) -> np.ndarray:
    """2-D polynomial profile-DOF basis ``B`` over the plasma nodes.

    ``c_plasma = B · θ`` restricts the inferred ``jφ(R, Z)`` to a low-order
    polynomial in dimensionless ``ρ_R = (R−R0)/a``, ``ρ_Z = Z/a`` — the
    current-space translation of the p′/FF′ profile-DOF restriction:

    * order-1 → {1, ρ_R, ρ_Z}                              (3 DOF)
    * order-2 → + {ρ_R², ρ_Rρ_Z, ρ_Z²}                     (6 DOF)
    * order-4 → + cubic + quartic terms                    (15 DOF)

    Returns ``B`` of shape ``(n_plasma_node, n_dof)``.
    """
    rz = np.asarray(plasma_rz, dtype=np.float64)
    if rz.size == 0:
        return np.zeros((0, 1), dtype=np.float64)
    a = float(minor_radius) if minor_radius else 1.0
    rho_r = (rz[:, 0] - float(r0)) / a
    rho_z = rz[:, 1] / a
    cols: list[np.ndarray] = [np.ones_like(rho_r)]
    if order >= 1:
        cols += [rho_r, rho_z]
    if order >= 2:
        cols += [rho_r * rho_r, rho_r * rho_z, rho_z * rho_z]
    if order >= 4:
        cols += [
            rho_r**3,
            rho_r**2 * rho_z,
            rho_r * rho_z**2,
            rho_z**3,
            rho_r**4,
            rho_r**3 * rho_z,
            rho_r**2 * rho_z**2,
            rho_r * rho_z**3,
            rho_z**4,
        ]
    return np.column_stack(cols)


def passive_lowrank_basis(g_passive_rows: np.ndarray, rank: int) -> np.ndarray:
    """Truncated-SVD low-rank basis for the passive nuisance amplitudes.

    Given the row-restricted passive design ``G_passive[trust]`` of shape
    ``(n_sensor, n_passive)``, return the leading ``rank`` right singular
    vectors ``V_r`` (shape ``(n_passive, rank)``) so ``c_passive = V_r · ψ``.
    This caps the passive DOF (the real ``r → 0`` driver) at ``rank`` modes,
    well below the ~78 nominal columns (effective rank ~2).
    """
    g = np.asarray(g_passive_rows, dtype=np.float64)
    if g.size == 0 or rank <= 0:
        return np.zeros((g.shape[1] if g.ndim == 2 else 0, 0), dtype=np.float64)
    _, _, vt = np.linalg.svd(g, full_matrices=False)
    r = min(int(rank), vt.shape[0])
    return np.ascontiguousarray(vt[:r].T)


# --- the regularised inverse solve ------------------------------------


@dataclass
class TrustTarget:
    """The trustworthy comparison target for one campaign operator.

    Filters the operator's predicted rows to the 76 B-probes + the 1 cleanly
    mapped flux loop, EXCLUDING both the unmatched ``fl_p2*`` loops (absent from
    ``G``) and the ~20 flagged non-unique ``fl_cc*``/``fl_p*`` loops (predicted
    at a placeholder silop index but with an AMBIGUOUS amb identity).  Comparing
    the residual against the flagged/excluded rows compares against garbage.
    """

    rows: np.ndarray  # int indices into operator.sensor_channels
    channels: list[str]
    kinds: list[str]

    @property
    def n(self) -> int:
        return int(self.rows.size)


def trustworthy_target(
    operator: ForwardOperator, available_channels: set[str] | None = None
) -> TrustTarget:
    """Build the trustworthy sensor target (76 B-probes + 1 clean flux loop).

    ``available_channels`` (optional) restricts the target to channels actually
    present for a given shot — a few B-probes are absent in some shots/campaigns
    (one dead channel must not nullify the whole row), so the per-shot loader
    passes the present-and-finite channel set and the design rows are sub-selected
    consistently.  The trustworthy RULE (B-probe OR cleanly-mapped flux loop,
    never a flagged/excluded loop) is unchanged.
    """
    flagged = set(operator.flagged_channels)
    rows: list[int] = []
    chans: list[str] = []
    kinds: list[str] = []
    for i, (c, k) in enumerate(
        zip(operator.sensor_channels, operator.sensor_kind, strict=True)
    ):
        is_trust = k == "b_probe" or (k == "flux_loop" and c not in flagged)
        if not is_trust:
            continue
        if available_channels is not None and c not in available_channels:
            continue
        rows.append(i)
        chans.append(c)
        kinds.append(k)
    return TrustTarget(rows=np.array(rows, dtype=int), channels=chans, kinds=kinds)


def robust_sensor_scale(
    raw_trust: np.ndarray,
    quiescent_mask: np.ndarray | None = None,
    floor: float = 1e-9,
) -> np.ndarray:
    """Per-sensor robust scale ``W`` fit on QUIESCENT slices.

    ``raw_trust`` : ``(T, n_trust)`` raw amb at the trustworthy sensors.
    ``quiescent_mask`` : ``(T,)`` bool (``True`` = quiescent); if ``None`` all
    finite slices are used.  The scale is the per-sensor std over the quiescent
    slices so transients do not inflate it.  Returned with a floor so degenerate
    (dead) channels do not blow up the whitening.  This SAME scale enters both
    the weighted lstsq and the fractional residual (and the baseline) — a single
    coherent per-sensor scale.
    """
    x = np.asarray(raw_trust, dtype=np.float64)
    finite_rows = np.isfinite(x).all(axis=1)
    use = finite_rows & quiescent_mask if quiescent_mask is not None else finite_rows
    if not use.any():
        use = finite_rows
    if not use.any():
        return np.full(x.shape[1], floor)
    scale: np.ndarray = np.maximum(np.std(x[use], axis=0), floor)
    return scale


@dataclass
class InverseSolver:
    """Regularised inverse solve for one campaign operator + a fixed config.

    Holds the row-restricted, profile-DOF-reduced, column-normalised design
    blocks for a fixed ``(profile_order, passive_rank)`` so the per-slice solve
    is a tiny normal-equation solve.  λ is applied per-solve (it is the swept
    axis).  The plasma + passive amplitudes are recovered in physical-current
    space (A) for the near-vacuum sanity check.
    """

    operator: ForwardOperator
    target: TrustTarget
    sensor_scale: np.ndarray  # (n_trust,) per-sensor robust scale W
    profile_order: int
    passive_rank: int

    # built design (column-normalised, row-whitened) — filled in __post_init__
    _design: np.ndarray = field(init=False, repr=False)
    _col_norm: np.ndarray = field(init=False, repr=False)
    _b_plasma: np.ndarray = field(init=False, repr=False)
    _v_passive: np.ndarray = field(init=False, repr=False)
    _n_plasma_dof: int = field(init=False, repr=False)
    _penalty: np.ndarray = field(init=False, repr=False)
    _design_ss: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rows = self.target.rows
        w = (1.0 / self.sensor_scale)[:, None]  # row whitening
        g_pl = self.operator.g_plasma[rows]
        g_pa = self.operator.g_passive[rows]
        # profile-DOF plasma basis (current-space translation of p'/FF' DOF)
        self._b_plasma = plasma_poly_basis(
            self.operator.plasma_rz,
            self.profile_order,
            self.operator.r0,
            self.operator.minor_radius,
        )
        # low-rank passive nuisance basis (sized from the measured eff-rank)
        self._v_passive = passive_lowrank_basis(w * g_pa, self.passive_rank)
        # reduced, row-whitened design blocks
        a_pl = (w * g_pl) @ self._b_plasma  # (n_trust, n_plasma_dof)
        a_pa = (w * g_pa) @ self._v_passive  # (n_trust, passive_rank)
        self._n_plasma_dof = a_pl.shape[1]
        design = np.hstack([a_pl, a_pa]) if a_pa.size else a_pl
        # column-normalise (whiten columns to unit norm on the trust rows) so a
        # single scalar λ penalises every direction coherently; rescale on apply.
        col_norm = np.linalg.norm(design, axis=0)
        col_norm = np.where(col_norm > 0, col_norm, 1.0)
        self._col_norm = col_norm
        self._design = design / col_norm

        # PHYSICAL-amplitude Tikhonov penalty (the actual GS SOFT PRIOR): the
        # plasma + passive blocks are 99 %-COLLINEAR (a plasma poly mode and a
        # passive SVD mode produce near-identical smooth fields), so an
        # unconstrained solve trades HUGE canceling currents between them (~25 kA
        # of spurious plasma current at near-vacuum).  A column-normalised ridge
        # does NOT suppress this (it penalises coefficients, not physical current
        # amplitude).  The physical penalty M = blkdiag(BᵀB, VᵀV) acts on
        # ||c_plasma||² + ||c_passive||² (current-space L2) — directly enforcing
        # near-vacuum c_plasma≈0 and breaking the collinear cancellation.  In the
        # column-normalised coefficient space the penalty is N⁻¹·M·N⁻¹ with
        # N = diag(col_norm).  Scaled by the design's mean diagonal so λ is
        # dimensionless and comparable across cells.
        btb = self._b_plasma.T @ self._b_plasma
        vtv = (
            self._v_passive.T @ self._v_passive
            if self._v_passive.size
            else np.zeros((0, 0))
        )
        m_phys = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
        npl = self._n_plasma_dof
        m_phys[:npl, :npl] = btb
        if vtv.size:
            m_phys[npl:, npl:] = vtv
        inv_n = 1.0 / col_norm
        self._penalty = (inv_n[:, None] * m_phys) * inv_n[None, :]
        # normalise the penalty so λ is comparable across cells (trace match to
        # the data term) — λ is then the soft-prior strength relative to the fit.
        ata = self._design.T @ self._design
        ptr = float(np.trace(self._penalty))
        atr = float(np.trace(ata))
        self._penalty = self._penalty * (atr / ptr) if ptr > 0 else self._penalty
        self._design_ss = atr

    def n_total_dof(self) -> int:
        return int(self._design.shape[1])

    def n_plasma_dof(self) -> int:
        return int(self._n_plasma_dof)

    def solve(
        self, raw_trust_slice: np.ndarray, i_pf: np.ndarray, lam: float
    ) -> dict[str, Any]:
        """Solve one slice for the reduced amplitudes + the fractional residual.

        ``raw_trust_slice`` : ``(n_trust,)`` raw amb at the trustworthy sensors.
        ``i_pf`` : KNOWN per-coil PF currents [A] (from
        :meth:`ForwardOperator.assemble_pf_currents`).  ``lam`` : the λ on the
        column-normalised design.

        Returns ``{residual_frac, residual_abs, theta_plasma, psi_passive,
        c_plasma, c_passive, pred_trust}``.  The plasma/passive amplitudes are
        rescaled back to physical-current space (A) for the sanity check.
        """
        rows = self.target.rows
        w = 1.0 / self.sensor_scale
        # subtract the KNOWN PF term — solve only for the INFERRED currents
        pf_trust = (self.operator.g_pf[rows] @ np.asarray(i_pf, dtype=np.float64))
        b = (np.asarray(raw_trust_slice, dtype=np.float64) - pf_trust) * w
        a = self._design
        n_par = a.shape[1]
        ata = a.T @ a
        # PHYSICAL-amplitude Tikhonov (the GS soft prior): λ·M on
        # ||c_plasma||²+||c_passive||² (current-space L2), + a tiny ridge floor so
        # the λ=0 corner is still solvable (the blocks are highly collinear).
        floor = _RIDGE_FLOOR * np.trace(ata) / max(n_par, 1) * np.eye(n_par)
        coef_n = np.linalg.solve(ata + lam * self._penalty + floor, a.T @ b)
        # un-normalise the columns
        coef = coef_n / self._col_norm
        theta = coef[: self._n_plasma_dof]
        psi = coef[self._n_plasma_dof :]
        c_plasma = self._b_plasma @ theta
        c_passive = self._v_passive @ psi if psi.size else np.zeros(0)
        # reconstruct the prediction at the trustworthy rows (whitened)
        pred_w = a @ coef_n + pf_trust * w
        raw_w = np.asarray(raw_trust_slice, dtype=np.float64) * w
        resid_w = pred_w - raw_w
        denom = float(np.linalg.norm(raw_w))
        resid_abs = float(np.linalg.norm(resid_w))
        resid_frac = resid_abs / denom if denom > 0 else float("nan")
        return {
            "residual_frac": resid_frac,
            "residual_abs": resid_abs,
            "theta_plasma": theta,
            "psi_passive": psi,
            "c_plasma": c_plasma,
            "c_passive": c_passive,
            "pred_trust": pred_w / w,
        }


# --- the λ × profile-DOF frontier -------------------------------------


def sweep_frontier(
    operator: ForwardOperator,
    raw_trust: np.ndarray,
    i_pf_per_slice: np.ndarray,
    quiescent_mask: np.ndarray,
    *,
    lambda_grid: tuple[float, ...] = LAMBDA_GRID,
    profile_dof_grid: tuple[int, ...] = PROFILE_DOF_GRID,
    passive_rank: int = _PASSIVE_RANK_DEFAULT,
    passive_rank_grid: tuple[int, ...] | None = None,
    target: TrustTarget | None = None,
) -> dict[str, Any]:
    """Report the λ × profile-DOF frontier on a set of slices.

    For each ``(profile_order, λ)`` cell, solve every slice and report the
    QUIESCENT-slice residual statistics — the frontier is read on quiescent
    slices because that is where ``r`` should be small-but-non-trivial (a slice
    consistent with smooth GS force balance).  A cell with median quiescent
    ``r`` collapsing to ~0 is trivial (too much DOF); a cell with ``r`` ~ 1 is
    starved (too little DOF).  The instrument's resolution lives in between.

    ``raw_trust`` : ``(T, n_trust)`` raw amb at the trustworthy sensors.
    ``i_pf_per_slice`` : ``(T, n_coil)`` KNOWN per-coil PF currents [A].
    ``quiescent_mask`` : ``(T,)`` bool, ``True`` = quiescent.

    Returns the frontier table + the per-sensor scale + the suggested operating
    point (min profile-DOF with non-trivial quiescent ``r``, smallest λ).
    """
    target = target or trustworthy_target(operator)
    scale = robust_sensor_scale(raw_trust[:, : target.n], quiescent_mask)
    rank_grid = passive_rank_grid or (passive_rank,)

    qmask = np.asarray(quiescent_mask, dtype=bool)
    finite = np.isfinite(raw_trust[:, : target.n]).all(axis=1)
    cells: list[dict[str, Any]] = []
    for order in profile_dof_grid:
        for rank in rank_grid:
            solver = InverseSolver(operator, target, scale, order, rank)
            for lam in lambda_grid:
                rfrac_q: list[float] = []
                rfrac_all: list[float] = []
                for t in range(raw_trust.shape[0]):
                    if not finite[t]:
                        continue
                    out = solver.solve(
                        raw_trust[t, : target.n], i_pf_per_slice[t], lam
                    )
                    rf = out["residual_frac"]
                    if not np.isfinite(rf):
                        continue
                    rfrac_all.append(rf)
                    if qmask[t]:
                        rfrac_q.append(rf)
                q = np.array(rfrac_q) if rfrac_q else np.array([np.nan])
                cells.append(
                    {
                        "profile_order": int(order),
                        "n_plasma_dof": solver.n_plasma_dof(),
                        "passive_rank": int(rank),
                        "n_total_dof": solver.n_total_dof(),
                        "lambda": float(lam),
                        "quiescent_residual_median": float(np.nanmedian(q)),
                        "quiescent_residual_p10": float(np.nanpercentile(q, 10)),
                        "quiescent_residual_p90": float(np.nanpercentile(q, 90)),
                        "n_quiescent_slices": int(len(rfrac_q)),
                        "n_all_slices": int(len(rfrac_all)),
                    }
                )

    # operating-point selection (ANTI-TUNING): the MIN profile-DOF whose
    # quiescent residual is NON-trivial (not collapsed to ~0) at the SMALLEST λ.
    # NOT the cell that maximises detection AUROC — that is the subtle
    # "tune to pass".
    op_point = _select_operating_point(cells)
    return {
        "schema": "gs-frontier-v0",
        "signature_key": operator.signature_key,
        "lambda_grid": list(lambda_grid),
        "profile_dof_grid": list(profile_dof_grid),
        "passive_rank_grid": list(rank_grid),
        "n_trust_sensor": target.n,
        "trivial_residual_floor": _TRIVIAL_FLOOR,
        "starved_residual_ceiling": _STARVED_CEILING,
        "cells": cells,
        "operating_point": op_point,
        "sensor_scale_median": float(np.median(scale)),
        "framing_note": (
            "fractional residual ||W(pred-raw)||/||W*raw|| with W a per-sensor "
            "robust scale fit on quiescent slices; bears on extrapolation-"
            "coordinates (SURFACED, not locked)."
        ),
    }


# Anti-tuning thresholds — WRITTEN BEFORE THE RUN (operating-point selection
# is by the SANITY rule, never by detection AUROC).
_TRIVIAL_FLOOR = 0.02
"""A quiescent median residual below this is TRIVIAL (the DOF over-fit r→0)."""
_STARVED_CEILING = 0.9
"""A quiescent median residual above this is STARVED (too little DOF to fit)."""


def _select_operating_point(
    cells: list[dict[str, Any]], require_near_vacuum: bool = False
) -> dict[str, Any]:
    """Select the operating point by the SANITY rule (anti-tuning).

    The min profile-DOF whose quiescent median residual is NON-trivial
    (``> _TRIVIAL_FLOOR``) and not starved (``< _STARVED_CEILING``) AND — when
    ``require_near_vacuum`` — whose cell is near-vacuum-SOUND (``near_vacuum_ok``;
    the inferred plasma current at near-vacuum is a small fraction of flat-top),
    at the SMALLEST λ that keeps it in band.  Near-vacuum soundness is a
    correctness criterion defined with ZERO reference to the detection labels, so
    gating on it is sound regularisation, not tuning (it pushes λ off the
    least-regularised λ=0 corner, which is correct).  Returns ``selected=False``
    (caller reports FAIL) if no cell qualifies — an honest negative.
    """
    candidates = [
        c
        for c in cells
        if _TRIVIAL_FLOOR
        < c["quiescent_residual_median"]
        < _STARVED_CEILING
        and (not require_near_vacuum or c.get("near_vacuum_ok", False))
    ]
    if not candidates:
        reason = (
            "no cell both non-trivial AND near-vacuum-sound"
            if require_near_vacuum
            else "no cell in the non-trivial band"
        )
        return {"selected": False, "reason": reason}
    candidates.sort(key=lambda c: (c["n_plasma_dof"], c["lambda"]))
    chosen = candidates[0]
    return {
        "selected": True,
        "profile_order": chosen["profile_order"],
        "n_plasma_dof": chosen["n_plasma_dof"],
        "passive_rank": chosen["passive_rank"],
        "lambda": chosen["lambda"],
        "quiescent_residual_median": chosen["quiescent_residual_median"],
        "near_vacuum_ok": chosen.get("near_vacuum_ok"),
        "rule": (
            f"min plasma-DOF with non-trivial quiescent r (>{_TRIVIAL_FLOOR:g}) "
            f"and not starved (<{_STARVED_CEILING:g})"
            f"{' AND near-vacuum-sound' if require_near_vacuum else ''}, smallest "
            "lambda; selected by SANITY not detection AUROC"
        ),
    }


# --- residual time-series at a fixed operating point ------------------


def residual_series(
    operator: ForwardOperator,
    raw_trust: np.ndarray,
    i_pf_per_slice: np.ndarray,
    *,
    profile_order: int,
    passive_rank: int,
    lam: float,
    sensor_scale: np.ndarray,
    include_passive: bool = True,
    target: TrustTarget | None = None,
    statistic: str = "residual_frac",
) -> np.ndarray:
    """Per-slice residual at a FIXED operating point.

    ``statistic`` selects ``"residual_frac"`` (the dimensionless
    ``||W(pred-raw)||/||W*raw||``) or ``"residual_abs"`` (the absolute whitened
    misfit ``||W(pred-raw)||``).  Both are reported: the fractional form's
    instantaneous denominator partly tracks 1/field, a detection confound; the
    absolute form is the physically-motivated departure magnitude.

    ``include_passive=False`` runs the EDDY-CURRENT ABLATION — the same solve
    with the inferred ``G_passive`` term removed (``passive_rank=0``), so a
    downstream consumer can compare the residual with/without the passive term.
    """
    target = target or trustworthy_target(operator)
    rank = passive_rank if include_passive else 0
    solver = InverseSolver(operator, target, sensor_scale, profile_order, rank)
    out = np.full(raw_trust.shape[0], np.nan)
    finite = np.isfinite(raw_trust[:, : target.n]).all(axis=1)
    for t in range(raw_trust.shape[0]):
        if not finite[t]:
            continue
        out[t] = solver.solve(raw_trust[t, : target.n], i_pf_per_slice[t], lam)[
            statistic
        ]
    return out


# --- artifact I/O -----------------------------------------------------


def write_frontier(payload: dict[str, Any], out_path: Path | None = None) -> Path:
    """Write the compact λ × profile-DOF frontier artifact."""
    from pathlib import Path as _Path  # noqa: PLC0415

    out_path = out_path or (
        _Path(__file__).parent / "artifacts" / "gs_residual_frontier.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
