"""Classical radial force balance and vacuum decay-index diagnostics.

Analytic identities only — no fitting, no learned components, no solver
state.  These functions answer one question about the forward operator:
does the vacuum field it predicts at the plasma satisfy the field a
confined equilibrium REQUIRES there?

* :func:`shafranov_vertical_field` — the Shafranov vertical-field
  requirement for a circular large-aspect current ring,
  ``B_v = −μ0·Ip/(4πR)·(ln(8R/a) + βp + li/2 − 3/2)`` (signed: a positive
  toroidal plasma current needs a NEGATIVE B_z so the ``2πR·Ip·B_z`` ring
  force points inward against hoop expansion).
* :func:`decay_index` — the vacuum decay index
  ``n = −(R/B_z)·(∂B_z/∂R)``; the classic rigid-displacement stability
  window is ``0 < n < 3/2`` (n > 0 vertical, n < 3/2 radial).
* :func:`filament_bz` / :func:`known_coil_bz` / :func:`passive_circuit_bz`
  — midplane-capable B_z evaluation at ARBITRARY (R, Z) points from the
  same finite-area cylinder kernel (:func:`imas_ambix.gs.cylinder.
  hybrid_greens`), per-coil column merge, and solenoid response scale the
  sensor-ring operator uses (:func:`imas_ambix.gs.operator.build_operator`)
  — so the field diagnosed AT THE PLASMA is byte-consistent with the
  operator the vacuum campaign validated at the sensors.

Sign conventions match the operator Green's functions throughout: B_z is
per ampere of +φ current; a single loop carries positive B_z inside
itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.gs.operator import (
    _KNOWN_ROLES,
    MU0,
    SOLENOID_RESPONSE_SCALE,
    classify_circuits,
)

if TYPE_CHECKING:
    from imas_ambix.gs.geometry import GeometryTable, PFFilament

#: Classic rigid-displacement stability window on the vacuum decay index:
#: n > 0 for vertical stability, n < 3/2 for radial stability.
DECAY_INDEX_WINDOW = (0.0, 1.5)


def shafranov_vertical_field(
    ip_amperes: float,
    r_axis: float,
    minor_radius: float,
    betap_li2: float,
) -> float:
    """Signed vertical field a circular current ring needs for radial balance.

    ``betap_li2`` is the combination ``βp + li/2`` (the referee supplies both
    terms).  Returns the SIGNED ``B_z`` [T]: negative for a positive plasma
    current — the ring force per unit length is ``Ip·B_z r̂``, so balance
    against the outward hoop force requires ``sign(B_z) = −sign(Ip)``.
    """
    ip = float(ip_amperes)
    r0 = float(r_axis)
    a = float(minor_radius)
    if r0 <= 0.0 or a <= 0.0 or a >= r0 * 8.0:
        return float("nan")
    lam = np.log(8.0 * r0 / a) + float(betap_li2) - 1.5
    return float(-MU0 * ip / (4.0 * np.pi * r0) * lam)


def shafranov_vertical_field_elongated(
    ip_amperes: float,
    r_axis: float,
    minor_radius: float,
    elongation: float,
    betap_li2: float,
) -> float:
    """Elongation-corrected Shafranov requirement, ``ln(8R/(a·√κ))`` form.

    The large-aspect circular identity overstates the requirement for an
    elongated column: to leading order the external-inductance term sees the
    effective minor radius ``a·√κ`` (the ellipse's area-equivalent circle),
    so ``Λ = ln(8R/(a√κ)) + βp + li/2 − 3/2``.  Together with the circular
    form this brackets the identity's own uncertainty at tight aspect —
    the band a residual must clear before it can convict the coil model.
    ``elongation = 1`` reproduces :func:`shafranov_vertical_field` exactly.
    """
    kappa = float(elongation)
    if kappa <= 0.0 or not np.isfinite(kappa):
        return float("nan")
    return shafranov_vertical_field(
        ip_amperes, r_axis, float(minor_radius) * float(np.sqrt(kappa)), betap_li2
    )


def decay_index(r: np.ndarray, bz: np.ndarray) -> np.ndarray:
    """Vacuum decay index ``n(R) = −(R/B_z)·(∂B_z/∂R)`` along a midplane ray.

    ``r`` must be monotonic; the derivative is the second-order
    ``np.gradient`` on the given (possibly non-uniform) grid.  Points where
    ``B_z ≈ 0`` (sign reversal — no meaningful index) return NaN.
    """
    r = np.asarray(r, dtype=np.float64)
    bz = np.asarray(bz, dtype=np.float64)
    dbz = np.gradient(bz, r)
    floor = 1e-12 + 1e-6 * float(np.nanmax(np.abs(bz), initial=0.0))
    out = np.where(np.abs(bz) > floor, -(r / bz) * dbz, np.nan)
    return np.asarray(out, dtype=np.float64)


def _filament_field(
    r: np.ndarray,
    z: np.ndarray,
    filaments: list[PFFilament],
    component: int,
) -> np.ndarray:
    """One hybrid-kernel field component at (r, z) from a circuit's filaments.

    ``component`` indexes the :func:`hybrid_greens` output tuple
    (0 = ψ [Wb per A], 1 = B_R, 2 = B_z [T per A]); each filament is
    xmult-weighted with the same 1 cm section floor the operator uses;
    ``width = height = 0`` filaments reduce to the point-loop formulas
    inside :func:`hybrid_greens`.
    """
    rr = np.asarray(r, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    out = np.zeros(rr.shape, dtype=np.float64)
    for f in filaments:
        w = float(f.xmult)
        if w == 0.0:
            continue
        da = max(abs(float(f.width)), 0.01)
        dz = max(abs(float(f.height)), 0.01)
        fields = hybrid_greens(rr, zz, float(f.r), float(f.z), da, dz)
        out = out + w * fields[component]
    return out


def filament_bz(
    r: np.ndarray,
    z: np.ndarray,
    filaments: list[PFFilament],
) -> np.ndarray:
    """B_z [T per A] at (r, z) from one circuit's filaments (xmult-weighted)."""
    return _filament_field(r, z, filaments, 2)


def filament_psi(
    r: np.ndarray,
    z: np.ndarray,
    filaments: list[PFFilament],
) -> np.ndarray:
    """Total flux ψ [Wb per A] at (r, z) from one circuit's filaments."""
    return _filament_field(r, z, filaments, 0)


def _known_coil_columns(
    table: GeometryTable,
    r: np.ndarray,
    z: np.ndarray,
    component: int,
) -> tuple[list[str], np.ndarray]:
    """Per-KNOWN-coil field columns at arbitrary points, operator merge rule.

    Reproduces the operator's KNOWN-block assembly exactly — one column per
    driven amc channel, redundant fcoil circuits AVERAGED (each is already
    normalised to the full coil current), the solenoid column scaled by the
    vacuum-measured :data:`SOLENOID_RESPONSE_SCALE` — but evaluated at the
    caller's points instead of the sensor ring.
    """
    rr = np.asarray(r, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    classes = classify_circuits(table.pf_filaments, table.amc_current_channels)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)

    pf_by_chan: dict[str, list[int]] = {}
    for cc in classes:
        if cc.role in _KNOWN_ROLES:
            pf_by_chan.setdefault(cc.amc_channel, []).append(cc.circuit)

    channels: list[str] = []
    cols: list[np.ndarray] = []
    for chan in sorted(pf_by_chan):
        circs = sorted(pf_by_chan[chan])
        merged = np.mean(
            [_filament_field(rr, zz, by_circ[c], component) for c in circs], axis=0
        )
        if chan == "sol_current":
            merged = merged * SOLENOID_RESPONSE_SCALE
        channels.append(chan)
        cols.append(merged)
    if not cols:
        return channels, np.zeros(rr.shape + (0,), dtype=np.float64)
    return channels, np.stack(cols, axis=-1)


def known_coil_bz(
    table: GeometryTable,
    r: np.ndarray,
    z: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Per-KNOWN-coil B_z columns [T per A] at arbitrary (r, z) points.

    Returns ``(channels, cols)`` with ``channels`` in the same sorted order
    as ``ForwardOperator.pf_amc_channels``, so ``cols @ i_pf`` consumes the
    ``load_shot_windows`` current vectors directly.
    """
    return _known_coil_columns(table, r, z, 2)


def known_coil_psi(
    table: GeometryTable,
    r: np.ndarray,
    z: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Per-KNOWN-coil flux columns [Wb per A] at arbitrary (r, z) points.

    Same assembly as :func:`known_coil_bz` with the ψ component — the coil
    flux-linkage row a plasma circuit needs when it joins the one-matrix
    coupled flux solve.
    """
    return _known_coil_columns(table, r, z, 0)


def passive_circuit_bz(
    table: GeometryTable,
    circuits: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Per-passive-circuit B_z columns [T per A] at arbitrary (r, z) points.

    ``circuits`` is the sorted circuit-id vector of a
    :class:`~imas_ambix.latent.temporal_operator.PassiveCircuitSystem`;
    columns are returned in that order so ``cols @ i_vessel`` consumes its
    predicted circuit currents directly.
    """
    rr = np.asarray(r, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    cols = [filament_bz(rr, zz, by_circ[int(c)]) for c in np.asarray(circuits)]
    if not cols:
        return np.zeros(rr.shape + (0,), dtype=np.float64)
    return np.stack(cols, axis=-1)


def passive_circuit_psi(
    table: GeometryTable,
    circuits: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Per-passive-circuit flux columns [Wb per A] at arbitrary (r, z) points.

    ψ analogue of :func:`passive_circuit_bz` — the vessel→plasma flux
    linkage a plasma row reads in the one-matrix coupled flux solve.
    """
    rr = np.asarray(r, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    cols = [filament_psi(rr, zz, by_circ[int(c)]) for c in np.asarray(circuits)]
    if not cols:
        return np.zeros(rr.shape + (0,), dtype=np.float64)
    return np.stack(cols, axis=-1)


def coil_group(channel: str) -> str:
    """Waterfall group of one amc channel: case / sol / p2…p6 / other."""
    name = channel.lower()
    if "case" in name:
        return "case"
    for g in ("sol", "p2", "p3", "p4", "p5", "p6"):
        if name.startswith(g):
            return g
    return "other"


__all__ = [
    "DECAY_INDEX_WINDOW",
    "coil_group",
    "decay_index",
    "filament_bz",
    "filament_psi",
    "known_coil_bz",
    "known_coil_psi",
    "passive_circuit_bz",
    "passive_circuit_psi",
    "shafranov_vertical_field",
    "shafranov_vertical_field_elongated",
]
