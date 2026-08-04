"""Geometry-only Green's-function FORWARD operator for the GS soft-prior.

:mod:`imas_ambix.gs.geometry` tabulates the per-campaign machine geometry;
this module turns that fixed geometry into the linear forward map

    prediction(sensor)  =  G_pf · I_pf  +  G_plasma · c_plasma  +  G_passive · c_passive

where each ``G_·`` is a **geometry-only** matrix (one per campaign signature) and

* ``I_pf``      — KNOWN active-PF-coil currents, assembled from the RAW ``amc``
                  coil-current channels via each filament's circuit + ``xmult``;
* ``c_plasma``  — the INFERRED toroidal plasma-current distribution (a low-dim
                  ``jφ(R, Z)`` basis on a limiter-masked grid — locked
                  ``latent-to-psi-representation = current-distribution-greens``);
* ``c_passive`` — the INFERRED passive / eddy currents (a nuisance term — the
                  ``amm`` computed current values are an EFIT-wall-model OUTPUT
                  and are EXCLUDED — never a known source; only the passive
                  *geometry* is used).

**Scope boundary.**  This module builds all three column blocks (pure
geometry), assembles the KNOWN ``I_pf`` term, and validates the vacuum
round-trip (PF-only, no plasma).  *Solving* for ``c_plasma`` / ``c_passive``
and the ``λ × profile-DOF`` sweep belong to the solver, not here.  The default
plasma basis is a documented coarse limiter-masked grid so ``G`` is a complete
forward map; the basis resolution is the solver's to tune.

Physics — axisymmetric circular-filament Green's functions (Jackson §5.5 /
FreeGS).  Each PF/plasma/passive element is a toroidal current loop at
``(a, z0)``; its contribution at a sensor ``(R, Z)`` is

* poloidal flux ``ψ`` [Wb per A]   — for flux-loop sensors;
* field components ``B_R``, ``B_Z`` [T per A] — projected onto the probe
  orientation ``B = B_R·cos θ + B_Z·sin θ`` with ``θ = angle_deg`` from the
  geometry table (``magpr_ang`` — a 90° probe reads ``B_Z``, a 0° probe reads ``B_R``).

with ``m = k² = 4aR / ((a+R)² + (Z−z0)²)`` (scipy ``ellipk``/``ellipe`` take
the parameter ``m``, **not** the modulus ``k`` — getting this wrong silently
corrupts every value, so it is pinned by a test).

SI denorm (decision, documented — does NOT pre-empt ``extrapolation-coordinates``)
-------------------------------------------------------------------------------
The operator works in **raw SI**: currents in amperes, ``μ0`` carried
explicitly, so ``G`` outputs flux in **Wb** and field in **T** — directly
comparable to raw ``amb`` (``fl_* : Wb``, ``ccbv/obr/obv : T``).  The ``amc``
coil channels are stored in ``kA · turn`` (amp-turns; ``turns = 1`` throughout
the MAST ``fcoil`` table, so the per-filament weight is exactly ``xmult``), so
``I_filament[A] = I_amc_circuit[kA·turn] · 1000 · xmult``.  We deliberately do
**not** non-dimensionalise here: a raw-SI forward map is the framing-neutral
choice and leaves the open ``extrapolation-coordinates`` decision (dimensionless
``R/R0``, ``ψ/(μ0 Ip R0)``, …) to a later consumer.  The fixed MAST constants
``MAST_R0`` / ``MAST_A`` are carried through from the geometry table for that later
framing but are **not** used to rescale ``G`` here.

Never reads any reconstructed EFIT output (no ``psirz`` / profiles / separatrix);
geometry comes from the geometry table only, and the only signal DATA read is
the RAW ``amc`` coil currents for the KNOWN term.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import ellipe, ellipk  # type: ignore[import-untyped]

from imas_ambix.data.paths import local_shot_path
from imas_ambix.gs import circuits as circuits_mod
from imas_ambix.gs.geometry import (
    MAST_A,
    MAST_R0,
    CircuitDrive,
    GeometryTable,
    PFFilament,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

COIL_MODEL_VERSION = "source-stated-drives"
"""Version marker for the circuit -> current-channel assignment + ``G_pf``
column construction (``classify_circuits`` / ``build_operator``).  Bump this
any time either changes -- e.g. a different geometric match radius, a new
circuit special-case, or a change to which circuits get merged into one
``G_pf`` column -- so a downstream cache keyed on this string invalidates
instead of silently mixing predictions built under different assignment
rules.  ``"case-circuits-v2"`` is the fix that stopped folding MAST's 10
coil-CASE circuits into their co-located active coil's column (measured
impact: docs/mast-coil-circuits.html §6,
:mod:`imas_ambix.gs.artifacts.case_circuit_impact` — median 0.44σ / all 73
B-probes above 0.05σ / worst probe 17.8σ on shots 18502-18505); the prior,
unversioned behaviour is implicitly ``"v1"``.  ``"cylinder-sensors-v3"`` added
the finite-area cylinder kernel for near-pack B-probes; ``"-v4"`` applies
:data:`SOLENOID_RESPONSE_SCALE` to the P1 solenoid column; ``"-v5"`` collapses
filled rectangular coil packs to one thick-cylinder filament each
(:func:`imas_ambix.gs.geometry.collapse_rectangular_circuits`), changing the
G_pf column source set for a fixed physical coil.  ``"source-stated-drives"``
takes a source's own circuit -> channel -> ampere-turn declaration
(:attr:`~imas_ambix.gs.geometry.GeometryTable.circuit_drives`) in place of every
positional rule, and withholds :data:`SOLENOID_RESPONSE_SCALE` from a column
whose weight the source stated.  A table declaring no drives assembles exactly
as under ``"-v5"``."""

SOLENOID_RESPONSE_SCALE = 1.0825
"""Multiplicative correction to the P1 central-solenoid ``G_pf`` column.

Measured plasma-free on the pooled coil-only vacuum stratum (85 shots / ~111k
slices) by :mod:`scripts.solenoid_response_attribution`: the geometry forward
model UNDER-predicts the solenoid field-per-amp by ~8% (empirical/model scale
``1.0825 [1.0625, 1.1009]``, significantly ≠ 1).  The attribution adjudicated
the mechanism on the vacuum pool — a single uniform scale (this constant) beats
both a per-axial-band free profile (ΔBIC −12.3, the band profile is flat) and
an axial-extent/offset perturbation (ΔBIC −8.4, which returns the identity),
so the solenoid's modelled EXTENT and POSITION are correct and only its overall
RESPONSE is low.  ``k > 1`` rules out an undriven winding section (that would
under-drive → ``k < 1``); the residual is a turn-count (effective amp-turns
``Σxmult = 328`` ≈ 8% low) or a ``sol_current`` channel-scale — degenerate in
the forward map, so it is carried here as one response constant on the column,
applied by default (a vacuum-derived machine-description correction, never
per-shot tuning).  Set to ``1.0`` to recover the un-corrected ``-v3`` column.

Because that residual is degenerate between a turn count and a channel scale,
this constant IS an amp-turn statement under another name: it says the
solenoid's effective weight is ``328 × 1.0825 = 355.06`` ampere turns per
ampere of ``sol_current``.  A source that states the weight itself therefore
supersedes it, and :func:`build_operator` skips the correction on any column
whose weight came from the source -- otherwise the same 8% enters twice, once
as a turn count and once as a response."""

# --- Physical constants -----------------------------------------------

MU0 = 4.0e-7 * np.pi
"""Vacuum permeability [T·m/A]."""

_KA_TURN_TO_A = 1.0e3
"""amc coil currents are stored in ``kA · turn``; ``turns = 1`` throughout the
MAST ``fcoil`` table so the per-filament weight is exactly ``xmult`` and the
SI conversion is a flat ×1000 (kA → A)."""

# Numerical floor so on-axis / coincident points don't divide by zero.
_R_FLOOR = 1.0e-9


# --- Green's functions (per unit current, SI) -------------------------


def greens_psi(rs: np.ndarray, zs: np.ndarray, ar: float, az: float) -> np.ndarray:
    """Total poloidal flux ``Φ`` [Wb per A] at sensors from a loop at ``(ar, az)``.

    Axisymmetric circular filament of unit current — the flux *threading* the
    observation loop, ``Φ = 2π R A_φ`` with the standard vector potential

        A_φ(R, Z) = (μ0 / (π k)) · √(a / R) · [(1 − k²/2) K(k²) − E(k²)]

    so

        Φ(R, Z) = (2 μ0 / k) · √(a R) · [(1 − k²/2) K(k²) − E(k²)]

    with ``k² = 4 a R / ((a + R)² + (Z − az)²)``.  This is the TOTAL flux (Wb) an
    ``amb`` ``fl_*`` loop measures — NOT the stream function ``Φ/2π`` — and it
    is consistent with :func:`greens_bz_br` via ``B_Z = (1/(2π R)) ∂Φ/∂R`` and
    ``B_R = −(1/(2π R)) ∂Φ/∂Z`` (pinned by
    :func:`test_psi_and_b_finite_difference_consistency`).
    """
    r = np.asarray(rs, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    dz = z - az
    denom = (ar + r) ** 2 + dz**2
    k2 = 4.0 * ar * r / np.maximum(denom, _R_FLOOR)
    k2 = np.clip(k2, 0.0, 1.0 - 1.0e-12)
    k = np.sqrt(k2)
    big_k = ellipk(k2)
    big_e = ellipe(k2)
    pref = 2.0 * MU0 * np.sqrt(ar * np.maximum(r, _R_FLOOR)) / np.maximum(k, _R_FLOOR)
    psi = pref * ((1.0 - 0.5 * k2) * big_k - big_e)
    # at R→0 the loop encloses no flux at the axis sensor → Φ→0
    return np.where(r < _R_FLOOR, 0.0, psi)


def greens_bz_br(
    rs: np.ndarray, zs: np.ndarray, ar: float, az: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(B_Z, B_R)`` [T per A] at sensors ``(rs, zs)`` from a loop at ``(ar, az)``.

    Standard axisymmetric forms (Jackson §5.5).  On-axis (``R→0``) ``B_R→0`` and
    ``B_Z`` reduces to the textbook ``μ0 a² / (2 (a² + Δz²)^{3/2})`` — pinned by
    :func:`test_on_axis_field_matches_textbook`.
    """
    r = np.asarray(rs, dtype=np.float64)
    z = np.asarray(zs, dtype=np.float64)
    dz = z - az
    denom = (ar + r) ** 2 + dz**2
    sq = np.sqrt(np.maximum(denom, _R_FLOOR))
    k2 = 4.0 * ar * r / np.maximum(denom, _R_FLOOR)
    k2 = np.clip(k2, 0.0, 1.0 - 1.0e-12)
    big_k = ellipk(k2)
    big_e = ellipe(k2)
    d2 = (ar - r) ** 2 + dz**2
    pre = MU0 / (2.0 * np.pi)
    bz = pre / sq * (big_k + (ar**2 - r**2 - dz**2) / np.maximum(d2, _R_FLOOR) * big_e)
    br_full = (
        pre
        * dz
        / (np.maximum(r, _R_FLOOR) * sq)
        * (-big_k + (ar**2 + r**2 + dz**2) / np.maximum(d2, _R_FLOOR) * big_e)
    )
    br = np.where(r < _R_FLOOR, 0.0, br_full)
    return bz, br


def _project_bprobe(
    bz: np.ndarray, br: np.ndarray, angle_deg: np.ndarray
) -> np.ndarray:
    """Project ``(B_Z, B_R)`` onto each probe orientation.

    ``B_probe = B_R·cos θ + B_Z·sin θ`` with ``θ = angle_deg``.  A 90° probe
    (``magpr_ang = 90`` → vertical/Bv) reads ``B_Z``; a 0° probe (``= 0`` →
    radial/Br) reads ``B_R``.  The general projection (not a two-branch
    if/else) handles oblique probes and is pinned to those two cases by a test.
    This is the crux the table's orientation-constrained mapping exists to protect:
    co-located ``obr``/``obv`` pairs differ ONLY by ``angle_deg``.
    """
    th = np.deg2rad(np.asarray(angle_deg, dtype=np.float64))
    proj: np.ndarray = br * np.cos(th) + bz * np.sin(th)
    return proj


# --- Source-element classification (which circuits are KNOWN PF coils) -

# amc channel-name → physical PF coil.  Maps the geometric coil identity to the
# RAW amc current channel that drives it.  We map ONLY the active PF coils +
# solenoid we can identify by geometry; everything else is INFERRED nuisance.
# Verified by geometric centroid in :func:`classify_circuits` (verify-and-flag,
# never fabricate).  The keys are the canonical coil labels; the
# values are the preferred ``*_current`` amc channel for that coil.
_PF_COIL_AMC = {
    "sol": "sol_current",  # P1 central solenoid (circuit 1; Rc≈0.14)
    "p2iu": "p2iu_coil_current",
    "p2il": "p2il_coil_current",
    "p2ou": "p2ou_coil_current",
    "p2ol": "p2ol_coil_current",
    "p3u": "p3u_coil_current",
    "p3l": "p3l_coil_current",
    "p4u": "p4u_coil_current",
    "p4l": "p4l_coil_current",
    "p5u": "p5u_coil_current",
    "p5l": "p5l_coil_current",
    "p6u": "p6u_current",
    "p6l": "p6l_current",
}
"""Canonical MAST PF-coil label → preferred RAW ``amc`` current channel.

Falls back to the bare ``<coil>_current`` channel when the ``*_coil_current``
variant is absent for a campaign.  Coils whose geometry we cannot pin to an amc
channel are flagged and INFERRED, never guessed."""

# Approximate (Rc, Zc) centroids of the MAST PF coils [m], from the MAST machine
# description (device geometry, NOT EFIT reconstruction).  Used ONLY to label a
# circuit's amc channel by nearest centroid; the actual filament (R, Z) always
# come from the geometry table.  A circuit unmatched within _COIL_MATCH_M is
# INFERRED.
_PF_COIL_CENTROID = {
    "sol": (0.14, 0.0),
    "p2iu": (0.475, 1.75),
    "p2il": (0.475, -1.75),
    "p2ou": (0.528, 1.72),
    "p2ol": (0.527, -1.71),
    "p3u": (1.10, 1.10),
    "p3l": (1.10, -1.09),
    "p4u": (1.50, 1.10),
    "p4l": (1.50, -1.10),
    "p5u": (1.65, 0.50),
    "p5l": (1.65, -0.50),
    "p6u": (1.43, 0.90),
    "p6l": (1.43, -0.90),
}
_COIL_MATCH_M = 0.08
"""A circuit centroid within this distance of a known PF-coil centroid is
labelled that coil (KNOWN amc-driven); else the circuit is INFERRED passive."""

_CASE_BY_CIRCUIT_ID: dict[int, circuits_mod.CaseCircuit] = {
    c.circuit_id: c for c in circuits_mod.case_circuits()
}
"""``pfSystems.xml`` case-circuit id -> its :class:`~imas_ambix.gs.circuits.
CaseCircuit` description.  Measured directly (``docs/mast-coil-circuits.html``
§6, three sample shots, both ``fcoil`` signatures): the ``efm`` circuit
numbering for ids 1-23 agrees 1:1 with ``pfSystems.xml``'s own ``pfCircuit``
numbering — circuit 14 IS "P2U case", exactly as in the machine description.
:func:`classify_circuits` uses this authoritative id correspondence (never
distance alone) to tell a coil's CASE circuit apart from the coil itself once
geometry has already flagged the two as neighbours (see below)."""


@dataclass(frozen=True)
class CircuitClass:
    """Classification of one fcoil circuit: KNOWN active PF, KNOWN case, or
    INFERRED passive."""

    circuit: int
    centroid_r: float
    centroid_z: float
    n_filament: int
    sum_xmult: float
    role: str  # "known_pf" | "known_case" | "inferred_passive"
    coil_label: str  # "" for inferred
    amc_channel: str  # "" for inferred
    flag: str  # "" if confident, else a reason (verify-and-flag, never fabricate)
    source_stated_weight: bool = False
    """Whether the SOURCE supplied this column's ampere-turns-per-ampere.

    A stated weight is the machine description's own claim, so a calibration a
    consumer fitted to correct a DIFFERENT source's weight must not also be
    applied to it -- that would count the same correction twice."""


_KNOWN_ROLES = ("known_pf", "known_case")
"""Roles carrying a real (non-fabricated) driven current -> a G_pf column."""


def _declared_classes(
    by_circ: dict[int, list[PFFilament]],
    circuit_drives: Sequence[CircuitDrive],
    amc_channels: Sequence[str],
) -> list[CircuitClass]:
    """Classify from the source's own circuit -> channel declaration.

    Every reconstruction below this is skipped, because each exists to recover
    part of what a declaration states outright: the centroid radius recovers
    which conductors are supplied, the channel-name convention recovers which
    channel supplies one, and the case-id table recovers the coil/case split the
    radius cannot see.  Running any of them against a declaration can only
    disagree with it.

    A declared circuit whose channel this campaign does not publish is INFERRED
    with the missing channel named: the description says the conductor is
    supplied, and the acquisition set says nothing measured it here, and both are
    true.  Nothing is substituted for the absent measurement.
    """
    avail = set(amc_channels)
    by_circuit = {drive.circuit: drive for drive in circuit_drives}
    out: list[CircuitClass] = []
    for circ in sorted(by_circ):
        fs = by_circ[circ]
        w = np.array([f.xmult for f in fs], dtype=np.float64)
        rr = np.array([f.r for f in fs], dtype=np.float64)
        zz = np.array([f.z for f in fs], dtype=np.float64)
        wsum = float(w.sum())
        cr = float((w * rr).sum() / wsum) if wsum else float(rr.mean())
        cz = float((w * zz).sum() / wsum) if wsum else float(zz.mean())

        role, coil_label, amc_channel, flag = "inferred_passive", "", "", ""
        stated = False
        drive = by_circuit.get(circ)
        if drive is not None:
            if drive.channel in avail:
                role = "known_case" if "_case_current" in drive.channel else "known_pf"
                coil_label = drive.conductor
                amc_channel = drive.channel
                stated = True
            else:
                flag = (
                    f"source declares circuit {circ} driven by "
                    f"'{drive.channel}' at {drive.ampere_turns_per_ampere:g} "
                    "ampere turns per ampere, but this campaign does not publish "
                    "that channel -> INFERRED"
                )
        out.append(
            CircuitClass(
                circuit=circ,
                centroid_r=cr,
                centroid_z=cz,
                n_filament=len(fs),
                sum_xmult=wsum,
                role=role,
                coil_label=coil_label,
                amc_channel=amc_channel,
                flag=flag,
                source_stated_weight=stated,
            )
        )
    return out


def classify_circuits(
    filaments: Sequence[PFFilament],
    amc_channels: Sequence[str],
    active_circuits: Sequence[int] = (),
    circuit_drives: Sequence[CircuitDrive] = (),
) -> list[CircuitClass]:
    """Classify each fcoil circuit as KNOWN active PF, KNOWN case, or INFERRED.

    Verify-and-flag (never fabricate): a circuit is labelled
    KNOWN only when its filament centroid sits within :data:`_COIL_MATCH_M` of
    a known MAST coil centroid AND its driving ``amc`` channel actually exists
    for this campaign.  Every other circuit — the singleton structural
    conductors (the ``1004−938 = 167−101 = 66`` extra fc1004 elements, ~½ of
    which coincide with ``amm`` passive geometry) and any coil we cannot pin —
    is INFERRED passive / eddy nuisance.  ``amm`` computed currents are NEVER
    read (EFIT-wall-model outputs, not measurements).

    Case-circuit correction (measured, ``docs/mast-coil-circuits.html`` §6)
    -------------------------------------------------------------------------
    8 of MAST's 10 coil-CASE circuits sit within :data:`_COIL_MATCH_M` of their
    co-located ACTIVE coil's centroid (the case is a physically distinct,
    separately-supplied structural conductor a couple of cm from the winding it
    encloses) — geometry alone cannot tell them apart.  The nearest-centroid
    match below is therefore only the FIRST pass; before accepting it as
    "known_pf", we check :data:`_CASE_BY_CIRCUIT_ID` — the authoritative
    ``pfSystems.xml`` id correspondence (:mod:`imas_ambix.gs.circuits`) — for
    whether THIS SPECIFIC circuit id is actually the matched coil's case, not
    the coil itself.  If so the circuit is driven by its own measured
    ``*_case_current`` channel (``role = "known_case"``, its own dedicated
    G_pf column — never merged with the active coil's) rather than by the
    active coil's amp-turn channel.  A case with no channel for this campaign
    (P6U/P6L: ``pfSystems.xml`` constrains them to zero, no amc channel at all)
    or an absent channel falls back to INFERRED, exactly like any other
    unmapped circuit.

    Sources that state which conductors are supplied
    -------------------------------------------------------------------------
    Both passes above reconstruct, from position, a fact the ``efm`` filament
    list does not record: which circuits carry a supplied current.  A source
    that records it — an IMAS ``pf_active`` / ``pf_passive`` split — passes its
    supplied circuits as ``active_circuits``, and those become the only
    candidates for a KNOWN role; every other circuit is induced structure and
    is INFERRED regardless of how near a coil centroid it sits.  That is what
    the centroid radius cannot decide on such a source: its structure is
    resolved into per-element circuits, many of which lie inside
    :data:`_COIL_MATCH_M` of the winding they enclose, so the geometric pass
    alone promotes a coil's own case, supports and neighbouring segments to
    driven columns and drives them all with the winding's measured current.
    The case correction cannot rescue those either — :data:`_CASE_BY_CIRCUIT_ID`
    is keyed by ``efm``'s circuit numbering, which no other source shares — so
    it is not consulted when the source has stated the split itself.

    Whether a structural conductor that IS separately supplied on the machine
    can be driven by its measured channel then depends on the source recording
    that supply.  One that files every case as passive states that it does not,
    and those conductors keep inferred currents rather than borrowing a
    channel on the strength of a centroid match.

    Sources that also state WHICH channel supplies each conductor
    -------------------------------------------------------------------------
    ``circuit_drives`` is the full declaration -- circuit, channel and the
    ampere turns one ampere of it drives -- and it displaces every rule above;
    see :func:`_declared_classes`.  Supplying it makes ``active_circuits``
    redundant, since a declared drive is what being supplied means.
    """
    if circuit_drives:
        by_circ_declared: dict[int, list[PFFilament]] = {}
        for f in filaments:
            by_circ_declared.setdefault(f.circuit, []).append(f)
        return _declared_classes(by_circ_declared, circuit_drives, amc_channels)

    avail = set(amc_channels)
    declared_active = set(active_circuits)
    by_circ: dict[int, list[PFFilament]] = {}
    for f in filaments:
        by_circ.setdefault(f.circuit, []).append(f)

    out: list[CircuitClass] = []
    for circ in sorted(by_circ):
        fs = by_circ[circ]
        w = np.array([f.xmult for f in fs], dtype=np.float64)
        rr = np.array([f.r for f in fs], dtype=np.float64)
        zz = np.array([f.z for f in fs], dtype=np.float64)
        wsum = float(w.sum())
        # weighted centroid (xmult is the current-share weight)
        cr = float((w * rr).sum() / wsum) if wsum else float(rr.mean())
        cz = float((w * zz).sum() / wsum) if wsum else float(zz.mean())

        # nearest known coil centroid
        best_label, best_d = "", np.inf
        for label, (lr, lz) in _PF_COIL_CENTROID.items():
            d = float(np.hypot(cr - lr, cz - lz))
            if d < best_d:
                best_label, best_d = label, d

        role, coil_label, amc_channel, flag = "inferred_passive", "", "", ""
        eligible = (circ in declared_active) if declared_active else True
        if eligible and best_d <= _COIL_MATCH_M:
            case = None if declared_active else _CASE_BY_CIRCUIT_ID.get(circ)
            if case is not None and case.geometry_confusable_with == best_label:
                # This efm circuit IS the coil's dedicated case circuit (id
                # matches pfSystems.xml 1:1) — never drive it by the active
                # coil's current, even though geometry alone would confuse them.
                if not case.constrained_zero and case.l1_case_channel in avail:
                    role = "known_case"
                    coil_label = f"{best_label}_case"
                    amc_channel = case.l1_case_channel or ""
                elif case.constrained_zero:
                    flag = (
                        f"case circuit '{case.name}' (id={circ}) constrained to"
                        " zero by pfSystems.xml (no amc channel) → INFERRED"
                    )
                else:
                    flag = (
                        f"case circuit '{case.name}' (id={circ}) channel"
                        f" '{case.l1_case_channel}' absent from this campaign"
                        " → INFERRED"
                    )
            else:
                pref = _PF_COIL_AMC.get(best_label, "")
                fallback = f"{best_label}_current"
                chan = (
                    pref if pref in avail else (fallback if fallback in avail else "")
                )
                if chan:
                    role, coil_label, amc_channel = "known_pf", best_label, chan
                else:
                    flag = (
                        f"coil '{best_label}' matched by geometry"
                        f" (d={best_d * 1e3:.0f}mm) but no amc channel"
                        " present → INFERRED"
                    )
        out.append(
            CircuitClass(
                circuit=circ,
                centroid_r=cr,
                centroid_z=cz,
                n_filament=len(fs),
                sum_xmult=wsum,
                role=role,
                coil_label=coil_label,
                amc_channel=amc_channel,
                flag=flag,
            )
        )
    return out


# --- The forward operator ---------------------------------------------


def _default_plasma_basis(
    table: GeometryTable,
    nr: int = 9,
    nz: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """A coarse limiter-masked ``(R, Z)`` grid of unit toroidal current elements.

    The INFERRED plasma ``jφ(R, Z)`` is parameterised on this grid (one column of
    ``G_plasma`` per retained node).  The resolution (``nr × nz``) is a documented
    DEFAULT so ``G`` is a complete forward map; the actual basis-DOF + the GS
    ``λ`` are the solver's sweep, NOT decided here.  Nodes outside the limiter contour
    are dropped (a plasma current element cannot live in the structure).
    """
    lr = np.asarray(table.limiter_r, dtype=np.float64)
    lz = np.asarray(table.limiter_z, dtype=np.float64)
    if lr.size >= 3:
        r_lo, r_hi = float(lr.min()), float(lr.max())
        z_lo, z_hi = float(lz.min()), float(lz.max())
    else:  # degenerate limiter → fall back to nominal MAST extent
        r_lo, r_hi = 0.2, 1.5
        z_lo, z_hi = -1.0, 1.0
    rg = np.linspace(r_lo, r_hi, nr)
    zg = np.linspace(z_lo, z_hi, nz)
    mesh_r, mesh_z = np.meshgrid(rg, zg)
    rr = mesh_r.ravel()
    zz = mesh_z.ravel()
    inside = (
        _inside_polygon(rr, zz, lr, lz) if lr.size >= 3 else np.ones(rr.shape, bool)
    )
    return rr[inside], zz[inside]


def _inside_polygon(
    px: np.ndarray, py: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    """Ray-casting point-in-polygon (limiter mask); no shapely dependency."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    n = vx.size
    inside = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        cond = ((vy[i] > py) != (vy[j] > py)) & (
            px < (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i] + 1e-30) + vx[i]
        )
        inside ^= cond
        j = i
    return inside


@dataclass
class ForwardOperator:
    """Geometry-only Green's-function forward operator for one campaign.

    Holds the three column blocks (KNOWN PF, INFERRED plasma, INFERRED passive)
    + the sensor index, and applies them.  All matrices are pure geometry — built
    once per campaign signature; applying is a CPU-tiny matmul.
    """

    signature_key: str
    sensor_channels: list[str]  # row order: predicted amb channels
    sensor_kind: list[str]  # "flux_loop" | "b_probe", parallel to rows
    g_pf: np.ndarray  # (n_sensor, n_known_coil)  [Wb or T per A]
    g_plasma: np.ndarray  # (n_sensor, n_plasma_node)
    g_passive: np.ndarray  # (n_sensor, n_passive_node)
    pf_circuits: list[int]  # representative circuit id per G_pf column
    pf_amc_channels: list[str]  # amc channel per G_pf column (one physical coil)
    pf_merged_circuits: list[list[int]]  # circuits averaged into each G_pf column
    plasma_rz: np.ndarray  # (n_plasma_node, 2) basis node (R, Z)
    passive_rz: np.ndarray  # (n_passive_node, 2) passive element (R, Z)
    circuit_classes: list[CircuitClass]
    excluded_channels: list[str]  # table-unmatched amb flux loops (not predicted)
    flagged_channels: list[str]  # non-unique amb (predicted at index, flagged)
    r0: float = MAST_R0
    minor_radius: float = MAST_A
    #: The Nova registry physical digest of the machine this operator describes,
    #: when it was resolved.  ``signature_key`` above identifies the
    #: DISCRETIZATION the matrices were built on; this identifies the MACHINE, so
    #: two operators built at different subdivisions of one device are
    #: recognisable as the same hardware.  Empty when identity was not resolved,
    #: which keeps an operator built without registry access fully usable.
    physical_digest: str = ""

    # ---- forward apply ----

    def predict(
        self,
        i_pf: np.ndarray,
        c_plasma: np.ndarray | None = None,
        c_passive: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward map → predicted sensor values [Wb for flux loops, T for probes].

        ``i_pf`` : KNOWN PF-coil currents [A], one per :attr:`pf_circuits` column
        (use :meth:`assemble_pf_currents` to build from raw amc).  ``c_plasma`` /
        ``c_passive`` : the INFERRED amplitudes [A] (default zero → vacuum/PF-only
        prediction).  Solving for them belongs to the solver.
        """
        i_pf = np.asarray(i_pf, dtype=np.float64)
        pred = self.g_pf @ i_pf
        if c_plasma is not None and self.g_plasma.size:
            pred = pred + self.g_plasma @ np.asarray(c_plasma, dtype=np.float64)
        if c_passive is not None and self.g_passive.size:
            pred = pred + self.g_passive @ np.asarray(c_passive, dtype=np.float64)
        out: np.ndarray = np.asarray(pred, dtype=np.float64)
        return out

    def vacuum_prediction(self, i_pf: np.ndarray) -> np.ndarray:
        """PF-only (no plasma, no eddy) prediction — the vacuum-field signature."""
        return self.predict(i_pf, c_plasma=None, c_passive=None)

    def assemble_pf_currents(self, amc_values: dict[str, float]) -> np.ndarray:
        """Assemble the KNOWN per-COIL PF current [A] from raw amc channels.

        ``amc_values`` maps amc channel name → its (scalar, single-time-slice)
        current in the RAW stored units (``kA · turn``).  One entry per G_pf
        column = one physical coil; the value is the mapped amc channel converted
        to amperes (``× 1000``; ``turns = 1`` so amp-turns = amps).  The
        per-filament ``xmult`` split AND the merge of the coil's redundant fcoil
        circuits are already folded into :attr:`g_pf` at build time, so each
        coil current is applied exactly once.  Missing channels contribute zero
        (and were already flagged at build).
        """
        out = np.zeros(len(self.pf_amc_channels), dtype=np.float64)
        for j, chan in enumerate(self.pf_amc_channels):
            if chan and chan in amc_values:
                out[j] = float(amc_values[chan]) * _KA_TURN_TO_A
        return out

    # ---- summary ----

    def shapes(self) -> dict[str, Any]:
        identity = (
            {"physical_digest": self.physical_digest} if self.physical_digest else {}
        )
        return {
            "signature_key": self.signature_key,
            **identity,
            "n_sensor": len(self.sensor_channels),
            "n_flux_loop": self.sensor_kind.count("flux_loop"),
            "n_b_probe": self.sensor_kind.count("b_probe"),
            "g_pf_shape": list(self.g_pf.shape),
            "g_plasma_shape": list(self.g_plasma.shape),
            "g_passive_shape": list(self.g_passive.shape),
            "n_known_coil": len(self.pf_amc_channels),
            "n_plasma_node": int(self.plasma_rz.shape[0]),
            "n_passive_node": int(self.passive_rz.shape[0]),
            "n_excluded_channel": len(self.excluded_channels),
            "n_flagged_channel": len(self.flagged_channels),
        }

    def to_summary(self, amc_channels: Sequence[str] | None = None) -> dict[str, Any]:
        by_circ = {c.circuit: c for c in self.circuit_classes}
        # one entry per G_pf column = one physical coil (merged redundant circuits)
        d = self.shapes()
        d["pf_coils"] = [
            {
                "amc_channel": chan,
                "coil_label": by_circ[circs[0]].coil_label,
                "merged_circuits": circs,
                "centroid": [
                    round(by_circ[circs[0]].centroid_r, 4),
                    round(by_circ[circs[0]].centroid_z, 4),
                ],
                "note": (
                    "EFIT represents this coil with >1 fcoil circuit (redundant "
                    "fine+coarse discretisations); averaged into one column so the"
                    " amc current is applied once (no double-count)."
                )
                if len(circs) > 1
                else "",
            }
            for chan, circs in zip(
                self.pf_amc_channels, self.pf_merged_circuits, strict=True
            )
        ]
        d["n_known_coil"] = len(self.pf_amc_channels)
        d["n_inferred_passive_circuit"] = sum(
            1 for c in self.circuit_classes if c.role == "inferred_passive"
        )
        d["circuit_flags"] = [
            {"circuit": c.circuit, "flag": c.flag}
            for c in self.circuit_classes
            if c.flag
        ]
        # auditability: surface the amc sibling channels at mapped coils that we
        # did NOT consume (e.g. p4u_case_current / p4u_feed_current alongside the
        # chosen p4u_coil_current) so the per-coil channel choice is inspectable.
        if amc_channels is not None:
            used = set(self.pf_amc_channels)
            prefixes = {ch.split("_")[0] for ch in self.pf_amc_channels if "_" in ch}
            d["unmapped_amc_siblings"] = sorted(
                ch
                for ch in amc_channels
                if ch not in used
                and ch.split("_")[0] in prefixes
                and ch.endswith("_current")
            )
        d["excluded_channels"] = self.excluded_channels
        d["flagged_channels"] = self.flagged_channels
        d["si_denorm"] = {
            "psi_units": "Wb",
            "b_units": "T",
            "current_units": "A",
            "amc_raw_units": "kA*turn",
            "amc_to_si_factor": _KA_TURN_TO_A,
            "note": (
                "raw-SI forward map (mu0 carried; turns=1 so per-filament weight"
                " = xmult). Dimensionless framing (extrapolation-coordinates) is"
                " deferred; MAST_R0/MAST_A carried but not used to rescale G."
            ),
        }
        return d


# --- Building G from a campaign geometry table ------------------------


def _sensor_rows(
    table: GeometryTable,
) -> tuple[
    list[str], list[str], np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
]:
    """Select the PREDICTED sensor rows + their (R, Z, angle) from the table's map.

    Predicts the cleanly-mapped flux loops + all mapped B-probes.  EXCLUDES the
    Table-unmatched flux loops (``fl_p2*`` placeholder/displaced — cannot predict);
    the non-unique flagged flux loops (``fl_cc*`` etc. sharing a placeholder
    silop index) are predicted at the index but listed in ``flagged_channels``
    because the amb identity is ambiguous (kept OUT of any comparison target by
    the consumer).  Returns row metadata + the excluded / flagged channel lists.
    """
    channels: list[str] = []
    kinds: list[str] = []
    rs: list[float] = []
    zs: list[float] = []
    angs: list[float] = []
    flagged: list[str] = []
    for m in table.sensor_map:
        if m.flag:
            flagged.append(m.amb_channel)
            # non-unique flux loops still have a usable (R,Z) index → predict,
            # but flag.  (A B-probe never reaches here flagged in practice.)
        channels.append(m.amb_channel)
        kinds.append(m.kind)
        rs.append(m.r)
        zs.append(m.z)
        angs.append(0.0 if m.angle_deg is None else float(m.angle_deg))
    excluded = list(table.unmatched_amb)
    return (
        channels,
        kinds,
        np.array(rs, dtype=np.float64),
        np.array(zs, dtype=np.float64),
        np.array(angs, dtype=np.float64),
        excluded,
        flagged,
    )


def _green_columns(
    src_r: np.ndarray,
    src_z: np.ndarray,
    weights: np.ndarray,
    sensor_r: np.ndarray,
    sensor_z: np.ndarray,
    sensor_ang: np.ndarray,
    is_flux: np.ndarray,
    src_dr: np.ndarray | None = None,
    src_dz: np.ndarray | None = None,
) -> np.ndarray:
    """Build one G column (Wb/T per unit source amplitude) at each sensor row.

    ``src_*`` / ``weights`` are the filament (R, Z) + per-filament weight of ONE
    source (a PF circuit's filaments weighted by ``xmult``, or a single plasma /
    passive node of weight 1).  Flux-loop rows get ``Σ w·ψ``; B-probe rows get
    ``Σ w·(B_R cosθ + B_Z sinθ)``.

    ``src_dr``/``src_dz`` are the conductor cross-section extents [m]: sensors
    within the near band of a winding pack get the finite-area cylinder kernel
    (uniform current over the rectangular section — smooth through the
    conductor) instead of the log-singular point filament, which biased the
    B-probes adjacent to the P4/P5/solenoid packs by 0.1–0.25σ at typical
    currents (measured against nova's cylinder Biot–Savart).  ``None``
    (default) reproduces the pure point-filament map exactly — the far-field
    behaviour is identical either way.
    """
    from imas_ambix.gs.cylinder import hybrid_greens  # noqa: PLC0415

    col = np.zeros(sensor_r.shape, dtype=np.float64)
    n = len(np.atleast_1d(src_r))
    dr = np.zeros(n) if src_dr is None else np.asarray(src_dr, dtype=np.float64)
    dz = np.zeros(n) if src_dz is None else np.asarray(src_dz, dtype=np.float64)
    for ar, az, w, adr, adz in zip(src_r, src_z, weights, dr, dz, strict=True):
        if w == 0.0:
            continue
        if adr > 0.0 or adz > 0.0:
            psi, br, bz = hybrid_greens(
                sensor_r, sensor_z, float(ar), float(az), float(adr), float(adz)
            )
        else:
            psi = greens_psi(sensor_r, sensor_z, float(ar), float(az))
            bz, br = greens_bz_br(sensor_r, sensor_z, float(ar), float(az))
        bproj = _project_bprobe(bz, br, sensor_ang)
        col = col + w * np.where(is_flux, psi, bproj)
    return col


def polygon_section_column(
    vertices: np.ndarray,
    xmult: float,
    sensor_r: np.ndarray,
    sensor_z: np.ndarray,
    sensor_ang: np.ndarray,
    is_flux: np.ndarray,
) -> np.ndarray:
    """One G column from an analytic polygon cross-section (Urankar Part V).

    The exact-shape counterpart of the single-source path in
    :func:`_green_columns`: flux-loop rows get ``xmult·ψ``, B-probe rows get
    ``xmult·(B_R cosθ + B_Z sinθ)``, with (ψ, B_R, B_Z) from
    :func:`imas_ambix.gs.polygon.polygon_greens` per ampere of TOTAL section
    current (the same sign/units contract as ``hybrid_greens``).  Replaces the
    bounding-box column for a slanted / trapezoidal / hollow passive at
    O(edges) cost, with none of the multi-filament proxy's Riemann error.
    """
    from imas_ambix.gs.polygon import polygon_greens  # noqa: PLC0415

    psi, br, bz = polygon_greens(sensor_r, sensor_z, np.asarray(vertices, np.float64))
    bproj = _project_bprobe(bz, br, sensor_ang)
    return xmult * np.where(is_flux, psi, bproj)


def build_operator(
    table: GeometryTable, *, resolve_identity: bool = False
) -> ForwardOperator:
    """Build the per-campaign :class:`ForwardOperator` from a geometry table.

    Pure geometry: assembles ``G_pf`` (KNOWN PF coils, columns = active circuits
    with their filaments weighted by ``xmult``), ``G_plasma`` (INFERRED plasma
    basis on the limiter-masked grid), and ``G_passive`` (INFERRED passive/eddy
    nodes — the fcoil structural circuits, NOT amm currents).  CPU-tiny.

    ``resolve_identity`` additionally stamps the machine's physical digest from
    the Nova registry onto the operator.  Off by default so the historical build
    neither reads the registry nor changes its summary; the matrices are
    identical either way, because identity is provenance here and never an input.
    A registry miss is non-fatal — the operator is still correct geometry, it just
    carries no physical identity.
    """
    channels, kinds, srz_r, srz_z, srz_ang, excluded, flagged = _sensor_rows(table)
    is_flux = np.array([k == "flux_loop" for k in kinds], dtype=bool)

    classes = classify_circuits(
        table.pf_filaments,
        table.amc_current_channels,
        table.active_circuits,
        table.circuit_drives,
    )
    by_circ: dict[int, list[PFFilament]] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)

    def _circ_col(circ: int) -> np.ndarray:
        fs = by_circ[circ]
        fr = np.array([f.r for f in fs], dtype=np.float64)
        fz = np.array([f.z for f in fs], dtype=np.float64)
        fw = np.array([f.xmult for f in fs], dtype=np.float64)  # turns=1 → weight=xmult
        # efm carries SIGNED pack extents; physical size is |extent| with the
        # same 1 cm floor the solve-domain coil columns use (gs_solve)
        fdr = np.array([max(abs(f.width), 0.01) for f in fs], dtype=np.float64)
        fdz = np.array([max(abs(f.height), 0.01) for f in fs], dtype=np.float64)
        return _green_columns(
            fr, fz, fw, srz_r, srz_z, srz_ang, is_flux, src_dr=fdr, src_dz=fdz
        )

    # --- KNOWN block: ONE column per driven current source (per amc channel) ---
    # The EFIT fcoil model represents each physical PF coil with >1 circuit (a
    # fine interior grid + a coarse corner set), each ALREADY normalised to the
    # FULL coil current (Σxmult = 1).  Applying all of them with the same amc
    # current double-counts the coil (confirmed: a near-vacuum slice showed a
    # ~2× over-prediction on coil-dominated probes, and the redundant columns
    # agree to <1%).  We therefore MERGE same-amc circuits into one column by
    # AVERAGING their (full-coil) responses → the coil current is applied once.
    # A "known_case" circuit has its OWN dedicated amc_channel (its measured
    # case current, never the co-located active coil's) so it always lands in
    # its own column here — never merged with the active coil it sits next to.
    pf_by_chan: dict[str, list[int]] = {}
    stated_weight: dict[str, bool] = {}
    for cc in classes:
        if cc.role in _KNOWN_ROLES:
            pf_by_chan.setdefault(cc.amc_channel, []).append(cc.circuit)
            stated_weight[cc.amc_channel] = (
                stated_weight.get(cc.amc_channel, False) or cc.source_stated_weight
            )

    pf_circuits: list[int] = []  # representative (lowest) circuit per coil column
    pf_amc: list[str] = []
    pf_merged_circuits: list[list[int]] = []  # the circuits averaged into each col
    pf_cols: list[np.ndarray] = []
    for chan in sorted(pf_by_chan):
        circs = sorted(pf_by_chan[chan])
        cols = [_circ_col(c) for c in circs]
        merged = np.mean(cols, axis=0)  # redundant representations → average
        if chan == "sol_current" and not stated_weight.get(chan, False):
            # The solenoid's amp-turns-per-ampere is not stated by this source,
            # so the column carries the vacuum-measured response correction that
            # stands in for it (SOLENOID_RESPONSE_SCALE).  A source that states
            # the weight has already answered the question the correction was
            # measured to answer, and applying both counts it twice.
            merged = merged * SOLENOID_RESPONSE_SCALE
        pf_circuits.append(circs[0])
        pf_amc.append(chan)
        pf_merged_circuits.append(circs)
        pf_cols.append(merged)

    # --- INFERRED passive block: one column per passive(structural) circuit ---
    # A circuit with a wired PolygonSection uses the exact shaped Urankar kernel
    # in place of its axis-aligned bounding-box cylinder (opt-in — an empty
    # table.polygon_sections leaves every column byte-identical).
    poly_by_circ = {ps.circuit: ps for ps in table.polygon_sections}
    passive_rz: list[tuple[float, float]] = []
    passive_cols: list[np.ndarray] = []
    for cc in classes:
        if cc.role not in _KNOWN_ROLES:
            passive_rz.append((cc.centroid_r, cc.centroid_z))
            ps = poly_by_circ.get(cc.circuit)
            if ps is not None:
                passive_cols.append(
                    polygon_section_column(
                        ps.vertices, ps.xmult, srz_r, srz_z, srz_ang, is_flux
                    )
                )
            else:
                passive_cols.append(_circ_col(cc.circuit))

    g_pf = (
        np.column_stack(pf_cols)
        if pf_cols
        else np.zeros((srz_r.size, 0), dtype=np.float64)
    )
    g_passive = (
        np.column_stack(passive_cols)
        if passive_cols
        else np.zeros((srz_r.size, 0), dtype=np.float64)
    )

    # --- INFERRED plasma block: one column per limiter-masked grid node ---
    pr, pz = _default_plasma_basis(table)
    plasma_cols = [
        _green_columns(
            np.array([r]),
            np.array([z]),
            np.array([1.0]),
            srz_r,
            srz_z,
            srz_ang,
            is_flux,
        )
        for r, z in zip(pr, pz, strict=True)
    ]
    g_plasma = (
        np.column_stack(plasma_cols)
        if plasma_cols
        else np.zeros((srz_r.size, 0), dtype=np.float64)
    )

    physical_digest = ""
    if resolve_identity:
        from imas_ambix.gs.machine_identity import (  # noqa: PLC0415
            MachineIdentityError,
            identity_for_table,
        )

        try:
            physical_digest = identity_for_table(table).physical_digest
        except MachineIdentityError, ImportError, OSError, ValueError:
            physical_digest = ""

    return ForwardOperator(
        signature_key=table.signature.key,
        physical_digest=physical_digest,
        sensor_channels=channels,
        sensor_kind=kinds,
        g_pf=g_pf,
        g_plasma=g_plasma,
        g_passive=g_passive,
        pf_circuits=pf_circuits,
        pf_amc_channels=pf_amc,
        pf_merged_circuits=pf_merged_circuits,
        plasma_rz=np.column_stack([pr, pz]) if pr.size else np.zeros((0, 2)),
        passive_rz=np.array(passive_rz, dtype=np.float64)
        if passive_rz
        else np.zeros((0, 2)),
        circuit_classes=classes,
        excluded_channels=excluded,
        flagged_channels=flagged,
        r0=table.r0,
        minor_radius=table.minor_radius,
    )


# --- Raw amc current read (the KNOWN-term signal data) ----------------

_AMC_SKIP = ("time", "timesec", "status")


def read_amc_currents_at_index(shot_id: int, t_index: int) -> dict[str, float]:
    """Read RAW amc coil/plasma currents at one time index (single slice).

    Returns ``{channel: value}`` in the RAW stored units (``kA · turn`` for
    coils, ``kA`` for plasma).  This is the only signal DATA the operator reads —
    the KNOWN PF-coil source term.  Plasma current is NOT consumed as a known
    source (it is INFERRED) but is returned for normalization/diagnostics.
    NEVER reads efm/amm computed currents.
    """
    import zarr  # noqa: PLC0415

    root = local_shot_path(shot_id, tier="level1")
    store: Any = zarr.open(str(root), mode="r")
    amc = store["amc"]
    out: dict[str, float] = {}
    for name in amc.array_keys():
        if name in _AMC_SKIP:
            continue
        arr = amc[name]
        if arr.ndim != 1 or arr.shape[0] == 0:
            continue
        idx = min(int(t_index), arr.shape[0] - 1)
        out[name] = float(arr[idx])
    return out


# --- Artifact I/O -----------------------------------------------------


def build_all_operators(
    tables: dict[str, GeometryTable],
    *,
    resolve_identity: bool = False,
) -> dict[str, ForwardOperator]:
    """Build one :class:`ForwardOperator` per campaign signature."""
    return {
        key: build_operator(t, resolve_identity=resolve_identity)
        for key, t in tables.items()
    }


def write_operator_summary(
    operators: dict[str, ForwardOperator],
    tables: dict[str, GeometryTable] | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write the compact committed operator summary (shapes, coil map, flags).

    ``tables`` (optional) supplies each campaign's amc channel list so the
    summary can surface the unmapped amc sibling channels at mapped coils.
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    out_path = out_path or (
        _Path(__file__).parent / "artifacts" / "gs_operator_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "gs-operator-summary-v0",
        "coil_model_version": COIL_MODEL_VERSION,
        "latent_to_psi_representation": "current-distribution-greens",
        "si_denorm": "raw-SI (Wb, T, A; mu0 carried; amc kA*turn x1000)",
        "amm_currents": "EXCLUDED (EFIT-wall-model output, never a known source)",
        "passive_currents": (
            "INFERRED nuisance (fcoil structural circuits, not amm-read)"
        ),
        "plasma_current": (
            "INFERRED (low-dim jphi basis); measured Ip is a solver constraint,"
            " not a known term"
        ),
        "known_pf_merge": (
            "each physical PF coil is represented by >1 fcoil circuit (redundant "
            "fine+coarse discretisations, each Σxmult=1); same-amc circuits are "
            "AVERAGED into one G_pf column so the coil current is applied once "
            "(a near-vacuum slice showed ~2x over-prediction without the merge)."
        ),
        "r0": MAST_R0,
        "minor_radius": MAST_A,
        "n_campaigns": len(operators),
        "campaigns": {
            k: o.to_summary(
                amc_channels=tables[k].amc_current_channels
                if tables and k in tables
                else None
            )
            for k, o in operators.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# --- amm / fcoil passive double-representation reconciliation ----------


def passive_amm_coincidence(
    table: GeometryTable, tol_m: float = 0.02
) -> dict[str, Any]:
    """Diagnose the fcoil-structural vs amm passive-geometry double-representation.

    The INFERRED passive columns use the fcoil structural-circuit geometry (a
    single source).  This reports how many of those circuits coincide with an
    ``amm`` passive-structure ``(R, Z)`` within ``tol_m`` — documenting the
    overlap so a downstream consumer does not double-count passive elements.
    ``amm`` current VALUES are never read (adjudication).
    """
    classes = classify_circuits(
        table.pf_filaments,
        table.amc_current_channels,
        table.active_circuits,
        table.circuit_drives,
    )
    passive = [c for c in classes if c.role == "inferred_passive"]
    amm = np.array([[p.r, p.z] for p in table.passive_structures], dtype=np.float64)
    n_coin = 0
    if amm.size:
        for c in passive:
            d = np.hypot(amm[:, 0] - c.centroid_r, amm[:, 1] - c.centroid_z)
            if float(d.min()) <= tol_m:
                n_coin += 1
    return {
        "n_inferred_passive_circuit": len(passive),
        "n_amm_passive_structure": len(table.passive_structures),
        "n_coincident_within_tol": n_coin,
        "tol_m": tol_m,
        "passive_geometry_source": "fcoil-structural-circuits",
        "note": (
            "passive/eddy currents are INFERRED on the fcoil structural circuits"
            " (single source); amm geometry overlaps but amm CURRENTS are excluded"
            " (EFIT-wall-model output, never a known source)."
        ),
    }
