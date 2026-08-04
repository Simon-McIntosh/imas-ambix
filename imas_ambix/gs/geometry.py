"""Per-campaign machine-geometry table for the GS Green's-function operator.

Stage-2 grounds the Stage-1 latent in raw magnetics.  The Green's-function
forward operator (T2) predicts the ``amb`` flux-loops + B-probes from a
toroidal current distribution; doing so needs the *machine geometry* — every
sensor's ``(R, Z)`` + orientation, the PF-coil + passive-structure filament
``(R, Z, turns)``, and the limiter contour.  That geometry is the one piece
**not** present in the raw signal data, so we tabulate it here.

Geometry source — ``gs-geometry-source = efm-static-geometry`` (LOCKED, user-
cleared 2026-05-30)
-------------------------------------------------------------------------------
We read **only** EFIT's STATIC SETUP / machine-description arrays — the fixed
*a-priori* every solver is *given*, categorically distinct from EFIT's
*reconstructed* output.  The discriminator is mechanical and auditable:

* a setup-geometry array is indexed by a **geometry dimension** (``magpr_n``,
  ``fcoil_segs_n``, ``nlimiter``) and has **no leading time axis** (the EFIT
  time base is length 50 in this corpus);
* a solver-output array has a **leading-50 time axis** (``f(A)`` / ``f(B)``
  time bases) or is a derived reconstructed scalar/profile.

The arrays we read (and ONLY these), with their classification proof:

==================  =======  ========================================
efm array           shape    why it is setup-geometry (not output)
==================  =======  ========================================
magpr_r             (78,)    B-probe R, indexed by magpr_n; no time axis
magpr_z             (78,)    B-probe Z, indexed by magpr_n; no time axis
magpr_ang           (78,)    B-probe orientation (deg), indexed by magpr_n
magpr_len           (78,)    B-probe effective length, indexed by magpr_n
silop_r             (46,)    flux-loop R, indexed by magpr_n; no time axis
silop_z             (46,)    flux-loop Z, indexed by magpr_n; no time axis
fcoil_r             (1004,)  PF filament R centroid, indexed by fcoil_segs_n
fcoil_z             (1004,)  PF filament Z centroid, indexed by fcoil_segs_n
fcoil_turns         (1004,)  PF filament turns, indexed by fcoil_segs_n
fcoil_width         (1004,)  PF filament R extent (finite-size, optional)
fcoil_height        (1004,)  PF filament Z extent (finite-size, optional)
fcoil_circ          (1004,)  circuit number each filament belongs to (1..167)
fcoil_xmult         (1004,)  weight of each filament's share of coil current
limiterr            (37,)    limiter R contour, indexed by nlimiter
limiterz            (37,)    limiter Z contour, indexed by nlimiter
==================  =======  ========================================

**EXCLUDED (solver output — never read as input or label):** ``magpr_c`` /
``magpr_x`` / ``silop_c`` / ``silop_x`` / ``fcoil_c`` / ``fcoil_x`` (all
``(50, N)`` fitted/experimental currents — the tempting prefix-neighbours of
the arrays above), ``psirz``, ``pprime``, ``ffprime``, ``lcfs_r/z``, ``qpsi_c``,
``plasma_current_*``, ``magnetic_axis_*``, ``betap``, ``li``, and every other
``(50, …)`` reconstructed quantity.

Orientation is resolved **authoritatively from ``magpr_ang``**, never from the
``amb`` channel name/description.  This matters: the ``amb`` ``obv*`` channels
carry "Br" in their description (a copy-paste bug) yet ``magpr_ang = 90``
confirms they are *vertical* probes.  See :func:`map_amb_sensors`.

Per-campaign keying
-------------------
The EFIT setup is not constant across the corpus.  Within the in-use S7 shot
set the ``fcoil`` discretisation is 1004 *or* 938 filaments, and ``magpr_z``
drifts up to ~13 mm between campaigns (``silop`` positions are stable).  A
single global table would silently mix incompatible geometries, so every table
is keyed by a :class:`SetupSignature` — a hash of the rounded valid sensor /
filament positions + counts.  ``silop`` arrays are sometimes zero-padded to 78
with trailing ``NaN``; only the valid (finite) entries count.

Dimensionless framing (open: ``extrapolation-coordinates``)
-----------------------------------------------------------
The decision on dimensionless coordinates (R/R0, ψ-normalisation, …) is **not**
resolved here.  The table carries the raw machine-absolute ``(R, Z, …)`` plus
the fixed MAST major/minor-radius constants (:data:`MAST_R0`, :data:`MAST_A`)
needed to *later* express geometry in dimensionless groups, without locking
any particular framing now.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from imas_ambix.data.paths import MANIFEST_DIR, local_shot_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

# --- Fixed MAST machine constants (for later dimensionless framing) ----
#
# These are device constants, not per-shot reconstructions.  They let a
# downstream consumer express geometry in dimensionless coordinates (R/R0,
# n/n_Greenwald via the minor radius, q*) WITHOUT re-introducing any EFIT
# dependency.  We carry them in the artifact but do NOT lock the
# `extrapolation-coordinates` decision here.

MAST_R0 = 0.85
"""MAST nominal plasma major radius [m] (device constant, not per-shot).

A *nominal* value, NOT derived from the limiter contour — a downstream
consumer that needs a geometry-exact R0/a should derive it from the table's
``limiter_r`` / ``limiter_z`` instead.  Carried only so a later dimensionless
framing has a fixed reference; the ``extrapolation-coordinates`` decision is
left open."""

MAST_A = 0.65
"""MAST nominal plasma minor radius [m] (device constant; the quantity used by
the Greenwald density limit).  Nominal, not a limiter dimension — see the
:data:`MAST_R0` note on deriving a geometry-exact value from the contour."""

# --- efm static-geometry array names (the auditable read-list) --------

EFM_GEOMETRY_ARRAYS: tuple[str, ...] = (
    "magpr_r",
    "magpr_z",
    "magpr_ang",
    "magpr_len",
    "silop_r",
    "silop_z",
    "fcoil_r",
    "fcoil_z",
    "fcoil_turns",
    "fcoil_width",
    "fcoil_height",
    "fcoil_circ",
    "fcoil_xmult",
    "limiterr",
    "limiterz",
)
"""The ONLY efm arrays this module reads — all static setup geometry.

Auditable against the strict ``gs-geometry-source`` boundary: every entry has
no leading time axis and is indexed by a geometry dimension.  This module
never opens any other efm array (no ``psirz``, no ``*_c`` / ``*_x`` fitted
currents, no profiles, no shape parameters).
"""

GEOMETRY_TABLE_VERSION = "amb-schema-canonical-v3"
"""Bump whenever the derivation of :class:`GeometryTable` (its sensor channel
SET in particular) changes for a FIXED efm geometry / signature digest — a
downstream cache keyed only on ``SetupSignature.key`` would otherwise silently
mix tables built under different derivations.  ``v1``: the sensor channel set
is geometry-determined (see :func:`canonical_amb_channels`) rather than an
artifact of one shot's own ``amb`` zarr schema.  ``v2``: filled rectangular
coil packs are collapsed to one thick-cylinder filament each
(:func:`collapse_rectangular_circuits`) — same finite-cross-section field,
fewer sources.  ``v3``: the eight slanted vessel/P2 passive sections
(crowns + P2 arms/divertor plates) carry their true parallelogram cross-section
as a :class:`PolygonSection` (:func:`mast_slanted_polygon_sections`) instead of
the axis-aligned bounding box — same area (hence same ring resistance), shaped
field. A downstream ``.npz`` cache keyed only on ``SetupSignature.key`` (the
passive circuit system) must be rebuilt across this bump.
"""

# Sensor-name → expected orientation (degrees).  Bv (vertical) probes read the
# Z-component of B; Br (radial) probes read the R-component.  We use the name
# ONLY to restrict the magpr candidate set so co-located obr/obv pairs map to
# the correct member; the STORED orientation always comes from magpr_ang.
_VERTICAL_PREFIXES = ("ccbv", "obv")
_RADIAL_PREFIXES = ("obr",)

# amb channel-description "r=..., z=..." parser (description used as a lookup
# KEY only; stored R,Z,ang come from efm).
_RZ_RE = re.compile(r"r\s*=\s*(-?\d+\.?\d*)\s*,?\s*z\s*=\s*(-?\d+\.?\d*)", re.I)

# Match flag thresholds.
_AMBIGUOUS_M = 5e-3  # 2nd-nearest within 5 mm of nearest → ambiguous
_MAX_RESIDUAL_M = 0.05  # nearest > 50 mm → unmatched (no plausible sensor)

# How far a stored orientation may sit from the angle a channel name implies and
# still describe the same sensitive axis.  A poloidal set separates its axes by
# 90°, so any band far below that resolves the same radial/vertical pair; the
# band exists because a stored angle is only as exact as its source's units — a
# value held in radians and converted to degrees lands a fraction of a degree
# off a whole number, which is a rounding artefact and not a different axis.
_ORIENTATION_MATCH_DEG = 1.0

# amb columns that are NOT sensors (time bases / status flags).  These vary by
# campaign — later campaigns add a ``timesec`` array — so exclude by name.
_AMB_NON_SENSOR = ("time", "timesec", "status")


# --- Small geometry records -------------------------------------------


@dataclass(frozen=True)
class BProbe:
    """A magnetic field probe: ``(R, Z)``, orientation angle, effective length."""

    index: int  # index into the efm magpr_* arrays
    r: float
    z: float
    angle_deg: float  # authoritative orientation from magpr_ang
    length: float


@dataclass(frozen=True)
class FluxLoop:
    """A poloidal flux loop: ``(R, Z)`` (no orientation)."""

    index: int  # index into the efm silop_* arrays
    r: float
    z: float


@dataclass(frozen=True)
class PFFilament:
    """A PF-coil filament element: centroid ``(R, Z)``, turns, circuit, weight."""

    r: float
    z: float
    turns: float
    width: float
    height: float
    circuit: int
    xmult: float


def collapse_rectangular_circuits(
    filaments: list[PFFilament],
    *,
    pos_tol: float = 0.01,
    mom_tol: float = 0.15,
    floor: float = 0.01,
) -> list[PFFilament]:
    """Replace a circuit's filament lattice with one equivalent thick-cylinder
    filament ONLY when that single rectangle reproduces the lattice's field.

    A coil pack described as a lattice of many co-current filaments has the
    SAME far and near field as a single finite-cross-section cylinder carrying
    the summed current (the analytic double-Newton / Biot-Savart integral is
    over the whole cross-section either way) -- but one cylinder is O(N) cheaper
    to evaluate.  The equivalence holds only when the filaments actually TILE a
    uniform rectangle; it fails for a staggered / ragged / hollow pack, where a
    single bounding-box rectangle would mis-place the current or over-state the
    cross-section.  The gate is therefore a FIELD-FIDELITY (multipole-match)
    check, not a crude area-fill count — raw area fill conflates benign
    inter-winding insulation gaps (a regular grid is still field-equivalent to
    a uniform rectangle) with a genuine shape feature (which is not).

    A circuit collapses iff all three hold:
      * all filaments carry same-sign weight (a uniform pack), AND
      * the current-weighted centroid coincides with the bounding-box centre to
        within ``pos_tol`` of the box size (dipole match — rejects staggered
        packs whose current sits off-centre), AND
      * the current-weighted second moments ⟨Δr²⟩, ⟨Δz²⟩ (including each
        filament's own w²/12, h²/12) match those of a uniform filled rectangle
        of the bounding box (``W²/12``, ``H²/12``) to within ``mom_tol``
        (quadrupole match — rejects hollow frames and gapped packs whose current
        is not uniformly spread over the box).
    Anything that fails is left as its exact filament lattice (the ground-truth
    field), the ``Σ w·greens`` the operator sums over the circuit regardless.

    The single cylinder takes the bounding-box centre and extents, ``xmult`` =
    Σ``xmult`` (so ``Σ w·greens`` is preserved), and ``turns`` = Σ``turns``;
    because it collapses only when the centroid is box-centred, box centre and
    current centroid coincide.  Single-filament circuits pass through unchanged.
    ``floor`` is the physical size floor (matches the coil-column build) used
    when a pack extent is thin.
    """
    by_circ: dict[int, list[PFFilament]] = {}
    order: list[int] = []
    for f in filaments:
        if f.circuit not in by_circ:
            order.append(f.circuit)
        by_circ.setdefault(f.circuit, []).append(f)

    out: list[PFFilament] = []
    for circ in order:
        fs = by_circ[circ]
        if len(fs) == 1:
            out.append(fs[0])
            continue
        xm = np.array([f.xmult for f in fs], dtype=np.float64)
        if not (np.all(xm >= 0.0) or np.all(xm <= 0.0)):
            out.extend(fs)  # mixed-sign pack (e.g. a wound reversal) — keep
            continue
        r = np.array([f.r for f in fs], dtype=np.float64)
        z = np.array([f.z for f in fs], dtype=np.float64)
        w = np.array([max(abs(f.width), floor) for f in fs], dtype=np.float64)
        h = np.array([max(abs(f.height), floor) for f in fs], dtype=np.float64)
        r_lo, r_hi = (r - w / 2).min(), (r + w / 2).max()
        z_lo, z_hi = (z - h / 2).min(), (z + h / 2).max()
        box_w, box_h = r_hi - r_lo, z_hi - z_lo
        if box_w <= 0.0 or box_h <= 0.0:
            out.extend(fs)  # degenerate box — keep the filament lattice
            continue
        r_c, z_c = 0.5 * (r_lo + r_hi), 0.5 * (z_lo + z_hi)
        # dipole match: current-weighted centroid vs bounding-box centre
        wsum = float(xm.sum())
        cw_r = float((xm * r).sum() / wsum)
        cw_z = float((xm * z).sum() / wsum)
        pos_off = max(abs(cw_r - r_c) / box_w, abs(cw_z - z_c) / box_h)
        # quadrupole match: current-weighted 2nd moment about box centre
        # (incl. each filament's own w²/12, h²/12) vs a uniform filled box
        mom_r = float((xm * ((r - r_c) ** 2 + w**2 / 12.0)).sum() / wsum)
        mom_z = float((xm * ((z - z_c) ** 2 + h**2 / 12.0)).sum() / wsum)
        mom_dev = max(
            abs(mom_r / (box_w**2 / 12.0) - 1.0),
            abs(mom_z / (box_h**2 / 12.0) - 1.0),
        )
        if pos_off > pos_tol or mom_dev > mom_tol:
            # staggered / hollow / gapped pack — a single rectangle would
            # mis-place or mis-size the current; keep the exact lattice
            out.extend(fs)
            continue
        out.append(
            PFFilament(
                r=r_c,
                z=z_c,
                turns=float(sum(f.turns for f in fs)),
                width=box_w,
                height=box_h,
                circuit=circ,
                xmult=wsum,
            )
        )
    return out


@dataclass(frozen=True)
class PassiveStructure:
    """A passive (vessel) structure element with ``(R, Z)`` parsed from amm.

    The amm current *values* are EFIT-wall-model OUTPUTS (the amm group's own
    description: "calculated induced currents in toroidal vessel elements for
    input to EFIT").  T1 captures only the geometry ``(R, Z)``; whether the amm
    currents may be used as a "known source" by the operator (T2) is flagged
    for the orchestrator — it bears on the never-efm principle.
    """

    name: str
    r: float
    z: float
    obsolete: bool


@dataclass(frozen=True, eq=False)
class PolygonSection:
    """A conductor cross-section as an (R, Z) polygon — the faithful field source
    for a slanted / trapezoidal / hollow section, evaluated with the analytic
    Urankar Part V kernel (:func:`imas_ambix.gs.polygon.polygon_greens`) instead
    of an axis-aligned bounding box or a Riemann-limited multi-filament tiling.

    Keyed to the fcoil ``circuit`` whose bounding-box representation it REPLACES
    in the forward operator (an opt-in override — a table with no polygon
    sections builds byte-identically to before).  ``vertices`` are the (n, 2)
    corners in either orientation, no repeated closing vertex.  ``xmult`` is the
    current-share weight and MUST equal the replaced circuit's summed ``xmult``
    so the column's per-amplitude scaling is unchanged (only the shape differs).
    """

    circuit: int
    vertices: np.ndarray  # (n, 2) (R, Z) corners
    xmult: float = 1.0
    name: str = ""


@dataclass(frozen=True)
class CircuitDrive:
    """The measured channel that supplies one circuit, and how hard it drives it.

    A source that publishes an electrical description says which channel feeds
    which conductor and what one ampere of it means there, because that
    conversion differs per channel: a channel measuring a conductor's own
    current drives its turn count, one already multiplied out before publication
    drives one ampere turn per ampere, and one feeding parallel branches splits
    between them.  Reconstructing that from position and a turn count is what
    :func:`~imas_ambix.gs.operator.classify_circuits` has to do for a source
    that states none of it; where a source does state it, this carries the
    statement through unchanged.

    ``ampere_turns_per_ampere`` is the TOTAL the circuit carries per ampere of
    ``channel``, and it is already folded into the circuit's per-filament
    :attr:`PFFilament.xmult` -- how it divides between a multi-element conductor
    is a property of the elements, not of the channel.  It is retained here
    because a weight is a claim about the machine, and a consumer comparing two
    descriptions needs the number the source stated rather than a product it has
    to unpick.  ``evidence`` records how that claim was arrived at, so a fitted
    weight stays distinguishable from a measured one.

    A stated weight SUPERSEDES both the conductor's turn count and any
    calibration a consumer derived to correct another source's turn count; see
    :data:`~imas_ambix.gs.operator.SOLENOID_RESPONSE_SCALE`.
    """

    circuit: int
    channel: str
    ampere_turns_per_ampere: float
    evidence: str = ""
    conductor: str = ""


def parallelogram_vertices(
    r: float, z: float, width: float, height: float, angle_deg: float
) -> np.ndarray:
    """(4, 2) corners of a parallelogram cross-section centred at ``(r, z)``.

    ``width`` / ``height`` are the radial / vertical extents; ``angle_deg`` tilts
    the two vertical (side) edges away from the Z-axis, so ``angle_deg = 0`` is an
    axis-aligned rectangle and the top / bottom edges stay horizontal — the crown
    / P2-arm / stability-plate shape.  A corner at ±height/2 shifts radially by
    ``(height/2)·tan(angle_deg)``; the enclosed area stays ``width·height``.
    Turns the MAST data-catalog ``*_r/_z/_width/_height/_shapeAngle``
    parametrisation of the slanted passives into explicit vertices for a
    :class:`PolygonSection`.
    """
    dr = 0.5 * height * np.tan(np.deg2rad(angle_deg))
    hw, hh = 0.5 * width, 0.5 * height
    return np.array(
        [
            (r - hw - dr, z - hh),
            (r + hw - dr, z - hh),
            (r + hw + dr, z + hh),
            (r - hw + dr, z + hh),
        ]
    )


def shaped_section_vertices(
    r: float, z: float, width: float, height: float, angle1: float, angle2: float
) -> np.ndarray:
    """(4, 2) parallelogram corners under the MAST Data Catalog shape-angle rule.

    Transcribes the ``pf_passive`` vertex convention the catalog publishes for a
    sheared section (``*_shapeAngle1`` / ``*_shapeAngle2`` in degrees):
    ``angle1`` tilts the top/bottom edges (a Z-shear that grows with the radial
    offset), ``angle2`` tilts the side edges (an R-shear that grows with the
    vertical offset); ``0`` for both is an axis-aligned rectangle.  The enclosed
    area stays ``width·height`` for either shear, so the conductor's true
    cross-section — and therefore its ring resistance — is unchanged from the
    bounding box; only the field distribution shifts.

    The ``angle1 = 0`` case is exactly :func:`parallelogram_vertices` at
    ``angle_deg = 90 − angle2`` (verified to machine precision), the pure side-
    edge shear the crown sections use; the transposed ``angle2 = 0`` branch (the
    P2-arm shear) is not expressible through that R-shear helper, hence this
    faithful two-angle transcription.
    """
    dr, dz = abs(width), abs(height)
    a1t = np.tan(np.deg2rad(angle1)) if angle1 > 0 else 0.0
    a2t = 1.0 / np.tan(np.deg2rad(angle2)) if angle2 > 0 else 0.0
    rr = np.array(
        [
            r - dr / 2 - dz / 2 * a2t,
            r + dr / 2 - dz / 2 * a2t,
            r + dr / 2 + dz / 2 * a2t,
            r - dr / 2 + dz / 2 * a2t,
        ]
    )
    zz = np.array(
        [
            z - dz / 2 - dr / 2 * a1t,
            z - dz / 2 + dr / 2 * a1t,
            z + dz / 2 + dr / 2 * a1t,
            z + dz / 2 - dr / 2 * a1t,
        ]
    )
    return np.column_stack([rr, zz])


#: The eight slanted MAST in-vessel passive sections — the vessel end-column
#: crowns and the P2 arm / divertor-plate structures — as published by the MAST
#: Data Catalog ``pf_passive`` group (authoritative machine geometry, constant
#: across campaigns).  Each entry is ``(name, ref_r, ref_z, shapeAngle1,
#: shapeAngle2)`` in metres / degrees; the reference centroid matches the entry
#: to its ``inferred_passive`` fcoil circuit, and the shear angles reshape that
#: circuit's box into its true parallelogram.  The remaining ~70 passive
#: sections are genuine axis-aligned rectangles (both shape angles zero) and
#: need no override.
_MAST_SLANTED_PASSIVES: tuple[tuple[str, float, float, float, float], ...] = (
    ("botcol", 0.2354, -2.0250, 0.0, 295.324),
    ("topcol", 0.2356, 2.0250, 0.0, 64.676),
    ("p2larm", 0.3532, -1.6308, 45.0, 0.0),
    ("p2larm_out", 0.6827, -1.6506, 0.0, 320.0),
    ("p2ldivpl", 0.6198, -1.6337, 320.0, 0.0),
    ("p2uarm", 0.3532, 1.6308, 315.0, 0.0),
    ("p2uarm_out", 0.6827, 1.6506, 0.0, 40.0),
    ("p2udivpl", 0.6198, 1.6337, 40.0, 0.0),
)

#: match tolerance between a catalog reference centroid and an fcoil circuit
#: centroid [m] — the eight elements match their circuit to < 1 mm; a loose
#: 1 cm guard tolerates minor per-campaign discretisation drift while never
#: mis-attaching to a neighbouring structure.
_SLANTED_MATCH_TOL_M = 0.01


def mast_slanted_polygon_sections(
    pf_filaments: list[PFFilament],
) -> list[PolygonSection]:
    """:class:`PolygonSection` overrides for MAST's eight slanted passives.

    Matches each catalog slanted element (:data:`_MAST_SLANTED_PASSIVES`) to the
    nearest fcoil circuit by centroid and builds its true parallelogram section
    from that circuit's own filament ``(r, z, width, height)`` plus the catalog
    shear angles — co-located with the box it replaces (identical centroid and
    area, so the dipole moment and ring resistance are unchanged; only the shape
    differs).  A circuit that is not a single filament, or whose centroid sits
    beyond :data:`_SLANTED_MATCH_TOL_M`, is skipped (verify-and-flag: never
    fabricate a match).  Empty for a non-MAST filament set.
    """
    by_circ: dict[int, list[PFFilament]] = {}
    for f in pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    centroids = {
        c: (float(np.mean([f.r for f in g])), float(np.mean([f.z for f in g])))
        for c, g in by_circ.items()
    }
    out: list[PolygonSection] = []
    for name, ref_r, ref_z, a1, a2 in _MAST_SLANTED_PASSIVES:
        best_c, best_d = None, np.inf
        for c, (cr, cz) in centroids.items():
            d = float(np.hypot(cr - ref_r, cz - ref_z))
            if d < best_d:
                best_c, best_d = c, d
        if best_c is None or best_d > _SLANTED_MATCH_TOL_M:
            continue
        group = by_circ[best_c]
        if len(group) != 1:
            continue  # slanted passives are single-filament in the efm set
        f = group[0]
        vertices = shaped_section_vertices(
            f.r, f.z, abs(f.width), abs(f.height), a1, a2
        )
        out.append(
            PolygonSection(
                circuit=best_c,
                vertices=vertices,
                xmult=sum(x.xmult for x in group),
                name=name,
            )
        )
    return out


@dataclass(frozen=True)
class SensorMapping:
    """One amb sensor column mapped to its efm geometry index."""

    amb_channel: str
    kind: str  # "b_probe" | "flux_loop"
    efm_index: int
    r: float
    z: float
    angle_deg: float | None  # None for flux loops
    residual_m: float  # nearest-neighbour distance amb-desc(R,Z) → efm
    flag: str  # "" if confidently mapped, else a reason


# --- Setup signature --------------------------------------------------


@dataclass(frozen=True)
class SetupSignature:
    """A per-campaign fingerprint of the static geometry.

    Two shots/entries share a signature iff their valid sensor / filament
    positions and counts agree to the rounding precision.  Used as the table
    key so geometry from different campaigns -- or different machines -- is
    never silently mixed.

    Machine tagging (byte-stability contract, LOCKED)
    --------------------------------------------------
    ``machine`` defaults to ``"mast"`` and :attr:`key` special-cases that
    default to the untagged prefix (``mp{..}-fl{..}-fc{..}-lim{..}-{digest}``)
    -- the exact format every MAST signature has always used.  Every existing
    on-disk cache (``imas_ambix/latent/artifacts/patch_scoping/g_pg_<key>_*``)
    and trained checkpoint resolves through this string, so it MUST stay
    byte-identical for MAST; this field's default plus the ``"mast"``
    special-case in :attr:`key` is what guarantees that (a plain always-on
    prefix would have changed every existing key).  Any other machine gets a
    ``"{machine}-"`` prefix, so a second machine's table can never collide
    with, or accidentally resolve through, a MAST cache entry.
    """

    n_bprobe: int
    n_fluxloop: int  # valid (finite) silop entries only
    n_pf_filament: int
    n_limiter: int
    digest: str  # 16-hex hash of the rounded position arrays
    machine: str = "mast"

    @property
    def key(self) -> str:
        prefix = "" if self.machine == "mast" else f"{self.machine}-"
        return (
            f"{prefix}mp{self.n_bprobe}-fl{self.n_fluxloop}-"
            f"fc{self.n_pf_filament}-lim{self.n_limiter}-{self.digest}"
        )


def _finite(a: np.ndarray) -> np.ndarray:
    """Return the leading run of finite entries (silop is NaN-padded to 78)."""
    a = np.asarray(a, dtype=np.float64)
    out: np.ndarray = a[np.isfinite(a)]
    return out


def _round_hash(arrays: Iterable[np.ndarray], decimals: int = 4) -> str:
    """Stable 16-hex digest of rounded float arrays (campaign fingerprint)."""
    h = hashlib.sha256()
    for a in arrays:
        r = np.round(np.asarray(a, dtype=np.float64), decimals)
        # canonical bytes — round to int counts to avoid -0.0 / +0.0 noise.
        h.update(np.ascontiguousarray(r).tobytes())
    return h.hexdigest()[:16]


def round_geometry_hash(arrays: Iterable[np.ndarray], decimals: int = 4) -> str:
    """Public alias of :func:`_round_hash` for use by other machine readers.

    Any reader building a :class:`SetupSignature` for a non-MAST machine
    should hash its own static position arrays with this (not reinvent a
    digest scheme) so every machine's signature has the same stability
    guarantees (stable under repeat, sensitive to sub-cm drift).
    """
    return _round_hash(arrays, decimals=decimals)


# --- The per-campaign geometry table ----------------------------------


@dataclass
class GeometryTable:
    """Machine geometry for one campaign / machine-description entry.

    Holds the static geometry (B-probes, flux loops, PF filaments, limiter)
    plus the sensor → geometry mapping and the enumerated current-source
    channel names (MAST: amc PF/plasma, amm passive).  The operator (T2)
    consumes this; readers only tabulate.  Produced by any
    :class:`MachineGeometryReader` implementation -- the MAST fields
    (``sensor_map`` identity mapping, ``amc_current_channels``) are a MAST
    reader convention, not a universal requirement; a non-MAST reader may
    leave ``amc_current_channels`` / ``passive_structures`` empty.
    """

    signature: SetupSignature
    shots: list[int]  # in-use shots that carry this signature
    b_probes: list[BProbe]
    flux_loops: list[FluxLoop]
    pf_filaments: list[PFFilament]
    limiter_r: list[float]
    limiter_z: list[float]
    sensor_map: list[SensorMapping]  # amb channel → geometry
    passive_structures: list[PassiveStructure]  # amm candidate sources
    amc_current_channels: list[str]  # amc PF/plasma current channels (names)
    unmatched_amb: list[str]  # amb sensor channels that could not be mapped
    r0: float = MAST_R0
    minor_radius: float = MAST_A
    provenance_flags: list[str] = field(default_factory=list)
    """Honest data-quality caveats a reader could not resolve (verify-and-flag,
    never fabricate) -- e.g. an unsupported element shape dropped, a sentinel
    "empty" value coerced to zero, or a non-axisymmetric sensor approximated
    by a single point.  Always ``[]`` for the MAST reader (nothing to flag);
    populated by :mod:`imas_ambix.gs.imas_geometry` where it applies."""
    active_circuits: list[int] = field(default_factory=list)
    """The circuits the SOURCE states are actively supplied conductors.

    A source that separates its supplied windings from its induced structure --
    an IMAS ``pf_active`` / ``pf_passive`` split, say -- can name the supplied
    ones here, and :func:`~imas_ambix.gs.operator.classify_circuits` then takes
    that statement instead of inferring the split from where a circuit's
    centroid falls.  The inference exists because a source like ``efm``'s
    ``fcoil`` table is one undifferentiated filament list in which a coil's
    structural neighbours are indistinguishable from the winding by position
    alone; where a source does make the distinction, guessing it again can only
    lose information.

    Empty means the source does not distinguish, which leaves the geometric
    classification exactly as it was -- so every existing reader's operator is
    byte-identical."""
    circuit_drives: list[CircuitDrive] = field(default_factory=list)
    """Which measured channel supplies each circuit, where the SOURCE states it.

    This is the stronger form of :attr:`active_circuits`: that says a circuit is
    supplied, this says by what and at what scale.  Classification takes these
    verbatim -- no centroid match, no channel-name convention, no case-id table
    -- because all three exist to reconstruct what a declaration already
    contains, and a source that resolves its structure finely enough to name
    element groups is exactly the source those reconstructions fail on.

    Empty means the source declares no drives, leaving classification as it was.
    """
    polygon_sections: list[PolygonSection] = field(default_factory=list)
    """Analytic polygon cross-sections that REPLACE the axis-aligned bounding box
    of specific fcoil circuits in the forward operator (keyed by
    :attr:`PolygonSection.circuit`).  Empty for every existing reader — the
    operator is byte-identical when this is ``[]`` — and populated only where a
    faithful slanted / trapezoidal / hollow section is wired in."""

    # ---- summary helpers ----

    @property
    def machine(self) -> str:
        """Machine tag, read off :attr:`signature` (never stored twice)."""
        return self.signature.machine

    @property
    def n_pf_circuits(self) -> int:
        return len({f.circuit for f in self.pf_filaments})

    def coverage(self) -> dict[str, int]:
        n_mapped = sum(1 for m in self.sensor_map if not m.flag)
        n_flagged = sum(1 for m in self.sensor_map if m.flag)
        return {
            "n_bprobe": len(self.b_probes),
            "n_fluxloop": len(self.flux_loops),
            "n_pf_filament": len(self.pf_filaments),
            "n_pf_circuit": self.n_pf_circuits,
            "n_limiter": len(self.limiter_r),
            "n_amb_sensor_columns": len(self.sensor_map) + len(self.unmatched_amb),
            "n_amb_mapped": n_mapped,
            "n_amb_flagged": n_flagged,
            "n_amb_unmatched": len(self.unmatched_amb),
            "n_passive_structure": len(self.passive_structures),
            "n_amc_current_channel": len(self.amc_current_channels),
        }

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "signature": asdict(self.signature),
            "signature_key": self.signature.key,
            "shots": self.shots,
            "r0": self.r0,
            "minor_radius": self.minor_radius,
            "b_probes": [asdict(p) for p in self.b_probes],
            "flux_loops": [asdict(p) for p in self.flux_loops],
            "pf_filaments": [asdict(p) for p in self.pf_filaments],
            "limiter_r": self.limiter_r,
            "limiter_z": self.limiter_z,
            "sensor_map": [asdict(m) for m in self.sensor_map],
            "passive_structures": [asdict(p) for p in self.passive_structures],
            "amc_current_channels": self.amc_current_channels,
            "unmatched_amb": self.unmatched_amb,
            "provenance_flags": self.provenance_flags,
            "coverage": self.coverage(),
        }
        return d


# --- Machine-geometry reader interface ---------------------------------
#
# The Green's-function forward operator (:mod:`imas_ambix.gs.operator`) and
# every downstream consumer (:class:`~imas_ambix.latent.patch_basis.PatchBasis`,
# :class:`~imas_ambix.latent.gs_solve.EquilibriumGrid`) depend on
# :class:`GeometryTable` ONLY -- never on how it was produced.  A
# :class:`MachineGeometryReader` is the seam that lets a second (third, ...)
# machine plug in: implement ``read() -> GeometryTable`` against whatever
# machine-native format that device's static geometry lives in, and nothing
# downstream of the table changes.  :class:`MastZarrGeometryReader` below
# adapts the existing FAIR-MAST efm reader to this interface with NO change
# to its read logic; :mod:`imas_ambix.gs.imas_geometry` implements the same
# interface against IMAS pf_active / wall / magnetics static IDSs.


@runtime_checkable
class MachineGeometryReader(Protocol):
    """A machine-specific reader that produces one :class:`GeometryTable`.

    ``machine`` is the tag that becomes :attr:`SetupSignature.machine` (and
    therefore the cache-key prefix); every implementation must set the same
    string on every table it returns.
    """

    machine: str

    def read(self) -> GeometryTable: ...


# --- Low-level efm geometry read (the audited boundary) ---------------


def _open_group(shot_id: int, group: str) -> Any:
    """Open one zarr group of a level-1 shot (FAIR-MAST zarr; never h5py)."""
    import zarr  # noqa: PLC0415

    root = local_shot_path(shot_id, tier="level1")
    store: Any = zarr.open(str(root), mode="r")
    return store[group]


def read_efm_geometry(shot_id: int) -> dict[str, np.ndarray]:
    """Read ONLY the static efm geometry arrays for one shot.

    Reads exactly the arrays in :data:`EFM_GEOMETRY_ARRAYS` — the auditable
    read-list — and no others.  Never touches any reconstructed efm output.
    """
    efm = _open_group(shot_id, "efm")
    out: dict[str, np.ndarray] = {}
    for name in EFM_GEOMETRY_ARRAYS:
        if name in efm:
            out[name] = np.asarray(efm[name][:], dtype=np.float64)
    return out


def setup_signature(geom: dict[str, np.ndarray]) -> SetupSignature:
    """Compute the per-campaign :class:`SetupSignature` from efm geometry."""
    mr = _finite(geom["magpr_r"])
    sr = _finite(geom["silop_r"])
    sz = _finite(geom["silop_z"])
    fr = geom["fcoil_r"]
    lim = geom["limiterr"]
    digest = _round_hash(
        [
            _finite(geom["magpr_r"]),
            _finite(geom["magpr_z"]),
            _finite(geom["magpr_ang"]),
            sr,
            sz,
            fr,
            geom["fcoil_z"],
            geom["fcoil_turns"],
            geom["limiterr"],
            geom["limiterz"],
        ]
    )
    return SetupSignature(
        n_bprobe=int(mr.size),
        n_fluxloop=int(min(sr.size, sz.size)),
        n_pf_filament=int(fr.size),
        n_limiter=int(_finite(lim).size),
        digest=digest,
    )


# --- amb sensor mapping (orientation from magpr_ang, authoritatively) -


def _parse_amb_rz(desc: str) -> tuple[float, float] | None:
    m = _RZ_RE.search(desc or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _expected_angle(channel: str) -> float | None:
    name = channel.lower()
    if name.startswith(_VERTICAL_PREFIXES):
        return 90.0
    if name.startswith(_RADIAL_PREFIXES):
        return 0.0
    return None


def map_amb_sensors(
    geom: dict[str, np.ndarray],
    amb_channels: Sequence[tuple[str, str]],
) -> tuple[list[SensorMapping], list[str]]:
    """Map each amb sensor column to its efm geometry index.

    ``amb_channels`` is a sequence of ``(channel_name, description)``.  The
    description's ``r=…, z=…`` is used **only** as a lookup key; the stored
    ``(R, Z, angle)`` always comes from efm.

    B-probes (``ccbv*`` / ``obr*`` / ``obv*``) map to ``magpr`` with the
    candidate set **restricted to the name-expected orientation subset** so
    co-located radial/vertical pairs (``obrNN`` ⟂ ``obvNN`` at identical
    ``(R, Z)``) resolve to the correct member — plain nearest-neighbour on
    ``(R, Z)`` alone is degenerate there and grabs the wrong orientation.  The
    restriction admits any stored angle within :data:`_ORIENTATION_MATCH_DEG`
    of the expected one, so a source holding its orientations in radians maps
    as well as one holding whole degrees; the two axes are 90° apart, so the
    band cannot merge them.

    Flux loops (``fl_*``) map to ``silop`` by nearest-neighbour (no
    orientation).  amb flux-loop descriptions are known to carry placeholder /
    duplicated ``(R, Z)``; those are flagged rather than force-mapped.

    Returns ``(mappings, unmatched_channel_names)``.
    """
    mr = geom["magpr_r"]
    mz = geom["magpr_z"]
    mang = geom["magpr_ang"]
    sr = _finite(geom["silop_r"])
    sz = _finite(geom["silop_z"])
    # silop_r/silop_z share the leading-finite run.
    n_sil = min(sr.size, sz.size)
    sr, sz = sr[:n_sil], sz[:n_sil]

    mappings: list[SensorMapping] = []
    unmatched: list[str] = []
    # track flux-loop index multiplicity to flag colliding amb descriptions.
    fl_claims: dict[int, list[str]] = {}

    for channel, desc in amb_channels:
        if channel in _AMB_NON_SENSOR:
            continue
        rz = _parse_amb_rz(desc)
        name = channel.lower()
        is_flux = name.startswith("fl")

        if rz is None:
            unmatched.append(channel)
            continue
        r, z = rz

        if not is_flux:
            exp = _expected_angle(channel)
            # restrict the magpr candidate set to the name-expected orientation
            # so co-located radial/vertical probes resolve to the correct
            # member; fall back to all magpr for an unknown B-probe family.
            cand = (
                np.arange(mr.size)
                if exp is None
                else np.where(np.abs(mang - exp) <= _ORIENTATION_MATCH_DEG)[0]
            )
            if cand.size == 0:
                unmatched.append(channel)
                continue
            d = np.hypot(mr[cand] - r, mz[cand] - z)
            order = np.argsort(d)
            i = int(cand[order[0]])
            d0 = float(d[order[0]])
            d1 = float(d[order[1]]) if order.size > 1 else np.inf
            flag = ""
            if d0 > _MAX_RESIDUAL_M:
                unmatched.append(channel)
                continue
            if (d1 - d0) < _AMBIGUOUS_M:
                flag = f"ambiguous: 2nd-nearest within {_AMBIGUOUS_M * 1e3:.0f}mm"
            if exp is not None and abs(float(mang[i]) - exp) > _ORIENTATION_MATCH_DEG:
                flag = (flag + "; " if flag else "") + "name/angle mismatch"
            mappings.append(
                SensorMapping(
                    amb_channel=channel,
                    kind="b_probe",
                    efm_index=i,
                    r=float(mr[i]),
                    z=float(mz[i]),
                    angle_deg=float(mang[i]),
                    residual_m=d0,
                    flag=flag,
                )
            )
        else:
            d = np.hypot(sr - r, sz - z)
            order = np.argsort(d)
            i = int(order[0])
            d0 = float(d[order[0]])
            if d0 > _MAX_RESIDUAL_M:
                # placeholder / displaced amb description — cannot trust.
                unmatched.append(channel)
                continue
            fl_claims.setdefault(i, []).append(channel)
            mappings.append(
                SensorMapping(
                    amb_channel=channel,
                    kind="flux_loop",
                    efm_index=i,
                    r=float(sr[i]),
                    z=float(sz[i]),
                    angle_deg=None,
                    residual_m=d0,
                    flag="",
                )
            )

    # second pass: flag flux loops where >1 amb channel claims one silop index
    # (duplicated amb descriptions — the mapping is not one-to-one there).
    for i, claims in fl_claims.items():
        if len(claims) > 1:
            for m_idx, m in enumerate(mappings):
                if m.kind == "flux_loop" and m.efm_index == i:
                    mappings[m_idx] = SensorMapping(
                        amb_channel=m.amb_channel,
                        kind=m.kind,
                        efm_index=m.efm_index,
                        r=m.r,
                        z=m.z,
                        angle_deg=m.angle_deg,
                        residual_m=m.residual_m,
                        flag=f"non-unique: silop[{i}] claimed by {claims}",
                    )

    return mappings, unmatched


# --- amm passive-structure geometry -----------------------------------

# amm descriptions embed "R=...  Z=..." (note the spaced 2-token form, distinct
# from the amb "r=...,z=..." form); multi-channel arrays carry one per line.
_AMM_RZ_RE = re.compile(r"R\s*=\s*(-?\d+\.?\d*)\s+Z\s*=\s*(-?\d+\.?\d*)")


def read_amm_passive(shot_id: int) -> list[PassiveStructure]:
    """Capture amm passive-structure geometry ``(R, Z)`` from descriptions.

    Reads only the amm array **descriptions** (and array shapes to expand
    multi-channel structures) — never the current time-series values, which
    are an EFIT-wall-model output (flagged for the orchestrator).
    """
    amm = _open_group(shot_id, "amm")
    out: list[PassiveStructure] = []
    for name in sorted(amm.array_keys()):
        if name in ("time", "passnumber", "tcutoff", "tolerance", "substeps"):
            continue
        if name.endswith("_channel"):
            continue
        desc = amm[name].attrs.get("description", "") or ""
        obsolete = "obsolete" in desc.lower()
        matches = _AMM_RZ_RE.findall(desc)
        for k, (r, z) in enumerate(matches):
            label = name if len(matches) == 1 else f"{name}[{k}]"
            out.append(
                PassiveStructure(name=label, r=float(r), z=float(z), obsolete=obsolete)
            )
    return out


def read_amc_current_channels(shot_id: int) -> list[str]:
    """Enumerate amc PF/plasma current channel names (no time-series read)."""
    amc = _open_group(shot_id, "amc")
    skip = {"time", "timesec", "status"}
    return sorted(k for k in amc.array_keys() if k not in skip)


def read_amb_channels(shot_id: int) -> list[tuple[str, str]]:
    """Return amb ``(channel_name, description)`` pairs (no signal-data read)."""
    amb = _open_group(shot_id, "amb")
    out: list[tuple[str, str]] = []
    for name in sorted(amb.array_keys()):
        if name in _AMB_NON_SENSOR:
            continue
        out.append((name, amb[name].attrs.get("description", "") or ""))
    return out


def canonical_amb_channels(
    shot_ids: Sequence[int], *, max_shots: int | None = None
) -> list[tuple[str, str]]:
    """Union of amb ``(channel, description)`` pairs across ``shot_ids``.

    A single shot's ``amb`` zarr group can genuinely lack an array that other
    shots sharing the identical :class:`SetupSignature` digest DO carry — a
    per-shot data-acquisition gap (a channel disabled or dropped for that
    shot), not a geometry difference (the efm arrays the digest hashes are
    unaffected).  Resolving a campaign's sensor channel SET from only one
    shot therefore makes it an artifact of THAT shot's own availability
    (:func:`build_table_for_shot` calling this shot "unlucky" would silently
    lose the channel for the whole campaign).  Scanning several shots and
    taking the union of every discovered ``(name, description)`` pair makes
    the resulting set geometry-determined instead — per-shot absence is left
    to be resolved as a masked-absent value downstream (the raw-signal
    alignment path keys on the GLOBAL ``feature_schema()``, not on any one
    shot's schema, so a channel present in the table but genuinely unread on
    a given shot naturally comes back all-NaN there).

    Cheap: reads only zarr array keys + description attributes, never signal
    data.  ``max_shots`` bounds the scan (``None`` = every shot given).
    """
    seen: dict[str, str] = {}
    ids = list(shot_ids) if max_shots is None else list(shot_ids)[:max_shots]
    for s in ids:
        try:
            chans = read_amb_channels(int(s))
        except KeyError, FileNotFoundError, OSError, ValueError:
            continue
        for name, desc in chans:
            seen.setdefault(name, desc)
    return sorted(seen.items())


# --- Build one table for a representative shot ------------------------


def build_table_for_shot(
    shot_id: int, *, amb_channels: list[tuple[str, str]] | None = None
) -> GeometryTable:
    """Build a :class:`GeometryTable` from one representative shot.

    ``amb_channels`` (optional) overrides the amb channel candidates fed to
    :func:`map_amb_sensors` — pass :func:`canonical_amb_channels` over every
    shot sharing this campaign's signature to get a sensor channel SET that
    does not depend on which single shot happens to build the table (see its
    docstring).  Defaults to this shot's own :func:`read_amb_channels` when
    omitted, preserving the original single-shot behaviour for callers that
    only ever see one shot of a campaign.

    The geometry is per-campaign-constant, so a single shot of a campaign is a
    valid source for that campaign's table.
    """
    geom = read_efm_geometry(shot_id)
    sig = setup_signature(geom)

    mr, mz, mang, mlen = (
        geom["magpr_r"],
        geom["magpr_z"],
        geom["magpr_ang"],
        geom["magpr_len"],
    )
    b_probes = [
        BProbe(
            index=i,
            r=float(mr[i]),
            z=float(mz[i]),
            angle_deg=float(mang[i]),
            length=float(mlen[i]),
        )
        for i in range(mr.size)
        if np.isfinite(mr[i])
    ]

    sr, sz = _finite(geom["silop_r"]), _finite(geom["silop_z"])
    n_sil = min(sr.size, sz.size)
    flux_loops = [
        FluxLoop(index=i, r=float(sr[i]), z=float(sz[i])) for i in range(n_sil)
    ]

    fr, fz, ft = geom["fcoil_r"], geom["fcoil_z"], geom["fcoil_turns"]
    fw = geom.get("fcoil_width", np.zeros_like(fr))
    fh = geom.get("fcoil_height", np.zeros_like(fr))
    fc = geom.get("fcoil_circ", np.zeros_like(fr))
    fx = geom.get("fcoil_xmult", np.ones_like(fr))
    pf_filaments = collapse_rectangular_circuits(
        [
            PFFilament(
                r=float(fr[i]),
                z=float(fz[i]),
                turns=float(ft[i]),
                width=float(fw[i]),
                height=float(fh[i]),
                circuit=int(fc[i]),
                xmult=float(fx[i]),
            )
            for i in range(fr.size)
        ]
    )

    lim_r = _finite(geom["limiterr"])
    lim_z = _finite(geom["limiterz"])
    n_lim = min(lim_r.size, lim_z.size)

    amb_ch = amb_channels if amb_channels is not None else read_amb_channels(shot_id)
    sensor_map, unmatched = map_amb_sensors(geom, amb_ch)

    passive = read_amm_passive(shot_id)
    amc_channels = read_amc_current_channels(shot_id)

    return GeometryTable(
        signature=sig,
        shots=[shot_id],
        b_probes=b_probes,
        flux_loops=flux_loops,
        pf_filaments=pf_filaments,
        limiter_r=lim_r[:n_lim].tolist(),
        limiter_z=lim_z[:n_lim].tolist(),
        sensor_map=sensor_map,
        passive_structures=passive,
        amc_current_channels=amc_channels,
        unmatched_amb=unmatched,
        polygon_sections=mast_slanted_polygon_sections(pf_filaments),
    )


@dataclass(frozen=True)
class MastZarrGeometryReader:
    """Adapts :func:`build_table_for_shot` to :class:`MachineGeometryReader`.

    Pure adapter -- read logic is untouched, so every MAST
    :class:`SetupSignature` this produces is byte-identical to calling
    ``build_table_for_shot`` directly (the pre-existing call path every
    script and cache still uses).
    """

    shot_id: int
    machine: str = "mast"

    def read(self) -> GeometryTable:
        return build_table_for_shot(self.shot_id)


# --- Corpus-wide extraction (campaign discovery + table build) --------


def discover_signatures(
    shot_ids: Iterable[int],
) -> dict[str, tuple[SetupSignature, list[int]]]:
    """Group shots by :class:`SetupSignature` (campaign discovery).

    Reads only the efm static geometry per shot.  Missing / unreadable shots
    are skipped silently (logged by the caller).
    """
    groups: dict[str, tuple[SetupSignature, list[int]]] = {}
    for shot in shot_ids:
        try:
            geom = read_efm_geometry(shot)
            if "magpr_r" not in geom or "fcoil_r" not in geom:
                continue
            sig = setup_signature(geom)
        except KeyError, FileNotFoundError, OSError, ValueError:
            continue
        key = sig.key
        if key not in groups:
            groups[key] = (sig, [])
        groups[key][1].append(int(shot))
    return groups


def extract_campaign_tables(
    shot_ids: Sequence[int],
    sample_per_campaign: int = 1,
) -> dict[str, GeometryTable]:
    """Discover campaigns over ``shot_ids`` and build one table per campaign.

    The sensor channel set is resolved from :func:`canonical_amb_channels`
    over EVERY shot discovered for that signature, not just one representative
    — geometry-determined rather than an artifact of which shot happens to
    build the table (see its docstring).  The representative shot used for the
    non-amb geometry (b-probes, flux loops, PF filaments, limiter) still comes
    from the group; if it individually fails to build (missing amm/amc, say),
    the next candidate in the group is tried before the whole signature is
    given up on.  The table's ``shots`` list records every in-use shot found
    with that signature.
    """
    groups = discover_signatures(shot_ids)
    tables: dict[str, GeometryTable] = {}
    for key, (_sig, shots) in groups.items():
        canonical_amb = canonical_amb_channels(shots)
        table = None
        for rep in shots:
            try:
                table = build_table_for_shot(rep, amb_channels=canonical_amb)
                break
            except KeyError, FileNotFoundError, OSError, ValueError:
                continue
        if table is None:
            continue
        table.shots = sorted(shots)
        tables[key] = table
    return tables


# --- Artifact I/O -----------------------------------------------------


def write_tables(
    tables: dict[str, GeometryTable],
    out_dir: Path | None = None,
) -> Path:
    """Write the full per-campaign geometry tables as JSON under MANIFEST_DIR.

    Returns the path to the written artifact.
    """
    out_dir = out_dir or MANIFEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gs_geometry_tables.json"
    payload = {
        "schema": "gs-geometry-v0",
        "source": "efm-static-geometry",
        "efm_arrays_read": list(EFM_GEOMETRY_ARRAYS),
        "r0": MAST_R0,
        "minor_radius": MAST_A,
        "shots_are_sampled_representatives": True,
        "shots_note": (
            "each campaign's 'shots' is the SAMPLED set found with that "
            "signature, NOT the full in-use population. Match a new shot to a "
            "campaign by recomputing setup_signature(...).key, never by "
            "membership in this list."
        ),
        "n_campaigns": len(tables),
        "campaigns": {k: t.to_dict() for k, t in tables.items()},
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def summarise(tables: dict[str, GeometryTable]) -> dict[str, object]:
    """Build the compact committed summary (counts, keys, unmatched list)."""
    return {
        "schema": "gs-geometry-summary-v0",
        "source": "efm-static-geometry",
        "efm_arrays_read": list(EFM_GEOMETRY_ARRAYS),
        "r0": MAST_R0,
        "minor_radius": MAST_A,
        "shots_are_sampled_representatives": True,
        "shots_note": (
            "n_shots / shot_range count the SAMPLED shots found with each "
            "signature, NOT the full in-use population; match new shots by "
            "recomputing the signature key, not by membership."
        ),
        "n_campaigns": len(tables),
        "campaigns": {
            k: {
                "n_shots_sampled": len(t.shots),
                "shot_range_sampled": [min(t.shots), max(t.shots)] if t.shots else [],
                "example_shots": t.shots[:3],
                "coverage": t.coverage(),
                "flagged_amb": [
                    {"channel": m.amb_channel, "flag": m.flag}
                    for m in t.sensor_map
                    if m.flag
                ],
                "unmatched_amb": t.unmatched_amb,
            }
            for k, t in tables.items()
        },
    }
