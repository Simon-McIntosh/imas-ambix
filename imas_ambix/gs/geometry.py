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
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

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
    """A per-campaign fingerprint of the efm static geometry.

    Two shots share a signature iff their valid sensor / filament positions
    and counts agree to the rounding precision.  Used as the table key so
    geometry from different campaigns is never silently mixed.
    """

    n_bprobe: int
    n_fluxloop: int  # valid (finite) silop entries only
    n_pf_filament: int
    n_limiter: int
    digest: str  # 16-hex hash of the rounded position arrays

    @property
    def key(self) -> str:
        return (
            f"mp{self.n_bprobe}-fl{self.n_fluxloop}-"
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


# --- The per-campaign geometry table ----------------------------------


@dataclass
class GeometryTable:
    """Machine geometry for one MAST campaign (one :class:`SetupSignature`).

    Holds the efm static geometry (B-probes, flux loops, PF filaments, limiter)
    plus the amb-sensor → geometry mapping and the enumerated current-source
    channel names (amc PF/plasma, amm passive).  The operator (T2) consumes
    this; T1 only tabulates.
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

    # ---- summary helpers ----

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
            "coverage": self.coverage(),
        }
        return d


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
    ``(R, Z)`` alone is degenerate there and grabs the wrong orientation.

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
            cand = np.arange(mr.size) if exp is None else np.where(mang == exp)[0]
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
            if exp is not None and abs(float(mang[i]) - exp) > 1.0:
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


# --- Build one table for a representative shot ------------------------


def build_table_for_shot(shot_id: int) -> GeometryTable:
    """Build a :class:`GeometryTable` from one representative shot.

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
    pf_filaments = [
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

    lim_r = _finite(geom["limiterr"])
    lim_z = _finite(geom["limiterz"])
    n_lim = min(lim_r.size, lim_z.size)

    amb_channels = read_amb_channels(shot_id)
    sensor_map, unmatched = map_amb_sensors(geom, amb_channels)

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
    )


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
        except (KeyError, FileNotFoundError, OSError, ValueError):
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

    One representative shot per campaign is used to build the full table; the
    table's ``shots`` list records every in-use shot found with that signature.
    """
    groups = discover_signatures(shot_ids)
    tables: dict[str, GeometryTable] = {}
    for key, (_sig, shots) in groups.items():
        rep = shots[0]
        table = build_table_for_shot(rep)
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
                "shot_range_sampled": [min(t.shots), max(t.shots)]
                if t.shots
                else [],
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
