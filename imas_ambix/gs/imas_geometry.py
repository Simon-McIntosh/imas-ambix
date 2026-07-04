"""IMAS static-IDS machine-description reader for the GS Green's-function operator.

Second implementation of :class:`~imas_ambix.gs.geometry.MachineGeometryReader`
(the first being :class:`~imas_ambix.gs.geometry.MastZarrGeometryReader`,
which wraps the FAIR-MAST ``efm`` reader).  Reads a machine's STATIC geometry
-- PF/CS coil geometry + turns (``pf_active``), the plasma-facing limiter
contour (``wall``), and pickup-coil / flux-loop positions (``magnetics``) --
straight from IMAS machine-description IDSs, and normalises it into the SAME
:class:`~imas_ambix.gs.geometry.GeometryTable` the MAST reader produces.
Every downstream consumer (the Green's-function operator,
:class:`~imas_ambix.latent.patch_basis.PatchBasis`,
:class:`~imas_ambix.latent.gs_solve.EquilibriumGrid`) is unaware which reader
built the table -- this is the seam the North-Star cross-machine-transfer
property depends on.

Access-layer rules (binding, imas-python only -- never h5py for physics data)
-------------------------------------------------------------------------------
* Every entry is opened at the DD version IT WAS WRITTEN IN
  (``ids_properties/version_put/data_dictionary``), never the installed
  default/latest.  A cross-major-version open is a **hard error** in
  imas-python (confirmed empirically: opening DD-3.x data with the
  installed default DD-4.x raises ``RuntimeError: ... different major
  version ...`` before any field, including ``ids_properties`` itself, can
  be read) and an approximate same-major version can silently miss renamed
  fields.  Because the on-disk version can only be read AFTER a
  version-matched open -- a chicken-and-egg the public imas-python API
  cannot resolve -- :func:`_peek_stored_dd_version` reads *only* the
  ``ids_properties/version_put/data_dictionary`` string attribute via
  h5py.  No physical/solver quantity is ever read this way; this is the
  documented "debugging schema issues" exception (never used for
  production data), scoped to exactly this one bootstrap step, one
  attribute, per IDS.
* ``DBEntry`` opens in the constructor -- ``.open()`` is never called
  afterwards.
* Every IDS read here is a static machine description; no time-dependent
  quantity is read, so ``homogeneous_time`` is not consulted.
* Every ``IDSFloat0D`` is cast with ``float(...)`` before arithmetic.

Machine-agnostic (no ITER-specific names)
------------------------------------------
Nothing here hardcodes a coil name, pulse number, or facility.  The caller
supplies the three IDS locations (a full ``imas:`` URI, or a legacy HDF5
pulse/run directory) and the reader introspects ``pf_active.coil[].element[]``,
``wall.description_2d[0].limiter.unit[]``, and
``magnetics.b_field_pol_probe[]`` / ``magnetics.flux_loop[]`` generically.

Known, honestly-flagged gaps (measured on the ITER machine-description gate,
recorded per-table in :attr:`~imas_ambix.gs.geometry.GeometryTable.provenance_flags`)
-------------------------------------------------------------------------------------
* ``pf_active`` elements come in more geometry shapes than MAST's rectangular
  ``fcoil`` filaments.  ``rectangle`` (r, z, width, height) and ``annulus``
  (r, z, radius_outer, converted to an equal-diameter square) are supported;
  any other shape (``outline`` / ``oblique`` / ``arcs_of_circle``) is FLAGGED
  and the element is DROPPED -- never fabricated as a rectangle.
* Some ``pf_active`` coils (bus-bar-equivalent / virtual coils) carry
  ``turns_with_sign`` as the IMAS EMPTY_FLOAT sentinel (no physical winding)
  -- treated as 0 turns and flagged, never silently coerced to a guess.
* IMAS ``magnetics.flux_loop.position`` can hold MULTIPLE ``(R, Z, phi)``
  points -- a "partial" flux loop segmented around a limited toroidal
  extent, unlike MAST's single-point axisymmetric ``silop``.  This reader
  represents each flux loop by the CENTROID of its position array, an
  approximation for non-axisymmetric partial loops; every loop for which
  this matters is flagged.
* ``pf_active`` here carries no ``circuit``-merge convention (unlike MAST's
  ``fcoil_circ``, which groups redundant fine/coarse discretisations of one
  physical coil) -- each named coil is its own circuit (``xmult = 1.0``
  throughout).  There is also no signal-channel concept analogous to MAST's
  ``amc`` (``amc_current_channels`` is always empty for an IMAS-sourced
  table): this reader tabulates GEOMETRY only, never a measured current.
  This is a real, load-bearing gap for
  :func:`imas_ambix.gs.operator.classify_circuits`, whose KNOWN-PF-coil
  match is hardcoded to MAST coil centroids/amc-channel names -- see the
  workstream report for the measured consequence downstream.
* ``magnetics.b_field_pol_probe.poloidal_angle`` is assumed to follow the
  same "angle of the sensor's sensitive axis from the R axis, toward Z,
  counter-clockwise in the poloidal plane" convention as MAST's
  ``magpr_ang`` (both feed the same ``B_R cosθ + B_Z sinθ`` projection in
  :mod:`imas_ambix.gs.operator`).  This is the standard IMAS DD convention
  but is NOT independently verified against a reference ITER field map in
  this reader -- flagged here for the record, not blocking (the vacuum-field
  gate checks coil geometry, not probe-orientation convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.gs.geometry import (
    BProbe,
    FluxLoop,
    GeometryTable,
    PFFilament,
    SensorMapping,
    SetupSignature,
    round_geometry_hash,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# IMAS EMPTY_FLOAT is -9e40; anything with |x| this large (or non-finite) is
# "not filled", never a genuine physical value at ITER/tokamak scale.
_EMPTY_FLOAT_FLOOR = 1.0e38

# Floor side length [m] for an annulus-equivalent square (avoids a
# zero-extent finite-area kernel on a vanishingly thin conductor).
_MIN_ANNULUS_SIDE_M = 0.01


def _is_empty(x: float) -> bool:
    """True if ``x`` is the IMAS EMPTY_FLOAT sentinel (or otherwise non-finite)."""
    return (not np.isfinite(x)) or abs(x) > _EMPTY_FLOAT_FLOOR


def _resolve_uri(path_or_uri: str | Path) -> str:
    """A bare legacy pulse/run directory is wrapped as an ``imas:hdf5`` URI.

    A string already starting with ``imas:`` is passed through unchanged
    (so a caller can point at any backend the local imas-python install
    supports).
    """
    s = str(path_or_uri)
    return s if s.startswith("imas:") else f"imas:hdf5?path={s}"


def _peek_stored_dd_version(path_or_uri: str | Path, ids_name: str) -> str:
    """Read the on-disk DD version string for one IDS via h5py (schema-debug only).

    imas-python cannot answer "what DD version is this file?" without first
    opening it AT that version -- a cross-major open hard-errors before any
    field, including ``ids_properties``, can be read (confirmed
    empirically).  This reads *only* the
    ``ids_properties/version_put/data_dictionary`` attribute string so the
    caller can then open the same entry correctly with imas-python for
    every actual field read.  See the module docstring's access-layer-rules
    note; only the legacy ``imas:hdf5?path=<dir>`` form (a directory holding
    ``master.h5``) can be peeked this way.
    """
    s = str(path_or_uri)
    if s.startswith("imas:") and not s.startswith("imas:hdf5?path="):
        raise ValueError(
            f"cannot peek the stored DD version for non-HDF5-legacy URI {s!r}; "
            "pass a legacy pulse/run directory (holding master.h5) instead"
        )
    directory = s[len("imas:hdf5?path=") :] if s.startswith("imas:") else s

    import h5py  # noqa: PLC0415

    master = Path(directory) / "master.h5"
    with h5py.File(master, "r") as f:
        grp = f[ids_name]
        raw = grp["ids_properties&version_put&data_dictionary"][()]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _open_static_ids(path_or_uri: str | Path, ids_name: str) -> Any:
    """Open one static IDS at its own stored DD version (imas-python only, closes after)."""
    import imas  # noqa: PLC0415

    dd_version = _peek_stored_dd_version(path_or_uri, ids_name)
    entry = imas.DBEntry(_resolve_uri(path_or_uri), "r", dd_version=dd_version)
    try:
        return entry.get(ids_name)
    finally:
        entry.close()


# --- pf_active -----------------------------------------------------------


def read_pf_active_filaments(pf_active: Any) -> tuple[list[PFFilament], list[str]]:
    """One :class:`PFFilament` per ``(coil, element)``; one circuit per named coil.

    Only ``rectangle`` and ``annulus`` element geometries are supported (the
    two shapes present in the ITER PF/CS machine description).  Any other
    shape is flagged and the element DROPPED -- never fabricated as a
    rectangle.  ``turns_with_sign`` carries the sign (verify-and-flag: an
    IMAS-EMPTY value is coerced to 0 turns and flagged, never guessed).
    """
    filaments: list[PFFilament] = []
    flags: list[str] = []
    for ci, coil in enumerate(pf_active.coil):
        name = str(coil.name)
        for ei, el in enumerate(coil.element):
            geo = el.geometry
            turns = float(el.turns_with_sign)
            if _is_empty(turns):
                flags.append(f"pf_active.coil[{ci}] '{name}'[{ei}]: turns_with_sign unset -> 0")
                turns = 0.0

            r_rect = float(geo.rectangle.r)
            if not _is_empty(r_rect):
                r, z = r_rect, float(geo.rectangle.z)
                width, height = float(geo.rectangle.width), float(geo.rectangle.height)
            else:
                r_ann = float(geo.annulus.r)
                if not _is_empty(r_ann):
                    r, z = r_ann, float(geo.annulus.z)
                    side = max(2.0 * float(geo.annulus.radius_outer), _MIN_ANNULUS_SIDE_M)
                    width = height = side
                else:
                    flags.append(
                        f"pf_active.coil[{ci}] '{name}'[{ei}]: unsupported element "
                        f"geometry (geometry_type={int(geo.geometry_type)}) -- dropped"
                    )
                    continue
            filaments.append(
                PFFilament(r=r, z=z, turns=turns, width=width, height=height, circuit=ci, xmult=1.0)
            )
    return filaments, flags


# --- wall / limiter --------------------------------------------------------


def _chain_limiter_units(
    units: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[float], list[float]]:
    """Greedily chain limiter units into one contour by nearest endpoint.

    Machine-agnostic: works whether a wall MD has one limiter unit (the
    single-unit-vessel convention) or several disjoint panels (ITER's
    First-Wall + Divertor pair) -- each remaining unit's nearer endpoint is
    attached (reversed if that end is closer) to the growing contour's tail.
    """
    if not units:
        return [], []
    r = list(units[0][0])
    z = list(units[0][1])
    remaining = list(units[1:])
    while remaining:
        best_i, best_reversed, best_d = 0, False, np.inf
        for i, (ur, uz) in enumerate(remaining):
            for reversed_, (er, ez) in (
                (False, (float(ur[0]), float(uz[0]))),
                (True, (float(ur[-1]), float(uz[-1]))),
            ):
                d = float(np.hypot(er - r[-1], ez - z[-1]))
                if d < best_d:
                    best_i, best_reversed, best_d = i, reversed_, d
        ur, uz = remaining.pop(best_i)
        if best_reversed:
            ur, uz = ur[::-1], uz[::-1]
        r.extend(ur.tolist())
        z.extend(uz.tolist())
    return r, z


def read_wall_limiter(wall: Any) -> tuple[list[float], list[float], list[str]]:
    """Chain every ``description_2d[0].limiter`` unit into ONE closed contour."""
    flags: list[str] = []
    if not len(wall.description_2d):
        return [], [], ["wall.description_2d is empty -- no limiter contour"]
    d2d = wall.description_2d[0]
    units = [
        (np.asarray(u.outline.r, dtype=np.float64), np.asarray(u.outline.z, dtype=np.float64))
        for u in d2d.limiter.unit
    ]
    units = [(r, z) for r, z in units if r.size and z.size]
    if not units:
        return [], [], ["wall.description_2d[0].limiter has no populated units"]
    if len(units) > 1:
        flags.append(
            f"wall limiter has {len(units)} units (e.g. first-wall + divertor); "
            "chained by nearest-endpoint into one contour"
        )
    r, z = _chain_limiter_units(units)
    return r, z, flags


# --- magnetics --------------------------------------------------------------


def read_magnetics_sensors(
    magnetics: Any,
) -> tuple[list[BProbe], list[FluxLoop], list[str]]:
    """B-probes read 1:1; flux loops collapse each's ``position`` array to its centroid."""
    flags: list[str] = []
    b_probes: list[BProbe] = []
    for i, p in enumerate(magnetics.b_field_pol_probe):
        length = float(p.length)
        if _is_empty(length):
            length = 0.0
        b_probes.append(
            BProbe(
                index=i,
                r=float(p.position.r),
                z=float(p.position.z),
                angle_deg=float(np.rad2deg(float(p.poloidal_angle))),
                length=length,
            )
        )

    flux_loops: list[FluxLoop] = []
    for i, fl in enumerate(magnetics.flux_loop):
        rs = np.asarray([float(pt.r) for pt in fl.position], dtype=np.float64)
        zs = np.asarray([float(pt.z) for pt in fl.position], dtype=np.float64)
        if rs.size == 0:
            flags.append(f"magnetics.flux_loop[{i}] '{str(fl.name)}': no position points -- dropped")
            continue
        if rs.size > 1:
            flags.append(
                f"magnetics.flux_loop[{i}] '{str(fl.name)}': {rs.size}-point partial "
                "loop represented by its centroid (non-axisymmetric approximation)"
            )
        flux_loops.append(FluxLoop(index=i, r=float(rs.mean()), z=float(zs.mean())))
    return b_probes, flux_loops, flags


# --- the reader -------------------------------------------------------------


@dataclass
class ImasGeometryReader:
    """Reads one machine's static geometry from IMAS ``pf_active``/``wall``/``magnetics``.

    Implements :class:`~imas_ambix.gs.geometry.MachineGeometryReader`.
    ``machine`` becomes the :class:`~imas_ambix.gs.geometry.SetupSignature`
    tag (and cache-key prefix) -- callers must pass a real tag (e.g.
    ``"iter"``); nothing here infers it from the data.  Each of the three
    paths is either a full ``imas:`` URI or a legacy HDF5 pulse/run
    directory (auto-wrapped by :func:`_resolve_uri`).
    """

    machine: str
    pf_active_path: str
    wall_path: str
    magnetics_path: str
    entry_id: int = 0  # provenance tag recorded in GeometryTable.shots

    def read(self) -> GeometryTable:
        pf_active = _open_static_ids(self.pf_active_path, "pf_active")
        wall = _open_static_ids(self.wall_path, "wall")
        magnetics = _open_static_ids(self.magnetics_path, "magnetics")

        pf_filaments, pf_flags = read_pf_active_filaments(pf_active)
        limiter_r, limiter_z, wall_flags = read_wall_limiter(wall)
        b_probes, flux_loops, mag_flags = read_magnetics_sensors(magnetics)

        sensor_map = [
            SensorMapping(
                amb_channel=f"bpol_{i:03d}",
                kind="b_probe",
                efm_index=i,
                r=p.r,
                z=p.z,
                angle_deg=p.angle_deg,
                residual_m=0.0,
                flag="",
            )
            for i, p in enumerate(b_probes)
        ] + [
            SensorMapping(
                amb_channel=f"floop_{i:03d}",
                kind="flux_loop",
                efm_index=i,
                r=fl.r,
                z=fl.z,
                angle_deg=None,
                residual_m=0.0,
                flag="",
            )
            for i, fl in enumerate(flux_loops)
        ]

        digest = round_geometry_hash(
            [
                np.array([p.r for p in b_probes]),
                np.array([p.z for p in b_probes]),
                np.array([p.angle_deg for p in b_probes]),
                np.array([f.r for f in flux_loops]),
                np.array([f.z for f in flux_loops]),
                np.array([f.r for f in pf_filaments]),
                np.array([f.z for f in pf_filaments]),
                np.array([f.turns for f in pf_filaments]),
                np.array(limiter_r),
                np.array(limiter_z),
            ]
        )
        signature = SetupSignature(
            n_bprobe=len(b_probes),
            n_fluxloop=len(flux_loops),
            n_pf_filament=len(pf_filaments),
            n_limiter=len(limiter_r),
            digest=digest,
            machine=self.machine,
        )

        lr = np.asarray(limiter_r, dtype=np.float64)
        r0 = float((lr.max() + lr.min()) / 2.0) if lr.size else 0.0
        minor_radius = float((lr.max() - lr.min()) / 2.0) if lr.size else 0.0

        return GeometryTable(
            signature=signature,
            shots=[self.entry_id],
            b_probes=b_probes,
            flux_loops=flux_loops,
            pf_filaments=pf_filaments,
            limiter_r=limiter_r,
            limiter_z=limiter_z,
            sensor_map=sensor_map,
            passive_structures=[],
            amc_current_channels=[],
            unmatched_amb=[],
            r0=r0,
            minor_radius=minor_radius,
            provenance_flags=pf_flags + wall_flags + mag_flags,
        )


__all__ = [
    "ImasGeometryReader",
    "read_magnetics_sensors",
    "read_pf_active_filaments",
    "read_wall_limiter",
]
