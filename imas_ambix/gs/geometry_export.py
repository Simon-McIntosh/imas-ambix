"""Flat per-channel sensor-geometry table — the positional encoding substrate.

A machine-agnostic world model attends over a heterogeneous, machine-varying
set of diagnostic channels.  For that attention to generalise across machines
(MAST -> JET -> ITER) every token must carry the physical geometry of ITS
sensor: where the sensor sits ``(R, Z, phi)``, how it is oriented (the field
component a B-probe reads; ``NaN`` for an isotropic point sensor like a flux
loop), and — for a line-integrated diagnostic — the chord it integrates along
``(r1, z1, r2, z2)``.  Plus the fixed machine geometry every channel shares:
the vessel/limiter contour, the PF-coil positions, and the nominal
major/minor radius.

This module turns the per-campaign machine-geometry projections
(B-probes, flux loops, PF filaments, limiter, the amb-sensor -> ``(R, Z, theta)``
map) into a **flat, one-row-per-token-channel** table keyed by the token
channel-name vocabulary the world-model dataset consumes (the ``signals_hf``
``channel_names``).  A channel with no known apparatus geometry — a pure scalar
like ``ip`` or a gas-valve setpoint — is **present and explicit** with
``sensor_kind="scalar"`` and ``NaN`` coordinates, never silently dropped, so a
consumer can build a complete ``(n_channels, n_geom_features)`` positional
array aligned 1:1 with the store's ``channel_names``.

Apparatus geometry only — never a reconstruction output
-------------------------------------------------------
Every coordinate here is **apparatus metadata** — the fixed *a-priori* setup a
solver is *given*. The private geometry kernel reads only the static-setup
arrays (``magpr_r/z/ang/len``, ``silop_r/z``, ``fcoil``
geometry, ``limiter``) and DELIBERATELY EXCLUDES every fitted/reconstructed
output (``*_c`` / ``*_x`` currents, ``psirz``, boundary, axis, ``q`` …).  This
module sources geometry ONLY through that boundary; it never imports or reads
equilibrium / boundary / psi / X-point.  Sensor geometry is freely usable —
the leakage firewall is on *code outputs* (plasma geometry), not apparatus
geometry.

Per-campaign aware
------------------
The efm setup is not constant across the corpus (the ``fcoil`` discretisation
and the ``magpr_z`` positions drift between campaigns).  The flat table is
built from one campaign's projections and carries their representation key so a
consumer never mixes geometry from incompatible campaigns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.gs.machine_geometry import (
    MachineGeometryService,
    OperatorGeometry,
    SensorGeometry,
    _project_operator_geometry,
    _project_sensor_features,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

# --- Sensor-kind taxonomy ---------------------------------------------
#
# The vocabulary a downstream positional encoder switches on.  Every token
# channel resolves to exactly one of these.

KIND_BPOL_PROBE = "bpol_probe"
KIND_FLUX_LOOP = "flux_loop"
KIND_INTERFEROMETER_CHORD = "interferometer_chord"
KIND_SXR_CHORD = "sxr_chord"
KIND_PIXEL = "pixel"
KIND_COIL = "coil"
KIND_SCALAR = "scalar"

SENSOR_KINDS: tuple[str, ...] = (
    KIND_BPOL_PROBE,
    KIND_FLUX_LOOP,
    KIND_INTERFEROMETER_CHORD,
    KIND_SXR_CHORD,
    KIND_PIXEL,
    KIND_COIL,
    KIND_SCALAR,
)

# --- The flat per-channel feature vector ------------------------------
#
# The ordered float feature columns a consumer reads as a dense
# (n_channels, n_geom_features) positional array.  sensor_kind is carried
# SEPARATELY as a string (it is categorical, not a coordinate) so the float
# block is purely numeric and NaN-fillable.  phi defaults to 0 (MAST sensors
# are axisymmetric in the poloidal model; the column exists so a 3-D machine
# can populate it).  normal_r / normal_z is the unit field-direction a B-probe
# reads, derived from angle_deg; NaN for an orientation-free sensor.

GEOMETRY_FEATURE_NAMES: tuple[str, ...] = (
    "r",  # sensor R [m]
    "z",  # sensor Z [m]
    "phi",  # sensor toroidal angle [rad] (0 for the axisymmetric poloidal model)
    "angle_deg",  # B-probe orientation [deg]; NaN where not applicable
    "normal_r",  # unit field-direction R component; NaN if orientation-free
    "normal_z",  # unit field-direction Z component; NaN if orientation-free
    "chord_r1",  # line-integrated diagnostic chord start R [m]; NaN for a point sensor
    "chord_z1",  # chord start Z [m]
    "chord_r2",  # chord end R [m]
    "chord_z2",  # chord end Z [m]
)

N_GEOMETRY_FEATURES = len(GEOMETRY_FEATURE_NAMES)


# --- Channel-name normalisation ---------------------------------------
#
# The token stores name channels in several conventions: the amb mapping keys
# on raw amb names (``ccbv01``, ``obr06``, ``fl_cc01``), while a re-encoded
# ``signals_hf`` store may carry separator-normalised names (``ccbv_01``).  We
# match on a separator-stripped, lower-cased key so ``ccbv_01`` and ``ccbv01``
# resolve to the same sensor.  A purely positional ``magnetics`` channel name
# (e.g. ``ccbv_01``) and the amb name (``ccbv01``) then collide-match cleanly.

_NORM_RE = re.compile(r"[\s_\-/.]+")


def _normalise(name: str) -> str:
    """Lower-case, strip separators — the key both vocabularies match on."""
    return _NORM_RE.sub("", str(name).lower())


# --- The flat per-channel record --------------------------------------


@dataclass(frozen=True)
class ChannelGeometry:
    """Apparatus geometry for one token channel (one row of the flat table)."""

    channel_name: str
    sensor_kind: str
    r: float
    z: float
    phi: float
    angle_deg: float
    normal_r: float
    normal_z: float
    chord_r1: float
    chord_z1: float
    chord_r2: float
    chord_z2: float

    def feature_row(self) -> list[float]:
        """The ordered float feature vector (matches GEOMETRY_FEATURE_NAMES)."""
        return [
            self.r,
            self.z,
            self.phi,
            self.angle_deg,
            self.normal_r,
            self.normal_z,
            self.chord_r1,
            self.chord_z1,
            self.chord_r2,
            self.chord_z2,
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_name": self.channel_name,
            "sensor_kind": self.sensor_kind,
            **dict(zip(GEOMETRY_FEATURE_NAMES, self.feature_row(), strict=True)),
        }


# --- The machine-geometry block (shared by every channel) -------------


@dataclass(frozen=True)
class MachineGeometry:
    """The fixed machine geometry every channel of a campaign shares."""

    limiter_r: tuple[float, ...]
    limiter_z: tuple[float, ...]
    pf_coil_r: tuple[float, ...]  # one R per PF circuit (turns-weighted centroid)
    pf_coil_z: tuple[float, ...]  # one Z per PF circuit
    r0: float
    minor_radius: float

    def to_dict(self) -> dict[str, object]:
        return {
            "limiter_r": list(self.limiter_r),
            "limiter_z": list(self.limiter_z),
            "pf_coil_r": list(self.pf_coil_r),
            "pf_coil_z": list(self.pf_coil_z),
            "r0": self.r0,
            "minor_radius": self.minor_radius,
        }


# --- The full flat geometry table -------------------------------------


@dataclass
class GeometryFields:
    """Flat per-channel sensor geometry for one campaign + the machine block.

    The substrate the world-model positional encoder consumes.  ``channels`` is
    keyed by the (separator-normalised) token channel name; ``machine`` is the
    fixed machine geometry shared by every channel.  ``signature_key`` is the
    campaign key so a consumer never mixes incompatible-campaign geometry.
    """

    signature_key: str
    shots: list[int]
    machine: MachineGeometry
    channels: dict[str, ChannelGeometry] = field(default_factory=dict)
    #: The Nova registry physical digest of the machine, when resolved.
    #: ``signature_key`` guards against mixing incompatible DISCRETIZATIONS;
    #: this records which MACHINE the geometry describes.  Empty when identity
    #: was not resolved, which leaves the exported payload byte-unchanged.
    physical_digest: str = ""

    def get(self, channel_name: str) -> ChannelGeometry | None:
        """Look up a channel by name (separator-insensitive)."""
        return self.channels.get(_normalise(channel_name))

    def feature_matrix(
        self, channel_names: Sequence[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Dense ``(n_channels, n_geom_features)`` float32 aligned to names.

        Returns ``(features, sensor_kinds)``: ``features[i]`` is the geometry
        feature row for ``channel_names[i]`` (all-NaN when the channel has no
        known geometry), and ``sensor_kinds[i]`` its categorical kind
        (``"scalar"`` for an unknown / pure-scalar channel).  This is the exact
        positional array the model attaches to its tokens.
        """
        names = list(channel_names)
        feats = np.full((len(names), N_GEOMETRY_FEATURES), np.nan, dtype=np.float32)
        kinds: list[str] = []
        for i, name in enumerate(names):
            cg = self.get(name)
            if cg is None:
                kinds.append(KIND_SCALAR)
                continue
            feats[i] = np.asarray(cg.feature_row(), dtype=np.float32)
            kinds.append(cg.sensor_kind)
        return feats, kinds

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "sensor-geometry-fields",
            "source": "efm-static-geometry",
            "signature_key": self.signature_key,
            **(
                {"physical_digest": self.physical_digest}
                if self.physical_digest
                else {}
            ),
            "shots": self.shots,
            "feature_names": list(GEOMETRY_FEATURE_NAMES),
            "sensor_kinds": list(SENSOR_KINDS),
            "machine": self.machine.to_dict(),
            "channels": {
                name: cg.to_dict() for name, cg in sorted(self.channels.items())
            },
        }


# --- Builders ---------------------------------------------------------


def _pf_circuit_centroids(
    geometry: Any,
) -> tuple[dict[int, tuple[float, float]], tuple[float, ...], tuple[float, ...]]:
    """Turns-weighted ``(R, Z)`` centroid per PF circuit.

    A PF circuit is many filaments; the coil's effective position is the
    turns-weighted centroid of its filaments.  Returns the per-circuit map plus
    the flat (R, Z) lists for the machine block (sorted by circuit number).
    """
    by_circ: dict[int, list] = {}
    filaments = getattr(geometry, "conductors", None)
    if filaments is None:
        filaments = geometry.pf_filaments
    for fil in filaments:
        by_circ.setdefault(int(fil.circuit), []).append(fil)
    centroids: dict[int, tuple[float, float]] = {}
    for circ, fils in by_circ.items():
        w = np.array([abs(f.turns) for f in fils], dtype=np.float64)
        r = np.array([f.r for f in fils], dtype=np.float64)
        z = np.array([f.z for f in fils], dtype=np.float64)
        wsum = float(w.sum())
        if wsum <= 0:
            centroids[circ] = (float(r.mean()), float(z.mean()))
        else:
            centroids[circ] = (float((w * r).sum() / wsum), float((w * z).sum() / wsum))
    circs = sorted(centroids)
    pf_r = tuple(centroids[c][0] for c in circs)
    pf_z = tuple(centroids[c][1] for c in circs)
    return centroids, pf_r, pf_z


def build_geometry_fields_from_table(
    table: Any,
    *,
    extra_channel_names: Iterable[str] | None = None,
    resolve_identity: bool = False,
) -> GeometryFields:
    """Project a private compatibility kernel to per-channel geometry.

    Every amb sensor mapping in ``table.sensor_map`` becomes a B-probe or
    flux-loop row keyed by its amb channel name.  Any ``extra_channel_names``
    (token channel names with no efm sensor — coils, chords, scalars) are added
    as classified rows (coil / chord kind with NaN coordinates, or scalar).

    Channel-name keying is separator-insensitive, so a re-encoded store's
    ``ccbv_01`` resolves to the amb ``ccbv01`` sensor.

    ``resolve_identity`` stamps the machine's Nova physical digest onto the
    result.  Off by default: the feature rows are unaffected either way, since
    identity is provenance and never feeds the encoding.  A registry miss is
    non-fatal, matching the surrounding best-effort geometry contract.
    """
    operator = _project_operator_geometry(table, resolve_identity=resolve_identity)
    channel_names: dict[str, str] = {}
    for mapping in operator.sensor_map:
        channel_names.setdefault(_normalise(mapping.amb_channel), mapping.amb_channel)
    for name in operator.available_current_channels:
        channel_names.setdefault(_normalise(name), name)
    for name in extra_channel_names or ():
        channel_names.setdefault(_normalise(name), str(name))
    requested = tuple(channel_names.values())
    matrix, kinds = _project_sensor_features(table, requested)
    sensors = SensorGeometry(
        identity=operator.identity,
        channels=requested,
        feature_names=GEOMETRY_FEATURE_NAMES,
        sensor_kinds=kinds,
        feature_matrix=matrix,
    )
    return _fields_from_projections(operator, sensors, table.shots)


def build_geometry_table(
    shot_id: int,
    *,
    extra_channel_names: Iterable[str] | None = None,
) -> GeometryFields:
    """Build the flat per-channel geometry table for one representative shot.

    Reads only the declared static machine description — no equilibrium,
    boundary or psi — and flattens it to the per-channel positional substrate.
    The selected description is constant across its declared shot range, so one
    shot is a valid source for that range's table.
    """
    shot = int(shot_id)
    service = MachineGeometryService()
    operator = service.operator(shot)
    channel_names: dict[str, str] = {}
    for mapping in operator.sensor_map:
        channel_names.setdefault(_normalise(mapping.amb_channel), mapping.amb_channel)
    for name in operator.available_current_channels:
        channel_names.setdefault(_normalise(name), name)
    for name in extra_channel_names or ():
        channel_names.setdefault(_normalise(name), str(name))
    sensors = service.sensors(shot, channel_names.values())
    return _fields_from_projections(operator, sensors, (shot,))


def _fields_from_projections(
    operator: OperatorGeometry,
    sensors: SensorGeometry,
    shots: Iterable[int],
) -> GeometryFields:
    """Build the outward DTO exclusively from facade projections."""
    _, pf_r, pf_z = _pf_circuit_centroids(operator)
    machine = MachineGeometry(
        limiter_r=operator.limiter_r,
        limiter_z=operator.limiter_z,
        pf_coil_r=pf_r,
        pf_coil_z=pf_z,
        r0=operator.r0,
        minor_radius=operator.minor_radius,
    )
    channels: dict[str, ChannelGeometry] = {}
    mappings = {
        _normalise(mapping.amb_channel): mapping for mapping in operator.sensor_map
    }
    for name, kind, row in zip(
        sensors.channels,
        sensors.sensor_kinds,
        sensors.feature_matrix,
        strict=True,
    ):
        values = dict(zip(GEOMETRY_FEATURE_NAMES, map(float, row), strict=True))
        mapping = mappings.get(_normalise(name))
        if mapping is not None and kind in (KIND_BPOL_PROBE, KIND_FLUX_LOOP):
            # SensorGeometry is float32 by contract, while the outward JSON
            # DTO retains exact source scalars. Read them from the operator
            # projection instead of widening the rounded dense encoding.
            values["r"] = float(mapping.r)
            values["z"] = float(mapping.z)
            if kind == KIND_BPOL_PROBE and mapping.angle_deg is not None:
                angle = float(mapping.angle_deg)
                values["angle_deg"] = angle
                radians = np.deg2rad(angle)
                values["normal_r"] = float(np.cos(radians))
                values["normal_z"] = float(np.sin(radians))
        channels[_normalise(name)] = ChannelGeometry(
            channel_name=name,
            sensor_kind=kind,
            **values,
        )
    return GeometryFields(
        signature_key=operator.identity.representation_key,
        physical_digest=operator.identity.physical_digest,
        shots=[int(shot) for shot in shots],
        machine=machine,
        channels=channels,
    )


# --- Artifact I/O -----------------------------------------------------


def write_geometry_fields(fields: GeometryFields, path: Path) -> Path:
    """Write the flat per-channel geometry table as JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields.to_dict(), indent=2))
    return path
