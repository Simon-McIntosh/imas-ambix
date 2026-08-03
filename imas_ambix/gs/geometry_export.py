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

This module turns the per-campaign :class:`imas_ambix.gs.geometry.GeometryTable`
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
solver is *given*.  The upstream :mod:`imas_ambix.gs.geometry` reads ONLY the
static-setup efm arrays (``magpr_r/z/ang/len``, ``silop_r/z``, ``fcoil``
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
built from one campaign's :class:`~imas_ambix.gs.geometry.GeometryTable` and
carries that table's :class:`~imas_ambix.gs.geometry.SetupSignature` key so a
consumer never mixes geometry from incompatible campaigns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.gs.geometry import (
    GeometryTable,
    build_table_for_shot,
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


# Channel-name prefixes whose channels are line-integrated diagnostics.  A
# matching channel is tagged with the chord kind even when no chord endpoints
# are known yet (the endpoint columns stay NaN — the schema is present and the
# kind is explicit, ready for a machine whose chord geometry IS tabulated).
_INTERFEROMETER_PREFIXES = ("interfer", "nbar", "density", "ne_", "ne")
_SXR_PREFIXES = ("sxr", "softxray", "xsx")

# Channel-name prefixes that are PF / plasma current actuators (coils).  These
# carry geometry (a PF-coil R/Z position), surfaced via the campaign's PF
# filaments grouped by circuit.  amc current-channel names look like ``p3u``,
# ``p4l``, ``p5``, ``solenoid``/``sol``, ``pfx`` …
_COIL_PREFIXES = ("p1", "p2", "p3", "p4", "p5", "p6", "pf", "sol", "tf", "ip")


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
    table: GeometryTable,
) -> tuple[dict[int, tuple[float, float]], tuple[float, ...], tuple[float, ...]]:
    """Turns-weighted ``(R, Z)`` centroid per PF circuit.

    A PF circuit is many filaments; the coil's effective position is the
    turns-weighted centroid of its filaments.  Returns the per-circuit map plus
    the flat (R, Z) lists for the machine block (sorted by circuit number).
    """
    by_circ: dict[int, list] = {}
    for fil in table.pf_filaments:
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


def _classify_kind(channel_name: str) -> str:
    """Tag a channel with no efm-sensor match by its name prefix.

    A line-integrated diagnostic (interferometer / SXR chord) keeps its chord
    kind even with no known endpoints; a coil / current actuator gets the coil
    kind; everything else is a pure scalar.
    """
    n = _normalise(channel_name)
    if n.startswith(_SXR_PREFIXES):
        return KIND_SXR_CHORD
    if n.startswith(_INTERFEROMETER_PREFIXES):
        return KIND_INTERFEROMETER_CHORD
    if n.startswith(_COIL_PREFIXES):
        return KIND_COIL
    return KIND_SCALAR


def _bprobe_row(channel_name: str, mapping) -> ChannelGeometry:
    """Build a B-probe channel row (orientation -> field-direction normal)."""
    ang = float(mapping.angle_deg) if mapping.angle_deg is not None else float("nan")
    if np.isfinite(ang):
        rad = np.deg2rad(ang)
        # angle_deg is the orientation of the probe; the field direction it
        # reads is (cos, sin) of that angle in the (R, Z) plane.  0 deg = radial
        # (reads B_R), 90 deg = vertical (reads B_Z).
        nr, nz = float(np.cos(rad)), float(np.sin(rad))
    else:
        nr = nz = float("nan")
    return ChannelGeometry(
        channel_name=channel_name,
        sensor_kind=KIND_BPOL_PROBE,
        r=float(mapping.r),
        z=float(mapping.z),
        phi=0.0,
        angle_deg=ang,
        normal_r=nr,
        normal_z=nz,
        chord_r1=float("nan"),
        chord_z1=float("nan"),
        chord_r2=float("nan"),
        chord_z2=float("nan"),
    )


def _flux_loop_row(channel_name: str, mapping) -> ChannelGeometry:
    """Build a flux-loop channel row (point sensor, no orientation/chord)."""
    return ChannelGeometry(
        channel_name=channel_name,
        sensor_kind=KIND_FLUX_LOOP,
        r=float(mapping.r),
        z=float(mapping.z),
        phi=0.0,
        angle_deg=float("nan"),
        normal_r=float("nan"),
        normal_z=float("nan"),
        chord_r1=float("nan"),
        chord_z1=float("nan"),
        chord_r2=float("nan"),
        chord_z2=float("nan"),
    )


def _coil_row(
    channel_name: str, centroids: dict[int, tuple[float, float]]
) -> ChannelGeometry:
    """Build a coil channel row.

    A coil's geometry is the turns-weighted centroid of its filaments.  Without
    a per-channel circuit mapping in the geometry source we cannot resolve WHICH
    circuit a named amc channel drives, so the position columns stay NaN but the
    kind is explicitly ``coil`` (the schema is present; a downstream wiring step
    can fill R/Z once an amc-channel -> circuit map is tabulated).
    """
    return ChannelGeometry(
        channel_name=channel_name,
        sensor_kind=KIND_COIL,
        r=float("nan"),
        z=float("nan"),
        phi=0.0,
        angle_deg=float("nan"),
        normal_r=float("nan"),
        normal_z=float("nan"),
        chord_r1=float("nan"),
        chord_z1=float("nan"),
        chord_r2=float("nan"),
        chord_z2=float("nan"),
    )


def _scalar_or_chord_row(channel_name: str) -> ChannelGeometry:
    """Build a row for a channel with no efm-sensor geometry.

    Pure scalars (``ip``, gas-valve setpoints) get ``scalar`` with all-NaN
    coordinates; a line-integrated diagnostic gets its chord kind with NaN
    endpoints (present, explicit, ready to fill).
    """
    kind = _classify_kind(channel_name)
    return ChannelGeometry(
        channel_name=channel_name,
        sensor_kind=kind,
        r=float("nan"),
        z=float("nan"),
        phi=0.0 if kind != KIND_SCALAR else float("nan"),
        angle_deg=float("nan"),
        normal_r=float("nan"),
        normal_z=float("nan"),
        chord_r1=float("nan"),
        chord_z1=float("nan"),
        chord_r2=float("nan"),
        chord_z2=float("nan"),
    )


def build_geometry_fields_from_table(
    table: GeometryTable,
    *,
    extra_channel_names: Iterable[str] | None = None,
    resolve_identity: bool = False,
) -> GeometryFields:
    """Flatten a :class:`GeometryTable` to a per-channel geometry table.

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
    centroids, pf_r, pf_z = _pf_circuit_centroids(table)
    machine = MachineGeometry(
        limiter_r=tuple(float(x) for x in table.limiter_r),
        limiter_z=tuple(float(x) for x in table.limiter_z),
        pf_coil_r=pf_r,
        pf_coil_z=pf_z,
        r0=float(table.r0),
        minor_radius=float(table.minor_radius),
    )

    channels: dict[str, ChannelGeometry] = {}
    for m in table.sensor_map:
        if m.kind == "b_probe":
            row = _bprobe_row(m.amb_channel, m)
        elif m.kind == "flux_loop":
            row = _flux_loop_row(m.amb_channel, m)
        else:  # defensive — gs.geometry only emits b_probe / flux_loop today
            row = _scalar_or_chord_row(m.amb_channel)
        channels[_normalise(m.amb_channel)] = row

    # amc current channels (coils) — present and explicitly coil-kinded.
    for name in table.amc_current_channels:
        key = _normalise(name)
        if key in channels:
            continue
        channels[key] = _coil_row(name, centroids)

    # any extra token channel names the caller supplies (chords / scalars /
    # re-encoded sensor names) — never silently dropped.
    for name in extra_channel_names or ():
        key = _normalise(name)
        if key in channels:
            continue
        channels[key] = _scalar_or_chord_row(name)

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

    return GeometryFields(
        signature_key=table.signature.key,
        physical_digest=physical_digest,
        shots=list(table.shots),
        machine=machine,
        channels=channels,
    )


def build_geometry_table(
    shot_id: int,
    *,
    extra_channel_names: Iterable[str] | None = None,
) -> GeometryFields:
    """Build the flat per-channel geometry table for one representative shot.

    Reads ONLY the static efm geometry (via
    :func:`imas_ambix.gs.geometry.build_table_for_shot`) — no equilibrium /
    boundary / psi — and flattens it to the per-channel positional substrate.
    The geometry is per-campaign-constant, so one shot of a campaign is a valid
    source for that campaign's table.
    """
    table = build_table_for_shot(shot_id)
    return build_geometry_fields_from_table(
        table, extra_channel_names=extra_channel_names
    )


# --- Artifact I/O -----------------------------------------------------


def write_geometry_fields(fields: GeometryFields, path: Path) -> Path:
    """Write the flat per-channel geometry table as JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields.to_dict(), indent=2))
    return path
