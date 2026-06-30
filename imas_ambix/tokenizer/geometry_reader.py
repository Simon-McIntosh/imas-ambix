"""Read per-channel sensor geometry aligned to a token store's channel order.

The world-model positional encoder needs, for every token channel it ingests,
the channel's apparatus geometry as a dense ``(n_channels, n_geom_features)``
array in the **same channel order** the store's ``channel_names`` declare.  Two
sources can supply it:

1. **the store itself** — if a ``signals_hf`` group was written with the
   optional ``geometry`` array (see ``store_v2.save_signal_hf_tokens``), it is
   already aligned and is returned directly; or
2. **a campaign geometry table** — a flat
   :class:`imas_ambix.gs.geometry_export.GeometryFields` built for the shot's
   campaign, projected onto the store's ``channel_names`` (NaN rows for any
   channel absent from the table).

Either way the result is aligned 1:1 with ``channel_names`` and carries a
``sensor_kind`` per channel (``"scalar"`` for an unknown / pure-scalar channel).

Boundary guard
--------------
Geometry is an INPUT-side positional field, never an eval-only reconstruction
target.  Every store path this reader opens is routed through
:func:`imas_ambix.tokenizer.store_targets.assert_not_target_path`, so a
``token_root`` resolving under the eval-only target store is hard-refused
before any read — geometry can never become a vector for target leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.gs.geometry_export import (
    GEOMETRY_FEATURE_NAMES,
    KIND_BPOL_PROBE,
    KIND_COIL,
    KIND_FLUX_LOOP,
    KIND_INTERFEROMETER_CHORD,
    KIND_SCALAR,
    KIND_SXR_CHORD,
    N_GEOMETRY_FEATURES,
    GeometryFields,
)
from imas_ambix.tokenizer.store_targets import assert_not_target_path
from imas_ambix.tokenizer.store_v2 import (
    STORE_GENERATION,
    load_signal_hf_tokens,
    signal_hf_token_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class AlignedGeometry:
    """Per-channel geometry aligned 1:1 with a store's ``channel_names``.

    ``features`` is ``(n_channels, n_geom_features)`` float32 (all-NaN rows for
    channels with no known geometry); ``sensor_kinds`` is the parallel list of
    categorical kinds; ``feature_names`` names the columns; ``channel_names`` is
    the order both are aligned to.
    """

    channel_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    features: np.ndarray
    sensor_kinds: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return self.features.shape[0]


def align_geometry_to_channels(
    fields: GeometryFields,
    channel_names: Sequence[str],
) -> AlignedGeometry:
    """Project a campaign :class:`GeometryFields` onto a channel-name order.

    Returns geometry aligned 1:1 with ``channel_names`` — ``features[i]`` is the
    geometry row for ``channel_names[i]`` (all-NaN when that channel has no
    known geometry), ``sensor_kinds[i]`` its categorical kind.  Channel-name
    matching is separator-insensitive (so a store's ``ccbv_01`` resolves to the
    amb ``ccbv01`` sensor).
    """
    feats, kinds = fields.feature_matrix(channel_names)
    return AlignedGeometry(
        channel_names=tuple(str(c) for c in channel_names),
        feature_names=tuple(GEOMETRY_FEATURE_NAMES),
        features=feats,
        sensor_kinds=tuple(kinds),
    )


def read_store_geometry(
    shot_id: int,
    group: str,
    *,
    token_root: Path | None = None,
    store_generation: str = STORE_GENERATION,
) -> AlignedGeometry | None:
    """Read geometry already attached to a ``signals_hf`` store, aligned to it.

    Routes the store path through the boundary guard before opening — a path
    resolving under the eval-only target root is hard-refused.  The actual read
    uses the module-level ``TOKEN_ROOT`` (as the rest of the v2 store does);
    ``token_root``, when given, is additionally guarded so a caller cannot point
    the reader at a target-rooted location.  Returns ``None`` when the store
    carries no geometry array (every legacy store) — a caller can then fall back
    to :func:`align_geometry_to_channels` with a campaign table.
    """
    path = signal_hf_token_path(shot_id, group, store_generation)
    assert_not_target_path(path)
    if token_root is not None:
        explicit = (
            token_root
            / store_generation
            / "signals_hf"
            / str(shot_id)
            / f"{group}.zarr"
        )
        assert_not_target_path(explicit)
    loaded = load_signal_hf_tokens(shot_id, group, store_generation=store_generation)
    if loaded.geometry is None:
        return None
    names = loaded.attrs.channel_names
    feat_names = loaded.attrs.geometry_feature_names or GEOMETRY_FEATURE_NAMES
    kinds = loaded.attrs.geometry_sensor_kinds
    if not kinds:
        kinds = tuple(KIND_SCALAR for _ in names)
    return AlignedGeometry(
        channel_names=tuple(str(c) for c in names),
        feature_names=tuple(str(f) for f in feat_names),
        features=np.asarray(loaded.geometry, dtype=np.float32),
        sensor_kinds=tuple(str(k) for k in kinds),
    )


def geometry_for_channels(
    channel_names: Sequence[str],
    *,
    fields: GeometryFields | None = None,
    shot_id: int | None = None,
    group: str | None = None,
    token_root: Path | None = None,
    store_generation: str = STORE_GENERATION,
) -> AlignedGeometry:
    """Per-channel geometry aligned to ``channel_names``, from store or table.

    Resolution order:

    1. if ``shot_id`` and ``group`` are given AND the store carries a geometry
       array, that already-aligned geometry is returned (the boundary guard is
       applied);
    2. else if ``fields`` is given, the campaign table is projected onto
       ``channel_names``;
    3. else an all-NaN / all-``scalar`` table is returned (geometry unknown but
       still present and explicit — never a silent drop).

    The result is always ``(len(channel_names), n_geom_features)`` aligned 1:1
    with ``channel_names``.
    """
    if shot_id is not None and group is not None:
        attached = read_store_geometry(
            shot_id, group, token_root=token_root, store_generation=store_generation
        )
        if attached is not None:
            return attached
    if fields is not None:
        return align_geometry_to_channels(fields, channel_names)
    names = tuple(str(c) for c in channel_names)
    return AlignedGeometry(
        channel_names=names,
        feature_names=tuple(GEOMETRY_FEATURE_NAMES),
        features=np.full((len(names), N_GEOMETRY_FEATURES), np.nan, dtype=np.float32),
        sensor_kinds=tuple(KIND_SCALAR for _ in names),
    )


# ---------------------------------------------------------------------------
# Staged-magnetics name -> geometry (the EFIT-class position/shape sensor)
# ---------------------------------------------------------------------------
#
# The staged magnetics store (and the world-model read path) names its columns
# from the L2 magnetics IDS arrays, NOT from the amb/efm sensor vocabulary the
# campaign geometry table keys on.  A poloidal-field-probe column is
# ``b_field_pol_probe_{kind}_field[i]`` (``i`` the i-th probe of that kind), a
# flux-loop column is ``flux_loop_flux[i]``, and the plasma current is the bare
# scalar ``ip``.  None of these resolve against the campaign table's ``ccbv_01``
# keys, so the spatial probe would see NaN geometry for every magnetic sensor —
# exactly the lane the model must attend over.
#
# The L2 magnetics IDS itself carries the geometry, aligned 1:1 with the field
# arrays BY CONSTRUCTION: ``b_field_pol_probe_{kind}_r/_z`` give each probe's
# position at the SAME column index ``i`` as the field, and ``flux_loop_r/_z``
# (indexed by ``flux_loop_geometry_channel``) give the flux-loop positions,
# matched to ``flux_loop_channel[i]`` by name.  Reading them directly is more
# authoritative than the campaign table (which has count gaps from the amb
# nearest-neighbour mapping) and needs no per-channel name translation.

#: Regex over a staged poloidal-B-probe column name -> ``(kind, index)``.
_BPROBE_COL_RE = re.compile(r"^b_field_pol_probe_(?P<kind>[a-z]+)_field\[(?P<i>\d+)\]$")
#: Regex over a staged flux-loop column name -> ``index``.
_FLUX_COL_RE = re.compile(r"^flux_loop_flux\[(?P<i>\d+)\]$")

#: Probe-kind -> the field component (and unit field-direction normal) it reads.
#: ``ccbv`` / ``obv`` are VERTICAL probes (read B_z, normal ``(0, 1)``); ``obr``
#: is a RADIAL probe (reads B_r, normal ``(1, 0)``).  The poloidal orientation
#: is fixed by the probe family, not the IDS ``phi_1/phi_2`` (which are the
#: probe's TOROIDAL extent, not its in-plane field direction).
_VERTICAL_PROBE_KINDS = ("ccbv", "obv")
_RADIAL_PROBE_KINDS = ("obr",)

#: Sensor-kind for a device-global signed scalar with no spatial geometry (the
#: plasma current ``ip``).  Matches the probe's ``SENSOR_KIND_VOCAB`` entry so
#: the kind embedding gives Ip a distinct slot.
KIND_GLOBAL_SCALAR = "global_scalar"


def _normalise_loop_name(name: str) -> str:
    """Lower-case, strip separators + the ``amb`` prefix (flux-loop name key)."""
    s = re.sub(r"[\s_\-/.]+", "", str(name).lower())
    return s[3:] if s.startswith("amb") else s


def _l2_magnetics_geometry(shot_id: int):
    """Read the L2 magnetics IDS geometry arrays for a shot (lazy, cached-free).

    Returns a dict with the per-kind probe ``(r, z)`` arrays (aligned 1:1 with
    the field columns) and the flux-loop name->``(r, z)`` map, or ``None`` when
    the L2 store / magnetics group is absent.  Reads ONLY the apparatus geometry
    arrays (positions + channel names) — never a reconstructed quantity.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415

    path = LEVEL2_DIR / f"{int(shot_id)}.zarr"
    if not path.exists():
        return None
    try:
        root = zarr.open_group(str(path), mode="r")
        if "magnetics" not in set(root.group_keys()):
            return None
        grp = root["magnetics"]
    except Exception:  # noqa: BLE001
        return None
    keys = set(grp.array_keys())
    probes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in (*_VERTICAL_PROBE_KINDS, *_RADIAL_PROBE_KINDS):
        rk = f"b_field_pol_probe_{kind}_r"
        zk = f"b_field_pol_probe_{kind}_z"
        if rk in keys and zk in keys:
            probes[kind] = (
                np.asarray(grp[rk], dtype=np.float64).reshape(-1),
                np.asarray(grp[zk], dtype=np.float64).reshape(-1),
            )
    loops: dict[str, tuple[float, float]] = {}
    if {"flux_loop_channel", "flux_loop_geometry_channel"} <= keys:
        fch = [str(x) for x in np.asarray(grp["flux_loop_channel"]).reshape(-1)]
        gch = [
            str(x) for x in np.asarray(grp["flux_loop_geometry_channel"]).reshape(-1)
        ]
        fr = np.asarray(grp["flux_loop_r"], dtype=np.float64).reshape(-1)
        fz = np.asarray(grp["flux_loop_z"], dtype=np.float64).reshape(-1)
        geo_by_name = {
            _normalise_loop_name(g): (float(fr[j]), float(fz[j]))
            for j, g in enumerate(gch)
            if j < fr.size and j < fz.size
        }
        for c in fch:
            loops[c] = geo_by_name.get(_normalise_loop_name(c), (np.nan, np.nan))
        loops["_order"] = fch  # preserve column order for index lookup
    return {"probes": probes, "loops": loops}


def magnetics_geometry_for_channels(
    channel_names: Sequence[str],
    shot_id: int,
) -> AlignedGeometry:
    """Resolve STAGED-MAGNETICS column names to per-channel apparatus geometry.

    The staged magnetics store names its columns from the L2 magnetics IDS:
    ``b_field_pol_probe_{kind}_field[i]`` (poloidal B-probe), ``flux_loop_flux[i]``
    (flux loop) and ``ip`` (plasma current scalar).  This resolves each to its
    physical geometry by reading the L2 magnetics IDS geometry arrays directly,
    aligned 1:1 with the field column index by construction:

    * a B-probe column gets ``(r, z)`` from ``b_field_pol_probe_{kind}_r/_z[i]``,
      ``sensor_kind = bpol_probe``, and a unit field-direction normal fixed by
      the probe family (``ccbv``/``obv`` vertical -> ``(0, 1)``; ``obr`` radial
      -> ``(1, 0)``);
    * a flux-loop column gets ``(r, z)`` matched by ``flux_loop_channel[i]`` to
      the IDS geometry-channel positions, ``sensor_kind = flux_loop``;
    * ``ip`` (and any unrecognised column) is a ``scalar`` with NaN coordinates —
      a learned no-geometry token, never a silent drop.

    The result is always ``(len(channel_names), n_geom_features)`` aligned 1:1
    with ``channel_names``; if the L2 store is unreachable, every row is the
    all-NaN ``scalar`` fallback (geometry unknown but present and explicit).
    """
    names = [str(c) for c in channel_names]
    feats = np.full((len(names), N_GEOMETRY_FEATURES), np.nan, dtype=np.float32)
    kinds: list[str] = [KIND_SCALAR] * len(names)
    src = _l2_magnetics_geometry(int(shot_id))
    if src is None:
        return AlignedGeometry(
            channel_names=tuple(names),
            feature_names=tuple(GEOMETRY_FEATURE_NAMES),
            features=feats,
            sensor_kinds=tuple(kinds),
        )
    probes = src["probes"]
    loops = src["loops"]
    loop_order = loops.get("_order", [])
    # column layout in GEOMETRY_FEATURE_NAMES:
    # (r, z, phi, angle_deg, normal_r, normal_z, chord_r1, chord_z1, chord_r2, chord_z2)
    for i, name in enumerate(names):
        m = _BPROBE_COL_RE.match(name)
        if m is not None:
            kind = m.group("kind")
            idx = int(m.group("i"))
            rz = probes.get(kind)
            if rz is not None and idx < rz[0].size:
                r, z = float(rz[0][idx]), float(rz[1][idx])
                if kind in _VERTICAL_PROBE_KINDS:
                    nr, nz, ang = 0.0, 1.0, 90.0
                elif kind in _RADIAL_PROBE_KINDS:
                    nr, nz, ang = 1.0, 0.0, 0.0
                else:
                    nr = nz = ang = np.nan
                feats[i, 0] = r
                feats[i, 1] = z
                feats[i, 2] = 0.0
                feats[i, 3] = ang
                feats[i, 4] = nr
                feats[i, 5] = nz
                kinds[i] = KIND_BPOL_PROBE
            continue
        m = _FLUX_COL_RE.match(name)
        if m is not None:
            idx = int(m.group("i"))
            if idx < len(loop_order):
                r, z = loops.get(loop_order[idx], (np.nan, np.nan))
                feats[i, 0] = r
                feats[i, 1] = z
                feats[i, 2] = 0.0
                kinds[i] = KIND_FLUX_LOOP
            continue
        # ``ip`` is the device-global plasma current — a SIGNED scalar with no
        # spatial geometry; tag it with the dedicated global-scalar kind so the
        # probe gives it a distinct slot rather than burying it among per-sensor
        # scalars.  Any other unrecognised column stays plain scalar / all-NaN.
        if name == "ip":
            kinds[i] = KIND_GLOBAL_SCALAR
    return AlignedGeometry(
        channel_names=tuple(names),
        feature_names=tuple(GEOMETRY_FEATURE_NAMES),
        features=feats,
        sensor_kinds=tuple(kinds),
    )


# ---------------------------------------------------------------------------
# PF-active coil-current name -> coil geometry (the EFIT current actuators)
# ---------------------------------------------------------------------------
#
# EFIT places the boundary / X-point from {magnetics + COIL CURRENTS + Ip +
# machine geometry}.  The L2 ``pf_active`` group carries ``coil_current``
# (n_coil, n_time) named by ``current_channel`` (e.g. ``AMC_P3U FEED CURRENT``)
# plus per-circuit filament geometry arrays (``p3_upper_r/_z`` …).  A coil's
# effective position is the centroid of its filament R/Z; we map each current
# channel to its circuit's geometry by name so each coil-current token carries
# the coil's (R, Z) as its positional code.

#: ``current_channel`` name token (after AMC_ prefix) -> filament-geometry prefix.
#: ``P2IL`` -> ``p2_inner_lower``, ``P3U`` -> ``p3_upper``, ``SOL`` -> ``sol`` …
_COIL_NAME_TO_PREFIX = {
    "p2il": "p2_inner_lower",
    "p2iu": "p2_inner_upper",
    "p2ol": "p2_outer_lower",
    "p2ou": "p2_outer_upper",
    "p3l": "p3_lower",
    "p3u": "p3_upper",
    "p4l": "p4_lower",
    "p4u": "p4_upper",
    "p5l": "p5_lower",
    "p5u": "p5_upper",
    "p6l": "p6_lower",
    "p6u": "p6_upper",
    "sol": "sol",
}


def _coil_token(channel_name: str) -> str:
    """Extract the coil token from a ``current_channel`` name.

    ``"AMC_P3U FEED CURRENT"`` -> ``"p3u"``; ``"AMC_SOL CURRENT"`` -> ``"sol"``.
    """
    s = str(channel_name).lower()
    s = re.sub(r"^amc[\s_]*", "", s)
    s = s.split()[0] if s.split() else s
    return re.sub(r"[\s_\-/.]+", "", s)


def pf_active_geometry_for_channels(
    channel_names: Sequence[str],
    shot_id: int,
) -> AlignedGeometry:
    """Resolve PF-active coil-current channel names to coil geometry.

    Reads the L2 ``pf_active`` group's per-circuit filament geometry arrays and
    maps each ``coil_current`` channel (named by ``current_channel``, e.g.
    ``AMC_P3U FEED CURRENT``) to its circuit's filament centroid ``(R, Z)``,
    ``sensor_kind = coil``.  An unmapped / unrecognised channel stays ``scalar``
    with NaN coordinates (present + explicit, never dropped).  Returns
    ``(len(channel_names), n_geom_features)`` aligned 1:1 with ``channel_names``.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415

    names = [str(c) for c in channel_names]
    feats = np.full((len(names), N_GEOMETRY_FEATURES), np.nan, dtype=np.float32)
    kinds: list[str] = [KIND_SCALAR] * len(names)
    path = LEVEL2_DIR / f"{int(shot_id)}.zarr"
    centroids: dict[str, tuple[float, float]] = {}
    if path.exists():
        try:
            grp = zarr.open_group(str(path), mode="r")["pf_active"]
            keys = set(grp.array_keys())
            for prefix in set(_COIL_NAME_TO_PREFIX.values()):
                rk, zk = f"{prefix}_r", f"{prefix}_z"
                if rk in keys and zk in keys:
                    r = np.asarray(grp[rk], dtype=np.float64).reshape(-1)
                    z = np.asarray(grp[zk], dtype=np.float64).reshape(-1)
                    if r.size and z.size:
                        centroids[prefix] = (float(np.mean(r)), float(np.mean(z)))
        except Exception:  # noqa: BLE001
            centroids = {}
    for i, name in enumerate(names):
        prefix = _COIL_NAME_TO_PREFIX.get(_coil_token(name))
        rz = centroids.get(prefix) if prefix else None
        if rz is not None:
            feats[i, 0] = rz[0]
            feats[i, 1] = rz[1]
            feats[i, 2] = 0.0
            kinds[i] = KIND_COIL
    return AlignedGeometry(
        channel_names=tuple(names),
        feature_names=tuple(GEOMETRY_FEATURE_NAMES),
        features=feats,
        sensor_kinds=tuple(kinds),
    )


# ---------------------------------------------------------------------------
# Toroidal saddle-array geometry (the ONE ingestible toroidal field series)
# ---------------------------------------------------------------------------
#
# The L2 magnetics group's ``b_field_tor_probe_saddle_voltage`` is a (12, n_time)
# array of saddle-loop voltages at 12 DISTINCT toroidal angles φ (the only
# toroidal sensor in this dataset that carries a time series — the
# ``b_field_tor_probe_cc`` 36-probe array is geometry-only).  The per-channel φ
# (and R, Z) come from the ``b_field_tor_probe_saddle_m_phi/_r/_z`` polygon
# arrays (each loop is a 28-vertex polygon spanning ~330° of toroidal arc); the
# loop's representative toroidal position is the CIRCULAR mean of its vertices'
# φ.  The 12 loops sit at 15°, 45°, …, 345° — evenly spaced 30° around the
# torus — so a periodic-φ positional encoding can resolve the toroidal mode.

#: The saddle band whose Z-centroid (~0) best represents the voltage channels.
_SADDLE_BAND = "saddle_m"
#: Sensor-kind for a toroidal saddle loop (a toroidal-field pickup).  Reuses the
#: bpol-probe embedding slot semantics is wrong — it is a distinct toroidal
#: sensor, so tag it as its own kind string the probe vocab carries.
KIND_TOROIDAL_SADDLE = "toroidal_saddle"


def _circular_mean_deg(phi_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees (handles the 0/360 seam)."""
    rad = np.deg2rad(np.asarray(phi_deg, dtype=np.float64).reshape(-1))
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)


def saddle_toroidal_geometry(shot_id: int):
    """Read the L2 saddle toroidal array: ``(names, geom (C,10), kinds)``.

    Returns the 12 saddle-voltage channel names, their per-channel geometry
    (R, Z, **φ in RADIANS** in the ``phi`` column — matching the schema's
    radian φ convention, all else NaN) and the toroidal-saddle sensor kind, or
    ``None`` when the L2 saddle arrays are absent.  φ is the CIRCULAR mean of
    each loop's polygon vertices; stored RAW (interpretable), the periodic
    ``(sin, cos)`` transform is applied at the model input, not here.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415

    path = LEVEL2_DIR / f"{int(shot_id)}.zarr"
    if not path.exists():
        return None
    try:
        grp = zarr.open_group(str(path), mode="r")["magnetics"]
        keys = set(grp.array_keys())
    except Exception:  # noqa: BLE001
        return None
    vkey = "b_field_tor_probe_saddle_voltage"
    if vkey not in keys:
        return None
    chnames = (
        [str(x) for x in np.asarray(grp[f"{vkey}_channel"]).reshape(-1)]
        if f"{vkey}_channel" in keys
        else [f"{vkey}[{i}]" for i in range(np.asarray(grp[vkey]).shape[0])]
    )
    n = len(chnames)
    feats = np.full((n, N_GEOMETRY_FEATURES), np.nan, dtype=np.float32)
    pk = f"b_field_tor_probe_{_SADDLE_BAND}_phi"
    rk = f"b_field_tor_probe_{_SADDLE_BAND}_r"
    zk = f"b_field_tor_probe_{_SADDLE_BAND}_z"
    if {pk, rk, zk} <= keys:
        phi = np.asarray(grp[pk], dtype=np.float64)  # (n, n_vertex) deg
        r = np.asarray(grp[rk], dtype=np.float64)
        z = np.asarray(grp[zk], dtype=np.float64)
        for i in range(min(n, phi.shape[0])):
            feats[i, 0] = float(np.mean(r[i]))
            feats[i, 1] = float(np.mean(z[i]))
            # φ stored in RADIANS (schema convention); circular mean over the
            # loop's polygon vertices, taken on the circle so the 0/2π seam is
            # handled before the value is stored.
            feats[i, 2] = float(np.deg2rad(_circular_mean_deg(phi[i])))
    kinds = tuple(KIND_TOROIDAL_SADDLE for _ in chnames)
    return tuple(chnames), feats, kinds


def saddle_toroidal_geometry_for_channels(
    channel_names: Sequence[str],
    shot_id: int,
) -> AlignedGeometry:
    """Per-channel saddle geometry aligned to ``channel_names`` (NaN if absent)."""
    src = saddle_toroidal_geometry(int(shot_id))
    names = [str(c) for c in channel_names]
    feats = np.full((len(names), N_GEOMETRY_FEATURES), np.nan, dtype=np.float32)
    kinds: list[str] = [KIND_SCALAR] * len(names)
    if src is not None:
        src_names, src_feats, src_kinds = src
        by_name = {n: i for i, n in enumerate(src_names)}
        for i, name in enumerate(names):
            j = by_name.get(name)
            if j is not None:
                feats[i] = src_feats[j]
                kinds[i] = src_kinds[j]
    return AlignedGeometry(
        channel_names=tuple(names),
        feature_names=tuple(GEOMETRY_FEATURE_NAMES),
        features=feats,
        sensor_kinds=tuple(kinds),
    )


# ---------------------------------------------------------------------------
# L2 light-path signal_hf names -> geometry (all-signals consolidation)
# ---------------------------------------------------------------------------
#
# The L2 light-path signal_hf streams name channels ``{group}.{var}[i]`` (e.g.
# ``soft_x_rays.horizontal_cam_lower[3]``, ``pf_active.coil_current[2]``,
# ``summary.ip``, ``gas_injection.valve_target_voltage[0]``,
# ``interferometer.n_e_line``).  To route EVERY stream through the shared
# space-time relational encoder with geometry — not just the magnetics — this
# resolver maps those names to per-channel geometry read DIRECTLY from the L2
# group, so each sensor token carries its apparatus geometry:
#
#   * soft_x_rays cameras -> CHORD geometry: the L2 group carries
#     ``{cam}_origin_r/_z`` + ``{cam}_endpoint_r/_z`` (+ ``{cam}_phi``) per
#     channel, written into the chord endpoint columns; kind = sxr_chord.
#   * pf_active coil currents -> COIL geometry: each ``coil_current[i]`` maps to
#     its circuit's filament centroid (R, Z) via
#     :func:`pf_active_geometry_for_channels` (the L2 ``current_channel`` order);
#     kind = coil.
#   * interferometer ``n_e_line`` -> a line-integrated chord with NO per-channel
#     endpoints in this store: kind = interferometer_chord, coords NaN (an
#     explicit placeholder — the chord schema is present, endpoints to be
#     tabulated from a static MAST interferometer geometry).
#   * everything else (summary, gas_injection scalars) -> scalar, NaN coords.
#
# Reads only the L2 light-path group's apparatus-geometry arrays — never a
# reconstructed quantity.

#: ``{group}.{var}[i]`` (or ``{group}.{var}``) channel-name parser.
_L2_HF_COL_RE = re.compile(r"^(?P<group>[a-z_]+)\.(?P<var>.+?)(?:\[(?P<i>\d+)\])?$")

#: pf_active current variables whose ``[i]`` maps to a coil position.
_PF_ACTIVE_CURRENT_VARS = ("coil_current", "solenoid_current")


def l2_signal_hf_geometry_for_channels(
    channel_names: Sequence[str],
    shot_id: int,
) -> AlignedGeometry:
    """Resolve L2 light-path ``{group}.{var}[i]`` names to per-channel geometry.

    Soft-x-ray cameras get chord endpoints from the L2 ``soft_x_rays`` group;
    pf_active coil currents get their coil centroid ``(R, Z)``; interferometer
    ``n_e_line`` gets a kind=``interferometer_chord`` placeholder (NaN coords);
    everything else stays ``scalar`` with NaN coords.  Always aligned 1:1 with
    ``channel_names``; an unreadable L2 store yields the all-scalar fallback.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.paths import LEVEL2_DIR  # noqa: PLC0415

    names = [str(c) for c in channel_names]
    feats = np.full((len(names), N_GEOMETRY_FEATURES), np.nan, dtype=np.float32)
    kinds: list[str] = [KIND_SCALAR] * len(names)

    path = LEVEL2_DIR / f"{int(shot_id)}.zarr"
    root = None
    if path.exists():
        try:
            root = zarr.open_group(str(path), mode="r")
        except Exception:  # noqa: BLE001
            root = None

    # pf_active coil geometry, resolved once via the dedicated coil resolver
    # (keyed by the L2 current_channel order); index i -> the i-th coil.
    pf_coil_feats = None
    if root is not None and "pf_active" in set(root.group_keys()):
        try:
            grp = root["pf_active"]
            if "current_channel" in set(grp.array_keys()):
                cc = [str(x) for x in np.asarray(grp["current_channel"]).reshape(-1)]
                ag = pf_active_geometry_for_channels(cc, int(shot_id))
                pf_coil_feats = (ag.features, ag.sensor_kinds)
        except Exception:  # noqa: BLE001
            pf_coil_feats = None

    for k, name in enumerate(names):
        m = _L2_HF_COL_RE.match(name)
        if m is None:
            continue
        group = m.group("group")
        var = m.group("var")
        idx = int(m.group("i")) if m.group("i") is not None else 0
        if group == "soft_x_rays" and root is not None:
            _fill_sxr_chord(feats, kinds, k, root, var, idx)
        elif group == "pf_active" and var in _PF_ACTIVE_CURRENT_VARS:
            if pf_coil_feats is not None and idx < pf_coil_feats[0].shape[0]:
                feats[k] = pf_coil_feats[0][idx]
                kinds[k] = pf_coil_feats[1][idx]
        elif group == "interferometer":
            # line-integrated, no per-channel endpoints in this store.
            kinds[k] = KIND_INTERFEROMETER_CHORD
        # summary / gas_injection / anything else -> scalar (NaN coords).
    return AlignedGeometry(
        channel_names=tuple(names),
        feature_names=tuple(GEOMETRY_FEATURE_NAMES),
        features=feats,
        sensor_kinds=tuple(kinds),
    )


def _fill_sxr_chord(feats, kinds, k, root, cam, idx):
    """Fill a soft-x-ray camera channel's chord endpoints from the L2 group."""
    try:
        sg = root["soft_x_rays"]
        keys = set(sg.array_keys())
    except Exception:  # noqa: BLE001
        return
    o_r, o_z = f"{cam}_origin_r", f"{cam}_origin_z"
    e_r, e_z = f"{cam}_endpoint_r", f"{cam}_endpoint_z"
    if not ({o_r, o_z, e_r, e_z} <= keys):
        return
    try:
        r1 = float(np.asarray(sg[o_r], dtype=np.float64).reshape(-1)[idx])
        z1 = float(np.asarray(sg[o_z], dtype=np.float64).reshape(-1)[idx])
        r2 = float(np.asarray(sg[e_r], dtype=np.float64).reshape(-1)[idx])
        z2 = float(np.asarray(sg[e_z], dtype=np.float64).reshape(-1)[idx])
    except (IndexError, ValueError):
        return
    # a dead (0,0)->(0,0) channel carries no usable chord — leave it scalar/NaN.
    if r1 == 0.0 and z1 == 0.0 and r2 == 0.0 and z2 == 0.0:
        return
    phi_key = f"{cam}_phi"
    phi = 0.0
    if phi_key in keys:
        try:
            phi = float(
                np.deg2rad(np.asarray(sg[phi_key], dtype=np.float64).reshape(-1)[idx])
            )
        except (IndexError, ValueError):
            phi = 0.0
    # columns: (r, z, phi, angle_deg, normal_r, normal_z, c_r1, c_z1, c_r2, c_z2)
    feats[k, 0] = 0.5 * (r1 + r2)  # chord midpoint as the representative R
    feats[k, 1] = 0.5 * (z1 + z2)  # chord midpoint Z
    feats[k, 2] = phi
    feats[k, 6] = r1
    feats[k, 7] = z1
    feats[k, 8] = r2
    feats[k, 9] = z2
    kinds[k] = KIND_SXR_CHORD
