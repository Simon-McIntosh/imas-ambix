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
    KIND_FLUX_LOOP,
    KIND_SCALAR,
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
        # ``ip`` and any other unrecognised column stay scalar / all-NaN.
    return AlignedGeometry(
        channel_names=tuple(names),
        feature_names=tuple(GEOMETRY_FEATURE_NAMES),
        features=feats,
        sensor_kinds=tuple(kinds),
    )
