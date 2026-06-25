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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.gs.geometry_export import (
    GEOMETRY_FEATURE_NAMES,
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
