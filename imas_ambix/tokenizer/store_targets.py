"""Eval-only TARGET store — the world-model prediction targets, walled off.

The Level-2 equilibrium reconstruction (ψ, j_φ, q, boundary, magnetic axis,
X-point, global scalars) and the reconstruction-derived globals
(``greenwald_density``, ``line_average_n_e`` = ``ESM_NE_BAR``) are what the
world model is supposed to **predict**.  Feeding any of them back in is
leakage.  This module stores them in a place — and with an attribute
contract — from which they can never be concatenated into the input token
stream.

Three physical walls keep targets out of the inputs
----------------------------------------------------
**Wall 1 — separate root.**  Targets live under
:data:`imas_ambix.data.paths.TARGET_ROOT`
(``/work/projects/imas_gpu/mast-targets``).  That root is *not* a child of
:data:`imas_ambix.data.paths.TOKEN_ROOT`
(``/work/projects/imas_gpu/mast-tokens``), so any glob the input loader runs
over ``TOKEN_ROOT/v2/...`` is structurally incapable of reaching a target.

**Wall 2 — no registry allocation.**  A target quantity is *raw physical
values*, never a decodable token id.  This module deliberately does **not**
import or call :mod:`imas_ambix.tokenizer.registry`, and
:class:`TargetV2Attrs` deliberately **omits** the ``tokenizer_name`` and
``vocab_version`` fields that :class:`imas_ambix.tokenizer.store_v2.StoreV2Attrs`
carries.  Without a tokenizer name and a vocab version there is no
registry-shift that could fold a target array into the input vocabulary —
the type itself refuses to participate.

**Wall 3 — guard enumerator.**  :func:`input_group_roots` is the single
function the world-model input data-loader uses to enumerate group roots.
It returns only ``TOKEN_ROOT/v2/signals_hf`` and ``TOKEN_ROOT/v2/frames``
and :func:`assert_not_target_path` hard-refuses any path that resolves under
``TARGET_ROOT``.  ``tests/tokenizer/test_targets_boundary.py`` exercises both
so the boundary is proven, not asserted.

On-disk layout (rooted at :data:`TARGET_ROOT`)::

    mast-targets/
      <shot_id>/
        equilibrium.zarr   # ψ / j_φ / q / boundary / axis / globals + masks
        derived_globals.zarr  # banned reconstruction-derived scalars + masks
        programmed.zarr    # programmed / demanded reconstruction waveforms

Each ``<quantity>.zarr`` group holds, per quantity, the raw physical array on
its **native time grid** (never resampled to a model grid) plus a matching
boolean *finite/validity* mask (``<name>__valid``) — NaN-outside-the-pulse
is recorded by the mask, never silently zero-filled.  The required
:class:`TargetV2Attrs` block records ``quantity_names``, ``units``,
``grid_r`` / ``grid_z`` (the equilibrium R/Z grid, empty for non-gridded
groups), ``time`` (the native time base), and ``original_window`` (the
``[t_start, t_end]`` of finite data so a consumer can place the target back
on the absolute axis).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.data.paths import TARGET_ROOT, TOKEN_ROOT

if TYPE_CHECKING:
    from pathlib import Path

# NOTE (Wall 2): this module MUST NOT import imas_ambix.tokenizer.registry.
# A target never gets a decodable token id, so there is no allocation step
# and no registry shift that could fold it into the input vocabulary.

# Store schema generation for the target store.  Distinct from the v2 token
# store generation — this tracks the on-disk *target* array/attribute layout.
TARGET_STORE_GENERATION = "v2"

# The world-model INPUT token store sub-roots, relative to ``TOKEN_ROOT``.
# The input data-loader enumerates ONLY these.  ``signals_hf`` is the
# native-cadence phase-preserving v2 signal store; ``frames`` is the camera
# token store.  TARGET_ROOT appears in NEITHER — that is Wall 1.
INPUT_GROUP_SUBPATHS: tuple[tuple[str, str], ...] = (
    (TARGET_STORE_GENERATION, "signals_hf"),
    (TARGET_STORE_GENERATION, "frames"),
)

# The required attribute keys for a target group.  A reader validates against
# this set so a half-written or stale-schema store is rejected loudly.  Note
# what is ABSENT versus the token store: no ``tokenizer_name``, no
# ``vocab_version`` (Wall 2).
REQUIRED_TARGET_ATTRS: tuple[str, ...] = (
    "quantity_names",
    "units",
    "grid_r",
    "grid_z",
    "time",
    "original_window",
)

# Attribute keys that would mark a group as input-vocabulary-eligible.  A
# target group must carry NONE of these — :meth:`TargetV2Attrs.to_attrs`
# never emits them, and :func:`assert_no_vocab_attrs` checks an on-disk store.
FORBIDDEN_TARGET_ATTRS: tuple[str, ...] = ("tokenizer_name", "vocab_version")


@dataclass(frozen=True)
class TargetV2Attrs:
    """Required attribute block for one target group.

    Mirrors the writer/reader discipline of
    :class:`imas_ambix.tokenizer.store_v2.StoreV2Attrs` — constructing this
    object validates field types so an inconsistent attribute set fails at
    write time, not read time — but **deliberately omits** ``tokenizer_name``
    and ``vocab_version`` (Wall 2).  Without them a target array carries no
    handle by which it could be concatenated into the input token vocabulary.

    Attributes
    ----------
    quantity_names:
        The physical quantities stored in this group (one data array +
        one ``<name>__valid`` mask each).
    units:
        Parallel tuple of unit strings, one per quantity.
    grid_r, grid_z:
        The equilibrium R / Z grid coordinates (m).  Empty for non-gridded
        groups (derived globals, programmed waveforms).
    time:
        The native time base (s) shared by every quantity in the group.
        Never resampled to a model grid.
    original_window:
        ``[t_start, t_end]`` of the finite-data span — the pulse window —
        so a consumer can place the target on the absolute time axis.
    """

    quantity_names: tuple[str, ...]
    units: tuple[str, ...]
    grid_r: tuple[float, ...]
    grid_z: tuple[float, ...]
    time: tuple[float, ...]
    original_window: tuple[float, float]
    # Free-form per-group metadata (uda names, source labels, coverage, …).
    # Stored JSON-encoded; never part of the required contract.
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.quantity_names) != len(self.units):
            raise ValueError(
                f"quantity_names ({len(self.quantity_names)}) and units "
                f"({len(self.units)}) length mismatch"
            )
        if len(self.original_window) != 2:
            raise ValueError("original_window must be (t_start, t_end)")
        # Wall 2, enforced at construction: a TargetV2Attrs can never carry a
        # tokenizer name or a vocab version — they are not fields of this
        # dataclass.  Guard against a caller smuggling them via ``metadata``.
        for forbidden in FORBIDDEN_TARGET_ATTRS:
            if forbidden in self.metadata:
                raise ValueError(
                    f"target attrs metadata must not carry {forbidden!r} — "
                    "a target gets no token-vocabulary handle (Wall 2)"
                )

    def to_attrs(self) -> dict[str, object]:
        """Return the JSON-safe ``.attrs`` dict written to Zarr.

        Emits the required target contract and NEVER ``tokenizer_name`` /
        ``vocab_version`` (Wall 2).
        """
        return {
            "quantity_names": list(self.quantity_names),
            "units": list(self.units),
            "grid_r": [float(x) for x in self.grid_r],
            "grid_z": [float(x) for x in self.grid_z],
            "time": [float(x) for x in self.time],
            "original_window": [
                float(self.original_window[0]),
                float(self.original_window[1]),
            ],
            "target_store_generation": TARGET_STORE_GENERATION,
            "is_eval_only_target": True,
            "metadata": json.dumps(_json_safe(self.metadata)),
        }

    @classmethod
    def from_attrs(cls, attrs: dict) -> TargetV2Attrs:
        """Reconstruct from an on-disk ``.attrs`` dict (validates contract).

        Rejects a store missing any required key, and rejects a store that
        somehow carries a forbidden token-vocabulary attribute (Wall 2).
        """
        missing = [k for k in REQUIRED_TARGET_ATTRS if k not in attrs]
        if missing:
            raise ValueError(f"target attrs missing required keys: {missing}")
        leaked = [k for k in FORBIDDEN_TARGET_ATTRS if k in attrs]
        if leaked:
            raise ValueError(
                f"target store carries forbidden token-vocab attrs {leaked} — "
                "a target must never be input-vocabulary eligible (Wall 2)"
            )
        meta_raw = attrs.get("metadata", "{}")
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
        win = attrs["original_window"]
        return cls(
            quantity_names=tuple(str(q) for q in attrs["quantity_names"]),
            units=tuple(str(u) for u in attrs["units"]),
            grid_r=tuple(float(x) for x in attrs["grid_r"]),
            grid_z=tuple(float(x) for x in attrs["grid_z"]),
            time=tuple(float(x) for x in attrs["time"]),
            original_window=(float(win[0]), float(win[1])),
            metadata=meta,
        )


@dataclass(frozen=True)
class TargetGroup:
    """In-memory representation of one target group.

    ``arrays`` maps a quantity name to its raw physical array (native time
    grid, the time axis last to mirror the L2 ``(..., time)`` layout).
    ``masks`` maps the same name to a boolean finite/validity mask of equal
    shape — ``True`` marks a usable value, ``False`` marks NaN-outside-window
    or otherwise invalid samples.
    """

    arrays: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    attrs: TargetV2Attrs


def _json_safe(obj: object) -> object:
    """Recursively coerce to a JSON-serialisable form (numpy-aware)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def target_group_path(
    shot_id: int, group: str, *, target_root: Path | None = None
) -> Path:
    """Canonical Zarr path for one shot's target group.

    ``TARGET_ROOT/<shot_id>/<group>.zarr``.  The path may not yet exist.
    """
    root = target_root or TARGET_ROOT
    return root / str(shot_id) / f"{group}.zarr"


# ---------------------------------------------------------------------------
# Wall 3 — the input-group enumerator + the boundary guard
# ---------------------------------------------------------------------------


def input_group_roots(*, token_root: Path | None = None) -> list[Path]:
    """Return the group roots the world-model INPUT loader may enumerate.

    Exactly ``TOKEN_ROOT/v2/signals_hf`` and ``TOKEN_ROOT/v2/frames`` — the
    native-cadence signal token store and the camera frame token store.
    :data:`TARGET_ROOT` is NOT among them (Wall 1): it is not a child of
    ``TOKEN_ROOT`` so no glob over these roots can reach a target.

    The world-model dataset builder must obtain its input group roots from
    here so the enumeration set is defined in exactly one place.
    """
    root = token_root or TOKEN_ROOT
    return [root / gen / name for gen, name in INPUT_GROUP_SUBPATHS]


def assert_not_target_path(path: Path, *, target_root: Path | None = None) -> Path:
    """Hard-refuse a path that resolves under :data:`TARGET_ROOT`.

    The input loader calls this on any path it is about to open, so an
    eval-only target can never be admitted into the input stream even if a
    caller hands it an explicit target path.  Returns ``path`` unchanged when
    it is clean; raises :class:`ValueError` when it points into the target
    store.
    """
    from pathlib import Path as _Path

    troot = (target_root or TARGET_ROOT).resolve()
    resolved = _Path(path).resolve()
    if resolved == troot or troot in resolved.parents:
        raise ValueError(
            f"refusing to open {resolved} — it resolves under TARGET_ROOT "
            f"({troot}); eval-only reconstruction targets must never enter "
            "the world-model input stream (Wall 1/Wall 3)"
        )
    return path


def enumerate_input_group_paths(
    *, token_root: Path | None = None, target_root: Path | None = None
) -> list[Path]:
    """Enumerate every per-shot input group ``.zarr`` under the input roots.

    This is the REAL enumerator the input loader uses: it scans only the
    roots returned by :func:`input_group_roots`, and every path it would
    yield is run through :func:`assert_not_target_path` so the target store
    can never leak in even if it were somehow symlinked under an input root.
    """
    paths: list[Path] = []
    for root in input_group_roots(token_root=token_root):
        if not root.exists():
            continue
        for shot_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for store in sorted(shot_dir.glob("*.zarr")):
                assert_not_target_path(store, target_root=target_root)
                paths.append(store)
    return paths


def assert_no_vocab_attrs(attrs: dict) -> dict:
    """Refuse an on-disk attribute block that carries token-vocab handles.

    A target store must carry none of :data:`FORBIDDEN_TARGET_ATTRS`.  Used
    by the boundary guard test and as a defensive check anywhere a target
    store's attrs are read.
    """
    leaked = [k for k in FORBIDDEN_TARGET_ATTRS if k in attrs]
    if leaked:
        raise ValueError(
            f"attrs carry forbidden token-vocab keys {leaked} — not a target"
        )
    return attrs


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


def save_target_group(
    shot_id: int,
    group: str,
    arrays: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    attrs: TargetV2Attrs,
    *,
    target_root: Path | None = None,
) -> Path:
    """Write one target group to Zarr under :data:`TARGET_ROOT`.

    Validates that every quantity has a matching finite/validity mask of
    equal shape and that the attribute ``quantity_names`` agrees with the
    arrays, **before** any data is written, so a malformed group is never
    persisted.  Masks are never optional — NaN-outside-the-pulse is recorded
    explicitly, never silently zero-filled.
    """
    import zarr

    names = tuple(attrs.quantity_names)
    if set(arrays) != set(names):
        raise ValueError(
            f"arrays keys {sorted(arrays)} disagree with attrs.quantity_names "
            f"{sorted(names)}"
        )
    if set(masks) != set(names):
        raise ValueError(
            f"masks keys {sorted(masks)} disagree with attrs.quantity_names "
            f"{sorted(names)}"
        )
    prepared: dict[str, np.ndarray] = {}
    prepared_masks: dict[str, np.ndarray] = {}
    for name in names:
        data = np.asarray(arrays[name], dtype=np.float64)
        mask = np.asarray(masks[name], dtype=bool)
        if mask.shape != data.shape:
            raise ValueError(
                f"{name}: mask shape {mask.shape} != data shape {data.shape}"
            )
        prepared[name] = data
        prepared_masks[name] = mask

    path = target_group_path(shot_id, group, target_root=target_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="w")
    for name in names:
        store.create_array(name, data=prepared[name])
        store.create_array(f"{name}__valid", data=prepared_masks[name])
    out_attrs = attrs.to_attrs()
    # Wall 2 belt-and-braces: never let a forbidden key reach the on-disk attrs.
    assert_no_vocab_attrs(out_attrs)
    out_attrs.update(
        {
            "shot_id": int(shot_id),
            "group": str(group),
            "n_quantities": len(names),
        }
    )
    store.attrs.update(out_attrs)
    return path


def load_target_group(
    shot_id: int, group: str, *, target_root: Path | None = None
) -> TargetGroup:
    """Read one target group back from Zarr.

    Validates the required-attribute contract on read; a store missing a
    required key — or carrying a forbidden token-vocab key — raises rather
    than silently defaulting.
    """
    import zarr

    path = target_group_path(shot_id, group, target_root=target_root)
    store = zarr.open_group(str(path), mode="r")
    raw_attrs = dict(store.attrs)
    attrs = TargetV2Attrs.from_attrs(raw_attrs)
    arrays: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name in attrs.quantity_names:
        arrays[name] = np.asarray(store[name], dtype=np.float64)
        masks[name] = np.asarray(store[f"{name}__valid"], dtype=bool)
    return TargetGroup(arrays=arrays, masks=masks, attrs=attrs)
