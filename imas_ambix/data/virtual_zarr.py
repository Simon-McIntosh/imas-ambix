"""Read-only, lazy canonical views over immutable physical Zarr arrays.

VirtualiZarr-style manifests are valuable for inventorying and concatenating
references, but a reference alone cannot change a decoded value's unit, COCOS
sign, or calibration.  Ambix therefore keeps the physical Zarr store untouched
and places a very small transform-aware view above it.  Constructing the view,
listing signals, and inspecting metadata read no chunks.  ``array[index]``
passes the index directly to the physical Zarr array and applies the compiled
affine transform to only the returned chunk or slice.

The class deliberately refuses implicit whole-array coercion and all writes.
Callers must request an explicit slice, which makes accidental corpus
materialisation and accidental source mutation fail loudly.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.cocos import CANONICAL_COCOS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from imas_ambix.data.signal_map import CompiledSignal, CompiledSignalMap, SignalMap


class VirtualZarrError(ValueError):
    """Raised when a virtual binding cannot be resolved safely."""


class VirtualArray:
    """One lazy physical Zarr array exposed in canonical units and convention."""

    def __init__(self, source: Any, signal: CompiledSignal) -> None:
        self._source = source
        self.signal = signal

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._source.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunks(self) -> tuple[int, ...] | None:
        chunks = getattr(self._source, "chunks", None)
        return None if chunks is None else tuple(chunks)

    @property
    def dtype(self) -> np.dtype:
        source_dtype = np.dtype(getattr(self._source, "dtype", float))
        sample = np.empty((), dtype=source_dtype)
        return np.multiply(sample, self.signal.scale).dtype

    @property
    def attrs(self) -> Mapping[str, Any]:
        rule = self.signal.rule
        return MappingProxyType(
            {
                "cocos": CANONICAL_COCOS,
                "map_semantic_id": rule.semantic_id,
                "standard_name": rule.standard_name,
                "target_index": rule.target_index,
                "target_path": rule.target_path,
                "units": rule.target_unit,
            }
        )

    def __getitem__(self, selection: Any) -> np.ndarray:
        return self.signal.apply(self._source[selection])

    def __setitem__(self, selection: Any, value: Any) -> None:
        del selection, value
        raise TypeError("a virtual canonical array is read-only")

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        del dtype, copy
        raise TypeError(
            "implicit whole-array reads are disabled; request an explicit slice"
        )


class VirtualZarrView:
    """A group-like collection of lazy canonical arrays for one shot."""

    def __init__(self, source: Any, compiled_map: CompiledSignalMap) -> None:
        self._source = source
        self.compiled_map = compiled_map
        self._arrays: dict[str, VirtualArray] = {}

    @classmethod
    def open(
        cls,
        source: str,
        signal_map: SignalMap,
        *,
        shot: int,
    ) -> VirtualZarrView:
        """Open a physical Zarr group read-only and attach a compiled map."""

        import zarr

        return cls(zarr.open_group(source, mode="r"), signal_map.compile(shot))

    @property
    def attrs(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "cocos": CANONICAL_COCOS,
                "map_digest": self.compiled_map.digest,
                "shot": self.compiled_map.shot,
            }
        )

    def keys(self) -> tuple[str, ...]:
        return tuple(signal.rule.semantic_id for signal in self.compiled_map)

    def __contains__(self, semantic_id: object) -> bool:
        return isinstance(semantic_id, str) and semantic_id in self.keys()

    def __getitem__(self, semantic_id: str) -> VirtualArray:
        cached = self._arrays.get(semantic_id)
        if cached is not None:
            return cached
        signal = self.compiled_map[semantic_id]
        rule = signal.rule
        try:
            group = self._source[rule.source_group]
            source_array = group[rule.source_array]
        except (KeyError, TypeError) as error:
            raise VirtualZarrError(
                f"source array {rule.source_group}/{rule.source_array} for "
                f"{semantic_id!r} is absent"
            ) from error
        array = VirtualArray(source_array, signal)
        self._arrays[semantic_id] = array
        return array

    def __setitem__(self, semantic_id: str, value: Any) -> None:
        del semantic_id, value
        raise TypeError("a virtual canonical Zarr view is read-only")

    def for_target(
        self, target_path: str, target_index: int | None = None
    ) -> VirtualArray:
        """Return the unique virtual array serving one DD path and structure index."""

        matches = [
            signal.rule.semantic_id
            for signal in self.compiled_map
            if signal.rule.target_path == target_path
            and signal.rule.target_index == target_index
        ]
        if len(matches) != 1:
            raise VirtualZarrError(
                f"target {(target_path, target_index)!r} has {len(matches)} served rows"
            )
        return self[matches[0]]


__all__ = ["VirtualArray", "VirtualZarrError", "VirtualZarrView"]
