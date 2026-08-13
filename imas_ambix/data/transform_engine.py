"""Apply declarative machine maps through format-scoped store engines.

The engine registry varies only with the physical store format.  Map selection,
binding traversal, sign handling, missing-array accounting, and DD-path
emission are shared by every format and every machine catalog.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, Self

import imas
import numpy as np
import zarr
from imas.ids_struct_array import IDSStructArray

from imas_ambix.cocos import canonical_factor
from imas_ambix.data.machine_map import (
    ChannelBinding,
    MachineMap,
    MachineMapCatalog,
    map_for_shot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class TransformEngineError(RuntimeError):
    """Base error for invalid or unreadable machine-description transforms."""


class SourceUnavailableError(TransformEngineError):
    """Raised when a requested pulse store is not available to an engine."""


class BindingTransformError(TransformEngineError):
    """Raised when a declared binding cannot be transformed unambiguously."""


_CATALOG_SOURCE_COCOS = object()


class StoreArrays(Protocol):
    """Minimal array-reading seam shared by physical store formats."""

    def read(self, binding: ChannelBinding) -> np.ndarray:
        """Read one declared source array or raise ``KeyError`` if absent."""


class StoreEngine(Protocol):
    """Open one pulse through a physical store format."""

    format_name: str

    def open(
        self, root: Path | str, shot: int, dd_version: str
    ) -> AbstractContextManager[StoreArrays]:
        """Open the requested pulse as a collection of named arrays."""


@dataclass(frozen=True)
class EmittedArray:
    """One source array emitted with its declared Data Dictionary identity."""

    binding_name: str
    source_group: str
    source_array: str
    dd_path: str
    source_unit: str
    target_unit: str
    cocos_transformation: str | None
    cocos_factor: float
    values: np.ndarray


@dataclass(frozen=True)
class MachineDescription:
    """The observable outcome of applying one range-scoped machine map."""

    shot: int
    store_format: str
    machine_map: MachineMap
    dd_version: str
    source_cocos: int | None
    status: str
    arrays: tuple[EmittedArray, ...]
    missing_bindings: tuple[str, ...]
    detail: str

    @property
    def emitted_array_count(self) -> int:
        """Return the number of source arrays emitted into DD paths."""
        return len(self.arrays)

    @property
    def arrays_by_dd_path(self) -> Mapping[str, tuple[EmittedArray, ...]]:
        """Group emissions by DD path without discarding repeated structures."""
        grouped: defaultdict[str, list[EmittedArray]] = defaultdict(list)
        for array in self.arrays:
            grouped[array.dd_path].append(array)
        return MappingProxyType(
            {path: tuple(arrays) for path, arrays in grouped.items()}
        )


class _ZarrArrays(AbstractContextManager["_ZarrArrays"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._group = None

    def __enter__(self) -> Self:
        if not self.path.is_dir():
            raise SourceUnavailableError(f"zarr pulse store is absent: {self.path}")
        try:
            self._group = zarr.open_group(self.path, mode="r")
        except (OSError, ValueError) as error:
            raise SourceUnavailableError(
                f"zarr pulse store cannot be opened: {self.path}: {error}"
            ) from error
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._group = None

    def read(self, binding: ChannelBinding) -> np.ndarray:
        if self._group is None:
            raise TransformEngineError("zarr pulse store is not open")
        return np.asarray(
            self._group[f"{binding.source_group}/{binding.source_array}"][...]
        )


class ZarrTransformEngine:
    """Read logical source groups from one Zarr directory per pulse."""

    format_name = "zarr"

    def open(self, root: Path | str, shot: int, dd_version: str) -> _ZarrArrays:
        """Open ``<root>/<shot>.zarr`` without mutating the source store."""
        return _ZarrArrays(Path(root) / f"{int(shot)}.zarr")


def _struct_array_positions(
    node: object, components: tuple[str, ...]
) -> tuple[int, ...]:
    positions: list[int] = []
    for index, component in enumerate(components[:-1]):
        child = getattr(node, component)
        if isinstance(child, IDSStructArray):
            if not len(child):
                raise KeyError("/".join(components))
            positions.append(index)
            node = child[0]
        else:
            node = child
    return tuple(positions)


def _read_ids_path(
    node: object,
    components: tuple[str, ...],
    preserved_positions: frozenset[int],
    index: int = 0,
) -> np.ndarray:
    child = getattr(node, components[index])
    if index == len(components) - 1:
        return np.asarray(child.value)
    if isinstance(child, IDSStructArray):
        rows = tuple(
            _read_ids_path(item, components, preserved_positions, index + 1)
            for item in child
        )
        if not rows:
            raise KeyError("/".join(components))
        if index in preserved_positions or len(rows) > 1:
            return np.stack(rows)
        return rows[0]
    return _read_ids_path(child, components, preserved_positions, index + 1)


class _NetCDFStoreArrays(AbstractContextManager["_NetCDFStoreArrays"]):
    def __init__(self, path: Path, dd_version: str) -> None:
        self.path = path
        self.dd_version = dd_version

    def __enter__(self) -> Self:
        if not self.path.is_dir():
            raise SourceUnavailableError(f"netCDF pulse store is absent: {self.path}")
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self, binding: ChannelBinding) -> np.ndarray:
        source = self.path / f"{binding.name}.nc"
        if not source.is_file():
            raise KeyError(binding.name)
        ids_name, relative_path = binding.dd_path.split("/", maxsplit=1)
        try:
            with imas.DBEntry(source, "r", dd_version=self.dd_version) as entry:
                ids = entry.get(ids_name, autoconvert=False)
                components = tuple(relative_path.split("/"))
                positions = _struct_array_positions(ids, components)
                leaf_rank = ids.metadata[relative_path].ndim
                structural_rank = binding.source_rank - leaf_rank
                if structural_rank < 0 or structural_rank > len(positions):
                    raise BindingTransformError(
                        f"binding {binding.name!r} source rank {binding.source_rank} "
                        f"cannot be reconstructed from leaf rank {leaf_rank} and "
                        f"{len(positions)} structural arrays"
                    )
                preserved = (
                    frozenset(positions[-structural_rank:])
                    if structural_rank
                    else frozenset()
                )
                return _read_ids_path(ids, components, preserved)
        except OSError as error:
            raise SourceUnavailableError(
                f"netCDF binding store cannot be opened: {source}: {error}"
            ) from error


class NetCDFTransformEngine:
    """Read logical source groups from one netCDF file per pulse."""

    format_name = "netcdf"

    def open(self, root: Path | str, shot: int, dd_version: str) -> _NetCDFStoreArrays:
        """Open the IMAS-netCDF binding files for one pulse."""
        return _NetCDFStoreArrays(Path(root) / str(int(shot)), dd_version)


_TRANSFORM_ENGINES: Mapping[str, StoreEngine] = MappingProxyType(
    {
        "netcdf": NetCDFTransformEngine(),
        "zarr": ZarrTransformEngine(),
    }
)
TRANSFORM_ENGINE_FORMATS = tuple(_TRANSFORM_ENGINES)


def get_transform_engine(store_format: str) -> StoreEngine:
    """Return the single registered engine for ``store_format``."""
    try:
        return _TRANSFORM_ENGINES[store_format]
    except KeyError as error:
        raise TransformEngineError(
            f"unsupported store format {store_format!r}; "
            f"available={TRANSFORM_ENGINE_FORMATS}"
        ) from error


def _apply_sign_convention(values: np.ndarray, binding: ChannelBinding) -> np.ndarray:
    if binding.sign_convention in {"identity", "not-applicable"}:
        emitted = np.array(values, copy=True)
    elif binding.sign_convention == "negate":
        emitted = np.negative(values)
    else:
        raise BindingTransformError(
            f"binding {binding.name!r} has unresolved sign convention "
            f"{binding.sign_convention!r}"
        )
    emitted.setflags(write=False)
    return emitted


@cache
def _target_cocos_transformation(dd_version: str, dd_path: str) -> str | None:
    """Resolve the nearest COCOS class declared on a DD target or its parents."""

    ids_name, relative_path = dd_path.split("/", maxsplit=1)
    metadata = imas.IDSFactory(dd_version).new(ids_name).metadata
    components = relative_path.split("/")
    for size in range(len(components), 0, -1):
        node = metadata["/".join(components[:size])]
        transformation = getattr(node, "cocos_label_transformation", None)
        if transformation:
            return str(transformation)
    return None


def _apply_cocos_convention(
    values: np.ndarray,
    binding: ChannelBinding,
    dd_version: str,
    source_cocos: int | None,
) -> tuple[np.ndarray, str | None, float]:
    transformation = _target_cocos_transformation(dd_version, binding.dd_path)
    if transformation is None:
        return values, None, 1.0
    if source_cocos is None:
        raise BindingTransformError(
            f"binding {binding.name!r} targets COCOS-dependent DD path "
            f"{binding.dd_path!r} ({transformation}) but has no declared source "
            "COCOS convention"
        )
    try:
        factor = canonical_factor(transformation, source_cocos=int(source_cocos))
    except (KeyError, TypeError, ValueError) as error:
        raise BindingTransformError(
            f"binding {binding.name!r} targets unsupported COCOS transformation "
            f"{transformation!r} at {binding.dd_path!r} from source COCOS "
            f"{source_cocos!r}"
        ) from error
    emitted = np.multiply(values, factor)
    emitted.setflags(write=False)
    return emitted, transformation, factor


def _emit_arrays(
    source: StoreArrays,
    bindings: tuple[ChannelBinding, ...],
    dd_version: str,
    source_cocos: int | None,
) -> tuple[tuple[EmittedArray, ...], tuple[str, ...]]:
    emitted: list[EmittedArray] = []
    missing: list[str] = []
    for binding in bindings:
        try:
            values = source.read(binding)
        except KeyError:
            missing.append(binding.name)
            continue
        signed_values = _apply_sign_convention(values, binding)
        transformed_values, transformation, factor = _apply_cocos_convention(
            signed_values,
            binding,
            dd_version,
            (
                binding.source_cocos_override
                if binding.source_cocos_override is not None
                else source_cocos
            ),
        )
        emitted.append(
            EmittedArray(
                binding_name=binding.name,
                source_group=binding.source_group,
                source_array=binding.source_array,
                dd_path=binding.dd_path,
                source_unit=binding.source_unit,
                target_unit=binding.target_unit,
                cocos_transformation=transformation,
                cocos_factor=factor,
                values=transformed_values,
            )
        )
    return tuple(emitted), tuple(missing)


def transform_machine_description(
    catalog: MachineMapCatalog,
    shot: int,
    store_format: str,
    store_root: Path | str,
    *,
    source_cocos: int | None | object = _CATALOG_SOURCE_COCOS,
) -> MachineDescription:
    """Select a map by shot and emit its available arrays through one engine.

    A missing pulse is represented as a result so callers can retain the
    distinction between a declared source-only catalog and observed corpus
    data.  Missing arrays within an available pulse are reported individually.
    Other read or transform failures remain visible exceptions.
    """
    shot_id = int(shot)
    machine_map = map_for_shot(catalog, shot_id)
    engine = get_transform_engine(store_format)
    bindings = catalog.bindings_for(machine_map)
    declared_source_cocos = (
        catalog.cocos_for_binding()
        if source_cocos is _CATALOG_SOURCE_COCOS
        else source_cocos
    )
    try:
        with engine.open(store_root, shot_id, catalog.dd_version) as source:
            arrays, missing = _emit_arrays(
                source,
                bindings,
                catalog.dd_version,
                declared_source_cocos,
            )
    except SourceUnavailableError as error:
        return MachineDescription(
            shot=shot_id,
            store_format=engine.format_name,
            machine_map=machine_map,
            dd_version=catalog.dd_version,
            source_cocos=declared_source_cocos,
            status="source-unavailable",
            arrays=(),
            missing_bindings=tuple(binding.name for binding in bindings),
            detail=str(error),
        )

    return MachineDescription(
        shot=shot_id,
        store_format=engine.format_name,
        machine_map=machine_map,
        dd_version=catalog.dd_version,
        source_cocos=declared_source_cocos,
        status="emitted",
        arrays=arrays,
        missing_bindings=missing,
        detail=(
            f"emitted {len(arrays)} arrays; "
            f"{len(missing)} declared arrays are absent from the pulse store"
        ),
    )
