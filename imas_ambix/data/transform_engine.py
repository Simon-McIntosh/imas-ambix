"""Apply declarative machine maps through format-scoped store engines.

The engine registry varies only with the physical store format.  Map selection,
binding traversal, sign handling, missing-array accounting, and DD-path
emission are shared by every format and every machine catalog.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, Self

import imas
import numpy as np
import zarr
from imas.ids_struct_array import IDSStructArray

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
    values: np.ndarray


@dataclass(frozen=True)
class MachineDescription:
    """The observable outcome of applying one range-scoped machine map."""

    shot: int
    store_format: str
    machine_map: MachineMap
    dd_version: str
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


def _read_ids_path(node: object, components: tuple[str, ...]) -> np.ndarray:
    child = getattr(node, components[0])
    if len(components) == 1:
        return np.asarray(child.value)
    if isinstance(child, IDSStructArray):
        rows = tuple(_read_ids_path(item, components[1:]) for item in child)
        if not rows:
            raise KeyError("/".join(components))
        return rows[0] if len(rows) == 1 else np.stack(rows)
    return _read_ids_path(child, components[1:])


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
                return _read_ids_path(ids, tuple(relative_path.split("/")))
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


def _emit_arrays(
    source: StoreArrays, bindings: tuple[ChannelBinding, ...]
) -> tuple[tuple[EmittedArray, ...], tuple[str, ...]]:
    emitted: list[EmittedArray] = []
    missing: list[str] = []
    for binding in bindings:
        try:
            values = source.read(binding)
        except KeyError:
            missing.append(binding.name)
            continue
        emitted.append(
            EmittedArray(
                binding_name=binding.name,
                source_group=binding.source_group,
                source_array=binding.source_array,
                dd_path=binding.dd_path,
                source_unit=binding.source_unit,
                target_unit=binding.target_unit,
                values=_apply_sign_convention(values, binding),
            )
        )
    return tuple(emitted), tuple(missing)


def transform_machine_description(
    catalog: MachineMapCatalog,
    shot: int,
    store_format: str,
    store_root: Path | str,
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
    try:
        with engine.open(store_root, shot_id, catalog.dd_version) as source:
            arrays, missing = _emit_arrays(source, bindings)
    except SourceUnavailableError as error:
        return MachineDescription(
            shot=shot_id,
            store_format=engine.format_name,
            machine_map=machine_map,
            dd_version=catalog.dd_version,
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
        status="emitted",
        arrays=arrays,
        missing_bindings=missing,
        detail=(
            f"emitted {len(arrays)} arrays; "
            f"{len(missing)} declared arrays are absent from the pulse store"
        ),
    )
