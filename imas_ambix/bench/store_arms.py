"""Construct equivalent native, IMAS-netCDF, and DD-annotated zarr payloads.

The fixed benchmark payload is a short, finite slice of the MAST level-2
``summary`` group.  Values are copied without conversion so strict bitwise
comparisons can detect any storage-arm drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imas
import numpy as np
import xarray as xr
import zarr
from imas.backends.netcdf.ids_tensorizer import IDSTensorizer
from imas.ids_data_type import IDSDataType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FIXED_MAST_SHOT = 11766
FIXED_SAMPLE_START = 0
FIXED_SAMPLE_STOP = 64
FIXED_IDS_NAME = "summary"


@dataclass(frozen=True)
class ChannelBinding:
    """Bind one native level-2 array to its numeric Data Dictionary leaf."""

    native_name: str
    dd_path: str


FIXED_CHANNELS = (
    ChannelBinding("time", "time"),
    ChannelBinding("ip", "global_quantities/ip/value"),
    ChannelBinding("power_radiated", "global_quantities/power_radiated/value"),
)


@dataclass(frozen=True)
class PayloadArray:
    """One unchanged source array and the semantics attached to it."""

    values: np.ndarray
    dd_path: str
    units: str


@dataclass(frozen=True)
class ShotPayload:
    """The fixed shot slice shared by all storage arms."""

    shot: int
    ids_name: str
    sample_start: int
    sample_stop: int
    arrays: Mapping[str, PayloadArray]

    @property
    def sample_count(self) -> int:
        """Number of samples in every channel."""
        return self.sample_stop - self.sample_start


@dataclass(frozen=True)
class WriterReceipt:
    """Public implementation seam exercised by a payload writer."""

    entrypoint: str
    base_class: str | None
    called_methods: tuple[str, ...]
    conditional_methods: tuple[str, ...] = ()
    locally_defined_methods: tuple[str, ...] = ()
    supporting_methods: tuple[str, ...] = ()
    private_names: tuple[str, ...] = ()


NETCDF_WRITER_RECEIPT = WriterReceipt(
    entrypoint="imas.DBEntry.put",
    base_class=None,
    called_methods=("put",),
)

ZARR_WRITER_RECEIPT = WriterReceipt(
    entrypoint="imas_ambix.bench.store_arms.IDSZarrWriter.write",
    base_class="imas.backends.netcdf.ids_tensorizer.IDSTensorizer",
    called_methods=(
        "include_coordinate_paths",
        "collect_filled_data",
        "determine_data_shapes",
        "get_dimensions",
        "tensorize",
        "get_attributes",
    ),
    conditional_methods=(
        "get_shape_dimensions",
        "get_shape_attributes",
    ),
    locally_defined_methods=(),
    supporting_methods=("filter_coordinates",),
)


_FILL_VALUES = {
    IDSDataType.INT: np.int32(-(2**31) + 1),
    IDSDataType.STR: "",
    IDSDataType.FLT: np.nan,
    IDSDataType.CPX: np.nan * (1 + 1j),
}


def read_fixed_native_payload(
    level2_root: Path | str,
    *,
    shot: int = FIXED_MAST_SHOT,
    sample_start: int = FIXED_SAMPLE_START,
    sample_stop: int | None = FIXED_SAMPLE_STOP,
    channels: Sequence[ChannelBinding] = FIXED_CHANNELS,
) -> ShotPayload:
    """Read the fixed MAST slice directly from its native level-2 arrays."""
    shot_path = Path(level2_root) / f"{shot}.zarr"
    group = zarr.open_group(shot_path, mode="r")[FIXED_IDS_NAME]
    arrays: dict[str, PayloadArray] = {}
    actual_stop: int | None = None

    for binding in channels:
        source = group[binding.native_name]
        values = np.asarray(source[sample_start:sample_stop])
        if values.ndim != 1:
            raise ValueError(
                f"{binding.native_name!r} does not provide the requested 1D slice"
            )
        channel_stop = sample_start + values.shape[0]
        if actual_stop is None:
            actual_stop = channel_stop
        if channel_stop != actual_stop:
            raise ValueError("all payload channels must have an equal sample count")
        if sample_stop is not None and channel_stop != sample_stop:
            raise ValueError(
                f"{binding.native_name!r} does not provide the requested 1D slice"
            )

        source_path = source.attrs.get("imas")
        dd_path = f"{FIXED_IDS_NAME}/{binding.dd_path}"
        if source_path:
            source_path = str(source_path).replace(".", "/")
            if dd_path != source_path and not dd_path.startswith(f"{source_path}/"):
                raise ValueError(
                    f"{binding.native_name!r} advertises {source_path!r}, "
                    f"not {dd_path!r}"
                )

        units = str(source.attrs.get("units", ""))
        if not units:
            raise ValueError(f"{binding.native_name!r} has no units metadata")
        arrays[binding.dd_path] = PayloadArray(values.copy(), dd_path, units)

    if actual_stop is None:
        raise ValueError("at least one channel is required")

    return ShotPayload(
        shot=shot,
        ids_name=FIXED_IDS_NAME,
        sample_start=sample_start,
        sample_stop=actual_stop,
        arrays=arrays,
    )


def payload_to_ids(payload: ShotPayload, *, dd_version: str | None = None) -> Any:
    """Populate an IDS with the fixed arrays and verify DD units at the boundary."""
    factory = imas.IDSFactory(dd_version)
    ids = factory.new(payload.ids_name)
    ids.ids_properties.homogeneous_time = 1

    for path, payload_array in payload.arrays.items():
        metadata = ids.metadata[path]
        if metadata.units != payload_array.units:
            raise ValueError(
                f"units mismatch for {path}: source={payload_array.units!r}, "
                f"DD={metadata.units!r}"
            )
        ids[path].value = payload_array.values.copy()

    return ids


def write_imas_netcdf(
    ids: Any, destination: Path | str, *, dd_version: str | None = None
) -> WriterReceipt:
    """Write an IDS through the public :class:`imas.DBEntry` netCDF path."""
    path = Path(destination)
    if path.suffix != ".nc":
        raise ValueError("IMAS netCDF destinations must end in '.nc'")
    with imas.DBEntry(path, "w", dd_version=dd_version) as entry:
        entry.put(ids)
    return NETCDF_WRITER_RECEIPT


def read_imas_netcdf(
    source: Path | str,
    paths: Sequence[str],
    *,
    ids_name: str = FIXED_IDS_NAME,
    dd_version: str | None = None,
) -> dict[str, np.ndarray]:
    """Read selected arrays through the public :class:`imas.DBEntry` path."""
    with imas.DBEntry(Path(source), "r", dd_version=dd_version) as entry:
        ids = entry.get(ids_name, autoconvert=False)
        return {path: np.asarray(ids[path].value) for path in paths}


class IDSZarrWriter(IDSTensorizer):
    """Write zarr as a public sibling subclass of imas-python's ``IDS2NC``."""

    def __init__(self, ids: Any, paths: Sequence[str]) -> None:
        normalized_paths = [ids.metadata[path].path_string for path in paths]
        super().__init__(ids, normalized_paths)

    def to_dataset(self) -> xr.Dataset:
        """Tensorize selected IDS paths into a DD-annotated xarray dataset."""
        self.include_coordinate_paths()
        self.collect_filled_data()
        self.determine_data_shapes()

        data_vars: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
        coordinate_names: set[str] = set()
        for path in self.filled_data:
            metadata = self.ids.metadata[path]
            if metadata.data_type in (IDSDataType.STRUCTURE, IDSDataType.STRUCT_ARRAY):
                continue

            name = path.replace("/", ".")
            dimensions = self.get_dimensions(path)
            data = self.tensorize(path, _FILL_VALUES[metadata.data_type])
            attrs = self.get_attributes(path, _FILL_VALUES)
            attrs["dd_path"] = f"{self.ids.metadata.name}/{path}"
            attrs["units"] = metadata.units or "1"
            coordinate_names.update(attrs.get("coordinates", "").split())
            data_vars[name] = (dimensions, data, attrs)

            if path in self.shapes and metadata.ndim:
                shape_name = f"{name}:shape"
                shape_attrs = self.get_shape_attributes(name)
                shape_attrs.update(
                    dd_path=f"{self.ids.metadata.name}/{path}", units="1"
                )
                data_vars[shape_name] = (
                    self.get_shape_dimensions(path),
                    self.shapes[path],
                    shape_attrs,
                )

        coordinates = {
            name: data_vars.pop(name) for name in coordinate_names if name in data_vars
        }
        return xr.Dataset(data_vars=data_vars, coords=coordinates)

    def write(self, destination: Path | str) -> WriterReceipt:
        """Write the tensorized dataset through xarray's public zarr serializer."""
        dataset = self.to_dataset()
        dataset.to_zarr(Path(destination), mode="w", consolidated=False)
        return ZARR_WRITER_RECEIPT


def read_dd_zarr(source: Path | str, paths: Sequence[str]) -> dict[str, np.ndarray]:
    """Read DD-path-selected arrays from the zarr arm."""
    group = zarr.open_group(Path(source), mode="r")
    return {path: np.asarray(group[path.replace("/", ".")][:]) for path in paths}
