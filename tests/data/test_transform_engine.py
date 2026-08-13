"""Format-scoped machine-map transforms over authoritative store paths."""

from __future__ import annotations

from pathlib import Path

import imas
import numpy as np
import pytest
import zarr
from imas.ids_data_type import IDSDataType
from imas.ids_struct_array import IDSStructArray

from imas_ambix.bench.store_arms import read_imas_netcdf, write_imas_netcdf
from imas_ambix.data.machine_map import ChannelBinding, load_packaged_machine_map
from imas_ambix.data.transform_engine import (
    TRANSFORM_ENGINE_FORMATS,
    transform_machine_description,
)

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
TRANSITION_SHOTS = (11_766, 12_417, 12_533)


def _assert_array_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype.kind in "fc" or expected.dtype.kind in "fc":
        assert np.array_equal(actual, expected, equal_nan=True)
    else:
        assert np.array_equal(actual, expected)


def _array_positions(ids: object, relative_path: str) -> tuple[int, ...]:
    positions: list[int] = []
    node = ids
    for index, component in enumerate(relative_path.split("/")[:-1]):
        node = getattr(node, component)
        if isinstance(node, IDSStructArray):
            positions.append(index)
            node.resize(1)
            node = node[0]
    return tuple(positions)


def _assign_values(
    ids: object, relative_path: str, values: np.ndarray
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    metadata = ids.metadata[relative_path]
    if metadata.data_type in {IDSDataType.STRUCTURE, IDSDataType.STRUCT_ARRAY}:
        raise ValueError("the declared DD path identifies a structure, not a leaf")

    extra_dimensions = values.ndim - metadata.ndim
    array_positions = _array_positions(ids, relative_path)
    if not 0 <= extra_dimensions <= len(array_positions):
        raise ValueError(
            f"source rank {values.ndim} cannot populate DD leaf rank {metadata.ndim}"
        )

    components = relative_path.split("/")
    expanded_positions = set(
        array_positions[-extra_dimensions:] if extra_dimensions else ()
    )
    concrete_paths: list[str] = []

    def assign(node: object, index: int, value: object, prefix: str) -> None:
        component = components[index]
        child = getattr(node, component)
        if index == len(components) - 1:
            child.value = value
            concrete_paths.append(f"{prefix}{component}")
            return
        if isinstance(child, IDSStructArray):
            count = len(value) if index in expanded_positions else 1
            child.resize(count)
            if index in expanded_positions:
                for item_index in range(count):
                    assign(
                        child[item_index],
                        index + 1,
                        value[item_index],
                        f"{prefix}{component}[{item_index}]/",
                    )
            else:
                assign(
                    child[0],
                    index + 1,
                    value,
                    f"{prefix}{component}[0]/",
                )
            return
        assign(child, index + 1, value, f"{prefix}{component}/")

    assign(ids, 0, values, "")
    return tuple(concrete_paths), tuple(values.shape[:extra_dimensions])


def _binding_ids(
    factory: imas.IDSFactory, binding: ChannelBinding, values: np.ndarray
) -> tuple[object, tuple[str, ...], tuple[int, ...]]:
    ids_name, relative_path = binding.dd_path.split("/", maxsplit=1)
    ids = factory.new(ids_name)
    ids.ids_properties.homogeneous_time = 1
    concrete_paths, structural_shape = _assign_values(ids, relative_path, values)
    metadata = ids.metadata[relative_path]
    if relative_path == "description_2d/limiter/unit/outline/z":
        for concrete_path in concrete_paths:
            ids[concrete_path.removesuffix("/z") + "/r"].value = np.zeros_like(values)
    if (
        ids_name in {"magnetics", "pf_active"}
        and metadata.ndim
        and relative_path != "time"
    ):
        ids.time = np.arange(values.shape[-1], dtype=np.float64)
    return ids, concrete_paths, structural_shape


def _public_netcdf_values(
    source: Path,
    binding: ChannelBinding,
    concrete_paths: tuple[str, ...],
    structural_shape: tuple[int, ...],
    dd_version: str,
) -> np.ndarray:
    ids_name = binding.dd_path.split("/", maxsplit=1)[0]
    arrays = read_imas_netcdf(
        source,
        concrete_paths,
        ids_name=ids_name,
        dd_version=dd_version,
    )
    values = tuple(arrays[path] for path in concrete_paths)
    if not structural_shape:
        return values[0]
    return np.stack(values).reshape(structural_shape + values[0].shape)


def _write_netcdf_fixture(
    destination: Path,
    factory: imas.IDSFactory,
    binding: ChannelBinding,
    values: np.ndarray,
    dd_version: str,
) -> None:
    ids, concrete_paths, structural_shape = _binding_ids(factory, binding, values)
    receipt = write_imas_netcdf(ids, destination, dd_version=dd_version)
    assert receipt.entrypoint == "imas.DBEntry.put"
    authoritative = _public_netcdf_values(
        destination,
        binding,
        concrete_paths,
        structural_shape,
        dd_version,
    )
    _assert_array_equal(authoritative, values)


@pytest.mark.skipif(
    not all((LEVEL2_ROOT / f"{shot}.zarr").is_dir() for shot in TRANSITION_SHOTS),
    reason="FAIR-MAST level-2 transition stores are not mounted",
)
def test_two_format_engines_emit_three_range_scoped_descriptions(tmp_path):
    catalog = load_packaged_machine_map("mast")
    factory = imas.IDSFactory(catalog.dd_version)
    netcdf_root = tmp_path / "netcdf"
    exception_reasons: dict[str, str] = {}
    singleton_structural_bindings: set[tuple[int, str]] = set()
    receipts = []

    for shot in TRANSITION_SHOTS:
        zarr_result = transform_machine_description(catalog, shot, "zarr", LEVEL2_ROOT)
        assert zarr_result.status == "emitted"
        assert zarr_result.machine_map.first_shot == shot

        direct_store = zarr.open_group(LEVEL2_ROOT / f"{shot}.zarr", mode="r")
        declared_bindings = catalog.bindings_for(zarr_result.machine_map)
        executable_bindings = tuple(
            binding
            for binding in declared_bindings
            if f"{binding.source_group}/{binding.source_array}" in direct_store
        )
        unavailable_bindings = tuple(
            binding
            for binding in declared_bindings
            if f"{binding.source_group}/{binding.source_array}" not in direct_store
        )
        direct_values: dict[str, np.ndarray] = {}
        for emitted in zarr_result.arrays:
            direct = np.asarray(
                direct_store[f"{emitted.source_group}/{emitted.source_array}"][...]
            )
            direct_values[emitted.binding_name] = direct
            _assert_array_equal(emitted.values, direct)
            ids_name, relative_path = emitted.dd_path.split("/", maxsplit=1)
            if (
                direct.shape == (1,)
                and factory.new(ids_name).metadata[relative_path].ndim == 0
            ):
                singleton_structural_bindings.add((shot, emitted.binding_name))

        shot_directory = netcdf_root / str(shot)
        shot_directory.mkdir(parents=True)
        bindings = {
            binding.name: binding
            for binding in catalog.bindings_for(zarr_result.machine_map)
        }
        for emitted in zarr_result.arrays:
            binding = bindings[emitted.binding_name]
            try:
                _write_netcdf_fixture(
                    shot_directory / f"{binding.name}.nc",
                    factory,
                    binding,
                    direct_values[binding.name],
                    catalog.dd_version,
                )
            except ValueError as error:
                exception_reasons[binding.source_array] = str(error)

        netcdf_result = transform_machine_description(
            catalog, shot, "netcdf", netcdf_root
        )
        assert netcdf_result.status == "emitted"
        assert netcdf_result.machine_map == zarr_result.machine_map
        for emitted in netcdf_result.arrays:
            _assert_array_equal(emitted.values, direct_values[emitted.binding_name])
        receipts.append(
            (
                shot,
                zarr_result,
                netcdf_result,
                executable_bindings,
                unavailable_bindings,
            )
        )

    format_gaps = tuple(
        zarr_result.emitted_array_count - netcdf_result.emitted_array_count
        for _, zarr_result, netcdf_result, _, _ in receipts
    )
    assert (format_gaps, len(exception_reasons)) == (
        tuple(0 for _ in receipts),
        0,
    )
    assert singleton_structural_bindings
    print(f"SINGLETON_STRUCTURAL_ARRAYS count={len(singleton_structural_bindings)}")

    for (
        shot,
        zarr_result,
        netcdf_result,
        executable_bindings,
        unavailable_bindings,
    ) in receipts:
        executable_binding_names = tuple(
            binding.name for binding in executable_bindings
        )
        unavailable_binding_names = tuple(
            binding.name for binding in unavailable_bindings
        )
        executable_binding_count = len(executable_bindings)
        assert zarr_result.emitted_array_count == executable_binding_count
        assert netcdf_result.emitted_array_count == executable_binding_count
        assert tuple(array.binding_name for array in zarr_result.arrays) == (
            executable_binding_names
        )
        assert tuple(array.binding_name for array in netcdf_result.arrays) == (
            executable_binding_names
        )
        assert zarr_result.missing_bindings == unavailable_binding_names
        assert netcdf_result.missing_bindings == unavailable_binding_names
        print(
            "EMITTED_ARRAY_COUNT "
            f"shot={shot} zarr={zarr_result.emitted_array_count} "
            f"netcdf={netcdf_result.emitted_array_count} "
            f"executable_bindings={executable_binding_count} "
            f"source_unavailable={len(unavailable_bindings)}"
        )


@pytest.mark.parametrize("store_format", TRANSFORM_ENGINE_FORMATS)
def test_source_only_catalog_uses_the_same_no_corpus_entry_point(
    tmp_path, store_format
):
    catalog = load_packaged_machine_map("diii-d")
    result = transform_machine_description(
        catalog, 170_000, store_format, tmp_path / "unmounted-corpus"
    )

    assert result.status == "source-unavailable"
    assert result.store_format == store_format
    assert result.emitted_array_count == 0
    assert result.missing_bindings == ("diii-d-plasma-current",)
    assert "pulse store is absent" in result.detail


def test_engine_registry_is_format_scoped_and_has_no_machine_conditionals():
    import imas_ambix.data.transform_engine as engine_module

    source = Path(engine_module.__file__).read_text().lower()
    assert TRANSFORM_ENGINE_FORMATS == ("netcdf", "zarr")
    assert len(TRANSFORM_ENGINE_FORMATS) == 2
    assert "mast" not in source
    assert "diii-d" not in source
    assert "d3d" not in source
