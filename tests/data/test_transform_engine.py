"""Format-scoped machine-map transforms over authoritative store paths."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import imas
import numpy as np
import pytest
import zarr
from imas.ids_data_type import IDSDataType
from imas.ids_struct_array import IDSStructArray

from imas_alambic.machine_map import (
    ChannelBinding,
    MachineMapError,
    discover_bundles,
    load_packaged_machine_map,
    map_for_shot,
)
from imas_alambic.transform_engine import (
    TRANSFORM_ENGINE_FORMATS,
    BindingTransformError,
    TransformEngineError,
    get_transform_engine,
    transform_machine_description,
)
from imas_ambix.bench.store_arms import read_imas_netcdf, write_imas_netcdf
from imas_ambix.data.cocos_convention import (
    MAST_LEVEL2_SIGN_TABLE,
    MAST_SOURCE_COCOS,
    MAST_TO_COCOS_17_FACTORS,
)
from imas_ambix.data.paths import JT60SA_DESCRIPTION_DIR

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
TRANSITION_SHOTS = (11_766, 12_417, 12_533)
JT60SA_OP1_DESCRIPTION_ROOT = JT60SA_DESCRIPTION_DIR
JT60SA_OP1_PF_ACTIVE = JT60SA_OP1_DESCRIPTION_ROOT / "OP1" / "pf_active.nc"
_COIL_ELEMENT_DD_PATH = "pf_active/coil/element/geometry/rectangle/r"
try:
    discover_bundles()
    _BUNDLE_DISCOVERY_AVAILABLE = True
except MachineMapError:
    _BUNDLE_DISCOVERY_AVAILABLE = False
needs_bundle_discovery = pytest.mark.skipif(
    not _BUNDLE_DISCOVERY_AVAILABLE,
    reason="IMAS_ALAMBIC_MAP_PATH must name a readable bundle",
)


def _coil_element_binding(struct_array_entry=None):
    return ChannelBinding(
        name="jt60sa-coil-element-r",
        source_group="pf_active",
        source_array="rectangle_r",
        source_rank=1,
        source_role="value",
        source_location="file:///machine_description/OP1/pf_active.nc",
        dd_path=_COIL_ELEMENT_DD_PATH,
        source_unit="m",
        target_unit="m",
        sign_convention="identity",
        evidence="synthetic coil-element rectangle radius",
        source_cocos_override=None,
        struct_array_entry=struct_array_entry,
    )


def _probe_angle_binding(sign_convention="identity"):
    return ChannelBinding(
        name="jt60sa-probe-poloidal-angle",
        source_group="magnetics",
        source_array="poloidal_angle",
        source_rank=1,
        source_role="value",
        source_location="file:///machine_description/poloidal_angle",
        dd_path="magnetics/b_field_pol_probe/poloidal_angle",
        source_unit="rad",
        target_unit="rad",
        sign_convention=sign_convention,
        evidence="synthetic directed probe poloidal angle",
        source_cocos_override=None,
    )


def _binding_payload(**overrides):
    payload = {
        "name": "synthetic-binding",
        "source_group": "pf_active",
        "source_array": "rectangle_r",
        "source_rank": 1,
        "source_role": "value",
        "source_location": "file:///synthetic/pf_active.nc",
        "dd_path": _COIL_ELEMENT_DD_PATH,
        "source_unit": "m",
        "target_unit": "m",
        "sign_convention": "identity",
        "evidence": "synthetic",
    }
    payload.update(overrides)
    return payload


def _write_ragged_coil_store(destination, dd_version, coils):
    factory = imas.IDSFactory(dd_version)
    ids = factory.new("pf_active")
    ids.ids_properties.homogeneous_time = 1
    ids.coil.resize(len(coils))
    for index, (name, values) in enumerate(coils):
        ids.coil[index].name = name
        ids.coil[index].element.resize(len(values))
        for element_index, value in enumerate(values):
            ids.coil[index].element[element_index].geometry.rectangle.r = value
    with imas.DBEntry(destination, "w", dd_version=dd_version) as entry:
        entry.put(ids)


def _catalog_with_only_plasma_current():
    catalog = load_packaged_machine_map("mast")
    binding = next(
        binding
        for bindings in catalog.binding_sets.values()
        for binding in bindings
        if binding.name == "mast-magnetics-ip"
    )
    binding_set = "plasma-current-only"
    machine_map = replace(
        catalog.maps[0],
        first_shot=min(row.shot for row in MAST_LEVEL2_SIGN_TABLE),
        last_shot=max(row.shot for row in MAST_LEVEL2_SIGN_TABLE),
        transition=None,
        binding_set=binding_set,
        drive_topology=None,
    )
    return replace(
        catalog,
        binding_sets=MappingProxyType({binding_set: (binding,)}),
        maps=(machine_map,),
        validation_gaps=(),
        source_qualifications=(),
        drive_topologies=(),
        structure_assemblies=(),
    )


def _write_plasma_current_store(root: Path, shot: int, values: np.ndarray) -> None:
    store = zarr.open_group(root / f"{shot}.zarr", mode="w")
    magnetics = store.create_group("magnetics")
    magnetics.create_array("ip", data=values)


def _assert_array_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype.kind in "fc" or expected.dtype.kind in "fc":
        assert np.array_equal(actual, expected, equal_nan=True)
    else:
        assert np.array_equal(actual, expected)


def _apply_expected_cocos_factor(values: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return values
    return np.multiply(values, factor)


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
@needs_bundle_discovery
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
        assert zarr_result.machine_map == map_for_shot(catalog, shot)

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
            _assert_array_equal(
                emitted.values,
                _apply_expected_cocos_factor(direct, emitted.cocos_factor),
            )
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
            _assert_array_equal(
                emitted.values,
                _apply_expected_cocos_factor(
                    direct_values[emitted.binding_name], emitted.cocos_factor
                ),
            )
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
@needs_bundle_discovery
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


@needs_bundle_discovery
def test_netcdf_static_store_reads_one_directory_for_every_shot_in_the_map(tmp_path):
    catalog = _catalog_with_only_plasma_current()
    machine_map = replace(
        catalog.maps[0],
        name="synthetic-static-map",
        first_shot=100,
        last_shot=200,
        transition=None,
    )
    catalog = replace(
        catalog,
        maps=(machine_map,),
        description_store_layout="static-over-map",
    )
    binding = catalog.bindings_for(machine_map)[0]

    factory = imas.IDSFactory(catalog.dd_version)
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    static_directory = tmp_path / machine_map.name
    static_directory.mkdir(parents=True)
    _write_netcdf_fixture(
        static_directory / "magnetics.nc",
        factory,
        binding,
        values,
        catalog.dd_version,
    )

    for shot in (101, 199):
        result = transform_machine_description(catalog, shot, "netcdf", tmp_path)
        assert result.status == "emitted"
        assert result.machine_map == machine_map
        assert result.emitted_array_count == 1
        _assert_array_equal(result.arrays[0].values, values)

    assert not (tmp_path / "101").exists()
    assert not (tmp_path / "199").exists()
    assert {entry.name for entry in tmp_path.iterdir()} == {machine_map.name}
    print(f"STATIC_STORE map={machine_map.name} shot_dirs=0 reads=2 emitted=1")


def test_zarr_engine_refuses_a_static_over_map_layout(tmp_path):
    engine = get_transform_engine("zarr")
    with pytest.raises(TransformEngineError, match="static-over-map"):
        engine.open(tmp_path, 101, "4.1.1", store_layout="static-over-map")


@needs_bundle_discovery
def test_zarr_catalog_declaring_static_over_map_is_refused_not_read_per_shot(
    tmp_path,
):
    catalog = _catalog_with_only_plasma_current()
    machine_map = replace(
        catalog.maps[0],
        name="synthetic-zarr-static-map",
        first_shot=100,
        last_shot=200,
        transition=None,
    )
    catalog = replace(
        catalog,
        maps=(machine_map,),
        description_store_format="zarr",
        description_store_layout="static-over-map",
    )
    with pytest.raises(TransformEngineError, match="static-over-map"):
        transform_machine_description(catalog, 150, "zarr", tmp_path)


def test_channel_binding_struct_array_entry_slot_round_trips_and_rejects_bad_values():
    selected = ChannelBinding.from_dict(
        _binding_payload(struct_array_entry="CS1"), "binding"
    )
    assert selected.struct_array_entry == "CS1"
    omitted = ChannelBinding.from_dict(_binding_payload(), "binding")
    assert omitted.struct_array_entry is None

    for bad_value in ("", 5, None):
        with pytest.raises(MachineMapError):
            ChannelBinding.from_dict(
                _binding_payload(struct_array_entry=bad_value), "binding"
            )
    print("STRUCT_ENTRY_SLOT accepted=CS1 omitted=None rejected=empty,nonstring")


def test_netcdf_binding_reads_one_named_entry_of_a_ragged_struct_array(tmp_path):
    dd_version = "4.1.1"
    store_directory = tmp_path / "OP1"
    store_directory.mkdir()
    expected = {"A": [0.0, 1.0, 2.0], "B": [10.0, 11.0], "C": [20.0]}
    _write_ragged_coil_store(
        store_directory / "pf_active.nc",
        dd_version,
        tuple(expected.items()),
    )

    engine = get_transform_engine("netcdf")
    machine_map = SimpleNamespace(name="OP1")
    with engine.open(
        str(tmp_path), 100_001, dd_version, machine_map, "static-over-map"
    ) as source:
        for name, values in expected.items():
            array = source.read(_coil_element_binding(name))
            assert array.tolist() == values

        with pytest.raises(TransformEngineError) as missing:
            source.read(_coil_element_binding("ZZ"))
        assert "ZZ" in str(missing.value)
        assert "jt60sa-coil-element-r" in str(missing.value)

        with pytest.raises(TransformEngineError) as ragged:
            source.read(_coil_element_binding())
        message = str(ragged.value)
        assert _COIL_ELEMENT_DD_PATH in message
        assert "[3, 2, 1]" in message

    lengths = [len(values) for values in expected.values()]
    print(f"STRUCT_ENTRY lengths={lengths} ragged_counts=[3, 2, 1]")


def test_selecting_binding_counts_only_levels_below_the_consumed_struct_array(
    tmp_path,
):
    dd_version = "4.1.1"
    store_directory = tmp_path / "OP1"
    store_directory.mkdir()
    _write_ragged_coil_store(
        store_directory / "pf_active.nc",
        dd_version,
        (("A", [0.0, 1.0, 2.0]), ("B", [10.0, 11.0]), ("C", [20.0])),
    )

    engine = get_transform_engine("netcdf")
    machine_map = SimpleNamespace(name="OP1")
    with engine.open(
        str(tmp_path), 100_001, dd_version, machine_map, "static-over-map"
    ) as source:
        selected = _coil_element_binding("C")
        assert source.read(selected).shape == (1,)
        assert source.read(replace(selected, source_rank=0)).shape == ()
        with pytest.raises(BindingTransformError) as refused:
            source.read(replace(selected, source_rank=2))
        message = str(refused.value)
        assert "jt60sa-coil-element-r" in message
        assert "2" in message
    print("SELECTOR_RANK consumed=coil rank0=() rank1=(1,) rank2=refused")


def test_selecting_binding_over_an_empty_struct_array_is_refused(tmp_path):
    dd_version = "4.1.1"
    store_directory = tmp_path / "OP1"
    store_directory.mkdir()
    _write_ragged_coil_store(store_directory / "pf_active.nc", dd_version, ())

    engine = get_transform_engine("netcdf")
    machine_map = SimpleNamespace(name="OP1")
    with (
        engine.open(
            str(tmp_path), 100_001, dd_version, machine_map, "static-over-map"
        ) as source,
        pytest.raises(BindingTransformError) as refused,
    ):
        source.read(_coil_element_binding("A"))
    message = str(refused.value)
    assert "jt60sa-coil-element-r" in message
    print("SELECTOR_EMPTY refused=binding-transform-error not=keyerror")


def test_zarr_engine_refuses_a_binding_selecting_a_struct_array_entry(tmp_path):
    dd_version = "4.1.1"
    group = zarr.open_group(tmp_path / "5.zarr", mode="w")
    group.create_group("pf_active").create_array("rectangle_r", data=np.zeros(3))

    engine = get_transform_engine("zarr")
    with (
        engine.open(str(tmp_path), 5, dd_version) as source,
        pytest.raises(TransformEngineError, match="struct-array entry"),
    ):
        source.read(_coil_element_binding("A"))
    print("STRUCT_ENTRY_ZARR refused=declared slot")


@pytest.mark.skipif(
    not JT60SA_OP1_PF_ACTIVE.is_file(),
    reason="JT-60SA OP1 machine description is not mounted",
)
def test_static_netcdf_store_selects_each_real_coil_by_name():
    with imas.DBEntry(JT60SA_OP1_PF_ACTIVE, "r") as entry:
        dd_version = entry.dd_version
        ids = entry.get("pf_active", autoconvert=False)
        names = [str(coil.name) for coil in ids.coil]
    assert len(names) == 12

    engine = get_transform_engine("netcdf")
    machine_map = SimpleNamespace(name="OP1")
    lengths: list[int] = []
    with engine.open(
        str(JT60SA_OP1_DESCRIPTION_ROOT),
        100_001,
        dd_version,
        machine_map,
        "static-over-map",
    ) as source:
        for name in names:
            lengths.append(len(source.read(_coil_element_binding(name))))

    assert lengths == [40, 40, 40, 40, 16, 16, 16, 16, 16, 16, 6, 6]
    print(f"REAL_STORE coils={len(names)} lengths={lengths}")


def test_engine_registry_is_format_scoped_and_has_no_machine_conditionals():
    import imas_alambic.transform_engine as engine_module

    source = Path(engine_module.__file__).read_text().lower()
    assert TRANSFORM_ENGINE_FORMATS == ("netcdf", "zarr")
    assert len(TRANSFORM_ENGINE_FORMATS) == 2
    assert "machine_map.machine" not in source
    assert "catalog.source" not in source


@needs_bundle_discovery
def test_every_bound_cocos_target_receives_its_target_path_factor():
    import imas_alambic.transform_engine as engine_module

    class ConstantArrays:
        def read(self, binding: ChannelBinding) -> np.ndarray:
            return np.ones((1,), dtype=np.float64)

    catalog = load_packaged_machine_map("mast")
    bindings = tuple(
        binding
        for binding_set in catalog.binding_sets.values()
        for binding in binding_set
    )
    emitted, missing = engine_module._emit_arrays(
        ConstantArrays(),
        bindings,
        catalog.dd_version,
        MAST_SOURCE_COCOS,
    )
    transformed = tuple(array for array in emitted if array.cocos_transformation)
    non_unity = tuple(array for array in transformed if array.cocos_factor != 1.0)

    assert missing == ()
    assert len(transformed) == 11
    assert non_unity == ()
    ip_like = tuple(
        array for array in transformed if array.cocos_transformation == "ip_like"
    )
    assert {array.binding_name for array in ip_like} == {
        "mast-magnetics-ip",
        "mast-pf-active-coil-current",
        "mast-pf-active-solenoid-current",
    }
    assert all(array.cocos_factor == 1.0 for array in ip_like)
    print(
        f"COCOS_BOUND_TARGETS dependent={len(transformed)} "
        f"ip_like={len(ip_like)} factor_before=-1 factor_after=+1 "
        f"non_unity={len(non_unity)}"
    )


@needs_bundle_discovery
def test_cocos_dependent_binding_rejects_an_undeclared_source_convention(tmp_path):
    catalog = _catalog_with_only_plasma_current()
    row = MAST_LEVEL2_SIGN_TABLE[0]
    values = np.asarray([row.plasma_current_a], dtype=np.float64)
    _write_plasma_current_store(tmp_path, row.shot, values)

    with pytest.raises(BindingTransformError, match="no declared source COCOS"):
        transform_machine_description(
            catalog,
            row.shot,
            "zarr",
            tmp_path,
            source_cocos=None,
        )


@needs_bundle_discovery
def test_both_polarities_round_trip_exactly_through_engine_cocos_transform(tmp_path):
    catalog = _catalog_with_only_plasma_current()
    current_signs: list[int] = []
    inverse_factor = 1.0 / MAST_TO_COCOS_17_FACTORS["ip_like"]

    for row in MAST_LEVEL2_SIGN_TABLE:
        source_values = np.asarray(
            [row.plasma_current_a, row.plasma_current_a / 2.0],
            dtype=np.float64,
        )
        _write_plasma_current_store(tmp_path, row.shot, source_values)
        result = transform_machine_description(
            catalog,
            row.shot,
            "zarr",
            tmp_path,
            source_cocos=MAST_SOURCE_COCOS,
        )

        assert result.emitted_array_count == 1
        emitted = result.arrays[0]
        assert emitted.cocos_transformation == "ip_like"
        assert emitted.cocos_factor == 1.0
        restored = np.multiply(emitted.values, inverse_factor)
        assert np.array_equal(restored, source_values)
        current_signs.append(row.plasma_current_sign)

    assert current_signs.count(-1) == 2
    assert current_signs.count(+1) == 2


def test_poloidal_angle_applies_no_cocos_factor_and_takes_sense_from_binding():
    import imas_alambic.transform_engine as engine_module

    values = np.asarray([0.0, 1.5707963267948966, -0.5], dtype=np.float64)

    for sign_convention, expected_factor in (("identity", 1.0), ("negate", -1.0)):
        binding = _probe_angle_binding(sign_convention)
        emitted, transformation, factor = engine_module._apply_cocos_convention(
            values, binding, "4.1.1", None
        )

        assert transformation == "pol_angle_like"
        assert factor == expected_factor
        assert np.array_equal(emitted, values * expected_factor)


def test_poloidal_angle_refuses_an_unvalidated_sign_convention():
    import imas_alambic.transform_engine as engine_module

    binding = _probe_angle_binding("unknown-unvalidated")

    with pytest.raises(BindingTransformError, match="unknown-unvalidated"):
        engine_module._apply_cocos_convention(
            np.asarray([0.0], dtype=np.float64), binding, "4.1.1", None
        )


def test_poloidal_angle_emit_path_negates_exactly_once():
    import imas_alambic.transform_engine as engine_module

    class ConstantArrays:
        def read(self, binding: ChannelBinding) -> np.ndarray:
            return np.asarray([1.0, -2.0], dtype=np.float64)

    binding = _probe_angle_binding("negate")
    emitted, missing = engine_module._emit_arrays(
        ConstantArrays(), (binding,), "4.1.1", MAST_SOURCE_COCOS
    )

    assert missing == ()
    assert emitted[0].cocos_transformation == "pol_angle_like"
    assert emitted[0].cocos_factor == -1.0
    assert np.array_equal(emitted[0].values, np.asarray([-1.0, 2.0]))
