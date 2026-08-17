"""Machine maps are closed-world data declarations, not reader code."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import imas
import numpy as np
import pytest
import zarr
from imas.ids_struct_array import IDSStructArray

from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.geometry_transitions import (
    build_geometry_transitions,
    load_geometry_table_payload,
)
from imas_ambix.data.machine_map import (
    LINKML_SCHEMA_PATH,
    MachineMapError,
    assert_transition_alignment,
    load_linkml_schema,
    load_machine_map,
    load_packaged_machine_map,
    map_for_shot,
)
from imas_ambix.data.manifest import load_index
from imas_ambix.data.transform_engine import (
    BindingTransformError,
    transform_machine_description,
)
from imas_ambix.gs.operator import classify_circuits

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
MACHINE_DESCRIPTION_GROUPS = ("magnetics", "pf_active", "pf_passive", "wall")
EXPECTED_BOUND_CHANNEL_COUNTS = {
    "magnetics": 58,
    "pf_active": 71,
    "pf_passive": 110,
    "wall": 3,
}
POSITIONAL_PHI_TARGETS = {
    "b_field_pol_probe_cc_phi": "magnetics/b_field_pol_probe/position/phi",
    "b_field_pol_probe_ccbv_phi_1": "magnetics/b_field_pol_probe/position/phi",
    "b_field_pol_probe_ccbv_phi_2": "magnetics/b_field_pol_probe/position/phi",
    "b_field_pol_probe_obr_phi_1": "magnetics/b_field_pol_probe/position/phi",
    "b_field_pol_probe_obr_phi_2": "magnetics/b_field_pol_probe/position/phi",
    "b_field_pol_probe_obv_phi_1": "magnetics/b_field_pol_probe/position/phi",
    "b_field_pol_probe_obv_phi_2": "magnetics/b_field_pol_probe/position/phi",
    "b_field_pol_probe_omv_phi": "magnetics/b_field_pol_probe/position/phi",
    "b_field_tor_probe_cc_phi": "magnetics/b_field_phi_probe/position/phi",
    "b_field_tor_probe_saddle_l_phi": "magnetics/flux_loop/position/phi",
    "b_field_tor_probe_saddle_m_phi": "magnetics/flux_loop/position/phi",
    "b_field_tor_probe_saddle_u_phi": "magnetics/flux_loop/position/phi",
}
COLOCATED_FULL_LOOP_ADDRESSES = (
    "fl_cc01",
    "fl_cc02",
    "fl_cc03",
    "fl_cc04",
    "fl_cc05",
    "fl_cc07",
    "fl_cc09",
    "fl_cc10",
)
SENSOR_DESCRIPTION_POSITION = re.compile(
    r"r\s*=\s*([-+0-9.]+)\s*,?\s*z\s*=\s*([-+0-9.]+)", re.IGNORECASE
)
LEVEL2_SENSOR_GEOMETRY_FAMILIES = (
    "b_field_pol_probe_ccbv",
    "b_field_pol_probe_obr",
    "b_field_pol_probe_obv",
    "flux_loop",
)
PASSIVE_LOOP_FAMILIES = (
    "botcol",
    "coil_cases",
    "endcrown_l",
    "endcrown_u",
    "incon",
    "lhorw",
    "mid",
    "p2larm",
    "p2ldivpl",
    "p2uarm",
    "p2udivpl",
    "ring",
    "rodgr",
    "topcol",
    "uhorw",
    "vertw",
)
CASE_CURRENT_JOINS = {
    "fcoil-circuit-014": "p2u_case_current",
    "fcoil-circuit-015": "p2l_case_current",
    "fcoil-circuit-016": "p3u_case_current",
    "fcoil-circuit-017": "p3l_case_current",
    "fcoil-circuit-018": "p4u_case_current",
    "fcoil-circuit-019": "p4l_case_current",
    "fcoil-circuit-020": "p5u_case_current",
    "fcoil-circuit-021": "p5l_case_current",
}
LEGACY_DESCRIPTION_FIELDS = (
    "magpr_r",
    "magpr_z",
    "magpr_ang",
    "magpr_len.cc",
    "magpr_len.ccbv",
    "magpr_len.obr",
    "magpr_len.obv",
    "magpr_len.omv",
    "silop_r",
    "silop_z",
    "silop_additional_positions",
    "pf_active.element.r",
    "pf_active.element.z",
    "pf_active.element.width",
    "pf_active.element.height",
    "pf_passive.loop.name",
    "pf_passive.element.name",
    "pf_passive.element.r",
    "pf_passive.element.z",
    "pf_passive.element.width",
    "pf_passive.element.height",
    "pf_active.element.turns_with_sign",
    "pf_passive.element.turns_with_sign",
    "fcoil_circ",
    "fcoil_xmult",
    "amc_current_channels",
    "unmatched_amb",
    "conductor_element_join",
    "r0",
    "minor_radius",
    "polygon_sections",
    "limiterr",
    "limiterz",
)


def _level1_sensor_positions(group: Any) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for name in group.array_keys():
        if not name.startswith(("ccbv", "obr", "obv", "fl_")):
            continue
        description = str(group[name].attrs.get("description", ""))
        match = SENSOR_DESCRIPTION_POSITION.search(description)
        if match is not None:
            positions[name.casefold()] = (float(match.group(1)), float(match.group(2)))
    return positions


def _level2_sensor_positions(group: Any) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for family in LEVEL2_SENSOR_GEOMETRY_FAMILIES:
        names = tuple(
            str(value)
            for value in np.asarray(group[f"{family}_geometry_channel"][:]).reshape(-1)
        )
        radial = np.asarray(group[f"{family}_r"][:], dtype=np.float64).reshape(-1)
        vertical = np.asarray(group[f"{family}_z"][:], dtype=np.float64).reshape(-1)
        assert len(names) == radial.size == vertical.size
        positions.update(zip(names, zip(radial, vertical, strict=True), strict=True))
    return positions


def _dd_leaf(dd_version: str, path: str):
    ids_name, leaf = path.split("/", maxsplit=1)
    return imas.IDSFactory(dd_version).new(ids_name).metadata[leaf]


def _dd_struct_array_count(dd_version: str, path: str) -> int:
    ids_name, relative_path = path.split("/", maxsplit=1)
    node = imas.IDSFactory(dd_version).new(ids_name)
    count = 0
    for component in relative_path.split("/")[:-1]:
        node = getattr(node, component)
        if isinstance(node, IDSStructArray):
            count += 1
            node.resize(1)
            node = node[0]
    return count


def _source_data_kind(data_type: object) -> str:
    declaration = json.dumps(data_type, sort_keys=True).lower()
    if "string" in declaration or "utf" in declaration:
        return "STR"
    if "float" in declaration:
        return "FLT"
    if "int" in declaration:
        return "INT"
    raise AssertionError(f"unsupported source data type {data_type!r}")


def _plasma_current_catalog_document(source_cocos: int) -> dict[str, Any]:
    document = json.loads((LINKML_SCHEMA_PATH.parent / "mast.json").read_text())
    binding = next(
        item
        for item in document["binding_sets"][0]["bindings"]
        if item["name"] == "mast-magnetics-ip"
    )
    document["source_cocos"] = source_cocos
    document["binding_sets"] = [{"name": "plasma-current-only", "bindings": [binding]}]
    machine_map = document["maps"][0]
    machine_map.update(
        {
            "name": "plasma-current-only",
            "first_shot": 1,
            "last_shot": 1,
            "transition": None,
            "binding_set": "plasma-current-only",
            "drive_topology": None,
        }
    )
    document["maps"] = [machine_map]
    document["validation_gaps"] = []
    document["source_qualifications"] = []
    document["sensor_identity_rules"] = []
    document["identity_qualifications"] = []
    document["flux_loop_position_declarations"] = []
    document["circuit_current_joins"] = []
    document["drive_topologies"] = []
    document["structure_assemblies"] = []
    document["acquisition_declarations"] = []
    document["description_supplements"] = []
    machine_map["description_supplement"] = None
    return document


@lru_cache(maxsize=1)
def _machine_description_inventory() -> tuple[
    list[int], dict[str, dict[str, dict[str, set[object]]]]
]:
    shots: list[int] = []
    inventory: dict[str, dict[str, dict[str, set[object]]]] = {
        group: {} for group in MACHINE_DESCRIPTION_GROUPS
    }
    for entry in os.scandir(LEVEL2_ROOT):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        shot = int(entry.name.removesuffix(".zarr"))
        metadata_path = Path(entry.path) / "zarr.json"
        metadata = json.loads(metadata_path.read_text())["consolidated_metadata"][
            "metadata"
        ]
        shots.append(shot)
        for group in MACHINE_DESCRIPTION_GROUPS:
            prefix = f"{group}/"
            for path, row in metadata.items():
                if not path.startswith(prefix) or row.get("node_type") != "array":
                    continue
                name = path.removeprefix(prefix)
                observation = inventory[group].setdefault(
                    name,
                    {"units": set(), "ranks": set(), "data_kinds": set()},
                )
                unit = str(row.get("attributes", {}).get("units", ""))
                observation["units"].add(unit)
                observation["ranks"].add(len(row["shape"]))
                observation["data_kinds"].add(_source_data_kind(row["data_type"]))
    return sorted(shots), inventory


def test_linkml_schema_is_declarative_and_has_no_executable_language():
    schema = load_linkml_schema()
    assert LINKML_SCHEMA_PATH.name == "schema.yaml"
    assert schema["classes"]["MachineMapCatalog"]["tree_root"] is True
    assert "SourceQualification" in schema["classes"]
    assert "SensorIdentityRule" in schema["classes"]
    assert "IdentityQualification" in schema["classes"]
    assert "FluxLoopPositionDeclaration" in schema["classes"]
    assert "CircuitConnection" in schema["classes"]
    assert "CircuitCurrentJoin" in schema["classes"]
    assert "AcquisitionDeclaration" in schema["classes"]
    assert "DescriptionSupplement" in schema["classes"]
    assert "DriveTopology" in schema["classes"]
    assert "PointFluxLoopDeclaration" in schema["classes"]
    assert "PolygonSectionDeclaration" in schema["classes"]
    assert "StructureAssembly" in schema["classes"]
    assert "source_cocos" in schema["classes"]["MachineMapCatalog"]["slots"]
    assert "circuit_current_joins" in schema["classes"]["MachineMapCatalog"]["slots"]
    assert "source_cocos_override" in schema["classes"]["ChannelBinding"]["slots"]
    assert "sensor_identity_key" in schema["classes"]["AcquisitionDeclaration"]["slots"]
    assert (
        "sensor_identity_rule" in schema["classes"]["AcquisitionDeclaration"]["slots"]
    )
    assert "sensor_identity_rules" in schema["classes"]["MachineMapCatalog"]["slots"]
    assert "identity_qualifications" in schema["classes"]["MachineMapCatalog"]["slots"]
    assert (
        "flux_loop_position_declarations"
        in schema["classes"]["MachineMapCatalog"]["slots"]
    )
    assert set(schema["enums"]["BindingRole"]["permissible_values"]) == {
        "value",
        "identifier",
        "dimension-coordinate",
    }
    assert set(schema["enums"]["SourceStatus"]["permissible_values"]) == {
        "corpus-observed",
        "legacy-only",
        "range-absent",
    }
    assert set(schema["enums"]["IdentityCaseRule"]["permissible_values"]) == {
        "case-fold"
    }
    assert set(schema["enums"]["IdentityNumericTokenRule"]["permissible_values"]) == {
        "integer-value"
    }
    assert set(schema["enums"]["FluxLoopPositionVerdict"]["permissible_values"]) == {
        "nominal-table",
        "reconstruction",
        "undecided",
    }
    forbidden = {"condition", "expression", "code_hook", "transform_code"}
    assert forbidden.isdisjoint(schema["slots"])


def test_flux_loop_position_verdicts_are_range_scoped_and_evidence_carrying():
    catalog = load_packaged_machine_map("mast")
    declarations = catalog.flux_loop_position_declarations
    counts = {
        verdict: sum(item.position_verdict == verdict for item in declarations)
        for verdict in ("reconstruction", "nominal-table", "undecided")
    }

    assert len(declarations) == 19
    assert counts == {
        "reconstruction": 14,
        "nominal-table": 2,
        "undecided": 3,
    }
    assert len(catalog.flux_loop_positions_for(12_000)) == 5
    assert len(catalog.flux_loop_positions_for(21_978)) == 14
    assert {
        (item.range_first_shot, item.range_last_shot)
        for item in declarations
    } == {(11_766, 12_416), (12_417, 30_471)}
    assert all(
        "docs/notes/vacuum-loop-adjudication.md" in item.evidence
        for item in declarations
    )

    undecided = {
        (item.range_first_shot, item.acquisition_address)
        for item in declarations
        if item.position_verdict == "undecided"
    }
    assert undecided == {
        (11_766, "fl_p2u_2"),
        (12_417, "fl_p2l_1"),
        (12_417, "fl_p2l_2"),
    }
    assert all(
        item.declared_r is None and item.declared_z is None
        for item in declarations
        if item.position_verdict == "undecided"
    )
    assert all(
        np.isfinite((item.declared_r, item.declared_z)).all()
        for item in declarations
        if item.position_verdict != "undecided"
    )

    late = {
        item.acquisition_address: item
        for item in catalog.flux_loop_positions_for(21_978)
    }
    assert (late["fl_p3l_1"].declared_r, late["fl_p3l_1"].declared_z) == (
        1.163,
        -1.08259,
    )
    assert late["fl_p3l_1"].position_verdict == "nominal-table"
    assert (late["fl_p4l_1"].declared_r, late["fl_p4l_1"].declared_z) == (
        1.5984,
        -1.04443,
    )
    assert late["fl_p4l_1"].position_verdict == "reconstruction"


def test_late_range_declares_the_measurable_acquisition_population():
    catalog = load_packaged_machine_map("mast")
    machine_map = map_for_shot(catalog, 21_978)
    supplement = next(
        item
        for item in catalog.description_supplements
        if item.name == machine_map.description_supplement
    )
    acquisition = next(
        item
        for item in catalog.acquisition_declarations
        if item.name == supplement.acquisition_declaration
    )
    topology = next(
        item
        for item in catalog.drive_topologies
        if item.name == machine_map.drive_topology
    )
    absence = next(
        item
        for item in catalog.source_qualifications
        if item.name == "mast-amb-fl-cc02-after-16605"
    )

    assert (machine_map.first_shot, machine_map.last_shot) == (21_978, 22_086)
    assert topology.current_channel_declaration == acquisition.name
    assert len(topology.connections) == 938
    assert machine_map.source_representation_signature == (
        "mp78-fl46-fc938-lim37-532938247d31ec5c"
    )
    assert map_for_shot(catalog, 21_977).source_representation_signature is None
    assert map_for_shot(catalog, 22_087).source_representation_signature is None
    assert len(acquisition.current_channels) == 45
    assert len(acquisition.sensor_addresses) == 100
    assert len(acquisition.unmatched_sensor_addresses) == 4
    measured = set(acquisition.sensor_addresses).difference(
        acquisition.unmatched_sensor_addresses
    )
    assert len(measured) == 96
    assert {"ccbv10", "fl_p6u_1"}.issubset(measured)
    assert {"fl_cc02", "fl_cc10"}.isdisjoint(acquisition.sensor_addresses)
    assert absence.source_array == "fl_cc02"
    assert absence.source_status == "range-absent"
    assert (absence.range_first_shot, absence.range_last_shot) == (16_606, 30_471)
    assert "last amb/fl_cc02 source at shot 16605" in absence.evidence
    assert "must not be selected" in absence.reason


def test_machine_catalogs_declare_source_cocos_without_binding_overrides():
    mast = load_packaged_machine_map("mast")
    diii_d = load_packaged_machine_map("diii-d")

    assert mast.source_cocos == 3
    assert diii_d.source_cocos is None
    for catalog in (mast, diii_d):
        overrides = tuple(
            binding
            for bindings in catalog.binding_sets.values()
            for binding in bindings
            if binding.source_cocos_override is not None
        )
        assert overrides == ()


def test_engine_uses_catalog_source_cocos_and_binding_override(tmp_path):
    values = np.asarray([2.0, -3.0], dtype=np.float64)
    store = zarr.open_group(tmp_path / "stores" / "1.zarr", mode="w")
    store.create_group("magnetics").create_array("ip", data=values)

    receipts = []
    for source_cocos, expected_factor in ((1, 1.0), (4, -1.0)):
        document = _plasma_current_catalog_document(source_cocos)
        path = tmp_path / f"source-cocos-{source_cocos}.json"
        path.write_text(json.dumps(document))
        catalog = load_machine_map(path)
        result = transform_machine_description(catalog, 1, "zarr", tmp_path / "stores")

        assert result.source_cocos == source_cocos
        assert result.arrays[0].cocos_factor == expected_factor
        assert np.array_equal(result.arrays[0].values, values * expected_factor)
        receipts.append((source_cocos, expected_factor))

    override_document = _plasma_current_catalog_document(4)
    override_document["binding_sets"][0]["bindings"][0]["source_cocos_override"] = 1
    override_path = tmp_path / "binding-override.json"
    override_path.write_text(json.dumps(override_document))
    override_result = transform_machine_description(
        load_machine_map(override_path), 1, "zarr", tmp_path / "stores"
    )
    assert override_result.arrays[0].cocos_factor == 1.0
    assert np.array_equal(override_result.arrays[0].values, values)
    print(f"SOURCE_COCOS_FACTOR_RECEIPT declarations={receipts} override=(1, 1.0)")


def test_engine_rejects_undeclared_catalog_cocos_for_dependent_binding(tmp_path):
    document = _plasma_current_catalog_document(0)
    path = tmp_path / "undeclared-cocos.json"
    path.write_text(json.dumps(document))
    store = zarr.open_group(tmp_path / "stores" / "1.zarr", mode="w")
    store.create_group("magnetics").create_array(
        "ip", data=np.asarray([1.0], dtype=np.float64)
    )

    with pytest.raises(BindingTransformError, match="no declared source COCOS"):
        transform_machine_description(
            load_machine_map(path), 1, "zarr", tmp_path / "stores"
        )


def test_mast_catalog_accounts_for_every_machine_description_array_in_the_corpus():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    shots, inventory = _machine_description_inventory()
    bindings = catalog.binding_sets["mast-machine-description"]

    assert len(shots) == 11_573
    assert (shots[0], shots[-1]) == (11_766, 30_471)
    assert catalog.bound_channel_counts == EXPECTED_BOUND_CHANNEL_COUNTS
    assert catalog.bound_channel_count == 242
    assert catalog.qualified_channel_counts == {
        "amb": 1,
        "magnetics": 16,
        "machine_description": 1,
        "pf_passive": 1,
    }
    assert catalog.qualified_channel_count == 19
    assert sum(len(catalog.bindings_for(item)) for item in catalog.maps) == (
        len(catalog.maps) * catalog.bound_channel_count
    )
    assert catalog.validation_gaps == ()
    assert (
        sum(
            item.source_status == "corpus-observed"
            for item in catalog.source_qualifications
        )
        == 13
    )
    assert (
        sum(
            item.source_status == "legacy-only"
            for item in catalog.source_qualifications
        )
        == 5
    )
    assert (
        sum(
            item.source_status == "range-absent"
            for item in catalog.source_qualifications
        )
        == 1
    )

    bindings_by_group: dict[str, dict[str, object]] = {
        group: {
            binding.source_array: binding
            for binding in bindings
            if binding.source_group == group
        }
        for group in MACHINE_DESCRIPTION_GROUPS
    }
    qualifications_by_group: dict[str, dict[str, object]] = {
        group: {
            item.source_array: item
            for item in catalog.source_qualifications
            if item.source_group == group and item.source_status == "corpus-observed"
        }
        for group in MACHINE_DESCRIPTION_GROUPS
    }
    for group in MACHINE_DESCRIPTION_GROUPS:
        bound_sources = set(bindings_by_group[group])
        qualified_sources = set(qualifications_by_group[group])
        assert bound_sources.isdisjoint(qualified_sources)
        assert bound_sources | qualified_sources == set(inventory[group])

    for binding in bindings:
        leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
        assert (leaf.units or "1") == binding.target_unit
        observed_units = inventory[binding.source_group][binding.source_array]["units"]
        nonempty_units = {unit for unit in observed_units if unit}
        if binding.source_role == "value":
            assert binding.sign_convention == "identity"
            expected_source_unit = (
                "degree" if binding.target_unit == "rad" else binding.target_unit
            )
            assert binding.source_unit == expected_source_unit
            if binding.source_unit in {"degree", "m"}:
                assert nonempty_units == {"SI, degrees, m"}
            elif binding.source_unit == "s":
                assert {unit.lower() for unit in nonempty_units} == {"s"}
            else:
                assert nonempty_units == {binding.source_unit}
        else:
            assert binding.source_unit == binding.target_unit == "1"
            assert binding.sign_convention == "not-applicable"
            assert not nonempty_units


def test_every_binding_targets_a_writable_shape_compatible_dd_leaf():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    _, inventory = _machine_description_inventory()

    for machine in ("mast", "diii-d"):
        catalog = load_packaged_machine_map(machine)
        for bindings in catalog.binding_sets.values():
            for binding in bindings:
                leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
                assert leaf.data_type.name not in {"STRUCTURE", "STRUCT_ARRAY"}
                if machine != "mast":
                    assert leaf.data_type.name == "FLT"
                    continue

                observation = inventory[binding.source_group][binding.source_array]
                assert observation["data_kinds"] == {leaf.data_type.name}
                assert observation["ranks"] == {binding.source_rank}
                for source_rank in observation["ranks"]:
                    expanded_dimensions = source_rank - leaf.ndim
                    assert expanded_dimensions >= 0, binding.name
                    if expanded_dimensions:
                        assert expanded_dimensions <= _dd_struct_array_count(
                            catalog.dd_version, binding.dd_path
                        ), binding.name


def test_saddle_trajectories_are_twelve_named_loops_with_twenty_eight_points():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    bindings = {
        item.source_array: item
        for item in catalog.binding_sets["mast-machine-description"]
    }
    qualified_sources = {item.source_array for item in catalog.source_qualifications}
    metadata = json.loads((LEVEL2_ROOT / "12417.zarr" / "zarr.json").read_text())[
        "consolidated_metadata"
    ]["metadata"]

    executable_sources = {
        f"b_field_tor_probe_saddle_{band}_{axis}"
        for band in "lmu"
        for axis in ("r", "z")
    }
    qualified_phi_sources = {f"b_field_tor_probe_saddle_{band}_phi" for band in "lmu"}
    assert executable_sources.isdisjoint(qualified_sources)
    assert qualified_phi_sources.issubset(qualified_sources)
    for source in executable_sources:
        binding = bindings[source]
        axis = source.rsplit("_", maxsplit=1)[1]
        assert binding.source_rank == 2
        assert tuple(metadata[f"magnetics/{source}"]["shape"]) == (12, 28)
        assert binding.dd_path == f"magnetics/flux_loop/position/{axis}"
        assert "axis 0 matches the 12" in binding.evidence
        assert "axis 1 matches coordinate 0..27" in binding.evidence

    qualifications = {item.source_array: item for item in catalog.source_qualifications}
    for source in qualified_phi_sources:
        qualification = qualifications[source]
        assert qualification.source_shape == (12, 28)
        assert "magnetics/flux_loop/position/phi" in qualification.reason
        assert "silop_dphi" in qualification.reason
        assert "no saddle trajectory" in qualification.evidence

    saddle_assemblies = {
        item.name: item
        for item in catalog.structure_assemblies
        if item.name.startswith("mast-magnetics-saddle-")
    }
    assert len(saddle_assemblies) == 3
    for band in "lmu":
        assembly = saddle_assemblies[f"mast-magnetics-saddle-{band}-flux-loops"]
        assert assembly.structure_path == "magnetics/flux_loop/position"
        assert assembly.type_path == "magnetics/flux_loop/type/index"
        assert assembly.type_index == 2
        name_binding = bindings[f"b_field_tor_probe_saddle_{band}_geometry_channel"]
        assert assembly.name_binding == name_binding.name
        assert name_binding.dd_path == "magnetics/flux_loop/name"
        assert len(assembly.member_bindings) == 2

    coordinate = next(
        item
        for item in catalog.source_qualifications
        if item.source_array == "coordinate"
    )
    assert coordinate.source_shape == (28,)
    assert "no writable coordinate-index leaf" in coordinate.reason


@pytest.mark.skipif(
    not all(
        (LEVEL1_ROOT / f"{shot}.zarr").is_dir()
        and (LEVEL2_ROOT / f"{shot}.zarr").is_dir()
        for shot in (11_766, 12_417, 21_978)
    ),
    reason="local level-1 and level-2 magnetics stores are not mounted",
)
def test_position_phi_requires_an_authoritative_level1_source():
    catalog = load_packaged_machine_map("mast")
    bindings = {
        item.source_array: item
        for item in catalog.binding_sets["mast-machine-description"]
    }
    qualifications = {item.source_array: item for item in catalog.source_qualifications}

    position_leaf = _dd_leaf(
        catalog.dd_version, "magnetics/b_field_pol_probe/position/phi"
    )
    orientation_leaf = _dd_leaf(
        catalog.dd_version, "magnetics/b_field_pol_probe/toroidal_angle"
    )
    assert (position_leaf.data_type.name, position_leaf.ndim, position_leaf.units) == (
        "FLT",
        0,
        "rad",
    )
    assert "Toroidal angle" in position_leaf.documentation
    assert "sensor normal vector" in orientation_leaf.documentation
    assert all(
        item.dd_path != "magnetics/b_field_pol_probe/toroidal_angle"
        for item in bindings.values()
    )

    for shot in (11_766, 12_417, 21_978):
        level1_path = LEVEL1_ROOT / f"{shot}.zarr"
        level2_path = LEVEL2_ROOT / f"{shot}.zarr"
        assert (level1_path / ".zmetadata").is_file()
        assert (level2_path / "zarr.json").is_file()
        level1 = zarr.open_consolidated(level1_path, mode="r")
        level2 = zarr.open_group(level2_path, mode="r")
        efm_names = set(level1["efm"].array_keys())
        assert {"magpr_r", "magpr_z", "magpr_ang", "magpr_len"}.issubset(efm_names)
        assert {"magpr_phi", "silop_phi"}.isdisjoint(efm_names)
        if "silop_dphi" in efm_names:
            extent = np.asarray(level1["efm"]["silop_dphi"][:], dtype=np.float64)
            finite_extent = extent[np.isfinite(extent)]
            assert finite_extent.size == 46
            assert np.allclose(finite_extent, 2.0 * np.pi, rtol=0.0, atol=2.0e-7)
        level2_metadata = level2["magnetics"]
        for source_array, target_path in POSITIONAL_PHI_TARGETS.items():
            assert source_array in level2_metadata
            assert source_array not in bindings
            qualification = qualifications[source_array]
            assert qualification.source_status == "corpus-observed"
            assert qualification.source_shape == level2_metadata[source_array].shape
            assert target_path in qualification.reason
            assert "level-1" in qualification.reason.casefold() or (
                "level 1" in qualification.reason.casefold()
            )
        print(
            "LEVEL1_POSITION_PHI_PUBLICATION "
            f"shot={shot} probe_position_phi=absent "
            "point_loop_position_phi=absent saddle_position_phi=absent "
            f"point_loop_extent={'present' if 'silop_dphi' in efm_names else 'absent'} "
            f"qualified={len(POSITIONAL_PHI_TARGETS)}"
        )


@pytest.mark.skipif(
    not all((LEVEL1_ROOT / f"{shot}.zarr").is_dir() for shot in (11_766, 12_417)),
    reason="local level-1 magnetics stores are not mounted",
)
def test_full_toroidal_flux_loops_use_acquisition_address_identity():
    catalog = load_packaged_machine_map("mast")
    acquisition = next(
        item for item in catalog.acquisition_declarations if item.name.endswith("12417")
    )
    point_assembly = next(
        item
        for item in catalog.structure_assemblies
        if item.name == "mast-magnetics-poloidal-flux-loops"
    )

    assert acquisition.sensor_identity_key == "acquisition-address"
    assert point_assembly.type_path == "magnetics/flux_loop/type/index"
    assert point_assembly.type_index == 1
    assert point_assembly.member_bindings == (
        "mast-magnetics-flux-loop-r",
        "mast-magnetics-flux-loop-z",
    )
    assert all(not name.endswith("-phi") for name in point_assembly.member_bindings)
    assert "no discrete position/phi" in point_assembly.evidence
    for supplement in catalog.description_supplements:
        for loop in supplement.point_flux_loops:
            assert loop.type_path == point_assembly.type_path
            assert loop.type_index == point_assembly.type_index

    collocated = tuple(
        address
        for address in acquisition.sensor_addresses
        if address in COLOCATED_FULL_LOOP_ADDRESSES
    )
    assert collocated == COLOCATED_FULL_LOOP_ADDRESSES
    assert len(set(collocated)) == len(collocated) == 8
    level1 = zarr.open_consolidated(LEVEL1_ROOT / "12417.zarr", mode="r")
    level1_positions = _level1_sensor_positions(level1["amb"])
    assert {level1_positions[address] for address in collocated} == {(0.18, 1.215)}

    for shot in (12_417, 21_978):
        efm = zarr.open_consolidated(LEVEL1_ROOT / f"{shot}.zarr", mode="r")["efm"]
        extent = np.asarray(efm["silop_dphi"][:], dtype=np.float64)
        finite_extent = extent[np.isfinite(extent)]
        assert finite_extent.size == 46
        assert np.allclose(finite_extent, 2.0 * np.pi, rtol=0.0, atol=2.0e-7)
        print(
            "FULL_TOROIDAL_FLUX_LOOP_EXTENT "
            f"shot={shot} published={extent.size} finite={finite_extent.size} "
            f"minimum_rad={finite_extent.min():.15g} "
            f"maximum_rad={finite_extent.max():.15g}"
        )
    early_efm = zarr.open_consolidated(LEVEL1_ROOT / "11766.zarr", mode="r")["efm"]
    assert "silop_dphi" not in set(early_efm.array_keys())
    print("FULL_TOROIDAL_FLUX_LOOP_EXTENT shot=11766 publication=absent")
    for address in collocated:
        print(
            "COLOCATED_FLUX_LOOP_IDENTITY "
            f"address={address} r=0.180 z=1.215 "
            "position_phi=not-applicable toroidal_extent_rad=2*pi "
            f"identity_key={acquisition.sensor_identity_key}"
        )


@pytest.mark.skipif(
    not (LEVEL1_ROOT / "12417.zarr").is_dir()
    or not (LEVEL2_ROOT / "12417.zarr").is_dir(),
    reason="local level-1 and level-2 magnetics stores are not mounted",
)
def test_declared_identity_rule_closes_the_malformed_flux_loop_spelling():
    catalog = load_packaged_machine_map("mast")
    acquisition = next(
        item for item in catalog.acquisition_declarations if item.name.endswith("12417")
    )
    qualification = next(
        item
        for item in catalog.identity_qualifications
        if item.name == "mast-level2-fl-cc010-spelling"
    )

    assert acquisition.sensor_identity_rule == "mast-acquisition-address"
    assert qualification.malformed_identity == "FL_CC010"
    assert qualification.canonical_identity == "fl_cc10"
    assert qualification.sensor_identity_rule == acquisition.sensor_identity_rule

    def normalise(identity: str) -> str:
        return catalog.normalise_sensor_identity(acquisition, identity)

    assert normalise(qualification.malformed_identity) == normalise(
        qualification.canonical_identity
    )
    assert normalise("FL_CC001") == normalise("fl_cc1")
    assert normalise("FL_P06U_001") == normalise("fl_p6u_1")

    level1 = zarr.open_consolidated(LEVEL1_ROOT / "12417.zarr", mode="r")
    level2 = zarr.open_group(LEVEL2_ROOT / "12417.zarr", mode="r")
    level1_positions = _level1_sensor_positions(level1["amb"])
    level2_positions = _level2_sensor_positions(level2["magnetics"])
    assert qualification.canonical_identity in level1_positions
    assert qualification.malformed_identity in level2_positions

    declared = set(acquisition.sensor_addresses)
    raw_level2_identities = {name.casefold() for name in level2_positions}
    unmatched_before = declared.difference(raw_level2_identities)
    assert unmatched_before == {"fl_cc10", "fl_p6u_1"}

    normalised_level2 = {normalise(name) for name in level2_positions}
    assert len(normalised_level2) == len(level2_positions)
    unmatched_after = {
        address for address in declared if normalise(address) not in normalised_level2
    }
    assert unmatched_after == {"fl_p6u_1"}
    assert len(unmatched_before) == 2
    assert len(unmatched_after) == 1
    print(
        "SENSOR_IDENTITY_NORMALISATION "
        f"shot=12417 malformed={qualification.malformed_identity} "
        f"canonical={qualification.canonical_identity} "
        f"normalised={normalise(qualification.malformed_identity)} "
        f"unmatched_before={len(unmatched_before)} "
        f"unmatched_after={len(unmatched_after)} "
        f"remaining={tuple(sorted(unmatched_after))}"
    )


@pytest.mark.skipif(
    not all(
        (LEVEL1_ROOT / f"{shot}.zarr").is_dir()
        and (LEVEL2_ROOT / f"{shot}.zarr").is_dir()
        for shot in (11_766, 12_417, 21_978)
    ),
    reason="local level-1 and level-2 geometry stores are not mounted",
)
def test_level1_and_level2_geometry_are_compared_by_sensor_identity():
    catalog = load_packaged_machine_map("mast")
    acquisitions = {item.name: item for item in catalog.acquisition_declarations}
    supplements = {item.name: item for item in catalog.description_supplements}

    for shot in (11_766, 12_417, 21_978):
        machine_map = map_for_shot(catalog, shot)
        level1_path = LEVEL1_ROOT / f"{shot}.zarr"
        level2_path = LEVEL2_ROOT / f"{shot}.zarr"
        level1 = zarr.open_consolidated(level1_path, mode="r")
        level2 = zarr.open_group(level2_path, mode="r")
        raw_level1_positions = _level1_sensor_positions(level1["amb"])
        raw_level2_positions = _level2_sensor_positions(level2["magnetics"])
        acquisition = acquisitions[
            supplements[machine_map.description_supplement].acquisition_declaration
        ]

        level1_positions = {
            catalog.normalise_sensor_identity(acquisition, name): position
            for name, position in raw_level1_positions.items()
        }
        level2_positions = {
            catalog.normalise_sensor_identity(acquisition, name): position
            for name, position in raw_level2_positions.items()
        }
        assert len(level1_positions) == len(raw_level1_positions)
        assert len(level2_positions) == len(raw_level2_positions)
        declared_by_identity = {
            catalog.normalise_sensor_identity(acquisition, address): address
            for address in acquisition.sensor_addresses
        }
        declared_addresses = set(declared_by_identity)
        assert set(level1_positions) == declared_addresses

        shared_identities = tuple(sorted(set(level1_positions) & set(level2_positions)))
        separations = {
            declared_by_identity[identity]: float(
                np.hypot(
                    level1_positions[identity][0] - level2_positions[identity][0],
                    level1_positions[identity][1] - level2_positions[identity][1],
                )
            )
            for identity in shared_identities
        }
        agreeing = tuple(name for name, value in separations.items() if value == 0.0)
        differing = tuple(name for name, value in separations.items() if value != 0.0)
        assert len(agreeing) + len(differing) == len(shared_identities)
        assert all(np.isfinite(tuple(separations.values())))
        missing_level2 = tuple(
            sorted(
                declared_by_identity[identity]
                for identity in declared_addresses.difference(shared_identities)
            )
        )
        maximum_name = max(separations, key=separations.__getitem__)
        print(
            "LEVEL1_LEVEL2_GEOMETRY_COMPARISON "
            f"shot={shot} shared={len(shared_identities)} agreeing={len(agreeing)} "
            f"differing={len(differing)} max_separation_m="
            f"{separations[maximum_name]:.15g} max_identity={maximum_name} "
            f"missing_level2={missing_level2}"
        )
        for prefix in ("ccbv", "obr", "obv", "fl_"):
            family = {
                name: value
                for name, value in separations.items()
                if name.startswith(prefix)
            }
            print(
                "LEVEL1_LEVEL2_GEOMETRY_FAMILY "
                f"shot={shot} family={prefix.removesuffix('_')} "
                f"shared={len(family)} "
                f"agreeing={sum(value == 0.0 for value in family.values())} "
                f"differing={sum(value != 0.0 for value in family.values())} "
                f"max_separation_m={max(family.values(), default=0.0):.15g}"
            )


def test_passive_loop_elements_and_shape_angles_are_executable_oblique_geometry():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    bindings = {
        item.source_array: item
        for item in catalog.binding_sets["mast-machine-description"]
        if item.source_group == "pf_passive"
    }
    expected_rectangle_sources = {
        f"{family}_{suffix}"
        for family in ("coil_cases",)
        for suffix in ("geometry_channel", "r", "z", "width", "height")
    }
    oblique_families = tuple(
        family for family in PASSIVE_LOOP_FAMILIES if family != "coil_cases"
    )
    expected_oblique_sources = {
        f"{family}_{suffix}"
        for family in oblique_families
        for suffix in (
            "geometry_channel",
            "r",
            "z",
            "width",
            "height",
            "shapeAngle1",
            "shapeAngle2",
        )
    }
    assert set(bindings) == expected_rectangle_sources | expected_oblique_sources
    assert len(bindings) == 110

    for family in oblique_families:
        name_binding = bindings[f"{family}_geometry_channel"]
        assert name_binding.dd_path == "pf_passive/loop/element/name"
        expected_targets = {
            "r": "r",
            "z": "z",
            "width": "length_alpha",
            "height": "length_beta",
            "shapeAngle1": "alpha",
            "shapeAngle2": "beta",
        }
        for source_suffix, target_suffix in expected_targets.items():
            binding = bindings[f"{family}_{source_suffix}"]
            assert binding.dd_path == (
                "pf_passive/loop/element/geometry/oblique/" + target_suffix
            )
        assert "first side inclination" in bindings[f"{family}_shapeAngle1"].evidence
        assert "second side inclination" in bindings[f"{family}_shapeAngle2"].evidence
        assert all(
            "must not be converted as lengths" in bindings[f"{family}_{angle}"].evidence
            for angle in ("shapeAngle1", "shapeAngle2")
        )

    metadata = json.loads((LEVEL2_ROOT / "12417.zarr" / "zarr.json").read_text())[
        "consolidated_metadata"
    ]["metadata"]
    for family in oblique_families:
        for angle in ("shapeAngle1", "shapeAngle2"):
            binding = bindings[f"{family}_{angle}"]
            assert binding.source_rank == len(
                metadata[f"pf_passive/{family}_{angle}"]["shape"]
            )
            assert binding.source_unit == "degree"

    oblique_assemblies = {
        item.name: item
        for item in catalog.structure_assemblies
        if item.name.startswith("mast-pf-passive-") and item.type_index == 3
    }
    assert len(oblique_assemblies) == 15
    assert {item.type_index for item in oblique_assemblies.values()} == {3}
    assert {item.type_path for item in oblique_assemblies.values()} == {
        "pf_passive/loop/element/geometry/geometry_type"
    }


def test_every_legacy_description_field_has_a_binding_or_qualification():
    catalog = load_packaged_machine_map("mast")
    bindings = catalog.binding_sets["mast-machine-description"]
    binding_names = {
        (item.source_group, item.source_array): item.name for item in bindings
    }
    qualification_names = {
        (item.source_group, item.source_array): item.name
        for item in catalog.source_qualifications
    }
    topology_names = tuple(item.name for item in catalog.drive_topologies)
    acquisition_names = tuple(item.name for item in catalog.acquisition_declarations)
    supplement_names = tuple(item.name for item in catalog.description_supplements)

    def binding_matches(group: str, suffix: str) -> tuple[str, ...]:
        return tuple(
            item.name
            for item in bindings
            if item.source_group == group and item.source_array.endswith(suffix)
        )

    accounting = {
        "magpr_r": tuple(
            item.name
            for item in bindings
            if item.source_group == "magnetics"
            and item.source_array.startswith("b_field_pol_probe_")
            and item.source_array.endswith("_r")
        ),
        "magpr_z": tuple(
            item.name
            for item in bindings
            if item.source_group == "magnetics"
            and item.source_array.startswith("b_field_pol_probe_")
            and item.source_array.endswith("_z")
        ),
        "magpr_ang": (qualification_names[("magnetics", "magpr_ang")],),
        "magpr_len.cc": (
            qualification_names[("magnetics", "b_field_pol_probe_cc_length")],
        ),
        "magpr_len.ccbv": (
            binding_names[("magnetics", "b_field_pol_probe_ccbv_length")],
        ),
        "magpr_len.obr": (
            binding_names[("magnetics", "b_field_pol_probe_obr_length")],
        ),
        "magpr_len.obv": (
            binding_names[("magnetics", "b_field_pol_probe_obv_length")],
        ),
        "magpr_len.omv": (
            qualification_names[("magnetics", "b_field_pol_probe_omv_length")],
        ),
        "silop_r": (binding_names[("magnetics", "flux_loop_r")],),
        "silop_z": (binding_names[("magnetics", "flux_loop_z")],),
        "silop_additional_positions": tuple(
            loop.name
            for supplement in catalog.description_supplements
            for loop in supplement.point_flux_loops
        ),
        "pf_active.element.r": binding_matches("pf_active", "_r"),
        "pf_active.element.z": binding_matches("pf_active", "_z"),
        "pf_active.element.width": binding_matches("pf_active", "_width"),
        "pf_active.element.height": binding_matches("pf_active", "_height"),
        "pf_passive.loop.name": (qualification_names[("pf_passive", "loop_names")],),
        "pf_passive.element.name": binding_matches("pf_passive", "_geometry_channel"),
        "pf_passive.element.r": binding_matches("pf_passive", "_r"),
        "pf_passive.element.z": binding_matches("pf_passive", "_z"),
        "pf_passive.element.width": binding_matches("pf_passive", "_width"),
        "pf_passive.element.height": binding_matches("pf_passive", "_height"),
        "pf_active.element.turns_with_sign": (*topology_names,),
        "pf_passive.element.turns_with_sign": (*topology_names,),
        "fcoil_circ": topology_names,
        "fcoil_xmult": topology_names,
        "amc_current_channels": acquisition_names,
        "unmatched_amb": acquisition_names,
        "conductor_element_join": tuple(
            item.name
            for item in catalog.structure_assemblies
            if item.element_identifiers
        ),
        "r0": supplement_names,
        "minor_radius": (qualification_names[("machine_description", "minor_radius")],),
        "polygon_sections": supplement_names,
        "limiterr": (binding_names[("wall", "limiter_r")],),
        "limiterz": (binding_names[("wall", "limiter_z")],),
    }

    assert tuple(accounting) == LEGACY_DESCRIPTION_FIELDS
    unaccounted = []
    for field, declarations in accounting.items():
        print(f"LEGACY_FIELD_ACCOUNTING field={field} declarations={declarations}")
        if not declarations:
            unaccounted.append(field)
    assert unaccounted == []

    assert len(catalog.drive_topologies) == 3
    for topology in catalog.drive_topologies:
        assert topology.circuit_identity_source == "fcoil_circ"
        assert topology.current_scale_source == "fcoil_xmult"
        assert topology.current_channel_source == "amc_current_channels"
        assert (
            topology.current_channel_declaration in accounting["amc_current_channels"]
        )
        assert topology.passive_loop_names == PASSIVE_LOOP_FAMILIES


def test_case_current_joins_are_explicit_and_resolvable():
    catalog = load_packaged_machine_map("mast")
    joins = catalog.circuit_current_joins
    joins_by_circuit = {item.circuit_identifier: item.current_channel for item in joins}

    assert len(joins) == len(CASE_CURRENT_JOINS) == 8
    assert joins_by_circuit == CASE_CURRENT_JOINS
    assert len({item.current_channel for item in joins}) == 8
    assert len({item.conductor_identifier for item in joins}) == 8

    acquisitions = {
        item.name: set(item.current_channels)
        for item in catalog.acquisition_declarations
    }
    for topology in catalog.drive_topologies:
        topology_circuits = {item.circuit_identifier for item in topology.connections}
        assert set(CASE_CURRENT_JOINS).issubset(topology_circuits)
        assert set(CASE_CURRENT_JOINS.values()).issubset(
            acquisitions[topology.current_channel_declaration]
        )
    for join in joins:
        assert join.conductor_identifier == (
            f"mast-pf-passive-coil-cases/{join.current_channel}"
        )
        assert join.current_channel in join.evidence
        print(
            "CASE_CURRENT_JOIN "
            f"circuit={join.circuit_identifier} channel={join.current_channel} "
            f"conductor={join.conductor_identifier}"
        )

    assert load_packaged_machine_map("diii-d").circuit_current_joins == ()


@pytest.mark.skipif(
    not all(
        (LEVEL2_ROOT / f"{shot}.zarr").is_dir()
        for shot in (11_766, 12_417, 13_361, 21_978)
    ),
    reason="local level-2 geometry stores are not mounted",
)
def test_range_declarations_join_topology_and_supply_legacy_fields():
    catalog = load_packaged_machine_map("mast")
    supplements = {item.name: item for item in catalog.description_supplements}
    acquisitions = {item.name: item for item in catalog.acquisition_declarations}
    topologies = {item.name: item for item in catalog.drive_topologies}
    minor_radius = next(
        item
        for item in catalog.source_qualifications
        if item.name == "mast-machine-description-minor-radius"
    )

    assert "equilibrium/time_slice/boundary/minor_radius" in minor_radius.reason
    assert "pulse_schedule/position_control/minor_radius/reference" in (
        minor_radius.reason
    )
    assert "summary/boundary/minor_radius/value" in minor_radius.reason

    for shot in (11_766, 12_417, 13_361, 21_978):
        machine_map = map_for_shot(catalog, shot)
        description = transform_machine_description(
            catalog, shot, "zarr", LEVEL2_ROOT
        )
        emitted_by_binding = {
            item.binding_name: np.asarray(item.values) for item in description.arrays
        }
        emitted_element_identifiers: set[str] = set()
        for assembly in catalog.structure_assemblies:
            if not assembly.element_identifiers:
                continue
            emitted_names = emitted_by_binding[assembly.name_binding].reshape(-1)
            assert len(emitted_names) == len(assembly.element_identifiers)
            for identifier, emitted_name in zip(
                assembly.element_identifiers, emitted_names, strict=True
            ):
                assert identifier.rsplit("/", maxsplit=1)[-1] == str(emitted_name)
                emitted_element_identifiers.add(identifier)

        assert len(emitted_element_identifiers) == 938
        topology = topologies[machine_map.drive_topology]
        unresolved = {
            item.geometry_element_identifier for item in topology.connections
        }.difference(emitted_element_identifiers)
        expected_topology_rows = 938 if machine_map.first_shot >= 13_361 else 1_004
        assert len(topology.connections) == expected_topology_rows
        assert unresolved == set()

        supplement = supplements[machine_map.description_supplement]
        acquisition = acquisitions[supplement.acquisition_declaration]
        topology_acquisition = acquisitions[topology.current_channel_declaration]
        assert set(CASE_CURRENT_JOINS.values()).issubset(
            topology_acquisition.current_channels
        )
        assert set(CASE_CURRENT_JOINS.values()).issubset(
            acquisition.current_channels
        )
        expected_current_channels = 45 if machine_map.first_shot >= 16_606 else 44
        assert len(acquisition.current_channels) == expected_current_channels
        expected_unmatched = {
            11_766: 2,
            12_417: 8,
            13_361: 8,
            16_606: 4,
            21_978: 4,
            22_087: 4,
        }[machine_map.first_shot]
        assert len(acquisition.unmatched_sensor_addresses) == expected_unmatched
        assert set(acquisition.unmatched_sensor_addresses).issubset(
            acquisition.sensor_addresses
        )

        emitted_loop_count = emitted_by_binding["mast-magnetics-flux-loop-r"].size
        assert emitted_loop_count == 44
        assert len(supplement.point_flux_loops) == 2
        assert emitted_loop_count + len(supplement.point_flux_loops) == 46
        for loop in supplement.point_flux_loops:
            r_leaf = _dd_leaf(catalog.dd_version, loop.r_path)
            z_leaf = _dd_leaf(catalog.dd_version, loop.z_path)
            type_leaf = _dd_leaf(catalog.dd_version, loop.type_path)
            assert (r_leaf.data_type.name, r_leaf.ndim, r_leaf.units) == (
                "FLT",
                0,
                "m",
            )
            assert (z_leaf.data_type.name, z_leaf.ndim, z_leaf.units) == (
                "FLT",
                0,
                "m",
            )
            assert (type_leaf.data_type.name, type_leaf.ndim) == ("INT", 0)
            assert loop.type_index == 1

        r0_leaf = _dd_leaf(catalog.dd_version, supplement.reference_radius_path)
        assert supplement.reference_radius == 0.85
        assert supplement.reference_radius_unit == "m"
        assert (r0_leaf.data_type.name, r0_leaf.ndim, r0_leaf.units) == (
            "FLT",
            0,
            "m",
        )
        assert "Reference major radius of the device" in r0_leaf.documentation

        topology_circuits = {item.circuit_identifier for item in topology.connections}
        assert len(supplement.polygon_sections) == 4
        for section in supplement.polygon_sections:
            assert section.circuit_identifier in topology_circuits
            assert section.geometry_element_identifier in emitted_element_identifiers
            assert len(section.vertex_r) == len(section.vertex_z) == 4

        print(
            "CATALOG_PARITY_DECLARATIONS "
            f"shot={machine_map.first_shot} topology_rows={len(topology.connections)} "
            f"unjoined={len(unresolved)} emitted_elements="
            f"{len(emitted_element_identifiers)} point_loops="
            f"{emitted_loop_count + len(supplement.point_flux_loops)} "
            f"current_channels={len(acquisition.current_channels)} "
            f"sensor_addresses={len(acquisition.sensor_addresses)} "
            f"unmatched={len(acquisition.unmatched_sensor_addresses)} "
            f"polygons={len(supplement.polygon_sections)} r0="
            f"{supplement.reference_radius} minor_radius=qualified"
        )


@pytest.mark.skipif(
    not all(
        (LEVEL2_ROOT / f"{shot}.zarr").is_dir()
        for shot in (11_766, 12_417, 13_361, 21_978)
    ),
    reason="local level-2 geometry stores are not mounted",
)
def test_case_current_joins_materialize_the_operator_block_split():
    catalog = load_packaged_machine_map("mast")
    topologies = {item.name: item for item in catalog.drive_topologies}

    for shot in (11_766, 12_417, 13_361, 21_978):
        machine_map = map_for_shot(catalog, shot)
        description = transform_machine_description(catalog, shot, "zarr", LEVEL2_ROOT)
        adapted = geometry_table_from_description(description, catalog)
        topology = topologies[machine_map.drive_topology]
        circuit_order = tuple(
            dict.fromkeys(item.circuit_identifier for item in topology.connections)
        )
        circuit_index = {
            identifier: index + 1 for index, identifier in enumerate(circuit_order)
        }

        classes = classify_circuits(
            adapted.pf_filaments,
            adapted.amc_current_channels,
            adapted.active_circuits,
            adapted.circuit_drives,
        )
        assert len(adapted.circuit_drives) == 21
        assert len({item.circuit for item in adapted.circuit_drives}) == 21
        drives_by_circuit = {item.circuit: item for item in adapted.circuit_drives}
        classes_by_circuit = {item.circuit: item for item in classes}
        remaining_discrepancies = []
        for join in catalog.circuit_current_joins:
            connections = [
                item
                for item in topology.connections
                if item.circuit_identifier == join.circuit_identifier
            ]
            assert connections
            circuit = circuit_index[join.circuit_identifier]
            drive = drives_by_circuit[circuit]
            assert drive.channel == join.current_channel
            assert drive.conductor == join.conductor_identifier
            assert drive.ampere_turns_per_ampere == sum(
                item.turns * item.current_weight * item.direction
                for item in connections
            )
            circuit_class = classes_by_circuit[circuit]
            if (
                circuit_class.role != "known_case"
                or circuit_class.amc_channel != join.current_channel
            ):
                remaining_discrepancies.append(join.circuit_identifier)
        assert remaining_discrepancies == []

        print(
            "CASE_CURRENT_OPERATOR_SPLIT "
            f"shot={shot} declared_drives={len(adapted.circuit_drives)} "
            f"classified_circuits={len(classes)} "
            f"remaining={','.join(remaining_discrepancies) or 'none'}"
        )


def test_turn_and_current_weight_magnitudes_are_positive():
    for machine in ("mast", "diii-d"):
        catalog = load_packaged_machine_map(machine)
        for bindings in catalog.binding_sets.values():
            for binding in bindings:
                if (
                    "turn" in binding.source_array.lower()
                    or "turn" in binding.dd_path.lower()
                ):
                    assert binding.sign_convention != "negate"
        for qualification in catalog.source_qualifications:
            declaration = " ".join(
                (qualification.name, qualification.source_array, qualification.reason)
            ).lower()
            assert "turns_with_sign" not in declaration
        for topology in catalog.drive_topologies:
            assert topology.circuit_identifier_path == "pf_active/circuit/name"
            assert topology.supply_identifier_path == "pf_active/supply/name"
            assert topology.element_identifier_path == "pf_active/coil/element/name"
            assert topology.turns_path == "pf_active/coil/element/turns_with_sign"
            assert topology.connections_path == "pf_active/circuit/connections"
            assert all(connection.turns > 0 for connection in topology.connections)
            assert all(
                connection.current_weight > 0
                for connection in topology.connections
            )
            assert {connection.direction for connection in topology.connections} <= {
                -1,
                1,
            }
            circuit_identifier = _dd_leaf(
                catalog.dd_version, topology.circuit_identifier_path
            )
            supply_identifier = _dd_leaf(
                catalog.dd_version, topology.supply_identifier_path
            )
            element_identifier = _dd_leaf(
                catalog.dd_version, topology.element_identifier_path
            )
            turns = _dd_leaf(catalog.dd_version, topology.turns_path)
            connections = _dd_leaf(catalog.dd_version, topology.connections_path)
            assert circuit_identifier.data_type.name == "STR"
            assert supply_identifier.data_type.name == "STR"
            assert element_identifier.data_type.name == "STR"
            assert "identifier" in circuit_identifier.documentation
            assert "identifier" in supply_identifier.documentation
            assert "identifier" in element_identifier.documentation
            assert turns.data_type.name == "FLT"
            assert turns.units == "1"
            assert "Should be positive" in turns.documentation
            assert (connections.data_type.name, connections.ndim) == ("INT", 2)
            assert "positive side" in connections.documentation
            assert "negative side" in connections.documentation


def test_sparse_connectivity_reconstructs_every_legacy_signed_drive_element():
    catalog = load_packaged_machine_map("mast")
    payload = load_geometry_table_payload()
    topologies = {
        topology.source_location.rsplit("/", maxsplit=1)[-1]: topology
        for topology in catalog.drive_topologies
    }
    total_elements = 0
    direction_counts = {-1: 0, 1: 0}

    for signature, campaign in payload["campaigns"].items():
        topology = topologies[signature]
        filaments = campaign["pf_filaments"]
        expected_circuits = tuple(
            f"fcoil-circuit-{int(item['circuit']):03d}" for item in filaments
        )
        expected_elements = tuple(
            f"fcoil-element-{index:04d}" for index in range(len(filaments))
        )
        expected_supplies = tuple(
            f"fcoil-supply-{int(item['circuit']):03d}" for item in filaments
        )
        legacy_turns = np.asarray([float(item["turns"]) for item in filaments])
        legacy_current_weight = np.asarray(
            [abs(float(item["xmult"])) for item in filaments]
        )
        legacy_signed_drive = np.asarray(
            [float(item["turns"]) * float(item["xmult"]) for item in filaments]
        )
        circuit_identifiers = tuple(
            connection.circuit_identifier for connection in topology.connections
        )
        element_identifiers = tuple(
            connection.element_identifier for connection in topology.connections
        )
        supply_identifiers = tuple(
            connection.supply_identifier for connection in topology.connections
        )
        positive_turns = np.asarray(
            [connection.turns for connection in topology.connections]
        )
        positive_current_weight = np.asarray(
            [connection.current_weight for connection in topology.connections]
        )

        unique_circuits = tuple(dict.fromkeys(circuit_identifiers))
        circuit_rows = {
            identifier: index for index, identifier in enumerate(unique_circuits)
        }
        connectivity = np.zeros(
            (len(unique_circuits), len(topology.connections)), dtype=np.int8
        )
        for column, connection in enumerate(topology.connections):
            connectivity[circuit_rows[connection.circuit_identifier], column] = (
                connection.direction
            )
            direction_counts[connection.direction] += 1

        selected_direction = connectivity[
            [circuit_rows[identifier] for identifier in circuit_identifiers],
            np.arange(len(topology.connections)),
        ]
        reconstructed = positive_turns * positive_current_weight * selected_direction
        assert circuit_identifiers == expected_circuits
        assert supply_identifiers == expected_supplies
        assert element_identifiers == expected_elements
        assert np.count_nonzero(connectivity, axis=0).tolist() == [1] * len(filaments)
        assert np.array_equal(positive_turns, legacy_turns)
        assert np.array_equal(positive_current_weight, legacy_current_weight)
        assert np.array_equal(reconstructed, legacy_signed_drive)
        total_elements += len(filaments)
        negative_count = np.count_nonzero(reconstructed < 0)
        print(
            "CIRCUIT_DRIVE_RECONSTRUCTION "
            f"signature={signature} elements={len(filaments)} "
            f"circuits={len(unique_circuits)} negative={negative_count}"
        )

    assert total_elements == 2_946
    assert direction_counts == {-1: 0, 1: 2_946}
    assert {item.drive_topology for item in catalog.maps} == {
        item.name for item in catalog.drive_topologies
    }


def test_every_mast_map_range_is_one_geometry_transition():
    if not LEVEL2_ROOT.is_dir():
        pytest.skip("FAIR-MAST level-2 mirror is not mounted")
    catalog = load_packaged_machine_map("mast")
    shots, _ = _machine_description_inventory()
    transitions = build_geometry_transitions(
        shots,
        load_index(),
        load_geometry_table_payload(),
    )

    declared_ranges = [
        (item.first_shot, item.last_shot, item.transition) for item in catalog.maps
    ]
    transitions_by_name = {item.name: item for item in transitions}
    mismatches = [
        item.name
        for item in catalog.maps
        if item.transition not in transitions_by_name
        or not (
            transitions_by_name[item.transition].first_shot <= item.first_shot
            and item.last_shot <= transitions_by_name[item.transition].last_shot
        )
    ]

    assert declared_ranges == [
        (11_766, 12_416, "mast-geometry-11766-9425ae4a8bf3bc15"),
        (12_417, 13_360, "mast-geometry-12417-edd753d282903679"),
        (13_361, 16_605, "mast-geometry-12417-edd753d282903679"),
        (16_606, 21_977, "mast-geometry-12417-edd753d282903679"),
        (21_978, 22_086, "mast-geometry-12417-edd753d282903679"),
        (22_087, 30_471, "mast-geometry-12417-edd753d282903679"),
    ]
    assert len(transitions) == 2
    assert len(catalog.maps) == 6
    assert len(mismatches) == 0
    assert_transition_alignment(catalog, transitions)
    bindings = catalog.binding_sets["mast-machine-description"]
    assert all(catalog.bindings_for(item) == bindings for item in catalog.maps)
    assert len({item.name for item in bindings}) == catalog.bound_channel_count
    assert len({item.name for item in catalog.source_qualifications}) == (
        catalog.qualified_channel_count
    )
    topology_names = {item.name for item in catalog.drive_topologies}
    for transition in transitions:
        covering_maps = [
            item for item in catalog.maps if item.transition == transition.name
        ]
        assert covering_maps[0].first_shot == transition.first_shot
        assert covering_maps[-1].last_shot == transition.last_shot
        assert all(item.drive_topology in topology_names for item in covering_maps)
    assert len({item.drive_topology for item in catalog.maps}) == 3
    print(
        "CATALOG_TRANSITION_ALIGNMENT "
        f"maps={len(catalog.maps)} transitions={len(transitions)} "
        f"mismatches={len(mismatches)} executable={dict(catalog.bound_channel_counts)} "
        f"qualified={dict(catalog.qualified_channel_counts)}"
    )


def test_diii_d_public_map_retains_only_direct_plasma_current():
    catalog = load_packaged_machine_map("diii-d")
    machine_map = catalog.maps[0]
    bindings = catalog.bindings_for(machine_map)
    gaps = {gap.binding: gap.reason for gap in catalog.validation_gaps}

    assert catalog.source == "https://github.com/MIT-PSFC/disruption-py"
    assert catalog.source_revision == "dec5c58a3e3970bc6817f33efb615fea11057fce"
    assert machine_map.validation_state == "source-only"
    assert catalog.bound_channel_count == len(bindings) == len(gaps) == 1
    assert catalog.source_qualifications == ()
    assert gaps.keys() == {binding.name for binding in bindings}
    assert all("No DIII-D pulse corpus" in reason for reason in gaps.values())
    assert all(binding.sign_convention == "unknown-unvalidated" for binding in bindings)
    assert [
        (binding.source_group, binding.source_array, binding.dd_path)
        for binding in bindings
    ] == [("d3d", "ip", "magnetics/ip/data")]
    for binding in bindings:
        leaf = _dd_leaf(catalog.dd_version, binding.dd_path)
        assert leaf.data_type.name == "FLT"
        assert leaf.units == binding.target_unit
        assert catalog.source_revision in binding.evidence


def test_machine_maps_exclude_reconstruction_derived_bindings():
    disallowed = {"equilibrium", "q95", "q_95", "li_3", "betan", "beta_tor_norm"}
    for machine in ("mast", "diii-d"):
        catalog = load_packaged_machine_map(machine)
        for bindings in catalog.binding_sets.values():
            for binding in bindings:
                declaration = "/".join(
                    (binding.source_group, binding.source_array, binding.dd_path)
                ).lower()
                assert not any(term in declaration for term in disallowed)


def test_loader_rejects_an_undeclared_conditional_slot(tmp_path):
    source = json.loads((LINKML_SCHEMA_PATH.parent / "mast.json").read_text())
    source["maps"][0]["condition"] = "shot > 12000"
    invalid = tmp_path / "conditional.json"
    invalid.write_text(json.dumps(source))

    with pytest.raises(MachineMapError, match="extra=.+condition"):
        load_machine_map(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("turns", -1.0, "number > 0"),
        ("current_weight", 0.0, "number > 0"),
        ("direction", 0, "must be -1 or 1"),
    ),
)
def test_loader_rejects_nonpositive_magnitudes_and_unsigned_connections(
    tmp_path, field, value, message
):
    source = json.loads((LINKML_SCHEMA_PATH.parent / "mast.json").read_text())
    source["drive_topologies"][0]["connections"][0][field] = value
    invalid = tmp_path / f"invalid-{field}.json"
    invalid.write_text(json.dumps(source))

    with pytest.raises(MachineMapError, match=message):
        load_machine_map(invalid)
