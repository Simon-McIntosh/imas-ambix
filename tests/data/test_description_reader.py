"""Receipts for the sole declared-description acquisition boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from imas_ambix.data.description_reader import (
    DescriptionReadError,
    read_acquisition_channels,
    read_geometry_table,
)

LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
PRIVATE_GEOMETRY_MODULE = "imas_ambix.gs.geometry"

DESCRIPTION_READ_APIS = frozenset(
    {
        "build_table_for_shot",
        "canonical_amb_channels",
        "discover_signatures",
        "extract_campaign_tables",
        "read_amb_channels",
        "read_amc_current_channels",
        "read_amm_passive",
        "read_efm_geometry",
        "setup_signature",
    }
)


def _description_read_calls(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    direct: dict[str, str] = {}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == PRIVATE_GEOMETRY_MODULE:
            for alias in node.names:
                direct[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PRIVATE_GEOMETRY_MODULE:
                    modules.add(alias.asname or alias.name)

    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            imported = direct.get(node.func.id)
            if imported in DESCRIPTION_READ_APIS:
                calls.add(imported)
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in modules
            and node.func.attr in DESCRIPTION_READ_APIS
        ):
            calls.add(node.func.attr)
    return frozenset(calls)


def _description_consumers(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def consumers(source_root: Path) -> tuple[str, ...]:
        return tuple(
            sorted(
                str(path.relative_to(root))
                for path in source_root.rglob("*.py")
                if _description_read_calls(path)
            )
        )

    return consumers(root / "imas_ambix"), consumers(root / "scripts")


def test_description_reader_census_is_zero() -> None:
    root = Path(__file__).resolve().parents[2]
    library, scripts = _description_consumers(root)

    print(
        "DESCRIPTION_READER_CENSUS "
        f"library={len(library)} scripts={len(scripts)} "
        f"total={len(library) + len(scripts)}"
    )
    assert library == ()
    assert scripts == ()


def test_raw_description_entrypoints_are_absent() -> None:
    from imas_ambix.gs import machine_geometry

    assert all(not hasattr(machine_geometry, name) for name in DESCRIPTION_READ_APIS)


def test_facade_rejects_a_description_that_was_not_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import imas_ambix.data.description_reader as reader

    monkeypatch.setattr(
        reader,
        "load_packaged_machine_map",
        lambda machine: SimpleNamespace(
            description_store_format="zarr",
            description_store_root="LEVEL2_DIR",
            probe_angle_source="acquisition-address",
        ),
    )
    monkeypatch.setattr(
        reader,
        "transform_machine_description",
        lambda *args, **kwargs: SimpleNamespace(
            status="source-unavailable",
            detail="fixture is absent",
        ),
    )

    with pytest.raises(DescriptionReadError, match="source-unavailable"):
        read_geometry_table(12_417)


def test_declared_probe_angle_source_selects_the_angle_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import imas_ambix.data.description_reader as reader

    sentinel = SimpleNamespace(sensor_map=[], provenance_flags=[])

    def catalog_with_angle_source(angle_source: str) -> SimpleNamespace:
        return SimpleNamespace(
            description_store_format="zarr",
            description_store_root="LEVEL2_DIR",
            probe_angle_source=angle_source,
        )

    monkeypatch.setattr(
        reader,
        "transform_machine_description",
        lambda *args, **kwargs: SimpleNamespace(status="emitted", detail=""),
    )
    monkeypatch.setattr(
        reader,
        "geometry_table_from_description",
        lambda description, catalog: sentinel,
    )

    applied: list[object] = []
    monkeypatch.setattr(
        reader,
        "_supply_declared_probe_angles",
        lambda table: applied.append("declared") or sentinel,
    )
    monkeypatch.setattr(
        reader,
        "_supply_emitted_probe_angles",
        lambda description, table: applied.append("emitted") or sentinel,
    )

    monkeypatch.setattr(
        reader,
        "load_packaged_machine_map",
        lambda machine: catalog_with_angle_source("acquisition-address"),
    )
    assert read_geometry_table(1) is sentinel
    assert applied == ["declared"]

    applied.clear()
    monkeypatch.setattr(
        reader,
        "load_packaged_machine_map",
        lambda machine: catalog_with_angle_source("description"),
    )
    assert read_geometry_table(1) is sentinel
    assert applied == ["emitted"]


def test_catalog_with_no_description_store_refuses_a_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import imas_ambix.data.description_reader as reader

    monkeypatch.setattr(
        reader,
        "load_packaged_machine_map",
        lambda machine: SimpleNamespace(
            description_store_format=None,
            description_store_root=None,
            description_store_layout=None,
            probe_angle_source="description",
        ),
    )

    with pytest.raises(DescriptionReadError, match="'diii-d'.*no description store"):
        read_geometry_table(170_000, machine="diii-d")
    with pytest.raises(DescriptionReadError, match="'diii-d'.*no description store"):
        read_acquisition_channels((170_000,), machine="diii-d")

    captured: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        reader,
        "transform_machine_description",
        lambda *args, **kwargs: captured.append(args)
        or SimpleNamespace(status="source-unavailable", detail="absent"),
    )
    with pytest.raises(DescriptionReadError, match="source-unavailable"):
        read_geometry_table(
            170_000,
            machine="diii-d",
            store_format="zarr",
            store_root="/tmp/elsewhere",
        )
    assert captured and captured[0][2:] == ("zarr", "/tmp/elsewhere")


@pytest.mark.skipif(
    not (LEVEL2_ROOT / "21978.zarr").is_dir(),
    reason="local level-2 geometry stores are not mounted",
)
def test_real_description_supplies_every_probe_axis_and_acquisition_address() -> None:
    table = read_geometry_table(21_978, store_root=LEVEL2_ROOT)
    acquisition = read_acquisition_channels((21_978,), store_root=LEVEL2_ROOT)
    probes = tuple(item for item in table.sensor_map if item.kind == "b_probe")
    loops = tuple(item for item in table.sensor_map if item.kind == "flux_loop")

    assert table.signature.key == "mp78-fl46-fc938-lim37-532938247d31ec5c"
    assert len(table.sensor_map) == 96
    assert len(probes) == 77
    assert len(loops) == 19
    assert all(item.angle_deg in (-90.0, 0.0) for item in probes)
    addresses = {item.amb_channel for item in table.sensor_map}
    assert {"ccbv10", "fl_p6u_1"}.issubset(addresses)
    assert {"fl_cc02", "fl_cc10"}.isdisjoint(addresses)
    assert len(table.amc_current_channels) == 45
    assert set(table.unmatched_amb) == {
        "fl_p2l_1",
        "fl_p2l_3",
        "fl_p2u_1",
        "fl_p2u_3",
    }
    assert tuple(item.amb_channel for item in table.sensor_map) == tuple(
        item[0] for item in acquisition.sensors
    )
    assert acquisition.currents == tuple(table.amc_current_channels)
    assert any(
        "reviewed MAST acquisition-address convention" in item
        for item in table.provenance_flags
    )


def test_enkf_operator_uses_declared_target_and_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import imas_ambix.data.description_reader as reader
    import imas_ambix.gs.operator as operator_module
    from imas_ambix.statespace.enkf_baseline import _operator_for_shot

    reads: list[int] = []

    def declared_table(shot: int) -> SimpleNamespace:
        reads.append(int(shot))
        return SimpleNamespace(
            shot=int(shot),
            signature=SimpleNamespace(key="mast-declared"),
        )

    monkeypatch.setattr(reader, "read_geometry_table", declared_table)
    monkeypatch.setattr(
        operator_module,
        "build_operator",
        lambda table: ("declared-operator", table.shot),
    )

    cache: dict[str, object] = {}
    built = _operator_for_shot(
        21_978,
        cache,
        reps={"mast-declared": [21_983]},
    )

    assert built == ("declared-operator", 21_983)
    assert cache == {"mast-declared": built}
    assert reads == [21_978, 21_983]


def _probe_angle_description(values, target_unit="rad", names=("p1", "p2")):
    arrays = [
        SimpleNamespace(
            dd_path="magnetics/b_field_pol_probe/poloidal_angle",
            target_unit=target_unit,
            values=np.asarray(values, dtype=np.float64),
        )
    ]
    if names is not None:
        arrays.append(
            SimpleNamespace(
                dd_path="magnetics/b_field_pol_probe/name",
                target_unit="1",
                values=np.asarray(names, dtype=object),
            )
        )
    return SimpleNamespace(arrays=tuple(arrays))


def _probe_angle_table():
    from dataclasses import dataclass

    from imas_ambix.gs.geometry import SensorMapping

    @dataclass
    class _Table:
        sensor_map: list
        provenance_flags: list

    def mapping(channel, kind, angle_deg, flag):
        return SensorMapping(
            amb_channel=channel,
            kind=kind,
            efm_index=0,
            r=1.0,
            z=0.0,
            angle_deg=angle_deg,
            residual_m=0.0,
            flag=flag,
        )

    return _Table(
        sensor_map=[
            mapping("p1", "b_probe", None, "poloidal angle is absent"),
            mapping("fl1", "flux_loop", None, ""),
            mapping("p2", "b_probe", None, "poloidal angle is absent"),
        ],
        provenance_flags=[],
    )


def test_description_source_fills_probe_angles_from_the_emitted_array():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    description = _probe_angle_description([0.0, np.pi / 2.0])
    table = _supply_emitted_probe_angles(description, _probe_angle_table())

    angles = {item.amb_channel: item.angle_deg for item in table.sensor_map}
    assert angles["p1"] == pytest.approx(0.0)
    assert angles["p2"] == pytest.approx(90.0)
    assert angles["fl1"] is None
    assert all(item.flag == "" for item in table.sensor_map)
    assert any(
        "poloidal_angle" in flag for flag in table.provenance_flags
    )


def test_description_source_leaves_absent_angles_when_nothing_was_emitted():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    description = SimpleNamespace(arrays=())
    table = _supply_emitted_probe_angles(description, _probe_angle_table())

    assert all(item.angle_deg is None for item in table.sensor_map)
    assert table.provenance_flags == []


def test_description_source_refuses_a_probe_the_angles_do_not_cover():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    description = _probe_angle_description([0.0], names=("p1",))

    with pytest.raises(DescriptionReadError, match="no angle for mapped probe 'p2'"):
        _supply_emitted_probe_angles(description, _probe_angle_table())


def test_description_source_refuses_an_angle_for_an_unmapped_probe():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    description = _probe_angle_description(
        [0.0, np.pi / 2.0, np.pi / 4.0], names=("p1", "p2", "p9")
    )

    with pytest.raises(
        DescriptionReadError, match="identities are not mapped: p9"
    ):
        _supply_emitted_probe_angles(description, _probe_angle_table())


def test_description_source_refuses_an_unknown_angle_unit():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    description = _probe_angle_description([0.0, 90.0], target_unit="arcmin")

    with pytest.raises(
        DescriptionReadError, match="declares angle unit 'arcmin'"
    ):
        _supply_emitted_probe_angles(description, _probe_angle_table())


def test_description_source_refuses_when_the_name_array_is_not_exactly_one():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    # No name array at all: nothing names the probes the angles belong to.
    without_names = _probe_angle_description([0.0, np.pi / 2.0], names=None)
    with pytest.raises(
        DescriptionReadError, match="must carry exactly one .* found 0"
    ):
        _supply_emitted_probe_angles(without_names, _probe_angle_table())

    # Two name arrays: which one names the angles is ambiguous.
    doubled = _probe_angle_description([0.0, np.pi / 2.0])
    doubled.arrays = doubled.arrays + (
        SimpleNamespace(
            dd_path="magnetics/b_field_pol_probe/name",
            target_unit="1",
            values=np.asarray(("p1", "p2"), dtype=object),
        ),
    )
    with pytest.raises(
        DescriptionReadError, match="must carry exactly one .* found 2"
    ):
        _supply_emitted_probe_angles(doubled, _probe_angle_table())


def test_description_source_refuses_a_duplicate_probe_name():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    description = _probe_angle_description(
        [0.0, np.pi / 2.0], names=("p1", "p1")
    )

    with pytest.raises(
        DescriptionReadError, match="names probe 'p1' more than once"
    ):
        _supply_emitted_probe_angles(description, _probe_angle_table())


def test_description_source_refuses_a_name_count_that_disagrees_with_the_angles():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    description = _probe_angle_description([0.0, np.pi / 2.0], names=("p1",))

    with pytest.raises(
        DescriptionReadError, match="holds 1 names for 2 .* angles"
    ):
        _supply_emitted_probe_angles(description, _probe_angle_table())


def test_description_source_joins_a_permuted_angle_array_by_probe_identity():
    from imas_ambix.data.description_reader import _supply_emitted_probe_angles

    # The emitted name array orders the angles p2-then-p1, the reverse of the
    # sensor map.  A reader that joins by position swaps the two probes; one
    # that joins by identity gives each probe its own angle.
    description = _probe_angle_description(
        [np.deg2rad(80.0), np.deg2rad(10.0)], names=("p2", "p1")
    )
    table = _supply_emitted_probe_angles(description, _probe_angle_table())

    angles = {item.amb_channel: item.angle_deg for item in table.sensor_map}
    assert angles["p1"] == pytest.approx(10.0)
    assert angles["p2"] == pytest.approx(80.0)
    assert angles["fl1"] is None
