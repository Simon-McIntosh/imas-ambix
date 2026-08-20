"""Receipts for the sole declared-description acquisition boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

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

    monkeypatch.setattr(reader, "load_packaged_machine_map", lambda machine: object())
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
