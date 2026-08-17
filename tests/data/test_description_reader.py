"""Receipts for the declared-description compatibility boundary."""

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
        if isinstance(node, ast.ImportFrom) and node.module == "imas_ambix.gs.geometry":
            for alias in node.names:
                direct[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "imas_ambix.gs.geometry":
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


def test_library_description_reader_census_is_zero() -> None:
    root = Path(__file__).resolve().parents[2]
    library, scripts = _description_consumers(root)

    print(
        "DESCRIPTION_READER_CENSUS "
        f"library={len(library)} scripts={len(scripts)} "
        f"total={len(library) + len(scripts)}"
    )
    assert library == ()
    assert len(scripts) == 33


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

    assert probes
    assert all(item.angle_deg in (-90.0, 0.0) for item in probes)
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
