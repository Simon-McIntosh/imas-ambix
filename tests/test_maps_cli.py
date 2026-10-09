"""Contracts for the machine map release CLI.

The version state machine, the unchanged-tree skip and the release
annotations are exercised against a stubbed registry: the tag list, the
manifest annotations and the artifact push are replaced in the module under
test, so a test never reaches GHCR.  The stub's tag list is returned in the
registry's own order, which is not sorted, so the tests also show that the
module sorts it newest first before reading "the latest tag".
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from imas_ambix import maps_cli

if TYPE_CHECKING:
    import pytest

from imas_ambix.maps_cli import (
    BASELINE_VERSION,
    compute_next_version,
    latest_stable_tag,
    latest_tag,
    semver_tags,
    stage_bundle,
    tree_digest,
)

HANDOFF_FIXTURE = Path(__file__).parent / "fixtures" / "mapping_handoff_example.json"


def test_score_handoff_prints_table_and_writes_json(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from imas_alambic.machine_map import SensorIdentityRule
    from imas_ambix.data.generated_mapping_import import import_generated_mappings

    root = tmp_path / "maps"
    machine_dir = root / "jt-60sa"
    (machine_dir / "maps").mkdir(parents=True)
    (machine_dir / "machine_map.json").write_text("{}")
    (machine_dir / "maps" / "magnetics.json").write_text("{}")
    catalogue = SimpleNamespace(
        source_cocos=1,
        sensor_identity_rules=(
            SensorIdentityRule(
                name="identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="description names",
            ),
        ),
    )
    members = {
        "magnetics/b_field_pol_probe": ("011", "010"),
        "pf_active/coil": ("02", "01"),
    }
    document = json.loads(HANDOFF_FIXTURE.read_text())
    reference = import_generated_mappings(document, catalogue, members).maps[0]
    import imas_alambic.machine_map as machine_map_module
    import imas_alambic.signal_map as signal_map_module

    monkeypatch.setattr(maps_cli, "MAPS_DIR", root)
    monkeypatch.setattr(
        maps_cli, "_description_members", lambda *args, **kwargs: members
    )
    monkeypatch.setattr(machine_map_module, "load_machine_map", lambda path: catalogue)
    monkeypatch.setattr(signal_map_module, "load_signal_map", lambda path: reference)
    output = tmp_path / "score.json"

    result = CliRunner().invoke(
        maps_cli.maps,
        ["score-handoff", "jt-60sa", str(HANDOFF_FIXTURE), "--json", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert "exported imported agreeing" in result.output
    assert "sign_unscored" in result.output
    assert "no reference" in result.output
    written = json.loads(output.read_text())
    assert written["total"]["exported"] == 5
    assert written["total"]["imported"] == 4
    assert written["by_ids"]["magnetics"]["agreeing"] == 2
    assert written["by_ids"]["magnetics"]["sign_unscored"] == 0
    assert written["by_ids"]["pf_active"]["agreeing"] == "no reference"


def test_score_handoff_prints_a_conflicting_column(tmp_path, monkeypatch):
    """A row bound to a hand-built target by another array shows as conflicting."""
    from types import SimpleNamespace

    from imas_alambic.machine_map import SensorIdentityRule
    from imas_ambix.data.generated_mapping_import import import_generated_mappings

    root = tmp_path / "maps"
    machine_dir = root / "jt-60sa"
    (machine_dir / "maps").mkdir(parents=True)
    (machine_dir / "machine_map.json").write_text("{}")
    (machine_dir / "maps" / "magnetics.json").write_text("{}")
    catalogue = SimpleNamespace(
        source_cocos=1,
        sensor_identity_rules=(
            SensorIdentityRule(
                name="identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="description names",
            ),
        ),
    )
    members = {
        "magnetics/b_field_pol_probe": ("011", "010"),
        "pf_active/coil": ("02", "01"),
    }
    document = json.loads(HANDOFF_FIXTURE.read_text())
    reference = import_generated_mappings(document, catalogue, members).maps[0]
    first, second = reference.signals
    reference = dataclasses.replace(
        reference,
        signals=(first, dataclasses.replace(second, source_array="other")),
    )
    import imas_alambic.machine_map as machine_map_module
    import imas_alambic.signal_map as signal_map_module

    monkeypatch.setattr(maps_cli, "MAPS_DIR", root)
    monkeypatch.setattr(
        maps_cli, "_description_members", lambda *args, **kwargs: members
    )
    monkeypatch.setattr(machine_map_module, "load_machine_map", lambda path: catalogue)
    monkeypatch.setattr(signal_map_module, "load_signal_map", lambda path: reference)

    result = CliRunner().invoke(
        maps_cli.maps, ["score-handoff", "jt-60sa", str(HANDOFF_FIXTURE)]
    )

    assert result.exit_code == 0, result.output
    header = next(
        line for line in result.output.splitlines() if line.startswith("IDS")
    )
    assert header.split() == [
        "IDS",
        "exported",
        "imported",
        "agreeing",
        "conflicting",
        "chain_differs",
        "cross_structure",
        "sign_unscored",
        "unplaced",
        "refused",
    ]
    rows = {
        line.split()[0]: line.split()
        for line in result.output.splitlines()
        if not line.startswith("IDS") and not line.startswith("refused ")
    }
    # columns: IDS exported imported agreeing conflicting sign_unscored ...
    assert rows["magnetics"][4] == "1"
    assert rows["TOTAL"][4] == "1"


def test_score_handoff_prints_a_chain_differs_column(tmp_path, monkeypatch):
    """A draft on a chain the hand-built rule declares an alternate is no conflict."""
    from types import SimpleNamespace

    from imas_alambic.machine_map import SensorIdentityRule
    from imas_alambic.signal_map import AlternateSource
    from imas_ambix.data.generated_mapping_import import import_generated_mappings

    root = tmp_path / "maps"
    machine_dir = root / "jt-60sa"
    (machine_dir / "maps").mkdir(parents=True)
    (machine_dir / "machine_map.json").write_text("{}")
    (machine_dir / "maps" / "magnetics.json").write_text("{}")
    catalogue = SimpleNamespace(
        source_cocos=1,
        sensor_identity_rules=(
            SensorIdentityRule(
                name="identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="description names",
            ),
        ),
    )
    members = {
        "magnetics/b_field_pol_probe": ("011", "010"),
        "pf_active/coil": ("02", "01"),
    }
    document = json.loads(HANDOFF_FIXTURE.read_text())
    reference = import_generated_mappings(document, catalogue, members).maps[0]
    first, second = reference.signals
    # The hand-built rule serves the target through a different array and
    # declares the draft's array an alternate: the two chains measure the same
    # quantity, so the draft is a chain choice rather than a conflict.
    reference = dataclasses.replace(
        reference,
        signals=(
            first,
            dataclasses.replace(
                second,
                source_array="other",
                alternate_sources=(
                    AlternateSource(
                        source_group="MDAC",
                        source_array="magPbTC11",
                        evidence="the alternate chain measures the same field",
                    ),
                ),
            ),
        ),
    )
    import imas_alambic.machine_map as machine_map_module
    import imas_alambic.signal_map as signal_map_module

    monkeypatch.setattr(maps_cli, "MAPS_DIR", root)
    monkeypatch.setattr(
        maps_cli, "_description_members", lambda *args, **kwargs: members
    )
    monkeypatch.setattr(machine_map_module, "load_machine_map", lambda path: catalogue)
    monkeypatch.setattr(signal_map_module, "load_signal_map", lambda path: reference)

    result = CliRunner().invoke(
        maps_cli.maps, ["score-handoff", "jt-60sa", str(HANDOFF_FIXTURE)]
    )

    assert result.exit_code == 0, result.output
    rows = {
        line.split()[0]: line.split()
        for line in result.output.splitlines()
        if not line.startswith("IDS") and not line.startswith("refused ")
    }
    # columns: IDS exported imported agreeing conflicting chain_differs ...
    assert rows["magnetics"][4] == "0"
    assert rows["magnetics"][5] == "1"
    assert rows["TOTAL"][4] == "0"
    assert rows["TOTAL"][5] == "1"


def test_real_pf_active_map_rescores_coil_chain_alternates(tmp_path):
    """The declared JT-60SA alternates put the coil chain pairs under chain_differs.

    The HiTe and LKAT chains measure the same coil current, so a draft bound to
    the chain the hand-built rule names as an alternate is a chain choice: it is
    counted under chain_differs with both arrays, not under conflicting.  The
    declarations live in the map file, so dropping them moves the six coil pairs
    back under conflicting -- the mutation this test refuses.
    """
    from types import SimpleNamespace

    from imas_alambic.machine_map import SensorIdentityRule
    from imas_alambic.signal_map import load_signal_map
    from imas_ambix.data.handoff_score import score_handoff
    from imas_ambix.data.paths import JT60SA_MAP_DIR

    map_path = JT60SA_MAP_DIR / "maps" / "pf_active.json"
    if not map_path.is_file():
        import pytest

        pytest.skip("the JT-60SA development map bundle is not present")

    reference = load_signal_map(map_path)
    catalogue = SimpleNamespace(
        source_cocos=17,
        sensor_identity_rules=(
            SensorIdentityRule(
                name="identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="description names",
            ),
        ),
    )
    members = {
        "pf_active/coil": (
            "CS1",
            "CS2",
            "CS3",
            "CS4",
            "EF1",
            "EF2",
            "EF3",
            "EF4",
            "EF5",
            "EF6",
        )
    }
    # The draft binds EF1-EF5 on the HiTe chain and EF6 on the LKAT chain, the
    # chains counter to the hand-built rules, which serve the LKAT chain for
    # EF1-EF5 and the HiTe chain for EF6.
    chain_of = {
        "EF1": "curEF1HiTe",
        "EF2": "curEF2HiTe",
        "EF3": "curEF3HiTe",
        "EF4": "curEF4HiTe",
        "EF5": "curEF5HiTe",
        "EF6": "curEF6LKAT",
    }
    signals = [
        {
            "signal_id": f"jt-60sa:general/mmsys_{array.lower()}",
            "source_id": "jt-60sa:pf_active:coil_current",
            "data_source": "edas",
            "source_group": "MMSYS",
            "source_array": array,
            "member_identifier": member,
            "source_property": "value",
            "target_path": "pf_active/coil/current/data",
            "transform_expression": None,
            "source_units": "A",
            "target_units": "A",
            "cocos_label": "one_like",
            "confidence": 0.8,
            "evidence": "MMSYS coil current channel",
        }
        for member, array in chain_of.items()
    ]
    document = {
        "format": "imas-codex-mapping-handoff",
        "format_version": 1,
        "facility": "jt-60sa",
        "dd_version": "4.1.1",
        "exported_at": "2026-10-09T00:00:00Z",
        "ids": [
            {
                "ids_name": "pf_active",
                "mapping_id": "jt-60sa:pf_active",
                "status": "generated",
                "signals": signals,
                "unexpanded": [],
            }
        ],
    }
    score = score_handoff(document, catalogue, members, [reference])
    coil = score["by_structure"]["pf_active/coil"]
    assert coil["conflicting"] == 0
    assert {entry["imported_source_array"] for entry in coil["chain_differs"]} == set(
        chain_of.values()
    )
    assert all(
        entry["hand_built_source_array"].endswith(("LKAT", "HiTe"))
        and entry["hand_built_source_group"] == "MMSYS"
        for entry in coil["chain_differs"]
    )


def test_score_handoff_prints_one_row_per_structure_under_its_ids(
    tmp_path, monkeypatch
):
    """The printed table carries a row per structure beneath its IDS row."""
    from types import SimpleNamespace

    from imas_alambic.machine_map import SensorIdentityRule
    from imas_ambix.data.generated_mapping_import import import_generated_mappings

    root = tmp_path / "maps"
    machine_dir = root / "jt-60sa"
    (machine_dir / "maps").mkdir(parents=True)
    (machine_dir / "machine_map.json").write_text("{}")
    (machine_dir / "maps" / "magnetics.json").write_text("{}")
    catalogue = SimpleNamespace(
        source_cocos=1,
        sensor_identity_rules=(
            SensorIdentityRule(
                name="identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="description names",
            ),
        ),
    )
    members = {
        "magnetics/b_field_pol_probe": ("011", "010"),
        "pf_active/coil": ("02", "01"),
    }
    document = json.loads(HANDOFF_FIXTURE.read_text())
    reference = import_generated_mappings(document, catalogue, members).maps[0]
    import imas_alambic.machine_map as machine_map_module
    import imas_alambic.signal_map as signal_map_module

    monkeypatch.setattr(maps_cli, "MAPS_DIR", root)
    monkeypatch.setattr(
        maps_cli, "_description_members", lambda *args, **kwargs: members
    )
    monkeypatch.setattr(machine_map_module, "load_machine_map", lambda path: catalogue)
    monkeypatch.setattr(signal_map_module, "load_signal_map", lambda path: reference)

    result = CliRunner().invoke(
        maps_cli.maps, ["score-handoff", "jt-60sa", str(HANDOFF_FIXTURE)]
    )

    assert result.exit_code == 0, result.output
    rows = [
        line.split()
        for line in result.output.splitlines()
        if not line.startswith("IDS") and not line.startswith("refused ")
    ]
    # magnetics splits into two structures, each printed under it before the
    # next IDS; pf_active carries its own structure row.
    assert [row[0] for row in rows] == [
        "magnetics",
        "magnetics/b_field_pol_probe",
        "magnetics/flux_loop",
        "pf_active",
        "pf_active/coil",
        "TOTAL",
    ]
    numbers = {row[0]: row[1:3] for row in rows}
    assert numbers["magnetics"] == ["3", "2"]
    assert numbers["magnetics/b_field_pol_probe"] == ["2", "2"]
    assert numbers["magnetics/flux_loop"] == ["1", "0"]


def test_scoring_can_leave_an_ids_without_description_members_unplaced(tmp_path):
    from types import SimpleNamespace

    from imas_alambic.machine_map import SensorIdentityRule
    from imas_ambix.data.handoff_score import score_handoff

    catalogue = SimpleNamespace(
        dd_version="4.1.1",
        source_cocos=None,
        maps=(SimpleNamespace(name="phase"),),
        sensor_identity_rules=(
            SensorIdentityRule(
                name="identity",
                case_rule="case-fold",
                numeric_token_rule="integer-value",
                evidence="description names",
            ),
        ),
        description_store_root_path=lambda **kwargs: tmp_path,
    )
    document = {
        "format": "imas-codex-mapping-handoff",
        "format_version": 1,
        "facility": "jt-60sa",
        "dd_version": "4.1.1",
        "exported_at": "2026-10-09T11:22:02Z",
        "ids": [
            {
                "ids_name": "gas_injection",
                "mapping_id": "jt-60sa:gas_injection",
                "status": "generated",
                "signals": [
                    {
                        "signal_id": "jt-60sa:general/n2gas_flwinleta",
                        "source_id": "jt-60sa:gas_injection:pipe",
                        "data_source": "edas",
                        "source_group": "GAS",
                        "source_array": "gas1",
                        "member_identifier": "1",
                        "source_property": "value",
                        "target_path": "gas_injection/pipe/flow_rate/data",
                        "transform_expression": None,
                        "source_units": "Pa m3/s",
                        "target_units": "Pa m3/s",
                        "cocos_label": None,
                        "confidence": 0.7,
                        "evidence": "gas pipe flow",
                    }
                ],
                "unexpanded": [],
            }
        ],
    }

    members = maps_cli._description_members(
        catalogue, tmp_path, document, allow_missing=True
    )
    assert members == {}

    score = score_handoff(document, catalogue, members, [])
    row = score["by_ids"]["gas_injection"]
    assert row["agreeing"] == "no reference"
    assert row["unplaced"] == 1
    assert row["refused"] == 0
    assert row["unplaced_reasons"] == {
        "no machine-description store for gas_injection": 1,
    }


class FakeRegistry:
    """A registry whose tags, annotations and pushes live in memory."""

    def __init__(
        self, tags: list[str] | None = None, *, fail_push: bool = False
    ) -> None:
        self.raw_tags = list(tags or [])
        self.fail_push = fail_push
        self.annotations: dict[str, dict[str, str]] = {}
        self.pushes: list[tuple[str, dict[str, str]]] = []

    @property
    def tags(self) -> list[str]:
        return list(self.raw_tags)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def list_tags(registry, token, pkg_name):
            return self.tags

        def fetch_annotations(registry, tags, pkg_name):
            return {tag: dict(self.annotations.get(tag, {})) for tag in tags}

        def push_tree(ref, staging, annotations, token):
            if self.fail_push:
                raise RuntimeError("registry refused the push")
            self.pushes.append((ref, dict(annotations)))
            tag = annotations["org.opencontainers.image.version"]
            self.raw_tags.append(tag)
            self.annotations[tag] = dict(annotations)

        def pull_tree(ref, dest, token):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "pulled-from").write_text(ref)

        monkeypatch.setattr(maps_cli, "require_oras", lambda: None)
        monkeypatch.setattr(maps_cli, "list_registry_tags", list_tags)
        monkeypatch.setattr(maps_cli, "fetch_tag_annotations", fetch_annotations)
        monkeypatch.setattr(maps_cli, "push_tree", push_tree)
        monkeypatch.setattr(maps_cli, "pull_tree", pull_tree)


def write_tree(root: Path, *, version: str = "2026.10.07") -> None:
    """Create a minimal published tree with development inputs beside it."""
    (root / "maps").mkdir(parents=True, exist_ok=True)
    (root / "maps" / "magnetics.json").write_text('{"loops": 53}')
    (root / "machine_description").mkdir(parents=True, exist_ok=True)
    (root / "machine_description" / "store.json").write_text("{}")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "name": "imas-alambic-jt60sa",
                "version": version,
                "machine": "jt-60sa",
                "store_roots": {"description": "machine_description"},
            },
            indent=2,
        )
    )
    for excluded in ("source", "superseded"):
        (root / excluded).mkdir(parents=True, exist_ok=True)
        (root / excluded / "deck.txt").write_text("development input")


def test_draft_signals_replace_covered_ids_and_keep_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordered description gives indices; covered hand-built rules disappear."""
    from imas_alambic.machine_map import load_machine_map
    from imas_alambic.signal_map import load_signal_map
    from imas_ambix.data.tokamap_export import export_tokamap_directory

    machine_dir = maps_cli.MAPS_DIR / "jt-60sa"
    if not (machine_dir / "machine_map.json").is_file():
        import pytest

        pytest.skip("JT-60SA development bundle is unavailable")
    catalog = load_machine_map(machine_dir / "machine_map.json")
    hand_built = [
        load_signal_map(path) for path in sorted((machine_dir / "maps").glob("*.json"))
    ]
    reference = export_tokamap_directory(
        catalog,
        hand_built,
        directory=tmp_path / "reference",
        withhold_undeclared_cocos=True,
    )
    import imas_alambic.signal_map as signal_map_module

    original_loader = signal_map_module.load_signal_map

    def contaminated_loader(path: Path):
        loaded = original_loader(path)
        if loaded.system not in {"magnetics", "pf_active"}:
            return loaded
        covered_paths = {
            "magnetics/b_field_pol_probe/field/data",
            "pf_active/coil/current/data",
        }
        return dataclasses.replace(
            loaded,
            signals=tuple(
                dataclasses.replace(
                    rule,
                    source_array=f"hand-built-{rule.source_array}",
                    target_index=500 + (rule.target_index or 0),
                )
                if rule.target_path in covered_paths
                else rule
                for rule in loaded.signals
            ),
        )

    monkeypatch.setattr(signal_map_module, "load_signal_map", contaminated_loader)
    draft_dir = tmp_path / "draft"
    result = CliRunner().invoke(
        maps_cli.maps,
        [
            "tokamap",
            "jt-60sa",
            "--out",
            str(draft_dir),
            "--draft-signals",
            str(HANDOFF_FIXTURE),
            "--withhold-undeclared-cocos",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "unresolved magnetics" in result.output
    assert "No FacilitySignal member is linked" in result.output
    assert "(50 machine description, 20 signal)" in result.output

    expected = {
        "magnetics": {
            "b_field_pol_probe[9]/field/data": "magPbTC10",
            "b_field_pol_probe[10]/field/data": "magPbTC11",
        },
        "pf_active": {
            "coil[0]/current/data": "curCS1LKAT",
            "coil[1]/current/data": "curCS2LKAT",
        },
    }
    for ids_name, signal_sources in expected.items():
        for partition in reference.partitions:
            path = Path(ids_name) / str(partition) / "mappings.json"
            written = json.loads((draft_dir / path).read_text())
            original = json.loads((reference.directory / path).read_text())
            drafts = {
                key: value
                for key, value in written.items()
                if "validation_state=draft" in value.get("comment", "")
            }
            assert {
                key: value["args"]["source_array"] for key, value in drafts.items()
            } == signal_sources
            assert all(value["scale"] == 1.0 for value in drafts.values())
            assert set(written) == {
                key
                for key, value in original.items()
                if "validation_state=" not in value.get("comment", "")
            } | set(signal_sources)
            for key, value in original.items():
                if "validation_state=" not in value.get("comment", ""):
                    assert written[key] == value


def test_draft_signals_list_a_pending_cocos_rule_as_withheld(tmp_path: Path) -> None:
    """A COCOS-dependent hand-off row is withheld rather than given a sign."""
    import pytest

    machine_dir = maps_cli.MAPS_DIR / "jt-60sa"
    if not (machine_dir / "machine_map.json").is_file():
        pytest.skip("JT-60SA development bundle is unavailable")
    document = json.loads(HANDOFF_FIXTURE.read_text(encoding="utf-8"))
    row = document["ids"][0]["signals"][0]
    row["cocos_label"] = "ip_like"
    row["cocos_label_source"] = "inferred_forward"
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps(document), encoding="utf-8")

    result = CliRunner().invoke(
        maps_cli.maps,
        [
            "tokamap",
            "jt-60sa",
            "--out",
            str(tmp_path / "draft"),
            "--draft-signals",
            str(handoff),
            "--withhold-undeclared-cocos",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "withheld magnetics" in result.output
    assert "ip_like" in result.output
    assert "inferred_forward" in result.output


# --- version state machine -------------------------------------------------


def test_semver_tags_sort_newest_first_with_candidate_below_release() -> None:
    raw = ["2026.10.07", "v0.1.0", "v0.2.0-rc1", "v0.2.0", "v0.2.0-rc2", "latest"]
    assert semver_tags(raw) == ["v0.2.0", "v0.2.0-rc2", "v0.2.0-rc1", "v0.1.0"]


def test_semver_tags_discard_a_legacy_date_tag() -> None:
    assert "2026.10.07" not in semver_tags(["2026.10.07", "v0.1.0"])


def test_baseline_is_read_by_both_queries() -> None:
    assert latest_tag([]) == BASELINE_VERSION
    assert latest_stable_tag([]) == BASELINE_VERSION


def test_baseline_is_read_when_no_stable_tag_exists() -> None:
    assert latest_tag(["v0.1.0-rc1"]) == "v0.1.0-rc1"
    assert latest_stable_tag(["v0.1.0-rc1"]) == BASELINE_VERSION


def test_stable_bump_starts_a_candidate_series() -> None:
    assert compute_next_version("minor", final=False, tags=["v0.1.0"]) == "v0.2.0-rc1"


def test_stable_bump_final_releases_directly() -> None:
    assert compute_next_version("patch", final=True, tags=["v0.1.0"]) == "v0.1.1"


def test_automatic_call_from_stable_starts_a_patch_candidate() -> None:
    assert compute_next_version(None, final=False, tags=["v0.1.0"]) == "v0.1.1-rc1"


def test_automatic_call_on_a_new_package_publishes_a_patch_candidate() -> None:
    assert compute_next_version(None, final=False, tags=[]) == "v0.0.1-rc1"


def test_candidate_plain_call_increments_the_candidate() -> None:
    tags = ["v0.2.0-rc1", "v0.1.0"]
    assert compute_next_version(None, final=False, tags=tags) == "v0.2.0-rc2"


def test_candidate_final_promotes_to_stable() -> None:
    tags = ["v0.2.0-rc1", "v0.1.0"]
    assert compute_next_version(None, final=True, tags=tags) == "v0.2.0"


def test_candidate_bump_abandons_it_from_the_last_stable() -> None:
    tags = ["v0.6.0-rc3", "v0.5.0"]
    assert compute_next_version("minor", final=False, tags=tags) == "v0.6.0-rc1"


def test_candidate_bump_final_releases_directly_from_stable() -> None:
    tags = ["v0.6.0-rc3", "v0.5.0"]
    assert compute_next_version("major", final=True, tags=tags) == "v1.0.0"


def test_candidate_bump_with_no_stable_tag_bumps_the_baseline() -> None:
    candidates = compute_next_version("minor", final=False, tags=["v0.2.0-rc1"])
    assert candidates == "v0.1.0-rc1"


def test_candidate_allocation_skips_a_tag_the_package_carries() -> None:
    tags = ["v0.2.0-rc2", "v0.2.0-rc1"]
    assert compute_next_version(None, final=False, tags=tags) == "v0.2.0-rc3"


# --- release, pull and status against the stubbed registry -----------------


def test_tree_digest_ignores_the_bundle_version_field(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_tree(a, version="2026.10.07")
    write_tree(b, version="v0.2.0-rc1")
    assert tree_digest(a) == tree_digest(b)


def test_tree_digest_changes_when_a_map_file_changes(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_tree(a)
    write_tree(b)
    (b / "maps" / "magnetics.json").write_text('{"loops": 54}')
    assert tree_digest(a) != tree_digest(b)


def test_tree_digest_ignores_development_inputs(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_tree(a)
    write_tree(b)
    (b / "source" / "deck.txt").write_text("changed deck")
    (b / "superseded" / "deck.txt").write_text("changed superseded")
    assert tree_digest(a) == tree_digest(b)


def test_a_stray_bundle_temp_is_neither_hashed_nor_staged(tmp_path: Path) -> None:
    root = tmp_path / "jt-60sa"
    write_tree(root)
    baseline = tree_digest(root)
    # The atomic descriptor write leaves this name behind if the process dies
    # between its mkstemp and its os.replace.
    stray = root / "bundle.json.a1b2c3d4.tmp"
    stray.write_text('{"version": "v9.9.9"}')

    assert tree_digest(root) == baseline

    dest = tmp_path / "staging"
    stage_bundle(root, dest, "v0.2.0-rc1")
    assert not (dest / stray.name).exists()
    assert json.loads((dest / "bundle.json").read_text())["version"] == "v0.2.0-rc1"


def test_release_pushes_the_next_version_with_its_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0"])
    registry.install(monkeypatch)

    result = CliRunner().invoke(
        maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"]
    )

    assert result.exit_code == 0, result.output
    assert len(registry.pushes) == 1
    ref, annotations = registry.pushes[0]
    assert ref.endswith("/imas-alambic-jt60sa:v0.2.0-rc1")
    assert annotations["io.imas-ambix.machine"] == "jt-60sa"
    assert annotations["org.opencontainers.image.version"] == "v0.2.0-rc1"
    # A digest fixed for the fixture tree rather than recomputed on it here: a
    # recomputation would agree with any digest rule, so it could not catch a
    # change to how tree_digest hashes the published files.
    assert (
        annotations["io.imas-ambix.tree-digest"]
        == "df450b7015b8c06ded34cc5df6dcc5c2a91e1658e3018661af3bb96f83af53e0"
    )
    assert "v0.2.0-rc1" in (maps_dir / "jt-60sa" / "bundle.json").read_text()


def test_release_leaves_the_bundle_version_when_the_push_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    machine_dir = maps_dir / "jt-60sa"
    write_tree(machine_dir, version="2026.10.07")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0"], fail_push=True)
    registry.install(monkeypatch)
    before = json.loads((machine_dir / "bundle.json").read_text())

    result = CliRunner().invoke(
        maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"]
    )

    assert result.exit_code != 0
    after = json.loads((machine_dir / "bundle.json").read_text())
    assert registry.pushes == []
    assert after["version"] == before["version"] == "2026.10.07"


def test_release_leaves_the_local_bundle_unpublished_while_the_push_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    machine_dir = maps_dir / "jt-60sa"
    write_tree(machine_dir, version="2026.10.07")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0"])
    registry.install(monkeypatch)
    staged = maps_cli.push_tree
    seen: list[str] = []

    def observing_push(ref, staging, annotations, token):
        seen.append(json.loads((machine_dir / "bundle.json").read_text())["version"])
        staged(ref, staging, annotations, token)

    monkeypatch.setattr(maps_cli, "push_tree", observing_push)

    result = CliRunner().invoke(
        maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"]
    )

    assert result.exit_code == 0, result.output
    # The local descriptor still names the deployed version while the push runs,
    # so a reader concurrent with the push cannot see the unpublished one.
    assert seen == ["2026.10.07"], seen
    after = json.loads((machine_dir / "bundle.json").read_text())
    assert after["version"] == "v0.2.0-rc1"


def test_release_leaves_the_bundle_bytes_intact_when_the_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    machine_dir = maps_dir / "jt-60sa"
    write_tree(machine_dir, version="2026.10.07")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0"])
    registry.install(monkeypatch)
    original = (machine_dir / "bundle.json").read_bytes()

    def failing_replace(src, dst):
        raise OSError("cannot move the descriptor into place")

    monkeypatch.setattr(maps_cli.os, "replace", failing_replace)

    result = CliRunner().invoke(
        maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"]
    )

    assert result.exit_code != 0
    assert (machine_dir / "bundle.json").read_bytes() == original


def test_release_preserves_the_bundle_file_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    machine_dir = maps_dir / "jt-60sa"
    write_tree(machine_dir, version="2026.10.07")
    bundle = machine_dir / "bundle.json"
    os.chmod(bundle, 0o644)
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0"])
    registry.install(monkeypatch)

    result = CliRunner().invoke(
        maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"]
    )

    assert result.exit_code == 0, result.output
    # mkstemp's 0600 must not become the published descriptor's mode.
    assert (bundle.stat().st_mode & 0o777) == 0o644


def test_release_chmods_only_permission_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    machine_dir = maps_dir / "jt-60sa"
    write_tree(machine_dir, version="2026.10.07")
    os.chmod(machine_dir / "bundle.json", 0o640)
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0"])
    registry.install(monkeypatch)
    modes: list[int] = []
    real_chmod = os.chmod

    def recording_chmod(path, mode, **kwargs):
        modes.append(mode)
        real_chmod(path, mode, **kwargs)

    monkeypatch.setattr(maps_cli.os, "chmod", recording_chmod)

    result = CliRunner().invoke(
        maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"]
    )

    assert result.exit_code == 0, result.output
    # Passed the raw st_mode, a chmod argument carries S_IFREG (0o100000); the
    # fix passes the permission bits alone, so no type bits reach os.chmod.
    assert modes, "no chmod call was observed"
    assert [mode for mode in modes if mode & ~0o7777] == [], modes


def test_release_skips_the_push_when_the_tree_digest_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0"])
    registry.install(monkeypatch)
    runner = CliRunner()

    first = runner.invoke(maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"])
    second = runner.invoke(maps_cli.maps, ["release", "jt-60sa"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert len(registry.pushes) == 1, registry.pushes
    assert "nothing to release" in second.output


def test_release_creates_no_git_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["v0.1.0"])
    registry.install(monkeypatch)
    commands: list[list[str]] = []

    def recording_run(command, *args, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="0" * 40, stderr="")

    monkeypatch.setattr(maps_cli.subprocess, "run", recording_run)
    result = CliRunner().invoke(
        maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"]
    )

    assert result.exit_code == 0, result.output
    # The recorder is shown to see the git call the module does make, so its
    # silence on ``git tag`` is an observation and not an empty list.
    assert any(command[:2] == ["git", "rev-parse"] for command in commands)
    assert not any(command[:2] == ["git", "tag"] for command in commands), commands


def test_pull_defaults_to_the_latest_stable_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0", "v0.2.0-rc1"])
    registry.install(monkeypatch)
    dest = tmp_path / "out"

    result = CliRunner().invoke(maps_cli.maps, ["pull", "jt-60sa", "--dest", str(dest)])

    assert result.exit_code == 0, result.output
    assert (dest / "pulled-from").read_text().endswith(":v0.1.0")


def test_pull_fetches_an_explicit_version_into_a_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["v0.1.0", "v0.2.0-rc1"])
    registry.install(monkeypatch)
    dest = tmp_path / "out"

    result = CliRunner().invoke(
        maps_cli.maps,
        ["pull", "jt-60sa", "--version", "v0.2.0-rc1", "--dest", str(dest)],
    )

    assert result.exit_code == 0, result.output
    assert (dest / "pulled-from").read_text().endswith(":v0.2.0-rc1")


def test_pull_refuses_a_local_tree_matching_no_released_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    (maps_dir / "jt-60sa" / "maps" / "magnetics.json").write_text('{"loops": 54}')
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["v0.1.0"])
    registry.install(monkeypatch)
    dest = tmp_path / "out"
    write_tree(dest)

    result = CliRunner().invoke(
        maps_cli.maps, ["pull", "jt-60sa", "--version", "v0.1.0", "--dest", str(dest)]
    )

    assert result.exit_code != 0
    assert "unreleased map edits" in result.output
    assert not (dest / "pulled-from").exists()


def test_pull_force_overwrites_and_prints_what_it_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["v0.1.0"])
    registry.install(monkeypatch)
    dest = tmp_path / "out"
    write_tree(dest)

    result = CliRunner().invoke(
        maps_cli.maps,
        ["pull", "jt-60sa", "--version", "v0.1.0", "--dest", str(dest), "--force"],
    )

    assert result.exit_code == 0, result.output
    assert "bundle.json" in result.output
    assert (dest / "pulled-from").exists()


def test_pull_refuses_a_tag_the_package_does_not_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["v0.1.0"])
    registry.install(monkeypatch)

    result = CliRunner().invoke(
        maps_cli.maps,
        ["pull", "jt-60sa", "--version", "v9.9.9", "--dest", str(tmp_path / "o")],
    )

    assert result.exit_code != 0
    assert "carries no tag" in result.output


def test_status_reports_candidate_state_and_the_matching_local_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    machine_dir = maps_dir / "jt-60sa"
    write_tree(machine_dir)
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["v0.1.0", "v0.2.0-rc1"])
    registry.annotations["v0.2.0-rc1"] = {
        "io.imas-ambix.tree-digest": tree_digest(machine_dir)
    }
    registry.install(monkeypatch)

    result = CliRunner().invoke(maps_cli.maps, ["status", "jt-60sa"])

    assert result.exit_code == 0, result.output
    assert "candidate" in result.output
    assert "v0.1.0" in result.output
    assert "v0.2.0-rc1" in result.output
    assert "matches v0.2.0-rc1" in result.output
