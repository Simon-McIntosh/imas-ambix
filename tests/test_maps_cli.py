"""Contracts for the machine map release CLI.

The version state machine, the unchanged-tree skip and the release
annotations are exercised against a stubbed registry: the tag list, the
manifest annotations and the artifact push are replaced in the module under
test, so a test never reaches GHCR.  The stub's tag list is returned in the
registry's own order, which is not sorted, so the tests also show that the
module sorts it newest first before reading "the latest tag".
"""

from __future__ import annotations

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


def test_draft_signals_replace_covered_ids_and_keep_catalogue(tmp_path: Path) -> None:
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
