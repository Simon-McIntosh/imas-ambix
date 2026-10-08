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
    tree_digest,
)


class FakeRegistry:
    """A registry whose tags, annotations and pushes live in memory."""

    def __init__(self, tags: list[str] | None = None) -> None:
        self.raw_tags = list(tags or [])
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
    assert annotations["io.imas-ambix.tree-digest"] == tree_digest(maps_dir / "jt-60sa")
    assert "v0.2.0-rc1" in (maps_dir / "jt-60sa" / "bundle.json").read_text()


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
    CliRunner().invoke(maps_cli.maps, ["release", "jt-60sa", "--bump", "minor"])
    assert all("git" not in ref for ref, _ in registry.pushes)


def test_pull_defaults_to_the_latest_stable_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    maps_dir = tmp_path / "maps"
    write_tree(maps_dir / "jt-60sa")
    monkeypatch.setattr(maps_cli, "MAPS_DIR", maps_dir)
    registry = FakeRegistry(["2026.10.07", "v0.1.0", "v0.2.0-rc1"])
    registry.install(monkeypatch)
    dest = tmp_path / "out"

    result = CliRunner().invoke(
        maps_cli.maps, ["pull", "jt-60sa", "--dest", str(dest)]
    )

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
