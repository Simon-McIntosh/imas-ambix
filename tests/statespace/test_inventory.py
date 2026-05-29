"""Unit tests for imas_ambix.statespace.inventory.

Tests use a small synthetic Zarr directory tree — no GPFS access.
"""

from __future__ import annotations

from pathlib import Path

from imas_ambix.statespace.inventory import (
    InventoryResult,
    _list_shot_groups,
    build_inventory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_corpus(root: Path, shots: dict[int, list[str]]) -> None:
    """Create a minimal fake Zarr corpus directory structure."""
    for shot_id, groups in shots.items():
        shot_dir = root / f"{shot_id}.zarr"
        shot_dir.mkdir(parents=True)
        for group in groups:
            (shot_dir / group).mkdir()


# ---------------------------------------------------------------------------
# Tests for _list_shot_groups
# ---------------------------------------------------------------------------


class TestListShotGroups:
    def test_returns_sorted_groups(self, tmp_path: Path) -> None:
        shot_dir = tmp_path / "12345.zarr"
        shot_dir.mkdir()
        for g in ("xim", "ama", "ane", "efm"):
            (shot_dir / g).mkdir()
        shot_id, groups = _list_shot_groups(shot_dir)
        assert shot_id == 12345
        assert groups == tuple(sorted(["xim", "ama", "ane", "efm"]))

    def test_empty_shot(self, tmp_path: Path) -> None:
        shot_dir = tmp_path / "99999.zarr"
        shot_dir.mkdir()
        shot_id, groups = _list_shot_groups(shot_dir)
        assert shot_id == 99999
        assert groups == ()

    def test_excludes_files(self, tmp_path: Path) -> None:
        """Only directories are returned as groups."""
        shot_dir = tmp_path / "11111.zarr"
        shot_dir.mkdir()
        (shot_dir / "ama").mkdir()
        (shot_dir / ".zmetadata").write_text("{}")
        _, groups = _list_shot_groups(shot_dir)
        assert "ama" in groups
        assert ".zmetadata" not in groups


# ---------------------------------------------------------------------------
# Tests for InventoryResult
# ---------------------------------------------------------------------------


class TestInventoryResult:
    def _make_result(self) -> InventoryResult:
        return InventoryResult(
            shot_groups={
                100: ("ama", "ane", "xim"),
                101: ("ama", "ane"),
                102: ("ama", "xim", "rbb"),
                103: ("rbb",),
            },
            all_groups=["ama", "ane", "rbb", "xim"],
            n_shots=4,
        )

    def test_shots_with_group(self) -> None:
        r = self._make_result()
        assert set(r.shots_with_group("xim")) == {100, 102}
        assert set(r.shots_with_group("rbb")) == {102, 103}
        assert r.shots_with_group("missing") == []

    def test_shots_with_all_groups(self) -> None:
        r = self._make_result()
        # Only shot 100 has both ama and xim
        assert r.shots_with_all_groups("ama", "xim") == [100, 102]
        # Only shot 100 has all three
        assert r.shots_with_all_groups("ama", "ane", "xim") == [100]
        assert r.shots_with_all_groups("rbb", "xim") == [102]

    def test_group_coverage(self) -> None:
        r = self._make_result()
        cov = r.group_coverage()
        assert cov["ama"] == 3  # shots 100, 101, 102
        assert cov["xim"] == 2  # shots 100, 102
        assert cov["rbb"] == 2  # shots 102, 103

    def test_coavailability_matrix_shape(self) -> None:
        r = self._make_result()
        mat = r.coavailability_matrix()
        assert mat.shape == (4, 4)  # 4 shots × 4 groups
        assert mat.dtype == bool

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        r = self._make_result()
        path = tmp_path / "inventory.json"
        r.save(path)
        r2 = InventoryResult.load(path)
        assert r2.n_shots == r.n_shots
        assert r2.all_groups == r.all_groups
        assert r2.shot_groups == r.shot_groups

    def test_save_json_is_compact(self, tmp_path: Path) -> None:
        """Saved JSON should not have unnecessary whitespace."""
        r = self._make_result()
        path = tmp_path / "inventory.json"
        r.save(path)
        raw = path.read_text()
        # Compact separators — no ': ' or ', '
        assert "': '" not in raw and "', '" not in raw


# ---------------------------------------------------------------------------
# Tests for build_inventory
# ---------------------------------------------------------------------------


class TestBuildInventory:
    def test_build_from_fake_corpus(self, tmp_path: Path) -> None:
        """build_inventory should find all shots and groups."""
        shots = {
            10001: ["ama", "ane", "xim"],
            10002: ["ama", "rbb"],
            10003: [],
        }
        _make_fake_corpus(tmp_path, shots)
        result = build_inventory(level1_dir=tmp_path, max_workers=2)
        assert result.n_shots == 3
        assert 10001 in result.shot_groups
        assert set(result.shot_groups[10001]) == {"ama", "ane", "xim"}
        assert result.shot_groups[10002] == ("ama", "rbb")
        assert result.shot_groups[10003] == ()

    def test_build_all_groups_union(self, tmp_path: Path) -> None:
        """all_groups should be the union of all groups across shots."""
        shots = {
            1: ["ama", "xim"],
            2: ["ane", "rbb"],
        }
        _make_fake_corpus(tmp_path, shots)
        result = build_inventory(level1_dir=tmp_path, max_workers=2)
        assert set(result.all_groups) == {"ama", "xim", "ane", "rbb"}
        assert result.all_groups == sorted(result.all_groups)  # sorted

    def test_empty_directory(self, tmp_path: Path) -> None:
        """build_inventory on empty directory should return empty result."""
        result = build_inventory(level1_dir=tmp_path, max_workers=1)
        assert result.n_shots == 0
        assert result.all_groups == []
