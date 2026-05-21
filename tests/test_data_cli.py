"""Smoke tests for ``ambix data`` subcommands.

Each subcommand is exercised via :class:`click.testing.CliRunner` with:
- ``--help`` text verification (exit 0 + key option/name strings)
- required-arg error paths (exit != 0 for UsageError)
- a happy-path call with mocked dependencies (no network, no /work filesystem)

Patching strategy
-----------------
``imas_ambix.data.paths`` module-level constants (MIRROR_ROOT, LEVEL1_DIR,
LEVEL2_DIR, MANIFEST_DIR, PROBE_DIR) are monkeypatched on the ``cli``
module, which imports them at module load time.  For functions called
*inside* the CLI commands we patch the heavy-dependency callables directly
on the modules where they live.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from imas_ambix.data.cli import data

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def make_synthetic_shot_zarr(
    path: Path,
    group_name: str,
    *,
    t: int = 16,
    h: int = 32,
    w: int = 32,
    single_channel: bool = True,
) -> Path:
    """Create a tiny Zarr shot with one group for CLI smoke testing.

    Dimension names are included so xarray can open the arrays without
    raising "Zarr object is missing the `dimension_names` metadata".
    """
    import zarr

    shot_path = path
    shot_path.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(shot_path), mode="w")
    sub = g.create_group(group_name)
    if single_channel:
        sub.create_array(
            "data",
            data=np.zeros((t, h, w), dtype=np.uint16),
            dimension_names=["time", "y", "x"],
        )
    else:
        sub.create_array(
            "data",
            data=np.zeros((t, h, w, 3), dtype=np.uint16),
            dimension_names=["time", "y", "x", "c"],
        )
    sub.create_array(
        "time",
        data=np.arange(t, dtype=np.float64) * 0.01,
        dimension_names=["time"],
    )
    return shot_path


def _make_probe_report() -> object:
    """Build a minimal :class:`ProbeReport` for use as a mock return value."""
    from imas_ambix.data.probe import ProbeReport

    return ProbeReport(
        tier="level2",
        n_shots_in_index=100,
        n_shots_in_tier=100,
        sample_size=3,
        sustained_throughput_mbps=250.0,
        median_shot_size_mb=50.0,
        p95_shot_size_mb=200.0,
        extrapolated_total_size_tb=0.5,
        group_coverage={"magnetics": 3},
        camera_coverage_fraction=0.0,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        notes=[],
    )


def _make_tiny_manifest(tmp_path: Path, tier: str = "level2") -> Path:
    """Write a minimal ShotManifest JSON to tmp_path/manifest.json."""
    payload = {
        "tier": tier,
        "shot_ids": [30001, 30002],
        "groups": [],
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "source": "https://mastapp.site/parquet/level2/shots",
        "filter_description": "2 shots at level2",
        "total_in_index": 100,
    }
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def test_probe_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["probe", "--help"])
    assert result.exit_code == 0
    assert "probe" in result.output
    assert "--tier" in result.output
    assert "--sample-size" in result.output


def test_probe_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """probe calls run_probe and renders the result table."""
    report = _make_probe_report()

    import imas_ambix.data.cli as cli_mod

    monkeypatch.setattr(cli_mod, "PROBE_DIR", tmp_path / "probe")

    runner = CliRunner()
    with patch("imas_ambix.data.probe.run_probe", return_value=report):
        result = runner.invoke(
            data,
            [
                "probe",
                "--tier",
                "level2",
                "--sample-size",
                "3",
                "--output-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["inventory", "--help"])
    assert result.exit_code == 0
    assert "inventory" in result.output
    assert "--tier" in result.output
    assert "--sample-size" in result.output


def test_inventory_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """inventory with --shot-ids skips load_index + sample_shots."""
    fake_inv = {30001: ("magnetics", "equilibrium"), 30002: ("magnetics",)}
    fake_coverage = {"magnetics": 2, "equilibrium": 1}

    with (
        patch(
            "imas_ambix.data.manifest.inventory_groups",
            return_value=fake_inv,
        ),
        patch(
            "imas_ambix.data.manifest.group_coverage",
            return_value=fake_coverage,
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["inventory", "--shot-ids", "30001,30002", "--tier", "level2"],
        )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def test_manifest_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["manifest", "--help"])
    assert result.exit_code == 0
    assert "manifest" in result.output
    assert "--tier" in result.output
    assert "--emit-ids" in result.output


def test_manifest_happy_path_emit_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """manifest --emit-ids prints shot IDs (no index fetch needed via mock)."""
    import pandas as pd

    fake_df = pd.DataFrame({"shot_id": [30001, 30002, 30003]})

    with patch("imas_ambix.data.manifest.load_index", return_value=fake_df):
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["manifest", "--tier", "level2", "--emit-ids"],
        )
    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.strip().splitlines() if ln.strip()]
    assert len(lines) == 3
    assert all(ln.isdigit() for ln in lines)


def test_manifest_happy_path_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """manifest --output writes a valid JSON manifest file."""
    import pandas as pd

    fake_df = pd.DataFrame({"shot_id": [30001, 30002]})
    out_path = tmp_path / "manifest.json"

    with patch("imas_ambix.data.manifest.load_index", return_value=fake_df):
        runner = CliRunner()
        result = runner.invoke(
            data,
            ["manifest", "--tier", "level2", "--output", str(out_path)],
        )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert set(payload["shot_ids"]) == {30001, 30002}
    assert payload["tier"] == "level2"


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def test_targets_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["targets", "--help"])
    assert result.exit_code == 0
    assert "targets" in result.output
    # Click renders the arg name in the usage line
    assert "manifest" in result.output.lower()


def test_targets_missing_required_arg() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["targets"])
    assert result.exit_code != 0


def test_targets_happy_path(tmp_path: Path) -> None:
    """targets reads a manifest JSON and emits s5cmd cp lines."""
    manifest_path = _make_tiny_manifest(tmp_path)

    runner = CliRunner()
    result = runner.invoke(data, ["targets", str(manifest_path)])
    assert result.exit_code == 0, result.output
    assert "cp" in result.output
    assert "30001" in result.output or "30002" in result.output


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def test_download_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["download", "--help"])
    assert result.exit_code == 0
    assert "download" in result.output
    assert "--tier" in result.output
    assert "--manifest" in result.output


def test_download_missing_manifest_fails() -> None:
    """download without --manifest raises UsageError."""
    runner = CliRunner()
    result = runner.invoke(data, ["download", "--tier", "level2"])
    assert result.exit_code != 0
    assert "manifest" in result.output.lower()


def test_download_happy_path_slurm(tmp_path: Path) -> None:
    """download --plan-only prints a SLURM script when --manifest is given."""
    manifest_path = _make_tiny_manifest(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        data,
        [
            "download",
            "--tier",
            "level2",
            "--manifest",
            str(manifest_path),
            "--host",
            "slurm-sun",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "#SBATCH" in result.output
    assert "s5cmd" in result.output


def test_download_happy_path_login(tmp_path: Path) -> None:
    """download --host login prints a plain bash script."""
    manifest_path = _make_tiny_manifest(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        data,
        [
            "download",
            "--tier",
            "level2",
            "--manifest",
            str(manifest_path),
            "--host",
            "login",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "s5cmd" in result.output
    assert "#SBATCH" not in result.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["status", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output
    assert "--tier" in result.output


def test_status_no_mirror_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """status exits 0 and prints a warning when MIRROR_ROOT is missing."""
    import imas_ambix.data.cli as cli_mod

    monkeypatch.setattr(cli_mod, "MIRROR_ROOT", tmp_path / "nonexistent")

    runner = CliRunner()
    result = runner.invoke(data, ["status", "--tier", "level2"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "mirror" in out or "does not exist" in out or "nonexistent" in out


def test_status_happy_path_with_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """status shows mirror progress when MIRROR_ROOT + manifest exist."""
    import imas_ambix.data.cli as cli_mod

    # Build a fake mirror layout: mirror_root/level2/shots/ with two .zarr dirs
    mirror = tmp_path / "mast"
    level2_shots = mirror / "level2" / "shots"
    level2_shots.mkdir(parents=True)
    (level2_shots / "30001.zarr").mkdir()
    (level2_shots / "30002.zarr").mkdir()

    manifests_dir = mirror / "manifests"
    manifests_dir.mkdir()
    payload = {
        "tier": "level2",
        "shot_ids": [30001, 30002, 30003],
        "groups": [],
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "source": "http://example.com",
        "filter_description": "3 shots",
        "total_in_index": 100,
    }
    (manifests_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(cli_mod, "MIRROR_ROOT", mirror)
    monkeypatch.setattr(cli_mod, "LEVEL2_DIR", level2_shots)
    monkeypatch.setattr(cli_mod, "MANIFEST_DIR", manifests_dir)

    runner = CliRunner()
    result = runner.invoke(data, ["status", "--tier", "level2"])
    assert result.exit_code == 0, result.output
    assert "2" in result.output  # 2 local shots found
    assert "3" in result.output  # 3 in manifest


# ---------------------------------------------------------------------------
# du
# ---------------------------------------------------------------------------


def test_du_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["du", "--help"])
    assert result.exit_code == 0
    assert "du" in result.output
    assert "--tier" in result.output
    assert "--sample-size" in result.output


def test_du_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """du with mocked sum_sizes_from_bucket prints the summary table."""
    import pandas as pd

    fake_df = pd.DataFrame({"shot_id": [30001, 30002]})
    fake_sizes = {30001: (1_000_000, 10), 30002: (2_000_000, 20)}

    with (
        patch("imas_ambix.data.manifest.load_index", return_value=fake_df),
        patch(
            "imas_ambix.data.probe.sample_shots",
            return_value=[30001, 30002],
        ),
        patch(
            "imas_ambix.data.manifest.sum_sizes_from_bucket",
            return_value=fake_sizes,
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(data, ["du", "--tier", "level2", "--sample-size", "2"])
    assert result.exit_code == 0, result.output
    # Table should contain byte totals (3 000 000 combined)
    out = result.output
    assert "3,000,000" in out or "3000000" in out or "total bytes" in out.lower()


# ---------------------------------------------------------------------------
# encode-shot
# ---------------------------------------------------------------------------


def test_encode_shot_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["encode-shot", "--help"])
    assert result.exit_code == 0
    assert "encode-shot" in result.output
    assert "--shot" in result.output
    assert "--tokenizer" in result.output


def test_encode_shot_missing_required_arg() -> None:
    """encode-shot without --shot should fail with exit != 0."""
    runner = CliRunner()
    result = runner.invoke(data, ["encode-shot"])
    assert result.exit_code != 0


def test_encode_shot_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """encode-shot writes token Zarr using the placeholder tokenizer."""
    import imas_ambix.data.persist as persist_mod

    # Build a synthetic shot Zarr at the LEVEL1_DIR location
    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "99001.zarr"
    make_synthetic_shot_zarr(shot_zarr, "rbb", t=8, h=32, w=32)

    # Redirect TOKEN_ROOT so the output lands in tmp_path
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    # Patch the LEVEL1_DIR inside the encode_shot_cmd function body
    with patch("imas_ambix.data.paths.LEVEL1_DIR", level1_shots):
        runner = CliRunner()
        result = runner.invoke(
            data,
            [
                "encode-shot",
                "--shot",
                "99001",
                "--camera",
                "rbb",
                "--tokenizer",
                "placeholder",
                "--max-frames",
                "4",
                "--vocab-version",
                "v1",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Saved" in result.output or "saved" in result.output


# ---------------------------------------------------------------------------
# tokens-status
# ---------------------------------------------------------------------------


def test_tokens_status_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["tokens-status", "--help"])
    assert result.exit_code == 0
    assert "tokens-status" in result.output
    assert "--vocab-version" in result.output


def test_tokens_status_no_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """tokens-status exits 0 and warns when TOKEN_ROOT doesn't exist."""
    with patch("imas_ambix.data.paths.TOKEN_ROOT", tmp_path / "nonexistent"):
        runner = CliRunner()
        result = runner.invoke(data, ["tokens-status", "--vocab-version", "v1"])
    assert result.exit_code == 0


def test_tokens_status_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tokens-status lists modalities + counts when token files are present."""
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.tokenizer.base import EncodedFrames

    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    # Write two frame token Zarrs via persist so the directory layout is correct
    fake_enc = EncodedFrames(
        token_ids=np.zeros((2, 4, 4), dtype=np.int32),
        shape=(2, 4, 4),
        tokenizer_name="frames_placeholder_v1",
        metadata={},
    )
    persist_mod.save_frame_tokens(
        shot_id=99001, camera="rbb", encoded=fake_enc, vocab_version="v1"
    )
    persist_mod.save_frame_tokens(
        shot_id=99002, camera="rbb", encoded=fake_enc, vocab_version="v1"
    )

    with patch("imas_ambix.data.paths.TOKEN_ROOT", tmp_path / "tokens"):
        runner = CliRunner()
        result = runner.invoke(data, ["tokens-status", "--vocab-version", "v1"])
    assert result.exit_code == 0, result.output
    # Should show at least one count row
    assert "rbb" in result.output or "frames" in result.output
