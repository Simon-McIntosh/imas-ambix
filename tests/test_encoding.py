"""Tests for imas_ambix.data.encoding (encode helpers) and the
bulk-encode-signals CLI subcommand.

All tests are fully offline — they use synthetic Zarr data in tmp_path,
the PlaceholderFrameTokenizer / UniformQuantizer, and monkeypatched paths.
The open-magvit2 tokenizer is never exercised (no weights on CPU test nodes).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import zarr
from click.testing import CliRunner

from imas_ambix.data.cli import data

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------


def make_frame_zarr(
    base_dir: Path,
    shot_id: int,
    camera: str = "rbb",
    *,
    t: int = 8,
    h: int = 16,
    w: int = 16,
) -> Path:
    """Create a minimal level-1 frame Zarr for shot_id under base_dir."""
    shot_path = base_dir / f"{shot_id}.zarr"
    shot_path.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(shot_path), mode="w")
    cam = g.create_group(camera)
    cam.create_array(
        "data",
        data=np.random.randint(0, 256, (t, h, w), dtype=np.uint16),
        dimension_names=["time", "y", "x"],
    )
    cam.create_array(
        "time",
        data=np.arange(t, dtype=np.float64) * 0.01,
        dimension_names=["time"],
    )
    return shot_path


def make_signal_zarr(
    base_dir: Path,
    shot_id: int,
    group: str = "magnetics",
    *,
    t: int = 20,
    n_channels: int = 3,
) -> Path:
    """Create a minimal level-2 signal Zarr for shot_id under base_dir."""
    shot_path = base_dir / f"{shot_id}.zarr"
    shot_path.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(shot_path), mode="w")
    grp = g.create_group(group)
    time_arr = np.linspace(0.0, 1.0, t)
    grp.create_array("time", data=time_arr, dimension_names=["time"])
    for i in range(n_channels):
        ch_name = f"channel_{i}"
        grp.create_array(
            ch_name,
            data=np.random.randn(t).astype(np.float64),
            dimension_names=["time"],
        )
    return shot_path


# ---------------------------------------------------------------------------
# encode_one_shot_frames
# ---------------------------------------------------------------------------


def test_encode_one_shot_frames_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Placeholder tokenizer: output file exists and report has correct fields."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.encoding import encode_one_shot_frames
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1 = tmp_path / "level1" / "shots"
    make_frame_zarr(level1, 1001, "rbb", t=8, h=16, w=16)

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    report = encode_one_shot_frames(
        1001, "rbb", PlaceholderFrameTokenizer, vocab_version="v1", max_frames=4
    )

    assert report.error is None
    assert report.shot_id == 1001
    assert report.modality == "frames"
    assert report.group_or_camera == "rbb"
    assert report.n_tokens > 0
    assert report.elapsed_s >= 0.0
    assert report.output_path.exists()


def test_encode_one_shot_frames_skip_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encoding is skipped when skip_existing=True and output already exists."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.encoding import encode_one_shot_frames
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1 = tmp_path / "level1" / "shots"
    make_frame_zarr(level1, 1002, "rbb")

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    # First encode
    r1 = encode_one_shot_frames(
        1002, "rbb", PlaceholderFrameTokenizer, vocab_version="v1"
    )
    assert r1.error is None
    assert r1.n_tokens > 0

    # Second encode with skip_existing=True (default overwrite=False)
    r2 = encode_one_shot_frames(
        1002, "rbb", PlaceholderFrameTokenizer, vocab_version="v1", overwrite=False
    )
    assert r2.error is None
    assert r2.n_tokens == 0  # skipped
    assert r2.elapsed_s == 0.0


def test_encode_one_shot_frames_missing_camera_populates_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing camera group → EncodeReport.error populated, no exception raised."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.encoding import encode_one_shot_frames
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1 = tmp_path / "level1" / "shots"
    make_frame_zarr(level1, 1003, "rbb")  # no "rir" group

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    report = encode_one_shot_frames(
        1003, "rir", PlaceholderFrameTokenizer, vocab_version="v1"
    )
    assert report.error is not None
    assert report.n_tokens == 0
    # Should not raise — error is captured in the report


def test_encode_one_shot_frames_missing_shot_populates_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entirely missing shot Zarr → EncodeReport.error, no exception raised."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.encoding import encode_one_shot_frames
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    level1 = tmp_path / "level1" / "shots"
    level1.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(paths_mod, "LEVEL1_DIR", level1)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    report = encode_one_shot_frames(
        9999, "rbb", PlaceholderFrameTokenizer, vocab_version="v1"
    )
    assert report.error is not None
    assert report.n_tokens == 0


# ---------------------------------------------------------------------------
# bulk_encode_signals (UniformQuantizer)
# ---------------------------------------------------------------------------


def test_bulk_encode_signals_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bulk_encode_signals encodes signal shots with UniformQuantizer."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.encoding import bulk_encode_signals
    from imas_ambix.tokenizer.signals import UniformQuantizer

    level2 = tmp_path / "level2" / "shots"
    for sid in [4001, 4002]:
        make_signal_zarr(level2, sid, "magnetics", t=20, n_channels=3)

    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    reports = bulk_encode_signals(
        shot_ids=[4001, 4002],
        group="magnetics",
        tokenizer_factory=UniformQuantizer,
        max_workers=1,
        skip_existing=False,
        vocab_version="v1",
    )

    assert len(reports) == 2
    for r in reports:
        assert r.error is None, f"shot {r.shot_id} failed: {r.error}"
        assert r.n_tokens > 0
        assert r.output_path.exists()


def test_bulk_encode_signals_error_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing signal group → error captured in report, no crash."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod
    from imas_ambix.data.encoding import bulk_encode_signals
    from imas_ambix.tokenizer.signals import UniformQuantizer

    level2 = tmp_path / "level2" / "shots"
    # Create shot with wrong group name
    make_signal_zarr(level2, 5001, "wrong_group", t=10, n_channels=2)

    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    reports = bulk_encode_signals(
        shot_ids=[5001],
        group="magnetics",  # not present
        tokenizer_factory=UniformQuantizer,
        max_workers=1,
        skip_existing=False,
    )

    assert len(reports) == 1
    assert reports[0].error is not None
    assert reports[0].n_tokens == 0


# ---------------------------------------------------------------------------
# CLI smoke — bulk-encode-signals
# ---------------------------------------------------------------------------


def test_bulk_encode_signals_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(data, ["bulk-encode-signals", "--help"])
    assert result.exit_code == 0
    assert "--shot-ids" in result.output
    assert "--tokenizer" in result.output
    assert "--group" in result.output


def test_bulk_encode_signals_cli_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI smoke: bulk-encode-signals with uniform tokenizer on 1 shot."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod

    level2 = tmp_path / "level2" / "shots"
    make_signal_zarr(level2, 7001, "magnetics", t=20, n_channels=3)

    out_json = tmp_path / "sig_report.json"

    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    with patch("imas_ambix.data.paths.LEVEL2_DIR", level2):
        runner = CliRunner()
        result = runner.invoke(
            data,
            [
                "bulk-encode-signals",
                "--shot-ids",
                "7001",
                "--group",
                "magnetics",
                "--tokenizer",
                "uniform",
                "--vocab-version",
                "v1",
                "--workers",
                "1",
                "--no-skip-existing",
                "--output",
                str(out_json),
            ],
        )
    assert result.exit_code == 0, result.output
    assert out_json.exists()
    payload = json.loads(out_json.read_text())
    assert payload["n_ok"] == 1
    assert payload["n_errored"] == 0


# ---------------------------------------------------------------------------
# --from-quality-index filter
# ---------------------------------------------------------------------------


def test_from_quality_index_filters_usable_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from-quality-index reads audit JSON and encodes only usable shots."""
    import imas_ambix.data.paths as paths_mod
    import imas_ambix.data.persist as persist_mod

    level2 = tmp_path / "level2" / "shots"
    for sid in [8001, 8002, 8003]:
        make_signal_zarr(level2, sid, "magnetics", t=20, n_channels=3)

    out_json = tmp_path / "enc_report.json"

    # Build a synthetic audit JSON: only 8001 and 8003 are usable
    audit = {
        "tier": "level2",
        "shot_ids": [8001, 8002, 8003],
        "aggregate": {},
        "per_shot": [
            {
                "shot_id": 8001,
                "quality_flags": {"usable_for_training": True},
                "overall_severity": "info",
            },
            {
                "shot_id": 8002,
                "quality_flags": {"usable_for_training": False},
                "overall_severity": "error",
            },
            {
                "shot_id": 8003,
                "quality_flags": {"usable_for_training": True},
                "overall_severity": "info",
            },
        ],
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2)
    monkeypatch.setattr(persist_mod, "TOKEN_ROOT", tmp_path / "tokens")

    with patch("imas_ambix.data.paths.LEVEL2_DIR", level2):
        runner = CliRunner()
        result = runner.invoke(
            data,
            [
                "bulk-encode-signals",
                "--from-quality-index",
                str(audit_path),
                "--group",
                "magnetics",
                "--tokenizer",
                "uniform",
                "--no-skip-existing",
                "--output",
                str(out_json),
            ],
        )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_json.read_text())
    # Only 8001 and 8003 should have been processed
    assert payload["n_ok"] == 2
    shot_ids_encoded = [e["shot_id"] for e in payload["per_shot"] if e["n_tokens"] > 0]
    assert sorted(shot_ids_encoded) == [8001, 8003]


def test_from_quality_index_empty_produces_no_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--from-quality-index with all non-usable shots produces no encoding."""
    import imas_ambix.data.paths as paths_mod

    level2 = tmp_path / "level2" / "shots"
    level2.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths_mod, "LEVEL2_DIR", level2)

    audit = {
        "per_shot": [
            {
                "shot_id": 9001,
                "quality_flags": {"usable_for_training": False},
            }
        ]
    }
    audit_path = tmp_path / "audit_empty.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        data,
        [
            "bulk-encode-signals",
            "--from-quality-index",
            str(audit_path),
            "--group",
            "magnetics",
            "--tokenizer",
            "uniform",
        ],
    )
    assert result.exit_code == 0
    assert "No shot IDs" in result.output or "no shot ids" in result.output.lower()
