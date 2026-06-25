"""Smoke tests for ``imas-ambix tokenize`` subcommands.

Each subcommand is exercised via :class:`click.testing.CliRunner` with:
- ``--help`` text verification (exit 0 + key option/name strings)
- required-arg error paths (exit != 0 for UsageError)
- a happy-path call with mocked filesystems / tokenizers

Note on open-magvit2
--------------------
The ``frames --tokenizer open-magvit2`` path is **not** tested here because
it requires the GPFS staging dir at
``/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/``.
Add an integration test (marked ``@pytest.mark.gpu`` or
``@pytest.mark.integration``) once the staging dir is in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from click.testing import CliRunner

from imas_ambix.tokenizer.cli import tokenize

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Shared helper
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


def make_synthetic_signal_zarr(path: Path, group_name: str, *, t: int = 64) -> Path:
    """Create a tiny Zarr shot with a 1-D signal group for CLI smoke testing."""
    import xarray as xr

    shot_path = path
    shot_path.mkdir(parents=True, exist_ok=True)
    group_dir = shot_path / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    time = np.linspace(0.0, 1.0, t)
    ds = xr.Dataset(
        {
            "ip": ("time", np.random.default_rng(0).standard_normal(t)),
            "beta": ("time", np.random.default_rng(1).standard_normal(t)),
        },
        coords={"time": time},
    )
    ds.to_zarr(str(group_dir), mode="w")
    return shot_path


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(tokenize, ["registry", "--help"])
    assert result.exit_code == 0
    assert "registry" in result.output


def test_registry_happy_path() -> None:
    """registry prints the vocabulary table including known tokenizer names."""
    runner = CliRunner()
    result = runner.invoke(tokenize, ["registry"])
    assert result.exit_code == 0, result.output
    # Placeholder tokenizer always registers itself
    assert "placeholder" in result.output.lower() or "frames" in result.output.lower()
    assert "total_vocab_size" in result.output


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(tokenize, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "inspect" in result.output
    assert "--shot" in result.output
    assert "--tier" in result.output


def test_inspect_missing_required_arg() -> None:
    """inspect without --shot should fail."""
    runner = CliRunner()
    result = runner.invoke(tokenize, ["inspect"])
    assert result.exit_code != 0


def test_inspect_shot_not_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inspect raises UsageError (exit != 0) when the shot path is absent."""
    import imas_ambix.tokenizer.cli as tok_cli

    monkeypatch.setattr(tok_cli, "LEVEL2_DIR", tmp_path / "shots")

    runner = CliRunner()
    result = runner.invoke(tokenize, ["inspect", "--shot", "99999", "--tier", "level2"])
    assert result.exit_code != 0
    assert "99999" in result.output or "not on disk" in result.output.lower()


def test_inspect_lists_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """inspect without --group lists the groups present in the shot."""
    import imas_ambix.tokenizer.cli as tok_cli

    level2_shots = tmp_path / "level2" / "shots"
    shot_zarr = level2_shots / "30001.zarr"
    make_synthetic_shot_zarr(shot_zarr, "magnetics", t=16, h=32, w=32)

    monkeypatch.setattr(tok_cli, "LEVEL2_DIR", level2_shots)

    runner = CliRunner()
    result = runner.invoke(tokenize, ["inspect", "--shot", "30001", "--tier", "level2"])
    assert result.exit_code == 0, result.output
    assert "magnetics" in result.output


def test_inspect_group_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """inspect --group prints a variable/shape/dtype table."""
    import imas_ambix.tokenizer.cli as tok_cli

    level2_shots = tmp_path / "level2" / "shots"
    shot_zarr = level2_shots / "30001.zarr"
    make_synthetic_shot_zarr(shot_zarr, "magnetics", t=16, h=32, w=32)

    monkeypatch.setattr(tok_cli, "LEVEL2_DIR", level2_shots)

    runner = CliRunner()
    result = runner.invoke(
        tokenize,
        [
            "inspect",
            "--shot",
            "30001",
            "--tier",
            "level2",
            "--group",
            "magnetics",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "data" in result.output
    assert "dtype" in result.output.lower() or "uint16" in result.output


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def test_frames_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(tokenize, ["frames", "--help"])
    assert result.exit_code == 0
    assert "frames" in result.output
    assert "--shot" in result.output
    assert "--tokenizer" in result.output
    assert "--max-frames" in result.output


def test_frames_missing_required_arg() -> None:
    """frames without --shot should fail."""
    runner = CliRunner()
    result = runner.invoke(tokenize, ["frames"])
    assert result.exit_code != 0


def test_frames_shot_not_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """frames exits non-zero when the camera group is absent."""
    import imas_ambix.tokenizer.cli as tok_cli

    monkeypatch.setattr(tok_cli, "LEVEL1_DIR", tmp_path / "shots")

    runner = CliRunner()
    result = runner.invoke(
        tokenize,
        [
            "frames",
            "--shot",
            "99999",
            "--camera",
            "rbb",
            "--tokenizer",
            "placeholder",
        ],
    )
    assert result.exit_code != 0


def test_frames_happy_path_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """frames --tokenizer placeholder encodes + decodes and prints MAE line."""
    import imas_ambix.tokenizer.cli as tok_cli

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "30001.zarr"
    make_synthetic_shot_zarr(shot_zarr, "rbb", t=8, h=32, w=32)

    monkeypatch.setattr(tok_cli, "LEVEL1_DIR", level1_shots)

    runner = CliRunner()
    result = runner.invoke(
        tokenize,
        [
            "frames",
            "--shot",
            "30001",
            "--camera",
            "rbb",
            "--tokenizer",
            "placeholder",
            "--max-frames",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    # Should log encode shape + decode MAE
    assert "encoded shape" in result.output
    assert "MAE" in result.output

    # NOTE: --tokenizer open-magvit2 is intentionally NOT tested here.
    # It requires the staging dir at
    # /work/projects/imas_gpu/mast-tokens/v1/open-magvit2/ plus GPU.
    # Add a @pytest.mark.integration test once that is available.


def test_frames_happy_path_with_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """frames --output writes a .npy file of token ids."""
    import imas_ambix.tokenizer.cli as tok_cli

    level1_shots = tmp_path / "level1" / "shots"
    shot_zarr = level1_shots / "30001.zarr"
    make_synthetic_shot_zarr(shot_zarr, "rbb", t=8, h=32, w=32)

    monkeypatch.setattr(tok_cli, "LEVEL1_DIR", level1_shots)

    out_file = tmp_path / "tokens.npy"
    runner = CliRunner()
    result = runner.invoke(
        tokenize,
        [
            "frames",
            "--shot",
            "30001",
            "--camera",
            "rbb",
            "--tokenizer",
            "placeholder",
            "--max-frames",
            "4",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    tokens = np.load(str(out_file))
    assert tokens.ndim >= 2
    assert tokens.dtype in (np.dtype("int32"), np.dtype("int64"))


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


def test_signals_help_text() -> None:
    runner = CliRunner()
    result = runner.invoke(tokenize, ["signals", "--help"])
    assert result.exit_code == 0
    assert "signals" in result.output
    assert "--shot" in result.output
    assert "--group" in result.output


def test_signals_missing_required_arg() -> None:
    """signals without --shot should fail."""
    runner = CliRunner()
    result = runner.invoke(tokenize, ["signals"])
    assert result.exit_code != 0


def test_signals_shot_not_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """signals exits non-zero when the group is absent."""
    import imas_ambix.tokenizer.cli as tok_cli

    monkeypatch.setattr(tok_cli, "LEVEL2_DIR", tmp_path / "shots")

    runner = CliRunner()
    result = runner.invoke(
        tokenize,
        ["signals", "--shot", "99999", "--group", "magnetics"],
    )
    assert result.exit_code != 0


def test_signals_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """signals encodes a 1-D signal group and prints the token-shape table."""
    import imas_ambix.tokenizer.cli as tok_cli

    level2_shots = tmp_path / "level2" / "shots"
    shot_zarr = level2_shots / "30001.zarr"
    make_synthetic_signal_zarr(shot_zarr, "magnetics", t=64)

    monkeypatch.setattr(tok_cli, "LEVEL2_DIR", level2_shots)

    runner = CliRunner()
    # Use the default n_bins=256 to avoid re-allocation conflicts with the
    # global registry singleton (which is already initialised by the module
    # import of tokenize.cli → UniformQuantizer().__post_init__).
    result = runner.invoke(
        tokenize,
        ["signals", "--shot", "30001", "--group", "magnetics"],
    )
    assert result.exit_code == 0, result.output
    # Table should include a "token shape" row
    assert "token shape" in result.output
    assert (
        "ip" in result.output
        or "beta" in result.output
        or "tokenized" in result.output.lower()
    )


def test_signals_happy_path_with_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """signals --output writes a .npy file."""
    import imas_ambix.tokenizer.cli as tok_cli

    level2_shots = tmp_path / "level2" / "shots"
    shot_zarr = level2_shots / "30001.zarr"
    make_synthetic_signal_zarr(shot_zarr, "magnetics", t=64)

    monkeypatch.setattr(tok_cli, "LEVEL2_DIR", level2_shots)

    out_file = tmp_path / "sig_tokens.npy"
    runner = CliRunner()
    # Use default n_bins=256 to avoid registry re-allocation conflicts.
    result = runner.invoke(
        tokenize,
        [
            "signals",
            "--shot",
            "30001",
            "--group",
            "magnetics",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    tokens = np.load(str(out_file))
    assert tokens.ndim == 2  # (T, n_channels)
