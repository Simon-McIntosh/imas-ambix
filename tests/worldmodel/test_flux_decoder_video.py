from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np
import xarray as xr
from PIL import Image

from imas_ambix.worldmodel.flux_decoder_video import render_session_video


class _Decoded(NamedTuple):
    image: object
    decode_wall: float
    decoder_identity: str


class _StubDecoder:
    decoder_identity = "synthetic-checkpoint:stub-vq:synthetic-corpus"

    def __init__(self) -> None:
        self.count = 0

    def decode(self, frame: object) -> _Decoded:
        del frame
        self.count += 1
        image = np.full((256, 256, 3), self.count * 30, dtype=np.uint8)
        return _Decoded(image, self.count / 1000.0, self.decoder_identity)


def _write_session(path: Path, *, count: int) -> None:
    session = xr.Dataset(
        {
            "action_name": ("time", ["prime", "elongation+", "gap-"][:count]),
            "wall_seconds": ("time", np.arange(1, count + 1) / 100.0),
        },
        coords={"time": np.arange(count, dtype=np.float64) * 0.005},
    )
    session.to_netcdf(path, group="steering", engine="h5netcdf")


def _animation_shape(path: Path) -> tuple[int, tuple[int, int]]:
    with Image.open(path) as animation:
        return animation.n_frames, animation.size


def test_labeller_video_pairs_only_written_converged_slices(tmp_path: Path) -> None:
    session_path = tmp_path / "21858.nc"
    checkpoint = tmp_path / "decoder.pt"
    output = tmp_path / "comparison.gif"
    checkpoint.write_bytes(b"synthetic untrained checkpoint")
    _write_session(session_path, count=3)
    session_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "shot": 21858,
                "slices": [
                    {"row": 0, "written": True, "converged": True},
                    {"row": 1, "written": True, "converged": False},
                    {"row": 2, "written": True, "converged": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    real = np.stack(
        [np.full((48, 256), 10, dtype=np.uint8), np.full((48, 256), 20, dtype=np.uint8)]
    )

    receipt = render_session_video(
        session_path,
        checkpoint,
        output,
        decoder=_StubDecoder(),
        real_frames=real,
        frame_deltas=np.array([0.0001, -0.0002]),
    )

    assert _animation_shape(output) == (2, (512, 288))
    assert receipt["mode"] == "labeller"
    assert receipt["shot"] == 21858
    assert receipt["manifest_slice_count"] == 3
    assert receipt["admitted_before_camera_join"] == 2
    assert receipt["frame_count"] == 2
    assert receipt["max_abs_camera_delta_s"] == 0.0002
    assert receipt["vq_route"] == "stub"
    recorded = json.loads(output.with_suffix(".receipt.json").read_text())
    assert recorded["output_sha256"] == receipt["output_sha256"]


def test_steering_video_burns_action_and_wall_on_decoded_frames(tmp_path: Path) -> None:
    session_path = tmp_path / "steering.nc"
    checkpoint = tmp_path / "decoder.pt"
    output = tmp_path / "steering.gif"
    checkpoint.write_bytes(b"synthetic untrained checkpoint")
    _write_session(session_path, count=3)

    receipt = render_session_video(
        session_path,
        checkpoint,
        output,
        decoder=_StubDecoder(),
    )

    assert _animation_shape(output) == (3, (256, 288))
    assert receipt["mode"] == "steering"
    assert receipt["actions"] == ["prime", "elongation+", "gap-"]
    with Image.open(output) as animation:
        first = np.asarray(animation.convert("RGB"))
    assert np.any(first[:32] != 0)


def test_serve_launcher_resolves_fixed_paths_in_dry_run() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = root / "scripts/slurm/playable_camera_serve.sbatch"
    syntax = subprocess.run(["bash", "-n", str(launcher)], check=False)
    assert syntax.returncode == 0
    result = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PLAYABLE_PORT=18506" in result.stdout
    assert (
        "PLAYABLE_DECODER=imas_ambix.worldmodel.flux_conditioned_decoder:"
        "FluxConditionedDecoder" in result.stdout
    )
    assert (
        "PLAYABLE_PYTHON=/home/ITER/mcintos/Code/imas-ambix/.venv/bin/python"
        in result.stdout
    )
    assert "DRY_RUN_EXIT_STATUS=0" in result.stdout
