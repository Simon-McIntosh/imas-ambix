from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr
from PIL import Image, ImageSequence

from imas_ambix.worldmodel.flux_label_dataset import (
    EXPECTED_CARRIER_IDENTITY,
    EXPECTED_POLICY_DIGEST,
)
from imas_ambix.worldmodel.label_cartoon_video import (
    CAMERA_FILENAME,
    LABEL_FILENAME,
    RECEIPT_FILENAME,
    _write_gif,
    render_label_cartoon_pair,
)

SHOT = 21858


def _write_session(root: Path) -> Path:
    session_path = root / f"{SHOT}.nc"
    count = 4
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    levels = np.linspace(0.0, 1.0, 11)
    surface_r = np.empty((11, angles.size, count), dtype=np.float64)
    surface_z = np.empty_like(surface_r)
    for time_index in range(count):
        for level_index, level in enumerate(levels):
            scale = 0.08 + 0.34 * level
            surface_r[level_index, :, time_index] = (
                1.05 + 0.01 * time_index + scale * np.cos(angles)
            )
            surface_z[level_index, :, time_index] = 1.7 * scale * np.sin(angles)
    session = xr.Dataset(
        {
            "flux_surface_r": (
                ("flux_surface", "poloidal_angle", "time"),
                surface_r,
            ),
            "flux_surface_z": (
                ("flux_surface", "poloidal_angle", "time"),
                surface_z,
            ),
            "flux_surface_psi_norm": ("flux_surface", levels),
            "flux_surface_psi": (
                ("flux_surface", "time"),
                np.repeat(levels[:, None], count, axis=1),
            ),
            "magnetic_axis_r": ("time", np.linspace(1.04, 1.07, count)),
            "magnetic_axis_z": ("time", np.zeros(count)),
            "x_point_r": (
                ("x_point", "time"),
                np.asarray([[0.92] * count, [1.2] * count]),
            ),
            "x_point_z": (
                ("x_point", "time"),
                np.asarray([[-0.48] * count, [0.48] * count]),
            ),
            "strike_points_r": (
                ("strike_point", "time"),
                np.full((1, count), np.nan),
            ),
            "strike_points_z": (
                ("strike_point", "time"),
                np.full((1, count), np.nan),
            ),
            "divertor_leg_r": (
                ("divertor_leg", "divertor_vertex", "time"),
                np.full((1, 1, count), np.nan),
            ),
            "divertor_leg_z": (
                ("divertor_leg", "divertor_vertex", "time"),
                np.full((1, 1, count), np.nan),
            ),
            "divertor_leg_finite": (
                ("divertor_leg", "time"),
                np.zeros((1, count), dtype=bool),
            ),
            "lcfs_r": (
                ("boundary_vertex", "time"),
                np.full((1, count), np.nan),
            ),
            "lcfs_z": (
                ("boundary_vertex", "time"),
                np.full((1, count), np.nan),
            ),
            "n_boundary_coords": ("time", np.zeros(count, dtype=np.int32)),
            "rho_face_norm": ("rho_face", np.asarray([0.0, 1.0])),
            "p_prime_face": (
                ("rho_face", "time"),
                np.zeros((2, count)),
            ),
            "ff_prime_face": (
                ("rho_face", "time"),
                np.zeros((2, count)),
            ),
            "branch_guard_ok": (
                "time",
                np.asarray([True, True, False, True]),
            ),
        },
        coords={"time": np.asarray([0.01, 0.02, 0.03, 0.04])},
    )
    session.to_netcdf(session_path, group="steering", engine="h5netcdf")
    slices = [
        {"row": index, "time": float(time), "written": True, "converged": True}
        for index, time in enumerate(session.time.values)
    ]
    session_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "shot": SHOT,
                "policy_digest": EXPECTED_POLICY_DIGEST,
                "carrier_identity": EXPECTED_CARRIER_IDENTITY,
                "slices": slices,
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        session_path.with_suffix(".npz"),
        row=np.arange(count, dtype=np.int32),
        time=np.asarray(session.time.values, dtype=np.float64),
        conditioned=np.asarray([False, False, True, False]),
        conditioned_branch_guard_ok=np.asarray([True, True, False, True]),
    )
    return session_path


def _write_level1(root: Path) -> None:
    store = zarr.open_group(str(root / f"{SHOT}.zarr"), mode="w")
    camera = store.create_group("rbb")
    camera.create_array("time", data=np.asarray([0.0101, 0.0199, 0.0402]))
    frame = np.arange(72, dtype=np.float32).reshape(6, 12)
    camera.create_array("data", data=np.stack((frame, frame + 50.0, frame + 100.0)))
    thomson = store.create_group("atm")
    thomson.create_array("radius", data=np.asarray([0.75, 1.05, 1.35]))
    thomson.create_array("scat_length", data=np.asarray([0.03, 0.04, 0.03]))
    core = store.create_group("ayc")
    core.create_array("time", data=np.asarray([0.01, 0.02, 0.04]))
    core.create_array(
        "radius",
        data=np.asarray([[0.74, 1.04, 1.34], [0.75, 1.05, 1.35], [0.76, 1.06, 1.36]]),
    )
    core.create_array("te", data=np.full((3, 3), 100.0))
    core.create_array("ne", data=np.full((3, 3), 1.0e19))
    edge = store.create_group("aye")
    edge.create_array("time", data=np.asarray([0.01, 0.02, 0.04, 1.0]))
    edge.create_array(
        "r",
        data=np.asarray([[1.3, 1.4], [1.31, 1.41], [1.32, 1.42], [1.33, 1.43]]),
    )
    edge.create_array("te", data=np.full((4, 2), 80.0))
    edge.create_array("ne", data=np.full((4, 2), 8.0e18))


def _geometry() -> SimpleNamespace:
    identity = SimpleNamespace(
        representation_key="mast-era",
        representation_digest="representation-digest",
        derivation_id="machine-geometry",
        physical_digest="physical-digest",
        registry_digest="registry-digest",
    )
    conductors = (
        SimpleNamespace(r=0.34, z=-0.8, width=0.1, height=0.2, circuit=1),
        SimpleNamespace(r=1.72, z=0.8, width=0.12, height=0.2, circuit=2),
    )
    polygon_sections = (
        SimpleNamespace(
            circuit=3,
            vertices=np.asarray(
                [[0.28, -0.25], [0.34, -0.3], [0.37, -0.22], [0.31, -0.18]]
            ),
        ),
    )
    return SimpleNamespace(
        identity=identity,
        limiter_r=(0.4, 1.7, 1.7, 0.4, 0.4),
        limiter_z=(-1.2, -1.2, 1.2, 1.2, -1.2),
        conductors=conductors,
        polygon_sections=polygon_sections,
    )


def _animation_shape(path: Path) -> tuple[int, tuple[int, int]]:
    with Image.open(path) as animation:
        return sum(1 for _ in ImageSequence.Iterator(animation)), animation.size


def test_renderer_writes_aligned_tight_clipped_gifs_and_receipt(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    level1_root = tmp_path / "level1"
    output_dir = tmp_path / "media"
    session_root.mkdir()
    level1_root.mkdir()
    session_path = _write_session(session_root)
    _write_level1(level1_root)

    receipt = render_label_cartoon_pair(
        session_path,
        output_dir,
        level1_root=level1_root,
        geometry=_geometry(),
        width=120,
        fps=8,
    )

    label_frames, label_size = _animation_shape(output_dir / LABEL_FILENAME)
    camera_frames, camera_size = _animation_shape(output_dir / CAMERA_FILENAME)
    # The two in-span label images are deliberately byte-identical; keeping the
    # literal count here detects any GIF writer that coalesces them.
    assert label_frames == camera_frames == 2
    assert label_size[0] == camera_size[0] == 120
    assert camera_size == (120, 60)
    expected_label_height = round(
        120 * receipt["data_window_m"]["height_to_width_ratio"]
    )
    assert abs(label_size[1] - expected_label_height) <= 1
    assert receipt["slice_counts"] == {
        "manifest": 4,
        "written": 4,
        "converged": 4,
        "guard_eligible_converged": 3,
        "guard_failed": 1,
        "conditioned": 1,
        "free": 3,
        "free_guarded": 3,
        "inside_camera_span": 2,
        "outside_camera_span": 1,
        "paired": 2,
    }
    assert receipt["pairing"]["fraction_of_camera_span_converged"] == 1.0
    assert receipt["pairing"]["fraction_of_all_guard_eligible_converged"] == 2 / 3
    assert receipt["pairing"]["max_abs_delta_s"] < 0.00021
    assert receipt["frame_times_s"] == [0.02, 0.04]
    assert len(receipt["camera_time_deltas_s"]) == 2
    assert receipt["policy_digest"] == EXPECTED_POLICY_DIGEST
    assert receipt["carrier_identity"] == EXPECTED_CARRIER_IDENTITY
    assert receipt["percentile_intensity_limits"]["percentiles"] == [1.0, 99.5]
    recorded = json.loads((output_dir / RECEIPT_FILENAME).read_text())
    assert recorded["label_gif"]["pixel_size"] == list(label_size)
    assert recorded["camera_gif"]["pixel_size"] == list(camera_size)
    assert recorded["label_gif"]["frame_count"] == 2
    assert recorded["camera_gif"]["frame_count"] == 2
    assert recorded["gif_writer"] == "ffmpeg-palettegen-paletteuse"
    assert recorded["label_rendering"] == {
        "stack": "nova.media",
        "figure_facecolor": "white",
        "coil_facecolor": "none",
        "surface_count": 11,
        "boundary_source": "outermost nested surface at normalised flux one",
    }
    assert recorded["label_provenance"]["conditioned_frame_count"] == 1
    assert recorded["label_provenance"]["free_guarded_frame_count"] == 3
    assert [item["name"] for item in recorded["thomson_provenance"]] == [
        "core",
        "edge",
    ]
    with Image.open(output_dir / LABEL_FILENAME) as animation:
        first_label = np.asarray(animation.convert("RGB"))
    white_fraction = np.mean(np.all(first_label == 255, axis=-1))
    assert white_fraction > 0.5
    for gif_key, sheet_name, tile_height in (
        ("label_gif", "label-cartoon-frames.png", label_size[1]),
        ("camera_gif", "camera-stream-frames.png", camera_size[1]),
    ):
        contact_sheet = recorded[gif_key]["contact_sheet"]
        sheet_path = output_dir / sheet_name
        assert contact_sheet["frame_indices"] == [0, 1]
        assert contact_sheet["writer_receipt"]["tile_indices"] == [0, 1]
        assert contact_sheet["writer_receipt"]["tiles"] == 2
        assert contact_sheet["writer_receipt"]["columns"] == 2
        assert contact_sheet["writer_receipt"]["rows"] == 1
        assert contact_sheet["pixel_size"] == [2 * 120 + 8, tile_height]
        with Image.open(sheet_path) as still:
            assert still.format == "PNG"
            assert list(still.size) == contact_sheet["pixel_size"]


def test_pil_fallback_refuses_identical_consecutive_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="byte-identical consecutive GIF frames"):
        _write_gif([frame, frame.copy()], tmp_path / "identical.gif", fps=10)


def test_renderer_refuses_a_session_outside_the_pinned_policy(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    session_path = _write_session(session_root)
    manifest_path = session_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["policy_digest"] = "wrong-policy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        render_label_cartoon_pair(session_path, tmp_path / "media")
    except ValueError as error:
        assert "policy_digest" in str(error)
    else:
        raise AssertionError("a mismatched policy digest was accepted")
