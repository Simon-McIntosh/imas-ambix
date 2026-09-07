"""Render aligned Nova flux-label and MAST camera GIFs.

The label movie is drawn directly in the physical R-Z plane with one fixed
data window, while the camera movie keeps its source aspect ratio.  Both
movies use the same output width and contain exactly the same paired times.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from nova.media.gif import write_contact_sheet
from numpy.typing import NDArray

from imas_ambix.camdyn.dataset import level1_shot_path
from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.gs.machine_geometry import MachineGeometryService
from imas_ambix.worldmodel.flux_decoder_video import (
    DEFAULT_FPS,
    DEFAULT_SESSION_ROOT,
    _as_rgb_uint8,
    _manifest_selection,
    _nearest_indices,
    _read_session,
    _sha256,
    _source_revision,
    _write_video,
)
from imas_ambix.worldmodel.flux_label_dataset import (
    EXPECTED_CARRIER_IDENTITY,
    EXPECTED_POLICY_DIGEST,
    MAX_FRAME_DELTA_SECONDS,
    _load_companion,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from imas_ambix.gs.machine_geometry import OperatorGeometry

FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.uint8]

DEFAULT_WIDTH = 512
DEFAULT_CAMERA = "rbb"
LABEL_FILENAME = "label-cartoon.gif"
CAMERA_FILENAME = "camera-stream.gif"
RECEIPT_FILENAME = "receipt.json"
MIN_PAIRING_FRACTION = 0.9
RENDER_SCALE = 2
CAMERA_TIMESTAMP_RESOLUTION_SECONDS = 1.0e-5
CONTACT_SHEET_FRAME_COUNT = 6
CONTACT_SHEET_COLUMNS = 2


@dataclass(frozen=True, slots=True)
class _Selection:
    session_indices: tuple[int, ...]
    slice_times: FloatArray
    camera_indices: NDArray[np.int64]
    camera_times: FloatArray
    camera_deltas: FloatArray
    manifest_slice_count: int
    written_slice_count: int
    converged_slice_count: int
    guard_eligible_slice_count: int
    guard_failed_slice_count: int
    camera_span_slice_count: int
    outside_camera_span_count: int


@dataclass(frozen=True, slots=True)
class _ThomsonGeometry:
    radius: FloatArray
    scattering_length: FloatArray


@dataclass(frozen=True, slots=True)
class _GifProperties:
    frame_count: int
    width: int
    height: int


def _load_manifest(session_path: Path, shot: int) -> tuple[dict[str, Any], Path]:
    manifest_path = session_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path} is not an atomically complete session")
    if int(manifest.get("shot", -1)) != shot:
        raise ValueError(f"{manifest_path} shot identity does not match {shot}")
    observed_policy = str(manifest.get("policy_digest", ""))
    observed_carrier = str(manifest.get("carrier_identity", ""))
    if observed_policy != EXPECTED_POLICY_DIGEST:
        raise ValueError(
            f"{manifest_path} policy_digest {observed_policy!r} does not match "
            f"the pinned digest {EXPECTED_POLICY_DIGEST!r}"
        )
    if observed_carrier != EXPECTED_CARRIER_IDENTITY:
        raise ValueError(
            f"{manifest_path} carrier_identity {observed_carrier!r} does not match "
            f"the pinned identity {EXPECTED_CARRIER_IDENTITY!r}"
        )
    return manifest, manifest_path


def _selected_session_rows(
    session_path: Path,
    session: Any,
    manifest: Mapping[str, Any],
) -> tuple[list[int], FloatArray, dict[str, int]]:
    slices = manifest.get("slices")
    if not isinstance(slices, list):
        raise ValueError(f"{session_path.with_suffix('.manifest.json')} has no rows")

    mode, _, base_selected, manifest_slice_count = _manifest_selection(
        session_path, int(session.sizes["time"])
    )
    if mode != "labeller":
        raise ValueError(f"{session_path} is not a labeller session")

    companion_path = session_path.with_suffix(".npz")
    if not companion_path.is_file():
        raise FileNotFoundError(companion_path)
    rows, companion_times, conditioned, guard_ok = _load_companion(
        companion_path, slices
    )
    session_times = np.asarray(session["time"], dtype=np.float64).reshape(-1)
    if session_times.shape != companion_times.shape or not np.allclose(
        session_times, companion_times, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(f"{session_path} times are not aligned to its companion")

    selected = [
        index
        for index in base_selected
        if not (bool(conditioned[index]) and not bool(guard_ok[index]))
    ]
    base_set = set(base_selected)
    guard_failed = sum(
        index in base_set and bool(conditioned[index]) and not bool(guard_ok[index])
        for index in range(len(rows))
    )
    written = sum(bool(row.get("written", False)) for row in slices)
    converged = sum(
        bool(row.get("written", False)) and bool(row.get("converged", False))
        for row in slices
    )
    counts = {
        "manifest": manifest_slice_count,
        "written": written,
        "converged": converged,
        "guard_eligible": len(selected),
        "guard_failed": guard_failed,
    }
    return selected, session_times[np.asarray(selected, dtype=np.int64)], counts


def _load_level1_geometry(
    shot: int,
    camera_group: str,
    *,
    level1_root: Path,
) -> tuple[Any, Any, FloatArray, _ThomsonGeometry, Path]:
    import zarr  # noqa: PLC0415

    path = level1_shot_path(shot, level1_dir=level1_root)
    if not path.exists():
        raise FileNotFoundError(path)
    store = zarr.open_group(str(path), mode="r")
    groups = set(store.group_keys())
    if camera_group not in groups:
        raise KeyError(f"{path} does not contain camera group {camera_group!r}")
    camera = store[camera_group]
    if not {"data", "time"}.issubset(set(camera.array_keys())):
        raise KeyError(f"{path}/{camera_group} must contain data and time")
    camera_times = np.asarray(camera["time"], dtype=np.float64).reshape(-1)
    if (
        not camera_times.size
        or not np.isfinite(camera_times).all()
        or np.any(np.diff(camera_times) <= 0.0)
    ):
        raise ValueError(f"{path}/{camera_group}/time must be finite and increasing")

    if "atm" not in groups:
        raise KeyError(f"{path} does not contain the atm Thomson group")
    thomson = store["atm"]
    if not {"radius", "scat_length"}.issubset(set(thomson.array_keys())):
        raise KeyError(f"{path}/atm must contain radius and scat_length")
    radius = np.asarray(thomson["radius"], dtype=np.float64).reshape(-1)
    scattering_length = np.asarray(thomson["scat_length"], dtype=np.float64).reshape(-1)
    if radius.shape != scattering_length.shape:
        raise ValueError("Thomson radius and scat_length do not align")
    finite = np.isfinite(radius) & np.isfinite(scattering_length)
    if not finite.any():
        raise ValueError("Thomson scattering-volume geometry is entirely non-finite")
    return (
        store,
        camera["data"],
        camera_times,
        _ThomsonGeometry(radius[finite], np.abs(scattering_length[finite])),
        path,
    )


def _pair_slices(
    session_path: Path,
    session: Any,
    manifest: Mapping[str, Any],
    camera_times: FloatArray,
    *,
    max_delta_s: float,
    minimum_pairing_fraction: float,
) -> _Selection:
    selected, slice_times, counts = _selected_session_rows(
        session_path, session, manifest
    )
    indices, deltas = _nearest_indices(camera_times, slice_times)
    edge_tolerance = min(max_delta_s, CAMERA_TIMESTAMP_RESOLUTION_SECONDS)
    epsilon = np.finfo(np.float64).eps
    in_span = (slice_times >= camera_times[0] - edge_tolerance - epsilon) & (
        slice_times <= camera_times[-1] + edge_tolerance + epsilon
    )
    within_tolerance = np.abs(deltas) <= max_delta_s + np.finfo(np.float64).eps
    keep = in_span & within_tolerance
    span_count = int(in_span.sum())
    paired_count = int(keep.sum())
    if span_count == 0:
        raise ValueError("no guard-eligible converged slice overlaps the camera span")
    pairing_fraction = paired_count / span_count
    if pairing_fraction < minimum_pairing_fraction:
        raise ValueError(
            f"only {paired_count}/{span_count} camera-span slices pair within "
            f"{max_delta_s:.6g} s ({pairing_fraction:.3%})"
        )
    selected_array = np.asarray(selected, dtype=np.int64)
    return _Selection(
        session_indices=tuple(int(value) for value in selected_array[keep]),
        slice_times=slice_times[keep],
        camera_indices=indices[keep],
        camera_times=camera_times[indices[keep]],
        camera_deltas=deltas[keep],
        manifest_slice_count=counts["manifest"],
        written_slice_count=counts["written"],
        converged_slice_count=counts["converged"],
        guard_eligible_slice_count=counts["guard_eligible"],
        guard_failed_slice_count=counts["guard_failed"],
        camera_span_slice_count=span_count,
        outside_camera_span_count=int((~in_span).sum()),
    )


def _camera_images(
    frames: NDArray[Any], width: int
) -> tuple[list[ImageArray], list[float]]:
    from matplotlib import colormaps  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    values = np.asarray(frames)
    if values.ndim == 4 and values.shape[-1] in {3, 4}:
        values = np.mean(values[..., :3], axis=-1)
    if values.ndim != 3:
        raise ValueError(
            f"camera data must be (time, height, width[, channels]), got {values.shape}"
        )
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("selected camera stack has no finite intensity")
    low, high = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("selected camera stack has a degenerate percentile range")
    scaled = np.clip((values.astype(np.float64) - low) / (high - low), 0.0, 1.0)
    coloured = np.asarray(colormaps["inferno"](scaled, bytes=True)[..., :3])
    coloured = _as_rgb_uint8(coloured)
    source_height, source_width = coloured.shape[1:3]
    output_height = max(1, int(round(width * source_height / source_width)))
    images = [
        np.asarray(
            Image.fromarray(frame).resize(
                (width, output_height), Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
        for frame in coloured
    ]
    return images, [float(low), float(high)]


def _gif_properties(path: Path) -> _GifProperties:
    from PIL import Image, ImageSequence  # noqa: PLC0415

    with Image.open(path) as animation:
        width, height = animation.size
        frame_count = sum(1 for _ in ImageSequence.Iterator(animation))
    return _GifProperties(frame_count=frame_count, width=width, height=height)


def _ffmpeg_gif(
    frames: Sequence[ImageArray], output: Path, fps: int, ffmpeg: str
) -> None:
    stack = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames])
    if stack.ndim != 4 or stack.shape[-1] != 3:
        raise ValueError(f"GIF frames must be RGB images, got {stack.shape}")
    if any(frame.shape != stack.shape[1:] for frame in frames):
        raise ValueError("GIF frames must have one common pixel size")
    height, width = stack.shape[1:3]
    raw_video = stack.tobytes()
    input_options = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
    ]
    with tempfile.TemporaryDirectory(prefix="ambix-label-gif-") as directory:
        palette = Path(directory) / "palette.png"
        subprocess.run(
            [
                *input_options,
                "-vf",
                "palettegen=stats_mode=full",
                "-frames:v",
                "1",
                "-y",
                str(palette),
            ],
            input=raw_video,
            check=True,
        )
        subprocess.run(
            [
                *input_options,
                "-i",
                str(palette),
                "-filter_complex",
                "[0:v][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                "-fps_mode",
                "passthrough",
                "-loop",
                "0",
                "-y",
                str(output),
            ],
            input=raw_video,
            check=True,
        )


def _write_gif(frames: Sequence[ImageArray], output: Path, fps: int) -> str:
    if not frames:
        raise ValueError("cannot write an empty GIF")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        _ffmpeg_gif(frames, output, fps, ffmpeg)
        route = "ffmpeg-palettegen-paletteuse"
    else:
        identical_pairs = [
            index
            for index, (previous, current) in enumerate(
                zip(frames, frames[1:], strict=False), start=1
            )
            if np.array_equal(previous, current)
        ]
        if identical_pairs:
            raise RuntimeError(
                "ffmpeg is unavailable and PIL would merge byte-identical "
                f"consecutive GIF frames at indices {identical_pairs}"
            )
        _write_video(frames, output, fps)
        route = "pil-nonidentical-fallback"

    properties = _gif_properties(output)
    if properties.frame_count != len(frames):
        raise RuntimeError(
            f"{route} wrote {properties.frame_count} GIF frames from "
            f"{len(frames)} inputs"
        )
    return route


def _slice_array(session: Any, name: str, index: int) -> FloatArray:
    value = session[name]
    if "time" in value.dims:
        value = value.isel(time=index)
    return np.asarray(value, dtype=np.float64)


def _rectangular_coils(geometry: OperatorGeometry) -> list[tuple[float, ...]]:
    polygon_circuits = {int(section.circuit) for section in geometry.polygon_sections}
    return [
        (
            float(coil.r - abs(coil.width) / 2.0),
            float(coil.z - abs(coil.height) / 2.0),
            float(abs(coil.width)),
            float(abs(coil.height)),
        )
        for coil in geometry.conductors
        if int(coil.circuit) not in polygon_circuits
    ]


def _finite_points(*arrays: NDArray[Any]) -> tuple[FloatArray, FloatArray]:
    r = np.concatenate(
        [np.asarray(array[0], dtype=np.float64).reshape(-1) for array in arrays]
    )
    z = np.concatenate(
        [np.asarray(array[1], dtype=np.float64).reshape(-1) for array in arrays]
    )
    finite = np.isfinite(r) & np.isfinite(z)
    return r[finite], z[finite]


def _data_window(
    session: Any,
    indices: Sequence[int],
    geometry: OperatorGeometry,
    thomson: _ThomsonGeometry,
) -> tuple[float, float, float, float]:
    point_sets: list[NDArray[Any]] = [
        np.vstack((geometry.limiter_r, geometry.limiter_z)),
        np.vstack(
            (
                thomson.radius - thomson.scattering_length / 2.0,
                np.zeros_like(thomson.radius),
            )
        ),
        np.vstack(
            (
                thomson.radius + thomson.scattering_length / 2.0,
                np.zeros_like(thomson.radius),
            )
        ),
    ]
    for rectangle in _rectangular_coils(geometry):
        r, z, width, height = rectangle
        point_sets.append(
            np.asarray(((r, r + width), (z, z + height)), dtype=np.float64)
        )
    point_sets.extend(
        np.asarray(section.vertices, dtype=np.float64).T
        for section in geometry.polygon_sections
    )
    for index in indices:
        point_sets.extend(
            (
                np.vstack(
                    (
                        _slice_array(session, "flux_surface_r", index).reshape(-1),
                        _slice_array(session, "flux_surface_z", index).reshape(-1),
                    )
                ),
                np.asarray(
                    (
                        [_slice_array(session, "magnetic_axis_r", index).item()],
                        [_slice_array(session, "magnetic_axis_z", index).item()],
                    )
                ),
                np.vstack(
                    (
                        _slice_array(session, "x_point_r", index).reshape(-1),
                        _slice_array(session, "x_point_z", index).reshape(-1),
                    )
                ),
            )
        )
    r_values, z_values = _finite_points(*point_sets)
    if not r_values.size:
        raise ValueError("label geometry has no finite R-Z points")
    r_min, r_max = float(r_values.min()), float(r_values.max())
    z_min, z_max = float(z_values.min()), float(z_values.max())
    if r_max <= r_min or z_max <= z_min:
        raise ValueError("label geometry has a degenerate R-Z data window")
    return r_min, r_max, z_min, z_max


def _label_image(
    session: Any,
    index: int,
    geometry: OperatorGeometry,
    thomson: _ThomsonGeometry,
    window: tuple[float, float, float, float],
    width: int,
) -> ImageArray:
    from matplotlib import colormaps  # noqa: PLC0415
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415
    from matplotlib.patches import Polygon, Rectangle  # noqa: PLC0415

    r_min, r_max, z_min, z_max = window
    height = max(1, int(round(width * (z_max - z_min) / (r_max - r_min))))
    render_width = width * RENDER_SCALE
    render_height = height * RENDER_SCALE
    figure = Figure(
        figsize=(render_width / 100.0, render_height / 100.0),
        dpi=100,
        facecolor="black",
    )
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0), facecolor="black")
    axes.set_xlim(r_min, r_max)
    axes.set_ylim(z_min, z_max)
    axes.set_aspect("equal", adjustable="box")
    axes.set_axis_off()
    inferno = colormaps["inferno"]

    for r, z, coil_width, coil_height in _rectangular_coils(geometry):
        axes.add_patch(
            Rectangle(
                (r, z),
                coil_width,
                coil_height,
                facecolor=inferno(0.24),
                edgecolor=inferno(0.48),
                linewidth=0.45 * RENDER_SCALE,
                alpha=0.7,
            )
        )
    for section in geometry.polygon_sections:
        axes.add_patch(
            Polygon(
                np.asarray(section.vertices, dtype=np.float64),
                closed=True,
                facecolor=inferno(0.24),
                edgecolor=inferno(0.48),
                linewidth=0.45 * RENDER_SCALE,
                alpha=0.7,
            )
        )
    axes.plot(
        geometry.limiter_r,
        geometry.limiter_z,
        color=inferno(0.72),
        linewidth=0.75 * RENDER_SCALE,
    )
    for radius, length in zip(thomson.radius, thomson.scattering_length, strict=True):
        axes.plot(
            (radius - length / 2.0, radius + length / 2.0),
            (0.0, 0.0),
            color=inferno(0.5),
            linewidth=0.35 * RENDER_SCALE,
            alpha=0.35,
        )

    surface_r = _slice_array(session, "flux_surface_r", index)
    surface_z = _slice_array(session, "flux_surface_z", index)
    levels = _slice_array(session, "flux_surface_psi_norm", index).reshape(-1)
    if surface_r.shape != surface_z.shape or surface_r.shape[0] != levels.size:
        raise ValueError("flux surfaces and psi_norm levels do not align")
    finite_levels = levels[np.isfinite(levels)]
    if not finite_levels.size:
        raise ValueError("flux-surface levels are entirely non-finite")
    level_min, level_max = float(finite_levels.min()), float(finite_levels.max())
    span = level_max - level_min
    for r_values, z_values, level in zip(surface_r, surface_z, levels, strict=True):
        finite = np.isfinite(r_values) & np.isfinite(z_values)
        if finite.sum() < 2 or not np.isfinite(level):
            continue
        fraction = 1.0 if span <= 0.0 else (float(level) - level_min) / span
        axes.plot(
            r_values[finite],
            z_values[finite],
            color=inferno(0.28 + 0.7 * fraction),
            linewidth=(0.55 + 0.55 * fraction) * RENDER_SCALE,
        )

    axis_r = float(_slice_array(session, "magnetic_axis_r", index).item())
    axis_z = float(_slice_array(session, "magnetic_axis_z", index).item())
    if np.isfinite(axis_r) and np.isfinite(axis_z):
        axes.plot(
            axis_r,
            axis_z,
            marker="o",
            markersize=2.4 * RENDER_SCALE,
            markerfacecolor=inferno(0.98),
            markeredgewidth=0.0,
        )
    x_r = _slice_array(session, "x_point_r", index).reshape(-1)
    x_z = _slice_array(session, "x_point_z", index).reshape(-1)
    finite_x = np.isfinite(x_r) & np.isfinite(x_z)
    if finite_x.any():
        axes.plot(
            x_r[finite_x],
            x_z[finite_x],
            linestyle="none",
            marker="x",
            markersize=3.2 * RENDER_SCALE,
            markeredgewidth=0.7 * RENDER_SCALE,
            color=inferno(0.9),
        )
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
    from PIL import Image  # noqa: PLC0415

    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _output_receipt(
    *,
    shot: int,
    camera_group: str,
    session_path: Path,
    manifest_path: Path,
    level1_path: Path,
    selection: _Selection,
    window: tuple[float, float, float, float],
    intensity_limits: list[float],
    label_path: Path,
    camera_path: Path,
    label_properties: _GifProperties,
    camera_properties: _GifProperties,
    label_contact_sheet: Mapping[str, object],
    camera_contact_sheet: Mapping[str, object],
    gif_writer: str,
    geometry: OperatorGeometry,
    fps: int,
) -> dict[str, object]:
    r_min, r_max, z_min, z_max = window
    paired_fraction = len(selection.session_indices) / selection.camera_span_slice_count
    full_coverage_fraction = (
        len(selection.session_indices) / selection.guard_eligible_slice_count
    )
    return {
        "shot": shot,
        "camera_group": camera_group,
        "policy_digest": EXPECTED_POLICY_DIGEST,
        "carrier_identity": EXPECTED_CARRIER_IDENTITY,
        "session": str(session_path.resolve()),
        "session_sha256": _sha256(session_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "level1_store": str(level1_path.resolve()),
        "slice_counts": {
            "manifest": selection.manifest_slice_count,
            "written": selection.written_slice_count,
            "converged": selection.converged_slice_count,
            "guard_eligible_converged": selection.guard_eligible_slice_count,
            "conditioned_guard_failed": selection.guard_failed_slice_count,
            "inside_camera_span": selection.camera_span_slice_count,
            "outside_camera_span": selection.outside_camera_span_count,
            "paired": len(selection.session_indices),
        },
        "pairing": {
            "maximum_allowed_abs_delta_s": MAX_FRAME_DELTA_SECONDS,
            "max_abs_delta_s": float(np.max(np.abs(selection.camera_deltas))),
            "fraction_of_camera_span_converged": paired_fraction,
            "fraction_of_all_guard_eligible_converged": full_coverage_fraction,
        },
        "frame_times_s": [float(value) for value in selection.slice_times],
        "camera_frame_times_s": [float(value) for value in selection.camera_times],
        "camera_time_deltas_s": [float(value) for value in selection.camera_deltas],
        "percentile_intensity_limits": {
            "percentiles": [1.0, 99.5],
            "values": intensity_limits,
            "scope": "whole paired camera stack",
            "colormap": "inferno",
        },
        "data_window_m": {
            "r": [r_min, r_max],
            "z": [z_min, z_max],
            "height_to_width_ratio": (z_max - z_min) / (r_max - r_min),
        },
        "label_gif": {
            "path": str(label_path.resolve()),
            "sha256": _sha256(label_path),
            "frame_count": label_properties.frame_count,
            "pixel_size": [label_properties.width, label_properties.height],
            "contact_sheet": dict(label_contact_sheet),
        },
        "camera_gif": {
            "path": str(camera_path.resolve()),
            "sha256": _sha256(camera_path),
            "frame_count": camera_properties.frame_count,
            "pixel_size": [camera_properties.width, camera_properties.height],
            "contact_sheet": dict(camera_contact_sheet),
        },
        "gif_writer": gif_writer,
        "thomson_geometry": "atm scattering-volume locus on the z=0 laser plane",
        "machine_geometry_identity": {
            "representation_key": geometry.identity.representation_key,
            "representation_digest": geometry.identity.representation_digest,
            "derivation_id": geometry.identity.derivation_id,
            "physical_digest": geometry.identity.physical_digest,
            "registry_digest": geometry.identity.registry_digest,
        },
        "fps": fps,
        "source_revision": _source_revision(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def render_label_cartoon_pair(
    session_path: Path,
    output_dir: Path,
    *,
    camera_group: str = DEFAULT_CAMERA,
    level1_root: Path = LEVEL1_DIR,
    geometry: OperatorGeometry | None = None,
    width: int = DEFAULT_WIDTH,
    fps: int = DEFAULT_FPS,
    max_frame_delta_s: float = MAX_FRAME_DELTA_SECONDS,
    minimum_pairing_fraction: float = MIN_PAIRING_FRACTION,
) -> dict[str, object]:
    """Write the aligned label and camera GIFs and their evidence receipt."""
    session_path = Path(session_path)
    output_dir = Path(output_dir)
    if not session_path.is_file():
        raise FileNotFoundError(session_path)
    if not session_path.stem.isdigit():
        raise ValueError("the labeller session filename must be a numeric shot")
    if width <= 0 or fps <= 0:
        raise ValueError("width and fps must be positive")
    if not 0.0 < minimum_pairing_fraction <= 1.0:
        raise ValueError("minimum_pairing_fraction must be in (0, 1]")
    shot = int(session_path.stem)
    manifest, manifest_path = _load_manifest(session_path, shot)
    session = _read_session(session_path)
    store, camera_data, camera_times, thomson, level1_path = _load_level1_geometry(
        shot, camera_group, level1_root=level1_root
    )
    del store
    selection = _pair_slices(
        session_path,
        session,
        manifest,
        camera_times,
        max_delta_s=max_frame_delta_s,
        minimum_pairing_fraction=minimum_pairing_fraction,
    )
    frames = np.stack(
        [np.asarray(camera_data[int(index)]) for index in selection.camera_indices]
    )
    camera_frames, intensity_limits = _camera_images(frames, width)
    actual_geometry = geometry or MachineGeometryService().operator(shot)
    window = _data_window(session, selection.session_indices, actual_geometry, thomson)
    label_frames = [
        _label_image(session, index, actual_geometry, thomson, window, width)
        for index in selection.session_indices
    ]
    if len(label_frames) != len(camera_frames):
        raise RuntimeError("label and camera frame counts diverged")

    label_path = output_dir / LABEL_FILENAME
    camera_path = output_dir / CAMERA_FILENAME
    label_contact_path = label_path.with_name(f"{label_path.stem}-frames.png")
    camera_contact_path = camera_path.with_name(f"{camera_path.stem}-frames.png")
    receipt_path = output_dir / RECEIPT_FILENAME
    collisions = [
        path
        for path in (
            label_path,
            camera_path,
            label_contact_path,
            camera_contact_path,
            receipt_path,
        )
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite {collisions[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_writer = _write_gif(label_frames, label_path, fps)
    camera_writer = _write_gif(camera_frames, camera_path, fps)
    if label_writer != camera_writer:
        raise RuntimeError(
            f"label and camera GIF writer routes diverged: "
            f"{label_writer!r} != {camera_writer!r}"
        )
    label_properties = _gif_properties(label_path)
    camera_properties = _gif_properties(camera_path)
    if label_properties.frame_count != len(label_frames):
        raise RuntimeError(
            f"label GIF contains {label_properties.frame_count} frames, expected "
            f"{len(label_frames)}"
        )
    if camera_properties.frame_count != len(camera_frames):
        raise RuntimeError(
            f"camera GIF contains {camera_properties.frame_count} frames, expected "
            f"{len(camera_frames)}"
        )
    if label_properties.frame_count != camera_properties.frame_count:
        raise RuntimeError(
            "written label and camera GIF frame counts are not identical"
        )
    if label_properties.width != camera_properties.width:
        raise RuntimeError("written label and camera GIF widths are not identical")

    label_contact_indices = np.unique(
        np.linspace(0, len(label_frames) - 1, CONTACT_SHEET_FRAME_COUNT).round()
    ).astype(int)
    camera_contact_indices = np.unique(
        np.linspace(0, len(camera_frames) - 1, CONTACT_SHEET_FRAME_COUNT).round()
    ).astype(int)
    from PIL import Image  # noqa: PLC0415

    label_contact_writer_receipt = write_contact_sheet(
        [Image.fromarray(frame) for frame in label_frames],
        label_contact_path.resolve(),
        columns=CONTACT_SHEET_COLUMNS,
        count=CONTACT_SHEET_FRAME_COUNT,
    )
    camera_contact_writer_receipt = write_contact_sheet(
        [Image.fromarray(frame) for frame in camera_frames],
        camera_contact_path.resolve(),
        columns=CONTACT_SHEET_COLUMNS,
        count=CONTACT_SHEET_FRAME_COUNT,
    )
    if label_contact_writer_receipt["tile_indices"] != label_contact_indices.tolist():
        raise RuntimeError("label contact-sheet frame selection diverged")
    if camera_contact_writer_receipt["tile_indices"] != camera_contact_indices.tolist():
        raise RuntimeError("camera contact-sheet frame selection diverged")
    with Image.open(label_contact_path) as image:
        if image.format != "PNG":
            raise RuntimeError("label contact sheet is not a PNG")
        label_contact_size = list(image.size)
    with Image.open(camera_contact_path) as image:
        if image.format != "PNG":
            raise RuntimeError("camera contact sheet is not a PNG")
        camera_contact_size = list(image.size)
    label_contact_sheet = {
        "writer_receipt": label_contact_writer_receipt,
        "frame_indices": label_contact_indices.tolist(),
        "pixel_size": label_contact_size,
        "sha256": _sha256(label_contact_path),
    }
    camera_contact_sheet = {
        "writer_receipt": camera_contact_writer_receipt,
        "frame_indices": camera_contact_indices.tolist(),
        "pixel_size": camera_contact_size,
        "sha256": _sha256(camera_contact_path),
    }
    receipt = _output_receipt(
        shot=shot,
        camera_group=camera_group,
        session_path=session_path,
        manifest_path=manifest_path,
        level1_path=level1_path,
        selection=selection,
        window=window,
        intensity_limits=intensity_limits,
        label_path=label_path,
        camera_path=camera_path,
        label_properties=label_properties,
        camera_properties=camera_properties,
        label_contact_sheet=label_contact_sheet,
        camera_contact_sheet=camera_contact_sheet,
        gif_writer=label_writer,
        geometry=actual_geometry,
        fps=fps,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shot", type=int)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    receipt = render_label_cartoon_pair(
        args.session_root / f"{args.shot}.nc",
        args.output_dir,
        camera_group=args.camera,
        level1_root=args.level1_root,
        width=args.width,
        fps=args.fps,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMERA_FILENAME",
    "LABEL_FILENAME",
    "RECEIPT_FILENAME",
    "render_label_cartoon_pair",
    "main",
]
