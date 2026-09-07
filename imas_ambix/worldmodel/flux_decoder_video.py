"""Render decoded camera videos from recorded Nova steering sessions.

Labeller sessions are paired with their nearest real ``rbb`` frames and shown
side by side.  Ordinary recorded steering sessions contain no camera payload,
so they render decoded frames alone with the action and solve wall time painted
into each frame.  OpenMAGVIT2 is loaded once for the complete token stack,
either in this interpreter or in its dedicated interpreter when its source
package is unavailable here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from imas_ambix.camdyn.dataset import frames_token_path, level1_shot_path
from imas_ambix.data.paths import LEVEL1_DIR, TOKEN_ROOT
from imas_ambix.data.stream_encode import REGISTRY_OFFSET

DEFAULT_SESSION_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_VQ_CHECKPOINT = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/weights/imagenet_256_L.ckpt"
)
DEFAULT_VQ_PYTHON = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python"
)
DEFAULT_FPS = 10
MAX_CAMERA_DELTA_SECONDS = 0.0025
VIDEO_HEIGHT = 256
VIDEO_WIDTH = 256
BANNER_HEIGHT = 32
CONTACT_SHEET_FRAME_COUNT = 6
HISTORY_FRAME_COUNT = 4
HISTORY_SPACING_SECONDS = 0.005
CAMERA_VOCAB_SIZE = 1 << 18

ImageArray = NDArray[np.uint8]
TokenArray = NDArray[np.int64]


class _SeedWindow(NamedTuple):
    selected: list[int]
    query_times: NDArray[np.float64]
    session_slice_indices: NDArray[np.int64]
    camera_frame_indices: NDArray[np.int64]
    camera_time_deltas: NDArray[np.float64]
    leading_slices_skipped: int


class FrameDecoder(Protocol):
    """Small surface used by the renderer and synthetic test decoder."""

    decoder_identity: str

    def decode(self, frame: object) -> object: ...


class _TokenCollector:
    """Capture predicted tokens while deferring VQ decode to one batch."""

    def __init__(self) -> None:
        self.tokens: list[TokenArray] = []

    def decode(self, tokens: TokenArray) -> ImageArray:
        self.tokens.append(np.asarray(tokens, dtype=np.int64).copy())
        return np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_session(path: Path) -> Any:
    import xarray as xr  # noqa: PLC0415

    with xr.open_dataset(path, group="steering", engine="h5netcdf") as source:
        return source.load()


def _manifest_selection(
    session_path: Path, session_count: int
) -> tuple[str, int | None, list[int], int]:
    manifest_path = session_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return "steering", None, list(range(session_count)), session_count

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    slices = payload.get("slices")
    if not isinstance(slices, list):
        raise ValueError(f"{manifest_path} has no slice-row list")
    selected: list[int] = []
    session_index = 0
    for row in slices:
        if not isinstance(row, Mapping):
            raise ValueError(f"{manifest_path} contains a non-object slice row")
        if not bool(row.get("written", False)):
            continue
        if bool(row.get("converged", False)):
            selected.append(session_index)
        session_index += 1
    if session_index != session_count:
        raise ValueError(
            f"{manifest_path} has {session_index} written rows but the session "
            f"contains {session_count} frames"
        )
    shot = int(payload.get("shot", session_path.stem))
    return "labeller", shot, selected, len(slices)


def _nearest_indices(
    reference: NDArray[np.float64], query: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    high = np.clip(
        np.searchsorted(reference, query, side="left"), 0, reference.size - 1
    )
    low = np.clip(high - 1, 0, reference.size - 1)
    choose_high = np.abs(reference[high] - query) <= np.abs(query - reference[low])
    indices = np.where(choose_high, high, low).astype(np.int64, copy=False)
    return indices, reference[indices] - query


def _resolve_seed_window(
    selected: Sequence[int],
    render_session_times: NDArray[np.float64],
    seed_session_times: NDArray[np.float64],
    seed_camera_times: NDArray[np.float64],
    *,
    requested_start_slice: int,
) -> _SeedWindow:
    """Tie four real-token history frames to the first rendered slice."""
    if not 0 <= requested_start_slice < render_session_times.size:
        raise IndexError(
            f"seed_slice {requested_start_slice} outside "
            f"{render_session_times.size} session rows"
        )
    candidates = [index for index in selected if index >= requested_start_slice]
    if not candidates:
        raise ValueError("the session has no admitted slice at or after seed_slice")
    if not seed_session_times.size or not seed_camera_times.size:
        raise ValueError("seed session and camera times must be non-empty")

    spacing = HISTORY_SPACING_SECONDS * np.arange(
        HISTORY_FRAME_COUNT, 0, -1, dtype=np.float64
    )
    for position, first_rendered_slice in enumerate(candidates):
        render_time = float(render_session_times[first_rendered_slice])
        query_times = render_time - spacing
        session_indices, session_deltas = _nearest_indices(
            seed_session_times, query_times
        )
        camera_indices, camera_deltas = _nearest_indices(seed_camera_times, query_times)
        full_history = bool(
            query_times[0] >= seed_session_times[0]
            and query_times[-1] <= seed_session_times[-1]
            and query_times[0] >= seed_camera_times[0]
            and query_times[-1] <= seed_camera_times[-1]
            and np.all(np.abs(session_deltas) <= MAX_CAMERA_DELTA_SECONDS)
            and np.all(np.abs(camera_deltas) <= MAX_CAMERA_DELTA_SECONDS)
            and np.all(seed_session_times[session_indices] < render_time)
            and np.all(seed_camera_times[camera_indices] < render_time)
        )
        if full_history:
            return _SeedWindow(
                selected=candidates[position:],
                query_times=query_times,
                session_slice_indices=session_indices,
                camera_frame_indices=camera_indices,
                camera_time_deltas=camera_deltas,
                leading_slices_skipped=position,
            )
    raise ValueError(
        "no admitted slice at or after seed_slice has four preceding history frames"
    )


def _camera_times(shot: int, *, level1_root: Path) -> NDArray[np.float64]:
    import zarr  # noqa: PLC0415

    path = level1_shot_path(shot, level1_dir=level1_root)
    store = zarr.open_group(str(path), mode="r")
    return np.asarray(store["rbb"]["time"], dtype=np.float64)


def _load_seed_tokens(
    shot: int,
    frame_indices: NDArray[np.int64],
    *,
    token_root: Path,
) -> TokenArray:
    import zarr  # noqa: PLC0415

    path = frames_token_path(shot, "rbb", token_root=token_root)
    store = zarr.open_group(str(path), mode="r")
    tokens = store["tokens"]
    if frame_indices.size != HISTORY_FRAME_COUNT:
        raise ValueError("seed history must contain four camera-frame indices")
    if int(frame_indices[-1]) >= int(tokens.shape[0]):
        raise ValueError("seed camera times and token frames have different lengths")
    stored = np.stack(
        [np.asarray(tokens[int(index)], dtype=np.int64) for index in frame_indices]
    )
    upper_bound = REGISTRY_OFFSET + CAMERA_VOCAB_SIZE
    if np.any(stored < REGISTRY_OFFSET) or np.any(stored >= upper_bound):
        raise ValueError("seed token history contains an out-of-range token id")
    return (stored - REGISTRY_OFFSET).astype(np.int64, copy=False)


def _load_real_frames(
    shot: int,
    times: NDArray[np.float64],
    *,
    level1_root: Path,
) -> tuple[ImageArray, NDArray[np.float64], NDArray[np.bool_]]:
    import zarr  # noqa: PLC0415

    path = level1_shot_path(shot, level1_dir=level1_root)
    store = zarr.open_group(str(path), mode="r")
    camera = store["rbb"]
    frame_times = np.asarray(camera["time"], dtype=np.float64)
    indices, deltas = _nearest_indices(frame_times, times)
    keep = np.abs(deltas) <= MAX_CAMERA_DELTA_SECONDS
    frames = np.stack(
        [np.asarray(camera["data"][int(index)]) for index in indices[keep]]
    )
    return _as_rgb_uint8(frames), deltas[keep], keep


def _as_rgb_uint8(frames: NDArray[Any]) -> ImageArray:
    values = np.asarray(frames)
    if values.dtype != np.uint8:
        finite = values[np.isfinite(values)]
        if not finite.size or float(finite.max()) <= float(finite.min()):
            values = np.zeros(values.shape, dtype=np.uint8)
        else:
            low = float(finite.min())
            scale = 255.0 / (float(finite.max()) - low)
            values = np.clip((values - low) * scale, 0.0, 255.0).astype(np.uint8)
    if values.ndim == 3:
        values = np.repeat(values[..., None], 3, axis=-1)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(
            f"camera frames must be (time, height, width[, 3]), got {values.shape}"
        )
    return np.asarray(values, dtype=np.uint8)


def _resize(image: ImageArray) -> ImageArray:
    from PIL import Image  # noqa: PLC0415

    source = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(
        source.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.BILINEAR)
    )


def _frame_value(frame: object, name: str, default: object) -> object:
    try:
        value = frame[name]  # type: ignore[index]
    except KeyError, TypeError:
        value = getattr(frame, name, default)
    if hasattr(value, "item"):
        return value.item()
    return value


def _compose_frame(
    decoded: ImageArray,
    *,
    real: ImageArray | None,
    action: str,
    keyframe_wall: float,
    slice_time: float,
) -> ImageArray:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    panels = [_resize(decoded)] if real is None else [_resize(real), _resize(decoded)]
    width = VIDEO_WIDTH * len(panels)
    canvas = Image.new("RGB", (width, VIDEO_HEIGHT + BANNER_HEIGHT), "black")
    for index, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (index * VIDEO_WIDTH, BANNER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    if real is not None:
        draw.text((6, BANNER_HEIGHT + 5), "real rbb", fill="white")
        draw.text((VIDEO_WIDTH + 6, BANNER_HEIGHT + 5), "decoded", fill="white")
    banner = (
        f"action: {action or 'recorded slice'} | keyframe wall: "
        f"{keyframe_wall * 1000.0:.1f} ms | t={slice_time:.6f} s"
    )
    draw.text((6, 9), banner, fill="white")
    return np.asarray(canvas, dtype=np.uint8)


def _pixel_error_receipt(
    real_frames: ImageArray | None, decoded_frames: Sequence[ImageArray]
) -> dict[str, object] | None:
    if real_frames is None or len(decoded_frames) < 2:
        return None
    real_rgb = _as_rgb_uint8(real_frames)
    real = np.stack([_resize(frame) for frame in real_rgb]).astype(np.float64)
    decoded = np.stack([_resize(frame) for frame in decoded_frames]).astype(np.float64)
    if real.shape != decoded.shape:
        raise ValueError(
            f"real and decoded stacks must share a shape, got {real.shape} and "
            f"{decoded.shape}"
        )
    decoded_per_frame = np.mean(np.abs(decoded - real), axis=(1, 2, 3))
    persistence_per_frame = np.mean(np.abs(real[1:] - real[:-1]), axis=(1, 2, 3))
    decoded_mae = float(np.mean(decoded_per_frame[1:]))
    persistence_mae = float(np.mean(persistence_per_frame))
    ratio = decoded_mae / persistence_mae if persistence_mae > 0.0 else None
    return {
        "definition": (
            "Mean absolute error in uint8 intensity levels over resized 256x256 "
            "frames. Frame zero is unscored so decoded and persistence means use "
            "the same subsequent-frame population; persistence repeats the "
            "previous admitted real frame."
        ),
        "decoded_frame_mae_u8": decoded_mae,
        "persistence_frame_mae_u8": persistence_mae,
        "decoded_to_persistence_ratio": ratio,
        "scored_frame_count": len(decoded_frames) - 1,
    }


def _write_video(frames: Sequence[ImageArray], output: Path, fps: int) -> int | None:
    if not frames:
        raise ValueError("cannot write an empty video")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if output.suffix.lower() == ".gif":
        from PIL import Image  # noqa: PLC0415

        images = [
            Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames
        ]
        images[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=int(round(1000.0 / fps)),
            loop=0,
            optimize=False,
        )
        with Image.open(output) as animation:
            gif_frame_count = animation.n_frames
        written_frame_count = len(frames)
        if gif_frame_count != written_frame_count:
            raise RuntimeError(
                "GIF frame count mismatch: "
                f"written_frame_count={written_frame_count}, "
                f"gif_frame_count={gif_frame_count}"
            )
        return gif_frame_count
    if output.suffix.lower() != ".mp4":
        raise ValueError("output suffix must be .gif or .mp4")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write mp4 output")
    height, width = frames[0].shape[:2]
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, input=np.stack(frames).tobytes(), check=True)
    return None


def _contact_sheet_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}-frames.png")


def write_video_contact_sheet(
    frames: Sequence[ImageArray], video_path: Path
) -> tuple[Path, list[int]]:
    """Write evenly spaced composed GIF frames as one vertically stacked PNG."""
    if not frames:
        raise ValueError("cannot write a contact sheet for an empty video")
    video_path = Path(video_path)
    if video_path.suffix.lower() != ".gif":
        raise ValueError("contact sheets are written only for GIF output")
    output = _contact_sheet_path(video_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    sample_count = CONTACT_SHEET_FRAME_COUNT
    indices = np.rint(np.linspace(0, len(frames) - 1, sample_count)).astype(int)
    selected = [np.asarray(frames[index], dtype=np.uint8) for index in indices]
    first_shape = selected[0].shape
    if any(frame.shape != first_shape for frame in selected):
        raise ValueError("contact-sheet frames must have a common shape")

    from PIL import Image  # noqa: PLC0415

    height, width = first_shape[:2]
    sheet = Image.new("RGB", (width, height * sample_count))
    for row, frame in enumerate(selected):
        sheet.paste(Image.fromarray(frame), (0, row * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(
        output,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    return output, indices.tolist()


def _runtime_decoder(
    checkpoint: Path,
    *,
    vq_checkpoint: Path,
    seed_session: Path,
    seed_slice: int,
    seed_frames: TokenArray,
    device: str,
    guidance_weight: float,
    temperature: float,
) -> tuple[FrameDecoder, _TokenCollector]:
    from imas_ambix.worldmodel.flux_conditioned_decoder import (  # noqa: PLC0415
        FluxConditionedDecoder,
    )

    if not seed_session.stem.isdigit():
        raise ValueError("the seed session filename must be a numeric shot")
    configuration = {
        "checkpoint": str(checkpoint),
        "vq_decoder_path": str(vq_checkpoint),
        "vq_decoder_id": f"imagenet_256_L:{_sha256(vq_checkpoint)}",
        "vq_stage": "stub",
        "guidance_weight": guidance_weight,
        "temperature": temperature,
        "device": device,
        "seed_shot": int(seed_session.stem),
        "seed_slice": seed_slice,
        "session_root": str(seed_session.parent),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8"
    ) as stream:
        json.dump(configuration, stream)
        stream.flush()
        previous = os.environ.get("IMAS_AMBIX_FLUX_DECODER")
        os.environ["IMAS_AMBIX_FLUX_DECODER"] = stream.name
        try:
            decoder = FluxConditionedDecoder()
        finally:
            if previous is None:
                os.environ.pop("IMAS_AMBIX_FLUX_DECODER", None)
            else:
                os.environ["IMAS_AMBIX_FLUX_DECODER"] = previous
    collector = _TokenCollector()
    decoder.reset(seed_frames)
    decoder.vq_decoder = collector
    return decoder, collector


def _stub_vq(tokens: TokenArray) -> ImageArray:
    values = np.bitwise_and(tokens, 255).astype(np.uint8)
    images = np.repeat(np.repeat(values, 16, axis=1), 16, axis=2)
    return np.repeat(images[..., None], 3, axis=3)


def _decode_vq(
    tokens: TokenArray,
    *,
    route: str,
    vq_checkpoint: Path,
    device: str,
    batch_size: int,
) -> tuple[ImageArray, str, str]:
    if route == "stub":
        return _stub_vq(tokens), "stub", "explicit deterministic test stage"

    root = vq_checkpoint.parent.parent
    import_failure = ""
    if route in {"auto", "in-process"}:
        try:
            from imas_ambix.bench.stream_worker import (  # noqa: PLC0415
                decode_batch,
                load_model,
            )

            model = load_model(root, device)
            try:
                images = decode_batch(
                    model,
                    tokens,
                    device,
                    model_forward_batch=batch_size,
                    target_hw=(VIDEO_HEIGHT, VIDEO_WIDTH),
                )
            finally:
                del model
            return (
                images,
                "in-process",
                "src.Open_MAGVIT2 imported in ambix interpreter",
            )
        except (ImportError, ModuleNotFoundError) as error:
            if route == "in-process":
                raise
            import_failure = f"{type(error).__name__}: {error}"

    if route not in {"auto", "persistent-subprocess"}:
        raise ValueError(
            "vq route must be auto, in-process, persistent-subprocess, or stub"
        )
    if not DEFAULT_VQ_PYTHON.is_file():
        raise FileNotFoundError(DEFAULT_VQ_PYTHON)
    if vq_checkpoint.resolve() != DEFAULT_VQ_CHECKPOINT.resolve():
        raise ValueError(
            "the subprocess route requires the frozen default VQ checkpoint"
        )
    from imas_ambix.camdyn.reconstruction_demo import (  # noqa: PLC0415
        run_decode_subprocess,
    )

    with tempfile.TemporaryDirectory(prefix="ambix-flux-video-") as directory:
        token_bundle = Path(directory) / "tokens.npz"
        image_bundle = Path(directory) / "images.npz"
        np.savez_compressed(
            token_bundle,
            grids=(tokens + REGISTRY_OFFSET)[None],
            index=json.dumps([]),
            meta=json.dumps([]),
        )
        run_decode_subprocess(token_bundle, image_bundle, device)
        with np.load(image_bundle, allow_pickle=False) as payload:
            images = np.asarray(payload["images"][0], dtype=np.uint8)
    detail = "OpenMAGVIT2 loaded once for all frames in its dedicated interpreter"
    if import_failure:
        detail += f"; ambix import failed with {import_failure}"
    return images, "persistent-subprocess", detail


def render_session_video(
    session_path: Path,
    checkpoint: Path,
    output: Path,
    *,
    decoder: FrameDecoder | None = None,
    real_frames: ImageArray | None = None,
    frame_deltas: NDArray[np.float64] | None = None,
    vq_route: str = "auto",
    vq_checkpoint: Path = DEFAULT_VQ_CHECKPOINT,
    seed_session: Path | None = None,
    seed_slice: int = 50,
    token_root: Path = TOKEN_ROOT,
    level1_root: Path = LEVEL1_DIR,
    device: str = "cpu",
    guidance_weight: float = 1.0,
    temperature: float = 1.0,
    fps: int = DEFAULT_FPS,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Decode one recorded session, write its video, and return the receipt."""
    session_path = Path(session_path)
    checkpoint = Path(checkpoint)
    output = Path(output)
    if not session_path.is_file() or not checkpoint.is_file():
        missing = session_path if not session_path.is_file() else checkpoint
        raise FileNotFoundError(missing)
    if fps <= 0:
        raise ValueError("fps must be positive")
    session = _read_session(session_path)
    session_times = np.asarray(session["time"], dtype=np.float64)
    mode, shot, selected, manifest_slices = _manifest_selection(
        session_path, int(session.sizes["time"])
    )
    admitted_before_camera = len(selected)
    actual_seed: Path | None = None
    seed_shot: int | None = None
    seed_window: _SeedWindow | None = None
    candidate_count_before_history = 0
    if decoder is None:
        actual_seed = seed_session or (
            session_path
            if session_path.stem.isdigit()
            else DEFAULT_SESSION_ROOT / "21858.nc"
        )
        if not actual_seed.stem.isdigit():
            raise ValueError("the seed session filename must be a numeric shot")
        seed_shot = int(actual_seed.stem)
        selected = [index for index in selected if index >= seed_slice]
        candidate_count_before_history = len(selected)
        if not selected:
            raise ValueError("the session has no admitted slice at or after seed_slice")
    times = session_times[selected]
    observed_deltas = frame_deltas
    if mode == "labeller":
        if real_frames is None:
            if shot is None:
                raise RuntimeError("a labeller session has no shot identity")
            real_frames, observed_deltas, keep = _load_real_frames(
                shot, times, level1_root=level1_root
            )
            selected = [
                index
                for index, keep_row in zip(selected, keep, strict=True)
                if keep_row
            ]
            times = times[keep]
        elif len(real_frames) != len(selected):
            raise ValueError("real frame count must equal the admitted labeller slices")
    elif real_frames is not None:
        raise ValueError("real frames are only valid for a labeller session")
    if decoder is None:
        if actual_seed is None or seed_shot is None:
            raise RuntimeError("runtime decoder seed source was not resolved")
        seed_source = _read_session(actual_seed)
        seed_session_times = np.asarray(seed_source["time"], dtype=np.float64)
        seed_camera_times = _camera_times(seed_shot, level1_root=level1_root)
        seed_window = _resolve_seed_window(
            selected,
            session_times,
            seed_session_times,
            seed_camera_times,
            requested_start_slice=seed_slice,
        )
        selected = seed_window.selected
        times = session_times[selected]
        skipped = seed_window.leading_slices_skipped
        if real_frames is not None:
            real_frames = real_frames[skipped:]
        if observed_deltas is not None:
            observed_deltas = observed_deltas[skipped:]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        selected = selected[:max_frames]
        times = times[:max_frames]
        if real_frames is not None:
            real_frames = real_frames[:max_frames]
    if not selected:
        raise ValueError("the session has no admitted frames to render")

    collector = None
    if decoder is None:
        if actual_seed is None or seed_shot is None or seed_window is None:
            raise RuntimeError("runtime decoder seed history was not resolved")
        seed_frames = _load_seed_tokens(
            seed_shot,
            seed_window.camera_frame_indices,
            token_root=token_root,
        )
        decoder, collector = _runtime_decoder(
            checkpoint,
            vq_checkpoint=vq_checkpoint,
            seed_session=actual_seed,
            seed_slice=int(seed_window.session_slice_indices[-1]),
            seed_frames=seed_frames,
            device=device,
            guidance_weight=guidance_weight,
            temperature=temperature,
        )

    decoded: list[ImageArray] = []
    decode_walls: list[float] = []
    actions: list[str] = []
    keyframe_walls: list[float] = []
    for index in selected:
        frame = session.isel(time=index)
        result = decoder.decode(frame)
        decoded.append(np.asarray(result.image, dtype=np.uint8))
        decode_walls.append(float(result.decode_wall))
        actions.append(str(_frame_value(frame, "action_name", "recorded slice")))
        keyframe_walls.append(float(_frame_value(frame, "wall_seconds", 0.0)))

    vq_started = perf_counter()
    route_detail = "decoder supplied by caller"
    actual_route = "stub" if decoder is not None and collector is None else vq_route
    if collector is not None:
        token_stack = np.stack(collector.tokens)
        decoded_stack, actual_route, route_detail = _decode_vq(
            token_stack,
            route=vq_route,
            vq_checkpoint=vq_checkpoint,
            device=device,
            batch_size=8,
        )
        decoded = [image for image in decoded_stack]
    vq_wall = perf_counter() - vq_started
    pixel_error = _pixel_error_receipt(real_frames, decoded)

    frames = [
        _compose_frame(
            image,
            real=None if real_frames is None else real_frames[position],
            action=actions[position],
            keyframe_wall=keyframe_walls[position],
            slice_time=float(times[position]),
        )
        for position, image in enumerate(decoded)
    ]
    receipt_path = output.with_suffix(".receipt.json")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite {receipt_path}")
    contact_sheet_path = _contact_sheet_path(output)
    if output.suffix.lower() == ".gif" and contact_sheet_path.exists():
        raise FileExistsError(f"refusing to overwrite {contact_sheet_path}")
    gif_frame_count = _write_video(frames, output, fps)
    contact_sheet_indices: list[int] = []
    if output.suffix.lower() == ".gif":
        contact_sheet_path, contact_sheet_indices = write_video_contact_sheet(
            frames, output
        )
    max_delta = (
        float(np.max(np.abs(observed_deltas)))
        if observed_deltas is not None and observed_deltas.size
        else None
    )
    receipt: dict[str, object] = {
        "session": str(session_path.resolve()),
        "session_sha256": _sha256(session_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "decoder_identity": decoder.decoder_identity,
        "vq_checkpoint": str(vq_checkpoint.resolve()),
        "vq_route": actual_route,
        "vq_route_detail": route_detail,
        "mode": mode,
        "shot": shot,
        "manifest_slice_count": manifest_slices,
        "admitted_before_camera_join": admitted_before_camera,
        "frame_count": len(frames),
        "written_frame_count": len(frames),
        "gif_frame_count": gif_frame_count,
        "max_abs_camera_delta_s": max_delta,
        "guidance_weight": guidance_weight,
        "temperature": temperature,
        "requested_start_slice": seed_slice if seed_window is not None else None,
        "first_rendered_slice": selected[0],
        "first_rendered_time_s": float(times[0]),
        "seed_provenance": (
            {
                "session": str(actual_seed.resolve()),
                "shot": seed_shot,
                "history_spacing_s": HISTORY_SPACING_SECONDS,
                "session_slice_indices": seed_window.session_slice_indices.tolist(),
                "slice_times_s": seed_window.query_times.tolist(),
                "camera_frame_indices": seed_window.camera_frame_indices.tolist(),
                "camera_frame_times_s": (
                    seed_window.query_times + seed_window.camera_time_deltas
                ).tolist(),
                "camera_time_deltas_s": seed_window.camera_time_deltas.tolist(),
                "seeded_short_frame_count": 0,
                "leading_render_slices_skipped_for_full_history": (
                    seed_window.leading_slices_skipped
                ),
                "candidate_render_slices_at_or_after_requested_start": (
                    candidate_count_before_history
                ),
            }
            if seed_window is not None and actual_seed is not None
            else None
        ),
        "pixel_error": pixel_error,
        "actions": actions,
        "median_decode_wall_s": float(np.median(decode_walls)),
        "max_decode_wall_s": float(np.max(decode_walls)),
        "vq_batch_wall_s": vq_wall,
        "fps": fps,
        "frame_shape": list(frames[0].shape),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "contact_sheet": (
            str(contact_sheet_path.resolve()) if contact_sheet_indices else None
        ),
        "contact_sheet_sha256": (
            _sha256(contact_sheet_path) if contact_sheet_indices else None
        ),
        "contact_sheet_frame_indices": contact_sheet_indices,
        "source_revision": _source_revision(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vq-checkpoint", type=Path, default=DEFAULT_VQ_CHECKPOINT)
    parser.add_argument(
        "--vq-route",
        choices=("auto", "in-process", "persistent-subprocess", "stub"),
        default="auto",
    )
    parser.add_argument("--seed-session", type=Path)
    parser.add_argument("--seed-slice", type=int, default=50)
    parser.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--guidance-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    receipt = render_session_video(
        args.session,
        args.checkpoint,
        args.output,
        vq_route=args.vq_route,
        vq_checkpoint=args.vq_checkpoint,
        seed_session=args.seed_session,
        seed_slice=args.seed_slice,
        token_root=args.token_root,
        level1_root=args.level1_root,
        device=args.device,
        guidance_weight=args.guidance_weight,
        temperature=args.temperature,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SESSION_ROOT",
    "DEFAULT_VQ_CHECKPOINT",
    "write_video_contact_sheet",
    "render_session_video",
    "main",
]
