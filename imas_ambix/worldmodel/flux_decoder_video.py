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
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from imas_ambix.camdyn.dataset import level1_shot_path
from imas_ambix.data.paths import LEVEL1_DIR
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

ImageArray = NDArray[np.uint8]
TokenArray = NDArray[np.int64]


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


def _write_video(frames: Sequence[ImageArray], output: Path, fps: int) -> None:
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
        return
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


def _runtime_decoder(
    checkpoint: Path,
    *,
    vq_checkpoint: Path,
    seed_session: Path,
    seed_slice: int,
    device: str,
    guidance_weight: float,
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
    level1_root: Path = LEVEL1_DIR,
    device: str = "cpu",
    guidance_weight: float = 1.0,
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
    mode, shot, selected, manifest_slices = _manifest_selection(
        session_path, int(session.sizes["time"])
    )
    admitted_before_camera = len(selected)
    times = np.asarray(session["time"].isel(time=selected), dtype=np.float64)
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
        actual_seed = seed_session or (
            session_path
            if session_path.stem.isdigit()
            else DEFAULT_SESSION_ROOT / "21858.nc"
        )
        decoder, collector = _runtime_decoder(
            checkpoint,
            vq_checkpoint=vq_checkpoint,
            seed_session=actual_seed,
            seed_slice=seed_slice,
            device=device,
            guidance_weight=guidance_weight,
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
    _write_video(frames, output, fps)
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
        "max_abs_camera_delta_s": max_delta,
        "actions": actions,
        "median_decode_wall_s": float(np.median(decode_walls)),
        "max_decode_wall_s": float(np.max(decode_walls)),
        "vq_batch_wall_s": vq_wall,
        "fps": fps,
        "frame_shape": list(frames[0].shape),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
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
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--guidance-weight", type=float, default=1.0)
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
        level1_root=args.level1_root,
        device=args.device,
        guidance_weight=args.guidance_weight,
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
    "render_session_video",
    "main",
]
