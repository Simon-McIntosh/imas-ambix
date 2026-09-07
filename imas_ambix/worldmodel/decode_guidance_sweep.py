"""Measure decoder guidance and temperature on a held-out camera session.

The sweep keeps checkpoint, history seed, sample seed, session slices, and VQ
decode fixed while varying only classifier-free guidance and temperature.  It
uses the production video renderer's session selection, camera join, batched VQ
route, frame composition, and contact-sheet writer so the visual and scalar
evidence describe the same decoded frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.worldmodel.flux_decoder_video import (
    DEFAULT_SESSION_ROOT,
    DEFAULT_VQ_CHECKPOINT,
    _compose_frame,
    _decode_vq,
    _load_real_frames,
    _manifest_selection,
    _read_session,
    _resize,
    _runtime_decoder,
    write_video_contact_sheet,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

DEFAULT_CHECKPOINT = Path(
    "/work/projects/imas_gpu/ambix/flux_decoder/overnight-20260907c/"
    "checkpoint-000077258.pt"
)
DEFAULT_SESSION = DEFAULT_SESSION_ROOT / "22086.nc"
DEFAULT_GUIDANCE_WEIGHTS = (1.0, 2.0, 3.5, 6.0)
DEFAULT_TEMPERATURES = (0.7, 1.0)
COLLAPSE_STD_FRACTION = 0.25
DEFAULT_SAMPLE_SEED = 22086

FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.uint8]
TokenArray = NDArray[np.int64]


def guided_token_probabilities(
    conditional_logits: torch.Tensor,
    unconditional_logits: torch.Tensor,
    *,
    guidance_weight: float,
    temperature: float,
) -> torch.Tensor:
    """Return the distribution sampled by classifier-free guided decoding."""
    if conditional_logits.shape != unconditional_logits.shape:
        raise ValueError("conditional and unconditional logits must have equal shape")
    if not np.isfinite(guidance_weight) or guidance_weight < 0.0:
        raise ValueError("guidance_weight must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = unconditional_logits + guidance_weight * (
        conditional_logits - unconditional_logits
    )
    return torch.softmax(logits / temperature, dim=-1)


def mean_collapse_mask(
    decoded_frames: NDArray[Any],
    real_frames: NDArray[Any],
    *,
    threshold_fraction: float = COLLAPSE_STD_FRACTION,
) -> NDArray[np.bool_]:
    """Mark decoded frames with less than a fraction of real-frame variation."""
    decoded = np.asarray(decoded_frames, dtype=np.float64)
    real = np.asarray(real_frames, dtype=np.float64)
    if decoded.shape != real.shape or decoded.ndim < 2:
        raise ValueError("decoded and real frame stacks must have equal shapes")
    if not np.isfinite(threshold_fraction) or threshold_fraction <= 0.0:
        raise ValueError("threshold_fraction must be finite and positive")
    axes = tuple(range(1, decoded.ndim))
    decoded_std = np.std(decoded, axis=axes)
    real_std = np.std(real, axis=axes)
    return np.asarray(
        (real_std > 0.0) & (decoded_std < threshold_fraction * real_std),
        dtype=np.bool_,
    )


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


def _cell_stem(guidance_weight: float, temperature: float) -> str:
    guidance = str(guidance_weight).replace(".", "p")
    heat = str(temperature).replace(".", "p")
    return f"guidance-{guidance}-temperature-{heat}"


def _score_cell(decoded: ImageArray, real: ImageArray) -> dict[str, object]:
    if decoded.shape != real.shape or decoded.ndim != 4:
        raise ValueError(
            "decoded and real videos must have equal four-dimensional shapes"
        )
    if decoded.shape[0] < 2:
        raise ValueError("at least two paired frames are required for persistence")
    decoded_float = decoded.astype(np.float32)
    real_float = real.astype(np.float32)
    target = real_float[1:]
    decoded_mae = float(np.mean(np.abs(decoded_float[1:] - target)))
    persistence_mae = float(np.mean(np.abs(real_float[:-1] - target)))
    ratio = decoded_mae / persistence_mae if persistence_mae > 0.0 else float("inf")
    collapse = mean_collapse_mask(decoded_float, real_float)
    return {
        "frame_count": int(decoded.shape[0]),
        "scored_frame_count": int(decoded.shape[0] - 1),
        "decoded_mean_absolute_error": decoded_mae,
        "persistence_mean_absolute_error": persistence_mae,
        "decoded_to_persistence_ratio": ratio,
        "mean_collapse_frame_count": int(np.count_nonzero(collapse)),
        "mean_collapse_frame_indices": np.flatnonzero(collapse).tolist(),
        "mean_collapse_std_fraction": COLLAPSE_STD_FRACTION,
    }


def _verdict_line(rows: Sequence[dict[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot form a verdict without sweep rows")
    best = min(rows, key=lambda row: float(row["decoded_to_persistence_ratio"]))
    winners = [row for row in rows if float(row["decoded_to_persistence_ratio"]) < 1.0]
    cell = (
        f"guidance_weight={best['guidance_weight']}, temperature={best['temperature']}"
    )
    ratio = float(best["decoded_to_persistence_ratio"])
    if winners:
        return (
            f"{len(winners)} of {len(rows)} cells beat persistence; best {cell} "
            f"at decoded-to-persistence MAE ratio {ratio:.6f}."
        )
    return (
        f"No cell beats persistence; best {cell} at decoded-to-persistence "
        f"MAE ratio {ratio:.6f}."
    )


def run_guidance_sweep(
    session_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    guidance_weights: Iterable[float] = DEFAULT_GUIDANCE_WEIGHTS,
    temperatures: Iterable[float] = DEFAULT_TEMPERATURES,
    vq_checkpoint: Path = DEFAULT_VQ_CHECKPOINT,
    level1_root: Path = LEVEL1_DIR,
    device: str = "cuda",
    seed_slice: int = 43,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    vq_route: str = "in-process",
) -> dict[str, object]:
    """Run every guidance-temperature cell and write contact sheets plus verdict."""
    session_path = Path(session_path)
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    for required in (session_path, checkpoint, vq_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    guidance_values = tuple(float(value) for value in guidance_weights)
    temperature_values = tuple(float(value) for value in temperatures)
    if not guidance_values or not temperature_values:
        raise ValueError("the sweep needs at least one guidance and temperature")
    if len(set(guidance_values)) != len(guidance_values):
        raise ValueError("guidance weights must be unique")
    if len(set(temperature_values)) != len(temperature_values):
        raise ValueError("temperatures must be unique")
    for guidance in guidance_values:
        guided_token_probabilities(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            guidance_weight=guidance,
            temperature=1.0,
        )
    for temperature in temperature_values:
        guided_token_probabilities(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            guidance_weight=1.0,
            temperature=temperature,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = output_dir / "verdict.json"
    if verdict_path.exists():
        raise FileExistsError(f"refusing to overwrite {verdict_path}")

    session = _read_session(session_path)
    mode, shot, selected, manifest_slice_count = _manifest_selection(
        session_path, int(session.sizes["time"])
    )
    if mode != "labeller" or shot is None:
        raise ValueError("the sweep requires a labeller session with a real camera")
    times = np.asarray(session["time"].isel(time=selected), dtype=np.float64)
    real_frames, camera_deltas, keep = _load_real_frames(
        shot, times, level1_root=level1_root
    )
    selected = [
        index for index, keep_row in zip(selected, keep, strict=True) if keep_row
    ]
    times = times[keep]
    if len(selected) < 2:
        raise ValueError("the camera join produced fewer than two paired frames")
    resized_real = np.stack([_resize(frame) for frame in real_frames])

    decoder, collector = _runtime_decoder(
        checkpoint,
        vq_checkpoint=vq_checkpoint,
        seed_session=session_path,
        seed_slice=seed_slice,
        device=device,
        guidance_weight=1.0,
    )
    seed_history = np.stack(decoder._history)
    tokens_by_cell: list[TokenArray] = []
    decode_walls_by_cell: list[list[float]] = []
    cell_parameters: list[tuple[float, float]] = []
    token_started = perf_counter()
    for guidance in guidance_values:
        for temperature in temperature_values:
            decoder.reset(seed_history)
            decoder.guidance_weight = guidance
            decoder.temperature = temperature
            decoder.generator.manual_seed(sample_seed)
            collector.tokens.clear()
            decode_walls: list[float] = []
            for index in selected:
                result = decoder.decode(session.isel(time=index))
                decode_walls.append(float(result.decode_wall))
            tokens_by_cell.append(np.stack(collector.tokens))
            decode_walls_by_cell.append(decode_walls)
            cell_parameters.append((guidance, temperature))
    token_decode_wall = perf_counter() - token_started
    decoder_identity = decoder.decoder_identity
    del decoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    combined_tokens = np.concatenate(tokens_by_cell, axis=0)
    vq_started = perf_counter()
    original_directory = Path.cwd()
    try:
        # OpenMAGVIT2 resolves its already-downloaded LPIPS weights relative to
        # this root even though inference never uses the loss module.
        os.chdir(vq_checkpoint.parent.parent)
        combined_images, actual_vq_route, vq_route_detail = _decode_vq(
            combined_tokens,
            route=vq_route,
            vq_checkpoint=vq_checkpoint,
            device=device,
            batch_size=8,
        )
    finally:
        os.chdir(original_directory)
    vq_wall = perf_counter() - vq_started

    rows: list[dict[str, object]] = []
    cursor = 0
    frame_count = len(selected)
    for (guidance, temperature), decode_walls in zip(
        cell_parameters, decode_walls_by_cell, strict=True
    ):
        decoded = np.asarray(
            combined_images[cursor : cursor + frame_count], dtype=np.uint8
        )
        cursor += frame_count
        score = _score_cell(decoded, resized_real)
        composed = [
            _compose_frame(
                decoded[position],
                real=real_frames[position],
                action=(f"guidance {guidance:g}, temperature {temperature:g}"),
                keyframe_wall=decode_walls[position],
                slice_time=float(times[position]),
            )
            for position in range(frame_count)
        ]
        cell_stem = _cell_stem(guidance, temperature)
        sheet_path, sheet_indices = write_video_contact_sheet(
            composed, output_dir / f"{cell_stem}.gif"
        )
        row: dict[str, object] = {
            "guidance_weight": guidance,
            "temperature": temperature,
            **score,
            "median_token_decode_wall_seconds": float(np.median(decode_walls)),
            "max_token_decode_wall_seconds": float(np.max(decode_walls)),
            "contact_sheet": sheet_path.name,
            "contact_sheet_sha256": _sha256(sheet_path),
            "contact_sheet_frame_indices": sheet_indices,
            "recognisable_plasma_column": None,
        }
        rows.append(row)

    if cursor != int(combined_images.shape[0]):
        raise RuntimeError("decoded VQ frame count does not match the sweep cells")
    payload: dict[str, object] = {
        "schema_version": 1,
        "session": str(session_path.resolve()),
        "session_sha256": _sha256(session_path),
        "shot": shot,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "vq_checkpoint": str(vq_checkpoint.resolve()),
        "vq_checkpoint_sha256": _sha256(vq_checkpoint),
        "decoder_identity": decoder_identity,
        "source_revision": _source_revision(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sample_seed": sample_seed,
        "seed_slice": seed_slice,
        "manifest_slice_count": manifest_slice_count,
        "paired_frame_count": frame_count,
        "max_abs_camera_delta_seconds": float(np.max(np.abs(camera_deltas))),
        "token_decode_wall_seconds": token_decode_wall,
        "vq_batch_wall_seconds": vq_wall,
        "vq_route": actual_vq_route,
        "vq_route_detail": vq_route_detail,
        "cells": rows,
        "verdict": _verdict_line(rows),
        "visual_verdict": "Pending direct review of the eight contact sheets.",
    }
    verdict_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def record_visual_assessment(
    output_dir: Path,
    *,
    recognisable_cells: Iterable[tuple[float, float]],
    note: str,
) -> dict[str, object]:
    """Record a direct contact-sheet judgement without rerunning GPU inference."""
    if not note.strip():
        raise ValueError("visual assessment note must be non-empty")
    verdict_path = Path(output_dir) / "verdict.json"
    payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    rows = payload.get("cells")
    if not isinstance(rows, list):
        raise ValueError("verdict contains no sweep cell list")
    selected = {
        (float(guidance), float(temperature))
        for guidance, temperature in recognisable_cells
    }
    available = {
        (float(row["guidance_weight"]), float(row["temperature"]))
        for row in rows
        if isinstance(row, dict)
    }
    unknown = selected - available
    if unknown:
        raise ValueError(f"visual assessment names unknown cells: {sorted(unknown)}")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("verdict contains a non-object sweep row")
        key = (float(row["guidance_weight"]), float(row["temperature"]))
        row["recognisable_plasma_column"] = key in selected
    if selected:
        cells = ", ".join(
            f"guidance_weight={guidance:g}, temperature={temperature:g}"
            for guidance, temperature in sorted(selected)
        )
        payload["visual_verdict"] = f"Recognisable plasma column in {cells}. {note}"
    else:
        payload["visual_verdict"] = (
            f"No sweep cell renders a recognisable plasma column. {note}"
        )
    verdict_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vq-checkpoint", type=Path, default=DEFAULT_VQ_CHECKPOINT)
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-slice", type=int, default=43)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--vq-route",
        choices=("auto", "in-process", "persistent-subprocess", "stub"),
        default="in-process",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed guidance and temperature sweep."""
    args = _parser().parse_args(argv)
    payload = run_guidance_sweep(
        args.session,
        args.checkpoint,
        args.output_dir,
        vq_checkpoint=args.vq_checkpoint,
        level1_root=args.level1_root,
        device=args.device,
        seed_slice=args.seed_slice,
        sample_seed=args.sample_seed,
        vq_route=args.vq_route,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "guided_token_probabilities",
    "mean_collapse_mask",
    "record_visual_assessment",
    "run_guidance_sweep",
    "main",
]
