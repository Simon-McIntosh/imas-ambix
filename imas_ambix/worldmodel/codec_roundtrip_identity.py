"""Measure the frozen camera codec without involving a learned predictor.

The command samples raw camera frames, sends them through the same frozen
OpenMAGVIT2 encoder and decoder used to build the camera-token corpus, and
compares the reconstructed pixels directly with the encoder input.  Encoding
and decoding happen in one persistent subprocess so the model is loaded once
and no autoregressive world-model state enters the measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from imas_ambix.data.stream_encode import DEFAULT_L1_ROOT, MODEL_FORWARD_BATCH
from imas_ambix.worldmodel.flux_decoder_video import (
    DEFAULT_VQ_CHECKPOINT,
    DEFAULT_VQ_PYTHON,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_MAGVIT_ROOT = DEFAULT_VQ_CHECKPOINT.parent.parent
DEFAULT_SHOT = 22086
DEFAULT_CAMERA = "rbb"
DEFAULT_FRAME_COUNT = 16
CONTACT_SHEET_FRAME_COUNT = 4

ImageArray = NDArray[np.uint8]


def fit_scale_and_offset(
    roundtrip: NDArray[Any], reference: NDArray[Any]
) -> tuple[float, float, float]:
    """Fit ``reference = scale * roundtrip + offset`` and return residual MAE."""
    decoded = np.asarray(roundtrip, dtype=np.float64).reshape(-1)
    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    if decoded.shape != target.shape or decoded.size == 0:
        raise ValueError("round-trip and reference arrays must be non-empty and equal")
    design = np.column_stack((decoded, np.ones(decoded.size, dtype=np.float64)))
    scale, offset = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = scale * decoded + offset
    residual_mae = float(np.mean(np.abs(target - fitted)))
    return float(scale), float(offset), residual_mae


def _frame_statistics(
    reference: ImageArray, roundtrip: ImageArray
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for index, (input_frame, decoded_frame) in enumerate(
        zip(reference, roundtrip, strict=True)
    ):
        rows.append(
            {
                "frame_index": index,
                "input_mean_u8": float(np.mean(input_frame, dtype=np.float64)),
                "input_std_u8": float(np.std(input_frame, dtype=np.float64)),
                "roundtrip_mean_u8": float(np.mean(decoded_frame, dtype=np.float64)),
                "roundtrip_std_u8": float(np.std(decoded_frame, dtype=np.float64)),
            }
        )
    return rows


def analyse_roundtrip(
    reference: ImageArray, roundtrip: ImageArray
) -> dict[str, object]:
    """Return intensity metrics and a conservative dynamic-range verdict."""
    source = np.asarray(reference)
    decoded = np.asarray(roundtrip)
    if source.shape != decoded.shape or source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError(
            "input and round-trip must share shape (frame, height, width, 3)"
        )
    if source.dtype != np.uint8 or decoded.dtype != np.uint8:
        raise TypeError("input and round-trip frames must be uint8")
    if source.shape[0] == 0:
        raise ValueError("at least one frame is required")

    difference = source.astype(np.float64) - decoded.astype(np.float64)
    mae = float(np.mean(np.abs(difference)))
    scale, offset, fitted_residual = fit_scale_and_offset(decoded, source)
    input_mean = float(np.mean(source, dtype=np.float64))
    input_std = float(np.std(source, dtype=np.float64))
    roundtrip_mean = float(np.mean(decoded, dtype=np.float64))
    roundtrip_std = float(np.std(decoded, dtype=np.float64))
    std_ratio = roundtrip_std / input_std if input_std > 0.0 else None
    mean_tolerance = max(8.0, 0.5 * input_std)
    exact_identity = bool(np.array_equal(source, decoded))
    faithful = exact_identity or (
        std_ratio is not None
        and 0.5 <= std_ratio <= 2.0
        and abs(roundtrip_mean - input_mean) <= mean_tolerance
    )
    if faithful:
        verdict = (
            "The codec preserves the input dynamic range; the token pipeline is "
            "cleared and the model is convicted."
        )
    else:
        verdict = (
            "The codec itself rescales the narrow input toward a broader, mid-grey "
            "range; the token pipeline normalisation is convicted, not the model."
        )

    return {
        "frame_count": int(source.shape[0]),
        "frame_shape": list(source.shape[1:]),
        "mean_absolute_error_u8": mae,
        "roundtrip_to_input_fit": {
            "scale": scale,
            "offset_u8": offset,
            "residual_mae_u8": fitted_residual,
        },
        "aggregate_intensity": {
            "input_mean_u8": input_mean,
            "input_std_u8": input_std,
            "roundtrip_mean_u8": roundtrip_mean,
            "roundtrip_std_u8": roundtrip_std,
            "roundtrip_to_input_std_ratio": std_ratio,
        },
        "per_frame_intensity": _frame_statistics(source, decoded),
        "decision_rule": {
            "mean_delta_tolerance_u8": mean_tolerance,
            "roundtrip_to_input_std_ratio_bounds": [0.5, 2.0],
            "exact_identity_is_faithful": True,
        },
        "dynamic_range_preserved": faithful,
        "verdict": verdict,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_codec_subprocess(
    *,
    shot: int,
    camera: str,
    frame_count: int,
    level1_root: Path,
    magvit_root: Path,
    vq_python: Path,
    vq_checkpoint: Path,
    device: str,
) -> tuple[ImageArray, ImageArray, Mapping[str, object]]:
    """Encode and decode sampled frames in one persistent codec process."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not vq_python.is_file():
        raise FileNotFoundError(vq_python)
    if not vq_checkpoint.is_file():
        raise FileNotFoundError(vq_checkpoint)
    worker = Path(__file__).resolve().parents[1] / "bench" / "stream_worker.py"

    with tempfile.TemporaryDirectory(prefix="ambix-codec-roundtrip-") as directory:
        scratch = Path(directory)
        output_dir = scratch / "arrays"
        worker_manifest = scratch / "worker.json"
        worker_report = scratch / "report.json"
        worker_manifest.write_text(
            json.dumps(
                {
                    "shots": [shot],
                    "camera": camera,
                    "l1_root": str(level1_root),
                    "magvit2_root": str(magvit_root),
                    "ckpt_path": str(vq_checkpoint),
                    "max_items_per_shot": frame_count,
                    "output_dir": str(output_dir),
                }
            ),
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [
                str(vq_python),
                str(worker),
                "--manifest",
                str(worker_manifest),
                "--device",
                device,
                "--model-forward-batch",
                str(MODEL_FORWARD_BATCH),
                "--report",
                str(worker_report),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=magvit_root,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "codec subprocess failed with exit "
                f"{completed.returncode}: {completed.stderr[-4000:]}"
            )
        report = json.loads(worker_report.read_text(encoding="utf-8"))
        if int(report.get("shots_ok", 0)) != 1:
            raise RuntimeError(f"codec subprocess did not complete the shot: {report}")
        source = np.load(output_dir / f"{shot}-src.npy", allow_pickle=False)
        decoded = np.load(output_dir / f"{shot}-decoded.npy", allow_pickle=False)
        return (
            np.asarray(source, dtype=np.uint8),
            np.asarray(decoded, dtype=np.uint8),
            report,
        )


def write_contact_sheet(
    reference: ImageArray,
    roundtrip: ImageArray,
    output: Path,
) -> list[int]:
    """Write four evenly spaced input/round-trip frame pairs."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    if reference.shape != roundtrip.shape or reference.shape[0] < 1:
        raise ValueError("contact-sheet arrays must be non-empty and shape-aligned")
    indices = np.rint(
        np.linspace(0, reference.shape[0] - 1, CONTACT_SHEET_FRAME_COUNT)
    ).astype(int)
    height, width = reference.shape[1:3]
    header = 22
    sheet = Image.new(
        "RGB", (2 * width, CONTACT_SHEET_FRAME_COUNT * (height + header)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(indices):
        y = row * (height + header)
        draw.text((5, y + 5), f"input frame {index}", fill="white")
        draw.text((width + 5, y + 5), f"codec round-trip {index}", fill="white")
        sheet.paste(Image.fromarray(reference[index]), (0, y + header))
        sheet.paste(Image.fromarray(roundtrip[index]), (width, y + header))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)
    return indices.tolist()


def generate_receipt(
    *,
    output_json: Path,
    contact_sheet: Path,
    shot: int = DEFAULT_SHOT,
    camera: str = DEFAULT_CAMERA,
    frame_count: int = DEFAULT_FRAME_COUNT,
    level1_root: Path = DEFAULT_L1_ROOT,
    magvit_root: Path = DEFAULT_MAGVIT_ROOT,
    vq_python: Path = DEFAULT_VQ_PYTHON,
    vq_checkpoint: Path = DEFAULT_VQ_CHECKPOINT,
    device: str = "cuda",
) -> dict[str, object]:
    """Run the isolated codec check and write its JSON and visual evidence."""
    reference, roundtrip, worker_report = run_codec_subprocess(
        shot=shot,
        camera=camera,
        frame_count=frame_count,
        level1_root=level1_root,
        magvit_root=magvit_root,
        vq_python=vq_python,
        vq_checkpoint=vq_checkpoint,
        device=device,
    )
    import zarr  # noqa: PLC0415

    camera_group = zarr.open_group(str(level1_root / f"{shot}.zarr"), mode="r")[camera]
    raw_camera_dtype = str(camera_group["data"].dtype)
    payload = analyse_roundtrip(reference, roundtrip)
    indices = write_contact_sheet(reference, roundtrip, contact_sheet)
    payload.update(
        {
            "shot": shot,
            "camera": camera,
            "input_dtype": str(reference.dtype),
            "raw_camera_dtype": raw_camera_dtype,
            "input_preprocessing": (
                "Raw uint8 is passed through unchanged before RGB replication; "
                "the codec input then follows the corpus uint8-to-[-1,1] transform."
            ),
            "roundtrip_dtype": str(roundtrip.dtype),
            "codec_route": "persistent-subprocess",
            "learned_predictor_in_loop": False,
            "tokenizer_name": worker_report.get("tokenizer_name"),
            "model_forward_batch": worker_report.get("model_forward_batch"),
            "vq_checkpoint": str(vq_checkpoint.resolve()),
            "vq_checkpoint_sha256": _sha256(vq_checkpoint),
            "contact_sheet": str(contact_sheet),
            "contact_sheet_sha256": _sha256(contact_sheet),
            "contact_sheet_frame_indices": indices,
            "codec_worker": dict(worker_report),
            "source_revision": _source_revision(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_json)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--level1-root", type=Path, default=DEFAULT_L1_ROOT)
    parser.add_argument("--magvit-root", type=Path, default=DEFAULT_MAGVIT_ROOT)
    parser.add_argument("--vq-python", type=Path, default=DEFAULT_VQ_PYTHON)
    parser.add_argument("--vq-checkpoint", type=Path, default=DEFAULT_VQ_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    payload = generate_receipt(
        output_json=args.output_json,
        contact_sheet=args.contact_sheet,
        shot=args.shot,
        camera=args.camera,
        frame_count=args.frame_count,
        level1_root=args.level1_root,
        magvit_root=args.magvit_root,
        vq_python=args.vq_python,
        vq_checkpoint=args.vq_checkpoint,
        device=args.device,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "analyse_roundtrip",
    "fit_scale_and_offset",
    "generate_receipt",
    "main",
    "run_codec_subprocess",
    "write_contact_sheet",
]
