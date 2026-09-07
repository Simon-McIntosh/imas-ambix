"""Render the held-out camera window and attach phase-resolved error evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from imas_ambix.worldmodel import flux_decoder_video as video

SESSION = Path("/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22086.nc")
CHECKPOINT = Path(
    "/work/projects/imas_gpu/ambix/flux_decoder/overnight-20260907c/"
    "checkpoint-000077258.pt"
)
OUTPUT = Path(__file__).with_name("full-window-22086.gif")
PHASE_BOUNDARY_S = 0.150


def _selected_frame_times() -> np.ndarray:
    session = video._read_session(SESSION)
    session_times = np.asarray(session["time"], dtype=np.float64)
    mode, shot, selected, _ = video._manifest_selection(
        SESSION, int(session.sizes["time"])
    )
    if mode != "labeller" or shot != 22086:
        raise RuntimeError(f"unexpected session identity: mode={mode}, shot={shot}")
    _, _, keep = video._load_real_frames(
        shot, session_times[selected], level1_root=video.LEVEL1_DIR
    )
    selected = [
        index for index, keep_row in zip(selected, keep, strict=True) if keep_row
    ]
    seed_window = video._resolve_seed_window(
        selected,
        session_times,
        session_times,
        video._camera_times(shot, level1_root=video.LEVEL1_DIR),
        requested_start_slice=0,
    )
    return session_times[seed_window.selected]


def _summary(decoded: np.ndarray, persistence: np.ndarray) -> dict[str, Any]:
    decoded_mean = float(np.mean(decoded))
    persistence_mean = float(np.mean(persistence))
    return {
        "decoded_frame_mae_u8": decoded_mean,
        "persistence_frame_mae_u8": persistence_mean,
        "decoded_to_persistence_ratio": (
            decoded_mean / persistence_mean if persistence_mean > 0.0 else None
        ),
        "scored_transition_count": int(decoded.size),
    }


def write_eight_frame_contact_sheet() -> tuple[Path, list[int]]:
    """Replace the default still with eight evenly spaced rendered frames."""
    from PIL import Image

    with Image.open(OUTPUT) as animation:
        indices = np.rint(np.linspace(0, animation.n_frames - 1, 8)).astype(int)
        frames = []
        for index in indices:
            animation.seek(int(index))
            frames.append(animation.convert("RGB").copy())
    width, height = frames[0].size
    sheet = Image.new("RGB", (width, height * len(frames)))
    for row, frame in enumerate(frames):
        sheet.paste(frame, (0, row * height))
    output = OUTPUT.with_name(f"{OUTPUT.stem}-frames.png")
    sheet.quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(
        output,
        format="PNG",
        optimize=True,
        compress_level=9,
    )
    return output, indices.tolist()


def main() -> int:
    frame_times = _selected_frame_times()
    captured: dict[str, np.ndarray] = {}
    original_pixel_error = video._pixel_error_receipt

    def pixel_error_with_arrays(real_frames: Any, decoded_frames: Any) -> Any:
        receipt = original_pixel_error(real_frames, decoded_frames)
        if receipt is None:
            return receipt
        real_rgb = video._as_rgb_uint8(real_frames)
        real = np.stack([video._resize(frame) for frame in real_rgb]).astype(np.float64)
        decoded = np.stack([video._resize(frame) for frame in decoded_frames]).astype(
            np.float64
        )
        captured["decoded"] = np.mean(np.abs(decoded - real), axis=(1, 2, 3))
        captured["persistence"] = np.mean(np.abs(real[1:] - real[:-1]), axis=(1, 2, 3))
        return receipt

    video._pixel_error_receipt = pixel_error_with_arrays
    try:
        receipt = video.render_session_video(
            SESSION,
            CHECKPOINT,
            OUTPUT,
            vq_route="persistent-subprocess",
            seed_session=SESSION,
            seed_slice=0,
            device="cuda",
            guidance_weight=2.0,
            temperature=1.0,
            sample_seed=22086,
            fps=10,
        )
    finally:
        video._pixel_error_receipt = original_pixel_error

    contact_sheet, contact_sheet_indices = write_eight_frame_contact_sheet()
    receipt["contact_sheet"] = str(contact_sheet.resolve())
    receipt["contact_sheet_frame_indices"] = contact_sheet_indices
    receipt["contact_sheet_frame_count"] = len(contact_sheet_indices)
    receipt["contact_sheet_sha256"] = video._sha256(contact_sheet)

    decoded_per_frame = captured["decoded"]
    persistence_per_transition = captured["persistence"]
    if decoded_per_frame.size != frame_times.size:
        raise RuntimeError(
            f"captured {decoded_per_frame.size} frames for {frame_times.size} times"
        )
    decoded_per_transition = decoded_per_frame[1:]
    scored_times = frame_times[1:]
    early = scored_times < PHASE_BOUNDARY_S
    late = ~early

    receipt["aggregate_error"] = _summary(
        decoded_per_transition, persistence_per_transition
    )
    receipt["phase_error"] = {
        "definition": (
            "Transitions are assigned by target-frame time; early is t < 0.150 s "
            "and late is t >= 0.150 s."
        ),
        "boundary_time_s": PHASE_BOUNDARY_S,
        "early": _summary(
            decoded_per_transition[early], persistence_per_transition[early]
        ),
        "late": _summary(
            decoded_per_transition[late], persistence_per_transition[late]
        ),
    }
    receipt["per_frame_error"] = {
        "frame_times_s": frame_times.tolist(),
        "decoded_frame_mae_u8": decoded_per_frame.tolist(),
        "scored_transition_target_times_s": scored_times.tolist(),
        "decoded_frame_mae_u8_scored": decoded_per_transition.tolist(),
        "persistence_frame_mae_u8_scored": persistence_per_transition.tolist(),
    }
    receipt["contaminated_publication"] = {
        "decoded_frame_mae_u8": 18.096,
        "persistence_frame_mae_u8": 6.253,
        "decoded_to_persistence_ratio": 2.894,
        "scored_transition_count": 45,
        "status": "withdrawn_due_to_misaligned_seed_history",
    }
    receipt["slurm_resources"] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": "gpu_0003_grpA",
        "account": "grpa",
        "gpu_count": 1,
        "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
        "qos": None,
        "memory": "80G",
    }
    receipt["publication_paths"] = {
        "gif": (
            "/imas-ambix/figures/physics-carried-playable-plasma/seed-window/"
            "full-window-22086.gif"
        ),
        "contact_sheet": (
            "/imas-ambix/figures/physics-carried-playable-plasma/seed-window/"
            "full-window-22086-frames.png"
        ),
        "receipt": (
            "/imas-ambix/figures/physics-carried-playable-plasma/seed-window/"
            "full-window-22086.receipt.json"
        ),
    }
    receipt_path = OUTPUT.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
