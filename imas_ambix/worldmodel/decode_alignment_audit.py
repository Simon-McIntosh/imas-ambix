"""Separate alignment, photometric, and residual content errors in paired videos.

The paired decoder videos place a 32-pixel annotation banner above a real
256-by-256 camera panel and a decoded panel of the same size.  This module
operates only on those rendered pixels: it does not reopen a camera card,
token store, labeller session, or model checkpoint.

Temporal lag is signed as ``real_index - decoded_index``.  Spatial displacement
is signed as ``decoded - real`` in image coordinates, where positive x points
right and positive y points down.  The translation to apply to the decoded
panel therefore has the opposite sign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PANEL_WIDTH = 256
PANEL_HEIGHT = 256
BANNER_HEIGHT = 32
DEFAULT_FRAME_SPACING_S = 0.005

FloatArray = NDArray[np.float64]
ImageStack = NDArray[np.uint8]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_gif_panels(
    path: Path,
    *,
    banner_height: int = BANNER_HEIGHT,
    panel_width: int = PANEL_WIDTH,
    panel_height: int = PANEL_HEIGHT,
) -> tuple[ImageStack, ImageStack]:
    """Load a paired GIF and return ``(real, decoded)`` RGB panel stacks."""
    from PIL import Image, ImageSequence  # noqa: PLC0415

    frames: list[NDArray[np.uint8]] = []
    with Image.open(path) as animation:
        for frame in ImageSequence.Iterator(animation):
            frames.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
    if not frames:
        raise ValueError(f"{path} contains no frames")
    expected_shape = (banner_height + panel_height, 2 * panel_width, 3)
    if any(frame.shape != expected_shape for frame in frames):
        shapes = sorted({frame.shape for frame in frames})
        raise ValueError(
            f"paired GIF frames must have shape {expected_shape}, got {shapes}"
        )
    stack = np.stack(frames)
    panels = stack[:, banner_height : banner_height + panel_height]
    real = panels[:, :, :panel_width]
    decoded = panels[:, :, panel_width:]
    return real, decoded


def mean_absolute_error_matrix(decoded: NDArray[Any], real: NDArray[Any]) -> FloatArray:
    """Return MAE for every decoded-frame/real-frame combination."""
    decoded_array, real_array = _validate_stacks(decoded, real)
    matrix = np.empty((decoded_array.shape[0], real_array.shape[0]), dtype=np.float64)
    real_float = real_array.astype(np.float64)
    for index, frame in enumerate(decoded_array):
        image_axes = tuple(range(1, real_float.ndim))
        matrix[index] = np.mean(
            np.abs(real_float - frame.astype(np.float64)), axis=image_axes
        )
    return matrix


def temporal_lag_curve(
    mae_matrix: NDArray[Any], *, min_overlap: int | None = None
) -> tuple[list[dict[str, float | int | bool]], int]:
    """Average matrix diagonals and return the best sufficiently supported lag."""
    matrix = np.asarray(mae_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("MAE matrix must be a non-empty finite two-dimensional array")
    max_overlap = min(matrix.shape)
    required = max(3, (max_overlap + 1) // 2) if min_overlap is None else min_overlap
    if required < 1 or required > max_overlap:
        raise ValueError(f"min_overlap must lie in [1, {max_overlap}], got {required}")

    curve: list[dict[str, float | int | bool]] = []
    for lag in range(-(matrix.shape[0] - 1), matrix.shape[1]):
        decoded_indices, real_indices = _lag_indices(matrix.shape, lag)
        error = float(np.mean(matrix[decoded_indices, real_indices]))
        curve.append(
            {
                "lag_frames": lag,
                "overlap_frame_count": int(decoded_indices.size),
                "mean_absolute_error_u8": error,
                "eligible_for_selection": bool(decoded_indices.size >= required),
            }
        )
    eligible = [point for point in curve if point["eligible_for_selection"]]
    best = min(
        eligible,
        key=lambda point: (
            float(point["mean_absolute_error_u8"]),
            abs(int(point["lag_frames"])),
        ),
    )
    return curve, int(best["lag_frames"])


def brightness_centroid(frame: NDArray[Any]) -> tuple[float, float]:
    """Return the intensity-weighted ``(x, y)`` centroid of one image."""
    values = np.asarray(frame, dtype=np.float64)
    if values.ndim == 3:
        values = np.mean(values, axis=-1)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("frame must be a finite two- or three-dimensional image")
    total = float(np.sum(values))
    if total <= 0.0:
        return float("nan"), float("nan")
    y_grid, x_grid = np.indices(values.shape, dtype=np.float64)
    return (
        float(np.sum(values * x_grid) / total),
        float(np.sum(values * y_grid) / total),
    )


def brightness_centroid_displacements(
    decoded: NDArray[Any], real: NDArray[Any]
) -> FloatArray:
    """Return per-frame ``decoded - real`` centroid displacement in pixels."""
    decoded_array, real_array = _validate_stacks(decoded, real, equal_count=True)
    displacement = np.empty((decoded_array.shape[0], 2), dtype=np.float64)
    for index, (decoded_frame, real_frame) in enumerate(
        zip(decoded_array, real_array, strict=True)
    ):
        decoded_x, decoded_y = brightness_centroid(decoded_frame)
        real_x, real_y = brightness_centroid(real_frame)
        displacement[index] = (decoded_x - real_x, decoded_y - real_y)
    return displacement


def least_squares_intensity_mapping(
    decoded: NDArray[Any], real: NDArray[Any]
) -> dict[str, float]:
    """Fit ``real = scale * decoded + offset`` and report its residual MAE."""
    decoded_array, real_array = _validate_stacks(decoded, real, equal_count=True)
    source = decoded_array.astype(np.float64).reshape(-1)
    target = real_array.astype(np.float64).reshape(-1)
    source_mean = float(np.mean(source))
    target_mean = float(np.mean(target))
    source_variance = float(np.sum((source - source_mean) ** 2))
    scale = (
        0.0
        if source_variance == 0.0
        else float(np.sum((source - source_mean) * (target - target_mean)))
        / source_variance
    )
    offset = target_mean - scale * source_mean
    residual = float(np.mean(np.abs(scale * source + offset - target)))
    return {
        "scale": scale,
        "offset_u8": offset,
        "residual_mean_absolute_error_u8": residual,
    }


def translate_images(images: NDArray[Any], *, dx_px: int, dy_px: int) -> NDArray[Any]:
    """Translate an image stack without wraparound, filling exposed pixels with zero."""
    array = np.asarray(images)
    if array.ndim not in (3, 4):
        raise ValueError("images must have shape (frame, height, width[, channel])")
    output = np.zeros_like(array)
    height, width = array.shape[1:3]
    source_x_start = max(0, -dx_px)
    source_x_stop = min(width, width - dx_px)
    source_y_start = max(0, -dy_px)
    source_y_stop = min(height, height - dy_px)
    if source_x_start >= source_x_stop or source_y_start >= source_y_stop:
        return output
    target_x_start = source_x_start + dx_px
    target_x_stop = source_x_stop + dx_px
    target_y_start = source_y_start + dy_px
    target_y_stop = source_y_stop + dy_px
    output[:, target_y_start:target_y_stop, target_x_start:target_x_stop, ...] = array[
        :, source_y_start:source_y_stop, source_x_start:source_x_stop, ...
    ]
    return output


def audit_alignment(
    real: NDArray[Any],
    decoded: NDArray[Any],
    *,
    frame_spacing_s: float = DEFAULT_FRAME_SPACING_S,
    min_overlap: int | None = None,
    reported_error_u8: float | None = None,
) -> dict[str, Any]:
    """Audit temporal, spatial, and intensity explanations for paired panels."""
    decoded_array, real_array = _validate_stacks(decoded, real, equal_count=True)
    matrix = mean_absolute_error_matrix(decoded_array, real_array)
    curve, best_lag = temporal_lag_curve(matrix, min_overlap=min_overlap)
    lag_zero = _curve_point(curve, 0)
    best_point = _curve_point(curve, best_lag)

    zero_displacement = brightness_centroid_displacements(decoded_array, real_array)
    spatial = _spatial_summary(zero_displacement)
    correction_dx = -int(np.rint(spatial["median_dx_px"]))
    correction_dy = -int(np.rint(spatial["median_dy_px"]))
    translated_zero = translate_images(
        decoded_array, dx_px=correction_dx, dy_px=correction_dy
    )
    spatial_residual = _paired_mae(translated_zero, real_array)
    intensity = least_squares_intensity_mapping(decoded_array, real_array)

    best_decoded, best_real = _pairs_at_lag(decoded_array, real_array, best_lag)
    best_displacement = brightness_centroid_displacements(best_decoded, best_real)
    best_spatial = _spatial_summary(best_displacement)
    best_dx = -int(np.rint(best_spatial["median_dx_px"]))
    best_dy = -int(np.rint(best_spatial["median_dy_px"]))
    translated_best = translate_images(best_decoded, dx_px=best_dx, dy_px=best_dy)
    translated_best_error = _paired_mae(translated_best, best_real)
    if translated_best_error >= float(best_point["mean_absolute_error_u8"]):
        translated_best = best_decoded
        translated_best_error = float(best_point["mean_absolute_error_u8"])
        best_dx = 0
        best_dy = 0
    joint_intensity = least_squares_intensity_mapping(translated_best, best_real)
    joint_intensity_error = float(joint_intensity["residual_mean_absolute_error_u8"])
    intensity_applied = joint_intensity_error < translated_best_error
    final_error = joint_intensity_error if intensity_applied else translated_best_error
    baseline_error = float(lag_zero["mean_absolute_error_u8"])
    temporal_share = max(
        0.0, baseline_error - float(best_point["mean_absolute_error_u8"])
    )
    spatial_share = max(
        0.0, float(best_point["mean_absolute_error_u8"]) - translated_best_error
    )
    intensity_share = max(0.0, translated_best_error - final_error)
    shares = {
        "temporal_lag": temporal_share,
        "spatial_translation": spatial_share,
        "intensity_scale": intensity_share,
        "genuine_content_error": final_error,
    }
    largest = max(shares, key=shares.__getitem__)
    label = largest.replace("_", " ")
    reference_error = baseline_error if reported_error_u8 is None else reported_error_u8
    verdict = (
        f"{label.capitalize()} accounts for the largest share of the "
        f"{reference_error:.2f}-pixel error ({shares[largest]:.2f} pixels)."
    )
    return {
        "frame_count": int(decoded_array.shape[0]),
        "frame_shape": list(decoded_array.shape[1:]),
        "reported_decoded_frame_mae_u8": reported_error_u8,
        "measured_lag_zero_mae_u8": baseline_error,
        "measured_minus_reported_mae_u8": None
        if reported_error_u8 is None
        else baseline_error - reported_error_u8,
        "mae_matrix_u8": matrix.tolist(),
        "temporal": {
            "lag_sign_convention": "real_index - decoded_index",
            "frame_spacing_s": frame_spacing_s,
            "selection_minimum_overlap_frames": min_overlap
            if min_overlap is not None
            else max(3, (decoded_array.shape[0] + 1) // 2),
            "best_lag_frames": best_lag,
            "best_lag_s": best_lag * frame_spacing_s,
            "lag_zero_mean_absolute_error_u8": baseline_error,
            "best_lag_mean_absolute_error_u8": float(
                best_point["mean_absolute_error_u8"]
            ),
            "per_lag": curve,
        },
        "spatial_translation": {
            "displacement_sign_convention": "decoded_minus_real_xy",
            "centroid_displacement_px_per_frame": zero_displacement.tolist(),
            **spatial,
            "translation_to_apply_dx_px": correction_dx,
            "translation_to_apply_dy_px": correction_dy,
            "residual_mean_absolute_error_u8": spatial_residual,
        },
        "intensity_mapping": {
            "definition": "real = scale * decoded + offset",
            **intensity,
        },
        "combined_correction": {
            "lag_frames": best_lag,
            "translation_to_apply_dx_px": best_dx,
            "translation_to_apply_dy_px": best_dy,
            "fitted_intensity_scale": float(joint_intensity["scale"]),
            "fitted_intensity_offset_u8": float(joint_intensity["offset_u8"]),
            "fitted_intensity_residual_mean_absolute_error_u8": joint_intensity_error,
            "intensity_mapping_applied": intensity_applied,
            "applied_intensity_scale": float(joint_intensity["scale"])
            if intensity_applied
            else 1.0,
            "applied_intensity_offset_u8": float(joint_intensity["offset_u8"])
            if intensity_applied
            else 0.0,
            "residual_mean_absolute_error_u8": final_error,
        },
        "independent_error_reduction_u8": {
            "temporal_lag": temporal_share,
            "spatial_translation": max(0.0, baseline_error - spatial_residual),
            "intensity_scale": max(
                0.0,
                baseline_error - float(intensity["residual_mean_absolute_error_u8"]),
            ),
        },
        "error_decomposition_u8": shares,
        "largest_error_component": largest,
        "verdict": verdict,
    }


def write_audit(
    audit: Mapping[str, Any],
    output_dir: Path,
    *,
    source_gif: Path | None = None,
    source_receipt: Path | None = None,
) -> tuple[Path, Path]:
    """Write the JSON receipt and two-panel diagnostic figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(audit)
    if source_gif is not None:
        payload["source_gif"] = _portable_path(source_gif)
        payload["source_gif_sha256"] = _sha256(source_gif)
    if source_receipt is not None:
        payload["source_receipt"] = _portable_path(source_receipt)
        payload["source_receipt_sha256"] = _sha256(source_receipt)

    json_path = output_dir / "verdict.json"
    figure_path = output_dir / "alignment-audit.png"
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_figure(payload, figure_path)
    return json_path, figure_path


def run_audit(
    gif_path: Path,
    receipt_path: Path,
    output_dir: Path,
    *,
    frame_spacing_s: float = DEFAULT_FRAME_SPACING_S,
    min_overlap: int | None = None,
) -> dict[str, Any]:
    """Audit one committed paired GIF and write its evidence artifacts."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    pixel_error = receipt.get("pixel_error", {})
    reported_error = (
        float(pixel_error["decoded_frame_mae_u8"])
        if isinstance(pixel_error, Mapping)
        and pixel_error.get("decoded_frame_mae_u8") is not None
        else None
    )
    real, decoded = split_gif_panels(gif_path)
    audit = audit_alignment(
        real,
        decoded,
        frame_spacing_s=frame_spacing_s,
        min_overlap=min_overlap,
        reported_error_u8=reported_error,
    )
    audit["source_shot"] = receipt.get("shot")
    audit["source_checkpoint_step"] = receipt.get("checkpoint_step")
    write_audit(
        audit,
        output_dir,
        source_gif=gif_path,
        source_receipt=receipt_path,
    )
    return audit


def _validate_stacks(
    decoded: NDArray[Any], real: NDArray[Any], *, equal_count: bool = False
) -> tuple[NDArray[Any], NDArray[Any]]:
    decoded_array = np.asarray(decoded)
    real_array = np.asarray(real)
    if decoded_array.ndim not in (3, 4) or real_array.ndim != decoded_array.ndim:
        raise ValueError(
            "frame stacks must have shape (frame, height, width[, channel])"
        )
    if decoded_array.shape[1:] != real_array.shape[1:]:
        raise ValueError("decoded and real frames must have the same image shape")
    if not decoded_array.shape[0] or not real_array.shape[0]:
        raise ValueError("frame stacks must be non-empty")
    if equal_count and decoded_array.shape[0] != real_array.shape[0]:
        raise ValueError("decoded and real frame counts must match")
    if not np.isfinite(decoded_array).all() or not np.isfinite(real_array).all():
        raise ValueError("frame stacks must be finite")
    return decoded_array, real_array


def _portable_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _lag_indices(
    shape: tuple[int, int], lag: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    start = max(0, -lag)
    stop = min(shape[0], shape[1] - lag)
    decoded_indices = np.arange(start, stop, dtype=np.int64)
    return decoded_indices, decoded_indices + lag


def _curve_point(
    curve: Sequence[Mapping[str, float | int | bool]], lag: int
) -> Mapping[str, float | int | bool]:
    return next(point for point in curve if int(point["lag_frames"]) == lag)


def _pairs_at_lag(
    decoded: NDArray[Any], real: NDArray[Any], lag: int
) -> tuple[NDArray[Any], NDArray[Any]]:
    decoded_indices, real_indices = _lag_indices((decoded.shape[0], real.shape[0]), lag)
    return decoded[decoded_indices], real[real_indices]


def _paired_mae(decoded: NDArray[Any], real: NDArray[Any]) -> float:
    return float(np.mean(np.abs(decoded.astype(np.float64) - real.astype(np.float64))))


def _spatial_summary(displacement: FloatArray) -> dict[str, float | int]:
    finite = displacement[np.isfinite(displacement).all(axis=1)]
    if not finite.size:
        raise ValueError("no frame pair has two defined brightness centroids")
    magnitudes = np.hypot(finite[:, 0], finite[:, 1])
    return {
        "finite_frame_count": int(finite.shape[0]),
        "mean_dx_px": float(np.mean(finite[:, 0])),
        "mean_dy_px": float(np.mean(finite[:, 1])),
        "median_dx_px": float(np.median(finite[:, 0])),
        "median_dy_px": float(np.median(finite[:, 1])),
        "mean_displacement_magnitude_px": float(np.mean(magnitudes)),
        "median_displacement_magnitude_px": float(np.median(magnitudes)),
    }


def _write_figure(audit: Mapping[str, Any], output: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    temporal = audit["temporal"]
    spatial = audit["spatial_translation"]
    curve = temporal["per_lag"]
    eligible = [point for point in curve if point["eligible_for_selection"]]
    displacements = np.asarray(
        spatial["centroid_displacement_px_per_frame"], dtype=np.float64
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].plot(
        [point["lag_frames"] for point in eligible],
        [point["mean_absolute_error_u8"] for point in eligible],
        color="#32688e",
        marker="o",
        markersize=3,
    )
    axes[0].axvline(
        temporal["best_lag_frames"], color="#b43c29", linestyle="--", linewidth=1.2
    )
    axes[0].set(
        xlabel="Lag (frames; real index minus decoded index)",
        ylabel="Mean absolute error (uint8 levels)",
        title="Temporal alignment",
    )
    axes[0].grid(alpha=0.2)

    finite = displacements[np.isfinite(displacements).all(axis=1)]
    axes[1].scatter(
        finite[:, 0], finite[:, 1], s=22, alpha=0.65, color="#32688e", label="frame"
    )
    axes[1].scatter(
        [spatial["mean_dx_px"]],
        [spatial["mean_dy_px"]],
        marker="x",
        s=90,
        linewidth=2,
        color="#df8f2d",
        label="mean",
    )
    axes[1].scatter(
        [spatial["median_dx_px"]],
        [spatial["median_dy_px"]],
        marker="+",
        s=110,
        linewidth=2,
        color="#b43c29",
        label="median",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].axvline(0.0, color="black", linewidth=0.7)
    axes[1].set(
        xlabel="Decoded minus real centroid dx (px)",
        ylabel="Decoded minus real centroid dy (px)",
        title="Brightness-centroid displacement",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle(str(audit["verdict"]), fontsize=10)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _default_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    figure_root = root / "docs/figures/physics-carried-playable-plasma"
    stem = "shot-22086-real-decoded-step57019"
    return (
        figure_root / f"{stem}.gif",
        figure_root / f"{stem}.receipt.json",
        figure_root / "alignment-audit",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the committed held-out-carrier alignment audit."""
    default_gif, default_receipt, default_output = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", type=Path, default=default_gif)
    parser.add_argument("--receipt", type=Path, default=default_receipt)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--frame-spacing-s", type=float, default=DEFAULT_FRAME_SPACING_S
    )
    parser.add_argument("--min-overlap", type=int)
    args = parser.parse_args(argv)
    audit = run_audit(
        args.gif,
        args.receipt,
        args.output_dir,
        frame_spacing_s=args.frame_spacing_s,
        min_overlap=args.min_overlap,
    )
    print(json.dumps({"verdict": audit["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
