from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.worldmodel.rollout_degradation_figure import (
    build_figure,
    load_rollout_degradation,
)


def _write_receipts(root: Path, *, boundary_time_s: float = 0.35) -> tuple[Path, Path]:
    full_path = root / "full-window.receipt.json"
    seeded_path = root / "seeded-window.receipt.json"
    full_path.write_text(
        json.dumps(
            {
                "per_frame_error": {
                    "decoded_frame_mae_u8_scored": [2, 4, 8, 16, 32, 64],
                    "persistence_frame_mae_u8_scored": [1, 2, 2, 4, 4, 8],
                    "scored_transition_target_times_s": [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                    ],
                },
                "phase_error": {"boundary_time_s": boundary_time_s},
                "seed_provenance": {"session_slice_indices": [10, 11, 12, 13]},
            }
        ),
        encoding="utf-8",
    )
    seeded_path.write_text(
        json.dumps(
            {
                "pixel_error": {
                    "decoded_to_persistence_ratio": 1.75,
                    "scored_frame_count": 5,
                }
            }
        ),
        encoding="utf-8",
    )
    return full_path, seeded_path


def _phase_patch_width(figure, phase: str) -> float:
    patch = next(item for item in figure.axes[0].patches if item.get_gid() == phase)
    return float(patch.get_width())


def test_figure_derives_curve_annotations_and_sources_from_receipts(
    tmp_path: Path,
) -> None:
    full_path, seeded_path = _write_receipts(tmp_path)

    data = load_rollout_degradation(full_path, seeded_path)
    figure = build_figure(data)
    all_text = "\n".join(
        text.get_text() for text in (*figure.axes[0].texts, *figure.texts)
    )

    assert data.transition_indices == (1, 2, 3, 4, 5, 6)
    assert data.decoded_error_u8 == (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    assert data.persistence_error_u8 == (1.0, 2.0, 2.0, 4.0, 4.0, 8.0)
    assert data.aggregate_ratio == pytest.approx(6.0)
    assert data.early_ratio == pytest.approx(2.8)
    assert data.late_ratio == pytest.approx(7.0)
    assert data.seeded_ratio == pytest.approx(1.75)
    assert data.seeded_transition_count == 5
    assert data.history_transition_count == 4
    assert "1.750× persistence over 5 transitions" in all_text
    assert "6.000× persistence over 6 transitions" in all_text
    assert "2.800× persistence · 3 transitions" in all_text
    assert "7.000× persistence · 3 transitions" in all_text
    assert str(full_path) in all_text
    assert str(seeded_path) in all_text


def test_phase_boundary_moves_the_shading(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    first_full, first_seeded = _write_receipts(first_root, boundary_time_s=0.35)
    second_root = tmp_path / "second"
    second_root.mkdir()
    second_full, second_seeded = _write_receipts(second_root, boundary_time_s=0.25)

    first = load_rollout_degradation(first_full, first_seeded)
    second = load_rollout_degradation(second_full, second_seeded)
    first_figure = build_figure(first)
    second_figure = build_figure(second)

    assert first.phase_boundary_transition == pytest.approx(3.5)
    assert second.phase_boundary_transition == pytest.approx(2.5)
    assert _phase_patch_width(first_figure, "early-phase") == pytest.approx(3.0)
    assert _phase_patch_width(second_figure, "early-phase") == pytest.approx(2.0)


def test_missing_full_window_error_block_raises(tmp_path: Path) -> None:
    full_path, seeded_path = _write_receipts(tmp_path)
    full_path.write_text(json.dumps({"phase_error": {"boundary_time_s": 0.35}}))

    with pytest.raises(ValueError, match="per_frame_error"):
        load_rollout_degradation(full_path, seeded_path)


def test_missing_seeded_error_block_raises(tmp_path: Path) -> None:
    full_path, seeded_path = _write_receipts(tmp_path)
    seeded_path.write_text(json.dumps({"frame_count": 6}))

    with pytest.raises(ValueError, match="pixel_error"):
        load_rollout_degradation(full_path, seeded_path)
