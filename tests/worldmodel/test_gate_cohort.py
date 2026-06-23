"""Unit tests for the screened eval-only gate cohort.

These cover the pure-numpy / filesystem parts — the train-disjoint candidate
enumeration, the training-shot extraction, and the binding LEAKAGE GUARD — so the
cohort builder is green BEFORE a GPU decode runs.  The per-shot GT-decode screen
needs the VQ stack and is exercised on a compute node.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.worldmodel.gate_cohort import (
    ScreenThresholds,
    assert_disjoint,
    enumerate_candidate_shots,
    load_cohort,
    training_shot_ids,
)

# ---------------------------------------------------------------------------
# training_shot_ids — read the manifest's shot set
# ---------------------------------------------------------------------------


def test_training_shot_ids_reads_dict_windows(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "windows": [
                    {"shot_id": 100, "camera_id": "rbb"},
                    {"shot_id": 100, "camera_id": "rco"},  # dup shot, one id
                    {"shot_id": 101, "camera_id": "rbb"},
                ],
                "held_out": [18502, 18503],
            }
        )
    )
    ids = training_shot_ids(manifest)
    assert ids == {100, 101, 18502, 18503}


def test_training_shot_ids_reads_list_schema_windows(tmp_path):
    # the list-row schema: shot_id is column 0.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": ["shot_id", "camera_id", "start_frame"],
                "windows": [[200, "rbb", 0], [201, "rbb", 5]],
            }
        )
    )
    assert training_shot_ids(manifest) == {200, 201}


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------


def test_assert_disjoint_passes_when_no_overlap():
    assert_disjoint([1, 2, 3], {4, 5, 6})  # no raise


def test_assert_disjoint_raises_on_leak():
    with pytest.raises(AssertionError) as exc:
        assert_disjoint([1, 2, 7], {7, 8, 9})
    assert "7" in str(exc.value)
    assert "leakage" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# candidate enumeration — train-disjoint + has the camera recording
# ---------------------------------------------------------------------------


def _mk_token_tree(root: Path, shots, *, camera="rbb", with_mag=None):
    """Fake the token store layout: frames/<shot>/<cam>.zarr + signals-magnetics."""
    with_mag = shots if with_mag is None else with_mag
    frames = root / "v1" / "frames"
    mag = root / "v1" / "signals-magnetics" / "magnetics"
    for s in shots:
        (frames / str(s) / f"{camera}.zarr").mkdir(parents=True)
    for s in with_mag:
        (mag / str(s) / "magnetics.zarr").mkdir(parents=True)


def test_enumerate_excludes_training_shots(tmp_path):
    _mk_token_tree(tmp_path, [10, 11, 12, 13])
    cand = enumerate_candidate_shots(
        token_root=tmp_path, camera="rbb", train_ids={11, 13}, cap=10
    )
    assert cand == [10, 12]  # train shots dropped, sorted ascending


def test_enumerate_requires_camera_and_magnetics(tmp_path):
    # shot 20 has rbb + mag; 21 has rbb but NO mag; 22 has neither (no rbb dir).
    _mk_token_tree(tmp_path, [20, 21], with_mag=[20])
    (tmp_path / "v1" / "frames" / "22").mkdir(parents=True)  # dir but no rbb.zarr
    cand = enumerate_candidate_shots(
        token_root=tmp_path, camera="rbb", train_ids=set(), cap=10
    )
    assert cand == [20]  # 21 dropped (no mag), 22 dropped (no rbb recording)
    # without the magnetics requirement, 21 is admitted too.
    cand2 = enumerate_candidate_shots(
        token_root=tmp_path,
        camera="rbb",
        train_ids=set(),
        cap=10,
        require_magnetics=False,
    )
    assert cand2 == [20, 21]


def test_enumerate_caps_candidate_count(tmp_path):
    _mk_token_tree(tmp_path, list(range(30, 50)))
    cand = enumerate_candidate_shots(
        token_root=tmp_path, camera="rbb", train_ids=set(), cap=5
    )
    assert len(cand) == 5
    assert cand == [30, 31, 32, 33, 34]  # deterministic ascending


# ---------------------------------------------------------------------------
# load_cohort
# ---------------------------------------------------------------------------


def test_load_cohort_reads_ids(tmp_path):
    p = tmp_path / "cohort.json"
    p.write_text(json.dumps({"cohort": [101, 202, 303], "summary": {}}))
    assert load_cohort(p) == [101, 202, 303]


def test_screen_thresholds_defaults_kill_dark_and_static():
    """The default thresholds reject the known degenerate gate shots by metric."""
    t = ScreenThresholds()
    # 18504 is dark (mean 2.7) and near-static (motion 2.3) -> both fail.
    assert t.min_brightness > 2.7
    assert t.min_transient_motion > 2.3
    # 18502 is bright (17) and moving (8.2) -> both pass.
    assert t.min_brightness <= 17.0
    assert t.min_transient_motion <= 8.2
