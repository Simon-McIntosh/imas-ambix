"""CPU-fast tests for the reconstruction-demo figure builder.

These exercise the pure-numpy / layout logic without torch, a GPU, or the
real corpus: scenario masks, the clip-box outline geometry, column-frame
selection, the predict↔decode token-bundle round-trip, and the matplotlib
figure assembly fed a synthetic decoded-image bundle.  The GPU predict
phase and the Open-MAGVIT2 decode phase are covered only by the live
SLURM run, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn import reconstruction_demo as rd


def test_scenario_masks():
    n_frames, frontier = 16, 8
    fr = rd.scenario_mask("frontier", n_frames, frontier)
    assert fr.shape == (n_frames, rd.GRID_H, rd.GRID_W)
    assert fr[:frontier].all() and not fr[frontier:].any()

    cl = rd.scenario_mask("clipped", n_frames, frontier)
    assert cl.shape == (n_frames, rd.GRID_H, rd.GRID_W)
    # fixed_section2 is a static sub-window: same mask every frame, partial.
    assert 0 < cl[0].mean() < 1.0
    assert np.array_equal(cl[0], cl[-1])

    so = rd.scenario_mask("signals_only", n_frames, frontier)
    assert not so.any()

    with pytest.raises(ValueError):
        rd.scenario_mask("bogus", n_frames, frontier)


def test_clip_box_only_for_clipped():
    assert rd.clip_box("frontier") is None
    assert rd.clip_box("signals_only") is None
    box = rd.clip_box("clipped")
    assert box is not None
    r0, r1, c0, c1 = box
    assert 0 <= r0 < r1 <= rd.GRID_H
    assert 0 <= c0 < c1 <= rd.GRID_W
    # the box must cover exactly the visible cells of fixed_section2 frame 0
    m0 = rd.scenario_mask("clipped", 1, 0)[0]
    assert m0[r0:r1, c0:c1].all()
    assert m0.sum() == (r1 - r0) * (c1 - c0)


def test_column_frames_frontier_are_post_frontier():
    cols = rd._column_frames(16, 8, "frontier")
    assert cols, "frontier must show at least one post-frontier column"
    assert all(c > 8 for c in cols)
    assert cols == sorted(set(cols))  # strictly increasing, unique

    spread = rd._column_frames(16, 8, "clipped")
    assert spread[0] == 0 and spread[-1] == 15
    assert spread == sorted(set(spread))


def test_zoh_carry_forward_frontier():
    """ZOH under a frontier mask = persistence of the last visible frame."""
    from imas_ambix.camdyn.arm_compare import _carry_forward_pred

    n_frames, frontier = 6, 3
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1 << 18, size=(n_frames, rd.GRID_H, rd.GRID_W))
    vis = rd.scenario_mask("frontier", n_frames, frontier)
    zoh = _carry_forward_pred(tokens, vis)
    # post-frontier frames carry forward frame (frontier-1); causal so the
    # prediction at f uses only frames < f.
    for f in range(frontier + 1, n_frames):
        assert np.array_equal(zoh[f], tokens[frontier - 1])


def test_zoh_signals_only_never_observed():
    """With no visible cell, ZOH is all -1 (never observed → black)."""
    from imas_ambix.camdyn.arm_compare import _carry_forward_pred

    tokens = np.arange(4 * rd.GRID_H * rd.GRID_W).reshape(4, rd.GRID_H, rd.GRID_W)
    vis = rd.scenario_mask("signals_only", 4, 0)
    zoh = _carry_forward_pred(tokens, vis)
    assert (zoh < 0).all()


def _fake_bundle(n_frames=16, scenarios=("frontier", "clipped", "signals_only")):
    """A predict-phase bundle for one window with deterministic token grids."""
    rng = np.random.default_rng(1)
    dt = 1.0 / 600.0
    ft = 0.30 + dt * np.arange(n_frames)
    true_tokens = rng.integers(
        0, 1 << 18, size=(n_frames, rd.GRID_H, rd.GRID_W), dtype=np.int64
    )
    entry = {
        "shot_id": 24065,
        "start": 100,
        "frame_time": ft,
        "dt": np.full(n_frames, dt),
        "valid": np.ones(n_frames, bool),
        "motion_fraction": 0.42,
        "true_tokens": true_tokens,
        "scenarios": {},
    }
    for sc in scenarios:
        vis = rd.scenario_mask(sc, n_frames, 8)
        from imas_ambix.camdyn.arm_compare import _carry_forward_pred

        entry["scenarios"][sc] = {
            "visible": vis,
            "pred_tokens": rng.integers(
                0, 1 << 18, size=(n_frames, rd.GRID_H, rd.GRID_W), dtype=np.int64
            ),
            "zoh_tokens": _carry_forward_pred(true_tokens, vis),
        }
    return [entry]


def test_token_bundle_roundtrip(tmp_path: Path):
    bundle = _fake_bundle()
    path = tmp_path / "tokens.npz"
    rd.save_token_bundle(bundle, path)
    data = np.load(str(path), allow_pickle=True)
    grids = np.asarray(data["grids"])
    index = json.loads(str(data["index"]))
    meta = json.loads(str(data["meta"]))

    assert grids.ndim == 4 and grids.shape[2:] == (rd.GRID_H, rd.GRID_W)
    assert len(meta) == 1
    # roles present: one true-window grid + (visible,pred,zoh) per scenario
    roles = {(e["scenario"], e["role"]) for e in index}
    assert ("_window", "true") in roles
    for sc in ("frontier", "clipped", "signals_only"):
        assert (sc, "pred") in roles
        assert (sc, "zoh") in roles
    # every index slot must point at a real grid
    for e in index:
        assert 0 <= e["slot"] < grids.shape[0]


def _fake_image_bundle(token_bundle_path: Path, image_bundle_path: Path):
    """Stand in for the magvit2 decode phase: 256² grayscale-as-RGB images."""
    data = np.load(str(token_bundle_path), allow_pickle=True)
    grids = np.asarray(data["grids"])
    n, f = grids.shape[0], grids.shape[1]
    rng = np.random.default_rng(2)
    images = rng.integers(0, 256, size=(n, f, 256, 256, 3), dtype=np.uint8)
    np.savez_compressed(
        image_bundle_path,
        images=images,
        index=data["index"],
        meta=data["meta"],
    )


def test_assemble_figure_writes_png(tmp_path: Path):
    bundle = _fake_bundle()
    tok = tmp_path / "tokens.npz"
    img = tmp_path / "images.npz"
    rd.save_token_bundle(bundle, tok)
    _fake_image_bundle(tok, img)

    data = np.load(str(img), allow_pickle=True)
    images = np.asarray(data["images"], dtype=np.uint8)
    index = json.loads(str(data["index"]))
    meta = json.loads(str(data["meta"]))

    raw_by_window = {
        0: np.random.default_rng(3).integers(
            0, 256, size=(16, *rd.ORIGINAL_HW), dtype=np.uint8
        )
    }

    for scenario, fname in rd.SCENARIO_FILE.items():
        out = tmp_path / fname
        rd.assemble_figure(
            scenario,
            meta,
            images,
            index,
            raw_by_window,
            frontier=8,
            out_path=out,
            show_decoded_gt=True,
        )
        assert out.exists() and out.stat().st_size > 0


def test_to_aspect_resizes_to_native():
    sq = np.random.default_rng(4).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    out = rd._to_aspect(sq)
    assert out.shape == rd.ORIGINAL_HW
