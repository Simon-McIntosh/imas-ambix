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


# ---------------------------------------------------------------------------
# Per-frame display normalisation (the clearer-evidence convention)
# ---------------------------------------------------------------------------


def test_display_limits_robust_percentile():
    """vmin/vmax are the 1st/99th percentile of the GT frame, not 0/255."""
    rng = np.random.default_rng(5)
    # a dim frame (max well below 255) with a couple of bright outliers
    frame = rng.integers(10, 40, size=rd.ORIGINAL_HW).astype(np.float64)
    frame[0, 0] = 255.0  # a single hot pixel must NOT set vmax
    vmin, vmax = rd.display_limits(frame)
    assert 0 <= vmin < vmax
    assert vmax < 60.0, "the lone hot pixel must be clipped by the 99th percentile"


def test_display_limits_degenerate_frame():
    """A flat frame falls back to a non-zero span (no divide-by-zero)."""
    vmin, vmax = rd.display_limits(np.full(rd.ORIGINAL_HW, 7.0))
    assert vmax > vmin

    vmin, vmax = rd.display_limits(np.full(rd.ORIGINAL_HW, np.nan))
    assert vmax > vmin


def test_normalise_for_display_clamps_to_limits():
    """Below vmin → black, above vmax → white, shared limits applied as given."""
    from imas_ambix.camdyn import recon_movie as mv

    img = np.array([[0.0, 50.0, 100.0, 200.0]], dtype=np.float64)
    out = mv.normalise_for_display(img, 50.0, 100.0)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0  # below vmin clamps to black
    assert out[0, 1] == 0  # at vmin
    assert out[0, 2] == 255  # at vmax
    assert out[0, 3] == 255  # above vmax clamps to white (honest over-shoot)


def test_normalise_for_display_rgb_to_gray():
    from imas_ambix.camdyn import recon_movie as mv

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 128
    out = mv.normalise_for_display(rgb, 0.0, 255.0)
    assert out.shape == (4, 4)


# ---------------------------------------------------------------------------
# GIF frame assembly from arrays (PIL, no matplotlib / imagemagick)
# ---------------------------------------------------------------------------


def test_side_by_side_frame_layout_and_scale():
    from imas_ambix.camdyn import recon_movie as mv

    gt = np.random.default_rng(6).integers(0, 256, size=(8, 12), dtype=np.uint8)
    model = np.random.default_rng(7).integers(0, 256, size=(8, 12), dtype=np.uint8)
    scale, gap = 3, 4
    frame = mv.side_by_side_frame(
        gt,
        model,
        scale=scale,
        gt_label="ground truth",
        model_label="dynamics",
        counter="f3 t+5.0ms",
        gap=gap,
    )
    assert frame.ndim == 3 and frame.shape[2] == 3  # RGB
    assert frame.shape[0] == 8 * scale  # height = scaled frame height
    # width = two scaled panes + the gap
    assert frame.shape[1] == 12 * scale * 2 + gap


def test_write_gif_roundtrip(tmp_path):
    from PIL import Image

    from imas_ambix.camdyn import recon_movie as mv

    frames = [np.full((16, 24, 3), v, dtype=np.uint8) for v in (10, 80, 160, 240)]
    out = tmp_path / "anim.gif"
    mv.write_gif(frames, out, duration_ms=100)
    assert out.exists() and out.stat().st_size > 0
    with Image.open(str(out)) as im:
        assert getattr(im, "n_frames", 1) == len(frames)


def test_draw_clip_box_marks_only_clipped():
    from imas_ambix.camdyn import recon_movie as mv

    img = np.zeros(rd.ORIGINAL_HW, dtype=np.uint8)
    box = rd.clip_box("clipped")
    mv._draw_clip_box(img, box, value=255)
    assert (img == 255).any(), "clip box must draw an outline"

    img2 = np.zeros(rd.ORIGINAL_HW, dtype=np.uint8)
    mv._draw_clip_box(img2, rd.clip_box("signals_only"), value=255)
    assert not (img2 == 255).any(), "no box for signals-only"


# ---------------------------------------------------------------------------
# Forecast horizon decimation selection
# ---------------------------------------------------------------------------


def test_forecast_stride_reachable_and_unreachable():
    from imas_ambix.camdyn import recon_movie as mv

    # 1 ms/frame, 16 frames spans 15 ms natively → 200 ms needs decimation.
    stride, ok = mv.forecast_stride_for(1e-3, 16, 200.0)
    assert ok and stride > 1
    reach = (16 - 1) * stride * 1e-3 * 1000.0
    assert reach >= 200.0

    # even fully decimated a tiny window cannot reach an absurd horizon if the
    # cadence is unknown/zero
    stride0, ok0 = mv.forecast_stride_for(0.0, 16, 200.0)
    assert stride0 == 1 and not ok0


def test_decimated_indices_span_horizon():
    from imas_ambix.camdyn import recon_movie as mv

    n_wide, n_target = 256, 16
    dt = 1e-3  # 1 ms/frame
    idx = mv.decimated_indices(n_wide, n_target, dt, 100.0)
    assert idx.ndim == 1
    assert idx.max() < n_wide
    assert idx.size >= 2
    assert np.all(np.diff(idx) > 0)  # strictly increasing
    # the kept frames span at least the requested horizon (within the window)
    span_ms = (idx[-1] - idx[0]) * dt * 1000.0
    assert span_ms >= 100.0 - 1e-6 or idx[-1] == max(i for i in range(n_wide))


def test_decimated_indices_short_window_passthrough():
    from imas_ambix.camdyn import recon_movie as mv

    idx = mv.decimated_indices(10, 16, 1e-3, 100.0)
    assert np.array_equal(idx, np.arange(10))


# ---------------------------------------------------------------------------
# Ramp-up window finder
# ---------------------------------------------------------------------------


def test_rampup_score_prefers_rising_current():
    from imas_ambix.camdyn import recon_movie as mv

    n = 16
    rising = np.linspace(50e3, 400e3, n)  # ramp-up: current climbing
    flat = np.full(n, 600e3)  # flat-top: already at peak
    bright_rise = np.linspace(20, 120, n)
    bright_flat = np.full(n, 200.0)

    s_rampup = mv.rampup_score(rising, bright_rise)
    s_flattop = mv.rampup_score(flat, bright_flat)
    assert s_rampup > s_flattop
    assert s_rampup > 0.0
    assert s_flattop == 0.0  # gated to zero (already near peak / no rise)


def test_rampup_score_degenerate():
    from imas_ambix.camdyn import recon_movie as mv

    assert mv.rampup_score(np.array([1.0]), np.array([1.0])) == 0.0
    assert mv.rampup_score(np.zeros(16), np.zeros(16)) == 0.0


# ---------------------------------------------------------------------------
# 3-panel strip (GT | static comparator | dynamics)
# ---------------------------------------------------------------------------


def test_panel_strip_three_panes_layout():
    from imas_ambix.camdyn import recon_movie as mv

    panes = [
        np.random.default_rng(s).integers(0, 256, size=(8, 12), dtype=np.uint8)
        for s in (11, 12, 13)
    ]
    scale, gap = 3, 4
    frame = mv.panel_strip(
        panes,
        ["ground truth", "baseline", "dynamics"],
        scale=scale,
        counter="f5 t+3.0ms",
        gap=gap,
    )
    assert frame.ndim == 3 and frame.shape[2] == 3  # RGB
    assert frame.shape[0] == 8 * scale
    # 3 panes + 2 separators
    assert frame.shape[1] == 3 * (12 * scale) + 2 * gap


# ---------------------------------------------------------------------------
# Forecast persistence comparator (freeze the last observed frame)
# ---------------------------------------------------------------------------


def test_persistence_tokens_freezes_last_observed():
    from imas_ambix.camdyn import recon_movie as mv

    n_frames, frontier = 16, 8
    rng = np.random.default_rng(14)
    tok = rng.integers(
        0, 1 << 18, size=(n_frames, rd.GRID_H, rd.GRID_W), dtype=np.int64
    )
    per = mv.persistence_tokens(tok, frontier)
    # observed half is unchanged
    assert np.array_equal(per[:frontier], tok[:frontier])
    # forecast half is the frozen frame (frontier-1)
    for f in range(frontier, n_frames):
        assert np.array_equal(per[f], tok[frontier - 1])


# ---------------------------------------------------------------------------
# ELM transient-spike scorer (Dα — window selection only, not a model input)
# ---------------------------------------------------------------------------


def test_elm_spike_score_finds_transient_burst():
    from imas_ambix.camdyn import recon_movie as mv

    n = 16
    base = np.full(n, 10.0)
    elmy = base.copy()
    elmy[9] = 120.0  # a sharp transient spike at frame 9 (rises in, falls out)
    score, peak = mv.elm_spike_score(elmy)
    assert peak == 9
    assert score > 0.0

    flat_score, _ = mv.elm_spike_score(base)
    assert flat_score == 0.0
    assert score > flat_score


def test_elm_spike_score_discounts_monotone_ramp():
    from imas_ambix.camdyn import recon_movie as mv

    n = 16
    ramp = np.linspace(10.0, 200.0, n)  # monotone rise — a ramp, not a burst
    burst = np.full(n, 10.0)
    burst[8] = 200.0
    s_ramp, _ = mv.elm_spike_score(ramp)
    s_burst, _ = mv.elm_spike_score(burst)
    assert s_burst > s_ramp  # the transient burst scores higher than the ramp


def test_elm_spike_score_degenerate():
    from imas_ambix.camdyn import recon_movie as mv

    assert mv.elm_spike_score(np.array([1.0]))[0] == 0.0
    s, _ = mv.elm_spike_score(np.array([np.nan, 1.0, 2.0]))
    assert s == 0.0


def test_camera_brightness_trace_edge_weighted():
    """The brightness proxy mixes whole-frame and bottom (divertor) rows."""
    from imas_ambix.camdyn import recon_movie as mv

    f, h, w = 8, 112, 156
    frames = np.zeros((f, h, w), dtype=np.float64)
    frames[3, -10:, :] = 200.0  # an edge burst only in the bottom rows, frame 3
    trace = mv.camera_brightness_trace(frames, edge_rows=30)
    assert trace.shape == (f,)
    assert int(np.argmax(trace)) == 3  # the edge burst frame is the brightest


def test_camera_elm_score_detects_transient_burst():
    """A sub-ms transient camera brightening scores as an ELM; flat does not."""
    from imas_ambix.camdyn import recon_movie as mv

    f, h, w = 16, 112, 156
    base = np.full((f, h, w), 20.0, dtype=np.float64)
    elmy = base.copy()
    elmy[9, -20:, :] += 180.0  # transient divertor burst at frame 9
    score, peak = mv.camera_elm_score(elmy)
    assert score > 0.0
    assert abs(peak - 9) <= 1  # the burst frame (high-pass may shift by 1)

    flat_score, _ = mv.camera_elm_score(base)
    assert score > flat_score


def test_three_row_panel_persistence_middle(tmp_path):
    """The panel can show persistence as the middle row (forecast mode)."""
    from imas_ambix.camdyn import recon_movie as mv

    n_frames = 16
    dt = 1.0 / 600.0
    meta_entry = {
        "shot_id": 24065,
        "frame_time": (0.30 + dt * np.arange(n_frames)).tolist(),
    }
    images = np.random.default_rng(15).integers(
        0, 256, size=(2, n_frames, 256, 256, 3), dtype=np.uint8
    )
    slot = {
        (0, "frontier", "persistence"): 0,
        (0, "frontier", "dynamics"): 1,
    }
    out = tmp_path / "panel.png"
    mv.assemble_three_row_panel(
        "frontier",
        meta_entry,
        images,
        slot,
        0,
        None,
        out_path=out,
        middle_role="persistence",
        middle_name="persistence",
        highlight_frame=10,
    )
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Three-row panel assembly (GT / baseline / dynamics) from a synthetic bundle
# ---------------------------------------------------------------------------


def test_assemble_three_row_panel_writes_png(tmp_path):
    from imas_ambix.camdyn import recon_movie as mv

    n_frames = 16
    dt = 1.0 / 600.0
    meta_entry = {
        "shot_id": 24065,
        "frame_time": (0.30 + dt * np.arange(n_frames)).tolist(),
    }
    # decoded bundle: window 0 carries baseline + dynamics for "clipped"
    images = np.random.default_rng(8).integers(
        0, 256, size=(2, n_frames, 256, 256, 3), dtype=np.uint8
    )
    slot = {(0, "clipped", "baseline"): 0, (0, "clipped", "dynamics"): 1}
    raw = np.random.default_rng(9).integers(
        0, 256, size=(n_frames, *rd.ORIGINAL_HW), dtype=np.uint8
    )
    out = tmp_path / "fig-cdw-recon-window.png"
    mv.assemble_three_row_panel(
        "clipped",
        meta_entry,
        images,
        slot,
        0,
        raw,
        out_path=out,
        title_extra="flat-top",
    )
    assert out.exists() and out.stat().st_size > 0


def test_bit_map_tokens_matches_rule():
    """The shared bit-head MAP decoder matches id = Σ_b (logit_b>0)<<b."""
    rng = np.random.default_rng(10)
    bits = 18
    logits = rng.standard_normal((3, rd.GRID_H, rd.GRID_W, bits))
    ids = rd._bit_map_tokens(logits)
    shifts = np.arange(bits, dtype=np.int64)
    expect = ((logits > 0).astype(np.int64) << shifts).sum(axis=-1)
    assert np.array_equal(ids, expect)
    assert ids.shape == (3, rd.GRID_H, rd.GRID_W)
