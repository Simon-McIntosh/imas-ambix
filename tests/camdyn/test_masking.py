"""Clip-mask sampler: all 5 modes + frozen named-geometry determinism."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.masking import (
    NAMED_GEOMETRIES,
    ClipMaskConfig,
    MaskMode,
    named_geometry_mask,
    sample_clip_mask,
)

N_FRAMES = 16
GRID = (16, 16)


def test_random_mode_shape_and_partial_visibility():
    cfg = ClipMaskConfig()
    rng = np.random.default_rng(0)
    mask, meta = sample_clip_mask(N_FRAMES, cfg, rng, mode=MaskMode.RANDOM)
    assert mask.shape == (N_FRAMES, *GRID)
    assert mask.dtype == bool
    # fixed clip → identical across frames
    assert np.all(mask[0] == mask[-1])
    # partially visible, not all/none
    frac = mask.mean()
    assert 0.0 < frac < 1.0
    assert meta["mode"] == "random"


def test_panning_mode_moves_across_frames():
    cfg = ClipMaskConfig(pan_max_speed=1.5)
    rng = np.random.default_rng(3)
    # draw until we get a pan with nonzero velocity (robust to a rare zero draw)
    for s in range(20):
        rng = np.random.default_rng(s)
        mask, meta = sample_clip_mask(N_FRAMES, cfg, rng, mode=MaskMode.PANNING)
        if not np.array_equal(mask[0], mask[-1]):
            break
    assert mask.shape == (N_FRAMES, *GRID)
    assert not np.array_equal(mask[0], mask[-1]), "panning clip should move"
    assert meta["mode"] == "panning"


def test_frontier_mode_visible_prefix_only():
    cfg = ClipMaskConfig()
    rng = np.random.default_rng(0)
    mask, meta = sample_clip_mask(N_FRAMES, cfg, rng, mode=MaskMode.FRONTIER)
    t = meta["frontier_t"]
    assert mask[:t].all()
    assert not mask[t:].any()
    assert 1 <= t < N_FRAMES


def test_full_mask_hides_everything():
    cfg = ClipMaskConfig()
    rng = np.random.default_rng(0)
    mask, meta = sample_clip_mask(N_FRAMES, cfg, rng, mode=MaskMode.FULL)
    assert not mask.any()
    assert meta["mode"] == "full"


def test_mixture_mode_draws_only_the_four_dynamic_modes():
    cfg = ClipMaskConfig()
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        _, meta = sample_clip_mask(N_FRAMES, cfg, rng, mode=None)
        seen.add(meta["mode"])
    # NAMED is the eval suite — never drawn in the mixture
    assert "named" not in seen
    assert seen <= {"random", "panning", "frontier", "full"}


def test_sampler_reproducible_with_same_seed():
    cfg = ClipMaskConfig()
    m1, _ = sample_clip_mask(N_FRAMES, cfg, np.random.default_rng(7), mode=None)
    m2, _ = sample_clip_mask(N_FRAMES, cfg, np.random.default_rng(7), mode=None)
    assert np.array_equal(m1, m2)


def test_curriculum_area_range_anneals():
    cfg = ClipMaskConfig(area_min=0.05, area_max=0.50)
    lo0, hi0 = cfg.curriculum_area_range(0.0)
    lo1, hi1 = cfg.curriculum_area_range(1.0)
    # early training: large visibility; late: full configured range
    assert hi0 == pytest.approx(1.0)
    assert lo0 >= 0.5
    assert (lo1, hi1) == pytest.approx((0.05, 0.50))
    # monotone shrink of the lower bound as progress increases
    los = [cfg.curriculum_area_range(p)[0] for p in (0.0, 0.5, 1.0)]
    assert los[0] >= los[1] >= los[2]


# --- frozen named-geometry suite ---


def test_named_suite_has_all_required_geometries():
    names = set(NAMED_GEOMETRIES)
    assert names == {
        "fixed_section2",
        "divertor_only",
        "centre_column_strip",
        "standard_pan",
        "frontier_half",
    }


@pytest.mark.parametrize("name", list(NAMED_GEOMETRIES))
def test_named_geometry_deterministic(name):
    m1 = named_geometry_mask(name, N_FRAMES, GRID)
    m2 = named_geometry_mask(name, N_FRAMES, GRID)
    assert m1.shape == (N_FRAMES, *GRID)
    assert m1.dtype == bool
    assert np.array_equal(m1, m2)


def test_named_window_partial_and_static():
    m = named_geometry_mask("fixed_section2", N_FRAMES, GRID)
    assert 0.0 < m.mean() < 1.0
    assert np.all(m[0] == m[-1])  # fixed window is static in time


def test_named_strip_is_centre_columns():
    m = named_geometry_mask("centre_column_strip", N_FRAMES, GRID)
    # only columns 6..9 visible, all rows
    assert m[0, :, 6:10].all()
    assert not m[0, :, :6].any()
    assert not m[0, :, 10:].any()


def test_named_pan_moves():
    m = named_geometry_mask("standard_pan", N_FRAMES, GRID)
    assert not np.array_equal(m[0], m[-1])


def test_named_frontier_half_splits_at_midpoint():
    m = named_geometry_mask("frontier_half", N_FRAMES, GRID)
    half = N_FRAMES // 2
    assert m[:half].all()
    assert not m[half:].any()


def test_unknown_named_geometry_raises():
    with pytest.raises(KeyError):
        named_geometry_mask("does_not_exist", N_FRAMES, GRID)
