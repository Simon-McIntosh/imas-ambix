"""Curation pipeline: coil-ramp excitation selector, plasma-presence filter,
long-horizon window recommendation, exposure-balancing transform.

These cover the NEW additions for the dynamic-excitation corpus, all
backward-compatible (the existing ``find_transient_window`` /
``assemble_controllable_window`` API is untouched — see
``test_controllable_model.py`` for the unchanged-API coverage).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# actuator_plan: coil-current channel indices + ramp-rate excitation selector
# ---------------------------------------------------------------------------


def test_coil_indices_exclude_ip_and_are_amc():
    import imas_ambix.worldmodel.actuator_plan as ap

    coil = set(ap.coil_current_channel_indices())
    ip = ap.plasma_current_channel_index()
    assert ip is not None
    # Ip is NOT a coil-drive column.
    assert ip not in coil
    # every coil column is an amc source and not the plasma current.
    for i in coil:
        c = ap.ACTUATOR_CHANNELS[i]
        assert c.source == "amc"
        assert c.key != "plasma_current"
    # coil columns are disjoint from gas + NBI.
    assert coil.isdisjoint(set(ap.gas_puff_channel_indices()))
    assert coil.isdisjoint(set(ap.nbi_channel_indices()))
    assert len(coil) >= 8  # the PF coils + solenoid + TF


def _patch_excitation(monkeypatch, ftime, values, missing):
    """Wire find_excitation_window's two reads to synthetic in-memory data."""
    import imas_ambix.camdyn.dataset as cd
    import imas_ambix.worldmodel.actuator_plan as ap
    import imas_ambix.worldmodel.spacetime_dataset as sd
    from imas_ambix.camdyn.conditioning import ConditioningSample

    monkeypatch.setattr(sd, "_frame_times", lambda *a, **k: ftime)
    monkeypatch.setattr(cd, "level1_shot_path", lambda *a, **k: "x")
    monkeypatch.setattr(
        ap,
        "load_conditioning",
        lambda *a, **k: ConditioningSample(
            shot_id=1,
            frame_time=ftime,
            channel_keys=[c.key for c in ap.ACTUATOR_CHANNELS],
            units=[c.unit for c in ap.ACTUATOR_CHANNELS],
            values=values,
            missing=missing,
        ),
    )


def _synthetic_actuator_window(
    n=120, *, ip_a, coil_ramp_at, coil_key="p4u_coil_current"
):
    """Build (ftime, values, missing) with Ip(t) and one coil ramping.

    ``ip_a`` is the per-frame plasma current in AMPERES — the physical value the
    selector reads directly (the loader normally scales the stored kA to A; the
    synthetic sample injects post-scale A).  ``coil_ramp_at`` is a (start, stop)
    frame slice where the named coil ramps linearly; elsewhere the coil is flat.
    """
    import imas_ambix.worldmodel.actuator_plan as ap

    ftime = np.linspace(0.0, (n - 1) / 600.0, n)  # 600 Hz
    C = ap.N_ACTUATOR_CHANNELS
    values = np.zeros((n, C), dtype=np.float64)
    missing = np.ones((n, C), dtype=np.float64)  # default: all missing

    ip_col = ap.plasma_current_channel_index()
    values[:, ip_col] = np.asarray(ip_a, dtype=np.float64)
    missing[:, ip_col] = 0.0

    ck = ap.ACTUATOR_CHANNEL_KEYS.index(coil_key)
    coil = np.full(n, 1.0e5)  # flat 100 kA baseline (level, not rate)
    s0, s1 = coil_ramp_at
    coil[s0:s1] = np.linspace(1.0e5, 2.0e5, s1 - s0)  # +100 kA ramp
    coil[s1:] = 2.0e5
    values[:, ck] = coil
    missing[:, ck] = 0.0
    return ftime, values, missing


def test_excitation_window_picks_the_coil_ramp(monkeypatch):
    import imas_ambix.worldmodel.actuator_plan as ap

    n, span = 120, 24
    # plasma present whole window (700 kA = 7e5 A), coil ramps over frames 60..84
    ip = np.full(n, 7.0e5)
    ftime, values, missing = _synthetic_actuator_window(
        n, ip_a=ip, coil_ramp_at=(60, 84)
    )
    _patch_excitation(monkeypatch, ftime, values, missing)
    res = ap.find_excitation_window(
        1, span, ip_present_threshold=2.0e4, min_ramp_rate=1.0
    )
    assert res.start_frame is not None
    # the window must overlap the ramp (frames 60..84).
    assert 48 <= res.start_frame <= 72
    assert res.score > 0.0
    assert res.max_abs_ip == pytest.approx(7.0e5)
    assert res.reason == ""


def test_excitation_window_rejects_vacuum(monkeypatch):
    import imas_ambix.worldmodel.actuator_plan as ap

    n, span = 120, 24
    # no plasma anywhere (1 kA = 1e3 A, below the 20 kA threshold) but a big ramp
    ip = np.full(n, 1.0e3)
    ftime, values, missing = _synthetic_actuator_window(
        n, ip_a=ip, coil_ramp_at=(40, 80)
    )
    _patch_excitation(monkeypatch, ftime, values, missing)
    res = ap.find_excitation_window(1, span, ip_present_threshold=2.0e4)
    assert res.start_frame is None
    assert res.reason == "no_plasma"


def test_excitation_window_rejects_flat_drive(monkeypatch):
    import imas_ambix.worldmodel.actuator_plan as ap

    n, span = 120, 24
    # plasma present, but the coil never ramps (flat) -> no excitation
    ip = np.full(n, 7.0e5)
    ftime, values, missing = _synthetic_actuator_window(n, ip_a=ip, coil_ramp_at=(0, 0))
    _patch_excitation(monkeypatch, ftime, values, missing)
    res = ap.find_excitation_window(
        1, span, ip_present_threshold=2.0e4, min_ramp_rate=1.0e3
    )
    assert res.start_frame is None
    assert res.reason == "flat_drive"


def test_excitation_window_allows_breakdown_start(monkeypatch):
    import imas_ambix.worldmodel.actuator_plan as ap

    n, span = 120, 24
    # Ip starts at ~0 (breakdown) and rises through the threshold — a window that
    # STARTS sub-threshold but becomes present must still be selectable.
    ip = np.concatenate([np.zeros(40), np.linspace(0, 7.0e5, 80)])
    ftime, values, missing = _synthetic_actuator_window(
        n, ip_a=ip, coil_ramp_at=(30, 70)
    )
    _patch_excitation(monkeypatch, ftime, values, missing)
    res = ap.find_excitation_window(
        1, span, ip_present_threshold=2.0e4, min_present_fraction=0.5, min_ramp_rate=1.0
    )
    assert res.start_frame is not None
    assert res.max_abs_ip >= 2.0e4


def test_find_transient_window_signature_unchanged():
    """The peer's gate calls find_transient_window — keep it importable + same API."""
    import inspect

    import imas_ambix.worldmodel.actuator_plan as ap

    sig = inspect.signature(ap.find_transient_window)
    params = list(sig.parameters)
    assert params[:2] == ["shot_id", "span"]
    # the kwargs the gate relies on are still present
    for kw in ("camera", "token_root", "channels", "level1_path", "min_variation"):
        assert kw in sig.parameters


# ---------------------------------------------------------------------------
# plasma_presence
# ---------------------------------------------------------------------------


def test_presence_accepts_present_window():
    from imas_ambix.worldmodel.plasma_presence import evaluate_presence

    absip = np.full(50, 5.0e5)  # 500 kA everywhere
    r = evaluate_presence(absip, threshold_a=2.0e4, min_present_fraction=0.5)
    assert r.present
    assert r.present_fraction == pytest.approx(1.0)
    assert r.max_abs_ip == pytest.approx(5.0e5)


def test_presence_rejects_vacuum():
    from imas_ambix.worldmodel.plasma_presence import evaluate_presence

    absip = np.full(50, 1.0e3)  # 1 kA — below 20 kA
    r = evaluate_presence(absip, threshold_a=2.0e4)
    assert not r.present
    assert r.max_abs_ip == pytest.approx(1.0e3)


def test_presence_breakdown_start_passes_with_fraction_gate():
    from imas_ambix.worldmodel.plasma_presence import evaluate_presence

    # first 40% sub-threshold (breakdown), then present — fraction 0.6 >= 0.5
    absip = np.concatenate([np.zeros(20), np.full(30, 5.0e5)])
    r = evaluate_presence(absip, threshold_a=2.0e4, min_present_fraction=0.5)
    assert r.present
    assert r.present_fraction == pytest.approx(0.6)


def test_presence_brief_flicker_rejected():
    from imas_ambix.worldmodel.plasma_presence import evaluate_presence

    # only 3/50 frames present — max clears threshold but fraction fails
    absip = np.zeros(50)
    absip[10:13] = 5.0e5
    r = evaluate_presence(absip, threshold_a=2.0e4, min_present_fraction=0.5)
    assert not r.present
    assert r.max_abs_ip == pytest.approx(5.0e5)


def test_presence_nan_frames_not_present():
    from imas_ambix.worldmodel.plasma_presence import frame_presence_mask

    absip = np.array([np.nan, np.nan, 5.0e5, 5.0e5])
    mask = frame_presence_mask(absip, threshold_a=2.0e4)
    assert mask.tolist() == [False, False, True, True]


# ---------------------------------------------------------------------------
# window_horizon
# ---------------------------------------------------------------------------


def test_horizon_low_fps_uses_stride_one():
    from imas_ambix.worldmodel.window_horizon import recommend_window

    # at 600 Hz, 0.25 s ~ 151 native frames; capped at 48 -> stride > 1
    r = recommend_window(600.0, target_horizon_s=0.25, max_n_frames=48)
    assert r.frame_stride >= 2
    assert r.n_frames <= 48
    assert r.covered_horizon_s == pytest.approx(0.25, abs=0.02)


def test_horizon_constant_physical_span_across_fps():
    from imas_ambix.worldmodel.window_horizon import recommend_window

    # the covered horizon stays ~constant across a 250x fps range — the point of
    # the per-shot rule (a fixed frame count would NOT do this).
    for fps in [500, 1500, 3000, 10000, 77000, 143000]:
        r = recommend_window(fps, target_horizon_s=0.25, max_n_frames=48)
        assert r.covered_horizon_s == pytest.approx(0.25, abs=0.03)
        assert 2 <= r.n_frames <= 48
        assert r.frame_stride >= 1
        assert 1 <= r.context_frames < r.n_frames


def test_horizon_tiny_fps_short_recording():
    from imas_ambix.worldmodel.window_horizon import recommend_window

    # 50 Hz: 0.25 s is ~13 frames, fits under the cap -> stride 1.
    r = recommend_window(50.0, target_horizon_s=0.25, max_n_frames=48)
    assert r.frame_stride == 1
    assert r.n_frames == r.native_span_frames


def test_horizon_nonfinite_fps_falls_back_to_reference():
    from imas_ambix.worldmodel.window_horizon import REFERENCE_FPS, recommend_window

    r = recommend_window(float("nan"), target_horizon_s=0.25)
    assert r.fps == pytest.approx(REFERENCE_FPS)


def test_window_config_for_builds_valid_config():
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeWindowConfig
    from imas_ambix.worldmodel.window_horizon import window_config_for

    cfg = window_config_for(3000.0, target_horizon_s=0.25, n_plan=8)
    assert isinstance(cfg, SpacetimeWindowConfig)
    assert cfg.n_frames >= 2
    assert cfg.frame_stride >= 1
    assert 1 <= cfg.context_frames < cfg.n_frames
    assert cfg.n_plan == 8


# ---------------------------------------------------------------------------
# exposure_balance
# ---------------------------------------------------------------------------


def test_percentile_normalise_shape_and_range():
    from imas_ambix.worldmodel.exposure_balance import percentile_normalise

    rng = np.random.default_rng(0)
    frames = (rng.random((6, 32, 32)) * 600).astype(np.uint16)
    out = percentile_normalise(frames)
    assert out.shape == frames.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_percentile_normalise_robust_to_outlier():
    """A single saturated pixel must NOT compress the bulk dynamic range.

    Global min/max would map the bulk (0..100) into 0..~3 because of one 60000
    outlier; percentile clipping keeps the bulk spread across the full range.
    """
    from imas_ambix.worldmodel.exposure_balance import (
        balance_exposure,
        percentile_normalise,
    )

    frames = np.zeros((4, 16, 16), dtype=np.uint16)
    # bulk signal spans 0..100 ...
    frames[:] = (np.linspace(0, 100, 16 * 16).reshape(16, 16)).astype(np.uint16)
    # ... plus one saturated outlier pixel per frame
    frames[:, 0, 0] = 60000
    pct = percentile_normalise(frames)
    glob = balance_exposure(frames, strategy="global")
    # Compare what each does to the BULK signal (exclude the outlier pixel): the
    # global stretch crushes the 0..100 bulk into ~0..0 because the 60000 pixel
    # sets the range; the percentile clip keeps the bulk spread across [0,255].
    bulk = (slice(None), slice(1, None), slice(None))
    assert int(pct[bulk].max()) > 200
    assert int(glob[bulk].max()) < 5  # the outlier crushes the bulk under global
    # the outlier pixel itself still maps to the top of the range under global.
    assert int(glob[:, 0, 0].max()) == 255


def test_percentile_normalise_rgb_layout_preserved():
    from imas_ambix.worldmodel.exposure_balance import percentile_normalise

    gray = (np.linspace(0, 200, 4 * 8 * 8).reshape(4, 8, 8)).astype(np.uint16)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    out = percentile_normalise(rgb)
    assert out.shape == rgb.shape
    # channels stay identical (grey-replicated)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_percentile_normalise_degenerate_flat_frame():
    from imas_ambix.worldmodel.exposure_balance import percentile_normalise

    frames = np.full((3, 8, 8), 7, dtype=np.uint16)
    out = percentile_normalise(frames)
    assert out.shape == frames.shape
    assert int(out.max()) == 0  # flat -> all zeros, matching the v0 contract


def test_profile_exposure_recommends_clahe_for_low_contrast():
    from imas_ambix.worldmodel.exposure_balance import profile_exposure

    # 10-bit sensor (max 1023) but bulk signal sits low (p99 ~ 50) -> low contrast
    frames = (np.random.default_rng(1).random((5, 16, 16)) * 50).astype(np.uint16)
    frames[0, 0, 0] = 1023  # set the sensor max high
    prof = profile_exposure(frames, low_contrast_p99_frac=0.25)
    assert prof.sensor_max == pytest.approx(1023.0)
    assert prof.low_contrast
    assert prof.recommended == "clahe"


def test_profile_exposure_recommends_percentile_for_full_range():
    from imas_ambix.worldmodel.exposure_balance import profile_exposure

    # bulk signal spans most of the 8-bit range -> percentile default
    frames = (np.linspace(0, 255, 5 * 16 * 16).reshape(5, 16, 16)).astype(np.uint16)
    prof = profile_exposure(frames)
    assert not prof.low_contrast
    assert prof.recommended == "percentile"


def test_balance_exposure_unknown_strategy_raises():
    from imas_ambix.worldmodel.exposure_balance import balance_exposure

    with pytest.raises(ValueError, match="unknown exposure strategy"):
        balance_exposure(np.zeros((2, 4, 4), dtype=np.uint16), strategy="nope")


# ---------------------------------------------------------------------------
# excitation_corpus: the curation orchestrator (held-out exclusion + ordering)
# ---------------------------------------------------------------------------


def _curated(shot_id, score, **kw):
    from imas_ambix.worldmodel.excitation_corpus import CuratedWindow

    return CuratedWindow(
        shot_id=int(shot_id),
        start_frame=kw.get("start_frame", 0),
        fps=kw.get("fps", 600.0),
        n_frames=kw.get("n_frames", 39),
        frame_stride=kw.get("frame_stride", 4),
        excitation_score=float(score),
        max_abs_ip=kw.get("max_abs_ip", 7.0e5),
        present_fraction=kw.get("present_fraction", 0.9),
    )


def test_select_curated_windows_excludes_held_out(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec

    # every shot "selects" with a score equal to its id (deterministic).
    monkeypatch.setattr(
        ec,
        "select_curated_window_for_shot",
        lambda sid, **k: _curated(sid, score=float(sid)),
    )
    shots = [10, 18502, 11, 18504, 12]
    out = ec.select_curated_windows(shots, held_out=(18502, 18504))
    sids = {c.shot_id for c in out}
    # held-out shots are gone; the others present.
    assert 18502 not in sids and 18504 not in sids
    assert sids == {10, 11, 12}


def test_select_curated_windows_orders_by_excitation(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec

    scores = {10: 5.0, 11: 99.0, 12: 50.0}
    monkeypatch.setattr(
        ec,
        "select_curated_window_for_shot",
        lambda sid, **k: _curated(sid, score=scores[sid]),
    )
    out = ec.select_curated_windows([10, 11, 12], held_out=())
    # most-excited first (the dynamic weighting).
    assert [c.shot_id for c in out] == [11, 12, 10]
    # a limit keeps the most-excited shots.
    top = ec.select_curated_windows([10, 11, 12], held_out=(), limit=2)
    assert [c.shot_id for c in top] == [11, 12]


def test_select_curated_windows_drops_rejected_shots(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec

    # shot 11 rejects (None); the rest select.
    def _sel(sid, **k):
        return None if sid == 11 else _curated(sid, score=float(sid))

    monkeypatch.setattr(ec, "select_curated_window_for_shot", _sel)
    out = ec.select_curated_windows([10, 11, 12], held_out=())
    assert {c.shot_id for c in out} == {10, 12}


def test_select_curated_window_for_shot_uses_horizon_and_excitation(monkeypatch):
    """Integration: the per-shot selector wires fps -> window -> excitation."""
    import imas_ambix.worldmodel.actuator_plan as ap
    import imas_ambix.worldmodel.excitation_corpus as ec
    from imas_ambix.worldmodel.actuator_plan import ExcitationWindow

    # fps fixed via a synthetic frame-time axis (600 Hz, 400 frames).
    ftime = np.arange(400) / 600.0
    monkeypatch.setattr(ec, "_frame_times", lambda *a, **k: ftime)
    captured = {}

    def _fake_excite(shot_id, span, **k):
        captured["span"] = span
        return ExcitationWindow(
            start_frame=42,
            score=1234.0,
            max_abs_ip=7.0e5,
            present_fraction=0.95,
            reason="",
        )

    monkeypatch.setattr(ec, "find_excitation_window", _fake_excite)
    # quiet the unused import warning by referencing ap once.
    assert ap.N_ACTUATOR_CHANNELS > 0

    cw = ec.select_curated_window_for_shot(123, target_horizon_s=0.25, max_n_frames=48)
    assert cw is not None
    assert cw.shot_id == 123
    assert cw.start_frame == 42
    assert cw.fps == pytest.approx(600.0, rel=0.01)
    # span handed to the selector matches the window shape derived from fps.
    assert captured["span"] == (cw.n_frames - 1) * cw.frame_stride + 1
    # 0.25 s at 600 Hz is ~151 native frames -> capped n_frames + stride > 1.
    assert cw.frame_stride >= 2
    assert cw.excitation_score == pytest.approx(1234.0)


def test_select_curated_window_for_shot_none_on_unreadable_fps(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec

    monkeypatch.setattr(ec, "_frame_times", lambda *a, **k: None)
    assert ec.select_curated_window_for_shot(5) is None
