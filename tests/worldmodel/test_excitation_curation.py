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


# ---------------------------------------------------------------------------
# excitation_corpus: multi-window tiling (every phase, disruptions included)
# ---------------------------------------------------------------------------


def test_classify_phase_ramp_flat_termination():
    import numpy as np

    from imas_ambix.worldmodel.excitation_corpus import _classify_phase

    thr = 2.0e4
    ramp = np.linspace(0.0, 7.0e5, 60)  # rising
    flat = np.full(60, 7.0e5)  # constant high
    term = np.linspace(7.0e5, 0.0, 60)  # falling / quench
    assert _classify_phase(ramp, present_threshold=thr) == "ramp"
    assert _classify_phase(flat, present_threshold=thr) == "flat_top"
    assert _classify_phase(term, present_threshold=thr) == "termination"


def _patch_enumerate(monkeypatch, ftime, values, missing):
    """Wire enumerate_shot_windows' reads to synthetic in-memory data."""
    import imas_ambix.camdyn.conditioning as cc
    import imas_ambix.camdyn.dataset as cd
    import imas_ambix.worldmodel.actuator_plan as ap
    import imas_ambix.worldmodel.excitation_corpus as ec
    from imas_ambix.camdyn.conditioning import ConditioningSample

    monkeypatch.setattr(ec, "_frame_times", lambda *a, **k: ftime)
    monkeypatch.setattr(cd, "level1_shot_path", lambda *a, **k: "x")
    sample = ConditioningSample(
        shot_id=1,
        frame_time=ftime,
        channel_keys=[c.key for c in ap.ACTUATOR_CHANNELS],
        units=[c.unit for c in ap.ACTUATOR_CHANNELS],
        values=values,
        missing=missing,
    )
    # enumerate_shot_windows imports load_conditioning from camdyn.conditioning
    # lazily; patch it there.
    monkeypatch.setattr(cc, "load_conditioning", lambda *a, **k: sample)


def test_enumerate_shot_windows_tiles_whole_pulse_with_phases(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # A full pulse: breakdown -> ramp -> flat-top -> quench, at 600 Hz over ~1.3s.
    n = 800
    ftime = np.arange(n) / 600.0
    ip = np.concatenate(
        [
            np.linspace(0, 7.0e5, 200),  # ramp
            np.full(400, 7.0e5),  # flat-top
            np.linspace(7.0e5, 0, 200),  # termination/quench
        ]
    )
    C = ec.ACTUATOR_CHANNELS
    nC = len(C)
    values = np.zeros((n, nC), dtype=np.float64)
    missing = np.ones((n, nC), dtype=np.float64)
    ip_col = ec.plasma_current_channel_index()
    values[:, ip_col] = ip
    missing[:, ip_col] = 0.0
    # one coil ramps continuously so every window has coil excitation.
    ck = ec.coil_current_channel_indices()[0]
    values[:, ck] = np.linspace(1.0e5, 3.0e5, n)
    missing[:, ck] = 0.0

    _patch_enumerate(monkeypatch, ftime, values, missing)
    ws = ec.enumerate_shot_windows(
        1,
        target_horizon_s=0.25,
        max_n_frames=48,
        window_time_stride_s=0.05,
        min_excitation=1.0,
    )
    # multiple windows over the pulse
    assert len(ws) > 3
    # ascending start frames
    assert [w.start_frame for w in ws] == sorted(w.start_frame for w in ws)
    # covers more than one phase, including a termination (disruption INCLUDED)
    phases = {w.phase for w in ws}
    assert "termination" in phases
    assert len(phases) >= 2
    # every kept window is plasma-present
    assert all(w.max_abs_ip >= 2.0e4 for w in ws)


def test_enumerate_curated_windows_excludes_all_held_out_windows(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec
    from imas_ambix.worldmodel.excitation_corpus import CuratedWindow

    # each shot yields 3 windows; held-out shots must contribute ZERO.
    def _enum(sid, **k):
        return [
            CuratedWindow(sid, st, 600.0, 39, 4, 1000.0, 7.0e5, 0.9, "flat_top")
            for st in (0, 10, 20)
        ]

    monkeypatch.setattr(ec, "enumerate_shot_windows", _enum)
    out = ec.enumerate_curated_windows(
        [10, 18502, 11, 18504, 18505, 12], held_out=(18502, 18504, 18505)
    )
    sids = {w.shot_id for w in out}
    assert sids == {10, 11, 12}  # all held-out shots fully gone
    assert len(out) == 9  # 3 shots x 3 windows
    # deterministic: ascending shot then start
    assert out == sorted(out, key=lambda w: (w.shot_id, w.start_frame))


def test_enumerate_curated_windows_caps_windows_per_shot(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec
    from imas_ambix.worldmodel.excitation_corpus import CuratedWindow

    def _enum(sid, **k):
        return [
            CuratedWindow(sid, st, 600.0, 39, 4, 1000.0, 7.0e5, 0.9, "flat_top")
            for st in range(0, 100, 10)  # 10 windows
        ]

    monkeypatch.setattr(ec, "enumerate_shot_windows", _enum)
    out = ec.enumerate_curated_windows([7], held_out=(), max_windows_per_shot=4)
    assert len(out) == 4
    # cap keeps first and last (span the whole recording incl. termination)
    starts = [w.start_frame for w in out]
    assert starts[0] == 0 and starts[-1] == 90


# ---------------------------------------------------------------------------
# excitation_corpus: full-shot windows (one window = whole plasma phase)
# ---------------------------------------------------------------------------


def test_plasma_phase_span_spans_breakdown_to_quench(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # 600 Hz; dark 0..50, plasma 50..550 (ramp+flat+quench), dark 550..600.
    n = 600
    ftime = np.arange(n) / 600.0
    absip = np.zeros(n)
    absip[50:250] = np.linspace(0, 7.0e5, 200)  # ramp
    absip[250:500] = 7.0e5  # flat-top
    absip[500:550] = np.linspace(7.0e5, 0, 50)  # quench
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))

    span = ec.find_plasma_phase_span(1, ip_present_threshold=2.0e4)
    assert span.valid
    # starts at first present frame (breakdown, ~frame 56 where ramp crosses 20kA)
    assert 50 <= span.start_frame <= 70
    # ends just past the last present frame (the quench), trimming trailing dark
    assert 540 <= span.end_frame <= 552
    assert span.end_frame <= 552  # trailing dark (550..600) trimmed
    assert span.present_fraction >= 0.9  # the span itself is mostly plasma
    assert span.max_abs_ip == pytest.approx(7.0e5)
    assert span.duration_s > 0.7  # ~0.8 s plasma phase


def test_plasma_phase_span_rejects_mostly_dark(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # plasma only flickers: present at the ends, long dark gap between -> the
    # SPAN (first..last present) is mostly dark -> rejected.
    n = 600
    ftime = np.arange(n) / 600.0
    absip = np.zeros(n)
    absip[10:30] = 5.0e5  # early blip
    absip[560:580] = 5.0e5  # late blip
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))

    span = ec.find_plasma_phase_span(
        1, ip_present_threshold=2.0e4, min_present_fraction=0.7
    )
    assert not span.valid
    assert span.reason == "mostly_dark"


def test_plasma_phase_span_rejects_vacuum(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    n = 200
    ftime = np.arange(n) / 600.0
    absip = np.full(n, 1.0e3)  # 1 kA everywhere, below 20 kA
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))
    span = ec.find_plasma_phase_span(1, ip_present_threshold=2.0e4)
    assert not span.valid
    assert span.reason == "no_plasma"


def test_plasma_phase_span_allows_start_in_plasma(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # recording BEGINS already in plasma (no dark breakdown lead-in) -> span
    # starts at frame 0.
    n = 300
    ftime = np.arange(n) / 600.0
    absip = np.concatenate([np.full(250, 7.0e5), np.linspace(7.0e5, 0, 50)])
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))
    span = ec.find_plasma_phase_span(1, ip_present_threshold=2.0e4)
    assert span.valid
    assert span.start_frame == 0


def test_select_fullshot_windows_excludes_held_out(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec
    from imas_ambix.worldmodel.excitation_corpus import CuratedWindow

    def _sel(sid, **k):
        return CuratedWindow(
            sid, 0, 600.0, 200, 1, 1000.0, 7.0e5, 0.9, "full", 200, 0.33
        )

    monkeypatch.setattr(ec, "select_fullshot_window", _sel)
    out = ec.select_fullshot_windows(
        [10, 18502, 11, 18504, 18505, 12], held_out=(18502, 18504, 18505)
    )
    sids = {w.shot_id for w in out}
    assert sids == {10, 11, 12}  # all held-out gone
    assert all(w.phase == "full" for w in out)
    assert all(w.end_frame > w.start_frame for w in out)
    assert all(w.plasma_duration_s > 0 for w in out)


def test_probe_plasma_activity_reports_without_excluding(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # an "active" held-out shot vs a "dark" one
    n = 300
    ftime = np.arange(n) / 600.0

    def _absip(shot_id, *a, **k):
        if shot_id == 18504:  # low-activity / brief — the dark-frame-artifact case
            v = np.zeros(n)
            v[10:20] = 3.0e4  # only a brief, weak blip
            return ftime, v
        v = np.full(n, 6.0e5)  # plasma-active throughout
        return ftime, v

    monkeypatch.setattr(ec, "_shot_abs_ip", _absip)
    act = ec.probe_plasma_activity([18502, 18504])
    # probe inspects held-out shots (does NOT exclude them)
    assert set(act) == {18502, 18504}
    # the ACTIVE shot: strong current, long plasma phase.
    assert act[18502]["max_abs_ip"] == pytest.approx(6.0e5)
    assert act[18502]["duration_s"] > 0.3
    # the BRIEF/weak shot: the artifact signals are SHORT duration + LOW max|Ip|
    # (the within-span present_fraction is ~1.0 by construction — the span IS the
    # plasma region — so duration + peak current are what flag a dark artifact).
    assert act[18504]["max_abs_ip"] < 1.0e5
    assert act[18504]["duration_s"] < 0.05  # ~10 frames @ 600 Hz


# --- robust sustained-presence detection (the ~20 ms ripple-artifact fix) ---


def test_sustained_mask_fills_dips_and_drops_blips():
    import numpy as np

    from imas_ambix.worldmodel.excitation_corpus import _sustained_present_mask

    # an isolated 2-frame blip at the start, then a sustained block with 2-frame
    # interior dips (the 8-on/2-off Ip ripple) -> close should fill the dips,
    # open should drop the leading blip.
    raw = np.zeros(60, dtype=bool)
    raw[2:4] = True  # isolated blip
    # sustained block 20..56 with 2-frame dips every 10 frames
    raw[20:56] = True
    raw[30:32] = False
    raw[42:44] = False
    clean = _sustained_present_mask(raw, gap_fill=8, min_run=4)
    # the leading 2-frame blip is dropped
    assert not clean[2:4].any()
    # the interior dips are filled -> one contiguous block 20..56
    assert clean[20:56].all()
    idx = np.flatnonzero(clean)
    assert idx[0] == 20 and idx[-1] == 55


def test_plasma_phase_span_robust_to_threshold_ripple(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # reproduce the artifact shape: Ip oscillates across the threshold (8 on /
    # 2 off) over a 200-frame sustained plasma, embedded in a longer recording.
    n = 412
    ftime = np.arange(n) / 600.0
    absip = np.zeros(n)
    # 100..300: ripple around the threshold (mean well above, dips just under)
    for s in range(100, 300, 10):
        absip[s : s + 8] = 7.0e5  # above
        absip[s + 8 : s + 10] = 1.0e4  # brief dip under 20 kA
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))

    span = ec.find_plasma_phase_span(
        1, ip_present_threshold=2.0e4, gap_fill_frames=8, min_sustained_frames=4
    )
    assert span.valid
    # the dips are FILLED -> present_fraction ~1.0 (not ~0.8 from the raw ripple)
    assert span.present_fraction > 0.95
    # span covers the WHOLE sustained region (~100..300), not a noisy sub-window
    assert span.start_frame <= 105
    assert span.end_frame >= 295


def test_plasma_phase_span_min_duration_drops_short_burst(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # a high-speed burst: 200 plasma frames packed into 20 ms (10 kHz) inside a
    # slow recording — physically real but too short for the full-shot regime.
    n = 412
    ftime = np.empty(n)
    ftime[:100] = np.arange(100) * 2e-3  # 500 Hz slow lead-in
    ftime[100:300] = 0.2 + np.arange(200) * 1e-4  # 10 kHz burst (20 ms)
    ftime[300:] = ftime[299] + np.arange(1, n - 299) * 2e-3  # slow tail
    absip = np.zeros(n)
    absip[100:300] = 7.0e5  # plasma only during the burst
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))

    # no floor: the span is valid but only ~20 ms
    s0 = ec.find_plasma_phase_span(1, ip_present_threshold=2.0e4, min_duration_s=0.0)
    assert s0.valid
    assert s0.duration_s < 0.03
    # with a 0.1 s floor: dropped as too_short
    s1 = ec.find_plasma_phase_span(1, ip_present_threshold=2.0e4, min_duration_s=0.1)
    assert not s1.valid
    assert s1.reason == "too_short"


# ---------------------------------------------------------------------------
# excitation_corpus: UNIFIED multi-camera, multi-timescale corpus
# ---------------------------------------------------------------------------


def test_campaign_for_shot_bands():
    from imas_ambix.worldmodel.excitation_corpus import campaign_for_shot

    # factual 5000-wide shot-id bands (exact MAST M-campaign boundaries are not
    # authoritatively published, so we label by id band, never a guessed name).
    assert campaign_for_shot(14000) == "lt_15k"
    assert campaign_for_shot(15085) == "15k-20k"  # tokenised corpus start
    assert campaign_for_shot(20316) == "20k-25k"
    assert campaign_for_shot(25000) == "25k-30k"
    assert campaign_for_shot(30473) == "30k-35k"  # corpus max
    assert campaign_for_shot(37000) == "ge_35k"


def test_select_unified_window_schema_and_timescale(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # SLOW shot: 0.4 s plasma phase at 600 Hz.
    n = 300
    ftime = np.arange(n) / 600.0
    absip = np.concatenate([np.full(250, 7.0e5), np.linspace(7.0e5, 0, 50)])
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))
    monkeypatch.setattr(ec, "_frame_times", lambda *a, **k: ftime)
    monkeypatch.setattr(ec, "_span_excitation_score", lambda *a, **k: 1234.0)

    row = ec.select_unified_window(20316, "rco", fast_max_duration_s=0.15)
    assert row is not None
    # exact corpus->model schema keys
    assert set(row) == {
        "shot_id",
        "camera_id",
        "campaign",
        "start_frame",
        "end_frame",
        "fps",
        "n_frames",
        "plasma_duration_s",
        "timescale",
        "excitation_score",
        "present_fraction",
        "frame_times",
    }
    assert row["camera_id"] == "rco"
    assert row["campaign"] == "20k-25k"  # shot 20316 id-band
    assert row["timescale"] == "slow"  # ~0.42 s > 0.15 s
    assert len(row["frame_times"]) == row["n_frames"]
    assert row["fps"] == pytest.approx(600.0, rel=0.01)


def test_select_unified_window_tags_fast_burst(monkeypatch):
    import numpy as np

    import imas_ambix.worldmodel.excitation_corpus as ec

    # FAST burst: 200 plasma frames in 20 ms (10 kHz) — admitted, tagged fast.
    n = 412
    ftime = np.empty(n)
    ftime[:100] = np.arange(100) * 2e-3
    ftime[100:300] = 0.2 + np.arange(200) * 1e-4
    ftime[300:] = ftime[299] + np.arange(1, n - 299) * 2e-3
    absip = np.zeros(n)
    absip[100:300] = 7.0e5
    monkeypatch.setattr(ec, "_shot_abs_ip", lambda *a, **k: (ftime, absip))
    monkeypatch.setattr(ec, "_frame_times", lambda *a, **k: ftime)
    monkeypatch.setattr(ec, "_span_excitation_score", lambda *a, **k: 50.0)

    row = ec.select_unified_window(20316, "rbb", fast_max_duration_s=0.15)
    assert row is not None
    assert row["timescale"] == "fast"  # ~20 ms < 150 ms — admitted, not dropped
    assert row["plasma_duration_s"] < 0.03


def test_select_unified_windows_multicamera_and_heldout(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec

    # every (shot,camera) yields a window; camera_frame_count says all present.
    monkeypatch.setattr(
        ec,
        "select_unified_window",
        lambda sid, cam, **k: {
            "shot_id": sid,
            "camera_id": cam,
            "campaign": "M8",
            "start_frame": 0,
            "end_frame": 100,
            "fps": 600.0,
            "n_frames": 100,
            "plasma_duration_s": 0.33,
            "timescale": "slow",
            "excitation_score": 1.0,
            "present_fraction": 0.9,
        },
    )
    import imas_ambix.worldmodel.spacetime_dataset as sd

    monkeypatch.setattr(sd, "camera_frame_count", lambda *a, **k: 500)
    out = ec.select_unified_windows(
        [20316, 18502, 20317],
        cameras=["rbb", "rco"],
        held_out=(18502,),
    )
    sids = {r["shot_id"] for r in out}
    assert sids == {20316, 20317}  # held-out 18502 fully excluded (BOTH cameras)
    # 2 shots x 2 cameras = 4 windows
    assert len(out) == 4
    assert {(r["shot_id"], r["camera_id"]) for r in out} == {
        (20316, "rbb"),
        (20316, "rco"),
        (20317, "rbb"),
        (20317, "rco"),
    }


def test_select_unified_windows_skips_absent_camera(monkeypatch):
    import imas_ambix.worldmodel.excitation_corpus as ec
    import imas_ambix.worldmodel.spacetime_dataset as sd

    # rgc absent (raises) for shot 20316; rbb present.
    def _count(sid, cam, **k):
        if cam == "rgc":
            raise FileNotFoundError("no rgc")
        return 500

    monkeypatch.setattr(sd, "camera_frame_count", _count)
    monkeypatch.setattr(
        ec,
        "select_unified_window",
        lambda sid, cam, **k: {"shot_id": sid, "camera_id": cam, "n_frames": 100},
    )
    out = ec.select_unified_windows([20316], cameras=["rbb", "rgc"], held_out=())
    assert {r["camera_id"] for r in out} == {"rbb"}  # rgc skipped, no crash
