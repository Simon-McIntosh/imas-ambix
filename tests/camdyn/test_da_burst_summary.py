"""Causality and matched-control tests for fast Dalpha conditioning."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.conditioning import (
    CONDITIONING_CHANNELS,
    causal_shuffle_summaries,
    load_conditioning,
    reduce_interframe_rms,
)
from imas_ambix.camdyn.train import TrainConfig


def test_interval_rms_uses_only_samples_at_or_before_each_frame():
    signal_time = np.array([0.25, 0.75, 1.25, 1.75, 2.01])
    signal_value = np.array([1.0, 3.0, 5.0, 7.0, 1.0e6])
    frame_time = np.array([1.0, 2.0])

    reduced, missing = reduce_interframe_rms(signal_time, signal_value, frame_time)

    np.testing.assert_allclose(reduced, [np.sqrt(5.0), np.sqrt(37.0)])
    np.testing.assert_array_equal(missing, 0.0)


def test_interval_rms_includes_sample_on_frame_boundary():
    reduced, missing = reduce_interframe_rms(
        np.array([0.0, 1.0, 2.0]),
        np.array([2.0, 4.0, 8.0]),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(reduced, [np.sqrt(10.0), np.sqrt(40.0)])
    np.testing.assert_array_equal(missing, 0.0)


def test_slow_profile_reduction_accepts_time_on_last_axis():
    signal_time = np.array([0.25, 0.75, 1.25, 1.75])
    radial_profile = np.array(
        [
            [1.0, 3.0, 5.0, 7.0],
            [1.0, 3.0, 5.0, 7.0],
        ]
    )
    reduced, missing = reduce_interframe_rms(
        signal_time, radial_profile, np.array([1.0, 2.0])
    )
    np.testing.assert_allclose(reduced, [np.sqrt(5.0), np.sqrt(37.0)])
    np.testing.assert_array_equal(missing, 0.0)


def test_load_conditioning_reduces_native_fast_carrier(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "shot.zarr"
    store = zarr.open_group(str(path), mode="w")
    xim = store.create_group("xim")
    signal_time = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64)
    signal_value = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    xim.create_array("time", data=signal_time)
    xim.create_array("da_hm10_t", data=signal_value)

    sample = load_conditioning(
        path,
        np.array([1.0, 2.0]),
        24065,
        channels=(CONDITIONING_CHANNELS.select("native")[-1],),
    )

    assert sample.channel_keys == ["dalpha_burst_rms"]
    np.testing.assert_allclose(sample.values[:, 0], [np.sqrt(5.0), np.sqrt(37.0)])
    np.testing.assert_array_equal(sample.missing[:, 0], 0.0)


def test_causal_shuffle_never_selects_a_future_summary():
    values = np.arange(32, dtype=np.float32)
    missing = np.zeros(32, dtype=np.float32)
    shuffled, shuffled_missing, source = causal_shuffle_summaries(
        values, missing, seed=24065
    )
    assert np.all(source <= np.arange(source.size))
    assert np.all(source[1:] < np.arange(source.size)[1:])
    assert np.any(source[2:] != np.arange(source.size)[2:])
    np.testing.assert_array_equal(shuffled, values[source])
    np.testing.assert_array_equal(shuffled_missing, missing[source])


@pytest.mark.parametrize(
    ("variant", "expected_key", "expected_source"),
    [
        ("native", "dalpha_burst_rms", "xim"),
        ("shuffled", "dalpha_burst_rms_shuffled", "xim"),
        ("slow_only", "dalpha_slow_rms", "ada"),
    ],
)
def test_burst_and_control_variants_are_width_matched_and_selectable(
    variant, expected_key, expected_source
):
    selected = CONDITIONING_CHANNELS.select(variant)
    assert len(selected) == len(CONDITIONING_CHANNELS)
    assert selected[-1].key == expected_key
    assert selected[-1].source == expected_source
    assert not any(c.is_fast_input and c.is_probe_target for c in selected)


def test_unknown_burst_variant_fails_closed():
    with pytest.raises(ValueError, match="unknown burst conditioning variant"):
        CONDITIONING_CHANNELS.select("future_window")


def test_channel_catalog_is_safe_for_spawned_data_loader_workers():
    restored = pickle.loads(pickle.dumps(CONDITIONING_CHANNELS))
    assert tuple(restored) == tuple(CONDITIONING_CHANNELS)
    assert restored.burst_variants == CONDITIONING_CHANNELS.burst_variants
    assert restored.select("shuffled")[-1].key == "dalpha_burst_rms_shuffled"


def test_matched_pair_configs_differ_only_in_temporal_attention_and_run_name():
    config_dir = (
        Path(__file__).resolve().parents[2] / "imas_ambix" / "camdyn" / "configs"
    )
    baseline = TrainConfig.load(config_dir / "da_burst_baseline.yaml").to_dict()
    dynamics = TrainConfig.load(config_dir / "da_burst_dynamics.yaml").to_dict()

    top_diff = {
        key
        for key in baseline
        if key not in {"model", "run_name"} and baseline[key] != dynamics[key]
    }
    model_diff = {
        key
        for key in baseline["model"]
        if baseline["model"][key] != dynamics["model"][key]
    }
    assert top_diff == set()
    assert model_diff == {"temporal_attention"}
    assert baseline["run_name"] == "da_burst_baseline"
    assert dynamics["run_name"] == "da_burst_dynamics"
    assert baseline["model"]["cond_channels"] == len(CONDITIONING_CHANNELS)
    assert dynamics["model"]["cond_channels"] == len(CONDITIONING_CHANNELS)
