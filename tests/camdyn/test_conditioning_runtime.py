"""Runtime dispatch tests for reduced conditioning channels."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from imas_ambix.camdyn.conditioning import (
    CONDITIONING_CHANNELS,
    load_conditioning,
    resample_to_frames,
)
from imas_ambix.camdyn.dataset import FrameWindowConfig, discover_token_shots
from imas_ambix.camdyn.loader import CamdynWindowStream
from imas_ambix.camdyn.masking import ClipMaskConfig
from imas_ambix.camdyn.train import TrainConfig, Trainer


def _specs(corpus):
    return discover_token_shots(
        token_root=corpus["token_root"],
        level1_dir=corpus["level1_dir"],
        shot_ids=corpus["shot_ids"],
        read_n_frames=True,
    )


def _install_burst_and_slow_traces(spec):
    import zarr

    store = zarr.open_group(str(spec.level1_path), mode="a")
    frame_time = np.asarray(store["rbb/time"], dtype=np.float64)
    dt = float(np.median(np.diff(frame_time)))

    fast_time = np.linspace(
        frame_time[0] - dt,
        frame_time[-1],
        frame_time.size * 8,
        dtype=np.float64,
    )
    fast_value = (
        2.0
        + 0.15 * np.arange(fast_time.size)
        + 1.25 * (np.arange(fast_time.size) % 3 == 0)
    ).astype(np.float32)
    xim = store.create_group("xim")
    xim.create_array("time", data=fast_time)
    xim.create_array("da_hm10_t", data=fast_value)

    slow_time = np.asarray(store["ada/time"], dtype=np.float64)
    slow_profile = np.stack(
        [
            20.0 + 0.3 * np.arange(slow_time.size),
            24.0 + 0.5 * np.arange(slow_time.size),
        ]
    ).astype(np.float32)
    store["ada"].create_array("dalpha_raw_full", data=slow_profile)
    return {
        "native": (fast_time, fast_value),
        "shuffled": (fast_time, fast_value),
        "slow_only": (slow_time, slow_profile.mean(axis=0)),
    }


def test_window_stream_dispatches_reduction_and_control_variant(synthetic_corpus):
    specs = _specs(synthetic_corpus)
    spec = specs[0]
    source_traces = _install_burst_and_slow_traces(spec)
    outputs = {}

    for variant in CONDITIONING_CHANNELS.burst_variants:
        channels = CONDITIONING_CHANNELS.select(variant)
        stream = CamdynWindowStream(
            [spec],
            FrameWindowConfig(n_frames=6, stride=4),
            ClipMaskConfig(),
            seed=11,
            max_windows=1,
            channels=channels,
        )
        item = next(iter(stream))
        outputs[variant] = item

        normalisation_path = load_conditioning(
            spec.level1_path,
            item["frame_time"],
            int(item["shot_id"]),
            channels=channels,
        )
        np.testing.assert_allclose(
            item["cond_values"], normalisation_path.values, rtol=0, atol=0
        )
        np.testing.assert_array_equal(item["cond_missing"], normalisation_path.missing)

        signal_time, signal_value = source_traces[variant]
        held, _ = resample_to_frames(signal_time, signal_value, item["frame_time"])
        assert not np.allclose(item["cond_values"][:, -1], held)

    for left, right in combinations(outputs, 2):
        assert not np.allclose(
            outputs[left]["cond_values"][:, -1],
            outputs[right]["cond_values"][:, -1],
        )
        np.testing.assert_array_equal(
            outputs[left]["cond_values"][:, :-1],
            outputs[right]["cond_values"][:, :-1],
        )
        np.testing.assert_array_equal(
            outputs[left]["cond_missing"][:, :-1],
            outputs[right]["cond_missing"][:, :-1],
        )


@pytest.mark.parametrize("variant", CONDITIONING_CHANNELS.burst_variants)
def test_training_config_selects_stream_channels(monkeypatch, variant):
    import imas_ambix.camdyn.train as train_module

    trainer = Trainer(TrainConfig(conditioning_variant=variant))
    captured = {}

    def fake_make_loader(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(train_module, "make_loader", fake_make_loader)
    trainer._make_loader(
        [],
        object(),
        seed=3,
        mode=None,
        progress=None,
        max_windows=1,
    )

    assert captured["channels"] == CONDITIONING_CHANNELS.select(variant)
    assert trainer.cfg.to_dict()["conditioning_variant"] == variant


def test_training_config_rejects_unknown_conditioning_variant():
    with pytest.raises(ValueError, match="unknown burst conditioning variant"):
        Trainer(TrainConfig(conditioning_variant="future_window"))
