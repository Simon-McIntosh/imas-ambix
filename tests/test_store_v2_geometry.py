"""Geometry-attachment + backward-compatibility tests for the v2 token store."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.gs.geometry_export import (
    GEOMETRY_FEATURE_NAMES,
    N_GEOMETRY_FEATURES,
)
from imas_ambix.tokenizer.geometry_reader import (
    AlignedGeometry,
    geometry_for_channels,
    read_store_geometry,
)
from imas_ambix.tokenizer.registry import VOCAB_VERSION
from imas_ambix.tokenizer.store_v2 import (
    StoreV2Attrs,
    load_signal_hf_tokens,
    save_signal_hf_tokens,
)


def _attrs(channel_names=("ccbv_01", "obr_06", "fl_cc01"), **over) -> StoreV2Attrs:
    base = dict(
        tokenizer_name="signal_hf_magnetics",
        vocab_version=VOCAB_VERSION,
        native_rate_hz=4_000.0,
        token_rate_hz=4_000.0,
        n_channels=len(channel_names),
        channel_names=tuple(channel_names),
        phase_preserving=False,
        original_window=(0.0, 0.32),
    )
    base.update(over)
    return StoreV2Attrs(**base)


def _tokens(n_tok=10, n_ch=3):
    rng = np.random.default_rng(7)
    tokens = rng.integers(1, 50, size=(n_tok, n_ch)).astype(np.int32)
    token_time = np.linspace(0.0, 0.32, n_tok)
    valid = np.ones((n_tok, n_ch), dtype=bool)
    return tokens, token_time, valid


# --- geometry round-trip ---------------------------------------------------


def test_geometry_roundtrips_through_store(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)

    tokens, token_time, valid = _tokens(n_tok=10, n_ch=3)
    geometry = np.arange(3 * N_GEOMETRY_FEATURES, dtype=np.float32).reshape(
        3, N_GEOMETRY_FEATURES
    )
    geometry[2, 3:] = np.nan  # a flux-loop's NaN orientation/chord must survive
    attrs = _attrs(
        geometry_feature_names=GEOMETRY_FEATURE_NAMES,
        geometry_sensor_kinds=("bpol_probe", "bpol_probe", "flux_loop"),
    )
    save_signal_hf_tokens(
        30460, "magnetics", tokens, token_time, valid, attrs, geometry=geometry
    )

    loaded = load_signal_hf_tokens(30460, "magnetics")
    assert loaded.geometry is not None
    assert loaded.geometry.shape == (3, N_GEOMETRY_FEATURES)
    np.testing.assert_array_equal(
        loaded.geometry[~np.isnan(loaded.geometry)],
        geometry[~np.isnan(geometry)],
    )
    assert np.isnan(loaded.geometry[2, 3:]).all()  # NaN preserved, not zero-filled
    assert loaded.attrs.geometry_feature_names == GEOMETRY_FEATURE_NAMES
    assert loaded.attrs.geometry_sensor_kinds == (
        "bpol_probe",
        "bpol_probe",
        "flux_loop",
    )


# --- backward compatibility: a geometry-less store still loads -------------


def test_geometry_less_store_loads_with_geometry_none(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)

    tokens, token_time, valid = _tokens()
    save_signal_hf_tokens(99, "xma", tokens, token_time, valid, _attrs())

    loaded = load_signal_hf_tokens(99, "xma")
    assert loaded.geometry is None
    # the geometry companion descriptors default to empty for a legacy store
    assert loaded.attrs.geometry_feature_names == ()
    assert loaded.attrs.geometry_sensor_kinds == ()
    # the existing required contract is unchanged
    np.testing.assert_array_equal(loaded.tokens, tokens)
    assert loaded.attrs.tokenizer_name == "signal_hf_magnetics"


def test_geometry_less_attrs_on_disk_unchanged():
    """A geometry-less attrs block must NOT emit the geometry companion keys."""
    a = _attrs()
    on_disk = a.to_attrs()
    assert "geometry_feature_names" not in on_disk
    assert "geometry_sensor_kinds" not in on_disk
    restored = StoreV2Attrs.from_attrs(on_disk)
    assert restored.geometry_feature_names == ()
    assert restored.geometry_sensor_kinds == ()


# --- writer validation -----------------------------------------------------


def test_save_rejects_wrong_geometry_channel_count(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)
    tokens, token_time, valid = _tokens(n_ch=3)
    bad = np.zeros((2, N_GEOMETRY_FEATURES), dtype=np.float32)  # 2 != 3 channels
    with pytest.raises(ValueError, match="geometry shape"):
        save_signal_hf_tokens(
            1, "xma", tokens, token_time, valid, _attrs(), geometry=bad
        )


def test_save_rejects_feature_name_count_mismatch(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)
    tokens, token_time, valid = _tokens(n_ch=3)
    geometry = np.zeros((3, N_GEOMETRY_FEATURES), dtype=np.float32)
    attrs = _attrs(geometry_feature_names=("r", "z"))  # 2 names != n_feat columns
    with pytest.raises(ValueError, match="feature columns"):
        save_signal_hf_tokens(
            1, "xma", tokens, token_time, valid, attrs, geometry=geometry
        )


# --- reader alignment ------------------------------------------------------


def test_read_store_geometry_aligns_to_channel_order(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)

    tokens, token_time, valid = _tokens(n_ch=3)
    geometry = np.arange(3 * N_GEOMETRY_FEATURES, dtype=np.float32).reshape(
        3, N_GEOMETRY_FEATURES
    )
    attrs = _attrs(
        geometry_feature_names=GEOMETRY_FEATURE_NAMES,
        geometry_sensor_kinds=("bpol_probe", "bpol_probe", "flux_loop"),
    )
    save_signal_hf_tokens(
        30460, "magnetics", tokens, token_time, valid, attrs, geometry=geometry
    )

    aligned = read_store_geometry(30460, "magnetics")
    assert isinstance(aligned, AlignedGeometry)
    assert aligned.channel_names == ("ccbv_01", "obr_06", "fl_cc01")
    assert aligned.feature_names == GEOMETRY_FEATURE_NAMES
    assert aligned.features.shape == (3, N_GEOMETRY_FEATURES)
    np.testing.assert_array_equal(aligned.features, geometry)
    assert aligned.sensor_kinds[2] == "flux_loop"


def test_read_store_geometry_returns_none_for_legacy_store(tmp_path, monkeypatch):
    import imas_ambix.tokenizer.store_v2 as mod

    monkeypatch.setattr(mod, "TOKEN_ROOT", tmp_path)
    tokens, token_time, valid = _tokens()
    save_signal_hf_tokens(7, "xma", tokens, token_time, valid, _attrs())
    assert read_store_geometry(7, "xma") is None


def test_geometry_for_channels_falls_back_to_all_nan():
    """With neither a store geometry nor a campaign table, geometry is all-NaN."""
    aligned = geometry_for_channels(["ip", "ccbv01", "interferometer_01"])
    assert aligned.features.shape == (3, N_GEOMETRY_FEATURES)
    assert np.all(np.isnan(aligned.features))
    assert all(k == "scalar" for k in aligned.sensor_kinds)
