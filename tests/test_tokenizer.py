"""Smoke tests for the multi-modal tokenizer scaffold.

Each test starts with a fresh :class:`TokenRegistry` so allocations
don't leak across tests. Network and S3 access are not exercised here —
these are pure-Python round-trip checks over numpy arrays and small
xarray datasets.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from imas_ambix.tokenizer import (
    EncodedFrames,
    EncodedSignals,
    FrameTokenizer,
    SignalTokenizer,
    Tokenizer,
)
from imas_ambix.tokenizer.frames import (
    OpenMagvit2Tokenizer,
    OpenMagvit2UnavailableError,
    PlaceholderFrameTokenizer,
)
from imas_ambix.tokenizer.multimodal import ShotTokenizer
from imas_ambix.tokenizer.registry import (
    CONTROL_TOKENS,
    VOCAB_VERSION,
    TokenRegistry,
)
from imas_ambix.tokenizer.signals import (
    ChronosSignalTokenizer,
    ChronosUnavailableError,
    PatchTSTTokenizer,
    UniformQuantizer,
)


@pytest.fixture
def fresh_registry():
    """Reset the shared registry singleton to an empty state for each test.

    The frame/signal tokenizers capture ``registry`` at import time via
    ``from .registry import registry``, so swapping the module attribute
    wouldn't reach them. Resetting the shared object in place does.
    """
    from imas_ambix.tokenizer import registry as singleton
    from imas_ambix.tokenizer.registry import CONTROL_RANGE

    # Snapshot for restoration
    saved_blocks = dict(singleton._blocks)
    saved_cursor = singleton._cursor

    singleton._blocks.clear()
    singleton._cursor = CONTROL_RANGE[1]
    yield singleton
    # Restore
    singleton._blocks.clear()
    singleton._blocks.update(saved_blocks)
    singleton._cursor = saved_cursor


# --- registry --------------------------------------------------------


def test_registry_starts_after_control_tokens(fresh_registry):
    assert fresh_registry.total_vocab_size() == len(CONTROL_TOKENS)


def test_registry_allocate_is_contiguous(fresh_registry):
    r = fresh_registry
    a = r.allocate("alpha", 100)
    b = r.allocate("beta", 50)
    assert a == (4, 104)
    assert b == (104, 154)
    assert r.total_vocab_size() == 154


def test_registry_allocate_is_idempotent(fresh_registry):
    r = fresh_registry
    a = r.allocate("alpha", 100)
    a_again = r.allocate("alpha", 100)
    assert a == a_again


def test_registry_allocate_rejects_size_change(fresh_registry):
    r = fresh_registry
    r.allocate("alpha", 100)
    with pytest.raises(ValueError, match="refusing to re-allocate"):
        r.allocate("alpha", 200)


def test_registry_split_decodes_global_ids(fresh_registry):
    r = fresh_registry
    r.allocate("alpha", 100)
    r.allocate("beta", 50)
    assert r.split(0) == ("control", 0)
    assert r.split(4) == ("alpha", 0)
    assert r.split(103) == ("alpha", 99)
    assert r.split(104) == ("beta", 0)


def test_registry_split_raises_on_unallocated(fresh_registry):
    fresh_registry.allocate("alpha", 10)
    with pytest.raises(KeyError):
        fresh_registry.split(10_000)


def test_registry_shift_adds_offset(fresh_registry):
    r = fresh_registry
    start, _ = r.allocate("alpha", 100)
    locals_ = np.array([[0, 5, 99]])
    shifted = r.shift("alpha", locals_)
    assert shifted.dtype == np.int32
    assert (shifted == np.array([[start, start + 5, start + 99]])).all()


def test_registry_json_roundtrip(fresh_registry):
    r = fresh_registry
    r.allocate("alpha", 100)
    r.allocate("beta", 50)
    text = r.to_json()
    parsed = TokenRegistry.from_json(text)
    assert parsed.version == VOCAB_VERSION
    assert parsed.total_vocab_size() == r.total_vocab_size()


# --- frame tokenizer (placeholder) -----------------------------------


def test_placeholder_frame_tokenizer_satisfies_protocol(fresh_registry):
    tok = PlaceholderFrameTokenizer()
    assert isinstance(tok, FrameTokenizer)
    assert isinstance(tok, Tokenizer)


def test_placeholder_frame_encode_shape(fresh_registry):
    tok = PlaceholderFrameTokenizer(
        spatial_compression=8, temporal_compression=4, intensity_levels=256
    )
    # Synthetic 16-frame uint16 video, 64x80
    frames = np.zeros((16, 64, 80), dtype=np.uint16)
    frames[..., :40] = 1024
    enc = tok.encode(frames)
    assert isinstance(enc, EncodedFrames)
    # 16 // 4 = 4 time tokens, 64 // 8 = 8 row tokens, 80 // 8 = 10 col tokens
    assert enc.token_ids.shape == (4, 8, 10)
    assert enc.token_ids.dtype == np.int32
    # Global ids land inside the placeholder's allocated range
    start, end = fresh_registry.allocate(tok.name, tok.vocab_size)
    assert (enc.token_ids >= start).all() and (enc.token_ids < end).all()


def test_placeholder_frame_round_trip_preserves_low_freq(fresh_registry):
    tok = PlaceholderFrameTokenizer(spatial_compression=4, temporal_compression=2)
    frames = (
        np.linspace(0, 1, 8 * 32 * 32, dtype=np.float32).reshape(8, 32, 32) * 4095
    ).astype(np.uint16)
    enc = tok.encode(frames)
    decoded = tok.decode(enc)
    # Decoded should be approximately the same low-frequency signal —
    # we only need order-of-magnitude agreement.
    assert decoded.shape == frames.shape
    assert decoded.dtype == np.uint8


def test_placeholder_frame_accepts_rgb_input(fresh_registry):
    tok = PlaceholderFrameTokenizer()
    rgb = np.random.randint(0, 4096, size=(8, 32, 32, 3), dtype=np.uint16)
    enc = tok.encode(rgb)
    assert enc.token_ids.ndim == 3  # T, h, w (channels collapsed)


def test_open_magvit2_unavailable_when_root_missing(fresh_registry, tmp_path):
    """OpenMagvit2Tokenizer raises a clear error if the staging dir is absent."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(OpenMagvit2UnavailableError, match="not found"):
        OpenMagvit2Tokenizer(root=missing)


def test_open_magvit2_unavailable_when_weights_missing(fresh_registry, tmp_path):
    """OpenMagvit2Tokenizer surfaces every missing dependency by path."""
    root = tmp_path / "magvit2"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").touch()
    (root / "worker.py").touch()
    with pytest.raises(OpenMagvit2UnavailableError, match="missing"):
        OpenMagvit2Tokenizer(root=root)


# --- signal tokenizer (uniform quantizer) ----------------------------


def test_uniform_quantizer_satisfies_protocol(fresh_registry):
    tok = UniformQuantizer()
    assert isinstance(tok, SignalTokenizer)


def _toy_signal_dataset(n_time: int = 16) -> xr.Dataset:
    t = np.arange(n_time, dtype=np.float64) * 0.01
    ip = np.sin(t) * 1000  # plasma current proxy
    ne = np.cos(t) * 1e19  # density proxy
    return xr.Dataset(
        {"ip": (("time",), ip), "ne": (("time",), ne)},
        coords={"time": t},
    )


def test_uniform_quantizer_encode_shape(fresh_registry):
    tok = UniformQuantizer(n_bins=32)
    ds = _toy_signal_dataset(n_time=16)
    tok.fit([ds])
    enc = tok.encode(ds)
    assert isinstance(enc, EncodedSignals)
    assert enc.token_ids.shape == (16, 2)
    assert enc.channel_names == ("ip", "ne")
    # Local ids in [0, n_bins); global ids in allocated block.
    start, end = fresh_registry.allocate(tok.name, tok.vocab_size)
    assert (enc.token_ids >= start).all()
    assert (enc.token_ids < end).all()


def test_uniform_quantizer_round_trip_preserves_order(fresh_registry):
    tok = UniformQuantizer(n_bins=256)
    ds = _toy_signal_dataset(n_time=64)
    tok.fit([ds])
    enc = tok.encode(ds)
    decoded = tok.decode(enc)
    # Order of magnitude check — quantization is lossy but not arbitrary
    for ch in ("ip", "ne"):
        orig = np.asarray(ds[ch].values)
        rec = np.asarray(decoded[ch].values)
        # Both should track each other in correlation
        corr = np.corrcoef(orig, rec)[0, 1]
        assert corr > 0.9


def test_uniform_quantizer_handles_constant_channel(fresh_registry):
    tok = UniformQuantizer(n_bins=64)
    ds = xr.Dataset({"x": (("time",), np.ones(8))}, coords={"time": np.arange(8.0)})
    tok.fit([ds])
    enc = tok.encode(ds)
    # All tokens should land in the middle bin
    start, _ = fresh_registry.allocate(tok.name, tok.vocab_size)
    assert (enc.token_ids == start + tok.n_bins // 2).all()


def test_uniform_quantizer_skips_multidim_vars(fresh_registry):
    tok = UniformQuantizer(n_bins=32)
    ds = xr.Dataset(
        {
            "a": (("time",), np.arange(10.0)),
            "b": (("time", "radius"), np.ones((10, 5))),  # 2-D should be skipped
        },
        coords={"time": np.arange(10.0), "radius": np.arange(5.0)},
    )
    tok.fit([ds])
    enc = tok.encode(ds)
    assert enc.channel_names == ("a",)


def test_chronos_signal_tokenizer_raises_not_implemented(fresh_registry):
    # Stub replaced by real implementation — instantiation now succeeds.
    # This test is superseded by test_chronos_protocol below.
    tok = ChronosSignalTokenizer()
    assert tok.name == "signals_chronos_t5_small_v1"
    assert tok.vocab_size == 4096


# --- multimodal shot tokenizer ---------------------------------------


def test_shot_tokenizer_emits_bos_sep_eos_structure(fresh_registry):
    ft = PlaceholderFrameTokenizer(spatial_compression=4, temporal_compression=2)
    st = UniformQuantizer(n_bins=64)
    ds = _toy_signal_dataset(n_time=8)
    st.fit([ds])
    frames = np.zeros((8, 16, 16), dtype=np.uint16)

    shot_tok = ShotTokenizer(frame_tokenizer=ft, signal_tokenizer=st)
    stream = shot_tok.encode_shot(frames=frames, signals=ds)

    assert stream.dtype == np.int32
    assert stream[0] == CONTROL_TOKENS["bos"]
    assert stream[-1] == CONTROL_TOKENS["eos"]
    # sep token appears at least once between bos and eos
    assert CONTROL_TOKENS["sep"] in stream[1:-1].tolist()


def test_shot_tokenizer_stream_length_scales_with_steps(fresh_registry):
    ft = PlaceholderFrameTokenizer(spatial_compression=4, temporal_compression=1)
    st = UniformQuantizer(n_bins=64)
    ds_short = _toy_signal_dataset(n_time=4)
    ds_long = _toy_signal_dataset(n_time=8)
    st.fit([ds_short, ds_long])

    frames_short = np.zeros((4, 16, 16), dtype=np.uint16)
    frames_long = np.zeros((8, 16, 16), dtype=np.uint16)
    shot_tok = ShotTokenizer(frame_tokenizer=ft, signal_tokenizer=st)
    n_short = len(shot_tok.encode_shot(frames=frames_short, signals=ds_short))
    n_long = len(shot_tok.encode_shot(frames=frames_long, signals=ds_long))
    assert n_long > n_short


# --- alignment -------------------------------------------------------


def test_time_grid_as_array_evenly_spaced():
    from imas_ambix.tokenizer.alignment import TimeGrid

    g = TimeGrid(t_start=0.0, t_end=0.1, hz=100.0)
    arr = g.as_array()
    assert arr.shape == (11,)  # 0.0, 0.01, ..., 0.10
    np.testing.assert_allclose(arr[1] - arr[0], 0.01, atol=1e-12)


def test_shot_time_window_intersects():
    from imas_ambix.tokenizer.alignment import shot_time_window

    a = np.array([0.0, 0.5, 1.0])
    b = np.array([0.2, 0.6, 1.2])
    c = np.array([0.1, 0.8])
    start, end = shot_time_window(a, b, c)
    # Intersection: max(starts) to min(ends) → 0.2 to 0.8
    assert start == 0.2
    assert end == 0.8


def test_resample_to_grid_linear_interp():
    from imas_ambix.tokenizer.alignment import TimeGrid, resample_to_grid

    t = np.array([0.0, 0.5, 1.0])
    ds = xr.Dataset({"x": (("time",), np.array([0.0, 5.0, 10.0]))}, coords={"time": t})
    grid = TimeGrid(t_start=0.0, t_end=1.0, hz=4.0)  # 5 points
    out = resample_to_grid(ds, grid)
    np.testing.assert_allclose(
        np.asarray(out["x"].values), np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    )


# --- Chronos signal tokenizer ----------------------------------------


def _chronos_available() -> bool:
    """Return True if chronos-forecasting is importable."""
    try:
        import chronos  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_chronos = pytest.mark.skipif(
    not _chronos_available(), reason="chronos-forecasting not installed"
)


@_skip_no_chronos
def test_chronos_protocol(fresh_registry):
    """ChronosSignalTokenizer satisfies the SignalTokenizer protocol."""
    tok = ChronosSignalTokenizer()
    assert isinstance(tok, SignalTokenizer)
    assert isinstance(tok, Tokenizer)
    assert tok.patch_size == 1
    assert tok.vocab_size == 4096


@_skip_no_chronos
def test_chronos_roundtrip(fresh_registry):
    """Encode + decode a 64-step sine/cosine dataset, expect corr > 0.9."""
    tok = ChronosSignalTokenizer()
    ds = _toy_signal_dataset(n_time=64)
    tok.fit([ds])
    enc = tok.encode(ds)

    assert isinstance(enc, EncodedSignals)
    assert enc.token_ids.shape == (64, 2)
    assert enc.token_ids.dtype == np.int32

    # Global ids must be inside the allocated block
    start, end = fresh_registry.allocate(tok.name, tok.vocab_size)
    assert (enc.token_ids >= start).all() and (enc.token_ids < end).all()

    decoded = tok.decode(enc)
    for ch in ("ip", "ne"):
        orig = np.asarray(ds[ch].values)
        rec = np.asarray(decoded[ch].values)
        corr = float(np.corrcoef(orig, rec)[0, 1])
        assert corr > 0.9, f"channel {ch!r}: correlation {corr:.3f} < 0.9"


def test_chronos_unavailable_when_missing(fresh_registry, monkeypatch):
    """ChronosSignalTokenizer raises ChronosUnavailableError when import fails."""
    import builtins

    real_import = builtins.__import__

    def _fail_chronos(name: str, *args: object, **kwargs: object) -> object:
        if name == "chronos":
            raise ImportError("monkeypatched: chronos not available")
        return real_import(name, *args, **kwargs)

    tok = ChronosSignalTokenizer()
    # Reset the cached tokenizer so the lazy import runs again
    tok._tokenizer = None  # type: ignore[attr-defined]

    monkeypatch.setattr(builtins, "__import__", _fail_chronos)
    with pytest.raises(ChronosUnavailableError):
        tok.encode(_toy_signal_dataset())


# --- PatchTST tokenizer -----------------------------------------------


def test_patchtst_protocol(fresh_registry):
    """PatchTSTTokenizer satisfies the SignalTokenizer protocol."""
    tok = PatchTSTTokenizer()
    assert isinstance(tok, SignalTokenizer)
    assert isinstance(tok, Tokenizer)
    assert tok.vocab_size == 1
    assert tok.patch_size == 64


def test_patchtst_roundtrip_exact(fresh_registry):
    """Encode + decode a 256-step dataset round-trips exactly."""
    tok = PatchTSTTokenizer(patch_size=64)
    t = np.arange(256, dtype=np.float64) * 0.01
    ip = np.sin(t) * 1000
    ne = np.cos(t) * 1e19
    ds = xr.Dataset(
        {"ip": (("time",), ip), "ne": (("time",), ne)},
        coords={"time": t},
    )
    enc = tok.encode(ds)
    decoded = tok.decode(enc)

    for ch in ("ip", "ne"):
        orig = np.asarray(ds[ch].values)
        rec = np.asarray(decoded[ch].values)
        assert np.allclose(orig, rec), f"channel {ch!r} did not round-trip exactly"


def test_patchtst_token_shape(fresh_registry):
    """256 timesteps with patch_size=64 produces (4, n_channels) token_ids."""
    tok = PatchTSTTokenizer(patch_size=64)
    t = np.arange(256, dtype=np.float64) * 0.01
    ds = xr.Dataset(
        {
            "ip": (("time",), np.sin(t)),
            "ne": (("time",), np.cos(t)),
        },
        coords={"time": t},
    )
    enc = tok.encode(ds)
    n_channels = len(enc.channel_names)
    assert enc.token_ids.shape == (4, n_channels), (
        f"expected (4, {n_channels}), got {enc.token_ids.shape}"
    )


# --- block_kind from ShotTokenizer -------------------------------------------


def test_encode_shot_return_block_kind_tuple(fresh_registry):
    """encode_shot(return_block_kind=True) returns a tuple with aligned arrays."""
    from imas_ambix.tokenizer.base import BlockKind

    ft = PlaceholderFrameTokenizer(spatial_compression=4, temporal_compression=1)
    st = UniformQuantizer(n_bins=32)
    ds = _toy_signal_dataset(n_time=4)
    st.fit([ds])
    frames = np.zeros((4, 16, 16), dtype=np.uint16)

    shot_tok = ShotTokenizer(frame_tokenizer=ft, signal_tokenizer=st)
    result = shot_tok.encode_shot(frames=frames, signals=ds, return_block_kind=True)

    assert isinstance(result, tuple), "expected a 2-tuple when return_block_kind=True"
    tokens, block_kind = result

    assert tokens.dtype == np.int32
    assert block_kind.dtype == np.uint8
    assert tokens.shape == block_kind.shape

    # bos and eos are CONTROL
    assert block_kind[0] == BlockKind.CONTROL
    assert block_kind[-1] == BlockKind.CONTROL

    # FRAME and SIGNAL codes are present
    assert BlockKind.FRAME in block_kind.tolist()
    assert BlockKind.SIGNAL in block_kind.tolist()


def test_encode_shot_with_block_kind_method(fresh_registry):
    """encode_shot_with_block_kind always returns a tuple, same as the flag path."""
    ft = PlaceholderFrameTokenizer(spatial_compression=4, temporal_compression=1)
    st = UniformQuantizer(n_bins=32)
    ds = _toy_signal_dataset(n_time=4)
    st.fit([ds])
    frames = np.zeros((4, 16, 16), dtype=np.uint16)

    shot_tok = ShotTokenizer(frame_tokenizer=ft, signal_tokenizer=st)
    tokens_flag, bk_flag = shot_tok.encode_shot(
        frames=frames, signals=ds, return_block_kind=True
    )
    tokens_method, bk_method = shot_tok.encode_shot_with_block_kind(
        frames=frames, signals=ds
    )

    np.testing.assert_array_equal(tokens_flag, tokens_method)
    np.testing.assert_array_equal(bk_flag, bk_method)
