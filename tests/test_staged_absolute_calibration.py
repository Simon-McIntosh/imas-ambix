"""Absolute calibration on the STAGED read path + corpus_compute parity.

The diagnostics->equilibrium oracle reads the "magnetics" stream through the
STAGED path (spacetime_dataset_v2._read_staged_signal -> _quantise_l2), which
historically z-scores PER-SHOT — so a signals_hf re-encode alone cannot move
the gate.  These tests prove:

- the raw-read helper's NAME list === the column order _quantise_l2 sees;
- absolute mode maps the SAME physical value to the SAME bin across two shots
  with different per-shot ranges (per-shot does NOT);
- calibration=None is byte-identical to the historic per-shot behaviour;
- corpus_compute's STAGED keys === the staged reader's column names, and its
  L2 keys === l2_input_build's expanded channel names (parity by construction).
"""

from __future__ import annotations

import numpy as np

from imas_ambix.calibration.signals import ChannelCalibration
from imas_ambix.worldmodel import spacetime_dataset_v2 as sd2


def _cal(name: str, mean: float, std: float) -> ChannelCalibration:
    return ChannelCalibration(
        name=name,
        mean=mean,
        std=std,
        min_value=mean - 5 * std,
        max_value=mean + 5 * std,
        q01=mean - 2 * std,
        q50=mean,
        q99=mean + 2 * std,
        n_samples=1000,
        n_shots=10,
    )


def _write_staged_store(
    token_root,
    group: str,
    shot_id: int,
    arrays: dict[str, np.ndarray],
    time: np.ndarray,
) -> None:
    """Write a synthetic staged store at the exact layout the reader expects."""
    import zarr

    path = (
        token_root
        / "v1"
        / f"signals-{group}"
        / group
        / str(int(shot_id))
        / f"{group}.zarr"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store["time"] = np.asarray(time, dtype=np.float64)
    for k, v in arrays.items():
        store[k] = np.asarray(v)


# ---------------------------------------------------------------------------
# raw-read helper <-> _quantise_l2 column parity
# ---------------------------------------------------------------------------


def test_raw_read_names_match_quantise_column_order(tmp_path):
    """The NAME list from _read_staged_raw is 1:1 with the columns _quantise_l2
    quantises — traces (sorted key) first, then 2-D profile columns named
    {key}[i] for the kept (strided) columns."""
    n_t = 20
    time = np.linspace(0.0, 1.0, n_t)
    arrays = {
        # two 1-D traces (sorted key order: bcoil, ipla)
        "ipla": np.linspace(0.0, 1.0, n_t),
        "bcoil": np.linspace(-1.0, 1.0, n_t),
        # one 2-D (T, 4) profile array
        "probes": np.outer(np.linspace(0, 1, n_t), np.arange(1, 5)),
    }
    _write_staged_store(tmp_path, "magnetics", 1, arrays, time)

    raw, names, t = sd2._read_staged_raw(
        "magnetics", 1, token_root=tmp_path, profile_r_stride=1
    )
    # traces sorted-key first (bcoil, ipla), then probes[0..3]
    assert names == [
        "bcoil",
        "ipla",
        "probes[0]",
        "probes[1]",
        "probes[2]",
        "probes[3]",
    ]
    assert raw.shape == (n_t, len(names))
    assert t.shape == (n_t,)

    # _quantise_l2 over the same raw must produce one column per name, in order.
    tok = sd2._quantise_l2(raw, channel_names=names)
    assert tok.shape == (n_t, len(names))

    # And the full read path yields exactly that token column count.
    tok2, _ = sd2._read_staged_signal(
        "magnetics", 1, token_root=tmp_path, profile_r_stride=1
    )
    assert tok2.shape == tok.shape


def test_profile_stride_drops_columns_and_renames(tmp_path):
    """A stride>1 keeps a subset of profile columns and names them by KEPT index."""
    n_t = 12
    time = np.linspace(0, 1, n_t)
    arrays = {"prof": np.outer(np.linspace(0, 1, n_t), np.arange(1, 9))}  # (T, 8)
    _write_staged_store(tmp_path, "ait", 5, arrays, time)

    raw, names, _ = sd2._read_staged_raw(
        "ait", 5, token_root=tmp_path, profile_r_stride=4
    )
    # columns 0 and 4 kept -> 2 columns, named by kept index 0,1
    assert names == ["prof[0]", "prof[1]"]
    assert raw.shape == (n_t, 2)


# ---------------------------------------------------------------------------
# absolute vs per-shot quantisation
# ---------------------------------------------------------------------------


def test_absolute_mode_same_value_same_bin_across_shots():
    name = "b_probe"
    cal = {name: _cal(name, mean=1.0, std=0.5)}
    probe = 0.5
    # two "shots" of the same channel, very different per-shot ranges
    col_a = np.array([probe, 0.6, 0.7, 0.8])[:, None]
    col_b = np.array([probe, 3.0, -2.0, 5.0])[:, None]

    bin_a = sd2._quantise_l2(col_a, channel_names=[name], calibration=cal)[0, 0]
    bin_b = sd2._quantise_l2(col_b, channel_names=[name], calibration=cal)[0, 0]
    assert int(bin_a) == int(bin_b)


def test_per_shot_mode_same_value_different_bins():
    probe = 0.5
    col_a = np.array([probe, 0.6, 0.7, 0.8])[:, None]
    col_b = np.array([probe, 3.0, -2.0, 5.0])[:, None]
    # no calibration => per-shot (historic)
    bin_a = sd2._quantise_l2(col_a)[0, 0]
    bin_b = sd2._quantise_l2(col_b)[0, 0]
    assert int(bin_a) != int(bin_b)


def test_quantise_byte_identical_without_calibration():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 3))
    base = sd2._quantise_l2(x)
    # passing names but calibration=None must be byte-identical
    same = sd2._quantise_l2(x, channel_names=["a", "b", "c"], calibration=None)
    np.testing.assert_array_equal(base, same)


def test_missing_channel_falls_back_with_warning(caplog):
    import logging

    cal = {"present": _cal("present", 0.0, 1.0)}
    x = np.stack(
        [np.array([0.0, 1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0, 40.0])], axis=1
    )
    sd2._WARNED_STAGED_CHANNELS.clear()
    with caplog.at_level(logging.WARNING):
        sd2._quantise_l2(x, channel_names=["present", "missing"], calibration=cal)
    assert any("missing" in r.getMessage() for r in caplog.records)


def test_read_window_signals_calibration_default_is_none_safe(tmp_path):
    """read_window_signals must accept calibration_by_group and, when omitted,
    behave byte-identically to the default (no calibration) on a staged read."""
    n_t = 16
    time = np.linspace(0.0, 1.0, n_t)
    arrays = {
        "ipla": np.linspace(0, 1, n_t),
        "probes": np.outer(np.linspace(0, 1, n_t), np.arange(1, 4)),
    }
    _write_staged_store(tmp_path, "magnetics", 7, arrays, time)

    spec = sd2.SignalModalitySpec(
        "magnetics",
        "magnetics",
        sd2._l2_vocab(),
        max_channels=48,
        kind="staged",
        profile_r_stride=1,
    )

    class _Sample:
        frame_time = np.linspace(0.0, 1.0, 6)

    out_default = sd2.read_window_signals(7, _Sample(), [spec], 4, token_root=tmp_path)
    out_none = sd2.read_window_signals(
        7, _Sample(), [spec], 4, token_root=tmp_path, calibration_by_group=None
    )
    assert "magnetics" in out_default
    np.testing.assert_array_equal(out_default["magnetics"], out_none["magnetics"])

    # With calibration supplied for the channels, the call still succeeds and
    # returns the same shape (values may differ — absolute binning).
    cal = {
        "ipla": _cal("ipla", 0.5, 0.3),
        "probes[0]": _cal("probes[0]", 0.0, 1.0),
        "probes[1]": _cal("probes[1]", 0.0, 1.0),
        "probes[2]": _cal("probes[2]", 0.0, 1.0),
    }
    out_abs = sd2.read_window_signals(
        7,
        _Sample(),
        [spec],
        4,
        token_root=tmp_path,
        calibration_by_group={"magnetics": cal},
    )
    assert out_abs["magnetics"].shape == out_default["magnetics"].shape


# ---------------------------------------------------------------------------
# corpus_compute parity: keys === read-time channel names
# ---------------------------------------------------------------------------


def test_corpus_compute_staged_keys_match_reader(tmp_path):
    from imas_ambix.calibration import corpus_compute as cc

    n_t = 18
    time = np.linspace(0, 1, n_t)
    # two shots with the same channel inventory
    for sid in (101, 102):
        arrays = {
            "ipla": np.linspace(0, sid, n_t),
            "probes": np.outer(np.linspace(0, 1, n_t), np.arange(1, 4)) * sid,
        }
        _write_staged_store(tmp_path, "magnetics", sid, arrays, time)

    # reader's column names (parity reference)
    _raw, reader_names, _t = sd2._read_staged_raw(
        "magnetics", 101, token_root=tmp_path, profile_r_stride=1
    )

    # corpus_compute STAGED path must produce exactly those keys.  Point it at
    # the synthetic token_root via monkeypatch-free explicit call: _read_staged_raw
    # defaults token_root to TOKEN_ROOT, so we patch the module's default here.
    import imas_ambix.data.paths as paths

    orig = paths.TOKEN_ROOT
    try:
        paths.TOKEN_ROOT = tmp_path
        cal = cc.compute_group_calibration("magnetics", shot_ids=[101, 102])
    finally:
        paths.TOKEN_ROOT = orig

    assert set(cal.keys()) == set(reader_names)
    # every channel saw both shots
    for c in cal.values():
        assert c.n_shots == 2


def test_corpus_compute_l2_keys_match_l2_reader(monkeypatch, tmp_path):
    """The L2 path keys === l2_input_build.read_group's expanded channel names.

    Uses a stubbed read_group so the test needs no on-disk L2 corpus — the
    point under test is that corpus_compute keys on ch.name verbatim.
    """
    from imas_ambix.calibration import corpus_compute as cc
    from imas_ambix.data import l2_input_build as l2b

    n_t = 10

    class _Ch:
        def __init__(self, name, values):
            self.name = name
            self.values = values

    class _Read:
        channels = [
            _Ch("summary.ip", np.linspace(0, 1, n_t)),
            _Ch("summary.power_nbi", np.linspace(1, 2, n_t)),
        ]

    def _fake_read_group(sid, spec, level2_dir):
        return _Read()

    monkeypatch.setattr(l2b, "read_group", _fake_read_group)
    cal = cc.compute_group_calibration("summary", shot_ids=[1, 2, 3])
    assert set(cal.keys()) == {"summary.ip", "summary.power_nbi"}
    for c in cal.values():
        assert c.n_shots == 3


def test_corpus_compute_known_groups_exclude_magnetics_from_l2():
    from imas_ambix.calibration import corpus_compute as cc

    assert "magnetics" in cc.STAGED_GROUPS
    assert "magnetics" not in cc.L2_GROUPS
    assert "xma" in cc.WINDOW_GROUPS
    # KNOWN_GROUPS is the union of the three families
    assert set(cc.KNOWN_GROUPS) == set(
        cc.WINDOW_GROUPS + cc.STAGED_GROUPS + cc.L2_GROUPS
    )
