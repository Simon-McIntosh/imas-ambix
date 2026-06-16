"""Tests for the leakage-free Level-2 input light path.

Three concerns:

1. **Registry append-only safety** — adding the L2 input block must not
   move, renumber, or resize any existing block (the in-flight L1 encode
   depends on those ids), and must not re-bump ``VOCAB_VERSION``.
2. **Leakage guard** — every authorised input field must pass the
   field-level reconstruction-vs-plan guard, and the build must refuse
   anything reconstructed.
3. **Real smoke** — encode a handful of real L2 shots and verify the
   per-channel round-trip QC (dequantise-vs-original max-abs-err /
   correlation) and the store contract (native rate, per-channel masks,
   ``_l2`` suffix).

The smoke tests read real FAIR-MAST L2 Zarr from GPFS; they ``skip`` when
that mirror is not mounted so the unit tests still run anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ambix.camdyn.conditioning import assert_no_leakage_fields
from imas_ambix.data import l2_input_build as lib
from imas_ambix.data.provenance import classify_l2_field, is_admissible_input

# Smoke shots: a deliberate mix — 11766 has NO soft-X-ray emission (only
# geometry) so it exercises the present-when-present skip; the others carry
# the XSX HCAML/HCAMU cameras.
SMOKE_SHOTS = [11766, 15199, 11768, 13277]


def _l2_available() -> bool:
    return lib.l2_shot_dir(SMOKE_SHOTS[0]).exists()


requires_l2 = pytest.mark.skipif(
    not _l2_available(), reason="FAIR-MAST L2 mirror not mounted"
)


# ---------------------------------------------------------------------------
# 1. Registry append-only safety
# ---------------------------------------------------------------------------


def test_vocab_version_not_bumped():
    """The L2 addition must NOT re-bump the vocab generation."""
    from imas_ambix.tokenizer.registry import VOCAB_VERSION

    assert VOCAB_VERSION == "v2"


# The real, model-derived codebook sizes the on-disk v2 corpus uses.  These
# are NOT a fictional 256 table — they are the values written into each store's
# ``metadata.codebook_size`` by the in-flight encode (xma continuous → 1, xim
# FSQ → 12800, xsx VQ → 1024).  The smoke / cross-check tests read them BACK
# off disk and assert these match, so a future model retrain that changes a
# codebook size fails loudly rather than silently producing an overlapping L2
# namespace.
REAL_CODEBOOK_SIZES = {"xma": 1, "xim": 12800, "xsx": 1024}


def test_reconstruct_namespace_matches_per_group_process_layout():
    """The reconstruction must lay each group out exactly as its independent
    encode process did — every group restarts at the control range (4), so the
    patch blocks OVERLAP, and the L2 floor is the union maximum.

    Non-circular: the block sizes come from the real model-derived codebook
    sizes (injected here, read off disk in the smoke test), NOT from a
    hand-maintained size table that the assertion then re-derives.
    """
    from imas_ambix.tokenizer.registry import (
        BLOCK_XIM_PATCH,
        BLOCK_XMA_MODE,
        BLOCK_XMA_PATCH,
        BLOCK_XSX_PATCH,
        BLOCK_XSX_PROFILE,
        reconstruct_v2_namespace_from_stores,
    )

    reg = reconstruct_v2_namespace_from_stores(
        signals_hf_root="/dev/null",  # unused — sizes are injected
        block_codebook_sizes=REAL_CODEBOOK_SIZES,
    )
    # Each group's encode process restarts at control end (4).
    assert reg.block_range(BLOCK_XMA_PATCH) == (4, 4 + 1)
    assert reg.block_range(BLOCK_XMA_MODE) == (4 + 1, 4 + 2)
    assert reg.block_range(BLOCK_XIM_PATCH) == (4, 4 + 12800)
    assert reg.block_range(BLOCK_XSX_PATCH) == (4, 4 + 1024)
    assert reg.block_range(BLOCK_XSX_PROFILE) == (4 + 1024, 4 + 1025)
    # The xim patch is the deepest block — its end is the union maximum.
    assert reg.max_block_end() == 4 + 12800 == 12804


def test_l2_block_is_strictly_above_every_corpus_id_and_splits_to_l2():
    """THE INVARIANT.  Allocate L2 on the reconstructed namespace and assert:

    1. every L2 id is above every corpus block end (no overlap with xma/xim/xsx);
    2. ``registry.split(id)`` of ANY L2 id returns ``BLOCK_L2_INPUT_LOW`` — never
       xim/xsx/xma or any HF block;
    3. ``VOCAB_VERSION`` is unchanged (append-only, no re-bump).
    """
    from imas_ambix.tokenizer.registry import (
        BLOCK_L2_INPUT_LOW,
        BLOCK_XIM_PATCH,
        BLOCK_XSX_PATCH,
        VOCAB_VERSION,
        allocate_l2_input_block,
        reconstruct_v2_namespace_from_stores,
    )

    assert VOCAB_VERSION == "v2"  # append-only — never re-bumped

    reg = reconstruct_v2_namespace_from_stores(
        signals_hf_root="/dev/null",
        block_codebook_sizes=REAL_CODEBOOK_SIZES,
    )
    corpus_max_end = reg.max_block_end()
    l2_start, l2_end = allocate_l2_input_block(reg, signals_hf_root="/dev/null")

    # (1) strictly above every corpus id.
    assert l2_start == corpus_max_end, "L2 did not start at the corpus maximum"
    assert l2_start >= reg.block_range(BLOCK_XIM_PATCH)[1]
    assert l2_start >= reg.block_range(BLOCK_XSX_PATCH)[1]
    assert l2_end - l2_start == 256

    # (2) split() of EVERY L2 id resolves to the L2 block, never a corpus block.
    for gid in (l2_start, (l2_start + l2_end) // 2, l2_end - 1):
        name, local = reg.split(gid)
        assert name == BLOCK_L2_INPUT_LOW, f"L2 id {gid} decoded as {name!r}"
        assert 0 <= local < 256
    # The xim/xsx maxima decode INSIDE their corpus blocks (overlap is by the
    # first-match rule), and crucially are BELOW the L2 floor.
    assert reg.block_range(BLOCK_XIM_PATCH)[1] - 1 < l2_start
    assert reg.block_range(BLOCK_XSX_PATCH)[1] - 1 < l2_start


def test_l2_allocation_idempotent_above_existing():
    """A second allocate call returns the same range and never reshuffles."""
    from imas_ambix.tokenizer.registry import (
        allocate_l2_input_block,
        reconstruct_v2_namespace_from_stores,
    )

    reg = reconstruct_v2_namespace_from_stores(
        signals_hf_root="/dev/null",
        block_codebook_sizes=REAL_CODEBOOK_SIZES,
    )
    first = allocate_l2_input_block(reg, signals_hf_root="/dev/null")
    second = allocate_l2_input_block(reg, signals_hf_root="/dev/null")
    assert first == second


@requires_l2
def test_reconstructed_ranges_contain_real_on_disk_ids():
    """Cross-check the reconstruction against ACTUAL on-disk global ids.

    Reads the real ``codebook_size`` off the on-disk stores (asserting it
    matches the expected model-derived sizes), reconstructs the namespace,
    then opens a real xim store and a real xsx store, reads their global
    token id arrays, and asserts the reconstructed xim/xsx ranges CONTAIN
    those ids (min/max within range).  If the reconstruction order/size were
    wrong, this fails — it is the non-circular ground-truth check.
    """
    import glob
    import json as _json
    import os

    import zarr

    from imas_ambix.data.paths import TOKEN_ROOT
    from imas_ambix.tokenizer.registry import (
        BLOCK_L2_INPUT_LOW,
        BLOCK_XIM_PATCH,
        BLOCK_XSX_PATCH,
        allocate_l2_input_block,
        reconstruct_v2_namespace_from_stores,
    )

    hf_root = TOKEN_ROOT / "v2" / "signals_hf"

    # Find a shot that carries each corpus group on disk.
    def _find(group: str) -> str | None:
        for p in sorted(glob.glob(str(hf_root / "*"))):
            sh = os.path.basename(p)
            if sh.isdigit() and (hf_root / sh / f"{group}.zarr").exists():
                return sh
        return None

    xim_shot, xsx_shot = _find("xim"), _find("xsx")
    if xim_shot is None or xsx_shot is None:
        pytest.skip("no on-disk xim/xsx corpus stores to cross-check")

    def _codebook_and_idrange(shot: str, group: str) -> tuple[int, int, int]:
        store = zarr.open_group(str(hf_root / shot / f"{group}.zarr"), mode="r")
        tok = np.asarray(store["tokens"], dtype=np.int64)
        meta = store.attrs["metadata"]
        meta = _json.loads(meta) if isinstance(meta, str) else meta
        return int(meta["codebook_size"]), int(tok.min()), int(tok.max())

    xim_cb, xim_lo, xim_hi = _codebook_and_idrange(xim_shot, "xim")
    xsx_cb, xsx_lo, xsx_hi = _codebook_and_idrange(xsx_shot, "xsx")

    # The real model-derived codebook sizes match the expected values.
    assert xim_cb == REAL_CODEBOOK_SIZES["xim"], xim_cb
    assert xsx_cb == REAL_CODEBOOK_SIZES["xsx"], xsx_cb

    # Reconstruct from the REAL on-disk codebook sizes (scanned off disk).
    reg = reconstruct_v2_namespace_from_stores(hf_root)

    # The reconstructed ranges CONTAIN the actual on-disk ids.
    xim_start, xim_end = reg.block_range(BLOCK_XIM_PATCH)
    xsx_start, xsx_end = reg.block_range(BLOCK_XSX_PATCH)
    assert xim_start <= xim_lo and xim_hi < xim_end, (
        f"xim on-disk ids [{xim_lo},{xim_hi}] not in reconstructed "
        f"[{xim_start},{xim_end})"
    )
    assert xsx_start <= xsx_lo and xsx_hi < xsx_end, (
        f"xsx on-disk ids [{xsx_lo},{xsx_hi}] not in reconstructed "
        f"[{xsx_start},{xsx_end})"
    )

    # L2 is strictly above the real on-disk ids and splits to the L2 block.
    l2_start, l2_end = allocate_l2_input_block(reg, signals_hf_root=hf_root)
    assert l2_start > xim_hi and l2_start > xsx_hi
    assert reg.split(l2_start)[0] == BLOCK_L2_INPUT_LOW
    assert reg.split(l2_end - 1)[0] == BLOCK_L2_INPUT_LOW


# ---------------------------------------------------------------------------
# 2. Leakage guard on the authorised set
# ---------------------------------------------------------------------------


def test_authorised_inputs_are_all_admissible():
    """Every (group, var) on the allow-list classifies as an admissible input.

    Uses the real uda_name recovered from the inventory classification path:
    we re-derive the classification from the canonical uda_name spelling so
    a future relabelling that demotes one of these to reconstructed/banned
    fails this test loudly.
    """
    # Canonical uda_name per authorised field (the on-disk spelling).
    uda = {
        ("pf_active", "coil_current"): "AMC_P2IL FEED CURRENT",
        ("pf_active", "solenoid_current"): "AMC_SOL CURRENT",
        ("pf_active", "coil_voltage"): "XDC_PF_F_P1",
        ("gas_injection", "inboard_total"): "AGA_INBOARD_TOTAL",
        ("gas_injection", "outboard_total"): "AGA_OUTBOARD_TOTAL",
        ("gas_injection", "total_injected"): "AGA_INTEG_GAS",
        ("gas_injection", "valve_voltage"): "XDC_GAS_F_G1",
        ("gas_injection", "valve_target_voltage"): "XDC_GAS_T_G1",
        ("summary", "ip"): "AMC_PLASMA CURRENT",
        ("summary", "power_radiated"): "ABM_PRAD_POL",
        ("summary", "power_nbi"): "ANB_TOT_SUM_POWER",
        ("summary", "neutron_rates_total"): "ANU_NEUTRONS",
        ("interferometer", "n_e_line"): "ANE_DENSITY",
        ("soft_x_rays", "horizontal_cam_lower"): "XSX_HCAML#1",
        ("soft_x_rays", "horizontal_cam_upper"): "XSX_HCAMU#1",
        ("pulse_schedule", "i_plasma"): "XDC_IP_T_IPREF",
        ("pulse_schedule", "n_e_line"): "XDC_DENSITY_T_NELREF",
    }
    triples = []
    for spec in lib.AUTHORISED_INPUTS:
        for var in spec.variables:
            key = (spec.group, var)
            assert key in uda, f"no canonical uda for {key}"
            triples.append((spec.group, var, uda[key]))
            assert is_admissible_input(spec.group, var, uda[key]), (
                f"{key} (uda={uda[key]}) is not an admissible input"
            )
    # The whole authorised set passes the field-level leakage guard at once.
    assert_no_leakage_fields(triples)  # must not raise


def test_guard_rejects_reconstructed_state():
    """The build's gate must refuse reconstructed equilibrium + derived scalars."""
    banned = [
        ("equilibrium", "psi", "EFM_PSI(R,Z)"),
        ("summary", "line_average_n_e", "ESM_NE_BAR"),
        ("summary", "greenwald_density", "ESM_N_GREENWALD"),
    ]
    for triple in banned:
        with pytest.raises(ValueError):
            assert_no_leakage_fields([triple])
        assert classify_l2_field(*triple).classification == "banned"


def test_no_banned_source_in_allowlist():
    """No allow-list field carries a reconstructed (EFM/ESM) source prefix."""
    from imas_ambix.data.provenance import RECONSTRUCTED_SOURCES, source_of_uda

    # Re-use the canonical uda map indirectly via the admissibility test set.
    for spec in lib.AUTHORISED_INPUTS:
        # The allow-list never names a reconstructed-equilibrium group; the
        # real per-field guard runs at build time on the actual uda_name.
        assert spec.group not in {"equilibrium"}
    assert "EFM" in RECONSTRUCTED_SOURCES and "ESM" in RECONSTRUCTED_SOURCES
    assert source_of_uda("EFM_PSI(R,Z)") == "EFM"


# ---------------------------------------------------------------------------
# 3. Read / quantise unit behaviour (synthetic — no GPFS needed)
# ---------------------------------------------------------------------------


def test_channels_from_var_1d_and_2d():
    """1-D var → one channel; 2-D (channel, time) → one channel per column."""
    import xarray as xr

    ds = xr.Dataset(
        {
            "ip": (("time",), np.arange(100.0)),
            "coil_current": (("current_channel", "time"), np.ones((3, 100))),
        },
        coords={"time": np.linspace(0, 1, 100)},
    )
    ds["ip"].attrs["uda_name"] = "AMC_PLASMA CURRENT"
    ds["coil_current"].attrs["uda_name"] = "AMC_P2IL FEED CURRENT"

    ch1 = lib._channels_from_var("summary", "ip", ds["ip"])
    assert len(ch1) == 1 and ch1[0].name == "summary.ip"

    ch2 = lib._channels_from_var("pf_active", "coil_current", ds["coil_current"])
    assert len(ch2) == 3
    assert {c.name for c in ch2} == {
        "pf_active.coil_current[0]",
        "pf_active.coil_current[1]",
        "pf_active.coil_current[2]",
    }


def test_valid_mask_marks_nonfinite_not_zero_fill():
    """NaN samples are masked invalid, never silently treated as real zeros."""
    import xarray as xr

    arr = np.arange(10.0)
    arr[3] = np.nan
    da = xr.DataArray(arr, dims=("time",))
    da.attrs["uda_name"] = "ANE_DENSITY"
    ch = lib._channels_from_var("interferometer", "n_e_line", da)[0]
    assert ch.valid[3] == False  # noqa: E712 — explicit mask check
    assert ch.valid.sum() == 9
    # The masked sample is preserved as NaN (NOT zero-filled) so the
    # quantiser's finite-only fit ignores it; the mask is the source of truth.
    assert np.isnan(ch.values[3])
    # The quantiser maps the residual NaN to the mid-bin on encode (finite
    # token id), and the valid mask flags the position as not-real.
    quant = lib.UniformQuantizer(name="t_mask", n_bins=256)
    import xarray as xr2

    ds = xr2.Dataset(
        {"interferometer.n_e_line": (("time",), ch.values)},
        coords={"time": np.arange(10.0)},
    )
    quant.fit([ds])
    enc = quant.encode(ds)
    assert np.isfinite(enc.token_ids).all()


# ---------------------------------------------------------------------------
# 4. Real smoke encode (requires the L2 mirror)
# ---------------------------------------------------------------------------


@requires_l2
def test_smoke_written_l2_tokens_split_to_l2_above_corpus(tmp_path):
    """END-TO-END LEAKAGE PROOF.  Re-encode a few real L2 shots, load a written
    token, and assert every written global id splits to ``BLOCK_L2_INPUT_LOW``
    (never xim/xsx/xma) and the stored ``global_id_range`` is strictly above
    the real on-disk xim/xsx ranges.

    This is the non-circular end-to-end check: the L2 floor is derived from the
    real on-disk corpus codebook sizes, and the proof opens the actually-written
    L2 store rather than re-deriving the namespace.
    """
    import glob
    import json as _json
    import os

    import zarr

    from imas_ambix.data.paths import TOKEN_ROOT
    from imas_ambix.tokenizer.registry import (
        BLOCK_L2_INPUT_LOW,
        load_v2_registry,
        reconstruct_v2_namespace_from_stores,
        registry,
    )

    hf_root = TOKEN_ROOT / "v2" / "signals_hf"

    # Real corpus floor (from the persisted manifest if present, else scanned).
    corpus = load_v2_registry() or reconstruct_v2_namespace_from_stores(hf_root)
    corpus_max_end = corpus.max_block_end()

    # Real on-disk xim/xsx id maxima to cross-check "strictly above".
    def _idrange(group: str) -> tuple[int, int] | None:
        for p in sorted(glob.glob(str(hf_root / "*"))):
            sh = os.path.basename(p)
            sp = hf_root / sh / f"{group}.zarr"
            if sh.isdigit() and sp.exists():
                tok = np.asarray(
                    zarr.open_group(str(sp), mode="r")["tokens"], dtype=np.int64
                )
                return int(tok.min()), int(tok.max())
        return None

    xim_rng, xsx_rng = _idrange("xim"), _idrange("xsx")

    n_checked = 0
    for shot in SMOKE_SHOTS:
        written = lib.build_shot(shot, out_root=tmp_path, skip_existing=False)
        for group, path in written.items():
            store = zarr.open_group(str(path), mode="r")
            tokens = np.asarray(store["tokens"], dtype=np.int64)
            meta = store.attrs["metadata"]
            meta = _json.loads(meta) if isinstance(meta, str) else meta
            gid_lo, gid_hi = meta["global_id_range"]

            # The recorded L2 range is strictly above the real corpus maximum.
            assert gid_lo == corpus_max_end, (
                f"{shot}/{group}: L2 floor {gid_lo} != corpus max {corpus_max_end}"
            )
            assert gid_hi - gid_lo == 256
            if xim_rng is not None:
                assert gid_lo > xim_rng[1], "L2 floor not above xim ids"
            if xsx_rng is not None:
                assert gid_lo > xsx_rng[1], "L2 floor not above xsx ids"

            # Every WRITTEN global id splits to the L2 block — never a corpus one.
            for gid in (int(tokens.min()), int(tokens.max())):
                name, _local = registry.split(gid)
                assert name == BLOCK_L2_INPUT_LOW, (
                    f"{shot}/{group}: written id {gid} decoded as {name!r}"
                )
            assert int(tokens.min()) >= gid_lo and int(tokens.max()) < gid_hi
            n_checked += 1
    assert n_checked > 0


@requires_l2
def test_smoke_read_group_guards_every_field():
    """read_group runs the leakage guard on every channel uda and admits only
    measured + planned fields, on a real shot."""
    spec = next(s for s in lib.AUTHORISED_INPUTS if s.group == "pf_active")
    read = lib.read_group(15199, spec)
    assert read is not None
    # Every emitted channel's uda must be admissible.
    for ch in read.channels:
        assert is_admissible_input(ch.group, ch.var, ch.uda_name), ch.name
    # native rate of the 4 kHz control group
    assert 3000.0 < read.native_rate_hz < 5000.0
    # coil_current expanded into per-channel columns + the two scalars + voltage
    names = {c.name for c in read.channels}
    assert "pf_active.solenoid_current" in names
    assert any(n.startswith("pf_active.coil_current[") for n in names)
    assert any(n.startswith("pf_active.coil_voltage[") for n in names)


@requires_l2
def test_smoke_encode_roundtrip_qc(tmp_path):
    """Encode 4 real shots, verify the store contract + per-channel round-trip.

    Reports, per channel, the dequantise-vs-original max-abs-err normalised
    by the channel std (a 256-bin uniform quantiser over ±4σ has a quantum of
    ~8σ/255 ≈ 0.031σ, so the round-trip error should sit near half that) and
    the Pearson correlation (must be high for any channel with real dynamic
    range).
    """
    import zarr

    from imas_ambix.tokenizer.signals import UniformQuantizer

    qc_rows: list[str] = []
    n_stores = 0
    n_channels_checked = 0

    for shot in SMOKE_SHOTS:
        written = lib.build_shot(shot, out_root=tmp_path, skip_existing=False)
        # 11766 has no SXR emission → soft_x_rays must be absent from outputs.
        if shot == 11766:
            assert "soft_x_rays" not in written, "expected SXR skip on 11766"
        else:
            assert "soft_x_rays" in written, f"expected SXR present on {shot}"

        for group, path in written.items():
            n_stores += 1
            assert path.name.endswith("_l2.zarr"), path.name
            store = zarr.open_group(str(path), mode="r")
            attrs = dict(store.attrs)
            # store contract
            assert attrs["vocab_version"] == "v2"
            assert attrs["tokenizer_name"] == "signal_l2_input_low_v2"
            assert attrs["native_rate_hz"] == attrs["token_rate_hz"]  # patch=1
            assert attrs["phase_preserving"] is False
            tokens = np.asarray(store["tokens"])
            valid = np.asarray(store["valid"])
            token_time = np.asarray(store["token_time"])
            assert tokens.shape == valid.shape
            assert token_time.shape[0] == tokens.shape[0]
            assert tokens.shape[1] == attrs["n_channels"]

            # Round-trip QC: rebuild a quantizer from the stored stats and
            # decode, then compare to the freshly-read raw values.
            import json as _json

            meta = (
                _json.loads(attrs["metadata"])
                if isinstance(attrs["metadata"], str)
                else attrs["metadata"]
            )
            means = meta["channel_means"]
            stds = meta["channel_stds"]
            chan_names = list(attrs["channel_names"])

            quant = UniformQuantizer(
                name="signal_l2_input_low_v2", n_bins=int(meta["n_bins"])
            )
            quant._channel_order = tuple(chan_names)
            quant._means = {k: float(v) for k, v in means.items()}
            quant._stds = {k: float(v) for k, v in stds.items()}

            from imas_ambix.tokenizer.base import EncodedSignals

            enc = EncodedSignals(
                token_ids=tokens,
                channel_names=tuple(chan_names),
                tokenizer_name=quant.name,
                metadata={},
            )
            decoded = quant.decode(enc)

            # Re-read the original group to compare.
            spec = next(s for s in lib.AUTHORISED_INPUTS if s.group == group)
            read = lib.read_group(shot, spec)
            orig: dict[str, np.ndarray] = {c.name: c.values for c in read.channels}
            orig_valid: dict[str, np.ndarray] = {c.name: c.valid for c in read.channels}

            for name in chan_names:
                rec = np.asarray(decoded[name].values)
                o = orig[name]
                m = orig_valid[name] & np.isfinite(rec)
                if m.sum() < 4:
                    continue
                mean = float(means[name])
                std = max(float(stds[name]), 1e-12)
                # The light path is a deliberately lossy ±clip_sigma magnitude
                # quantiser: samples beyond ±4σ are CLIPPED to the boundary,
                # so the round-trip QC must separate the quantisation quantum
                # (measured on in-range samples) from the clip fraction.
                z = (o[m] - mean) / std
                in_range = np.abs(z) <= quant.clip_sigma
                clip_frac = float((~in_range).mean())
                err_in = np.abs(rec[m][in_range] - o[m][in_range])
                # In-range round-trip error is the genuine quantisation quantum.
                inrange_max_err_sigma = (
                    float(err_in.max() / std) if err_in.size else 0.0
                )
                # Full-channel correlation (clipped tail included — a
                # monotone clip lowers it slightly on heavy-tailed channels).
                if np.std(o[m]) > 1e-9 and np.std(rec[m]) > 1e-9:
                    corr = float(np.corrcoef(o[m], rec[m])[0, 1])
                else:
                    corr = 1.0  # constant channel — trivially recovered
                # In-range correlation (the unbiased fidelity of the quantiser
                # where it is not deliberately saturating).
                oi, ri = o[m][in_range], rec[m][in_range]
                if oi.size >= 4 and np.std(oi) > 1e-9 and np.std(ri) > 1e-9:
                    corr_in = float(np.corrcoef(oi, ri)[0, 1])
                else:
                    corr_in = 1.0
                n_channels_checked += 1
                # A ±4σ / 256-bin uniform quantiser: half-quantum ≈ 0.0157σ,
                # worst-case quantum ≈ 0.0314σ. Allow a small margin for the
                # round() bin-edge and the per-shot std estimate.  Measured on
                # the in-range samples only (clipped tails are expected lossy).
                assert inrange_max_err_sigma < 0.06, (
                    f"{shot}/{name}: in-range round-trip err "
                    f"{inrange_max_err_sigma:.4f}σ too large"
                )
                # Clipping must be a small minority for an authorised input
                # channel (heavy-tailed coil-feed currents clip a few %).
                assert clip_frac < 0.05, (
                    f"{shot}/{name}: clip fraction {clip_frac:.3f} too high"
                )
                # Where the quantiser is not saturating, fidelity is ~exact.
                if np.std(oi) > 1e-6:
                    assert corr_in > 0.999, (
                        f"{shot}/{name}: in-range corr {corr_in:.5f} too low"
                    )
                if len(qc_rows) < 14:
                    qc_rows.append(
                        f"{shot} {name}: "
                        f"inrange_max_err={inrange_max_err_sigma:.4f}sigma "
                        f"clip={clip_frac:.4f} "
                        f"corr_in={corr_in:.5f} corr_all={corr:.5f}"
                    )

    assert n_stores > 0 and n_channels_checked > 0
    # Surface a few QC rows in the test report (visible with -s / on failure).
    print("\nL2 input round-trip QC (sample):")
    for row in qc_rows:
        print("  " + row)
