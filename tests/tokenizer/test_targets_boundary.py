"""Airtight-boundary tests for the eval-only TARGET store.

Three walls keep the world-model PREDICTION targets (the L2 equilibrium
reconstruction + reconstruction-derived globals) out of the input token
stream.  These tests prove the walls rather than assert them in prose:

* Wall 1 — ``TARGET_ROOT`` is not under ``TOKEN_ROOT``, so no input glob
  can structurally reach it.
* Wall 2 — :class:`TargetV2Attrs` has NO ``tokenizer_name`` / ``vocab_version``
  field, on the dataclass and on the on-disk attrs it writes.
* Wall 3 — the REAL input-group enumerator opens only
  ``TOKEN_ROOT/v2/{signals_hf,frames}`` and REFUSES ``TARGET_ROOT``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from imas_ambix.data.paths import TARGET_ROOT, TOKEN_ROOT
from imas_ambix.tokenizer.store_targets import (
    FORBIDDEN_TARGET_ATTRS,
    REQUIRED_TARGET_ATTRS,
    TargetV2Attrs,
    assert_not_target_path,
    enumerate_input_group_paths,
    input_group_roots,
    load_target_group,
    save_target_group,
)


def _make_attrs(**over) -> TargetV2Attrs:
    base = dict(
        quantity_names=("psi", "li"),
        units=("Wb / rad", ""),
        grid_r=(0.06, 0.5, 1.98),
        grid_z=(-2.0, 0.0, 2.0),
        time=(0.0, 0.005, 0.01),
        original_window=(0.0, 0.01),
    )
    base.update(over)
    return TargetV2Attrs(**base)


# ---------------------------------------------------------------------------
# Wall 1 — separate root
# ---------------------------------------------------------------------------


def test_target_root_not_under_token_root():
    """TARGET_ROOT must not be a child of TOKEN_ROOT (no input glob reaches it)."""
    assert TARGET_ROOT != TOKEN_ROOT
    assert TOKEN_ROOT.resolve() not in TARGET_ROOT.resolve().parents


# ---------------------------------------------------------------------------
# Wall 2 — no token-vocabulary handle on the attrs
# ---------------------------------------------------------------------------


def test_target_attrs_has_no_tokenizer_name_field():
    """The dataclass deliberately omits tokenizer_name (no input-vocab handle)."""
    field_names = {f.name for f in dataclasses.fields(TargetV2Attrs)}
    assert "tokenizer_name" not in field_names


def test_target_attrs_has_no_vocab_version_field():
    """The dataclass deliberately omits vocab_version (no registry shift)."""
    field_names = {f.name for f in dataclasses.fields(TargetV2Attrs)}
    assert "vocab_version" not in field_names


def test_target_attrs_instance_lacks_vocab_handles():
    """An instance carries neither forbidden attribute."""
    attrs = _make_attrs()
    for forbidden in FORBIDDEN_TARGET_ATTRS:
        assert not hasattr(attrs, forbidden)


def test_target_on_disk_attrs_omit_vocab_handles():
    """The serialised .attrs dict never carries a token-vocab key."""
    out = _make_attrs().to_attrs()
    for forbidden in FORBIDDEN_TARGET_ATTRS:
        assert forbidden not in out
    # but the required target contract is all present
    for required in REQUIRED_TARGET_ATTRS:
        assert required in out


def test_target_attrs_reject_smuggled_vocab_in_metadata():
    """A caller cannot smuggle a vocab handle in via metadata."""
    with pytest.raises(ValueError, match="vocab_version"):
        _make_attrs(metadata={"vocab_version": "v2"})


def test_from_attrs_rejects_store_with_vocab_key():
    """Reading a store that somehow carries a vocab key fails loudly."""
    bad = _make_attrs().to_attrs()
    bad["tokenizer_name"] = "sneaky"
    with pytest.raises(ValueError, match="forbidden token-vocab"):
        TargetV2Attrs.from_attrs(bad)


def test_from_attrs_requires_full_contract():
    """A store missing a required key is rejected, not silently defaulted."""
    incomplete = _make_attrs().to_attrs()
    del incomplete["time"]
    with pytest.raises(ValueError, match="missing required keys"):
        TargetV2Attrs.from_attrs(incomplete)


# ---------------------------------------------------------------------------
# Wall 3 — the REAL input-group enumerator opens only the input roots
# ---------------------------------------------------------------------------


def test_input_group_roots_are_exactly_signals_hf_and_frames():
    """The enumerator lists only TOKEN_ROOT/v2/{signals_hf,frames}."""
    roots = input_group_roots()
    assert roots == [
        TOKEN_ROOT / "v2" / "signals_hf",
        TOKEN_ROOT / "v2" / "frames",
    ]
    # and none of them is the target root
    for r in roots:
        assert TARGET_ROOT.resolve() not in r.resolve().parents
        assert r.resolve() != TARGET_ROOT.resolve()


def test_input_roots_all_under_token_root():
    """Every input group root is under TOKEN_ROOT — never TARGET_ROOT."""
    for r in input_group_roots():
        assert TOKEN_ROOT.resolve() in r.resolve().parents


def test_assert_not_target_path_refuses_target_root(tmp_path):
    """A path under TARGET_ROOT is hard-refused by the boundary guard."""
    fake_target = tmp_path / "mast-targets"
    leaked = fake_target / "13277" / "equilibrium.zarr"
    with pytest.raises(ValueError, match="TARGET_ROOT"):
        assert_not_target_path(leaked, target_root=fake_target)


def test_assert_not_target_path_passes_input_path(tmp_path):
    """A path under an input root passes the boundary guard unchanged."""
    input_store = tmp_path / "mast-tokens" / "v2" / "signals_hf" / "13277" / "xma.zarr"
    out = assert_not_target_path(input_store, target_root=tmp_path / "mast-targets")
    assert out == input_store


def test_enumerator_refuses_target_root_even_if_symlinked(tmp_path):
    """The REAL enumerator runs the guard on every path it yields.

    Build an isolated token root with one signals_hf store and one frames
    store, plus a separate target store; confirm the enumerator yields the
    two input stores and never the target.
    """
    token_root = tmp_path / "mast-tokens"
    target_root = tmp_path / "mast-targets"

    # two legitimate input stores
    sig = token_root / "v2" / "signals_hf" / "13277" / "xma.zarr"
    frm = token_root / "v2" / "frames" / "13277" / "rbb.zarr"
    for d in (sig, frm):
        d.mkdir(parents=True)
        (d / ".marker").write_text("x")

    # a target store under the separate root — must NOT be enumerated
    tgt = target_root / "13277" / "equilibrium.zarr"
    tgt.mkdir(parents=True)
    (tgt / ".marker").write_text("x")

    found = enumerate_input_group_paths(token_root=token_root, target_root=target_root)
    found_set = {p.resolve() for p in found}
    assert sig.resolve() in found_set
    assert frm.resolve() in found_set
    assert tgt.resolve() not in found_set
    # nothing yielded resolves under the target root
    for p in found:
        assert target_root.resolve() not in p.resolve().parents


def test_enumerator_hard_refuses_a_target_symlinked_under_input_root(tmp_path):
    """If a target store is symlinked under an input root, the guard fires."""
    token_root = tmp_path / "mast-tokens"
    target_root = tmp_path / "mast-targets"
    real_target = target_root / "13277" / "equilibrium.zarr"
    real_target.mkdir(parents=True)

    shot_dir = token_root / "v2" / "signals_hf" / "13277"
    shot_dir.mkdir(parents=True)
    # symlink the target store in under the input root
    link = shot_dir / "leaked.zarr"
    link.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(ValueError, match="TARGET_ROOT"):
        enumerate_input_group_paths(token_root=token_root, target_root=target_root)


# ---------------------------------------------------------------------------
# Writer/reader round-trip with finite masks + NaN-outside-window
# ---------------------------------------------------------------------------


def test_writer_round_trip_with_finite_mask(tmp_path):
    """Raw values + per-quantity finite mask round-trip; mask marks NaNs."""
    target_root = tmp_path / "mast-targets"
    psi = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])  # (chan, time)
    li = np.array([np.nan, 0.5, 0.6])
    arrays = {"psi": psi, "li": li}
    masks = {"psi": np.isfinite(psi), "li": np.isfinite(li)}
    attrs = _make_attrs()

    save_target_group(
        13277, "equilibrium", arrays, masks, attrs, target_root=target_root
    )
    grp = load_target_group(13277, "equilibrium", target_root=target_root)

    # arrays preserved (NaN preserved, never zero-filled)
    assert np.array_equal(np.isnan(grp.arrays["psi"]), np.isnan(psi))
    # every value the mask calls valid IS finite
    assert np.isfinite(grp.arrays["psi"][grp.masks["psi"]]).all()
    assert np.isfinite(grp.arrays["li"][grp.masks["li"]]).all()
    # and the mask is False exactly where the data is NaN
    assert np.array_equal(grp.masks["psi"], np.isfinite(psi))


def test_writer_rejects_mismatched_mask_shape(tmp_path):
    """A mask whose shape disagrees with its data is rejected at write."""
    target_root = tmp_path / "mast-targets"
    arrays = {"psi": np.zeros((2, 3)), "li": np.zeros(3)}
    masks = {"psi": np.ones((2, 2), dtype=bool), "li": np.ones(3, dtype=bool)}
    with pytest.raises(ValueError, match="mask shape"):
        save_target_group(
            13277, "equilibrium", arrays, masks, _make_attrs(), target_root=target_root
        )
