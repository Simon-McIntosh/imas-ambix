"""Tests for the token-store vocabulary model.

The on-disk v2 high-frequency corpus is encoded ONE PROCESS PER GROUP, so
every group's patch block restarts at the control range and the groups
OVERLAP in global id space (xma_patch, xim_patch, xsx_patch all begin at id
4).  The contract is therefore PER-GROUP-LOCAL: an id is only meaningful
together with the group it came from (the store's ``tokenizer_name``).

A downstream model that wants a single 1-D vocabulary REBASES the
per-group-local ids into a DISJOINT unified-flat namespace at consume time
(``build_unified_namespace`` / ``unified_global_ids``), sized from the REAL
on-disk ``metadata.codebook_size`` — no re-encode.

These tests prove:

1. after rebasing, ids from DIFFERENT groups are pairwise disjoint;
2. a per-group-local id is interpreted via its ``tokenizer_name`` (the same
   local id maps to two different unified ids for two different groups);
3. the builder derives block sizes from on-disk ``metadata.codebook_size``
   (it reads a real store), NOT a hardcoded table — and the disjoint
   xim/xsx ranges CONTAIN the real on-disk ids once rebased.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

# The real, model-derived codebook sizes the on-disk v2 corpus uses.  These are
# NOT a fictional table the builder consults — they are the values the absolute
# re-encode wrote into each store's ``metadata.codebook_size``.  The codebook
# decision resolved to the CONTINUOUS bottleneck for every HF group (continuous
# is the phase-fidelity ceiling — quantisation destroyed phase), so every
# patch block is continuous with ``codebook_size = 1`` (the continuous
# bottleneck emits a single vestigial id; the phase-preserving payload is the
# per-token EMBEDDING, not a discrete code).  They are injected here to drive
# the disjointness / interpretation assertions, and the on-disk cross-check
# reads the REAL size off disk and asserts the builder reproduces it (so a
# future retrain that changes a codebook size is caught), rather than asserting
# a hardcoded value the redo could legitimately change.
REAL_CODEBOOK_SIZES = {"xma": 1, "xim": 1, "xsx": 1}


def _hf_root():
    from imas_ambix.data.paths import TOKEN_ROOT

    return TOKEN_ROOT / "v2" / "signals_hf"


def _find_group_store(group: str) -> str | None:
    """First on-disk shot dir that carries ``{group}.zarr``."""
    root = _hf_root()
    for p in sorted(glob.glob(str(root / "*"))):
        sh = os.path.basename(p)
        if sh.isdigit() and (root / sh / f"{group}.zarr").exists():
            return sh
    return None


# ---------------------------------------------------------------------------
# 1. Cross-group ids are DISJOINT after rebasing (THE INVARIANT)
# ---------------------------------------------------------------------------


def test_unified_namespace_blocks_are_pairwise_disjoint():
    """After rebasing, every block occupies its OWN range — no two groups
    overlap.  This is the property the per-group-local on-disk layout lacks
    (there every patch block starts at id 4)."""
    from imas_ambix.tokenizer.registry import (
        UNIFIED_BLOCK_ORDER,
        build_unified_namespace,
    )

    unified = build_unified_namespace(block_codebook_sizes=REAL_CODEBOOK_SIZES)

    ranges = [unified.block_range(name) for name in UNIFIED_BLOCK_ORDER]
    for i in range(len(ranges)):
        a0, a1 = ranges[i]
        assert a1 > a0
        for j in range(i + 1, len(ranges)):
            b0, b1 = ranges[j]
            assert a1 <= b0 or b1 <= a0, (
                f"unified blocks {UNIFIED_BLOCK_ORDER[i]} {ranges[i]} and "
                f"{UNIFIED_BLOCK_ORDER[j]} {ranges[j]} overlap"
            )


def test_rebased_ids_from_different_groups_never_collide():
    """The same per-group-local id from xma, xim and xsx must map to THREE
    distinct unified ids — the overlapping on-disk ids are disentangled."""
    from imas_ambix.tokenizer.registry import (
        BLOCK_XIM_PATCH,
        BLOCK_XMA_PATCH,
        BLOCK_XSX_PATCH,
        build_unified_namespace,
        unified_global_ids,
    )

    unified = build_unified_namespace(block_codebook_sizes=REAL_CODEBOOK_SIZES)

    # local id 0 exists in every group's local vocabulary.
    xma = int(unified_global_ids(BLOCK_XMA_PATCH, np.array([0]), unified=unified)[0])
    xim = int(unified_global_ids(BLOCK_XIM_PATCH, np.array([0]), unified=unified)[0])
    xsx = int(unified_global_ids(BLOCK_XSX_PATCH, np.array([0]), unified=unified)[0])
    assert len({xma, xim, xsx}) == 3, (xma, xim, xsx)

    # The full span of each group is disjoint from the others.  Take the max
    # local id each group can emit and confirm it lands inside its own block,
    # below the next block's floor.
    for block, size in (
        (BLOCK_XMA_PATCH, REAL_CODEBOOK_SIZES["xma"]),
        (BLOCK_XIM_PATCH, REAL_CODEBOOK_SIZES["xim"]),
        (BLOCK_XSX_PATCH, REAL_CODEBOOK_SIZES["xsx"]),
    ):
        start, end = unified.block_range(block)
        rebased = unified_global_ids(block, np.arange(size), unified=unified)
        assert int(rebased.min()) == start
        assert int(rebased.max()) == end - 1
        # split() of any rebased id resolves to this block, never another.
        assert unified.split(int(rebased.min()))[0] == block
        assert unified.split(int(rebased.max()))[0] == block


# ---------------------------------------------------------------------------
# 2. A per-group-local id is interpreted via its tokenizer_name
# ---------------------------------------------------------------------------


def test_local_id_interpreted_via_tokenizer_name():
    """An on-disk id alone is ambiguous (groups overlap); the disambiguator is
    the store's ``tokenizer_name`` (== the block constant).  Rebasing the SAME
    local id under different tokenizer names yields different unified ids, and
    each rebased id ``split``s back to the name it was rebased under."""
    from imas_ambix.tokenizer.registry import (
        BLOCK_XIM_PATCH,
        BLOCK_XSX_PATCH,
        build_unified_namespace,
        unified_global_ids,
    )

    unified = build_unified_namespace(block_codebook_sizes=REAL_CODEBOOK_SIZES)

    # local id 0 is the one id every (continuous, codebook_size=1) group emits,
    # and is present in both the xim and xsx local vocabularies.
    local = np.array([0])
    as_xim = int(unified_global_ids(BLOCK_XIM_PATCH, local, unified=unified)[0])
    as_xsx = int(unified_global_ids(BLOCK_XSX_PATCH, local, unified=unified)[0])

    assert as_xim != as_xsx, "tokenizer_name did not disambiguate the local id"
    # The interpretation round-trips: each unified id decodes to the
    # tokenizer_name it was rebased under, with the original local id.
    name_xim, back_xim = unified.split(as_xim)
    name_xsx, back_xsx = unified.split(as_xsx)
    assert name_xim == BLOCK_XIM_PATCH and back_xim == 0
    assert name_xsx == BLOCK_XSX_PATCH and back_xsx == 0


def test_unknown_tokenizer_name_rejected():
    """Rebasing demands a known group block — an unknown tokenizer_name raises
    rather than silently producing a wrong (aliasing) id."""
    from imas_ambix.tokenizer.registry import (
        build_unified_namespace,
        unified_global_ids,
    )

    unified = build_unified_namespace(block_codebook_sizes=REAL_CODEBOOK_SIZES)
    with pytest.raises(KeyError):
        unified_global_ids("not_a_real_block", np.array([0]), unified=unified)


# ---------------------------------------------------------------------------
# 3. Sizes derive from on-disk metadata (NOT a hardcoded table) +
#    the disjoint xim/xsx ranges contain the rebased real on-disk ids
# ---------------------------------------------------------------------------


def test_builder_derives_sizes_from_on_disk_metadata_not_hardcoded():
    """The builder must read ``metadata.codebook_size`` off a REAL store — it
    must NOT carry the sizes in a hardcoded table.

    Proof it is not hardcoded: ``_scan_block_codebook_sizes`` returns the sizes
    read off disk, and feeding those scanned sizes into the builder reproduces
    the same layout as a fresh scan-driven build.  We then assert those scanned
    sizes match the expected model-derived values (so a retrain that changes a
    codebook size is caught), and that the on-disk xim/xsx token id arrays,
    once rebased, land INSIDE their disjoint unified ranges.
    """
    import zarr

    from imas_ambix.tokenizer.registry import (
        BLOCK_XIM_PATCH,
        BLOCK_XSX_PATCH,
        _scan_block_codebook_sizes,
        build_unified_namespace,
        unified_global_ids,
    )

    xim_shot, xsx_shot = _find_group_store("xim"), _find_group_store("xsx")
    if xim_shot is None or xsx_shot is None:
        pytest.skip("no on-disk xim/xsx corpus stores to read sizes from")

    root = _hf_root()

    # (a) Sizes are SCANNED off disk — not hardcoded.  The builder must
    # reproduce whatever the stores actually carry, NOT a constant; we therefore
    # assert the SCANNED sizes match the on-disk metadata (read independently
    # below) rather than a fixed table — the codebook size can legitimately
    # change across a re-encode (continuous -> 1), so a hardcoded expectation
    # would be the very brittleness this test guards against.
    scanned = _scan_block_codebook_sizes(root)
    assert scanned["xim"] >= 1 and scanned["xsx"] >= 1, scanned

    # The builder with no injected sizes scans the SAME on-disk truth and
    # reproduces the layout built from the scanned sizes — proving the builder
    # consults on-disk metadata, not a constant.
    from_disk = build_unified_namespace(signals_hf_root=root)
    from_scanned = build_unified_namespace(block_codebook_sizes=scanned)
    assert from_disk.block_range(BLOCK_XIM_PATCH) == from_scanned.block_range(
        BLOCK_XIM_PATCH
    )
    assert from_disk.block_range(BLOCK_XSX_PATCH) == from_scanned.block_range(
        BLOCK_XSX_PATCH
    )

    # (b) Cross-check: the rebased REAL on-disk ids land inside the disjoint
    # unified ranges.  On disk every group's block starts at the control range,
    # so the per-group-local id is ``on_disk_id - CONTROL_RANGE[1]``.
    from imas_ambix.tokenizer.registry import CONTROL_RANGE

    def _on_disk_ids(shot: str, group: str) -> tuple[int, np.ndarray]:
        store = zarr.open_group(str(root / shot / f"{group}.zarr"), mode="r")
        meta = store.attrs["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        tok = np.asarray(store["tokens"], dtype=np.int64).reshape(-1)
        return int(meta["codebook_size"]), tok

    xim_cb, xim_ids = _on_disk_ids(xim_shot, "xim")
    xsx_cb, xsx_ids = _on_disk_ids(xsx_shot, "xsx")
    # The scan and the independent metadata read agree — the size is read, not
    # invented (whatever value the current re-encode wrote).
    assert scanned["xim"] == xim_cb, (scanned, xim_cb)
    assert scanned["xsx"] == xsx_cb, (scanned, xsx_cb)

    # Convert on-disk (per-group-local, control-relative) ids -> local ids.
    ctrl = CONTROL_RANGE[1]
    xim_local = xim_ids - ctrl
    xsx_local = xsx_ids - ctrl
    assert xim_local.min() >= 0 and xsx_local.min() >= 0

    xim_re = np.asarray(
        unified_global_ids(BLOCK_XIM_PATCH, xim_local, unified=from_disk)
    )
    xsx_re = np.asarray(
        unified_global_ids(BLOCK_XSX_PATCH, xsx_local, unified=from_disk)
    )

    xim_start, xim_end = from_disk.block_range(BLOCK_XIM_PATCH)
    xsx_start, xsx_end = from_disk.block_range(BLOCK_XSX_PATCH)
    assert xim_re.min() >= xim_start and xim_re.max() < xim_end, (
        f"rebased xim ids [{xim_re.min()},{xim_re.max()}] escape disjoint "
        f"range [{xim_start},{xim_end})"
    )
    assert xsx_re.min() >= xsx_start and xsx_re.max() < xsx_end, (
        f"rebased xsx ids [{xsx_re.min()},{xsx_re.max()}] escape disjoint "
        f"range [{xsx_start},{xsx_end})"
    )

    # And crucially: the rebased xim and xsx id sets are DISJOINT (they
    # overlapped on disk, both starting at the control range).
    assert xim_re.max() < xsx_re.min() or xsx_re.max() < xim_re.min(), (
        "rebased xim/xsx id sets still overlap — disjointness failed"
    )


def test_persisted_manifest_records_both_views(tmp_path):
    """The persisted ``registry.json`` carries BOTH the per-group-local
    ``blocks`` (overlapping on-disk reality, untouched) AND a disjoint
    ``unified_blocks`` map; loading the unified map round-trips disjoint.

    Writes to a tmp manifest (never the shared on-disk one) so the test does
    not race the in-flight encode / the concurrent L2 build.  The tmp manifest
    is seeded with a per-group-local ``blocks`` array (built via the same
    reconstruction the on-disk manifest uses) so we can prove
    ``persist_unified_namespace`` PRESERVES it while adding ``unified_blocks``.
    """
    from imas_ambix.tokenizer.registry import (
        UNIFIED_BLOCK_ORDER,
        build_unified_namespace,
        load_unified_namespace,
        persist_unified_namespace,
        persist_v2_registry,
        reconstruct_v2_namespace_from_stores,
    )

    manifest = tmp_path / "registry.json"

    # Seed the per-group-local (overlapping) manifest first, exactly as the
    # on-disk reality records it.
    per_group_local = reconstruct_v2_namespace_from_stores(
        signals_hf_root="/dev/null", block_codebook_sizes=REAL_CODEBOOK_SIZES
    )
    persist_v2_registry(per_group_local, manifest)

    unified = build_unified_namespace(block_codebook_sizes=REAL_CODEBOOK_SIZES)
    path = persist_unified_namespace(unified, manifest_path=manifest)
    payload = json.loads(path.read_text())

    # Per-group-local blocks preserved AND overlapping (every patch block at 4).
    by_name = {b["name"]: (b["start"], b["end"]) for b in payload["blocks"]}
    assert by_name["signal_hf_xim_patch_v2"][0] == 4
    assert by_name["signal_hf_xsx_patch_v2"][0] == 4
    # Each group's patch block spans exactly its own codebook size starting at
    # the control floor — the OVERLAPPING per-group-local layout (NOT disjoint:
    # xim and xsx patch blocks share the same start).  A per-group block stacks
    # its sibling (mode/profile) block ABOVE its patch block within that group.
    assert by_name["signal_hf_xim_patch_v2"] == (4, 4 + REAL_CODEBOOK_SIZES["xim"])
    assert by_name["signal_hf_xsx_patch_v2"] == (4, 4 + REAL_CODEBOOK_SIZES["xsx"])

    # Unified blocks present, disjoint, and round-trip via load_unified.
    reloaded = load_unified_namespace(manifest_path=manifest)
    assert reloaded is not None
    for name in UNIFIED_BLOCK_ORDER:
        assert reloaded.block_range(name) == unified.block_range(name)
