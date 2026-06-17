"""Token id namespace registry.

The on-disk contract is PER-GROUP-LOCAL
----------------------------------------
Every concrete tokenizer (Open-MAGVIT2, Chronos, PatchTST, …) emits
**local** ids in ``[0, vocab_size)``.  The high-frequency v2 signal corpus
(xma / xim / xsx) is encoded by :mod:`imas_ambix.tokenizer.signal_hf_encode`,
which sizes each block's codebook **at runtime** from the trained model
(``patch_vocab = bottleneck.codebook_size``) — NOT from any compile-time
constant.

Crucially the encode runs **one process per group** (see the
``for GROUP in xma xim xsx`` loop in
``scripts/slurm/signal_tokenizer.sbatch`` — each group is a separate
``encode`` invocation with its own fresh :data:`registry`).  Each process
restarts its allocation at the control range, so on disk **every group's
patch block begins at id 4 and the groups OVERLAP in global id space**::

    on disk:   xma_patch  ∈ [4, 5)        (codebook_size 1)
               xim_patch  ∈ [4, 12804)    (codebook_size 12800)
               xsx_patch  ∈ [4, 1028)     (codebook_size 1024)

The registry packs a flat contiguous range only **within a single
process** — it never produces one global stream across groups, because no
single process ever encodes more than one group.  The flat-vocabulary
picture from earlier versions of this docstring is therefore NOT what is
realised on disk.

What this means for a consumer (the CONTRACT)
---------------------------------------------
Because group ids overlap, an id is meaningless without the **group it came
from**.  A consumer disambiguates one of two ways:

(a) **per-group-local channels** — keep each group as its own token channel
    and resolve an id with the store's ``tokenizer_name`` / group (recorded
    in every ``signals_hf/{shot}/{group}.zarr`` ``.attrs``).  This matches
    the on-disk reality and needs no data rewrite.  It is the default.

(b) **unified-flat stream** — when a downstream model wants a single 1-D
    vocabulary, REBASE the per-group-local ids to a disjoint unified
    namespace at *consume* time via :func:`unified_global_ids` /
    :func:`build_unified_namespace` below.  Each known block is assigned a
    DISJOINT contiguous range in canonical order, sized from the REAL
    on-disk ``metadata.codebook_size`` — no re-encode, no data rewrite.

The single ground truth for what ids the corpus actually occupies is the
on-disk store metadata (``metadata.codebook_size``) cross-checked against
the real global ids in the token arrays — never a hand-maintained size
table.

Vocab generations are versioned through :data:`VOCAB_VERSION` (held at
``"v2"`` — the rebasing utility is append-only and does NOT re-bump it).
Persisted token arrays under ``/work/projects/imas_gpu/mast-tokens/v{N}/``
store the version they were produced under.

Two persisted manifests, one file
----------------------------------
``TOKEN_ROOT/v2/registry.json`` is self-describing and records BOTH views:

* ``blocks`` — the PER-GROUP-LOCAL on-disk reality (overlapping; every
  patch block starts at the control range).  This is what
  :func:`reconstruct_v2_namespace_from_stores`, :meth:`from_json` and
  :func:`load_v2_registry` round-trip, and the floor the Level-2 input
  light path (:mod:`imas_ambix.data.l2_input_build`) places its block
  strictly above (:func:`allocate_l2_input_block`) so a decoded L2 id can
  never alias a corpus block.
* ``unified_blocks`` — the authoritative DISJOINT unified-flat map for
  consume-time rebasing (option (b)).  Written by
  :func:`persist_unified_namespace`.

Keeping both in one file means a consumer can pick its view without
re-deriving anything, and the per-group-local L2 floor is unaffected by the
unified map's existence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

VOCAB_VERSION = "v2"
"""Token-id allocation generation.

``v1`` — frame + low-frequency signal tokenizers on the 100 Hz model grid.
``v2`` — adds the native-cadence, phase-preserving high-frequency signal
tokenizers (xma fast magnetics, xim Dα/CII) and their cross-channel
mode-number tokens.  Bumped because the new tokenizers allocate fresh id
ranges; v1 cached tokens are unaffected but live in a separate store
generation (see :mod:`imas_ambix.tokenizer.store_v2`).
"""


# Control tokens — fixed-position reserved range.
CONTROL_TOKENS = {
    "pad": 0,
    "bos": 1,
    "eos": 2,
    "sep": 3,
}
CONTROL_RANGE = (0, len(CONTROL_TOKENS))


# --- v2 high-frequency signal tokenizer block names -----------------------
#
# Stable allocation names for the native-cadence, phase-preserving
# tokenizers introduced in vocab generation v2.  Each name is the key a
# tokenizer passes to :meth:`TokenRegistry.allocate`; encoders reuse the
# constant rather than hard-coding the string so the store and the registry
# stay in lock-step.  Naming is by what the block represents (modality +
# representation), never by any plan/stage label.

# Phase-aware patch-transformer codebook for the xma fast-magnetics
# Mirnov array (per-coil channel codes).
BLOCK_XMA_PATCH = "signal_hf_xma_patch_v2"
# Cross-channel poloidal mode-number tokens derived from the xma coil array.
BLOCK_XMA_MODE = "signal_hf_xma_mode_v2"
# Phase-aware patch-transformer codebook for the xim Dα/CII channels.
BLOCK_XIM_PATCH = "signal_hf_xim_patch_v2"

# Phase-aware patch-transformer codebook for the xsx soft-X-ray (SXR)
# horizontal-camera chord array (per-chord channel codes).
BLOCK_XSX_PATCH = "signal_hf_xsx_patch_v2"
# Cross-chord soft-X-ray emission-profile latent derived from the xsx chord
# array (the radial emission-profile analog of the xma poloidal mode block).
BLOCK_XSX_PROFILE = "signal_hf_xsx_profile_v2"


# --- Level-2 input light-path block (tier: level2) -------------------------
#
# Native-cadence, magnitude-only uniform-quantiser codes for the
# provenance-verified Level-2 *inputs* — the measured observables and the
# authorised planned (pulse-schedule demand / feed-forward) waveforms.  This
# is the leakage-free input light path (see
# :mod:`imas_ambix.data.l2_input_build`); it is NOT a second high-frequency
# patch-transformer corpus (that is the xma/xim/xsx work above).
#
# APPEND-ONLY: this allocates a fresh contiguous range ABOVE every id the
# real on-disk corpus uses and does NOT re-bump VOCAB_VERSION — the
# already-written / in-flight xma/xim/xsx token ids are untouched.  Its store
# groups carry the ``_l2`` filename suffix (``{group}_l2.zarr``) so they never
# collide with the L1 ``{group}.zarr`` an in-flight encode writes.
# ``L2_BLOCK_VOCAB`` matches the UniformQuantizer default bin count (256).
BLOCK_L2_INPUT_LOW = "signal_l2_input_low_v2"
L2_BLOCK_VOCAB = 256

# Tier tag for the Level-2 input blocks (distinct from the L1 high-frequency
# tier).  Carried in the manifest so a consumer can filter the L2 light-path
# token ids from the L1 corpus.
L2_TIER = "level2"


# --- Real on-disk corpus block layout (per-group, per-process) -------------
#
# Each high-frequency group is encoded by an INDEPENDENT process (see the
# ``for GROUP in ...`` loop in ``scripts/slurm/signal_tokenizer.sbatch``), so
# every group's allocation restarts at the control range.  Within one group's
# process the allocation order is exactly the order
# :func:`signal_hf_encode.encode_shots` calls ``registry.allocate``: the patch
# block first, then the cross-channel mode/profile block (a size-1 placeholder
# whose payload is carried in metadata) if the group is a coil/chord array.
#
# The block *sizes* are NOT known here — they are model-derived
# (``bottleneck.codebook_size``) and recorded only in each store's
# ``metadata.codebook_size``.  This table records solely the per-group block
# *order* and which store group reports each block's size, so a reconstruction
# can read the real sizes off disk and lay each group's blocks out exactly as
# its encode process did.
#
# (group_store, block_name, size_source) where size_source is either
# "codebook" — read from that store group's metadata.codebook_size — or an
# int literal — the fixed placeholder size the encoder allocates directly.
_HF_GROUP_LAYOUT: tuple[tuple[str, tuple[tuple[str, str | int], ...]], ...] = (
    ("xma", ((BLOCK_XMA_PATCH, "codebook"), (BLOCK_XMA_MODE, 1))),
    ("xim", ((BLOCK_XIM_PATCH, "codebook"),)),
    ("xsx", ((BLOCK_XSX_PATCH, "codebook"), (BLOCK_XSX_PROFILE, 1))),
)


@dataclass
class _Block:
    """One tokenizer's allocation in the global vocabulary."""

    name: str
    start: int
    end: int  # exclusive

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class TokenRegistry:
    """Mutable allocator that assigns contiguous id ranges to tokenizers.

    Two ways to add a block:

    * :meth:`allocate` — append a fresh contiguous range at the cursor (the
      normal path; idempotent per name).
    * :meth:`register_block` — record a block at an EXPLICIT ``(start, end)``
      range read from on-disk truth.  Used by
      :func:`reconstruct_v2_namespace_from_stores` to model the real corpus
      layout, where independently-encoded groups overlap.  Registering a
      block advances the cursor to the maximum block end so a subsequent
      :meth:`allocate` (e.g. the L2 input block) lands strictly above every
      real id.
    """

    version: str = VOCAB_VERSION
    _blocks: dict[str, _Block] = field(default_factory=dict)
    _cursor: int = field(default=CONTROL_RANGE[1])

    def allocate(self, name: str, vocab_size: int) -> tuple[int, int]:
        """Reserve ``vocab_size`` consecutive ids for ``name``.

        Repeated calls with the same name return the existing range —
        idempotent. Repeated calls with a different vocab_size for an
        existing name raise.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if name in self._blocks:
            block = self._blocks[name]
            if block.size != vocab_size:
                raise ValueError(
                    f"{name!r} already allocated with size {block.size}, "
                    f"refusing to re-allocate with size {vocab_size}"
                )
            return (block.start, block.end)

        start = self._cursor
        end = start + vocab_size
        self._blocks[name] = _Block(name, start, end)
        self._cursor = end
        return (start, end)

    def register_block(self, name: str, start: int, end: int) -> tuple[int, int]:
        """Record a block at an EXPLICIT ``[start, end)`` global-id range.

        For reconstructing the real on-disk corpus layout where
        independently-encoded groups overlap (every group's encode process
        restarts at the control range, so ``xma_patch``, ``xim_patch`` and
        ``xsx_patch`` all begin at id 4).  ``allocate`` cannot express that —
        it is strictly contiguous — so the reconstruction uses this.

        The cursor is advanced to ``max(cursor, end)`` so the next contiguous
        :meth:`allocate` (the L2 input block) starts strictly above every
        registered block, guaranteeing no overlap with any corpus id.  An
        idempotent re-register with the same range is a no-op; a conflicting
        range raises.
        """
        if end <= start:
            raise ValueError(f"{name!r}: end {end} must exceed start {start}")
        if name in self._blocks:
            block = self._blocks[name]
            if (block.start, block.end) != (start, end):
                raise ValueError(
                    f"{name!r} already registered at [{block.start}, {block.end}), "
                    f"refusing to re-register at [{start}, {end})"
                )
            return (block.start, block.end)
        self._blocks[name] = _Block(name, start, end)
        self._cursor = max(self._cursor, end)
        return (start, end)

    def split(self, global_id: int) -> tuple[str, int]:
        """Decode a global id into ``(tokenizer_name, local_id)``.

        When blocks overlap (the real on-disk corpus groups all start at the
        control range), the first matching block in insertion order wins —
        the L2 input block is registered ABOVE every corpus block, so an L2
        id never matches a corpus block.
        """
        if 0 <= global_id < CONTROL_RANGE[1]:
            return ("control", global_id)
        for block in self._blocks.values():
            if block.start <= global_id < block.end:
                return (block.name, global_id - block.start)
        raise KeyError(f"global_id {global_id} is unallocated")

    def shift(self, name: str, local_ids: object) -> object:
        """Add the allocated offset for ``name`` to a numpy array of local ids.

        Returns the array's element type unchanged. ``local_ids`` is
        expected to be a numpy array; this function is a thin convenience
        to keep call sites readable.
        """
        if name not in self._blocks:
            raise KeyError(f"{name!r} has not been allocated yet")
        import numpy as np

        offset = self._blocks[name].start
        arr = np.asarray(local_ids, dtype=np.int64)
        return (arr + offset).astype(np.int32)

    def block_range(self, name: str) -> tuple[int, int]:
        """Return the ``(start, end)`` range of an allocated/registered block."""
        if name not in self._blocks:
            raise KeyError(f"{name!r} is not allocated")
        b = self._blocks[name]
        return (b.start, b.end)

    def max_block_end(self) -> int:
        """Highest block end (exclusive) over every registered/allocated block.

        This is the first id strictly above every block — the floor the L2
        input block must start at to be leakage-free.  Falls back to the
        control range when no block exists yet.
        """
        if not self._blocks:
            return CONTROL_RANGE[1]
        return max(b.end for b in self._blocks.values())

    def total_vocab_size(self) -> int:
        """Inclusive count of every allocated id, control tokens included."""
        return self._cursor

    def to_json(self) -> str:
        """Serialise the registry for the manifest file."""
        return json.dumps(
            {
                "version": self.version,
                "control_tokens": CONTROL_TOKENS,
                "blocks": [
                    {"name": b.name, "start": b.start, "end": b.end}
                    for b in self._blocks.values()
                ],
                "total_vocab_size": self.total_vocab_size(),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> TokenRegistry:
        """Reconstruct from a manifest, preserving every block's exact range.

        Uses :meth:`register_block` (not :meth:`allocate`) so overlapping
        corpus blocks round-trip faithfully — the persisted manifest is the
        source of truth for the real ranges, including the deliberate
        per-group overlap.
        """
        payload = json.loads(text)
        out = cls(version=payload["version"])
        out._cursor = CONTROL_RANGE[1]  # reset before re-registering
        for block in payload["blocks"]:
            out.register_block(block["name"], block["start"], block["end"])
        return out


# ---------------------------------------------------------------------------
# Reconstruct the real corpus namespace from on-disk truth
# ---------------------------------------------------------------------------


def _read_codebook_size(zarr_group) -> int | None:
    """Read ``metadata.codebook_size`` from one store group's attrs."""
    attrs = dict(zarr_group.attrs)
    meta_raw = attrs.get("metadata", "{}")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
    cs = meta.get("codebook_size")
    return int(cs) if cs is not None else None


def _scan_block_codebook_sizes(signals_hf_root) -> dict[str, int]:
    """Scan the on-disk ``signals_hf`` stores for each block's real size.

    Walks shot directories until it has read a ``metadata.codebook_size`` for
    every codebook-sized block in :data:`_HF_GROUP_LAYOUT` (xma/xim/xsx
    patch).  Returns ``{store_group: codebook_size}``.  A block that uses a
    fixed placeholder size (mode/profile) is not scanned here — its size is a
    literal in the layout table.

    Raises if a required codebook block is never found on disk (a missing
    store means the namespace cannot be derived from truth — refuse to guess).
    """
    from pathlib import Path

    import zarr

    root = Path(signals_hf_root)
    # Which store groups carry a codebook size we must read.
    wanted = {
        grp
        for grp, blocks in _HF_GROUP_LAYOUT
        for (_name, src) in blocks
        if src == "codebook"
    }
    found: dict[str, int] = {}
    if not root.exists():
        raise FileNotFoundError(f"signals_hf root absent: {root}")
    for shot_dir in sorted(root.iterdir()):
        if not shot_dir.is_dir() or not shot_dir.name.isdigit():
            continue
        for grp in list(wanted - set(found)):
            store_path = shot_dir / f"{grp}.zarr"
            if not store_path.exists():
                continue
            try:
                store = zarr.open_group(str(store_path), mode="r")
                cs = _read_codebook_size(store)
            except Exception as exc:  # noqa: BLE001 — skip a half-written store
                logger.warning("could not read %s: %r", store_path, exc)
                continue
            if cs is not None and cs >= 1:
                found[grp] = cs
        if wanted <= set(found):
            break
    missing = wanted - set(found)
    if missing:
        raise ValueError(
            f"could not read codebook_size for corpus group(s) {sorted(missing)} "
            f"under {root} — refusing to reconstruct the namespace from a guess"
        )
    return found


def reconstruct_v2_namespace_from_stores(
    signals_hf_root,
    *,
    block_codebook_sizes: dict[str, int] | None = None,
) -> TokenRegistry:
    """Reconstruct the real v2 corpus namespace from the on-disk HF stores.

    Models the deliberate per-group overlap: each group is encoded by an
    independent process, so every group's blocks start at the control range.
    For each group in :data:`_HF_GROUP_LAYOUT` the blocks are laid out
    starting at ``CONTROL_RANGE[1]`` in the group's own encode order, using
    the real ``codebook_size`` read from that group's stores (or the fixed
    placeholder size for a mode/profile block).

    The returned registry's cursor sits at the maximum block end across all
    groups, so :func:`allocate_l2_input_block` places the L2 block strictly
    above every real corpus id.

    ``block_codebook_sizes`` (``{store_group: size}``) lets a caller inject
    sizes (tests); when ``None`` the sizes are scanned off disk.
    """
    sizes = (
        block_codebook_sizes
        if block_codebook_sizes is not None
        else _scan_block_codebook_sizes(signals_hf_root)
    )
    reg = TokenRegistry()
    for grp, blocks in _HF_GROUP_LAYOUT:
        cursor = CONTROL_RANGE[1]  # every group's encode process restarts here
        for name, src in blocks:
            size = sizes[grp] if src == "codebook" else int(src)
            if size < 1:
                raise ValueError(f"{name!r}: non-positive size {size}")
            reg.register_block(name, cursor, cursor + size)
            cursor += size
    return reg


def persist_v2_registry(reg: TokenRegistry, manifest_path=None):  # -> Path
    """Write ``reg`` to the v2 registry manifest (the single source of truth).

    ``manifest_path`` defaults to ``TOKEN_ROOT/v2/registry.json`` via
    :func:`imas_ambix.tokenizer.store_v2.registry_v2_path`.  Returns the
    written path.
    """
    from pathlib import Path

    from imas_ambix.tokenizer.store_v2 import registry_v2_path

    path = Path(manifest_path) if manifest_path is not None else registry_v2_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reg.to_json())
    return path


def load_v2_registry(manifest_path=None) -> TokenRegistry | None:
    """Load the persisted v2 registry manifest, or ``None`` if absent."""
    from pathlib import Path

    from imas_ambix.tokenizer.store_v2 import registry_v2_path

    path = Path(manifest_path) if manifest_path is not None else registry_v2_path()
    if not path.exists():
        return None
    return TokenRegistry.from_json(path.read_text())


def build_or_load_v2_registry(
    signals_hf_root,
    *,
    manifest_path=None,
    persist: bool = True,
) -> TokenRegistry:
    """Return the authoritative v2 registry: load the manifest, else rebuild.

    On a first call (no manifest) this reconstructs the namespace from the
    on-disk stores and persists it; subsequent calls load the persisted
    manifest so the allocation is stable and self-describing.
    """
    reg = load_v2_registry(manifest_path)
    if reg is not None:
        return reg
    reg = reconstruct_v2_namespace_from_stores(signals_hf_root)
    if persist:
        try:
            persist_v2_registry(reg, manifest_path)
        except OSError as exc:  # read-only / unwritable store root — still usable
            logger.warning("could not persist v2 registry manifest: %r", exc)
    return reg


# ---------------------------------------------------------------------------
# Consume-time rebasing to a DISJOINT unified-flat namespace (option (b))
# ---------------------------------------------------------------------------
#
# The on-disk corpus is per-group-local: every group's patch block starts at
# the control range, so the groups overlap.  A downstream model that wants a
# single 1-D vocabulary rebases the per-group-local ids into a disjoint
# unified stream at consume time — no re-encode.  Each known block (in the
# canonical :data:`_HF_GROUP_LAYOUT` order) is given a fresh CONTIGUOUS range
# above the control tokens; because :meth:`TokenRegistry.allocate` is strictly
# append-only, the resulting ranges are disjoint by construction.

# Canonical block order for the unified namespace — exactly the per-group,
# per-process encode order, flattened across groups.  Holding this as the
# single ordering source keeps the unified map deterministic.
UNIFIED_BLOCK_ORDER: tuple[str, ...] = tuple(
    name for _grp, blocks in _HF_GROUP_LAYOUT for (name, _src) in blocks
)

# Which store group reports each block's real size, and whether that size is
# read from ``metadata.codebook_size`` or is a fixed placeholder literal.
_BLOCK_SIZE_SOURCE: dict[str, tuple[str, str | int]] = {
    name: (grp, src) for grp, blocks in _HF_GROUP_LAYOUT for (name, src) in blocks
}


def build_unified_namespace(
    signals_hf_root=None,
    *,
    block_codebook_sizes: dict[str, int] | None = None,
) -> TokenRegistry:
    """Build the DISJOINT unified-flat namespace for consume-time rebasing.

    Lays every block in :data:`UNIFIED_BLOCK_ORDER` into a fresh contiguous
    range starting at the control range, in canonical order.  Unlike the
    per-group-local on-disk layout (where every group restarts at id 4 and the
    groups overlap), each block here gets its OWN disjoint span, so an id in
    the unified stream is unambiguous on its own.

    Sizes are the REAL model-derived sizes — read from each store group's
    ``metadata.codebook_size`` on disk (never a hardcoded table) for the
    codebook-sized blocks, and the encoder's fixed placeholder literal for the
    size-1 mode/profile blocks.  ``block_codebook_sizes`` (``{store_group:
    size}``) lets a caller inject the on-disk sizes (tests); when ``None`` the
    sizes are scanned off disk via :func:`_scan_block_codebook_sizes`.

    Because :meth:`TokenRegistry.allocate` is strictly append-only, the
    returned registry's blocks are pairwise DISJOINT.
    """
    if block_codebook_sizes is None:
        from imas_ambix.data.paths import TOKEN_ROOT

        if signals_hf_root is None:
            signals_hf_root = TOKEN_ROOT / VOCAB_VERSION / "signals_hf"
        block_codebook_sizes = _scan_block_codebook_sizes(signals_hf_root)

    reg = TokenRegistry()
    for name in UNIFIED_BLOCK_ORDER:
        grp, src = _BLOCK_SIZE_SOURCE[name]
        size = block_codebook_sizes[grp] if src == "codebook" else int(src)
        if size < 1:
            raise ValueError(f"{name!r}: non-positive size {size}")
        reg.allocate(name, size)  # contiguous append → disjoint by construction
    return reg


def unified_global_ids(
    group_block_name: str,
    local_ids: object,
    *,
    unified: TokenRegistry | None = None,
    signals_hf_root=None,
    block_codebook_sizes: dict[str, int] | None = None,
) -> object:
    """Rebase a group's PER-GROUP-LOCAL ids into the disjoint unified stream.

    ``group_block_name`` is the block constant the group was encoded under
    (e.g. :data:`BLOCK_XIM_PATCH` — also the store's ``tokenizer_name``); this
    is the only disambiguator the on-disk per-group-local ids carry.
    ``local_ids`` are the group's ids RELATIVE to its own block start (the
    per-group-local id ``= on-disk global id − group_block_start``; on disk
    every group's block starts at the control range, so for the on-disk
    arrays the local id is ``on_disk_id − CONTROL_RANGE[1]``).

    Returns ``local_ids`` offset by the block's DISJOINT start in the unified
    namespace, as int32.  Two different groups can carry the same local id and
    will map to two DIFFERENT unified ids — that is the whole point.

    ``unified`` reuses an already-built unified namespace (cheap path: build it
    once, rebase many groups); otherwise it is built from
    ``block_codebook_sizes`` / on-disk truth.
    """
    if unified is None:
        unified = build_unified_namespace(
            signals_hf_root, block_codebook_sizes=block_codebook_sizes
        )
    return unified.shift(group_block_name, local_ids)


def persist_unified_namespace(
    unified: TokenRegistry | None = None,
    *,
    manifest_path=None,
    signals_hf_root=None,
    block_codebook_sizes: dict[str, int] | None = None,
):  # -> Path
    """Write the disjoint unified map into ``TOKEN_ROOT/v2/registry.json``.

    The manifest carries BOTH views: the existing per-group-local ``blocks``
    array (the on-disk reality, untouched — preserved if a manifest already
    exists) PLUS a ``unified_blocks`` array recording the authoritative
    disjoint unified-flat layout.  Writing the unified map therefore never
    disturbs the per-group-local floor the L2 light path depends on.

    Returns the written path.
    """
    from pathlib import Path

    from imas_ambix.tokenizer.store_v2 import registry_v2_path

    if unified is None:
        unified = build_unified_namespace(
            signals_hf_root, block_codebook_sizes=block_codebook_sizes
        )

    path = Path(manifest_path) if manifest_path is not None else registry_v2_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the existing per-group-local ``blocks`` manifest if present;
    # only ADD/refresh the ``unified_blocks`` section.
    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.setdefault("version", VOCAB_VERSION)
    payload.setdefault("control_tokens", dict(CONTROL_TOKENS))
    payload["unified_blocks"] = [
        {"name": b.name, "start": b.start, "end": b.end}
        for b in unified._blocks.values()
    ]
    payload["unified_total_vocab_size"] = unified.total_vocab_size()
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_unified_namespace(manifest_path=None) -> TokenRegistry | None:
    """Load the persisted disjoint unified map, or ``None`` if not yet written.

    Reads the ``unified_blocks`` section of ``TOKEN_ROOT/v2/registry.json``
    (NOT the per-group-local ``blocks`` array).
    """
    from pathlib import Path

    from imas_ambix.tokenizer.store_v2 import registry_v2_path

    path = Path(manifest_path) if manifest_path is not None else registry_v2_path()
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    unified = payload.get("unified_blocks")
    if not unified:
        return None
    out = TokenRegistry(version=payload.get("version", VOCAB_VERSION))
    out._cursor = CONTROL_RANGE[1]
    for block in unified:
        out.register_block(block["name"], block["start"], block["end"])
    return out


# ---------------------------------------------------------------------------
# Level-2 input block allocation (above the real corpus maximum)
# ---------------------------------------------------------------------------


def allocate_l2_input_block(
    reg: TokenRegistry | None = None,
    *,
    vocab_size: int = L2_BLOCK_VOCAB,
    signals_hf_root=None,
) -> tuple[int, int]:
    """Allocate :data:`BLOCK_L2_INPUT_LOW` strictly ABOVE every real corpus id.

    The corpus block ranges come from on-disk truth — either ``reg`` already
    carries them (the normal path: pass the reconstructed/loaded registry) or,
    when ``reg`` is the bare default singleton with no corpus blocks, they are
    reconstructed from the stores under ``signals_hf_root`` and merged in so
    the L2 floor is the real corpus maximum, never a fictional table.

    Returns the ``(start, end)`` global-id range of the L2 input block, which
    satisfies ``start == reg.max_block_end()`` (idempotent on repeat calls).
    """
    reg = reg if reg is not None else registry

    # If the registry has no corpus blocks yet, seed it from on-disk truth so
    # the L2 floor is the REAL maximum corpus id, not a guess.  An already-
    # seeded registry (the manifest was loaded / reconstructed upstream) is
    # left untouched — register_block is idempotent for matching ranges.
    have_corpus = any(
        name in reg._blocks
        for _grp, blocks in _HF_GROUP_LAYOUT
        for (name, _src) in blocks
    )
    if not have_corpus:
        from imas_ambix.data.paths import TOKEN_ROOT

        if signals_hf_root is None:
            signals_hf_root = TOKEN_ROOT / VOCAB_VERSION / "signals_hf"
        corpus = reconstruct_v2_namespace_from_stores(signals_hf_root)
        for name, b in corpus._blocks.items():
            reg.register_block(name, b.start, b.end)

    return reg.allocate(BLOCK_L2_INPUT_LOW, vocab_size)


# The default singleton registry — most callers want this.
registry = TokenRegistry()
