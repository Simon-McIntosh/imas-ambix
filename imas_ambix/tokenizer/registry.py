"""Global token id namespace registry.

Every concrete tokenizer (Open-MAGVIT2, Chronos, PatchTST, …) emits
**local** ids in ``[0, vocab_size)``. The registry packs them into a
contiguous **global** range so the world-model transformer sees a flat
1-D vocabulary.

Layout:

```
[0, 4)                  control tokens          (pad, bos, eos, sep)
[4, 4+N1)               Open-MAGVIT2 frame      (visible + IR share)
[4+N1, 4+N1+N2)         Chronos low-freq signal
[4+N1+N2, ...)          PatchTST patch tokens
[final position, end)   scalar / metadata embeddings (own table)
```

The registry is a singleton (:data:`registry`). Modules that emit
tokens call :meth:`TokenRegistry.allocate` once during construction and
get back a ``(start, end)`` range. Encoded ids are produced via
``local_id + start``; decoding uses :meth:`TokenRegistry.split`.

Vocab generations are versioned through :data:`VOCAB_VERSION`. Bumping
the version means existing cached tokens are invalidated. Persisted
token arrays under ``/work/projects/imas_gpu/mast-tokens/v{N}/`` store
the version they were produced under.

On-disk truth is authoritative
-------------------------------
The high-frequency v2 signal corpus (xma / xim / xsx) is encoded by
:mod:`imas_ambix.tokenizer.signal_hf_encode`, which sizes each block's
codebook **at runtime** from the trained model
(``patch_vocab = bottleneck.codebook_size``) — NOT from any compile-time
constant.  Crucially the encode runs **one process per group**, so every
group's id allocation restarts at the control range: the on-disk
``xma_patch``, ``xim_patch`` and ``xsx_patch`` blocks all begin at id 4
and therefore *overlap*.  The single ground truth for what ids the corpus
actually occupies is the on-disk store metadata
(``metadata.codebook_size``) cross-checked against the real global ids in
the token arrays — never a hand-maintained size table.

The Level-2 input light path (:mod:`imas_ambix.data.l2_input_build`) runs
in a single process and must place its block **strictly above** every id
the corpus uses, so a decoded L2 id can never alias a corpus block.  The
helpers below reconstruct the real corpus namespace from the on-disk
stores, persist it to ``TOKEN_ROOT/v2/registry.json`` as the
self-describing manifest, and allocate the L2 block above the real
maximum.
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
