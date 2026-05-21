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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

VOCAB_VERSION = "v1"


# Control tokens — fixed-position reserved range.
CONTROL_TOKENS = {
    "pad": 0,
    "bos": 1,
    "eos": 2,
    "sep": 3,
}
CONTROL_RANGE = (0, len(CONTROL_TOKENS))


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
    """Mutable allocator that assigns contiguous id ranges to tokenizers."""

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

    def split(self, global_id: int) -> tuple[str, int]:
        """Decode a global id into ``(tokenizer_name, local_id)``."""
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
        payload = json.loads(text)
        out = cls(version=payload["version"])
        out._cursor = CONTROL_RANGE[1]  # reset before re-allocating
        for block in payload["blocks"]:
            out.allocate(block["name"], block["end"] - block["start"])
        return out


# The default singleton registry — most callers want this.
registry = TokenRegistry()
