"""Training-window loader for persisted per-shot token streams.

Each shot's token data lives in a Zarr file written by :mod:`persist`.
This module mmap's each shot lazily and yields fixed-size windows of
``(input_ids, labels, attn_mask, loss_mask)`` suitable for the
world-model training loop.

Loss-mask conventions (from plans/world-model-v0.md §6):

==========  =================
block_kind  loss_mask weight
==========  =================
0           0.0  (control tokens)
1           1.0  (frame tokens)
2           0.3  (signal tokens)
3           0.1  (action tokens)
==========  =================

If the Zarr does not contain a ``block_kind`` side array the loader
falls back to an all-ones mask and emits a one-time ``warnings.warn``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Block-kind → loss-mask weight map
# ---------------------------------------------------------------------------

BLOCK_WEIGHTS: dict[int, float] = {
    0: 0.0,  # control
    1: 1.0,  # frame
    2: 0.3,  # signal
    3: 0.1,  # action
}
"""Per-block-kind loss-mask weights (``plans/world-model-v0.md`` §4.1).

Keyed by :class:`~imas_ambix.tokenizer.base.BlockKind` integer code:

=====  =======  =============
Code   Name     loss_mask
=====  =======  =============
0      CONTROL  0.0
1      FRAME    1.0
2      SIGNAL   0.3
3      ACTION   0.1
=====  =======  =============
"""

# Private alias kept for backward compatibility with existing tests that
# import ``_BLOCK_KIND_WEIGHTS``.
_BLOCK_KIND_WEIGHTS = BLOCK_WEIGHTS

_DEFAULT_LOSS_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class ShotTokenSpec:
    """Metadata for a single shot's token stream.

    Attributes
    ----------
    shot_id:
        Integer shot identifier.
    n_tokens:
        Total length of the 1-D token stream (number of int32 values).
    path:
        Path to the Zarr root that stores the ``tokens`` array (and
        optionally ``block_kind``).
    """

    shot_id: int
    n_tokens: int
    path: Path


@dataclass
class WindowSamplerConfig:
    """Hyper-parameters that control how training windows are sampled.

    Attributes
    ----------
    context_length:
        Number of tokens in each training window.  Defaults to 16 384.
    stride:
        Minimum gap between window start positions when iterating through
        a shot.  Defaults to 4 096.
    seed:
        Base random seed.  Each ``__iter__`` call uses a fresh
        ``numpy.random.Generator`` seeded with this value so training
        epochs are reproducible when the data-loader is re-iterated.
    """

    context_length: int = 16384
    stride: int = 4096
    seed: int = 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ShotTokenDataset:
    """Iterable dataset that yields fixed-size training windows.

    Each window is a ``dict`` with four ``numpy.ndarray`` keys:

    ``input_ids``
        ``int32`` array of length ``context_length``.
    ``labels``
        ``int32`` array equal to ``input_ids`` (next-token prediction).
    ``attn_mask``
        ``int32`` array of all ones (no padding in fixed-size windows).
    ``loss_mask``
        ``float32`` array in ``[0, 1]`` derived from ``block_kind``.
        Falls back to all-ones when ``block_kind`` is absent from the
        Zarr store.

    Parameters
    ----------
    shot_specs:
        Sequence of :class:`ShotTokenSpec` objects, one per shot.
    config:
        Sampling configuration.

    Notes
    -----
    Zarr arrays are opened on demand inside ``__iter__`` to keep the
    dataset object cheap to construct and safe to share across processes.
    """

    def __init__(
        self,
        shot_specs: list[ShotTokenSpec],
        config: WindowSamplerConfig,
    ) -> None:
        self._specs = shot_specs
        self._config = config
        self._warned_no_block_kind: bool = False

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _loss_mask_from_block_kind(self, block_kind: np.ndarray) -> np.ndarray:
        """Convert a ``block_kind`` uint8 array to a float32 loss-mask."""
        mask = np.full(block_kind.shape, _DEFAULT_LOSS_WEIGHT, dtype=np.float32)
        for code, weight in BLOCK_WEIGHTS.items():
            mask[block_kind == code] = weight
        return mask

    def _window_from_spec(
        self,
        spec: ShotTokenSpec,
        start: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Load one window from a :class:`ShotTokenSpec` starting at *start*."""
        import zarr  # noqa: PLC0415

        cl = self._config.context_length
        store = zarr.open_group(str(spec.path), mode="r")
        tokens_arr = store["tokens"]
        end = start + cl
        input_ids = np.asarray(tokens_arr[start:end], dtype=np.int32)

        # Pad with zeros if the shot is shorter than context_length
        if input_ids.shape[0] < cl:
            input_ids = np.pad(input_ids, (0, cl - input_ids.shape[0]))

        # block_kind side data
        if "block_kind" in store:
            raw_kind = np.asarray(store["block_kind"][start:end], dtype=np.uint8)
            if raw_kind.shape[0] < cl:
                raw_kind = np.pad(raw_kind, (0, cl - raw_kind.shape[0]))
            loss_mask = self._loss_mask_from_block_kind(raw_kind)
        else:
            if not self._warned_no_block_kind:
                warnings.warn(
                    f"shot {spec.shot_id}: Zarr at {spec.path} has no 'block_kind' "
                    "array — falling back to uniform loss_mask=1.0. "
                    "Run a ShotTokenizer that writes block_kind to suppress this.",
                    stacklevel=2,
                )
                self._warned_no_block_kind = True
            loss_mask = np.ones(cl, dtype=np.float32)

        return {
            "input_ids": input_ids,
            "labels": input_ids.copy(),
            "attn_mask": np.ones(cl, dtype=np.int32),
            "loss_mask": loss_mask,
        }

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield windows, sampling shots uniformly at random."""
        rng = np.random.default_rng(self._config.seed)
        cl = self._config.context_length
        stride = self._config.stride

        # Build a flat list of (spec, start) windows
        windows: list[tuple[ShotTokenSpec, int]] = []
        for spec in self._specs:
            if spec.n_tokens < cl:
                # Shot is shorter than one window — yield from start (padded)
                windows.append((spec, 0))
                continue
            # Stride through the shot
            starts = list(range(0, spec.n_tokens - cl + 1, stride))
            for s in starts:
                windows.append((spec, s))

        if not windows:
            return

        # Shuffle windows each epoch
        indices = rng.permutation(len(windows))
        for idx in indices:
            spec, start = windows[int(idx)]
            yield self._window_from_spec(spec, start, rng)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        config: WindowSamplerConfig,
    ) -> ShotTokenDataset:
        """Build a :class:`ShotTokenDataset` from an ambix data manifest JSON.

        The manifest is expected to be a JSON object produced by
        ``ambix data manifest --output <path>`` with at minimum::

            {
                "shot_ids": [30001, 30002, ...],
                "vocab_version": "v1"
            }

        Each shot's Zarr path is resolved via
        :func:`~imas_ambix.data.persist.frames_token_path` for the
        default camera ``"rbb"``.  For multi-camera or signal manifests
        callers should construct :class:`ShotTokenSpec` objects directly.

        Parameters
        ----------
        manifest_path:
            Path to the JSON manifest file.
        config:
            Sampling configuration.

        Returns
        -------
        ShotTokenDataset
        """
        import zarr  # noqa: PLC0415

        from imas_ambix.data.persist import frames_token_path  # noqa: PLC0415

        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        shot_ids: list[int] = [int(s) for s in payload.get("shot_ids", [])]
        vocab_version: str = payload.get("vocab_version", "v1")
        camera: str = payload.get("camera", "rbb")

        specs: list[ShotTokenSpec] = []
        for shot_id in shot_ids:
            path = frames_token_path(shot_id, camera, vocab_version)
            if not path.exists():
                continue
            store = zarr.open_group(str(path), mode="r")
            n_tokens = int(store["tokens"].shape[0])
            specs.append(ShotTokenSpec(shot_id=shot_id, n_tokens=n_tokens, path=path))

        return cls(specs, config)


# ---------------------------------------------------------------------------
# Convenience builder for signal token streams
# ---------------------------------------------------------------------------


def build_signal_dataset(
    shot_ids: list[int],
    group: str,
    config: WindowSamplerConfig,
    vocab_version: str = "v1",
) -> ShotTokenDataset:
    """Build a :class:`ShotTokenDataset` from persisted signal tokens.

    Parameters
    ----------
    shot_ids:
        List of shot IDs to include.
    group:
        Signal group name (e.g. ``"magnetics"``).
    config:
        Sampling configuration.
    vocab_version:
        Token vocabulary version (default ``"v1"``).

    Returns
    -------
    ShotTokenDataset
        Dataset wrapping the signal token streams for the given shots.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.data.persist import signals_token_path  # noqa: PLC0415

    specs: list[ShotTokenSpec] = []
    for shot_id in shot_ids:
        path = signals_token_path(shot_id, group, vocab_version)
        if not path.exists():
            continue
        store = zarr.open_group(str(path), mode="r")
        tokens = store["tokens"]
        # Flatten multi-dim token arrays to 1-D for window sampling
        n_tokens = int(np.prod(tokens.shape))
        specs.append(ShotTokenSpec(shot_id=shot_id, n_tokens=n_tokens, path=path))

    return ShotTokenDataset(specs, config)
