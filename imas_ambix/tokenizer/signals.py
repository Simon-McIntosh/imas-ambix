"""Signal tokenizer wrappers.

Two implementations:

- :class:`UniformQuantizer` — per-channel mean/std scale + uniform
  quantization to ``n_bins`` levels. Works without any external
  dependency; suitable for the v0 round-trip plumbing.
- :class:`ChronosSignalTokenizer` (planned) — Amazon Chronos T5-small
  wrapper. The pretrained quantize-then-tokenize approach is described
  in ``plans/tokenizers.md`` §3.

Input contract: an ``xarray.Dataset`` whose data variables share a
single 1-D ``time`` dimension at the **model time grid** (see
:mod:`alignment`). Channels that don't share the time axis (e.g.
geometry constants like ``flux_loop_r``) are passed through as
dataset attributes, not tokenised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

from imas_ambix.tokenizer.base import EncodedSignals
from imas_ambix.tokenizer.registry import registry


def _time_indexed_channels(ds: xr.Dataset, time_dim: str = "time") -> list[str]:
    """Return data-var names whose only varying dim is ``time_dim``."""
    out = []
    for name, var in ds.data_vars.items():
        if time_dim not in var.dims:
            continue
        if var.ndim == 1:
            out.append(name)
        # Skip multi-dim vars (e.g. (time, major_radius)) — they need
        # their own per-dim handling; tracked as a v1 TODO in plans.
    return sorted(out)


@dataclass
class UniformQuantizer:
    """Per-channel mean/std normalise → uniform quantization → token ids.

    For each channel, compute mean and std across the training set
    (passed via :meth:`fit`). At encode time, ``(x - mean) / std`` is
    clipped to ``[-clip, +clip]`` then mapped linearly into
    ``[0, n_bins)``. Decoding inverts the map and returns the bin
    midpoint.

    ``vocab_size`` is ``n_bins`` shared across every channel. The token
    *position* (column index) tells the model which channel is being
    represented; the *value* tells it the quantised reading.
    """

    name: str = "signals_uniform_v1"
    n_bins: int = 256
    clip_sigma: float = 4.0

    def __post_init__(self) -> None:
        self.vocab_size = self.n_bins
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._channel_order: tuple[str, ...] = ()
        registry.allocate(self.name, self.vocab_size)

    @property
    def patch_size(self) -> int:  # one model timestep per token
        return 1

    def fit(self, datasets: list[xr.Dataset], time_dim: str = "time") -> None:
        """Accumulate per-channel mean and std over a list of training datasets."""
        import numpy as np

        # Discover the channel order from the first dataset.
        if not datasets:
            raise ValueError("need at least one dataset to fit")
        self._channel_order = tuple(_time_indexed_channels(datasets[0], time_dim))

        # Running mean and variance via Welford could be tidier, but for
        # v0 a simple two-pass over concatenated arrays is fine.
        per_channel: dict[str, list[np.ndarray]] = {n: [] for n in self._channel_order}
        for ds in datasets:
            for name in self._channel_order:
                if name not in ds:
                    continue
                arr = np.asarray(ds[name].values, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    per_channel[name].append(arr)

        for name, chunks in per_channel.items():
            if not chunks:
                self._means[name] = 0.0
                self._stds[name] = 1.0
                continue
            cat = np.concatenate(chunks)
            self._means[name] = float(cat.mean())
            std = float(cat.std())
            self._stds[name] = std if std > 1e-12 else 1.0

    def encode(self, ds: xr.Dataset) -> EncodedSignals:
        """Quantize an aligned-time dataset to per-channel global token ids."""
        import numpy as np

        if not self._channel_order:
            # No fit — fall back to per-encode normalisation
            channels = tuple(_time_indexed_channels(ds))
        else:
            channels = self._channel_order

        cols = []
        used: list[str] = []
        for name in channels:
            if name not in ds:
                continue
            arr = np.asarray(ds[name].values, dtype=np.float64)
            mean = self._means.get(name, float(np.nanmean(arr)))
            std = self._stds.get(name, float(np.nanstd(arr)) or 1.0)
            z = np.nan_to_num((arr - mean) / max(std, 1e-12), nan=0.0)
            z = z.clip(-self.clip_sigma, self.clip_sigma)
            local = (
                (((z + self.clip_sigma) / (2 * self.clip_sigma)) * (self.n_bins - 1))
                .round()
                .astype(np.int32)
            )
            local = local.clip(0, self.n_bins - 1)
            cols.append(local)
            used.append(name)

        if not cols:
            return EncodedSignals(
                token_ids=np.zeros((0, 0), dtype=np.int32),
                channel_names=(),
                tokenizer_name=self.name,
                metadata={},
            )
        stacked = np.stack(cols, axis=-1)  # (T, n_channels)
        global_ids = registry.shift(self.name, stacked)
        return EncodedSignals(
            token_ids=global_ids,
            channel_names=tuple(used),
            tokenizer_name=self.name,
            metadata={
                "n_bins": self.n_bins,
                "clip_sigma": self.clip_sigma,
                "fitted": bool(self._channel_order),
            },
        )

    def decode(self, tokens: EncodedSignals) -> xr.Dataset:
        """Decode quantized tokens back into an xarray Dataset."""
        import numpy as np
        import xarray as xr

        start, _ = registry.allocate(self.name, self.vocab_size)
        ids = np.asarray(tokens.token_ids, dtype=np.int64) - start
        out_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for col_idx, name in enumerate(tokens.channel_names):
            local = ids[..., col_idx]
            z = (local.astype(np.float64) / (self.n_bins - 1)) * (
                2 * self.clip_sigma
            ) - self.clip_sigma
            mean = self._means.get(name, 0.0)
            std = self._stds.get(name, 1.0)
            value = z * std + mean
            out_vars[name] = (("time",), value)
        return xr.Dataset({k: v for k, v in out_vars.items()})


@dataclass
class ChronosSignalTokenizer:
    """Amazon Chronos T5-small wrapper — not yet implemented.

    See ``plans/tokenizers.md`` §3:

    - Apache-2.0, https://github.com/amazon-science/chronos-forecasting
    - HF model: ``amazon/chronos-t5-small``
    - Quantise-then-tokenize approach: scale by mean/std, uniform-bin
      to T5 vocab ids, treat as a language sequence.
    - For v0 we use the published checkpoint as-is — only the per-channel
      mean / std calibration is re-computed on the MAST training split.

    Allocated as a separate name to keep us free to swap implementations
    without invalidating Chronos-emitted token streams.
    """

    name: str = "signals_chronos_t5_small_v1"
    vocab_size: int = 4096

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "ChronosSignalTokenizer is not yet wired up — see "
            "plans/tokenizers.md §3 for the rollout plan"
        )

    @property
    def patch_size(self) -> int:  # pragma: no cover
        return 1

    def encode(self, ds: xr.Dataset) -> EncodedSignals:  # pragma: no cover
        raise NotImplementedError

    def decode(self, tokens: EncodedSignals) -> xr.Dataset:  # pragma: no cover
        raise NotImplementedError
