"""Signal tokenizer wrappers.

Three implementations:

- :class:`UniformQuantizer` — per-channel mean/std scale + uniform
  quantization to ``n_bins`` levels. Works without any external
  dependency; suitable for the v0 round-trip plumbing.
- :class:`ChronosSignalTokenizer` — Amazon Chronos T5-small wrapper.
  Uses the ``MeanScaleUniformBins`` quantizer shipped with
  ``chronos-forecasting`` to map each channel into ``[0, 4095]`` local
  token ids. Requires ``chronos-forecasting`` to be installed.
- :class:`PatchTSTTokenizer` — identity passthrough for PatchTST. Raw
  float patches are preserved verbatim in ``metadata["patches"]`` so the
  world-model transformer can apply its learned patch-projection. The
  single token id emitted per patch is always zero (shifted into the
  registry range).

Input contract: an ``xarray.Dataset`` whose data variables share a
single 1-D ``time`` dimension at the **model time grid** (see
:mod:`alignment`). Channels that don't share the time axis (e.g.
geometry constants like ``flux_loop_r``) are passed through as
dataset attributes, not tokenised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

    from imas_ambix.calibration.signals import ChannelCalibration

from imas_ambix.tokenizer.base import EncodedSignals
from imas_ambix.tokenizer.registry import registry

logger = logging.getLogger(__name__)


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
        self._calibration: dict[str, ChannelCalibration] | None = None
        registry.allocate(self.name, self.vocab_size)

    @property
    def patch_size(self) -> int:  # one model timestep per token
        return 1

    def set_calibration(
        self, calibration: dict[str, ChannelCalibration] | None
    ) -> None:
        """Supply corpus-wide (absolute / SI) per-channel mean+std.

        When set, :meth:`encode` / :meth:`decode` standardise each channel
        against the CORPUS mean/std supplied here instead of any per-shot
        ``fit`` statistics.  Because the corpus stats are constant across
        every shot and every machine, the same physical value maps to the
        same bin everywhere — absolute magnitude survives tokenisation.

        A channel with no entry in ``calibration`` falls back to the
        per-shot / per-fit statistics (with a one-line warning) so a partial
        calibration is never silently magnitude-destroying.  Passing ``None``
        restores the default (per-fit / per-shot) behaviour exactly.
        """
        self._calibration = calibration

    def _resolve_stats(self, name: str, arr) -> tuple[float, float]:
        """Return ``(mean, std)`` for one channel under the active mode.

        Corpus-calibration mode (``_calibration`` set and the channel present)
        wins; otherwise the fitted stats; otherwise per-shot stats from ``arr``
        (with a warning when calibration is set but this channel is missing).
        """
        import numpy as np

        if self._calibration is not None:
            cal = self._calibration.get(name)
            if cal is not None:
                return float(cal.mean), max(float(cal.std), 1e-12)
            logger.warning(
                "UniformQuantizer: no corpus calibration for channel %r — "
                "falling back to per-shot stats (absolute magnitude not "
                "preserved for this channel)",
                name,
            )
        mean = self._means.get(name, float(np.nanmean(arr)))
        std = self._stds.get(name, float(np.nanstd(arr)) or 1.0)
        return mean, max(std, 1e-12)

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
            mean, std = self._resolve_stats(name, arr)
            z = np.nan_to_num((arr - mean) / std, nan=0.0)
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
                "calibration": "absolute"
                if self._calibration is not None
                else "per_shot",
            },
        )

    def decode(self, tokens: EncodedSignals) -> xr.Dataset:
        """Decode quantized tokens back into an xarray Dataset.

        Inverts :meth:`encode` with the SAME per-channel mean/std it encoded
        with — corpus (absolute) stats when calibration is set, otherwise the
        fitted / default per-shot stats.
        """
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
            mean, std = self._decode_stats(name)
            value = z * std + mean
            out_vars[name] = (("time",), value)
        return xr.Dataset({k: v for k, v in out_vars.items()})

    def _decode_stats(self, name: str) -> tuple[float, float]:
        """Per-channel ``(mean, std)`` for decode (no raw array available)."""
        if self._calibration is not None:
            cal = self._calibration.get(name)
            if cal is not None:
                return float(cal.mean), max(float(cal.std), 1e-12)
        return self._means.get(name, 0.0), self._stds.get(name, 1.0)


class ChronosUnavailableError(RuntimeError):
    """Raised when ``chronos-forecasting`` cannot be imported.

    Install with::

        uv pip install chronos-forecasting
        # or, via the project optional-deps:
        uv pip install "imas-ambix[train]"
    """


def _build_chronos_tokenizer() -> object:
    """Construct a :class:`chronos.MeanScaleUniformBins` tokenizer.

    Uses the published Chronos T5-small configuration:
    4096 token vocab, 2 special tokens (pad=0, eos=1),
    ``low_limit=-1.0`` / ``high_limit=1.0`` uniform bins — matching the
    HuggingFace ``amazon/chronos-t5-small`` ``config.json``.

    The tokenizer object is constructed from the published constants so
    that the model weights are **not** required for the
    quantize-then-tokenize encode/decode step. Only the tokenizer math
    (not the T5 transformer) is needed here.
    """
    try:
        from chronos import ChronosConfig  # noqa: PLC0415
    except ImportError as exc:
        raise ChronosUnavailableError(
            "chronos-forecasting is not installed. "
            "Install with: uv pip install chronos-forecasting"
        ) from exc

    cfg = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -1.0, "high_limit": 1.0},
        context_length=512,
        prediction_length=64,
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=False,  # we use _input_transform directly, skip eos
        model_type="seq2seq",
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    return cfg.create_tokenizer()


@dataclass
class ChronosSignalTokenizer:
    """Amazon Chronos T5-small signal tokenizer.

    Uses the ``MeanScaleUniformBins`` quantizer from
    ``chronos-forecasting`` to map normalised signal values into 4096
    discrete local token ids in ``[0, 4095]``.

    The ``chronos-forecasting`` package is imported lazily inside
    :meth:`encode` / :meth:`decode` so that the module loads cheaply even
    when the package is absent.  A :class:`ChronosUnavailableError` is
    raised on the first encode/decode call if the package is missing.

    Encode pipeline per channel:
    1. Mean/std-normalise the raw signal using ``fit``-computed statistics.
    2. Pass normalised values through ``MeanScaleUniformBins._input_transform``
       which applies Chronos's internal mean-scale normalisation and returns
       local token ids in ``[0, 4095]``.
    3. Shift local ids into the global registry range via
       ``registry.shift(self.name, ...)``.

    Decode pipeline per channel:
    1. Shift global ids back to local.
    2. Call ``MeanScaleUniformBins.output_transform`` with the stored
       Chronos scale to recover the mean-normalised float values.
    3. Denormalise by per-channel mean + std to recover original units.

    References:
        - https://github.com/amazon-science/chronos-forecasting (Apache-2.0)
        - HF model: ``amazon/chronos-t5-small``
    """

    name: str = "signals_chronos_t5_small_v1"
    vocab_size: int = 4096
    patch_size: int = 1

    def __post_init__(self) -> None:
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._channel_order: tuple[str, ...] = ()
        self._calibration: dict[str, ChannelCalibration] | None = None
        self._tokenizer: object | None = None
        registry.allocate(self.name, self.vocab_size)

    def set_calibration(
        self, calibration: dict[str, ChannelCalibration] | None
    ) -> None:
        """Supply corpus-wide (absolute / SI) per-channel mean+std.

        When set, the pre-Chronos mean/std normalisation uses the CORPUS
        statistics instead of any per-shot ``fit`` stats, so absolute
        magnitude is preserved into the Chronos quantiser.  A channel absent
        from ``calibration`` falls back to per-shot stats (with a warning).
        Passing ``None`` restores the default behaviour exactly.
        """
        self._calibration = calibration

    def _resolve_stats(self, name: str, arr) -> tuple[float, float]:
        """Return ``(mean, std)`` for one channel under the active mode."""
        import numpy as np

        if self._calibration is not None:
            cal = self._calibration.get(name)
            if cal is not None:
                return float(cal.mean), max(float(cal.std), 1e-12)
            logger.warning(
                "ChronosSignalTokenizer: no corpus calibration for channel "
                "%r — falling back to per-shot stats (absolute magnitude not "
                "preserved for this channel)",
                name,
            )
        mean = self._means.get(name, float(np.nanmean(arr)))
        std = self._stds.get(name, max(float(np.nanstd(arr)), 1e-12))
        return mean, max(std, 1e-12)

    def _get_tokenizer(self) -> object:
        """Return the cached Chronos tokenizer, building it on first call."""
        if self._tokenizer is None:
            self._tokenizer = _build_chronos_tokenizer()
        return self._tokenizer

    def fit(self, datasets: list[xr.Dataset], time_dim: str = "time") -> None:
        """Accumulate per-channel mean and std over a list of training datasets.

        Mirrors :meth:`UniformQuantizer.fit` — the statistics are used to
        normalise raw signal values to roughly zero-mean unit-variance
        before Chronos's internal mean-scale normalisation takes over.
        """
        import numpy as np

        if not datasets:
            raise ValueError("need at least one dataset to fit")
        self._channel_order = tuple(_time_indexed_channels(datasets[0], time_dim))

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
        """Encode an xarray Dataset into per-channel Chronos token ids.

        Each 1-D time-indexed channel is:
        1. Mean/std normalised using ``fit``-computed statistics.
        2. Passed through ``MeanScaleUniformBins`` to obtain local ids in
           ``[0, 4095]``.
        3. Shifted into the global registry namespace.

        The Chronos internal scale tensor and per-channel mean/std are
        stored in ``metadata`` for round-trip decoding.
        """
        import numpy as np
        import torch

        tok = self._get_tokenizer()

        channels = (
            self._channel_order
            if self._channel_order
            else tuple(_time_indexed_channels(ds))
        )

        cols: list[np.ndarray] = []
        used: list[str] = []
        means_out: dict[str, float] = {}
        stds_out: dict[str, float] = {}
        scales_out: dict[str, float] = {}

        for name in channels:
            if name not in ds:
                continue
            arr = np.asarray(ds[name].values, dtype=np.float64)
            mean, std = self._resolve_stats(name, arr)
            normalised = np.nan_to_num((arr - mean) / std, nan=0.0)

            t = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0)  # (1, T)
            ids, _mask, scale = tok._input_transform(t)  # type: ignore[attr-defined]
            local = ids[0].numpy().astype(np.int32)  # (T,)
            local = np.clip(local, 0, self.vocab_size - 1)

            cols.append(local)
            used.append(name)
            means_out[name] = mean
            stds_out[name] = std
            scales_out[name] = float(scale[0])

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
                "means": means_out,
                "stds": stds_out,
                "scale": scales_out,
            },
        )

    def decode(self, tokens: EncodedSignals) -> xr.Dataset:
        """Decode Chronos token ids back into an xarray Dataset.

        Inverts :meth:`encode`:
        1. Shift global ids back to local.
        2. Apply ``output_transform`` with the stored Chronos scale to
           recover the mean-normalised float values.
        3. Denormalise by per-channel mean + std.
        """
        import numpy as np
        import torch
        import xarray as xr

        tok = self._get_tokenizer()
        start, _ = registry.allocate(self.name, self.vocab_size)
        global_arr = np.asarray(tokens.token_ids, dtype=np.int64)  # (T, n_channels)
        local_arr = global_arr - start  # (T, n_channels)

        means: dict[str, float] = tokens.metadata.get("means", {})
        stds: dict[str, float] = tokens.metadata.get("stds", {})
        scales: dict[str, float] = tokens.metadata.get("scale", {})

        out_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for col_idx, name in enumerate(tokens.channel_names):
            local = local_arr[:, col_idx]  # (T,)
            ids_t = torch.tensor(local, dtype=torch.int64)
            # output_transform expects (B, n_samples, T)
            ids_3d = ids_t.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            ch_scale = float(scales.get(name, 1.0))
            scale_t = torch.tensor([ch_scale], dtype=torch.float32)
            recovered = tok.output_transform(ids_3d, scale_t)  # type: ignore[attr-defined]
            normalised = recovered[0, 0].numpy()  # (T,)

            mean = means.get(name, 0.0)
            std = stds.get(name, 1.0)
            value = normalised * std + mean
            out_vars[name] = (("time",), value)

        return xr.Dataset({k: v for k, v in out_vars.items()})


@dataclass
class PatchTSTTokenizer:
    """Identity passthrough tokenizer for PatchTST patch projection.

    In v0, the PatchTST patch-projection matrix is trained inside the
    world model's transformer (see ``plans/world-model-v0.md`` §2).
    This tokenizer therefore acts as an **identity passthrough**: it slices
    each 1-D time-indexed channel into non-overlapping ``patch_size``-sample
    patches and stores the raw float values verbatim in
    ``metadata["patches"]``.  The ``token_ids`` field is filled with zeros
    (the single "identity" id, shifted into the registry namespace) so that
    the token stream is structurally consistent with other tokenizers.

    Round-trip guarantee:
    ``np.allclose(decode(encode(ds))[ch], ds[ch])`` holds exactly for any
    dataset whose time length is a multiple of ``patch_size``. For
    non-multiples the final incomplete patch is zero-padded on encode and
    trimmed on decode to the original length.
    """

    name: str = "signals_patchtst_v1"
    patch_size: int = 64
    vocab_size: int = 1

    def __post_init__(self) -> None:
        registry.allocate(self.name, self.vocab_size)

    def encode(self, ds: xr.Dataset) -> EncodedSignals:
        """Slice each channel into patches; store raw floats in metadata.

        ``token_ids`` has shape ``(n_patches, n_channels)`` and is filled
        with the single "identity" id (zero, shifted into the registry
        range).  ``metadata["patches"]`` maps channel name to a
        ``numpy.ndarray`` of shape ``(n_patches, patch_size)``.
        ``metadata["original_lengths"]`` stores the un-padded time length
        per channel so :meth:`decode` can trim correctly.
        """
        import numpy as np

        channels = tuple(_time_indexed_channels(ds))
        if not channels:
            return EncodedSignals(
                token_ids=np.zeros((0, 0), dtype=np.int32),
                channel_names=(),
                tokenizer_name=self.name,
                metadata={"patches": {}, "original_lengths": {}},
            )

        patch_dict: dict[str, np.ndarray] = {}
        orig_lengths: dict[str, int] = {}
        n_patches_list: list[int] = []

        for name in channels:
            if name not in ds:
                continue
            arr = np.asarray(ds[name].values, dtype=np.float64)
            t = len(arr)
            orig_lengths[name] = t
            remainder = t % self.patch_size
            if remainder:
                pad_len = self.patch_size - remainder
                arr = np.concatenate([arr, np.zeros(pad_len, dtype=arr.dtype)])
            n_patches = len(arr) // self.patch_size
            n_patches_list.append(n_patches)
            patch_dict[name] = arr.reshape(n_patches, self.patch_size)

        # Use the first channel's patch count (all channels share the same
        # time axis so after padding they produce the same n_patches).
        n_patches = n_patches_list[0] if n_patches_list else 0
        n_channels = len(patch_dict)

        local_zeros = np.zeros((n_patches, n_channels), dtype=np.int32)
        global_ids = registry.shift(self.name, local_zeros)

        return EncodedSignals(
            token_ids=global_ids,
            channel_names=tuple(patch_dict.keys()),
            tokenizer_name=self.name,
            metadata={
                "patches": patch_dict,
                "original_lengths": orig_lengths,
            },
        )

    def decode(self, tokens: EncodedSignals) -> xr.Dataset:
        """Reconstruct the dataset from stored raw float patches.

        Concatenates patches along the time axis and trims to the original
        length stored in ``metadata["original_lengths"]``.
        """
        import xarray as xr

        patch_dict: dict[str, np.ndarray] = tokens.metadata.get("patches", {})
        orig_lengths: dict[str, int] = tokens.metadata.get("original_lengths", {})

        out_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for name in tokens.channel_names:
            if name not in patch_dict:
                continue
            patches = patch_dict[name]  # (n_patches, patch_size)
            flat = patches.reshape(-1)
            t = orig_lengths.get(name, len(flat))
            out_vars[name] = (("time",), flat[:t])

        return xr.Dataset({k: v for k, v in out_vars.items()})
