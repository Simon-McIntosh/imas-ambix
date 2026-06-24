"""Heteroscedastic probe: measured diagnostics -> plasma equilibrium geometry.

EVALUATOR-ONLY (binding firewall)
---------------------------------
This probe is part of the **independent equilibrium evaluator**, never the
world model.  It reads *measured-diagnostic signal tokens* (the same magnetics /
interferometer / soft-x-ray / Dα-boundary / … streams the joint world model
conditions on AND dreams) and *equilibrium labels*, and emits a geometry
prediction with a calibrated error budget.  It is NEVER imported into the WM
training / conditioning path.  Keep it physically downstream of the model — it
only consumes data and produces evaluator metrics.

Why a diagnostics probe (vs the camera probe next door)
-------------------------------------------------------
The camera-only oracle (:mod:`imas_ambix.worldmodel.equilibrium_probe`) FAILED
on the interior geometry — magnetic axis and X-point skill was ~0, because the
interior current distribution is not a camera observable.  But the MEASURED
magnetics (flux loops + B-field probes) are precisely the inputs an EFIT-class
reconstruction uses to *determine* the boundary / X-point, so a
diagnostics→equilibrium map should be feasible where the camera one was not.
The joint world model already dreams these diagnostics, so a feasible referee
here lets the controllability gate score the DREAMED diagnostics through this
same map.

What it is
----------
Each measured stream arrives as a ``(n_steps, n_channels)`` block of per-stream
LOCAL token ids (the dataset's :func:`read_window_signals` output).  Per stream
we:

  * embed the local ids with a per-stream embedding table (``vocab_s`` entries),
  * mean-pool the (small) channel axis after a per-stream channel projection so
    a stream's contribution is a fixed ``d_stream`` vector per time step,
  * concatenate the present streams' per-step vectors (a missing stream
    contributes a learned zero — handled by the caller omitting it and this
    module zero-filling its slot) into a per-step token,
  * run a tiny temporal Transformer / GRU over the ``n_steps`` tokens and pool,
  * feed a Gaussian head -> ``(mean, log_sigma)`` over the **standardised**
    12-D geometry target.

The probe operates in *standardised* target space (the caller supplies the
TRAIN-split per-component mean / std), so the heteroscedastic Gaussian NLL is
well-conditioned across the very different scales of axis_R (~0.8 m) and
axis_Z (~0 m).  De-standardisation back to metres happens at scoring time.

This module is forward + checkpoint IO only — no data loading, no training
loop (those live in the feasibility-oracle driver, which stays outside the WM
training path).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Minimum / maximum log-sigma the head emits (clamped for NLL stability).
LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 3.0


@dataclass(frozen=True)
class StreamSpec:
    """One measured conditioning stream the probe ingests.

    Attributes
    ----------
    name:
        Stream key (matches the dataset modality name, e.g. ``"magnetics"``).
    vocab:
        Local vocabulary size of the stream's token ids (its embedding rows).
    channels:
        Channel count the stream was probed at for the corpus (the spatial lane
        count of its ``(n_steps, channels)`` block).
    """

    name: str
    vocab: int
    channels: int


@dataclass
class DiagnosticsProbeConfig:
    """Architecture + IO configuration for :class:`DiagnosticsEquilibriumProbe`.

    Attributes
    ----------
    streams:
        The measured streams the probe fuses (name / vocab / channels).
    n_steps:
        Number of temporal positions each stream is resampled onto.
    target_dim:
        Geometry target dimensionality (12 for the standard label set).
    d_stream:
        Per-stream per-step embedding width (after channel pooling).
    d_model:
        Fused per-step token width fed to the temporal encoder.
    n_layers:
        Temporal Transformer-encoder layers.
    n_heads:
        Attention heads in the temporal encoder.
    head_hidden:
        Hidden width of the MLP head feeding the mean / log-sigma outputs.
    dropout:
        Dropout in the temporal encoder + head.
    """

    streams: list[StreamSpec] = field(default_factory=list)
    n_steps: int = 12
    target_dim: int = 12
    d_stream: int = 48
    d_model: int = 192
    n_layers: int = 3
    n_heads: int = 6
    head_hidden: int = 256
    dropout: float = 0.1


class _StreamEncoder(nn.Module):
    """Embed one stream's ``(B, n_steps, channels)`` ids -> ``(B, n_steps, d)``.

    Per local id is embedded (``vocab`` rows), then the channel axis is reduced
    by a learned per-channel weighting + mean-pool so the stream's per-step
    output is a fixed ``d_stream`` vector regardless of (capped) channel count.
    """

    def __init__(self, spec: StreamSpec, d_stream: int) -> None:
        super().__init__()
        self.name = spec.name
        self.channels = int(spec.channels)
        self.embed = nn.Embedding(int(spec.vocab), d_stream)
        # A per-channel linear mixing over the embedded channels: collapse the
        # channel axis with a learned projection then mean-pool, so the per-step
        # vector is channel-count independent and order-stable.
        self.proj = nn.Linear(d_stream, d_stream)
        self.norm = nn.LayerNorm(d_stream)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``(B, n_steps, channels) int64 -> (B, n_steps, d_stream)``."""
        emb = self.embed(ids)  # (B, S, C, d)
        emb = self.proj(emb)
        pooled = emb.mean(dim=2)  # mean-pool channels -> (B, S, d)
        return self.norm(pooled)


class DiagnosticsEquilibriumProbe(nn.Module):
    """Measured-diagnostic streams -> Gaussian head over 12-D geometry.

    Forward takes a dict ``{stream_name: (B, n_steps, channels) int64 ids}`` and
    returns ``(mean, log_sigma)``, each ``(B, target_dim)``, in STANDARDISED
    target space.  A stream absent from the dict contributes a learned zero (its
    fused slot is filled with zeros), so a window missing a stream is handled
    without a shape change.  Use :func:`gaussian_nll` for the training loss and
    :meth:`predict_metres` to map back to metres given the standardisation stats.
    """

    def __init__(self, config: DiagnosticsProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiagnosticsProbeConfig()
        cfg = self.config
        if not cfg.streams:
            raise ValueError("DiagnosticsProbeConfig.streams must be non-empty")

        self.encoders = nn.ModuleDict(
            {s.name: _StreamEncoder(s, cfg.d_stream) for s in cfg.streams}
        )
        self._stream_order = [s.name for s in cfg.streams]
        d_fused = cfg.d_stream * len(cfg.streams)

        # Project the concatenated per-stream per-step vectors to d_model and add
        # a learned positional embedding over the n_steps temporal axis.
        self.in_proj = nn.Linear(d_fused, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_steps, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.pool_norm = nn.LayerNorm(cfg.d_model)

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 2 * cfg.target_dim),
        )

    def forward(
        self, signals: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(mean, log_sigma)`` in standardised target space.

        Parameters
        ----------
        signals:
            ``{stream_name: (B, n_steps, channels) int64}`` local-id blocks for
            the present streams.  A configured stream missing from the dict
            contributes a zero vector for every step (handled here).

        Returns
        -------
        (mean, log_sigma): each ``(B, target_dim)``.
        """
        # Determine batch + device from any present stream.
        ref = next(iter(signals.values()))
        b = ref.shape[0]
        dev = ref.device
        cfg = self.config

        per_stream: list[torch.Tensor] = []
        for name in self._stream_order:
            if name in signals:
                per_stream.append(self.encoders[name](signals[name]))
            else:
                per_stream.append(torch.zeros(b, cfg.n_steps, cfg.d_stream, device=dev))
        fused = torch.cat(per_stream, dim=-1)  # (B, S, d_stream*n_streams)
        x = self.in_proj(fused) + self.pos  # (B, S, d_model)
        x = self.encoder(x)
        x = self.pool_norm(x.mean(dim=1))  # mean-pool over time
        out = self.head(x)
        mean, log_sigma = out.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mean, log_sigma

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    # -- prediction in physical units --------------------------------------

    def predict_metres(
        self,
        signals: dict[str, torch.Tensor],
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict ``(mean_m, sigma_m)`` in METRES (de-standardised).

        Parameters
        ----------
        signals:
            ``{stream_name: (B, n_steps, channels) int64}`` input batch.
        target_mean, target_std:
            ``(target_dim,)`` standardisation stats (the TRAIN-split mean / std
            used to standardise the labels).

        Returns
        -------
        (mean_m, sigma_m): each ``(B, target_dim)`` numpy arrays in metres.
        """
        self.eval()
        with torch.no_grad():
            mean, log_sigma = self.forward(signals)
        mu = mean.detach().cpu().float().numpy()
        sd = np.exp(log_sigma.detach().cpu().float().numpy())
        tmean = np.asarray(target_mean, dtype=np.float64)
        tstd = np.asarray(target_std, dtype=np.float64)
        mean_m = mu * tstd + tmean
        sigma_m = sd * tstd
        return mean_m.astype(np.float64), sigma_m.astype(np.float64)


def gaussian_nll(
    mean: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked heteroscedastic Gaussian negative log-likelihood.

    NLL per element = ``0.5 * (log(2π) + 2·log_sigma + ((y-μ)/σ)²)``, averaged
    over the *finite* (masked-in) elements only.  ``mask`` (``(B, D)`` bool /
    float) zeros out elements whose equilibrium label is undefined so masked
    targets never contribute a gradient.

    Returns a scalar tensor (mean over masked-in elements); zero if the batch
    has no finite labels.
    """
    m = mask.to(mean.dtype)
    inv_var = torch.exp(-2.0 * log_sigma)
    sq = (target - mean) ** 2
    nll = 0.5 * (np.log(2.0 * np.pi) + 2.0 * log_sigma + sq * inv_var)
    nll = nll * m
    denom = m.sum().clamp_min(1.0)
    return nll.sum() / denom


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------


def save_probe(
    path,
    model: DiagnosticsEquilibriumProbe,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    extra: dict | None = None,
) -> None:
    """Save the probe weights + config + standardisation stats to ``path``."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model.config)
    # dataclass list[StreamSpec] -> list[dict] for a plain torch.save payload.
    cfg["streams"] = [asdict(s) for s in model.config.streams]
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg,
        "target_mean": np.asarray(target_mean, dtype=np.float64),
        "target_std": np.asarray(target_std, dtype=np.float64),
        "extra": extra or {},
    }
    torch.save(payload, str(p))


def load_probe(path, *, map_location: str = "cpu"):
    """Load a probe saved by :func:`save_probe`.

    Returns ``(model, target_mean, target_std, extra)``.
    """
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg_d = dict(payload["config"])
    cfg_d["streams"] = [StreamSpec(**s) for s in cfg_d["streams"]]
    cfg = DiagnosticsProbeConfig(**cfg_d)
    model = DiagnosticsEquilibriumProbe(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return (
        model,
        np.asarray(payload["target_mean"], dtype=np.float64),
        np.asarray(payload["target_std"], dtype=np.float64),
        dict(payload.get("extra", {})),
    )


__all__ = [
    "LOG_SIGMA_MIN",
    "LOG_SIGMA_MAX",
    "StreamSpec",
    "DiagnosticsProbeConfig",
    "DiagnosticsEquilibriumProbe",
    "gaussian_nll",
    "save_probe",
    "load_probe",
]
