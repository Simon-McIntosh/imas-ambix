"""Heteroscedastic CNN probe: decoded camera frames -> plasma geometry.

EVALUATOR-ONLY (binding firewall)
---------------------------------
This probe is part of the **independent equilibrium evaluator**, never the
world model.  It reads *decoded camera pixels* and *equilibrium labels* and
emits a geometry prediction with a calibrated error budget; it is NEVER
imported into the WM training / conditioning path.  Keep it physically
downstream of the model.

What it is
----------
A small (~2-5M-parameter) convolutional network over a ``k``-frame grayscale
stack (channel-stacked, ``k`` ~ 3-5) of decoded rbb frames at 256x256.  It
predicts a Gaussian over the **standardised** 12-D geometry target
(:mod:`imas_ambix.worldmodel.equilibrium_labels`): a mean and a log-sigma per
component.  Training under the heteroscedastic Gaussian negative-log-likelihood
makes the per-component error budget *native* — the predicted ``sigma`` is the
model's own calibrated uncertainty, and the de-standardised RMSE in metres is
the feasibility metric.

The probe operates in *standardised* target space (zero-mean / unit-variance
per component, stats supplied by the caller from the TRAIN split) so the NLL is
well-conditioned across the very different scales of axis_R (~0.8 m) and
axis_Z (~0 m).  De-standardisation back to metres happens at scoring time.

This module is forward + checkpoint IO only — no data loading, no training
loop (those live in the feasibility-oracle driver, which must stay outside the
WM training path).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Minimum / maximum log-sigma the head emits (clamped for NLL stability).
LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 3.0


@dataclass
class ProbeConfig:
    """Architecture + IO configuration for :class:`EquilibriumProbe`.

    Attributes
    ----------
    in_frames:
        Number of channel-stacked grayscale frames (``k``).
    image_size:
        Square decoded-frame side length (256 for the Open-MAGVIT2 decoder).
    target_dim:
        Geometry target dimensionality (12 for the standard label set).
    width:
        Base channel width; the conv stack doubles it per downsample stage.
    n_stages:
        Number of stride-2 downsample stages.
    head_hidden:
        Hidden width of the MLP head feeding the mean / log-sigma outputs.
    """

    in_frames: int = 4
    image_size: int = 256
    target_dim: int = 12
    width: int = 32
    n_stages: int = 5
    head_hidden: int = 256


class _ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, optionally stride-2 downsampling."""

    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        # GroupNorm (batch-size independent — robust for tiny eval batches).
        groups = max(1, min(8, c_out // 8))
        self.norm = nn.GroupNorm(groups, c_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class EquilibriumProbe(nn.Module):
    """Small CNN -> Gaussian head over the standardised 12-D geometry target.

    Forward returns ``(mean, log_sigma)``, each ``(B, target_dim)``, in
    STANDARDISED target space.  Use :func:`gaussian_nll` for the training loss
    and :meth:`predict_metres` to map back to metres given the standardisation
    stats.
    """

    def __init__(self, config: ProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProbeConfig()
        cfg = self.config

        chans = [cfg.in_frames]
        c = cfg.width
        stages: list[nn.Module] = []
        # Stem (stride 1) then n_stages stride-2 downsamples; width doubles
        # each stage and is capped so the param count stays in the ~2-5M band.
        stages.append(_ConvBlock(cfg.in_frames, c, stride=1))
        chans.append(c)
        for _ in range(cfg.n_stages):
            c_next = min(c * 2, 256)
            stages.append(_ConvBlock(c, c_next, stride=2))
            c = c_next
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(c, cfg.head_hidden),
            nn.SiLU(),
            nn.Linear(cfg.head_hidden, 2 * cfg.target_dim),
        )

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(mean, log_sigma)`` in standardised target space.

        Parameters
        ----------
        frames:
            ``(B, k, H, W)`` float tensor — ``k`` channel-stacked grayscale
            frames, values typically in ``[0, 1]``.

        Returns
        -------
        (mean, log_sigma): each ``(B, target_dim)``.
        """
        x = self.features(frames)
        x = self.pool(x).flatten(1)
        out = self.head(x)
        mean, log_sigma = out.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mean, log_sigma

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    # -- prediction in physical units --------------------------------------

    def predict_metres(
        self,
        frames: torch.Tensor,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict ``(mean_m, sigma_m)`` in METRES (de-standardised).

        Parameters
        ----------
        frames:
            ``(B, k, H, W)`` input batch.
        target_mean, target_std:
            ``(target_dim,)`` standardisation stats (the TRAIN-split mean / std
            used to standardise the labels).

        Returns
        -------
        (mean_m, sigma_m): each ``(B, target_dim)`` numpy arrays in metres.
        """
        self.eval()
        with torch.no_grad():
            mean, log_sigma = self.forward(frames)
        mu = mean.detach().cpu().numpy()
        sd = np.exp(log_sigma.detach().cpu().numpy())
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
    frames never contribute a gradient.

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
    model: EquilibriumProbe,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    extra: dict | None = None,
) -> None:
    """Save the probe weights + config + standardisation stats to ``path``."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(model.config),
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
    cfg = ProbeConfig(**payload["config"])
    model = EquilibriumProbe(cfg)
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
    "ProbeConfig",
    "EquilibriumProbe",
    "gaussian_nll",
    "save_probe",
    "load_probe",
]
