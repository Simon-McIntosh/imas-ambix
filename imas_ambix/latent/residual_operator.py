"""Static residual operator: sensor tokens → profile-DOF corrections.

The per-slice learner of the learned-equilibrium ladder's plumbing rung: a
small permutation-invariant encoder over geometry-encoded sensor tokens that
emits corrections on the physics-degenerate profile DOF only, decoded through
:class:`~imas_ambix.latent.profile_greens_decoder.ProfileGreensDecoder` (the
exact Green's layer).  Honest expectation on static information: parity with
the classical spine — the rung exists to prove the differentiable stack and
the training plumbing, so later temporal wins are attributable to the time
axis, not to machinery.

Every sensor token carries its own geometry (R, Z, orientation, kind) — the
machine-agnostic positional encoding — so a machine-varying sensor set is
ingestible without fixed-vector assumptions.  Inputs are firewall-clean by
construction: raw magnetics, the classical solve's own prediction, Ip and
n_e.  EFIT never enters.

Label shards (one NPZ per shot, written by ``scripts/spine_label_factory.py``)
are loaded here so the training and gate-eval drivers share one featurizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class LabelShard:
    """One shot's spine-manufactured labels + raw payload snapshot."""

    shot: int
    meta: dict
    arrays: dict[str, np.ndarray]

    @property
    def n_slices(self) -> int:
        return int(self.arrays["i_cell"].shape[0])


def load_label_shards(paths: list[Path] | list[str]) -> list[LabelShard]:
    """Load label NPZ shards (+ sidecar provenance JSON) written by the factory."""
    shards: list[LabelShard] = []
    for p in sorted(Path(q) for q in paths):
        meta_path = p.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        shards.append(
            LabelShard(shot=int(meta.get("shot", -1)), meta=meta, arrays=arrays)
        )
    return shards


#: per-token feature order (documented for the checkpoint contract)
TOKEN_FEATURES = (
    "plasma_signature_whitened",  # (measured − vacuum) / scale, masked → 0
    "spine_residual_whitened",  # (measured − spine prediction) / scale, masked → 0
    "measured_mask",
    "is_flux_loop",
    "sensor_r",
    "sensor_z",
    "orientation_cos",
    "orientation_sin",
)


def slice_tokens(
    measured: np.ndarray,
    vacuum: np.ndarray,
    spine_pred: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
    sr: np.ndarray,
    sz: np.ndarray,
    sang_deg: np.ndarray,
    is_flux: np.ndarray,
    *,
    clip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one slice's ``(S, 8)`` sensor tokens + ``(S,)`` token mask.

    Whitened channels are clipped at ``clip`` robust-σ so a single outlier
    channel cannot dominate the pooled code (mirrors the solve's own
    heavy-tail handling).
    """
    s = np.clip(np.asarray(scale, dtype=np.float64), 1e-12, None)
    m = np.asarray(mask, dtype=bool) & np.isfinite(measured)
    meas = np.nan_to_num(np.asarray(measured, dtype=np.float64))
    x = np.clip((meas - vacuum) / s, -clip, clip)
    r = np.clip((meas - spine_pred) / s, -clip, clip)
    ang = np.deg2rad(np.asarray(sang_deg, dtype=np.float64))
    tokens = np.column_stack(
        [
            np.where(m, x, 0.0),
            np.where(m, r, 0.0),
            m.astype(np.float64),
            np.asarray(is_flux, dtype=np.float64),
            sr,
            sz,
            np.cos(ang),
            np.sin(ang),
        ]
    ).astype(np.float32)
    return tokens, m


def slice_globals(ip_amperes: float, n_e: float) -> np.ndarray:
    """(2,) firewall-safe global scalars, roughly unit-scaled."""
    ne = float(n_e) if np.isfinite(n_e) else 0.0
    return np.array([float(ip_amperes) / 1e6, ne / 1e20], dtype=np.float32)


class ResidualOperator(nn.Module):
    """Masked-pool sensor-token encoder → profile-DOF corrections.

    Zero-initialised output head: the untrained operator is the identity on
    the classical spine (``dc = 0``), so training starts exactly at parity.
    """

    def __init__(
        self,
        n_dof: int,
        *,
        token_dim: int = len(TOKEN_FEATURES),
        n_global: int = 2,
        width: int = 96,
        dc_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_dof = int(n_dof)
        self.dc_scale = float(dc_scale)
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width + n_global, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, n_dof),
        )
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        tokens: torch.Tensor,  # (B, S, F)
        token_mask: torch.Tensor,  # (B, S) bool — measured sensors
        global_feats: torch.Tensor,  # (B, n_global)
    ) -> torch.Tensor:
        h = self.token_mlp(tokens)  # (B, S, W)
        w = token_mask.to(h.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1.0)
        mean_pool = (h * w).sum(dim=1) / denom
        max_pool = torch.where(w > 0, h, h.new_full((), -1e30)).amax(dim=1)
        max_pool = torch.where(denom > 0, max_pool, torch.zeros_like(max_pool))
        code = torch.cat([mean_pool, max_pool, global_feats], dim=-1)
        return self.dc_scale * torch.tanh(self.head(code))


def save_checkpoint(path: Path | str, model: ResidualOperator, extra: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_dof": model.n_dof,
            "dc_scale": model.dc_scale,
            "token_features": list(TOKEN_FEATURES),
            **extra,
        },
        path,
    )


def load_checkpoint(path: Path | str) -> tuple[ResidualOperator, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualOperator(int(ckpt["n_dof"]), dc_scale=float(ckpt["dc_scale"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


__all__ = [
    "TOKEN_FEATURES",
    "LabelShard",
    "ResidualOperator",
    "load_checkpoint",
    "load_label_shards",
    "save_checkpoint",
    "slice_globals",
    "slice_tokens",
]
