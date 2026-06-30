"""Shared space-time relational encoder primitives for diagnostic signals.

ONE geometry-aware, phase-preserving tokenisation + relational-attention scheme,
shared by the eval-side diagnostics->equilibrium probe
(:mod:`imas_ambix.worldmodel.diagnostics_equilibrium_probe`) and — in a later
step — the world-model input path, so every diagnostic stream is ingested the
SAME way: one token per ``(sensor, time-step)``, each carrying

  * a VALUE embedding (the local token id, OR a continuous signed value),
  * a PERIODIC (sin/cos) projection of the sensor GEOMETRY (R, Z, φ, orientation
    normal; angular features encoded on the circle so the 0/2π seam is
    continuous — a sensor at 2π−ε is ADJACENT to one at 0+ε), with a learned
    has-geometry flag,
  * a per-sensor-kind embedding,
  * (optional, continuous lane) a complex-STFT phase lift over the step axis,
    so cross-sensor PHASE reaches the attention (a rotating toroidal mode is a
    phase ramp across φ).

NO pooling over the channel or time axis: every ``(sensor, step)`` token reaches
the relational attention; aggregation is a learned query token (attention-pool).
This module holds the reusable, target-agnostic building blocks; the eval probe
composes them with a Gaussian geometry head, and the live path can import the
same primitives so the two never diverge.

Pure model code — no data loading, no IO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

#: Geometry feature width (R, Z, phi, angle_deg, normal_r, normal_z, chord_r1,
#: chord_z1, chord_r2, chord_z2) — matches
#: :data:`imas_ambix.gs.geometry_export.GEOMETRY_FEATURE_NAMES`.
N_GEOM_FEATURES = 10

#: Sensor-kind vocabulary the per-kind embedding switches on.  Index 0 is the
#: catch-all / unknown kind so an unmapped channel still embeds.  Extends
#: :data:`imas_ambix.gs.geometry_export.SENSOR_KINDS` with an explicit "unknown"
#: head slot + a device-global scalar (Ip) + a toroidal-saddle kind.
SENSOR_KIND_VOCAB: tuple[str, ...] = (
    "unknown",
    "bpol_probe",
    "flux_loop",
    "interferometer_chord",
    "sxr_chord",
    "pixel",
    "coil",
    "scalar",
    # A device-global SIGNED scalar with no spatial geometry — the plasma
    # current Ip (the single most important equilibrium scalar: it sets the
    # Grad-Shafranov source).  A dedicated kind so a head can give it a clear,
    # distinct feature slot rather than burying it among per-sensor scalars.
    "global_scalar",
    # A toroidal saddle loop — a toroidal-field pickup at a distinct toroidal
    # angle φ; the toroidal array that resolves the toroidal mode number.
    "toroidal_saddle",
)
_KIND_INDEX = {k: i for i, k in enumerate(SENSOR_KIND_VOCAB)}

#: Geometry feature-column indices that are ANGLES ON A CIRCLE — encoded as
#: ``(sin, cos)`` before the geometry projection so the 0/2π seam is continuous.
#: ``phi`` is the toroidal angle (radians); ``angle_deg`` is the B-probe
#: orientation (degrees) — both periodic.  The orientation NORMAL
#: (normal_r, normal_z) is already a periodic unit vector, so it is left as-is.
_PERIODIC_ANGLE_COLUMNS = (
    (2, False),  # phi   — radians
    (3, True),  # angle_deg — degrees (converted before sin/cos)
)

#: Feature width after the periodic (sin, cos) expansion (one extra col / angle).
_PERIODIC_GEOM_FEATURES = N_GEOM_FEATURES + len(_PERIODIC_ANGLE_COLUMNS)


def sensor_kind_index(kind: str) -> int:
    """Map a sensor-kind string to its embedding row (0 = unknown)."""
    return _KIND_INDEX.get(str(kind), 0)


@dataclass(frozen=True)
class StreamSpec:
    """One measured conditioning stream the encoder ingests.

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


def encode_periodic_geometry(geom: torch.Tensor) -> torch.Tensor:
    """Map a raw geometry feature row to a SEAM-CONTINUOUS feature vector.

    Every angular column (toroidal ``phi``, B-probe ``angle_deg``) is replaced
    by its ``(sin, cos)`` pair so angles on a circle are continuous across the
    0/2π seam: a sensor at φ = 2π−ε lands ADJACENT to one at φ = 0+ε (their
    encoded vectors are close), whereas φ = 0 and φ = π land far apart.  A linear
    angle-in-degrees encoding would do the opposite (359° and 1° maximally far).

    NaN angles encode to ``(0, 0)`` (distinct from any real unit-circle point);
    every non-angular column is passed through unchanged.  Input
    ``(..., N_GEOM_FEATURES)`` -> output ``(..., _PERIODIC_GEOM_FEATURES)``.
    """
    cols = []
    periodic = dict(_PERIODIC_ANGLE_COLUMNS)
    for c in range(geom.shape[-1]):
        x = geom[..., c : c + 1]
        if c in periodic:
            rad = x * (np.pi / 180.0) if periodic[c] else x
            finite = torch.isfinite(rad)
            rad = torch.where(finite, rad, torch.zeros_like(rad))
            s = torch.where(finite, torch.sin(rad), torch.zeros_like(rad))
            cs = torch.where(finite, torch.cos(rad), torch.zeros_like(rad))
            cols.extend([s, cs])
        else:
            cols.append(x)
    return torch.cat(cols, dim=-1)


class GeometryEncoder(nn.Module):
    """Project a per-sensor geometry feature row to the token width.

    Angular columns (toroidal φ, B-probe orientation) are expanded to a periodic
    ``(sin, cos)`` encoding (:func:`encode_periodic_geometry`) so the 0/2π seam
    is continuous, then NaN-filled with 0, a learned "has-geometry" flag is
    concatenated (so a geometry-free scalar token is distinguishable from a
    sensor at the origin), then a small MLP -> ``d_model``.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(_PERIODIC_GEOM_FEATURES + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        """``(..., N_GEOM_FEATURES) -> (..., d_model)``; NaN-safe + seam-aware."""
        finite_any = torch.isfinite(geom).any(dim=-1, keepdim=True).to(geom.dtype)
        enc = encode_periodic_geometry(geom)  # (..., _PERIODIC_GEOM_FEATURES)
        finite = torch.isfinite(enc)
        filled = torch.where(finite, enc, torch.zeros_like(enc))
        feat = torch.cat([filled, finite_any], dim=-1)
        return self.proj(feat)


class StreamTokeniser(nn.Module):
    """Build per-``(sensor, step)`` tokens for one stream.

    Token = value embedding (id-embed OR continuous-value projection) + geometry
    projection + per-stream embedding + per-sensor-kind embedding (+ an optional
    complex-STFT phase lift on the continuous lane).  The temporal positional
    embedding is added by the parent (shared across streams so the time axis is a
    single coordinate system).  Returns ``(B, n_steps, C, d)``.
    """

    def __init__(
        self,
        spec: StreamSpec,
        d_model: int,
        geom_encoder: GeometryEncoder,
        *,
        continuous_value: bool,
        n_steps: int,
        stft_phase: bool,
    ) -> None:
        super().__init__()
        self.name = spec.name
        self.channels = int(spec.channels)
        self.continuous_value = bool(continuous_value)
        # STFT phase lift only makes sense on the continuous (real-valued) lane.
        self.stft_phase = bool(stft_phase and continuous_value)
        if self.continuous_value:
            self.value_proj = nn.Linear(1, d_model)
        else:
            self.embed = nn.Embedding(int(spec.vocab), d_model)
        if self.stft_phase:
            n_bins = n_steps // 2 + 1
            self.stft_proj = nn.Linear(2 * n_bins, d_model)
        self.geom = geom_encoder
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        ids: torch.Tensor,
        geom: torch.Tensor,
        kind_emb: torch.Tensor,
        stream_emb: torch.Tensor,
        values: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compose ``(B, n_steps, C) -> (B, n_steps, C, d)`` tokens."""
        if self.continuous_value:
            v = values if values is not None else ids.to(torch.float32)
            v = v.to(torch.float32)
            tok = self.value_proj(v.unsqueeze(-1))
        else:
            tok = self.embed(ids)  # (B, S, C, d)
        g = self.geom(geom).unsqueeze(1)  # (B, 1, C, d) -> broadcast over time
        k = kind_emb.unsqueeze(1)  # (B, 1, C, d)
        tok = tok + g + k + stream_emb.view(1, 1, 1, -1)
        if self.stft_phase:
            # complex STFT over the step axis per sensor: rFFT keeps PHASE
            # (real + imag), so a toroidal mode's phase ramp across φ is visible.
            spec = torch.fft.rfft(v, dim=1)  # (B, n_bins, C) complex
            phase = torch.cat([spec.real, spec.imag], dim=1)  # (B, 2*n_bins, C)
            phase = phase.transpose(1, 2)  # (B, C, 2*n_bins)
            stft = self.stft_proj(phase).unsqueeze(1)  # (B, 1, C, d)
            tok = tok + stft
        return self.norm(tok)


__all__ = [
    "N_GEOM_FEATURES",
    "SENSOR_KIND_VOCAB",
    "sensor_kind_index",
    "StreamSpec",
    "encode_periodic_geometry",
    "GeometryEncoder",
    "StreamTokeniser",
]
