"""Space-time probe: measured diagnostics -> plasma equilibrium geometry.

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

Why SPACE-TIME (not channel mean-pool)
--------------------------------------
The boundary is determined by the SPATIAL PATTERN across the sensor array —
which flux loop reads high relative to its neighbours, the gradient of B_z down
the centre-column ``ccbv`` chain — and by how that pattern EVOLVES.  A probe
that mean-pools the 94 magnetic sensors into one vector per step throws that
pattern away (the previous tokenised probe reached only axis_R skill ~0.05 that
way, far below the raw-float ceiling ~0.57).  So this probe instead builds one
token PER ``(stream, sensor, time-step)`` and lets a transformer attend over the
FULL ``(sensor × time)`` set — between-sensor (spatial) AND across-time
(temporal, incl. sensor phasing) — with each sensor's GEOMETRY as its positional
encoding (the camera lesson — ``16×16`` spatial position made cameras decodable
— generalised to every diagnostic).  No pooling over the channel or time axis;
aggregation is a learned attention-pool query, not a mean.

Token composition
-----------------
Each ``(stream, sensor, step)`` token sums:

  * a VALUE embedding — either the local token id (256-bin quantised) through a
    per-stream embedding table, OR (ablation) a projection of the CONTINUOUS
    standardised value, to test whether quantisation is the remaining ceiling;
  * a projection of the sensor GEOMETRY feature vector (R, Z, phi, orientation
    normal; NaN -> 0 with a learned "has-geometry" flag) — the positional code;
  * a temporal positional embedding (the step index);
  * a learned per-stream-type embedding;
  * a learned per-sensor-kind embedding.

A few MACHINE-geometry context tokens (vessel-contour extent + PF-coil R/Z) give
the model the machine frame.  A learned query token attention-pools the encoded
set into the Gaussian head over the 12-D target, in STANDARDISED target space
(the caller supplies the TRAIN-split per-component mean / std); de-standardisation
back to metres happens at scoring time.

This module is forward + checkpoint IO only — no data loading, no training loop
(those live in the feasibility-oracle driver, which stays outside the WM
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

#: Geometry feature width (R, Z, phi, angle_deg, normal_r, normal_z, chord_r1,
#: chord_z1, chord_r2, chord_z2) — matches
#: :data:`imas_ambix.gs.geometry_export.GEOMETRY_FEATURE_NAMES`.
N_GEOM_FEATURES = 10

#: Sensor-kind vocabulary the per-kind embedding switches on.  Index 0 is the
#: catch-all / unknown kind so an unmapped channel still embeds.  Order matches
#: :data:`imas_ambix.gs.geometry_export.SENSOR_KINDS` with an explicit "unknown"
#: head slot.
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
    # Grad-Shafranov source).  A dedicated kind so the head can give it a clear,
    # distinct feature slot rather than burying it among per-sensor scalars.
    "global_scalar",
)
_KIND_INDEX = {k: i for i, k in enumerate(SENSOR_KIND_VOCAB)}


def sensor_kind_index(kind: str) -> int:
    """Map a sensor-kind string to its embedding row (0 = unknown)."""
    return _KIND_INDEX.get(str(kind), 0)


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
    d_model:
        Per-token width fed to the space-time encoder.
    n_layers:
        Transformer-encoder layers attending over the (sensor × time) set.
    n_heads:
        Attention heads.
    head_hidden:
        Hidden width of the MLP head feeding the mean / log-sigma outputs.
    dropout:
        Dropout in the encoder + head.
    continuous_value:
        When True, embed the CONTINUOUS standardised per-sensor value (a 1-D
        linear projection) instead of the 256-bin token id — the ablation lever
        that tests whether quantisation is the remaining ceiling.  When False
        (default) the per-stream id embedding table is used.
    use_machine_tokens:
        When True, prepend a few machine-geometry context tokens (vessel-contour
        extent + PF-coil R/Z) so the model carries the machine frame.
    max_machine_tokens:
        Cap on the machine-geometry context tokens (informational; the caller
        builds the machine block, this only bounds expectations).
    """

    streams: list[StreamSpec] = field(default_factory=list)
    n_steps: int = 12
    target_dim: int = 12
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    head_hidden: int = 256
    dropout: float = 0.1
    continuous_value: bool = False
    use_machine_tokens: bool = True
    max_machine_tokens: int = 8


class _GeometryEncoder(nn.Module):
    """Project a per-sensor geometry feature row to the token width.

    NaN-fills with 0 and concatenates a learned "has-geometry" flag (so a
    geometry-free scalar token is distinguishable from a sensor that happens to
    sit at the origin), then a small MLP -> ``d_model``.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(N_GEOM_FEATURES + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        """``(..., N_GEOM_FEATURES) -> (..., d_model)``; NaN-safe."""
        finite = torch.isfinite(geom)
        has_geom = finite.any(dim=-1, keepdim=True).to(geom.dtype)
        filled = torch.where(finite, geom, torch.zeros_like(geom))
        feat = torch.cat([filled, has_geom], dim=-1)
        return self.proj(feat)


class _StreamTokeniser(nn.Module):
    """Build per-``(sensor, step)`` tokens for one stream.

    Token = value embedding (id-embed OR continuous-value projection) + geometry
    projection + per-stream embedding + per-sensor-kind embedding.  The temporal
    positional embedding is added by the parent (shared across streams so the
    time axis is a single coordinate system).  Returns ``(B, n_steps, C, d)``.
    """

    def __init__(
        self,
        spec: StreamSpec,
        d_model: int,
        geom_encoder: _GeometryEncoder,
        *,
        continuous_value: bool,
    ) -> None:
        super().__init__()
        self.name = spec.name
        self.channels = int(spec.channels)
        self.continuous_value = bool(continuous_value)
        if self.continuous_value:
            # one scalar standardised value -> d_model (the ablation path).
            self.value_proj = nn.Linear(1, d_model)
        else:
            self.embed = nn.Embedding(int(spec.vocab), d_model)
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
        """Compose ``(B, n_steps, C) -> (B, n_steps, C, d)`` tokens.

        Parameters
        ----------
        ids:
            ``(B, n_steps, C)`` int64 local token ids.
        geom:
            ``(B, C, N_GEOM_FEATURES)`` per-sensor geometry (broadcast over time).
        kind_emb:
            ``(B, C, d)`` per-sensor-kind embedding (broadcast over time).
        stream_emb:
            ``(d,)`` learned per-stream-type embedding.
        values:
            ``(B, n_steps, C)`` continuous standardised values (only used when
            ``continuous_value``; else ignored).
        """
        if self.continuous_value:
            v = values if values is not None else ids.to(torch.float32)
            tok = self.value_proj(v.unsqueeze(-1).to(torch.float32))
        else:
            tok = self.embed(ids)  # (B, S, C, d)
        g = self.geom(geom).unsqueeze(1)  # (B, 1, C, d) -> broadcast over time
        k = kind_emb.unsqueeze(1)  # (B, 1, C, d)
        tok = tok + g + k + stream_emb.view(1, 1, 1, -1)
        return self.norm(tok)


class DiagnosticsEquilibriumProbe(nn.Module):
    """Space-time transformer over (sensor × time) tokens -> 12-D geometry head.

    Forward takes a dict ``{stream_name: (B, n_steps, channels) int64 ids}`` plus
    a parallel dict of per-stream geometry / sensor-kind blocks, and returns
    ``(mean, log_sigma)``, each ``(B, target_dim)``, in STANDARDISED target
    space.  A configured stream absent from the ids dict contributes nothing
    (its tokens are simply not emitted), so a window missing a stream is handled
    without a shape change.  Use :func:`gaussian_nll` for the training loss and
    :meth:`predict_metres` to map back to metres given the standardisation stats.

    No pooling over the channel or time axis: every ``(sensor, step)`` token
    reaches attention, and aggregation is a learned query token (attention-pool),
    not a mean.
    """

    def __init__(self, config: DiagnosticsProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiagnosticsProbeConfig()
        cfg = self.config
        if not cfg.streams:
            raise ValueError("DiagnosticsProbeConfig.streams must be non-empty")

        self._stream_order = [s.name for s in cfg.streams]
        self._stream_index = {s.name: i for i, s in enumerate(cfg.streams)}

        self.geom_encoder = _GeometryEncoder(cfg.d_model)
        self.tokenisers = nn.ModuleDict(
            {
                s.name: _StreamTokeniser(
                    s,
                    cfg.d_model,
                    self.geom_encoder,
                    continuous_value=cfg.continuous_value,
                )
                for s in cfg.streams
            }
        )
        # learned per-stream-type + per-sensor-kind embeddings.
        self.stream_embed = nn.Embedding(len(cfg.streams), cfg.d_model)
        self.kind_embed = nn.Embedding(len(SENSOR_KIND_VOCAB), cfg.d_model)
        # temporal positional embedding over the n_steps axis.
        self.time_pos = nn.Parameter(torch.zeros(cfg.n_steps, cfg.d_model))
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        # machine-geometry context tokens: each carries a (R, Z) point + a flag
        # bit (vessel-extent corner vs PF-coil), through a small projection plus
        # a learned machine-token marker.
        self.use_machine_tokens = bool(cfg.use_machine_tokens)
        if self.use_machine_tokens:
            self.machine_proj = nn.Sequential(
                nn.Linear(3, cfg.d_model),  # (R, Z, is_coil)
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.machine_marker = nn.Parameter(torch.zeros(cfg.d_model))
            nn.init.trunc_normal_(self.machine_marker, std=0.02)

        # learned query token for attention-pool aggregation.
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.in_norm = nn.LayerNorm(cfg.d_model)
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
        self,
        signals: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor],
        sensor_kinds: dict[str, torch.Tensor],
        *,
        values: dict[str, torch.Tensor] | None = None,
        machine: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(mean, log_sigma)`` in standardised target space.

        Parameters
        ----------
        signals:
            ``{stream_name: (B, n_steps, channels) int64}`` local-id blocks for
            the present streams.
        geometry:
            ``{stream_name: (B, channels, N_GEOM_FEATURES) float32}`` per-sensor
            geometry aligned 1:1 with the stream's channel order (NaN where a
            channel has no known geometry).
        sensor_kinds:
            ``{stream_name: (B, channels) int64}`` per-sensor kind indices (see
            :func:`sensor_kind_index`).
        values:
            ``{stream_name: (B, n_steps, channels) float32}`` continuous
            standardised values, used only when the config sets
            ``continuous_value`` (the quantisation ablation).
        machine:
            ``(B, M, 3)`` machine-geometry context points ``(R, Z, is_coil)``,
            optional; ignored when machine tokens are disabled.

        Returns
        -------
        (mean, log_sigma): each ``(B, target_dim)``.
        """
        cfg = self.config
        present = [n for n in self._stream_order if n in signals]
        if not present:
            raise ValueError("forward: no present streams in signals dict")
        ref = signals[present[0]]
        b = ref.shape[0]
        dev = ref.device

        seq: list[torch.Tensor] = []
        for name in present:
            ids = signals[name]  # (B, S, C)
            geom = geometry[name].to(dev)  # (B, C, G)
            kinds = sensor_kinds[name].to(dev)  # (B, C)
            kind_emb = self.kind_embed(kinds)  # (B, C, d)
            stream_emb = self.stream_embed.weight[self._stream_index[name]]
            vals = values.get(name) if values is not None else None
            tok = self.tokenisers[name](ids, geom, kind_emb, stream_emb, vals)
            # add the shared temporal positional embedding over the step axis.
            n_s = tok.shape[1]
            tok = tok + self.time_pos[:n_s].view(1, n_s, 1, -1)
            # flatten (sensor, time) -> one sequence of tokens.
            tok = tok.reshape(b, n_s * tok.shape[2], cfg.d_model)
            seq.append(tok)

        x = torch.cat(seq, dim=1)  # (B, total_tokens, d)

        if self.use_machine_tokens and machine is not None and machine.shape[1] > 0:
            m = self.machine_proj(machine.to(dev).to(x.dtype))
            m = m + self.machine_marker.view(1, 1, -1)
            x = torch.cat([m, x], dim=1)

        # prepend the learned attention-pool query token.
        q = self.query.expand(b, -1, -1).to(x.dtype)
        x = torch.cat([q, x], dim=1)
        x = self.in_norm(x)
        x = self.encoder(x)
        pooled = self.pool_norm(x[:, 0])  # the query token's encoded state
        out = self.head(pooled)
        mean, log_sigma = out.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        return mean, log_sigma

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    # -- prediction in physical units --------------------------------------

    def predict_metres(
        self,
        signals: dict[str, torch.Tensor],
        geometry: dict[str, torch.Tensor],
        sensor_kinds: dict[str, torch.Tensor],
        target_mean: np.ndarray,
        target_std: np.ndarray,
        *,
        values: dict[str, torch.Tensor] | None = None,
        machine: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict ``(mean_m, sigma_m)`` in METRES (de-standardised).

        Parameters mirror :meth:`forward`; ``target_mean`` / ``target_std`` are
        the ``(target_dim,)`` TRAIN-split standardisation stats.
        """
        self.eval()
        with torch.no_grad():
            mean, log_sigma = self.forward(
                signals,
                geometry,
                sensor_kinds,
                values=values,
                machine=machine,
            )
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
    "N_GEOM_FEATURES",
    "SENSOR_KIND_VOCAB",
    "sensor_kind_index",
    "StreamSpec",
    "DiagnosticsProbeConfig",
    "DiagnosticsEquilibriumProbe",
    "gaussian_nll",
    "save_probe",
    "load_probe",
]
