"""Independent per-frame corruption of the camera-HISTORY EMBEDDINGS.

Why this exists (the corrected controllability lever)
-----------------------------------------------------
The M4 controllability gate showed the camera world model IGNORES its actuator
plan: zeroing the whole drive moved the prediction by ~0.  The deepest reason is
information-theoretic (Schmidt & Jiang 2026, the latent-action-collapse theorem):
when the PREDICTABLE component of the dynamics (the camera history) dominates the
CONTROLLABLE component (the plan), the action signal collapses and is ignored
**regardless of conditioning architecture** — the past frames alone suffice to
predict the next, so the plan carries no marginal information and the model
routes everything through the history.

The fix the literature converges on is to BOTTLENECK the history so the past
frames no longer suffice: corrupt the camera-history tokens with independent
per-frame noise (Diffusion Forcing, Chen et al. NeurIPS 2024; History-Guided
Video Diffusion, Song et al. 2026 — "noise the history to reduce over-reliance").
With the history made unreliable, the only way to lower the loss on the forecast
frames is to LEAN on the conditioning (the actuator plan), so the plan must carry
what the corrupted history cannot — the precondition for the controls becoming
load-bearing.

Critical correction over the prior gate
---------------------------------------
The prior M4 gate's 80% dropout was on the OBSERVATION SIGNALS, and
:mod:`imas_ambix.worldmodel.context_corruption` replaces a fraction of context
TOKEN IDS.  Neither bottlenecked the past CAMERA FRAMES reaching the dynamics
head: the model still saw a near-clean camera history (token-id replacement on a
2**18 codebook is a weak corruption — a replaced id still embeds to a generic
vector, and at low rates most of the frame survives) and coasted on it.  This
module corrupts the camera-history **embeddings** directly — the continuous
vectors the temporal attention actually reads — which is the channel the survey
identifies, and does it INDEPENDENTLY PER FRAME (Diffusion Forcing) rather than
with one shared rate, so a deep history frame can be heavily corrupted while a
recent one is light, breaking the smooth-extrapolation shortcut at every lag.

What it does
------------
Given the camera-frame embeddings ``(B, T, S, d)`` and the number of CONTEXT
frames (the ones the rollout re-feeds itself), for each (sample, context-frame)
it draws an independent corruption strength and:

* **additive Gaussian noise** scaled by that strength (continuous noising of the
  embedding — the discrete-token analogue would lose the graded control), and
* optionally **stochastically masks** the frame embedding toward zero (a
  Bernoulli drop of the whole frame's information at high strength), so the model
  also sees histories with entire frames missing — the strongest bottleneck.

The forecast-window frames (``>= context_frames``) are NEVER touched — they are
what the model predicts, and corrupting them would just add label noise.  At
inference the strength is 0 (clean history, "trust the context"), a regime the
sampler always includes so it stays well-trained.

The per-frame strength is also quantised to a small bin and returned, so the
model can CONDITION on "how corrupt is my history at each lag" via the existing
learned per-level embedding (:class:`SignalSpacetimeConfig.corruption_levels`),
exactly as the M2 anti-drift recipe conditions on a single history-corruption
level — here generalised to a per-frame level.

All operations are deterministic given a ``torch.Generator`` so a training step
is reproducible and one DDP rank's corruption is independent of another's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HistoryBottleneckConfig:
    """Knobs for the per-frame camera-history embedding bottleneck.

    Attributes
    ----------
    noise_std:
        Standard deviation of the additive Gaussian noise at FULL strength,
        in units of the per-(sample, frame) embedding RMS — so the corruption is
        scale-invariant to the embedding magnitude (a frame whose embedding has
        RMS ``r`` gets noise of std ``strength * noise_std * r``).  0 disables the
        additive-noise leg.
    mask_prob:
        Probability, at FULL strength, of additionally masking (scaling toward
        zero by ``mask_scale``) a whole context-frame's embedding — the strongest
        bottleneck (the frame's information is largely removed, not just noised).
        The per-frame mask draw is scaled by that frame's strength so light frames
        are rarely masked.  0 disables the masking leg.
    mask_scale:
        Residual fraction of a masked frame's embedding kept (0 = fully zeroed,
        1 = no masking).  A small positive value (e.g. 0.0) zeroes the frame's
        value content while the spatial / temporal / marker positional adds
        applied AFTER this module still locate it on the grid.
    min_strength, max_strength:
        Per-frame strength is drawn uniformly in ``[min_strength, max_strength]``
        for the corrupted fraction of samples; ``clean_fraction`` of samples are
        forced to strength 0 (a fully clean history) so the clean regime — the one
        the first rollout frames are in and inference uses — stays well-trained.
    clean_fraction:
        Fraction of samples per step whose ENTIRE history is left clean
        (strength 0 on every frame).
    independent_per_frame:
        When True (Diffusion Forcing), each context frame draws its OWN strength;
        when False, all context frames of a sample share one strength (the M2
        single-level recipe).  True is the lever the survey prescribes.
    levels:
        Number of discrete strength bins the per-frame level is quantised to (for
        the learned per-level conditioning embedding).  Bin 0 = clean.
    """

    noise_std: float = 1.0
    mask_prob: float = 0.5
    mask_scale: float = 0.0
    min_strength: float = 0.0
    max_strength: float = 1.0
    clean_fraction: float = 0.2
    independent_per_frame: bool = True
    levels: int = 8

    @property
    def enabled(self) -> bool:
        return self.max_strength > 0.0 and (
            self.noise_std > 0.0 or self.mask_prob > 0.0
        )

    def strength_to_bin(self, strength: float) -> int:
        """Map a per-frame strength in ``[0, max_strength]`` to a bin ``[0, levels)``.

        Strength exactly 0 is bin 0 (clean); a positive strength falls in one of
        the ``levels - 1`` equal-width sub-bins partitioning ``(0, max_strength]``
        so the conditioning embedding row is a monotone code of the strength.
        """
        if self.levels <= 1 or strength <= 0.0 or self.max_strength <= 0.0:
            return 0
        frac = min(max(strength / self.max_strength, 0.0), 1.0)
        b = 1 + int(frac * (self.levels - 1 - 1e-9))
        return int(min(max(b, 1), self.levels - 1))


def sample_frame_strengths(
    batch_size: int,
    context_frames: int,
    cfg: HistoryBottleneckConfig,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw per-(sample, context-frame) corruption strengths + their bin indices.

    Returns ``(strengths (B, ctx) float, bins (B, ctx) long)`` on ``device``.  A
    ``clean_fraction`` of samples get strength 0 on EVERY frame; the rest get a
    per-frame strength (independent across frames when ``independent_per_frame``)
    drawn in ``[min_strength, max_strength]``.  ``context_frames`` <= 0 returns
    empty ``(B, 0)`` tensors.
    """
    dev = device or torch.device("cpu")
    ctx = int(max(0, context_frames))
    if ctx == 0 or not cfg.enabled:
        return (
            torch.zeros(batch_size, ctx, device=dev),
            torch.zeros(batch_size, ctx, dtype=torch.long, device=dev),
        )
    # per-frame strength (or one per sample broadcast to all frames).
    if cfg.independent_per_frame:
        u = torch.rand(batch_size, ctx, generator=generator, device=dev)
    else:
        u = torch.rand(batch_size, 1, generator=generator, device=dev).expand(
            batch_size, ctx
        )
    span = cfg.max_strength - cfg.min_strength
    strengths = cfg.min_strength + u * span
    # force a clean fraction of samples to strength 0 on all frames.
    clean_draw = torch.rand(batch_size, 1, generator=generator, device=dev)
    is_clean = clean_draw < cfg.clean_fraction  # (B, 1)
    strengths = torch.where(is_clean, torch.zeros_like(strengths), strengths)
    strengths = strengths.contiguous()
    # quantise to bins (host loop over the small (B, ctx) grid).
    flat = strengths.reshape(-1).tolist()
    bins = torch.tensor(
        [cfg.strength_to_bin(float(s)) for s in flat],
        dtype=torch.long,
        device=dev,
    ).view(batch_size, ctx)
    return strengths, bins


def bottleneck_history_embeddings(
    cam: torch.Tensor,
    strengths: torch.Tensor,
    cfg: HistoryBottleneckConfig,
    *,
    context_frames: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Corrupt the CONTEXT camera-frame EMBEDDINGS per their per-frame strength.

    ``cam`` is ``(B, T, S, d)`` camera-frame embeddings; ``strengths`` is
    ``(B, ctx)`` per-(sample, context-frame) strength in ``[0, max_strength]``.
    Only the first ``context_frames`` frames are corrupted (the history the
    rollout re-feeds itself); the forecast frames are returned untouched so the
    prediction target carries no added noise.

    For each context frame, at its strength ``g``:

    * additive Gaussian noise of std ``g * noise_std * rms`` is added, where
      ``rms`` is that frame's per-(sample) embedding RMS (so the noise scales with
      the embedding magnitude — corruption is scale-invariant), and
    * with probability ``g * mask_prob`` the whole frame's embedding is scaled by
      ``mask_scale`` toward zero (the masking leg) — drawn per (sample, frame).

    Returns a NEW tensor (``cam`` is not modified in place).  When the bottleneck
    is disabled or there are no context frames the input is returned unchanged.
    """
    if not cfg.enabled or context_frames < 1 or cam.ndim != 4:
        return cam
    b, t, s, d = cam.shape
    ctx = int(min(context_frames, t))
    if ctx < 1 or strengths.numel() == 0:
        return cam
    g = strengths[:, :ctx].to(cam.device, cam.dtype).clamp_min(0.0)  # (B, ctx)
    out = cam.clone()
    ctx_emb = out[:, :ctx]  # (B, ctx, S, d) view into the clone

    # additive Gaussian noise scaled by per-frame strength and per-frame RMS.
    if cfg.noise_std > 0.0:
        # per-(sample, frame) RMS over (S, d), kept as (B, ctx, 1, 1).
        rms = (
            ctx_emb.detach()
            .float()
            .pow(2)
            .mean(dim=(2, 3), keepdim=True)
            .clamp_min(1e-12)
            .sqrt()
            .to(cam.dtype)
        )
        noise = torch.randn(
            (b, ctx, s, d), generator=generator, device=cam.device, dtype=cam.dtype
        )
        scale = (g * float(cfg.noise_std)).view(b, ctx, 1, 1)
        ctx_emb = ctx_emb + noise * scale * rms

    # stochastic whole-frame masking toward zero.
    if cfg.mask_prob > 0.0:
        mdraw = torch.rand(
            (b, ctx), generator=generator, device=cam.device, dtype=torch.float32
        )
        # mask probability scales with the frame's strength: light frames rarely
        # masked, full-strength frames masked with prob mask_prob.
        thresh = (g.float() * float(cfg.mask_prob)).clamp(0.0, 1.0)
        masked = (mdraw < thresh).view(b, ctx, 1, 1)
        keep = torch.where(
            masked,
            torch.full_like(masked, float(cfg.mask_scale), dtype=cam.dtype),
            torch.ones_like(masked, dtype=cam.dtype),
        )
        ctx_emb = ctx_emb * keep

    out = out.clone()
    out[:, :ctx] = ctx_emb
    return out


__all__ = [
    "HistoryBottleneckConfig",
    "bottleneck_history_embeddings",
    "sample_frame_strengths",
]
