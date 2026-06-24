"""Compact recurrent latent-dynamics world model (command-conditioned, token-decode).

Why this exists (the structural controllability fix)
-----------------------------------------------------
The token backbone (:mod:`imas_ambix.worldmodel.controllable_model`) injects the
demanded actuator plan through AdaLN-Zero *side*-conditioning: a pooled plan
summary modulates every transformer block, but the prediction can still be
produced from the (predictable, redundant) camera history alone — the plan is a
soft hint the model is free to ignore.  The powered ΔN-M controllability gate
confirmed exactly that: true-plan rollouts ≈ random-plan rollouts.  No amount of
auxiliary pressure makes a side-input *necessary*.

This model puts the command INSIDE the recurrent transition.  It is a
Dreamer-style RSSM (Recurrent State-Space Model): a per-frame latent state
``z_t = [h_t ; s_t]`` evolves under a recurrence whose INPUT is the command,

    h_t = GRUCell(input=[s_{t-1} ; cmd_emb_t], hidden=h_{t-1}),

so a different command necessarily produces a different deterministic recurrence
and therefore a different latent rollout — **controllability by construction**.
There is no path by which the rollout can be produced without consuming the
command, because the command is part of the state transition function, not an
optional modulation.  The decoded camera frames are a deterministic function of
the latent rollout, so a command edit propagates to the dreamt video.

Architecture (one camera view; T frames)
-----------------------------------------
* **Encoder (per frame)** — embed the frame's ``S`` spatial tokens with the
  camera token embedding (reused weights / layout from
  :class:`imas_ambix.worldmodel.spacetime_model.SpacetimeTransformer`: a value
  embedding over the ``vocab_size`` Open-MAGVIT2 codebook + factorised row/col
  spatial position), then pool over the ``S`` spatial tokens (mean pool) into a
  per-frame embedding ``e_t``.
* **Command (per frame)** — the demanded actuator vector for the window, with the
  measured-STATE columns (``masked_command_indices`` — Ip / density / tf) zeroed
  exactly as :meth:`ControllableSpacetimeTransformer._plan_summary` does (drive
  from commands, never read the realised plasma state off an always-on Ip), then
  resampled from the ``Pa`` actuator steps onto the ``T`` camera frames (nearest)
  and projected by a small MLP to ``cmd_emb_t``.
* **Transition** — a deterministic GRU recurrence (above) carrying ``h_t``, with a
  diagonal-Gaussian PRIOR ``p(s_t | h_t)`` and POSTERIOR ``q(s_t | h_t, e_t)``.
  Training uses the POSTERIOR sample (it sees the frame, so reconstruction is
  feasible) and a KL term ties the prior to it; a PLAY rollout uses the PRIOR
  (no observations) so the rollout depends only on (initial state, commands).
* **Decoder** — ``z_t`` is expanded to the ``S`` spatial positions via ``S``
  learned spatial query embeddings (+ the reused row/col position) conditioned on
  ``z_t``, then scored by the reused CHUNKED camera head over the
  ``vocab_size`` codebook (the full logit tensor is never materialised).  A
  per-stream diagnostic head decodes ``z_t`` to the measured-signal vocabularies
  (secondary, next-step CE with PAD ignore).
* **Loss** — ``camera_CE + diagnostic_weight*diagnostic_CE + beta*KL`` with KL
  FREE BITS (each latent dim's KL is clamped at a floor so the posterior cannot
  collapse the stochastic state to the prior and stop carrying information).  All
  reductions are mean-over-(B,T) so the magnitude is comparable to the token
  backbone's per-token CE.

The camera token embedding, the chunked camera head, the per-stream diagnostic
heads, and the row/col position embeds are warm-startable from a Phase-1
controllable checkpoint (:meth:`RSSMWorldModel.warm_start_from_phase1`,
shape-matched, ``strict=False``); the recurrent latent core (GRU, prior /
posterior / command MLPs, the latent->spatial decoder) stays fresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec

#: PAD local-id (an absent / sub-sampled-empty signal step) — the diagnostic CE
#: ignore index.  Mirrors :data:`imas_ambix.worldmodel.dataset.PAD_LOCAL_ID`.
PAD_LOCAL_ID = 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RSSMConfig:
    """Hyper-parameters for the command-conditioned recurrent latent world model.

    Attributes
    ----------
    vocab_size:
        Camera Open-MAGVIT2 codebook size (the per-token output classes).  The
        decode head is chunked over this axis so a 262144 vocab never materialises
        a ``(B·T·S, vocab)`` tensor.
    grid_h, grid_w:
        Frame token grid (16 x 16 on the real corpus); ``n_spatial = grid_h *
        grid_w`` spatial tokens per camera frame.
    d_model:
        Token-embedding / decoder hidden width (the reused camera embedding +
        chunked head run at this width).
    h_dim:
        Deterministic recurrent-state (``h_t``) width — the GRU hidden size.
    s_dim:
        Stochastic latent-state (``s_t``) width — the diagonal-Gaussian dimension.
    a_dim:
        Command-embedding (``cmd_emb_t``) width.
    cmd_hidden, latent_hidden, decoder_hidden:
        Hidden widths of the command MLP, the prior / posterior MLPs, and the
        latent->spatial decoder MLP respectively.
    min_std:
        Floor added to the softplus std of both the prior and posterior so the
        Gaussians never degenerate to a delta.
    beta:
        KL loss weight.
    free_bits:
        Per-dimension KL FLOOR in nats — each latent dim's KL is clamped UP to this
        value before summing, so the posterior is never penalised for matching the
        prior below the floor (prevents stochastic-state collapse).
    diagnostic_weight:
        Weight on the per-stream diagnostic next-step CE in the loss.
    action_contrastive:
        Turn the always-on action-contrastive InfoNCE term ON.  Keeps the command
        load-bearing through reconstruction training: the realised (posterior)
        latent must look more like the TRUE-command PRIOR rollout than a
        WRONG-command one, so a command edit demonstrably moves the latent the
        decoder reads.  OFF (default) leaves the model byte-identical to the plain
        ELBO RSSM (no projector is even built).  Honoured only when the model has a
        command path (``has_actuator``).
    action_contrastive_weight:
        Weight on the action-contrastive term in the total loss (default 1.0).
    contrastive_dim:
        Width of the shared projector that maps a latent into the InfoNCE space.
    action_contrastive_temperature:
        Temperature for the InfoNCE softmax (smaller = sharper).
    signal_streams:
        Measured-signal streams the diagnostic heads decode (name / vocab /
        channels — :class:`imas_ambix.worldmodel.spacetime_model_v2.
        SignalStreamSpec`).  Empty disables the diagnostic head + objective.
    actuator_channels:
        Width of the continuous actuator (command) vector.  0 disables the command
        path (the model is then an unconditioned recurrent autoencoder — useful
        only as an ablation; the controllability property needs commands).
    masked_command_indices:
        Actuator-vector columns to ZERO before conditioning — the measured STATES
        (plasma_current, ne_line_integrated) + the quasi-static tf_current.  These
        are NOT commands; masking them forces drive-from-commands.  Honoured
        exactly like :meth:`ControllableSpacetimeTransformer._plan_summary`.
    """

    vocab_size: int = 1 << 18
    grid_h: int = 16
    grid_w: int = 16
    d_model: int = 256
    h_dim: int = 256
    s_dim: int = 32
    a_dim: int = 64
    cmd_hidden: int = 128
    latent_hidden: int = 256
    decoder_hidden: int = 512
    min_std: float = 0.1
    beta: float = 1.0
    free_bits: float = 1.0
    diagnostic_weight: float = 0.5
    # action-contrastive (OFF by default -> byte-identical to the plain ELBO RSSM).
    action_contrastive: bool = False
    action_contrastive_weight: float = 1.0
    contrastive_dim: int = 128
    action_contrastive_temperature: float = 0.1
    signal_streams: tuple[SignalStreamSpec, ...] = field(default_factory=tuple)
    actuator_channels: int = 0
    masked_command_indices: tuple[int, ...] = ()

    @property
    def n_spatial(self) -> int:
        return int(self.grid_h * self.grid_w)

    @property
    def z_dim(self) -> int:
        """Full latent-state width ``[h_t ; s_t]``."""
        return int(self.h_dim + self.s_dim)

    @property
    def has_actuator(self) -> bool:
        return int(self.actuator_channels) > 0

    @property
    def has_diagnostics(self) -> bool:
        return len(self.signal_streams) > 0

    @property
    def has_action_contrastive(self) -> bool:
        """Action-contrastive (projector + term) iff ON and the model has a command."""
        return bool(self.action_contrastive) and self.has_actuator


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class RSSMOutput:
    """Forward output of :meth:`RSSMWorldModel.forward`.

    Attributes
    ----------
    loss:
        Total scalar loss ``camera_CE + diagnostic_weight*diagnostic_CE +
        beta*KL + action_contrastive_weight*action_contrastive`` (mean over (B, T)).
    camera_ce, diagnostic_ce, kl:
        The three ELBO components (each a scalar; ``diagnostic_ce`` is 0 when the
        model has no diagnostic streams or none are scored).
    action_contrastive:
        The action-contrastive InfoNCE term (a scalar; a zero-magnitude touch over
        the projector when the term is OFF or a wrong-command rollout cannot form —
        so it is always in the autograd graph and DDP-uniform).
    h, s:
        ``(B, T, h_dim)`` and ``(B, T, s_dim)`` — the per-frame deterministic and
        (posterior-sampled) stochastic states from the teacher-forced rollout.
    """

    loss: torch.Tensor
    camera_ce: torch.Tensor
    diagnostic_ce: torch.Tensor
    kl: torch.Tensor
    action_contrastive: torch.Tensor
    h: torch.Tensor
    s: torch.Tensor


@dataclass
class RSSMRollout:
    """Output of :meth:`RSSMWorldModel.rollout_prior` — the PLAY rollout.

    Attributes
    ----------
    frames:
        ``(B, n_steps, S)`` long argmax-decoded camera-token ids for each rolled
        step (the dreamt clip under the supplied commands).
    h, s:
        ``(B, n_steps, h_dim)`` / ``(B, n_steps, s_dim)`` — the rolled prior states.
    diagnostics:
        ``{name: (B, n_steps, C, vocab)}`` per-stream diagnostic logits at each
        rolled step (empty when the model has no diagnostic heads).
    """

    frames: torch.Tensor
    h: torch.Tensor
    s: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Diagonal-Gaussian latent head
# ---------------------------------------------------------------------------


class _GaussianHead(nn.Module):
    """MLP ``in_dim -> (mean, std)`` of a diagonal Gaussian over ``s_dim``.

    ``std`` is a softplus of the raw output plus ``min_std`` so it is strictly
    positive and floored away from a delta.
    """

    def __init__(self, in_dim: int, s_dim: int, hidden: int, min_std: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * s_dim),
        )
        self.s_dim = int(s_dim)
        self.min_std = float(min_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, raw_std = out[..., : self.s_dim], out[..., self.s_dim :]
        std = F.softplus(raw_std) + self.min_std
        return mean, std


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class RSSMWorldModel(nn.Module):
    """Command-conditioned recurrent latent-dynamics world model over camera tokens.

    See the module docstring for the architecture and the controllability-by-
    construction argument.  The public surface is :meth:`forward` (teacher-forced
    ELBO), :meth:`rollout_prior` (the PLAY / controllability rollout), and
    :meth:`warm_start_from_phase1` (load the reusable token / head / diagnostic
    weights from a Phase-1 checkpoint).
    """

    def __init__(self, cfg: RSSMConfig) -> None:
        super().__init__()
        self.config = cfg
        d = cfg.d_model

        # ── reused camera token embedding (value + factorised row/col position) ──
        # Same names/layout as SpacetimeTransformer so a Phase-1 checkpoint's
        # token_embed / row_embed / col_embed load by name (warm start).
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.row_embed = nn.Embedding(cfg.grid_h, d)
        self.col_embed = nn.Embedding(cfg.grid_w, d)
        # encoder pool -> per-frame embedding e_t (a learned projection after the
        # spatial mean-pool gives the latent core a clean frame summary).
        self.encoder_proj = nn.Linear(d, d)

        # ── command path (per frame) ──
        if cfg.has_actuator:
            self.cmd_mlp = nn.Sequential(
                nn.Linear(int(cfg.actuator_channels), cfg.cmd_hidden),
                nn.SiLU(),
                nn.Linear(cfg.cmd_hidden, cfg.a_dim),
            )
        a_dim = cfg.a_dim if cfg.has_actuator else 0

        # ── transition: GRU recurrence INPUT = [s_{t-1} ; cmd_emb_t] ──
        self.gru = nn.GRUCell(cfg.s_dim + a_dim, cfg.h_dim)
        # learned initial deterministic/stochastic state.
        self.h0 = nn.Parameter(torch.zeros(cfg.h_dim))
        self.s0 = nn.Parameter(torch.zeros(cfg.s_dim))
        # prior p(s_t | h_t); posterior q(s_t | h_t, e_t).
        self.prior_head = _GaussianHead(
            cfg.h_dim, cfg.s_dim, cfg.latent_hidden, cfg.min_std
        )
        self.posterior_head = _GaussianHead(
            cfg.h_dim + d, cfg.s_dim, cfg.latent_hidden, cfg.min_std
        )

        # ── latent -> spatial decoder ──
        # S learned spatial query embeddings; each decoded position conditions on
        # z_t (broadcast) + the reused row/col position + its spatial query, then a
        # small MLP -> the d-wide per-token hidden the camera head scores.
        self.spatial_query = nn.Parameter(torch.zeros(cfg.n_spatial, d))
        self.decoder_in = nn.Linear(cfg.z_dim, d)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d, cfg.decoder_hidden),
            nn.SiLU(),
            nn.Linear(cfg.decoder_hidden, d),
        )
        self.decoder_ln = nn.LayerNorm(d)
        # reused CHUNKED camera head over the codebook (weight-tied to token_embed,
        # the standard LM choice + halves the 2^18 x d table; aligns value /
        # prediction space).
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

        # ── per-stream diagnostic heads (joint generation, secondary) ──
        if cfg.has_diagnostics:
            self.diag_in = nn.Linear(cfg.z_dim, d)
            self.diagnostic_heads = nn.ModuleDict()
            for stream in cfg.signal_streams:
                head = nn.Linear(d, int(stream.vocab))
                self.diagnostic_heads[stream.name] = head

        self.apply(self._init_weights)
        # re-tie after init (apply may have re-init the head weight tensor view).
        self.head.weight = self.token_embed.weight

        # ── action-contrastive projector (built LAST, AFTER _init_weights) ──
        # Built last + only when ON so an OFF model is byte-identical to the plain
        # ELBO RSSM (same parameters, same RNG-consumption order during init), and a
        # warm-start from an OFF checkpoint sees the identical key set.
        self.has_action_contrastive = cfg.has_action_contrastive
        if self.has_action_contrastive:
            cd = int(cfg.contrastive_dim)
            proj = nn.Sequential(
                nn.Linear(cfg.z_dim, cd),
                nn.GELU(),
                nn.Linear(cd, cd),
            )
            for m in proj:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
            self.action_contrastive_proj = proj

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # -- introspection -----------------------------------------------------

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    # -- encoder -----------------------------------------------------------

    def _spatial_position(self, device: torch.device) -> torch.Tensor:
        """Factorised (row + col) spatial position over the raster grid -> (S, d)."""
        cfg = self.config
        rows = torch.arange(cfg.grid_h, device=device).repeat_interleave(cfg.grid_w)
        cols = torch.arange(cfg.grid_w, device=device).repeat(cfg.grid_h)
        return self.row_embed(rows) + self.col_embed(cols)  # (S, d)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """``(B, T, S) long -> (B, T, d)`` per-frame encoder embeddings ``e_t``.

        Value embedding + factorised spatial position, then mean-pool over the S
        spatial tokens and a learned projection.
        """
        cfg = self.config
        b, t, s = frames.shape
        emb = self.token_embed(frames)  # (B, T, S, d)
        emb = emb + self._spatial_position(frames.device).view(1, 1, s, cfg.d_model)
        pooled = emb.mean(dim=2)  # (B, T, d) — spatial mean-pool
        return self.encoder_proj(pooled)

    # -- command -----------------------------------------------------------

    def _mask_commands(self, values: torch.Tensor) -> torch.Tensor:
        """Zero the masked measured-STATE columns (drive-from-commands).

        Mirrors :meth:`ControllableSpacetimeTransformer._plan_summary`: the masked
        columns (Ip / density / tf) carry the realised plasma state, not a command,
        and an always-on Ip context lets the model read the state instead of
        deriving it from the commands.  Returns a CLONE with those columns zeroed.
        """
        cfg = self.config
        mask_idx = tuple(cfg.masked_command_indices or ())
        if not mask_idx:
            return values
        c = values.shape[-1]
        cols = [int(i) for i in mask_idx if 0 <= int(i) < c]
        if not cols:
            return values
        values = values.clone()
        values[..., cols] = 0.0
        return values

    def frame_commands(
        self, actuator: dict[str, torch.Tensor] | None, n_frames: int
    ) -> torch.Tensor | None:
        """``actuator -> (B, T, a_dim)`` per-camera-frame command embeddings.

        Takes ``actuator["values"]`` ``(B, Pa, C)`` (the demanded plan), ZEROES the
        masked-state columns (drive-from-commands), resamples the ``Pa`` actuator
        steps onto the ``T = n_frames`` camera frames by NEAREST step (the actuator
        plan is coarser than the camera cadence), and projects each frame's command
        vector with the command MLP.  Returns ``None`` when the model has no
        command path / no plan is supplied.
        """
        cfg = self.config
        if not cfg.has_actuator or actuator is None:
            return None
        values = actuator.get("values")
        if values is None or values.numel() == 0 or values.ndim != 3:
            return None
        b, pa, c = values.shape
        if pa == 0 or c == 0:
            return None
        values = self._mask_commands(values.to(self.cmd_mlp[0].weight.dtype))
        # nearest-step resample onto the T camera frames.
        if pa == n_frames:
            per_frame = values
        else:
            # map frame t (0..T-1) to its nearest actuator-step index.
            t_pos = torch.arange(n_frames, device=values.device, dtype=torch.float32)
            if n_frames > 1:
                step = t_pos / float(n_frames - 1) * float(pa - 1)
            else:
                step = torch.zeros(1, device=values.device)
            idx = step.round().long().clamp(0, pa - 1)  # (T,)
            per_frame = values[:, idx, :]  # (B, T, C)
        return self.cmd_mlp(per_frame)  # (B, T, a_dim)

    def wrong_frame_commands(
        self,
        actuator: dict[str, torch.Tensor] | None,
        n_frames: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | None:
        """A WRONG/random per-frame command embedding for the action-contrastive term.

        Builds the TRUE per-frame command vectors exactly as :meth:`frame_commands`
        (masked-state columns zeroed, resampled onto the ``T`` frames), then
        PERTURBS only the UNMASKED command columns with a per-sample random shift
        (the masked state columns ``masked_command_indices`` are HELD — they are not
        commands and a perturbation there would be a no-op anyway, since they are
        zeroed) before the command MLP.  The perturbation is a batch-roll of the true
        commands plus Gaussian noise on the unmasked columns, so a different sample's
        plan (a genuinely different drive) becomes the wrong command — guaranteed to
        differ from the true command whenever the unmasked columns are not all equal.
        Returns the projected wrong commands ``(B, T, a_dim)`` or ``None`` when no
        command path / no plan is supplied.
        """
        cfg = self.config
        if not cfg.has_actuator or actuator is None:
            return None
        values = actuator.get("values")
        if values is None or values.numel() == 0 or values.ndim != 3:
            return None
        b, pa, c = values.shape
        if pa == 0 or c == 0:
            return None
        values = self._mask_commands(values.to(self.cmd_mlp[0].weight.dtype))
        # the columns that ARE commands (NOT masked) — the only ones we perturb.
        mask_idx = {int(i) for i in (cfg.masked_command_indices or ())}
        cmd_cols = [j for j in range(c) if j not in mask_idx]
        wrong = values.clone()
        if cmd_cols:
            # roll the batch so each sample gets a DIFFERENT sample's true commands,
            # then add Gaussian noise on the command columns — a wrong drive.
            rolled = torch.roll(values, shifts=1, dims=0) if b > 1 else values
            if generator is not None:
                noise = torch.randn(
                    values.shape, generator=generator, device=values.device
                ).to(values.dtype)
            else:
                noise = torch.randn_like(values)
            cols = torch.as_tensor(cmd_cols, device=values.device, dtype=torch.long)
            wrong[..., cols] = rolled[..., cols] + noise[..., cols]
        # nearest-step resample onto the T frames + project (mirror frame_commands).
        if pa == n_frames:
            per_frame = wrong
        else:
            t_pos = torch.arange(n_frames, device=wrong.device, dtype=torch.float32)
            if n_frames > 1:
                step = t_pos / float(n_frames - 1) * float(pa - 1)
            else:
                step = torch.zeros(1, device=wrong.device)
            idx = step.round().long().clamp(0, pa - 1)
            per_frame = wrong[:, idx, :]
        return self.cmd_mlp(per_frame)

    def _gru_input(
        self, s_prev: torch.Tensor, cmd_t: torch.Tensor | None
    ) -> torch.Tensor:
        """Assemble the GRU input ``[s_{t-1} ; cmd_emb_t]`` (command in transition)."""
        if cmd_t is None:
            return s_prev
        return torch.cat([s_prev, cmd_t], dim=-1)

    # -- decoder -----------------------------------------------------------

    def decode_hidden(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, T, z_dim) -> (B, T, S, d)`` per-token decoder hiddens.

        Each of the ``S`` spatial positions conditions on ``z_t`` (broadcast) + the
        reused factorised row/col position + a learned per-position spatial query,
        through a small MLP, producing the d-wide hidden the chunked camera head
        scores.  The full logit tensor is NOT built here.
        """
        cfg = self.config
        b, t, _ = z.shape
        s = cfg.n_spatial
        d = cfg.d_model
        base = self.decoder_in(z)  # (B, T, d)
        spatial = self._spatial_position(z.device)  # (S, d)
        pos = spatial.view(1, 1, s, d) + self.spatial_query.view(1, 1, s, d)
        h = base.view(b, t, 1, d) + pos  # (B, T, S, d)
        h = h + self.decoder_mlp(self.decoder_ln(h))
        return h

    def _chunked_camera_ce(
        self, hidden: torch.Tensor, frames: torch.Tensor, *, chunk: int
    ) -> torch.Tensor:
        """Mean camera-token cross-entropy over (B, T, S), chunked over the vocab.

        ``hidden`` is ``(B, T, S, d)`` (decoded per-token hiddens for frame t);
        ``frames`` is ``(B, T, S)`` the RECONSTRUCTION target (frame t's own
        tokens — the RSSM reconstructs each observed frame from its latent).  The
        head logits never exceed ``chunk x vocab``.
        """
        b, t, s, d = hidden.shape
        flat_h = hidden.reshape(-1, d)
        flat_tgt = frames.reshape(-1).to(flat_h.device).long()
        n = flat_h.shape[0]
        total = hidden.new_zeros(())
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            logits = self.head(flat_h[start:stop])  # (chunk, vocab)
            total = total + F.cross_entropy(
                logits, flat_tgt[start:stop], reduction="sum"
            )
            del logits
        return total / max(n, 1)

    @torch.no_grad()
    def _chunked_camera_argmax(
        self, hidden: torch.Tensor, *, chunk: int
    ) -> torch.Tensor:
        """Argmax-decode camera tokens from hiddens ``(B, S, d) -> (B, S)``."""
        b, s, d = hidden.shape
        flat = hidden.reshape(-1, d)
        out = flat.new_zeros(flat.shape[0], dtype=torch.long)
        for start in range(0, flat.shape[0], chunk):
            stop = min(start + chunk, flat.shape[0])
            logits = self.head(flat[start:stop])
            out[start:stop] = logits.argmax(dim=-1)
            del logits
        return out.reshape(b, s)

    # -- diagnostics -------------------------------------------------------

    def diagnostic_logits(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """``(B, T, z_dim) -> {name: (B, T, C, vocab)}`` per-stream diagnostic logits.

        Each stream's ``C`` channels are decoded from the SAME per-frame latent
        (the latent is a compact whole-state summary; per-channel structure is left
        to the head's output rows).  A linear ``z -> d`` map feeds each stream head.
        Empty when the model has no diagnostic heads.
        """
        cfg = self.config
        if not cfg.has_diagnostics:
            return {}
        b, t, _ = z.shape
        feat = self.diag_in(z)  # (B, T, d)
        out: dict[str, torch.Tensor] = {}
        for stream in cfg.signal_streams:
            name = stream.name
            logits = self.diagnostic_heads[name](feat)  # (B, T, vocab)
            # broadcast the per-frame state over the stream's C channels.
            out[name] = logits.unsqueeze(2).expand(b, t, int(stream.channels), -1)
        return out

    def _diagnostic_ce(
        self, z: torch.Tensor, signals: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """Next-step diagnostic CE on the measured-signal tokens (PAD ignore).

        For each stream the per-frame latent at step ``j`` predicts the stream's
        token at step ``j+1`` (next-step), over the steps the stream supplies; PAD
        (id 0) is ignored.  Mean over scored positions, summed across streams.
        Returns a zero scalar when nothing is scored.
        """
        cfg = self.config
        if not cfg.has_diagnostics or not signals:
            return z.new_zeros(())
        feat = self.diag_in(z)  # (B, T, d)
        b, t, _ = feat.shape
        total_ce: torch.Tensor | None = None
        total_count = 0
        for stream in cfg.signal_streams:
            name = stream.name
            tok = signals.get(name)
            if tok is None or tok.numel() == 0 or tok.ndim != 3:
                continue
            ps, cs = int(tok.shape[1]), int(tok.shape[2])
            p = min(ps, t)
            if p < 2 or cs < 1:
                continue
            # the latent at step j predicts the stream token at j+1.
            logits = self.diagnostic_heads[name](feat[:, : p - 1])  # (B, p-1, vocab)
            v = logits.shape[-1]
            # broadcast the per-frame logits across the cs channels.
            logits = logits.unsqueeze(2).expand(b, p - 1, cs, v).reshape(-1, v)
            target = tok[:, 1:p, :cs].reshape(-1).to(logits.device).long()
            ce = F.cross_entropy(
                logits, target, ignore_index=PAD_LOCAL_ID, reduction="sum"
            )
            cnt = int((target != PAD_LOCAL_ID).sum())
            if cnt > 0:
                total_ce = ce if total_ce is None else total_ce + ce
                total_count += cnt
        if total_ce is None or total_count == 0:
            return z.new_zeros(())
        return total_ce / float(total_count)

    # -- KL with free bits -------------------------------------------------

    @staticmethod
    def _kl_diag_gaussian(
        q_mean: torch.Tensor,
        q_std: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        *,
        free_bits: float,
    ) -> torch.Tensor:
        """Mean-over-(B, T) KL(q || p) for diagonal Gaussians, with FREE BITS.

        ``q_*`` / ``p_*`` are ``(B, T, s_dim)``.  The per-dim KL is clamped UP to
        ``free_bits`` nats BEFORE the dim-sum, so the posterior is not penalised for
        matching the prior below the floor (prevents the stochastic state from
        collapsing onto the prior).  Reduces to a per-(B, T) scalar then means.
        """
        # closed-form per-dim KL(N(qm,qs) || N(pm,ps))
        var_ratio = (q_std / p_std) ** 2
        kl = 0.5 * (
            var_ratio
            + ((p_mean - q_mean) ** 2) / (p_std**2)
            - 1.0
            - torch.log(var_ratio)
        )  # (B, T, s_dim)
        if free_bits > 0.0:
            kl = torch.clamp(kl, min=float(free_bits))
        return kl.sum(dim=-1).mean()

    # -- action-contrastive on the latent ----------------------------------

    def _projector_zero_touch(self, ref: torch.Tensor) -> torch.Tensor:
        """A zero-magnitude sum over the contrastive projector's params (DDP-uniform).

        A batch / step where the action-contrastive term cannot score (no plan, no
        wrong-command rollout, T<1) would otherwise leave the projector grad-less and
        desync a DDP rank.  This touches every Linear in the projector with a
        ``*0.0`` contribution so it stays in the autograd graph with no effect.
        Returns a zero scalar when the model has no projector (term OFF).
        """
        if not getattr(self, "has_action_contrastive", False):
            return ref.new_zeros(())
        acc: torch.Tensor | None = None
        for m in self.action_contrastive_proj.modules():
            if isinstance(m, nn.Linear):
                s = m.weight.sum() + m.bias.sum()
                acc = s if acc is None else acc + s
        if acc is None:
            return ref.new_zeros(())
        return acc.to(ref.dtype) * 0.0

    def action_contrastive_loss(
        self,
        anchor: torch.Tensor,
        prior_true: torch.Tensor,
        prior_wrong: torch.Tensor | None,
        *,
        context_frames: int = 0,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """InfoNCE: the realised latent matches the TRUE-command prior, not a wrong one.

        ``anchor`` is the realised/posterior latent ``(B, T, z_dim)`` (the forward's
        teacher-forced rollout — STOP-GRADded here so the contrastive term moves the
        PREDICTIONS toward reality, not reality toward a prediction).  ``prior_true``
        is the TRUE-command PRIOR latent ``(B, T, z_dim)`` and ``prior_wrong`` the
        WRONG-command PRIOR latent ``(B, T, z_dim)``.  Pools the forecast-window
        latents (frames ``>= context_frames``) over time, projects each through the
        shared head, L2-normalises, and runs a temperature-scaled InfoNCE:

        * anchor = the realised latent (its own row);
        * positive = its TRUE-command prior latent (the diagonal of the true block);
        * negatives = EVERY sample's WRONG-command prior latent (the whole wrong
          block) PLUS the other samples' true-command priors masked out (different
          shots are legitimately different — only the WRONG-command rows compete).

        Cross-entropy keeps pulling the true-command prediction above the
        wrong-command ones with NO margin floor (unlike a relu-margin repulsion,
        which clamps to 0 once already separated and stops applying gradient — the
        failure that drove the token-backbone term to ~0).  Because the RSSM decode
        is latent-only, pulling the true-command prior latent toward the realised
        state and pushing the wrong-command one away DIRECTLY trains command
        sensitivity in the latent the decoder reads.

        Returns a scalar; a zero-magnitude projector touch when it cannot score
        (term OFF / no wrong-command rollout / empty window) so the projector stays
        in the autograd graph and a DDP rank never desyncs.
        """
        zt = self._projector_zero_touch(anchor)
        if not getattr(self, "has_action_contrastive", False) or prior_wrong is None:
            return zt
        b, t, _z = anchor.shape
        if t < 1:
            return zt
        ctx = max(0, min(int(context_frames), t - 1)) if t > 1 else 0
        a = anchor[:, ctx:].mean(dim=1).detach()  # realised next state (B, z) stop-grad
        pt = prior_true[:, ctx:].mean(dim=1)  # true-command prior next state (B, z)
        pw = prior_wrong[:, ctx:].mean(dim=1)  # wrong-command prior next state (B, z)
        z_a = F.normalize(self.action_contrastive_proj(a), dim=-1)  # (B, cd)
        z_t = F.normalize(self.action_contrastive_proj(pt), dim=-1)  # (B, cd)
        z_w = F.normalize(self.action_contrastive_proj(pw), dim=-1)  # (B, cd)
        tau = max(float(temperature), 1e-4)
        # anchor x true candidates (B, B); positive = the diagonal (own true prior).
        sim_true = z_a @ z_t.t()
        sim_wrong = z_a @ z_w.t()  # anchor x wrong candidates (B, B) — all negatives
        # mask the non-self TRUE columns to -inf so only the diagonal true (positive)
        # and the B wrong negatives compete in the softmax.
        eye = torch.eye(b, device=z_a.device, dtype=torch.bool)
        sim_true = sim_true.masked_fill(~eye, float("-inf"))
        logits = torch.cat([sim_true, sim_wrong], dim=1) / tau  # (B, 2B)
        labels = torch.arange(b, device=z_a.device)  # positive = diagonal of true
        return F.cross_entropy(logits, labels) + zt

    # -- teacher-forced ELBO forward ---------------------------------------

    def forward(
        self,
        batch: dict,
        *,
        chunk: int = 4096,
    ) -> RSSMOutput:
        """Teacher-forced ELBO over the camera window.

        ``batch`` is ``{"frames": (B, T, S) long, "actuator": {"values": (B, Pa, C)
        float, "missing": ...}, "signals": {name: (B, Ps, Cs) long}}`` (plan /
        missing are optional).  Encodes every frame (posterior), rolls the
        deterministic recurrence under the per-frame commands, samples each ``s_t``
        from the POSTERIOR (reparameterised), decodes the reconstruction + the
        diagnostics, and returns the ELBO components.

        Loss ``= camera_CE + diagnostic_weight*diagnostic_CE + beta*KL +
        action_contrastive_weight*action_contrastive`` (each a mean over (B, T)).
        The camera CE is the RECONSTRUCTION of each observed frame from its own
        latent.  The action-contrastive term (when ON) keeps the command load-bearing
        by training the TRUE-command PRIOR latent to match the realised state more
        closely than a WRONG-command one — see :meth:`action_contrastive_loss`.
        """
        cfg = self.config
        frames = batch["frames"]
        b, t, _s = frames.shape

        e = self.encode_frames(frames)  # (B, T, d) — posterior evidence
        cmd = self.frame_commands(batch.get("actuator"), t)  # (B, T, a_dim) | None
        do_ac = self.has_action_contrastive and cmd is not None
        cmd_wrong = (
            self.wrong_frame_commands(batch.get("actuator"), t) if do_ac else None
        )

        h_t = self.h0.view(1, -1).expand(b, -1).contiguous()  # (B, h_dim)
        s_prev = self.s0.view(1, -1).expand(b, -1).contiguous()  # (B, s_dim)

        h_list: list[torch.Tensor] = []
        s_list: list[torch.Tensor] = []
        prior_means: list[torch.Tensor] = []
        prior_stds: list[torch.Tensor] = []
        post_means: list[torch.Tensor] = []
        post_stds: list[torch.Tensor] = []
        # action-contrastive: per-frame PRIOR latents under the TRUE / WRONG command.
        prior_true_list: list[torch.Tensor] = []
        prior_wrong_list: list[torch.Tensor] = []

        for ti in range(t):
            cmd_t = cmd[:, ti] if cmd is not None else None
            h_prev = h_t  # det state INTO this step (shared by the true + wrong rolls)
            h_t = self.gru(self._gru_input(s_prev, cmd_t), h_prev)  # (B, h_dim)
            p_mean, p_std = self.prior_head(h_t)
            q_mean, q_std = self.posterior_head(torch.cat([h_t, e[:, ti]], dim=-1))
            # reparameterised posterior sample (training state).
            eps = torch.randn_like(q_std)
            s_t = q_mean + q_std * eps
            h_list.append(h_t)
            s_list.append(s_t)
            prior_means.append(p_mean)
            prior_stds.append(p_std)
            post_means.append(q_mean)
            post_stds.append(q_std)
            if do_ac:
                # TRUE-command 1-step PRIOR latent at this frame: the deterministic
                # state from the TRUE command + its prior mean.
                prior_true_list.append(torch.cat([h_t, p_mean], dim=-1))
                # WRONG-command 1-step PRIOR latent: re-run the SAME transition from
                # the SAME realised history (h_prev, s_prev) but under the WRONG
                # command, so the only thing that differs is the command (an
                # apples-to-apples counterfactual).
                cmd_w = cmd_wrong[:, ti] if cmd_wrong is not None else None
                h_w = self.gru(self._gru_input(s_prev, cmd_w), h_prev)
                pw_mean, _pw_std = self.prior_head(h_w)
                prior_wrong_list.append(torch.cat([h_w, pw_mean], dim=-1))
            s_prev = s_t

        h_seq = torch.stack(h_list, dim=1)  # (B, T, h_dim)
        s_seq = torch.stack(s_list, dim=1)  # (B, T, s_dim)
        z = torch.cat([h_seq, s_seq], dim=-1)  # (B, T, z_dim)

        # ── camera reconstruction CE ──
        dec_hidden = self.decode_hidden(z)  # (B, T, S, d)
        camera_ce = self._chunked_camera_ce(dec_hidden, frames, chunk=chunk)

        # ── diagnostics (secondary) ──
        diagnostic_ce = self._diagnostic_ce(z, batch.get("signals"))

        # ── KL(q || p) with free bits ──
        kl = self._kl_diag_gaussian(
            torch.stack(post_means, dim=1),
            torch.stack(post_stds, dim=1),
            torch.stack(prior_means, dim=1),
            torch.stack(prior_stds, dim=1),
            free_bits=cfg.free_bits,
        )

        # ── action-contrastive (always-on InfoNCE; zero-touch when off) ──
        if do_ac and prior_wrong_list:
            prior_true = torch.stack(prior_true_list, dim=1)  # (B, T, z_dim)
            prior_wrong = torch.stack(prior_wrong_list, dim=1)  # (B, T, z_dim)
            action_contrastive = self.action_contrastive_loss(
                z,
                prior_true,
                prior_wrong,
                context_frames=0,
                temperature=cfg.action_contrastive_temperature,
            )
        else:
            action_contrastive = self._projector_zero_touch(z)

        loss = (
            camera_ce
            + cfg.diagnostic_weight * diagnostic_ce
            + cfg.beta * kl
            + cfg.action_contrastive_weight * action_contrastive
        )
        return RSSMOutput(
            loss=loss,
            camera_ce=camera_ce,
            diagnostic_ce=diagnostic_ce,
            kl=kl,
            action_contrastive=action_contrastive,
            h=h_seq,
            s=s_seq,
        )

    # -- PLAY / controllability rollout ------------------------------------

    @torch.no_grad()
    def rollout_prior(
        self,
        context_frames: torch.Tensor,
        actuator: dict[str, torch.Tensor] | None,
        n_steps: int,
        *,
        chunk: int = 4096,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> RSSMRollout:
        """Roll the PRIOR forward under the commands (no obs) — the PLAY rollout.

        Encodes the ``context_frames`` ``(B, Tc, S)`` with the POSTERIOR to warm the
        recurrent state, then rolls the PRIOR forward for ``n_steps`` under the
        per-step commands (NO further observations): ``s_t ~ prior``, ``h_t =
        GRU([s_{t-1} ; cmd_t], h_{t-1})``, decoding each ``z_t`` to argmax camera
        tokens + diagnostics.  THIS is where the command is load-bearing — the
        rollout is a deterministic function of (warmed state, commands), so a
        different command necessarily yields a different rollout.

        ``actuator["values"]`` spans the FULL window (context + rollout); the
        per-frame command resample maps the actuator steps onto the ``Tc +
        n_steps`` camera-frame axis, and the rollout consumes the commands at the
        rollout-frame positions ``Tc .. Tc+n_steps-1``.  ``sample=False`` uses the
        prior MEAN (deterministic — the cleanest controllability probe); ``True``
        draws a reparameterised prior sample.
        """
        b, tc, _s = context_frames.shape
        device = context_frames.device
        total_t = tc + int(n_steps)

        # per-frame commands across the full (context + rollout) window.
        cmd = self.frame_commands(actuator, total_t)  # (B, total_t, a_dim) | None

        h_t = self.h0.view(1, -1).expand(b, -1).contiguous()
        s_prev = self.s0.view(1, -1).expand(b, -1).contiguous()

        # ── warm the state on the context frames (posterior) ──
        if tc > 0:
            e = self.encode_frames(context_frames)  # (B, Tc, d)
            for ti in range(tc):
                cmd_t = cmd[:, ti] if cmd is not None else None
                h_t = self.gru(self._gru_input(s_prev, cmd_t), h_t)
                q_mean, q_std = self.posterior_head(torch.cat([h_t, e[:, ti]], dim=-1))
                s_prev = q_mean  # use the posterior mean to seed (deterministic)

        # ── roll the PRIOR forward under the commands (no observations) ──
        h_list: list[torch.Tensor] = []
        s_list: list[torch.Tensor] = []
        for k in range(int(n_steps)):
            ti = tc + k
            cmd_t = cmd[:, ti] if cmd is not None else None
            h_t = self.gru(self._gru_input(s_prev, cmd_t), h_t)
            p_mean, p_std = self.prior_head(h_t)
            if sample:
                eps = (
                    torch.randn(p_std.shape, generator=generator, device=device)
                    if generator is not None
                    else torch.randn_like(p_std)
                )
                s_t = p_mean + p_std * eps
            else:
                s_t = p_mean
            h_list.append(h_t)
            s_list.append(s_t)
            s_prev = s_t

        h_seq = torch.stack(h_list, dim=1)  # (B, n_steps, h_dim)
        s_seq = torch.stack(s_list, dim=1)  # (B, n_steps, s_dim)
        z = torch.cat([h_seq, s_seq], dim=-1)  # (B, n_steps, z_dim)

        dec_hidden = self.decode_hidden(z)  # (B, n_steps, S, d)
        frames_out = torch.stack(
            [
                self._chunked_camera_argmax(dec_hidden[:, k], chunk=chunk)
                for k in range(int(n_steps))
            ],
            dim=1,
        )  # (B, n_steps, S)
        diagnostics = self.diagnostic_logits(z)
        return RSSMRollout(frames=frames_out, h=h_seq, s=s_seq, diagnostics=diagnostics)

    # -- warm start --------------------------------------------------------

    def warm_start_from_phase1(self, ckpt_path: str) -> dict[str, int]:
        """Load the REUSABLE token/head/diagnostic/position weights from a Phase-1 ckpt.

        Loads (shape-matched, ``strict=False``) the camera ``token_embed`` (and the
        weight-tied ``head``), the factorised ``row_embed`` / ``col_embed`` spatial
        position, and any per-stream ``diagnostic_heads.<name>`` whose shape
        matches, from a Phase-1 ControllableSpacetime checkpoint.  The recurrent
        latent core (GRU, prior / posterior MLPs, command MLP, the latent->spatial
        decoder) STAYS FRESH — those tensors have no Phase-1 counterpart.

        Accepts a raw ``state_dict`` payload or a ``{"model"|"model_state"|
        "state_dict": state_dict}`` wrapper.  Returns
        ``{"loaded": n, "fresh": m, "skipped_shape": k}`` tensor counts.
        """
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        src = payload
        for key in ("model_state_dict", "model", "model_state", "state_dict"):
            if isinstance(payload, dict) and key in payload:
                src = payload[key]
                break
        if not isinstance(src, dict):
            raise ValueError(f"could not find a state_dict in checkpoint {ckpt_path!r}")
        # strip a DDP "module." prefix if present.
        src = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in src.items()
        }
        return self._load_reusable(src)

    def _load_reusable(self, src: dict) -> dict[str, int]:
        """Copy shape-matched reusable tensors from ``src`` into ``self`` (in place).

        Reusable = the camera token embedding, the camera head (tied), the row/col
        position embeds, and the per-stream diagnostic heads.  Everything else
        (the latent core) is left at its fresh init.
        """
        own = self.state_dict()
        # ``head.`` is DELIBERATELY excluded: the head weight is TIED to
        # ``token_embed.weight`` (same storage), so loading ``token_embed.weight``
        # already sets the head.  Loading ``head.`` too would write the same
        # storage twice and the last-written value would win (a Phase-1 checkpoint
        # stores both tied keys with equal values, but a synthetic / mismatched one
        # need not).  Re-tie at the end to be safe.
        reusable_prefixes = (
            "token_embed.",
            "row_embed.",
            "col_embed.",
            "diagnostic_heads.",
        )
        to_load: dict[str, torch.Tensor] = {}
        skipped_shape = 0
        for name in own:
            if name.startswith("head."):
                continue
            if not name.startswith(reusable_prefixes):
                continue
            # prefer the exact key; fall back to ``head.weight`` for the tied
            # token embed when a checkpoint only stored the head side.
            chosen = name
            if (
                name == "token_embed.weight"
                and name not in src
                and "head.weight" in src
            ):
                chosen = "head.weight"
            if chosen not in src:
                continue
            if tuple(src[chosen].shape) == tuple(own[name].shape):
                to_load[name] = src[chosen]
            else:
                skipped_shape += 1
        missing, unexpected = self.load_state_dict({**own, **to_load}, strict=False)
        # re-tie the head after the load (token_embed may have been overwritten).
        self.head.weight = self.token_embed.weight
        loaded = len(to_load)
        fresh = len(own) - loaded
        return {"loaded": loaded, "fresh": fresh, "skipped_shape": skipped_shape}


__all__ = [
    "PAD_LOCAL_ID",
    "RSSMConfig",
    "RSSMOutput",
    "RSSMRollout",
    "RSSMWorldModel",
    "SignalStreamSpec",
]
