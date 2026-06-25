"""Phase-aware patch-transformer tokenizer for 1-D high-frequency signals.

A PatchTST / ViT-style autoencoder over windowed native-cadence signals.
Unlike the gen-1 signal tokenizers, this one is built to **preserve phase
and mode structure** — the interior-information study showed that
magnitude-only ``|rfft|`` features are structurally blind to the phase
wraps that encode poloidal/toroidal mode numbers, so the whole point here
is that the reconstruction keeps phase.

Architecture
------------
::

    raw window (n_channels, n_samples)
      → patchify: non-overlapping patches of `patch_size` samples
      → optional complex STFT lift: each patch → (real, imag) features
      → linear patch embedding → + positional encoding
      → Transformer encoder (pre-norm, GELU)
      → bottleneck:
          - FSQ            (finite-scalar quantisation, no codebook collapse)
          - VQ-VAE         (learned codebook + commitment loss)
          - continuous     (continuous embedding + mask — the CONTROL arm:
                            no quantisation, so phase cannot be quantisation-
                            destroyed; this is the fidelity ceiling)
      → Transformer decoder → linear patch un-embedding
      → reconstructed window

The bottleneck is swappable so the open codebook decision (FSQ vs VQ vs
continuous) can be resolved by **measuring** round-trip phase fidelity on a
holdout — see :func:`phase_error`, :func:`mode_number_recovery`, and the
training driver ``scripts/.../signal_tokenizer train``.

Cross-channel mode-number tokens
--------------------------------
For a toroidal/poloidal probe array (the xma ccbv Mirnov coils) the spatial
structure across channels carries the mode number.  :func:`mode_decomposition`
reuses the spatial-DFT decomposition the oracle probe computes (a DFT across
the angularly-distributed coil array at each time) to emit a small set of
mode-amplitude/phase tokens per patch — captured as a separate channel
block in the v2 store under ``BLOCK_XMA_MODE``.

This module loads cheaply (torch imported lazily inside the nn classes) so
the schema / loaders can be imported on a CPU-only node.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imas_ambix.calibration.signals import ChannelCalibration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PatchTokenizerConfig:
    """Hyper-parameters for the phase-aware patch-transformer tokenizer."""

    patch_size: int = 64
    """Native samples per patch (token cadence = native_rate / patch_size)."""

    seq_patches: int = 32
    """Patches per training sequence segment.  Windows are split into
    fixed-length segments of this many patches so shots of different
    duration yield equal-length, stackable training samples."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.0

    # Input representation.  When True, each patch is lifted to its complex
    # STFT (real + imag interleaved) before embedding — an explicitly
    # phase-preserving input.  When False, raw patch samples are embedded
    # directly (still phase-preserving — phase lives in the sample ordering).
    use_stft: bool = True

    # Bottleneck: "fsq" | "vq" | "continuous".
    bottleneck: str = "fsq"
    # FSQ per-dimension level counts (the product is the effective codebook
    # size).  Default [8,8,8,5,5] → 12 800 codes.
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    # VQ codebook size and commitment weight.
    vq_codebook_size: int = 1024
    vq_commitment: float = 0.25
    # Continuous bottleneck embedding dim (no quantisation).
    continuous_dim: int = 8

    def token_rate_hz(self, native_rate_hz: float) -> float:
        return native_rate_hz / self.patch_size


# ---------------------------------------------------------------------------
# Patchify + STFT lift (numpy — used by both the model and the encoders)
# ---------------------------------------------------------------------------


def patchify(x: np.ndarray, patch_size: int) -> tuple[np.ndarray, int]:
    """Split ``(C, T)`` into ``(C, n_patches, patch_size)``; pad the tail.

    Returns ``(patches, n_pad)`` where ``n_pad`` is the number of zero
    samples appended to the final patch (trimmed on decode).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    c, t = x.shape
    rem = t % patch_size
    n_pad = (patch_size - rem) if rem else 0
    if n_pad:
        x = np.concatenate([x, np.zeros((c, n_pad), dtype=x.dtype)], axis=1)
    n_patches = x.shape[1] // patch_size
    patches = x.reshape(c, n_patches, patch_size)
    return patches, n_pad


def unpatchify(patches: np.ndarray, n_pad: int) -> np.ndarray:
    """Inverse of :func:`patchify` — concatenate patches and trim padding."""
    c, n_patches, ps = patches.shape
    flat = patches.reshape(c, n_patches * ps)
    return flat[:, : flat.shape[1] - n_pad] if n_pad else flat


def stft_lift(patches: np.ndarray) -> np.ndarray:
    """Lift ``(..., patch_size)`` patches to interleaved real/imag rFFT.

    A per-patch rFFT keeps phase (real + imag) so the embedding sees the
    full complex spectrum, not just magnitude.  Output last-dim is
    ``2 * (patch_size // 2 + 1)``.
    """
    spec = np.fft.rfft(patches, axis=-1)
    return np.concatenate([spec.real, spec.imag], axis=-1).astype(np.float32)


def stft_unlift(feats: np.ndarray, patch_size: int) -> np.ndarray:
    """Inverse of :func:`stft_lift` — recover real patches from real/imag."""
    half = feats.shape[-1] // 2
    spec = feats[..., :half] + 1j * feats[..., half:]
    return np.fft.irfft(spec, n=patch_size, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-channel mode decomposition (reuses the oracle-probe spatial DFT)
# ---------------------------------------------------------------------------


def mode_decomposition(coil_array: np.ndarray, n_modes: int = 4) -> np.ndarray:
    """Spatial-DFT mode decomposition across an angular probe array.

    For a ``(T, n_coils)`` array of toroidally/poloidally distributed coils,
    compute the spatial DFT across coils at each time and return the complex
    amplitude of modes ``0 .. n_modes-1`` as interleaved real/imag:
    output shape ``(T, 2 * n_modes)``.

    This is the same decomposition the interior-information oracle probe uses
    (``oracle_probe._xma_mirnov_block`` spatial-DFT block) — reused here so the
    mode-number tokens are consistent with that study's mode features, but
    here phase is kept (real + imag) rather than magnitude only.
    """
    a = np.asarray(coil_array, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"coil_array must be (T, n_coils); got {a.shape}")
    # Subtract the per-time mean (the m=0 offset) so the spatial DFT resolves
    # the structured modes, mirroring oracle_probe's `snap - snap.mean()`.
    a = a - np.nanmean(a, axis=1, keepdims=True)
    a = np.nan_to_num(a, nan=0.0)
    spec = np.fft.rfft(a, axis=1)  # (T, n_coils//2 + 1) complex
    k = min(n_modes, spec.shape[1])
    out = np.zeros((a.shape[0], 2 * n_modes), dtype=np.float32)
    out[:, :k] = spec[:, :k].real
    out[:, n_modes : n_modes + k] = spec[:, :k].imag
    return out


# ---------------------------------------------------------------------------
# Bottlenecks
# ---------------------------------------------------------------------------


def _make_fsq(levels):
    import torch
    import torch.nn as nn

    class FSQ(nn.Module):
        """Finite-scalar quantisation (Mentzer et al. 2023).

        Each latent dim is bounded by tanh then rounded to one of
        ``levels[d]`` levels with a straight-through gradient.  No codebook,
        so no codebook collapse; the effective vocab is ``prod(levels)``.
        Phase survives quantisation only up to the level resolution — the
        point of the comparison is to measure exactly how much.
        """

        def __init__(self):
            super().__init__()
            self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
            self.n_dim = len(levels)
            self.codebook_size = int(np.prod(levels))

        def forward(self, z):
            # z: (..., n_dim).  Bound to [-1, 1], scale to half-level range.
            half = (self.levels - 1) / 2
            zb = torch.tanh(z) * half
            zq = zb + (torch.round(zb) - zb).detach()  # straight-through
            codes = self._to_codes(zq)
            return zq / half, codes, z.new_zeros(())  # normalised, ids, no aux loss

        def _to_codes(self, zq):
            half = (self.levels - 1) / 2
            idx = (zq + half).round().long().clamp_min(0)
            acc = 1.0
            mult = []
            for lev in self.levels.tolist():
                mult.append(acc)
                acc *= lev
            mult_t = torch.tensor(mult, device=zq.device)
            return (idx.float() * mult_t).sum(dim=-1).long()

    return FSQ()


def _make_vq(codebook_size, dim, commitment):
    import torch
    import torch.nn as nn

    class VQ(nn.Module):
        """Vector-quantised bottleneck with a learned codebook (VQ-VAE).

        Straight-through estimator with the standard codebook + commitment
        loss.  Codebook collapse is the known failure mode — reported via the
        active-code count at eval.
        """

        def __init__(self):
            super().__init__()
            self.codebook = nn.Embedding(codebook_size, dim)
            self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
            self.commitment = commitment
            self.codebook_size = codebook_size

        def forward(self, z):
            flat = z.reshape(-1, z.shape[-1])
            d = (
                flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1)
            )
            codes = d.argmin(1)
            zq = self.codebook(codes).view_as(z)
            cb_loss = torch.nn.functional.mse_loss(zq, z.detach())
            commit = torch.nn.functional.mse_loss(zq.detach(), z)
            aux = cb_loss + self.commitment * commit
            zq = z + (zq - z).detach()  # straight-through
            return zq, codes.view(z.shape[:-1]), aux

    return VQ()


def _make_continuous(dim):
    import torch.nn as nn

    class Continuous(nn.Module):
        """Continuous-embedding control bottleneck (no quantisation).

        Returns the latent unchanged with a zero id-stream.  This is the
        fidelity ceiling for the comparison: if FSQ/VQ phase error is far
        above this, quantisation is destroying phase and we should prefer the
        continuous+mask representation.
        """

        def __init__(self):
            super().__init__()
            self.codebook_size = 0
            self.continuous_dim = dim

        def forward(self, z):
            import torch

            ids = z.new_zeros(z.shape[:-1], dtype=torch.long)
            return z, ids, z.new_zeros(())

    return Continuous()


# ---------------------------------------------------------------------------
# The autoencoder
# ---------------------------------------------------------------------------


def build_model(cfg: PatchTokenizerConfig):
    """Construct the patch-transformer autoencoder ``nn.Module``.

    Built lazily (torch imported here) so the module imports on CPU-only
    nodes without a torch GPU init.
    """
    import torch
    import torch.nn as nn

    patch_feat_dim = 2 * (cfg.patch_size // 2 + 1) if cfg.use_stft else cfg.patch_size

    if cfg.bottleneck == "fsq":
        bottleneck = _make_fsq(cfg.fsq_levels)
        latent_dim = bottleneck.n_dim
    elif cfg.bottleneck == "vq":
        latent_dim = cfg.continuous_dim
        bottleneck = _make_vq(cfg.vq_codebook_size, latent_dim, cfg.vq_commitment)
    elif cfg.bottleneck == "continuous":
        latent_dim = cfg.continuous_dim
        bottleneck = _make_continuous(latent_dim)
    else:
        raise ValueError(f"unknown bottleneck {cfg.bottleneck!r}")

    def encoder_layer():
        return nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    class PatchTransformerAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.patch_feat_dim = patch_feat_dim
            self.embed = nn.Linear(patch_feat_dim, cfg.d_model)
            self.encoder = nn.TransformerEncoder(
                encoder_layer(), cfg.n_layers, enable_nested_tensor=False
            )
            self.to_latent = nn.Linear(cfg.d_model, latent_dim)
            self.bottleneck = bottleneck
            self.from_latent = nn.Linear(latent_dim, cfg.d_model)
            self.decoder = nn.TransformerEncoder(
                encoder_layer(), cfg.n_layers, enable_nested_tensor=False
            )
            self.unembed = nn.Linear(cfg.d_model, patch_feat_dim)
            self.register_buffer("_posbase", torch.zeros(1), persistent=False)

        def _pos(self, n, device):
            # Sinusoidal positional encoding over patch positions.
            pos = torch.arange(n, device=device).float().unsqueeze(1)
            i = torch.arange(cfg.d_model, device=device).float().unsqueeze(0)
            angle = pos / torch.pow(10000.0, (2 * (i // 2)) / cfg.d_model)
            pe = torch.zeros(n, cfg.d_model, device=device)
            pe[:, 0::2] = torch.sin(angle[:, 0::2])
            pe[:, 1::2] = torch.cos(angle[:, 1::2])
            return pe.unsqueeze(0)

        def encode(self, feats):
            # feats: (B, n_patches, patch_feat_dim)
            h = self.embed(feats)
            h = h + self._pos(h.shape[1], h.device)
            h = self.encoder(h)
            z = self.to_latent(h)
            zq, ids, aux = self.bottleneck(z)
            return zq, ids, aux

        def decode(self, zq):
            h = self.from_latent(zq)
            h = h + self._pos(h.shape[1], h.device)
            h = self.decoder(h)
            return self.unembed(h)

        def forward(self, feats):
            zq, ids, aux = self.encode(feats)
            recon = self.decode(zq)
            return recon, ids, aux

    return PatchTransformerAE()


# ---------------------------------------------------------------------------
# Phase-fidelity metrics (the codebook decision is MEASURED with these)
# ---------------------------------------------------------------------------


def reconstruction_crps(truth: np.ndarray, recon: np.ndarray) -> float:
    """Deterministic-forecast CRPS == mean absolute error of the reconstruction.

    For a point reconstruction the CRPS reduces to the MAE; reported in the
    signal's native units so it is comparable across bottlenecks.
    """
    t = np.asarray(truth, dtype=np.float64).reshape(-1)
    r = np.asarray(recon, dtype=np.float64).reshape(-1)
    m = np.isfinite(t) & np.isfinite(r)
    return float(np.mean(np.abs(t[m] - r[m]))) if m.any() else float("nan")


def phase_error(
    truth: np.ndarray,
    recon: np.ndarray,
    band: tuple[float, float] | None = None,
    dt: float = 1.0,
) -> float:
    """Mean absolute spectral phase error (radians) in an optional band.

    Computes the rFFT of truth and reconstruction along the last axis and
    returns the mean ``|Δphase|`` (wrapped to ``[-π, π]``) over bins whose
    truth magnitude is non-negligible.  This is the metric that distinguishes
    a phase-preserving bottleneck from a magnitude-only one — a tokenizer can
    have low amplitude error yet scramble phase.
    """
    t = np.asarray(truth, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    ft = np.fft.rfft(t, axis=-1)
    fr = np.fft.rfft(r, axis=-1)
    freqs = np.fft.rfftfreq(t.shape[-1], d=dt)
    dphi = np.angle(ft) - np.angle(fr)
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # wrap to [-pi, pi]
    mag = np.abs(ft)
    thresh = 0.05 * mag.max() if mag.size and mag.max() > 0 else 0.0
    mask = mag > thresh
    if band is not None:
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        mask = mask & band_mask[tuple([None] * (mask.ndim - 1) + [slice(None)])]
    return float(np.mean(np.abs(dphi[mask]))) if mask.any() else float("nan")


def mode_number_recovery(
    coil_truth: np.ndarray, coil_recon: np.ndarray, n_modes: int = 4
) -> dict[str, float]:
    """Mode-number recovery: per-mode amplitude + phase agreement.

    Decomposes truth and reconstruction with :func:`mode_decomposition` and
    reports, per mode ``1..n_modes-1`` (skipping the m=0 offset), the complex
    correlation magnitude (amplitude+phase agreement, 1.0 == perfect) and the
    mean absolute mode-phase error (radians).  A tokenizer that destroys phase
    shows mode-phase error → π/2-ish even with good amplitude.
    """
    mt = mode_decomposition(coil_truth, n_modes)
    mr = mode_decomposition(coil_recon, n_modes)
    out: dict[str, float] = {}
    corrs = []
    phase_errs = []
    for m in range(1, n_modes):
        ct = mt[:, m] + 1j * mt[:, n_modes + m]
        cr = mr[:, m] + 1j * mr[:, n_modes + m]
        denom = np.sqrt((np.abs(ct) ** 2).sum() * (np.abs(cr) ** 2).sum())
        corr = np.abs((ct * np.conj(cr)).sum()) / denom if denom > 0 else np.nan
        dphi = np.angle(ct) - np.angle(cr)
        dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
        w = np.abs(ct)
        pe = (
            float(np.sum(np.abs(dphi) * w) / np.sum(w)) if w.sum() > 0 else float("nan")
        )
        out[f"mode_{m}_complex_corr"] = float(corr)
        out[f"mode_{m}_phase_err"] = pe
        if np.isfinite(corr):
            corrs.append(corr)
        if np.isfinite(pe):
            phase_errs.append(pe)
    out["mean_complex_corr"] = float(np.mean(corrs)) if corrs else float("nan")
    out["mean_mode_phase_err"] = (
        float(np.mean(phase_errs)) if phase_errs else float("nan")
    )
    return out


# ---------------------------------------------------------------------------
# Trainer / encoder wrapper
# ---------------------------------------------------------------------------


@dataclass
class PatchTransformerTokenizer:
    """Trainable phase-aware patch-transformer tokenizer.

    Holds a config + (after :meth:`fit`) a trained autoencoder.  Encodes a
    native-cadence ``(C, T)`` signal window into per-patch codes plus the
    per-channel validity needed for the v2 store.  The model is loaded once
    (in :meth:`fit` / :meth:`load`) and reused across many windows — the
    in-process performant pattern (repo §2b).
    """

    cfg: PatchTokenizerConfig = field(default_factory=PatchTokenizerConfig)
    name: str = "signal_hf_xma_patch_v2"
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._model = None

    # -- feature prep -------------------------------------------------------

    def _normalise(
        self,
        x: np.ndarray,
        fit: bool,
        *,
        channel_names: Sequence[str] | None = None,
        corpus_calibration: dict[str, ChannelCalibration] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-channel z-score → ``(z, means, stds)`` (channel-count agnostic).

        Two modes:

        - **Per-window (default, ``corpus_calibration is None``):** each
          channel is standardised against *this window's* own per-channel
          mean/std.  This is amplitude-relative — the same physical value maps
          to a different code in every window — but robust to a shot having a
          different channel inventory.
        - **Absolute (``corpus_calibration`` supplied):** each channel
          (matched by ``channel_names[row]``) is standardised against its
          CORPUS mean/std, so the same physical value maps to the same code in
          every window and on every machine.  A channel with no calibration
          entry falls back to its per-window stats (with a one-line warning).

        IMPORTANT: absolute mode changes the *input distribution* the
        autoencoder sees.  A codebook trained under per-window normalisation
        is NOT valid for absolute mode and must be retrained.

        Returns ``(z, means, stds)`` so the caller can de-normalise the
        reconstruction with the SAME stats.
        """
        with np.errstate(invalid="ignore"):
            means = np.nan_to_num(np.nanmean(x, axis=1, keepdims=True))
            stds = np.nanstd(x, axis=1, keepdims=True)
        stds = np.where((stds > 1e-9) & np.isfinite(stds), stds, 1.0)

        if corpus_calibration is not None:
            names = list(channel_names) if channel_names is not None else []
            for row in range(x.shape[0]):
                name = names[row] if row < len(names) else None
                cal = corpus_calibration.get(name) if name is not None else None
                if cal is None:
                    logger.warning(
                        "PatchTransformerTokenizer: no corpus calibration for "
                        "channel %r — falling back to per-window stats "
                        "(absolute magnitude not preserved for this channel)",
                        name,
                    )
                    continue
                means[row, 0] = float(cal.mean)
                std = float(cal.std)
                stds[row, 0] = std if std > 1e-9 else 1.0

        z = np.nan_to_num((x - means) / stds, nan=0.0)
        return z, means, stds

    def _features(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        patches, n_pad = patchify(x, self.cfg.patch_size)  # (C, P, ps)
        feats = stft_lift(patches) if self.cfg.use_stft else patches
        return feats, n_pad

    # -- training -----------------------------------------------------------

    def fit(
        self,
        windows: list[np.ndarray],
        *,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_channels: int = 256,
        seed: int = 0,
        log_every: int = 5,
        logger=None,
    ) -> dict:
        """Train the autoencoder on a list of ``(C, T)`` signal windows.

        Each window's channels are normalised, patchified, optionally
        STFT-lifted, then fed as a sequence batch.  Reconstruction is in
        feature space (STFT real/imag if ``use_stft``) so phase is optimised
        directly.  Returns a small history dict.
        """
        import torch

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._model = build_model(self.cfg).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)

        # Pre-compute uniform-length training samples.  Each window's channels
        # are patchified, then split into fixed-length segments of
        # ``seq_patches`` patches so shots of different duration yield
        # stackable, equal-length sequences (the transformer is length-agnostic
        # per batch; a uniform length just lets us stack into one array).
        seq = self.cfg.seq_patches
        all_feats: list[np.ndarray] = []
        for w in windows:
            z, _, _ = self._normalise(np.asarray(w, dtype=np.float32), fit=False)
            feats, _ = self._features(z)  # (C, P, feat)
            n_patches = feats.shape[1]
            n_seg = n_patches // seq
            if n_seg == 0:
                # Shorter than one segment — pad the patch axis up to seq.
                pad = seq - n_patches
                feats = np.pad(feats, ((0, 0), (0, pad), (0, 0)))
                n_seg, n_patches = 1, seq
            for c in range(feats.shape[0]):
                for s in range(n_seg):
                    all_feats.append(feats[c, s * seq : (s + 1) * seq])
        if not all_feats:
            raise RuntimeError("no training segments — windows too short or empty")
        all_feats_arr = np.stack(all_feats, axis=0)  # (N, seq, feat)

        hist = {"loss": [], "recon": [], "aux": []}
        n = all_feats_arr.shape[0]
        for ep in range(epochs):
            perm = rng.permutation(n)
            ep_loss = ep_recon = ep_aux = 0.0
            nb = 0
            for i in range(0, n, batch_channels):
                idx = perm[i : i + batch_channels]
                fb = torch.tensor(all_feats_arr[idx], device=self.device)
                recon, _ids, aux = self._model(fb)
                recon_loss = torch.nn.functional.mse_loss(recon, fb)
                loss = recon_loss + aux
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach())
                ep_recon += float(recon_loss.detach())
                ep_aux += float(aux.detach())
                nb += 1
            hist["loss"].append(ep_loss / max(nb, 1))
            hist["recon"].append(ep_recon / max(nb, 1))
            hist["aux"].append(ep_aux / max(nb, 1))
            if logger is not None and (ep % log_every == 0 or ep == epochs - 1):
                logger.info(
                    "epoch %d/%d loss=%.4e recon=%.4e aux=%.4e",
                    ep + 1,
                    epochs,
                    hist["loss"][-1],
                    hist["recon"][-1],
                    hist["aux"][-1],
                )
        return hist

    # -- inference ----------------------------------------------------------

    def _reconstruct_feats(
        self, feats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        self._model.eval()
        with torch.no_grad():
            fb = torch.tensor(feats, device=self.device)
            zq, ids, _ = self._model.encode(fb)
            recon = self._model.decode(zq)
        return recon.cpu().numpy(), ids.cpu().numpy(), zq.cpu().numpy()

    def encode_window(
        self,
        x: np.ndarray,
        *,
        channel_names: Sequence[str] | None = None,
        corpus_calibration: dict[str, ChannelCalibration] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a ``(C, T)`` window → ``(codes (C,P), latent (C,P,d), recon)``.

        ``codes`` are local per-patch ids (0 for the continuous bottleneck);
        ``latent`` is the per-patch bottleneck embedding — for the continuous
        bottleneck this carries the phase-preserving payload (the discrete ids
        are vestigial), so the store persists it alongside the ids.  ``recon``
        is the reconstructed signal in native units (de-normalised).

        When ``corpus_calibration`` is supplied, each channel (matched by
        ``channel_names[row]``) is standardised against its CORPUS mean/std so
        absolute magnitude survives tokenisation — see :meth:`_normalise`.
        Absolute mode changes the input distribution; the codebook must be
        retrained under absolute normalisation for the codes to be valid.
        With ``corpus_calibration=None`` (default) behaviour is byte-identical
        to per-window normalisation.
        """
        if self._model is None:
            raise RuntimeError("call fit() or load() before encode_window()")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        z, means, stds = self._normalise(
            x,
            fit=False,
            channel_names=channel_names,
            corpus_calibration=corpus_calibration,
        )
        feats, n_pad = self._features(z)  # (C, P, feat)
        recon_feats, ids, latent = self._reconstruct_feats(feats)
        if self.cfg.use_stft:
            recon_patches = stft_unlift(recon_feats, self.cfg.patch_size)
        else:
            recon_patches = recon_feats
        recon_z = unpatchify(recon_patches, n_pad)  # (C, T) normalised
        recon = recon_z * stds + means  # de-normalise with the SAME per-window stats
        return (
            ids.astype(np.int64),
            latent.astype(np.float32),
            recon.astype(np.float32),
        )

    def roundtrip_metrics(
        self, x: np.ndarray, *, dt: float, is_coil_array: bool = False
    ) -> dict:
        """Full phase-fidelity QC for one window — the codebook-decision data.

        Returns reconstruction CRPS, banded phase error, active-code count,
        and (for a coil array) mode-number recovery.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        ids, _latent, recon = self.encode_window(x)
        out: dict[str, float] = {
            "recon_crps": reconstruction_crps(x, recon),
            "phase_err": phase_error(x, recon, dt=dt),
            "n_active_codes": int(np.unique(ids).size),
            "codebook_size": int(getattr(self._model.bottleneck, "codebook_size", 0)),
        }
        if is_coil_array:
            # mode decomposition expects (T, n_coils)
            out.update(mode_number_recovery(x.T, recon.T))
        return out

    # -- persistence of the trained weights ---------------------------------

    def save(self, path) -> None:
        import torch

        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "state_dict": self._model.state_dict(),
                "name": self.name,
            },
            str(path),
        )

    def load(self, path) -> None:
        import torch

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.cfg = PatchTokenizerConfig(**ckpt["cfg"])
        self._model = build_model(self.cfg).to(self.device)
        self._model.load_state_dict(ckpt["state_dict"])
        self.name = ckpt["name"]
