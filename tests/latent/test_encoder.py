"""Tests for the hybrid latent encoder (anchored ⊕ free ⊕ ψ).

The encoder maps the absolute-calibrated, geometry-tagged input features to the
shared hybrid latent:

* an **anchored block** supervised only on genuinely raw measured scalars
  (Ip, coil currents, line-averaged density) — the physics handle;
* the **plasma-current profile amplitudes θ** (the ψ representation the GS
  observation operator reads out);
* a **free block** for the closure physics (feeds the transport prior's learned
  η∥ / sources).

Coordinates are dimensionless (learned-soft, resolving the open
``dimensionless-coordinate-set`` decision): the encoder learns the map and a
soft standardisation regulariser nudges the anchored block toward zero-mean /
unit-variance dimensionless coordinates without hard-imposing a fixed basis.
"""

from __future__ import annotations

import torch

from imas_ambix.latent.encoder import HybridLatentEncoder, LatentConfig


def _cfg(**kw):
    base = dict(n_features=32, n_theta=3, n_anchored=4, n_free=8, hidden=32, depth=2)
    base.update(kw)
    return LatentConfig(**base)


def test_encoder_output_shapes():
    cfg = _cfg()
    enc = HybridLatentEncoder(cfg)
    x = torch.randn(5, cfg.n_features)
    lat = enc(x)
    assert lat.theta.shape == (5, cfg.n_theta)
    assert lat.anchored.shape == (5, cfg.n_anchored)
    assert lat.free.shape == (5, cfg.n_free)


def test_anchored_block_can_fit_raw_scalars():
    """The anchored head must be able to regress raw measured scalars."""
    torch.manual_seed(0)
    cfg = _cfg()
    enc = HybridLatentEncoder(cfg)
    x = torch.randn(64, cfg.n_features)
    # a fixed linear target so the map is learnable
    w = torch.randn(cfg.n_features, cfg.n_anchored)
    y = x @ w
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    lat0 = enc(x)
    loss0 = enc.anchored_loss(lat0, y).item()
    for _ in range(200):
        opt.zero_grad()
        loss = enc.anchored_loss(enc(x), y)
        loss.backward()
        opt.step()
    loss1 = enc.anchored_loss(enc(x), y).item()
    assert loss1 < 0.25 * loss0  # the anchored block genuinely learns


def test_anchored_loss_respects_mask():
    """Masked (missing) raw-scalar targets must not contribute to the loss."""
    cfg = _cfg()
    enc = HybridLatentEncoder(cfg)
    x = torch.randn(3, cfg.n_features)
    lat = enc(x)
    y = torch.zeros(3, cfg.n_anchored)
    mask = torch.zeros(3, cfg.n_anchored, dtype=torch.bool)  # nothing observed
    loss = enc.anchored_loss(lat, y, mask=mask)
    assert loss.item() == 0.0  # no observed target → zero contribution


def test_dimensionless_regulariser_is_nonneg_and_small_when_standardised():
    cfg = _cfg(n_anchored=6)
    enc = HybridLatentEncoder(cfg)
    lat = enc(torch.randn(128, cfg.n_features))
    reg = enc.dimensionless_regulariser(lat)
    assert reg.item() >= 0.0
    # a genuinely standardised block → ~0 penalty
    from imas_ambix.latent.encoder import HybridLatent

    std = torch.randn(256, cfg.n_anchored)
    std = (std - std.mean(0)) / std.std(0)
    lat_std = HybridLatent(
        theta=lat.theta, anchored=std, free=lat.free, free_logvar=None
    )
    assert enc.dimensionless_regulariser(lat_std).item() < 0.05


def test_probabilistic_free_block_exposes_belief():
    cfg = _cfg(probabilistic=True)
    enc = HybridLatentEncoder(cfg)
    lat = enc(torch.randn(4, cfg.n_features))
    assert lat.free_logvar is not None
    assert lat.free_logvar.shape == (4, cfg.n_free)
    kl = enc.kl_free_bits(lat, free_bits=0.1)
    assert torch.isfinite(kl) and kl.item() >= 0.0
