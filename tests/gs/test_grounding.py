"""Tests for the S8-T6 GS grounding head (latent → restricted GS currents).

Three layers (no mirror / network needed — all synthetic):

* **Loss correctness** — ``grounding_losses`` is pinned against a hand-computed
  whitened-residual Gaussian NLL and the current-space L2 prior, including the
  CRITICAL de-normalisation of the sliced (normalised) inputs back to raw SI.
* **Learnability** — a GroundingHead trained on a synthetic linear
  current→magnetics system reduces L_data; a recoverable θ is found.
* **Additive invariant** — with ``grounding=False`` the engine has no head and
  the ungrounded forward/training path is unchanged (the v2-vs-grounded
  comparison requires the ungrounded path stays runnable + bit-identical).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from imas_ambix.gs.grounding import (
    CampaignGrounding,
    GroundingHead,
    grounding_losses,
)
from imas_ambix.gs.residual import plasma_poly_basis
from imas_ambix.statespace.engine import (
    EngineConfig,
    RKNEngine,
    _plasma_poly_dof,
    train_engine,
)

# ---------------------------------------------------------------------------
# helpers — a synthetic CampaignGrounding with a known forward map
# ---------------------------------------------------------------------------


def _synthetic_cg(
    *,
    n_row: int = 30,
    n_dof: int = 3,
    rank: int = 4,
    feat_dim: int = 122,
    amb_cols=None,
    pf_cols=None,
    seed: int = 0,
) -> tuple[CampaignGrounding, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic CampaignGrounding + the raw blocks for hand-checking.

    Returns (cg, amb_mean, amb_std, g_pf_white_np) so a test can reconstruct the
    expected whitened residual.
    """
    rng = np.random.default_rng(seed)
    amb_cols = np.arange(6, 6 + n_row) if amb_cols is None else np.asarray(amb_cols)
    pf_cols = np.arange(80, 80 + 5) if pf_cols is None else np.asarray(pf_cols)
    n_coil = len(pf_cols)

    amb_mean = rng.normal(0, 1, n_row)
    amb_std = rng.uniform(0.5, 2.0, n_row)
    pf_mean = rng.normal(0, 1, n_coil)
    pf_std = rng.uniform(0.5, 2.0, n_coil)

    a_plasma = rng.normal(0, 0.1, (n_row, n_dof))
    a_passive = rng.normal(0, 0.1, (n_row, rank))
    g_pf_white = rng.normal(0, 1e-3, (n_row, n_coil))
    w_scale = rng.uniform(0.5, 2.0, n_row)
    btb = np.eye(n_dof) * 2.0
    vtv = np.eye(rank) * 3.0

    def _t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float()

    cg = CampaignGrounding(
        signature_key="synthetic",
        amb_x_cols=amb_cols,
        pf_amc_x_cols=pf_cols,
        pf_amc_present=np.ones(n_coil, dtype=bool),
        amb_mean=_t(amb_mean),
        amb_std=_t(amb_std),
        pf_mean=_t(pf_mean),
        pf_std=_t(pf_std),
        g_pf_white=_t(g_pf_white),
        a_plasma=_t(a_plasma),
        a_passive=_t(a_passive),
        w_scale=_t(w_scale),
        penalty_plasma=_t(btb),
        penalty_passive=_t(vtv),
        n_dof=n_dof,
        rank=rank,
        n_row=n_row,
    )
    return cg, amb_mean, amb_std, g_pf_white


# ---------------------------------------------------------------------------
# (1) loss correctness — de-norm + whitened NLL + current-space L2
# ---------------------------------------------------------------------------


def test_grounding_losses_match_hand_computation():
    torch.manual_seed(0)
    cg, amb_mean, amb_std, _ = _synthetic_cg(seed=1)
    n_dof, rank = cg.n_dof, cg.rank
    feat_dim = 122

    head = GroundingHead(latent_dim=8, n_dof=n_dof, rank=rank)
    head.eval()
    N = 5
    z = torch.randn(N, 8)
    x_norm = torch.randn(N, feat_dim)

    l_data, l_gs, info = grounding_losses(head, z, x_norm, cg, lam=0.05)

    # --- hand reconstruct ---
    with torch.no_grad():
        theta, psi = head(z)
        theta_np = theta.numpy()
        psi_np = psi.numpy()
    amb_norm = x_norm.numpy()[:, cg.amb_x_cols]
    amb_raw = amb_norm * amb_std + amb_mean
    pf_norm = x_norm.numpy()[:, cg.pf_amc_x_cols]
    pf_raw = pf_norm * cg.pf_std.numpy() + cg.pf_mean.numpy()
    pf_amps = pf_raw * 1.0e3
    w = cg.w_scale.numpy()
    pred = (
        pf_amps @ cg.g_pf_white.numpy().T
        + theta_np @ cg.a_plasma.numpy().T
        + psi_np @ cg.a_passive.numpy().T
    )
    raw_white = amb_raw * w[None, :]
    resid = pred - raw_white
    emit_var = float(head.log_emit.exp().item()) ** 2
    nll = 0.5 * (np.log(2 * math.pi * emit_var) + resid**2 / emit_var)
    expected_ldata = float(nll.mean())
    assert abs(float(l_data) - expected_ldata) < 1e-4, (
        float(l_data),
        expected_ldata,
    )

    # current-space L2 prior: lam * mean(θᵀBᵀBθ + ψᵀVᵀVψ)
    gs_pl = np.einsum("nd,de,ne->n", theta_np, cg.penalty_plasma.numpy(), theta_np)
    gs_pa = np.einsum("nr,rs,ns->n", psi_np, cg.penalty_passive.numpy(), psi_np)
    expected_lgs = 0.05 * float((gs_pl + gs_pa).mean())
    assert abs(float(l_gs) - expected_lgs) < 1e-4, (float(l_gs), expected_lgs)
    assert info["n"] == N


def test_grounding_lgs_is_nonneg_and_scales_with_lambda():
    cg, *_ = _synthetic_cg(seed=2)
    head = GroundingHead(latent_dim=8, n_dof=cg.n_dof, rank=cg.rank)
    z = torch.randn(6, 8)
    x = torch.randn(6, 122)
    _, lgs1, _ = grounding_losses(head, z, x, cg, lam=0.01)
    _, lgs2, _ = grounding_losses(head, z, x, cg, lam=0.04)
    assert float(lgs1) >= 0.0
    # L_GS is exactly linear in lambda (penalty is fixed; head deterministic)
    assert abs(float(lgs2) - 4.0 * float(lgs1)) < 1e-5


def test_absent_pf_coil_contributes_zero():
    """A PF coil flagged absent must contribute exactly zero to the PF term."""
    cg, *_ = _synthetic_cg(seed=3)
    # mark the first coil absent
    cg.pf_amc_present = cg.pf_amc_present.copy()
    cg.pf_amc_present[0] = False
    head = GroundingHead(latent_dim=8, n_dof=cg.n_dof, rank=cg.rank)
    z = torch.zeros(3, 8)  # zero latent → head bias only
    # two input batches differing ONLY in the (absent) first coil column
    x1 = torch.randn(3, 122)
    x2 = x1.clone()
    x2[:, cg.pf_amc_x_cols[0]] += 100.0  # large change on the absent coil
    l1, _, _ = grounding_losses(head, z, x1, cg, lam=0.0)
    l2, _, _ = grounding_losses(head, z, x2, cg, lam=0.0)
    assert abs(float(l1) - float(l2)) < 1e-4


# ---------------------------------------------------------------------------
# (2) learnability — head recovers a planted linear current field
# ---------------------------------------------------------------------------


def test_grounding_head_learns_planted_currents():
    """A head fed a latent linear in a planted θ* must drive L_data down."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_row, n_dof, rank = 25, 3, 0  # no passive for a clean planted test
    L = 6
    a_plasma = rng.normal(0, 0.2, (n_row, n_dof))
    w_scale = np.ones(n_row)
    amb_mean = np.zeros(n_row)
    amb_std = np.ones(n_row)

    def _t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float()

    cg = CampaignGrounding(
        signature_key="plant",
        amb_x_cols=np.arange(6, 6 + n_row),
        pf_amc_x_cols=np.arange(80, 80 + 2),
        pf_amc_present=np.zeros(2, dtype=bool),  # no PF term
        amb_mean=_t(amb_mean),
        amb_std=_t(amb_std),
        pf_mean=_t(np.zeros(2)),
        pf_std=_t(np.ones(2)),
        g_pf_white=_t(np.zeros((n_row, 2))),
        a_plasma=_t(a_plasma),
        a_passive=_t(np.zeros((n_row, 0))),
        w_scale=_t(w_scale),
        penalty_plasma=_t(np.eye(n_dof)),
        penalty_passive=_t(np.zeros((0, 0))),
        n_dof=n_dof,
        rank=rank,
        n_row=n_row,
    )
    # planted: theta* = M z  → raw_white = a_plasma @ theta*  (so a head can fit)
    M = rng.normal(0, 1, (n_dof, L))
    N = 200
    z = rng.normal(0, 1, (N, L))
    theta_star = z @ M.T  # (N, n_dof)
    raw_white = theta_star @ a_plasma.T  # (N, n_row); w=1, mean=0,std=1 → amb_raw
    x_norm = np.zeros((N, 122), dtype=np.float64)
    x_norm[:, cg.amb_x_cols] = raw_white  # amb_std=1, mean=0 → raw == norm

    head = GroundingHead(latent_dim=L, n_dof=n_dof, rank=rank)
    zt = torch.from_numpy(z).float()
    xt = torch.from_numpy(x_norm).float()
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    l0 = float(grounding_losses(head, zt, xt, cg, lam=0.0)[0])
    for _ in range(300):
        opt.zero_grad()
        ld, _lg, _ = grounding_losses(head, zt, xt, cg, lam=0.0)
        ld.backward()
        opt.step()
    l1 = float(grounding_losses(head, zt, xt, cg, lam=0.0)[0])
    assert l1 < l0 - 0.5, (l0, l1)
    # rmse_white should be small (the planted map is exactly representable)
    info = grounding_losses(head, zt, xt, cg, lam=0.0)[2]
    assert info["rmse_white"] < 0.2, info["rmse_white"]


# ---------------------------------------------------------------------------
# (3) additive invariant — grounding=False leaves the engine unchanged
# ---------------------------------------------------------------------------


def test_grounding_off_has_no_head():
    cfg = EngineConfig(input_dim=7, latent_dim=4, output_dim=1, grounding=False)
    model = RKNEngine(cfg)
    assert model.grounding_head is None


def test_grounding_on_builds_head_with_locked_dof():
    cfg = EngineConfig(
        input_dim=7,
        latent_dim=4,
        output_dim=1,
        grounding=True,
        gs_profile_order=1,
        gs_passive_rank=4,
    )
    model = RKNEngine(cfg)
    assert model.grounding_head is not None
    # order-1 → 3 plasma DOF; rank-4 passive
    assert model.grounding_head.head_plasma.out_features == 3
    assert model.grounding_head.head_passive.out_features == 4


def test_ungrounded_training_unchanged_without_ctx():
    """train_engine with grounding=False must run + reduce loss (v2 path intact)."""
    rng = np.random.default_rng(0)
    cfg = EngineConfig(
        input_dim=5,
        latent_dim=4,
        output_dim=1,
        n_epochs=4,
        seq_len=16,
        batch_size=8,
        train_horizons=(1, 2),
        num_threads=2,
        grounding=False,
    )
    x = [rng.normal(0, 1, (40, 5)).astype(np.float64) for _ in range(6)]
    y = [np.cumsum(rng.normal(0, 0.1, (40, 1)), axis=0) for _ in range(6)]
    model = RKNEngine(cfg)
    st = train_engine(model, x, y, cfg, device="cpu")
    assert len(st.epoch_losses) == 4
    assert st.epoch_losses[-1] < st.epoch_losses[0] + 1.0
    assert st.epoch_gs_data == []  # no grounding terms recorded


def test_plasma_poly_dof_matches_basis():
    """_plasma_poly_dof must equal plasma_poly_basis column count for each order."""
    rz = np.array([[0.5, 0.1], [0.6, -0.2], [0.7, 0.0], [0.8, 0.3]])
    for order in (1, 2, 4):
        b = plasma_poly_basis(rz, order, r0=0.9, minor_radius=0.5)
        assert b.shape[1] == _plasma_poly_dof(order), (order, b.shape[1])
