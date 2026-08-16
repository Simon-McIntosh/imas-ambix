"""GS grounding head: latent z → restricted GS currents → predicted raw magnetics.

The GS prior earns its place here by GROUNDING the RKN latent in raw magnetics
through the forward operator, not by detecting transients.  Read as a standalone
detector (:mod:`imas_ambix.gs.residual`) the GS force-balance residual is
physics-native but loses to a trivial ``|dB/dt|`` baseline, so detection is not
what this head is judged on.

What this module is
-------------------
A torch ``GroundingHead`` that maps the RKN latent ``z`` (L-d) to the INFERRED
GS currents in the restricted basis and pushes them through the forward
operator to predict the RAW trustworthy magnetics:

    θ_plasma (n_dof)   = head_plasma(z)          # 3 DOF at profile order-1
    ψ_passive (rank)   = head_passive(z)         # rank-4 passive SVD modes
    c_plasma           = B_poly · θ_plasma       # current-space jφ(R,Z) field
    c_passive          = V_passive · ψ_passive   # passive/eddy nuisance currents
    pred_trust_raw     = G_pf · i_pf  +  G_plasma · c_plasma  +  G_passive · c_passive

compared against the RAW ``amb`` at the 76 B-probes + 1 clean flux loop (the
``trustworthy_target``; flagged/excluded loops never enter the comparison).

The restricted basis is the SAME one the standalone monitor's
``InverseSolver`` uses (``residual.plasma_poly_basis`` order-1 → 3 DOF;
``residual.passive_lowrank_basis`` rank-4), with the SAME per-sensor robust
whitening ``W`` (``residual.robust_sensor_scale``).  By emitting EXACTLY those
DOF, the head inherits the structural near-vacuum soundness the frontier found
(the q1 monitor passed near-vacuum at λ=0 with order-1/rank-4: net-current
ratio 0.030 ≤ 0.25), so the 0.99 plasma/passive collinearity is controlled
STRUCTURALLY (low DOF + low-rank passive), not by brute λ — exactly what the
task prefers.

Two objective terms (added to the engine's joint loss in ``engine.py``)
-----------------------------------------------------------------------
* ``L_data`` — Gaussian NLL of the WHITENED raw-magnetics reconstruction:
  ``W·(raw_trust − G_pf·i_pf)`` vs ``A_plasma·θ + A_passive·ψ`` with a single
  learned emission log-scale.  Whitening mixes the 76 B-probe (Tesla) rows + 1
  flux-loop (Wb) row coherently (same ``W`` as the standalone residual).
* ``L_GS`` — the GS force-balance soft prior: the PHYSICAL-amplitude Tikhonov
  ``θᵀ(BᵀB)θ + ψᵀ(VᵀV)ψ`` (current-space L2 on ‖c_plasma‖²+‖c_passive‖²) —
  the same ``M = blkdiag(BᵀB, VᵀV)`` penalty ``residual.InverseSolver`` uses to
  break the collinear cancellation.  Soft (small λ) so data overrides; it
  biases the inferred currents toward small, smooth, force-balanced fields.

Alignment (the crux — see plan / commit message)
-------------------------------------------------
The grounding terms are evaluated on the SAME ``ShotRun`` windows the engine
trains on, so ``z_t`` and ``raw_trust_t`` are the SAME timestep BY CONSTRUCTION.
The trustworthy ``amb`` rows and the 13 PF ``amc`` coil currents are SLICED
straight out of ``ShotRun.X`` (raw, un-normalised, grid-aligned) via a fixed
per-campaign column-index map.  CRITICAL: the engine feeds the model NORMALISED
inputs, so the sliced columns are de-normalised back to raw SI
(``x_raw = x_norm · feature_std + feature_mean``) before the raw-SI operator.

Only the ~56/70 trustworthy ``amb`` channels that are in the engine's amb
feature schema (``_AMB_CHANNELS``, the >90%-present subset) can be sliced; the
others are dropped from the per-campaign static row set (the low-DOF fit is
heavily overdetermined — 7 params vs ~56 rows — so dropping a handful of rows is
immaterial; quantified in the artifact).  Shots whose geometry table cannot be
built (``amm`` passive geometry absent, ~⅓ of shots) have NO operator and train
Dα-only (the ungrounded path is untouched for them) — the grounded-shot /
grounded-timestep fraction + campaign skew are reported, not gated on.

Scope: import-only from ``gs.operator`` / ``gs.residual`` / ``gs.geometry`` /
``statespace.baseline``.  CPU; the operator forward is a ~77×7 matmul (after the
basis reduction), negligible vs the encoder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from imas_ambix.gs.operator import build_operator
from imas_ambix.gs.residual import (
    passive_lowrank_basis,
    plasma_poly_basis,
    robust_sensor_scale,
    trustworthy_target,
)

if TYPE_CHECKING:
    from imas_ambix.gs.operator import ForwardOperator
    from imas_ambix.statespace.baseline import ChannelStats

logger = logging.getLogger(__name__)

# Feature-layout offsets in ShotRun.X (baseline._FEATURE_SCHEMA_MAG_ANE order:
# [ama (6), amb (73), amc (42), ane (1)] = 122).  amb starts after ama; amc
# starts after ama+amb.  Resolved at runtime from the schema so a schema change
# cannot silently desync the slice map.
_VAR_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Per-campaign grounding operator: precomputed torch tensors + X-slice maps
# ---------------------------------------------------------------------------


@dataclass
class CampaignGrounding:
    """Everything needed to evaluate L_data + L_GS for one campaign signature.

    All matrices are PRECOMPUTED at build time (pure geometry + the restricted
    basis); per-window evaluation is a tiny batched matmul.
    """

    signature_key: str
    # X-column indices (into the 122-d ShotRun.X feature vector) for the static
    # trustworthy amb rows kept for this campaign, and the 13 PF amc coils.
    amb_x_cols: np.ndarray  # (n_row,) int  — trustworthy amb rows present in X
    pf_amc_x_cols: np.ndarray  # (n_coil,) int — the 13 PF coil amc channels in X
    pf_amc_present: np.ndarray  # (n_coil,) bool — coil channel present in X
    # Per-X-column de-normalisation affine (raw = norm*std + mean), gathered for
    # the sliced columns only (so the slice + de-norm are one indexed op).
    amb_mean: torch.Tensor  # (n_row,)
    amb_std: torch.Tensor  # (n_row,)
    pf_mean: torch.Tensor  # (n_coil,)
    pf_std: torch.Tensor  # (n_coil,)
    # Whitened, basis-reduced forward blocks (the operator forward in head space):
    #   pred_white = W·G_pf·i_pf  +  A_plasma·θ  +  A_passive·ψ
    g_pf_white: torch.Tensor  # (n_row, n_coil)   W·G_pf
    a_plasma: torch.Tensor  # (n_row, n_dof)    W·G_plasma·B_poly
    a_passive: torch.Tensor  # (n_row, rank)     W·G_passive·V_passive
    w_scale: torch.Tensor  # (n_row,)          the per-sensor whitening 1/scale
    # GS soft-prior penalty blocks (current-space L2): BᵀB on θ, VᵀV on ψ.
    penalty_plasma: torch.Tensor  # (n_dof, n_dof)   BᵀB
    penalty_passive: torch.Tensor  # (rank, rank)     VᵀV
    # bookkeeping
    n_dof: int
    rank: int
    n_row: int
    profile_order: int = 1  # order-1 default (real builder passes explicitly)
    # diagnostics: per-row fraction of imputed (absent) values across train data
    imputation_rate: float = 0.0
    # the raw numpy operator + target + scale, kept for the near-vacuum sanity
    operator: ForwardOperator | None = field(default=None, repr=False)
    sensor_scale_np: np.ndarray | None = field(default=None, repr=False)
    target_rows_np: np.ndarray | None = field(default=None, repr=False)


def _feature_offsets(feature_schema: dict[str, list[str]]) -> dict[str, int]:
    """Start offset (column index) of each feature group in ShotRun.X."""
    off: dict[str, int] = {}
    cur = 0
    for group, channels in feature_schema.items():
        off[group] = cur
        cur += len(channels)
    return off


def build_campaign_grounding(
    signature_key: str,
    operator: ForwardOperator,
    stats: ChannelStats,
    feature_schema: dict[str, list[str]],
    *,
    profile_order: int,
    passive_rank: int,
    lam: float,
    quiescent_raw_trust: np.ndarray | None = None,
    sensor_scale: np.ndarray | None = None,
) -> CampaignGrounding | None:
    """Build the precomputed grounding tensors for one campaign signature.

    Parameters
    ----------
    operator : the ForwardOperator for this campaign signature.
    stats : the engine's ChannelStats (per-X-column mean/std for de-norm).
    feature_schema : baseline._FEATURE_SCHEMA_MAG_ANE (column layout).
    profile_order, passive_rank, lam : the restricted-basis config (the
        head emits exactly these DOF so it inherits the frontier's near-vacuum
        soundness; lam scales the L_GS soft prior, NOT the design).
    quiescent_raw_trust : (n_q, n_row) optional raw amb at the (kept) trustworthy
        rows on quiescent slices, to fit the per-sensor whitening scale.  If
        None and ``sensor_scale`` is None, a unit scale is used (whitening then
        only does dimensional row-norm via the operator columns).
    sensor_scale : (n_row,) optional precomputed per-sensor scale (overrides the
        quiescent fit; used to share one scale across all of a campaign's shots).

    Returns None if the campaign has fewer than a usable number of sliceable
    trustworthy rows.
    """
    from imas_ambix.statespace.baseline import _AMB_CHANNELS  # noqa: PLC0415

    offsets = _feature_offsets(feature_schema)
    amb_off = offsets.get("amb")
    amc_off = offsets.get("amc")
    if amb_off is None or amc_off is None:
        return None
    amb_feat = feature_schema["amb"]
    amc_feat = feature_schema["amc"]
    amb_feat_idx = {c: amb_off + i for i, c in enumerate(amb_feat)}
    amc_feat_idx = {c: amc_off + i for i, c in enumerate(amc_feat)}
    amb_feat_set = set(_AMB_CHANNELS)

    # Full trustworthy target for this campaign (76 B-probes + 1 clean flux loop).
    target = trustworthy_target(operator)
    # Keep only the trustworthy rows whose channel is in the amb feature schema
    # (sliceable from ShotRun.X); record the operator-row index for each.
    keep_op_rows: list[int] = []
    keep_x_cols: list[int] = []
    for j, ch in enumerate(target.channels):
        if ch in amb_feat_set and ch in amb_feat_idx:
            keep_op_rows.append(int(target.rows[j]))
            keep_x_cols.append(amb_feat_idx[ch])
    n_row = len(keep_op_rows)
    if n_row < 20:  # too few sliceable rows → no usable grounding for this campaign
        logger.warning(
            "[grounding] campaign %s: only %d sliceable trustworthy rows — skipped",
            signature_key,
            n_row,
        )
        return None
    keep_op_rows_arr = np.array(keep_op_rows, dtype=int)
    keep_x_cols_arr = np.array(keep_x_cols, dtype=int)

    # PF coil amc columns in X (all 13 verified present in the amc feature set).
    pf_chans = operator.pf_amc_channels
    pf_x_cols = np.array([amc_feat_idx.get(c, -1) for c in pf_chans], dtype=int)
    pf_present = pf_x_cols >= 0
    pf_x_cols_safe = np.where(pf_present, pf_x_cols, 0)

    # De-normalisation affine for the sliced columns.
    fmean = np.asarray(stats.feature_mean, dtype=np.float64)
    fstd = np.asarray(stats.feature_std, dtype=np.float64)
    amb_mean = fmean[keep_x_cols_arr]
    amb_std = fstd[keep_x_cols_arr]
    pf_mean = fmean[pf_x_cols_safe]
    pf_std = fstd[pf_x_cols_safe]

    # --- the restricted basis (SAME as residual.InverseSolver) --------------
    b_poly = plasma_poly_basis(
        operator.plasma_rz, profile_order, operator.r0, operator.minor_radius
    )  # (n_plasma_node, n_dof)
    g_pl = operator.g_plasma[keep_op_rows_arr]  # (n_row, n_plasma_node)
    g_pa = operator.g_passive[keep_op_rows_arr]  # (n_row, n_passive_node)
    g_pf = operator.g_pf[keep_op_rows_arr]  # (n_row, n_coil)

    # per-sensor robust whitening scale W (fit on quiescent raw if available)
    if sensor_scale is not None:
        scale = np.asarray(sensor_scale, dtype=np.float64)
    elif quiescent_raw_trust is not None and quiescent_raw_trust.shape[0] > 0:
        scale = robust_sensor_scale(quiescent_raw_trust)
    else:
        scale = np.ones(n_row, dtype=np.float64)
    w = 1.0 / scale  # (n_row,)

    # low-rank passive basis fit on the WHITENED passive design (same as monitor)
    v_passive = passive_lowrank_basis(w[:, None] * g_pa, passive_rank)  # (n_pa, rank)

    # whitened, basis-reduced forward blocks
    a_plasma = (w[:, None] * g_pl) @ b_poly  # (n_row, n_dof)
    a_passive = (
        (w[:, None] * g_pa) @ v_passive if v_passive.size else np.zeros((n_row, 0))
    )
    g_pf_white = w[:, None] * g_pf  # (n_row, n_coil)

    # GS soft-prior penalty blocks (current-space L2): BᵀB, VᵀV.
    btb = b_poly.T @ b_poly  # (n_dof, n_dof)
    vtv = (
        v_passive.T @ v_passive if v_passive.size else np.zeros((0, 0))
    )  # (rank, rank)

    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).float()

    return CampaignGrounding(
        signature_key=signature_key,
        amb_x_cols=keep_x_cols_arr,
        pf_amc_x_cols=pf_x_cols_safe,
        pf_amc_present=pf_present,
        amb_mean=_t(amb_mean),
        amb_std=_t(amb_std),
        pf_mean=_t(pf_mean),
        pf_std=_t(pf_std),
        g_pf_white=_t(g_pf_white),
        a_plasma=_t(a_plasma),
        a_passive=_t(a_passive),
        w_scale=_t(w),
        penalty_plasma=_t(btb),
        penalty_passive=_t(vtv),
        n_dof=int(b_poly.shape[1]),
        rank=int(v_passive.shape[1]) if v_passive.size else 0,
        n_row=n_row,
        profile_order=int(profile_order),
        operator=operator,
        sensor_scale_np=scale,
        target_rows_np=keep_op_rows_arr,
    )


# ---------------------------------------------------------------------------
# The grounding head (torch module)
# ---------------------------------------------------------------------------


class GroundingHead(nn.Module):
    """Latent z → restricted GS current amplitudes (θ_plasma, ψ_passive).

    Two small linear heads off the latent mean.  The amplitudes are emitted in
    the restricted basis (order-1 plasma poly DOF + rank-4 passive SVD),
    so the inferred current field is structurally low-DOF and smooth — the same
    instrument resolution the standalone frontier found, now PREDICTED FROM z
    rather than solved per-slice.  A single learned emission log-scale sets the
    L_data Gaussian noise floor (in whitened units).
    """

    def __init__(self, latent_dim: int, n_dof: int, rank: int) -> None:
        super().__init__()
        self.head_plasma = nn.Linear(latent_dim, n_dof)
        self.head_passive = nn.Linear(latent_dim, rank) if rank > 0 else None
        # learned emission noise (whitened units); init ~0.3 so it does not
        # dominate the early fit.
        self.log_emit = nn.Parameter(torch.tensor(math.log(0.3)))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (B, L) → (θ_plasma (B, n_dof), ψ_passive (B, rank))."""
        theta = self.head_plasma(z)
        if self.head_passive is not None:
            psi = self.head_passive(z)
        else:
            psi = z.new_zeros((z.shape[0], 0))
        return theta, psi


# ---------------------------------------------------------------------------
# Loss evaluation (called from engine.py's training loop + eval)
# ---------------------------------------------------------------------------


def grounding_losses(
    head: GroundingHead,
    z_flat: torch.Tensor,
    x_norm_flat: torch.Tensor,
    cg: CampaignGrounding,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute (L_data, L_GS) for a flat batch of (z_t, x_norm_t) at one campaign.

    Parameters
    ----------
    head : the GroundingHead.
    z_flat : (N, L) filtered posterior latent means (the grounding anchor).
    x_norm_flat : (N, F) the NORMALISED engine inputs at the SAME timesteps.
    cg : the campaign's precomputed grounding tensors.
    lam : the L_GS soft-prior weight.

    Returns
    -------
    L_data : scalar Gaussian NLL of the whitened raw-magnetics reconstruction.
    L_GS   : scalar GS force-balance soft prior (current-space L2).
    info   : dict of detached diagnostics (rmse_white, mean |c_plasma| etc.).
    """
    dev = z_flat.device
    # --- slice + DE-NORMALISE the raw amb + PF amc columns from x_norm -------
    amb_norm = x_norm_flat[:, cg.amb_x_cols]  # (N, n_row)
    amb_raw = amb_norm * cg.amb_std.to(dev) + cg.amb_mean.to(dev)
    pf_norm = x_norm_flat[:, cg.pf_amc_x_cols]  # (N, n_coil)
    pf_raw = pf_norm * cg.pf_std.to(dev) + cg.pf_mean.to(dev)
    # raw amc is kA·turn; the operator's flat ×1000 → A (turns=1).  Absent coil
    # channels contribute zero.
    pf_amps = pf_raw * 1.0e3
    pf_amps = pf_amps * torch.from_numpy(cg.pf_amc_present.astype(np.float32)).to(dev)

    # --- forward map in whitened head space ---------------------------------
    theta, psi = head(z_flat)  # (N, n_dof), (N, rank)
    w_scale = cg.w_scale.to(dev)
    # whitened KNOWN PF term: W·G_pf·i_pf  (g_pf_white already = W·G_pf)
    pred_pf = pf_amps @ cg.g_pf_white.to(dev).T  # (N, n_row)
    pred_pl = theta @ cg.a_plasma.to(dev).T  # (N, n_row)
    if cg.rank > 0:
        pred_pa = psi @ cg.a_passive.to(dev).T
    else:
        pred_pa = pred_pl.new_zeros(pred_pl.shape)
    pred_white = pred_pf + pred_pl + pred_pa  # (N, n_row)
    raw_white = amb_raw * w_scale.unsqueeze(0)  # (N, n_row)
    resid = pred_white - raw_white  # (N, n_row)

    # --- L_data: Gaussian NLL with a single learned emission scale ----------
    emit_var = (head.log_emit.exp() ** 2).clamp_min(_VAR_FLOOR)
    nll = 0.5 * (torch.log(2.0 * math.pi * emit_var) + resid * resid / emit_var)
    l_data = nll.mean()

    # --- L_GS: physical-amplitude Tikhonov (current-space L2) ---------------
    # θᵀ(BᵀB)θ  +  ψᵀ(VᵀV)ψ  averaged over the batch.
    btb = cg.penalty_plasma.to(dev)
    gs_pl = torch.einsum("nd,de,ne->n", theta, btb, theta)  # (N,)
    if cg.rank > 0:
        vtv = cg.penalty_passive.to(dev)
        gs_pa = torch.einsum("nr,rs,ns->n", psi, vtv, psi)
    else:
        gs_pa = gs_pl.new_zeros(gs_pl.shape)
    l_gs = lam * (gs_pl + gs_pa).mean()

    with torch.no_grad():
        info = {
            "n": int(z_flat.shape[0]),
            "rmse_white": float(torch.sqrt((resid * resid).mean()).item()),
            "mean_abs_theta": float(theta.abs().mean().item()),
            "mean_abs_psi": float(psi.abs().mean().item()) if cg.rank else 0.0,
            "emit_scale": float(head.log_emit.exp().item()),
            "l_data": float(l_data.item()),
            "l_gs_raw": float((gs_pl + gs_pa).mean().item()),
        }
    return l_data, l_gs, info


@dataclass
class GroundingContext:
    """All grounding state the engine training loop needs.

    Bundles the per-campaign precomputed tensors + a per-training-window
    campaign-signature assignment (None where the window's shot has no operator
    → that window contributes Dα loss only).
    """

    by_signature: dict[str, CampaignGrounding]
    window_signature: list[str | None]  # one per training window (or None)
    gs_lambda: float
    gs_data_weight: float
    # diagnostics for the artifact
    n_grounded_windows: int = 0
    n_total_windows: int = 0
    grounded_timestep_fraction: float = 0.0
    campaign_window_counts: dict[str, int] = field(default_factory=dict)

    def for_signature(self, sig: str | None) -> CampaignGrounding | None:
        return self.by_signature.get(sig) if sig else None


# ---------------------------------------------------------------------------
# Build the per-campaign operators for a set of shots (loader)
# ---------------------------------------------------------------------------


def build_operators_for_shots(
    shot_ids: list[int],
    *,
    resolve_identity: bool = False,
) -> tuple[dict[str, ForwardOperator], dict[int, str]]:
    """Build the per-campaign ForwardOperators covering ``shot_ids``.

    Returns ``(operators_by_signature, campaign_of)`` where ``campaign_of`` maps
    each shot whose geometry table built successfully to its signature key.
    Shots whose declared description cannot be emitted are simply absent from
    ``campaign_of`` and train Dα-only; the ungrounded path is untouched for
    them.

    The grouping stays keyed by signature: each operator's matrices are built on
    one discretization, so a window must find the operator built from ITS
    geometry, not merely from the same machine.  ``resolve_identity`` stamps the
    machine's physical digest onto each operator as provenance, which is what
    makes a corpus spanning two real configurations detectable rather than
    silent.
    """
    from imas_ambix.data.description_reader import (  # noqa: PLC0415
        read_geometry_table,
    )

    operators: dict[str, ForwardOperator] = {}
    campaign_of: dict[int, str] = {}
    n_ok = n_fail = 0
    for s in shot_ids:
        try:
            table = read_geometry_table(int(s))
        except Exception:  # noqa: BLE001 — unavailable description → Dα-only
            n_fail += 1
            continue
        key = table.signature.key
        if key not in operators:
            operators[key] = build_operator(table, resolve_identity=resolve_identity)
        campaign_of[int(s)] = key
        n_ok += 1
    logger.info(
        "[grounding] built %d campaign operators; %d/%d shots have an operator "
        "(%d unavailable descriptions → Dα-only)",
        len(operators),
        n_ok,
        len(shot_ids),
        n_fail,
    )
    return operators, campaign_of


def build_grounding_context(
    train_runs: list,
    stats: ChannelStats,
    feature_schema: dict[str, list[str]],
    *,
    profile_order: int,
    passive_rank: int,
    lam: float,
    gs_data_weight: float,
    seq_len: int,
    seed: int,
    max_windows_per_shot: int = 8,
) -> GroundingContext:
    """Build the full GroundingContext for the engine training loop.

    Steps:
      1. Build a campaign operator per train shot (those whose geometry table
         builds — ~⅔; the rest train Dα-only).
      2. Per campaign, pool de-normalised RAW amb (at the sliceable trustworthy
         rows) over QUIESCENT slices to fit ONE per-sensor whitening scale, then
         build the CampaignGrounding tensors.
      3. Build the per-training-window signature list IN THE SAME ORDER as
         ``engine._build_training_windows(..., return_run_index=True)`` (same
         seq_len/seed/max_windows_per_shot) so the engine can map each window to
         its campaign by position.

    ``train_runs`` are ``engine.ShotRun`` objects (carry ``shot_id`` + raw ``X``).
    """
    from imas_ambix.statespace.baseline import (  # noqa: PLC0415
        _AMB_CHANNELS,
        compute_transient_mask,
    )

    shot_ids = [int(r.shot_id) for r in train_runs]
    operators, campaign_of = build_operators_for_shots(sorted(set(shot_ids)))

    offsets = _feature_offsets(feature_schema)
    amb_off = offsets["amb"]
    amb_feat = feature_schema["amb"]
    amb_feat_idx = {c: amb_off + i for i, c in enumerate(amb_feat)}
    amb_feat_set = set(_AMB_CHANNELS)

    # --- per-campaign: kept trustworthy rows + a pooled quiescent raw scale ---
    # Determine the kept (sliceable) rows once per campaign from its operator.
    kept_cols: dict[str, np.ndarray] = {}
    quiescent_raw: dict[str, list[np.ndarray]] = {}
    for sig, op in operators.items():
        target = trustworthy_target(op)
        cols = [
            amb_feat_idx[c]
            for c in target.channels
            if c in amb_feat_set and c in amb_feat_idx
        ]
        kept_cols[sig] = np.array(cols, dtype=int)
        quiescent_raw[sig] = []

    # pool quiescent RAW amb at the kept rows across each campaign's runs
    for r in train_runs:
        sig = campaign_of.get(int(r.shot_id))
        if sig is None or kept_cols[sig].size == 0:
            continue
        x = np.asarray(r.X, dtype=np.float64)  # raw, un-normalised
        cols = kept_cols[sig]
        amb_raw = x[:, cols]  # (T, n_row) raw amb at kept rows
        elm = compute_transient_mask(r.y)
        q = ~elm
        if q.any():
            quiescent_raw[sig].append(amb_raw[q])

    by_sig: dict[str, CampaignGrounding] = {}
    for sig, op in operators.items():
        if kept_cols[sig].size < 20:
            continue
        qr = np.concatenate(quiescent_raw[sig], axis=0) if quiescent_raw[sig] else None
        scale = robust_sensor_scale(qr) if qr is not None and qr.shape[0] else None
        # imputation diagnostic: fraction of kept-row values that equal the
        # de-norm mean (i.e. were absent/imputed) — a coarse proxy, reported.
        cg = build_campaign_grounding(
            sig,
            op,
            stats,
            feature_schema,
            profile_order=profile_order,
            passive_rank=passive_rank,
            lam=lam,
            sensor_scale=scale,
        )
        if cg is not None:
            by_sig[sig] = cg

    # --- per-window signature list (same order as the engine window builder) --
    rng = np.random.default_rng(seed)
    window_signature: list[str | None] = []
    campaign_counts: dict[str, int] = {}
    n_grounded_windows = 0
    grounded_steps = 0
    total_steps = 0
    for r in train_runs:
        x = np.asarray(r.X)
        n_steps = x.shape[0]
        if seq_len > n_steps:
            continue
        starts = list(range(0, n_steps - seq_len + 1, seq_len))
        if len(starts) > max_windows_per_shot:
            selidx = rng.choice(len(starts), size=max_windows_per_shot, replace=False)
            starts = [starts[i] for i in sorted(selidx)]
        sig = campaign_of.get(int(r.shot_id))
        grounded = sig is not None and sig in by_sig
        for _ in starts:
            window_signature.append(sig if grounded else None)
            total_steps += seq_len
            if grounded:
                n_grounded_windows += 1
                grounded_steps += seq_len
                campaign_counts[sig] = campaign_counts.get(sig, 0) + 1

    ctx = GroundingContext(
        by_signature=by_sig,
        window_signature=window_signature,
        gs_lambda=lam,
        gs_data_weight=gs_data_weight,
        n_grounded_windows=n_grounded_windows,
        n_total_windows=len(window_signature),
        grounded_timestep_fraction=(
            grounded_steps / total_steps if total_steps else 0.0
        ),
        campaign_window_counts=campaign_counts,
    )
    logger.info(
        "[grounding] context: %d/%d windows grounded (%.1f%% of timesteps); "
        "campaigns=%s; %d operators with sliceable rows",
        n_grounded_windows,
        len(window_signature),
        100.0 * ctx.grounded_timestep_fraction,
        {k: v for k, v in sorted(campaign_counts.items())},
        len(by_sig),
    )
    return ctx
