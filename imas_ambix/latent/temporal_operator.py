"""Temporal equilibrium operator: causal sequence model over sensor history.

The temporal rung of the learned-equilibrium ladder.  A causal transformer
trunk attends over per-slice sensor-token codes (the same geometry-encoded
featurizer as :mod:`imas_ambix.latent.residual_operator`) and two heads emit
only physics-degenerate DOF:

* profile-DOF corrections ``dc`` about the classical solution, decoded through
  the exact Green's layer (:class:`ProfileGreensDecoder`) — identical contract
  to the static operator;
* vessel-eddy mode amplitudes ``da`` that enter the field as MORE EXTERNAL
  CURRENTS through the exact passive Green's columns — the boundary push-out
  readout is unchanged by construction.

The eddy latent is physics-structured, not free: the passive conductors'
L/R eigenmodes (inductance from the finite-area cylinder kernels, resistance
from the conductor cross-sections at a nominal steel resistivity) define a
diagonal state-space block whose per-mode decay constants are LEARNABLE but
initialised at the physical L/R times, and whose drive is the physically
computable flux swing the coil and plasma histories induce in each mode.  The
eddy state at time t is therefore an L/R convolution of the current history —
exactly the quantity the per-slice static fit provably cannot see.

Zero-initialised output heads: the untrained operator reproduces the classical
spine byte-exactly (``dc = 0``, ``da = 0``), so training starts at parity and
every gain is attributable to the learned temporal structure.

Conventions: total poloidal flux Φ = 2πR·A_φ [Wb]; thick-cylinder
finite-area Green's kernels throughout (never point-filament).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from imas_ambix.latent.residual_operator import TOKEN_FEATURES

if TYPE_CHECKING:
    from pathlib import Path

    from imas_ambix.gs.geometry import GeometryTable
    from imas_ambix.latent.gs_solve import EquilibriumGrid

logger = logging.getLogger(__name__)

#: nominal stainless-steel resistivity [Ω·m] for the vessel L/R times — the
#: decay constants are learnable, so this sets the INITIALISATION scale only
STEEL_RESISTIVITY = 7.2e-7


# ---------------------------------------------------------------------------
# Passive L/R eigenbasis — pure geometry (+ nominal resistivity), per campaign
# ---------------------------------------------------------------------------
@dataclass
class PassiveEigenbasis:
    """L/R eigenmodes of the passive conductors, reduced to the k modes kept.

    ``tau`` are the physical decay times [s]; ``v`` (n_passive, k) is the
    L-orthonormal eigenvector block; ``a_sens`` (S, k) / ``g_grid`` (nz·nr, k)
    map a mode amplitude to sensors / grid flux; ``m_coil`` (k, C) and
    ``m_cells`` (k, n_cells) are the flux linkages each mode picks up per
    ampere of coil channel / plasma cell current — the eddy DRIVE couplings.
    Mode amplitudes are in the L-orthonormal eigencoordinates throughout.
    """

    tau: np.ndarray
    v: np.ndarray
    a_sens: np.ndarray
    g_grid: np.ndarray
    m_coil: np.ndarray
    m_cells: np.ndarray
    resistivity: float

    @property
    def n_modes(self) -> int:
        return int(self.tau.size)


def _passive_circuit_filaments(table: GeometryTable) -> list[list]:
    """Filament groups of every ``inferred_passive`` circuit (sorted order)."""
    from imas_ambix.gs import operator as op  # noqa: PLC0415

    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    passive = sorted(c.circuit for c in classes if c.role == "inferred_passive")
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    return [by_circ[c] for c in passive]


def _flux_at_filaments(obs_filaments: list, src_filaments: list, greens) -> float:
    """Mutual flux linkage [Wb/A] between two filament groups (xmult-weighted)."""
    acc = 0.0
    for fo in obs_filaments:
        for fs in src_filaments:
            psi, _br, _bz = greens(
                np.array([fo.r]),
                np.array([fo.z]),
                float(fs.r),
                float(fs.z),
                max(abs(fs.width), 0.01),
                max(abs(fs.height), 0.01),
            )
            acc += fo.xmult * fs.xmult * float(psi[0])
    return acc


def build_passive_eigenbasis(
    table: GeometryTable,
    grid: EquilibriumGrid,
    *,
    sensor_scale: np.ndarray,
    k: int = 12,
    resistivity: float = STEEL_RESISTIVITY,
) -> PassiveEigenbasis:
    """L/R eigenmode reduction of the passive set — pure geometry, per campaign.

    Inductance: mutual flux linkage between passive circuits through the
    finite-area cylinder kernel (centroid-linked; the tiny negative spectral
    tail of that approximation is clamped — L is SPD physically).  Resistance:
    toroidal-ring resistance ``2πR·ρ/(w·h)`` per filament at the nominal steel
    resistivity (initialisation scale only — decays are learnable downstream).
    Generalised eigenproblem ``R v = (1/τ) L v`` with L-orthonormal ``v``.

    Mode selection keeps the ``k`` modes with the largest history-relevance
    ``τ_m · ||a_sens_m / scale||`` — a slow mode the sensors can see is
    exactly a mode whose history the static fit cannot absorb.  Drive
    couplings by reciprocity: ``m_cells = g_grid[cells].T`` (flux a mode links
    per ampere of plasma cell current == flux the cell sees per mode ampere).
    """
    from scipy.linalg import eigh  # noqa: PLC0415

    from imas_ambix.gs import operator as op  # noqa: PLC0415
    from imas_ambix.gs.cylinder import hybrid_greens  # noqa: PLC0415
    from imas_ambix.latent.boundary_disc import (  # noqa: PLC0415
        passive_coupling_matrices,
    )

    groups = _passive_circuit_filaments(table)
    n_pass = len(groups)
    if n_pass == 0:
        raise ValueError("table has no inferred_passive circuits")

    lmat = np.zeros((n_pass, n_pass))
    for i, gi in enumerate(groups):
        for j in range(i, n_pass):
            m = _flux_at_filaments(gi, groups[j], hybrid_greens)
            lmat[i, j] = m
            lmat[j, i] = m
    # SPD clamp: the centroid-linked kernel leaves a tiny negative tail
    # (measured ~1e-3 of the leading eigenvalue); inductance is SPD physically
    w0, u0 = np.linalg.eigh(lmat)
    lmat = (u0 * np.clip(w0, 1e-4 * w0.max(), None)) @ u0.T

    r_diag = np.array(
        [
            sum(
                2.0
                * np.pi
                * f.r
                * resistivity
                / (max(abs(f.width), 0.01) * max(abs(f.height), 0.01))
                for f in g
            )
            / max(len(g), 1) ** 2
            for g in groups
        ]
    )
    w, v = eigh(np.diag(r_diag), lmat)  # R v = (1/τ) L v ; v L-orthonormal
    tau = 1.0 / np.clip(w, 1e-12, None)

    a_circ, g_circ = passive_coupling_matrices(grid, table)
    a_modes = a_circ @ v  # (S, n_pass)
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    relevance = tau * np.linalg.norm(a_modes / scale[:, np.newaxis], axis=0)
    keep = np.argsort(relevance)[::-1][: int(k)]
    keep = keep[np.argsort(tau[keep])[::-1]]  # slowest-first for readability

    v_k = v[:, keep]
    g_modes = g_circ @ v_k
    m_cells = g_modes[grid.cells, :].T  # reciprocity: (k, n_cells)

    # coil channel → mode flux linkage, mirroring build_operator's channel
    # merge (average same-channel circuits; solenoid response scale applied)
    classes = op.classify_circuits(table.pf_filaments, table.amc_current_channels)
    by_circ: dict[int, list] = {}
    for f in table.pf_filaments:
        by_circ.setdefault(f.circuit, []).append(f)
    pf_by_chan: dict[str, list[int]] = {}
    for cc in classes:
        if cc.role in op._KNOWN_ROLES:  # noqa: SLF001 — canonical role list
            pf_by_chan.setdefault(cc.amc_channel, []).append(cc.circuit)
    m_coil_circ = np.zeros((n_pass, len(pf_by_chan)))
    for c_idx, chan in enumerate(sorted(pf_by_chan)):
        cols = []
        for circ in sorted(pf_by_chan[chan]):
            cols.append(
                [_flux_at_filaments(g, by_circ[circ], hybrid_greens) for g in groups]
            )
        col = np.mean(np.asarray(cols), axis=0)
        if chan == "sol_current":
            col = col * op.SOLENOID_RESPONSE_SCALE
        m_coil_circ[:, c_idx] = col
    m_coil = v_k.T @ m_coil_circ  # (k, C)

    return PassiveEigenbasis(
        tau=tau[keep],
        v=v_k,
        a_sens=a_modes[:, keep],
        g_grid=g_modes,
        m_coil=m_coil,
        m_cells=m_cells,
        resistivity=float(resistivity),
    )


def physical_eddy_history(
    basis: PassiveEigenbasis,
    times: np.ndarray,
    i_pf: np.ndarray,
    i_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-ZOH integration of the mode eddy ODE along a slice sequence.

    Mode dynamics (L-orthonormal coordinates): ``da/dt + a/τ = −dΨ/dt`` with
    ``Ψ_m(t) = m_coil_m · i_pf(t) + m_cells_m · i_cell(t)`` the external flux
    the mode links.  With Ψ piecewise-linear between labelled slices the update
    is exact::

        a_k = e^{−Δt/τ} a_{k−1} − (τ/Δt)(1 − e^{−Δt/τ}) ΔΨ

    Returns ``(a_phys (T, k), u_drive (T, k))`` — the physical eddy state and
    the per-step flux swing ``−ΔΨ`` (the drive feature).  ``a_phys[0] = 0``:
    the first labelled slice is taken as the eddy reference (label sequences
    start above the Ip threshold; earlier transients are unobserved here).
    """
    times = np.asarray(times, dtype=np.float64)
    psi_m = (
        np.asarray(i_pf, dtype=np.float64) @ basis.m_coil.T
        + np.asarray(i_cell, dtype=np.float64) @ basis.m_cells.T
    )  # (T, k)
    n_t, k = psi_m.shape
    a = np.zeros((n_t, k))
    u = np.zeros((n_t, k))
    for t in range(1, n_t):
        dt = max(float(times[t] - times[t - 1]), 1e-6)
        decay = np.exp(-dt / basis.tau)
        coeff = basis.tau / dt * (1.0 - decay)
        dpsi = psi_m[t] - psi_m[t - 1]
        u[t] = -dpsi
        a[t] = decay * a[t - 1] + coeff * u[t]
    return a, u


def save_eigenbasis(path: Path | str, basis: PassiveEigenbasis) -> None:
    """Persist a campaign eigenbasis (the build is minutes of kernel sums)."""
    np.savez_compressed(
        path,
        tau=basis.tau,
        v=basis.v,
        a_sens=basis.a_sens,
        g_grid=basis.g_grid,
        m_coil=basis.m_coil,
        m_cells=basis.m_cells,
        resistivity=np.float64(basis.resistivity),
    )


def load_eigenbasis(path: Path | str) -> PassiveEigenbasis:
    with np.load(path) as z:
        return PassiveEigenbasis(
            tau=z["tau"],
            v=z["v"],
            a_sens=z["a_sens"],
            g_grid=z["g_grid"],
            m_coil=z["m_coil"],
            m_cells=z["m_cells"],
            resistivity=float(z["resistivity"]),
        )


# ---------------------------------------------------------------------------
# The causal temporal operator
# ---------------------------------------------------------------------------
class TemporalOperator(nn.Module):
    """Causal transformer over slice codes + diagonal L/R SSM eddy block.

    Inputs are sequences over one shot's labelled slices: per-slice sensor
    tokens (masked-pooled by the same encoder as the static operator),
    firewall-safe globals, the step Δt, and the physically-integrated eddy
    features (state + drive, standardised).  Outputs per step:

    * ``dc`` (B, T, n_dof) — bounded profile-DOF corrections (as R1);
    * ``da`` (B, T, k) — eddy mode amplitudes in PHYSICAL mode units
      (standardised internally, rescaled by ``eddy_std`` on output).

    The eddy block is a diagonal SSM: per-mode learnable decay rates
    initialised at the physical L/R times; drive = physical flux swing plus a
    zero-initialised trunk projection.  All output heads are zero-initialised,
    so the untrained operator is the identity on the classical spine.
    """

    def __init__(
        self,
        n_dof: int,
        tau_init: np.ndarray,
        eddy_std: np.ndarray,
        drive_std: np.ndarray,
        *,
        token_dim: int = len(TOKEN_FEATURES),
        n_global: int = 2,
        width: int = 96,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dc_scale: float = 0.3,
        eddy_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.n_dof = int(n_dof)
        self.n_modes = int(np.asarray(tau_init).size)
        self.dc_scale = float(dc_scale)
        self.eddy_scale = float(eddy_scale)
        self.width = int(width)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.n_global = int(n_global)

        self.log_tau = nn.Parameter(
            torch.log(torch.as_tensor(tau_init, dtype=torch.float32))
        )
        self.register_buffer(
            "eddy_std",
            torch.clamp(torch.as_tensor(eddy_std, dtype=torch.float32), min=1e-30),
        )
        self.register_buffer(
            "drive_std",
            torch.clamp(torch.as_tensor(drive_std, dtype=torch.float32), min=1e-30),
        )

        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        step_dim = 2 * width + n_global + 1 + 2 * self.n_modes  # +1 = log Δt
        self.step_proj = nn.Linear(step_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.dc_head = nn.Linear(d_model, n_dof)
        nn.init.zeros_(self.dc_head.weight)
        nn.init.zeros_(self.dc_head.bias)
        # eddy drive projection (adds to the physical drive) and output heads —
        # gate + trunk projection both zero-initialised: da == 0 at init while
        # the SSM state still carries the physically-driven trajectory
        self.drive_proj = nn.Linear(d_model, self.n_modes)
        nn.init.zeros_(self.drive_proj.weight)
        nn.init.zeros_(self.drive_proj.bias)
        self.eddy_gate = nn.Parameter(torch.zeros(self.n_modes))
        self.eddy_head = nn.Linear(d_model, self.n_modes)
        nn.init.zeros_(self.eddy_head.weight)
        nn.init.zeros_(self.eddy_head.bias)

    # -- featurization ------------------------------------------------------
    def encode_tokens(
        self, tokens: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """(B, T, S, F) sensor tokens → (B, T, 2·width) masked mean+max codes."""
        h = self.token_mlp(tokens)
        w = token_mask.to(h.dtype).unsqueeze(-1)
        n_valid = w.sum(dim=2)
        mean_pool = (h * w).sum(dim=2) / n_valid.clamp(min=1.0)
        max_pool = torch.where(w > 0, h, h.new_full((), -1e30)).amax(dim=2)
        # a fully-masked timestep (padded tail of a mixed-length batch) must
        # emit zeros, not the -1e30 max sentinel — test on the TRUE valid
        # count, never on a clamped denominator that is always positive
        max_pool = torch.where(n_valid > 0, max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T, S, F)
        token_mask: torch.Tensor,  # (B, T, S) bool
        global_feats: torch.Tensor,  # (B, T, n_global)
        dt: torch.Tensor,  # (B, T) step Δt [s]; dt[:, 0] ignored
        a_phys: torch.Tensor,  # (B, T, k) physical eddy state (native units)
        u_drive: torch.Tensor,  # (B, T, k) physical flux swing (native units)
        pad_mask: torch.Tensor | None = None,  # (B, T) bool — True = PADDING
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = tokens.shape[:2]
        code = self.encode_tokens(tokens, token_mask)
        a_std = a_phys / self.eddy_std
        u_std = u_drive / self.drive_std
        log_dt = torch.log(torch.clamp(dt, min=1e-6)).unsqueeze(-1)
        feats = torch.cat([code, global_feats, log_dt, a_std, u_std], dim=-1)
        x = self.step_proj(feats)
        causal = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.trunk(x, mask=causal, src_key_padding_mask=pad_mask)

        dc = self.dc_scale * torch.tanh(self.dc_head(h))

        # diagonal SSM over standardised mode amplitudes, exact-ZOH stepping
        tau = torch.exp(self.log_tau)  # (k,)
        drive = u_std + self.drive_proj(h)  # (B, T, k)
        dt_c = torch.clamp(dt, min=1e-6).unsqueeze(-1)
        decay = torch.exp(-dt_c / tau)
        coeff = tau / dt_c * (1.0 - decay)
        states = []
        s = torch.zeros(b, self.n_modes, device=x.device, dtype=x.dtype)
        for step in range(t):
            if step > 0:
                s = decay[:, step] * s + coeff[:, step] * drive[:, step]
            states.append(s)
        s_seq = torch.stack(states, dim=1)  # (B, T, k)

        da_std = self.eddy_scale * torch.tanh(
            self.eddy_gate * s_seq + self.eddy_head(h)
        )
        da = da_std * self.eddy_std
        if pad_mask is not None:
            # select, never multiply: any non-finite value a padded position
            # picks up would survive a multiplicative mask (NaN * 0 == NaN)
            # and poison the batch loss
            keep = (~pad_mask).unsqueeze(-1)
            dc = torch.where(keep, dc, torch.zeros_like(dc))
            da = torch.where(keep, da, torch.zeros_like(da))
        return dc, da


def save_checkpoint(path: Path | str, model: TemporalOperator, extra: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_dof": model.n_dof,
            "n_modes": model.n_modes,
            "dc_scale": model.dc_scale,
            "eddy_scale": model.eddy_scale,
            "width": model.width,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
            "n_global": model.n_global,
            "token_features": list(TOKEN_FEATURES),
            **extra,
        },
        path,
    )


def load_checkpoint(path: Path | str) -> tuple[TemporalOperator, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalOperator(
        int(ckpt["n_dof"]),
        np.ones(int(ckpt["n_modes"])),  # placeholders — state_dict overwrites
        np.ones(int(ckpt["n_modes"])),
        np.ones(int(ckpt["n_modes"])),
        width=int(ckpt["width"]),
        d_model=int(ckpt["d_model"]),
        n_heads=int(ckpt["n_heads"]),
        n_layers=int(ckpt["n_layers"]),
        n_global=int(ckpt["n_global"]),
        dc_scale=float(ckpt["dc_scale"]),
        eddy_scale=float(ckpt["eddy_scale"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


__all__ = [
    "STEEL_RESISTIVITY",
    "PassiveEigenbasis",
    "TemporalOperator",
    "build_passive_eigenbasis",
    "load_checkpoint",
    "load_eigenbasis",
    "physical_eddy_history",
    "save_checkpoint",
    "save_eigenbasis",
]
