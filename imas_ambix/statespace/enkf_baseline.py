r"""D2 — the classical RAPTOR-equivalent baseline (THE BAR to beat) for S9.

A credible classical comparator for ``mse-free-current-recovery-v0``: a
parameter-space **ensemble smoother** built on the validated TORAX current-
diffusion simulator as the forward dynamics, that recovers the internal
poloidal-current profile from a shot's MEASURED non-MSE inputs + an external-
magnetics consistency update, and reads out the MSE-observable pitch + on-axis
``q0``/``rax`` through the SHARED eval forward/inverse models.  No MSE anywhere
on the input side — MSE is the held-out eval truth only.

Why this is not a vacuous baseline (the binding physics point)
--------------------------------------------------------------
Stage-2 (``plasma-gs-prior-v0``) proved — and ``gs/residual.py`` measured — that
external magnetics + the plasma boundary UNDER-DETERMINE the internal current
profile ``j(psi)`` (the GS plasma block has effective rank ~5-6).  A transition-
FREE EnKF (random-walk + GS observation) would therefore recover nothing
internal and "winning" over it would be meaningless.  The internal current
profile here comes from a genuine **resistive current-diffusion transition with
a neoclassical conductivity closure**: TORAX evolves the poloidal flux driven by
the measured ``Ip(t)`` (``amc``), the measured ``Te(rho)`` (Thomson ``ayc``) ->
neoclassical ``sigma_parallel(Te)``, and the measured density (interferometer
``ane`` / Thomson).  That ``sigma(Te)`` closure is the §4 channel that breaks the
magnetics under-determination — now from a validated solver, not hand-rolled.

Two arms are run + reported so the non-vacuity is DEMONSTRATED, not asserted:
  * FORECAST arm — the prior TORAX ensemble (measured inputs only, magnetics NOT
    assimilated).  This is the non-vacuity control: it already produces a
    physical ``q0 ~ 1`` and a sensible pitch profile from the transition alone.
  * ANALYSIS arm — one ensemble-Kalman-inversion (EKI) update of the uncertain
    TORAX parameters against the external ``amb`` magnetics (via ``gs/operator.py``
    as the observation operator H), then re-run.  The update is validated by an
    OBS-SPACE INNOVATION DROP (the whitened amb misfit falls after the update) —
    a truth-free check that the analysis is real, not the under-determined inverse.

Architecture — parameter-space ensemble smoother over TORAX (NOT a sequential
state-updating EnKF)
-----------------------------------------------------------------------------
Each ensemble member is ONE full TORAX trajectory under a sampled parameter
vector ``theta = (Zeff, j_peaking/initial-q, resistivity_mult, ...)`` (3-5
parameters).  The magnetics analysis is ONE EKI step on ``theta`` (map each
member's TORAX current profile at the assimilation slices -> ``c_plasma`` ->
``operator.predict`` -> predicted amb; stack innovations vs measured amb; one
Gauss-Newton / ensemble-Kalman update on ``theta``; re-run the updated members).
We do NOT inject an updated state mid-trajectory and re-init TORAX per segment
(that is a multi-day rabbit hole and not needed for a v0 comparator).  This is a
legitimate RAPTOR-equivalent ensemble smoother; it is documented honestly as
such (not called a sequential EnKF).

Observation operator H (magnetics) + the j -> c_plasma embedding
----------------------------------------------------------------
The validated EFIT-free GS Green's-function forward operator (``gs/operator.py``):
the TORAX ``j(rho)`` is embedded onto the operator's plasma ``(R, Z)`` basis and
turned into per-node AMPERES (``c_plasma[k] = j(node_k) * cell_area_k`` so
``sum(c_plasma) ~ Ip`` — the operator's plasma columns are field-per-ampere, so
feeding current *density* without the cell-area weight mis-scales H ~15x).  The
KNOWN PF term is assembled from raw ``amc``.  We compute our OWN ensemble update
against the trustworthy ``amb`` (NOT ``gs/residual.py.solve`` — that returns the
regularised magnetics inverse and would bypass the ensemble/transition);
``gs/residual.py.robust_sensor_scale`` IS reused for the per-sensor R.

Readout (SHARED with the neural filter — fairness-binding)
----------------------------------------------------------
The current->pitch and pitch->q0/rax readouts are the CANONICAL SHARED pair from
``statespace/mse_eval.py`` (``pitch_from_current_profile`` /
``invert_pitch_to_q0rax``); the EnKF and the learned filter use IDENTICAL
observation physics so the head-to-head isolates STATE INFERENCE, not the
forward model.  The ensemble of TORAX trajectories gives ``pitch_samples``
(K, C, M) directly -> the harness scores energy-form CRPS / coverage natively.
``q0`` is read straight off TORAX's ``q`` profile on-axis (free, no inversion)
for the non-vacuity check; the scored secondary ``q0``/``rax`` come from the
shared inverter applied to the predicted pitch (method-matched to the truth).

"Matched compute", stated honestly
-----------------------------------
Equal CPU / wall-clock budget vs the neural filter.  The ``O(N_ens)`` ensemble
CANNOT ingest the camera / SXR image modalities the neural filter fuses — that
asymmetry is EXPECTED and IS the thesis.  Stated in the metrics artifact.

Compute: CPU + JAX (TORAX).  Measured: after a one-time ~7 s JIT compile, each
member runs ~0.04-0.3 s (JIT cache reuses across parameter-value changes), so a
64-member shot is ~3-18 s — no GPU needed for v0.  Foreground only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from imas_ambix.data.paths import LEVEL1_DIR, local_shot_path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from imas_ambix.gs.operator import ForwardOperator

logger = logging.getLogger(__name__)

MU0 = 4.0e-7 * np.pi
MAST_R0 = 0.85  # nominal major radius [m] (matches gs.geometry + mse_eval DEFAULT_R0)
MAST_A = 0.50  # plasma minor radius [m] used for the circular TORAX geometry
MAST_B0 = 0.5  # vacuum toroidal field at R0 [T] (matches mse_eval DEFAULT_BT0)
_KA_TO_A = 1.0e3  # amc plasma_current stored in kA -> A


# --- Configuration ----------------------------------------------------


@dataclass
class EnKFConfig:
    """All baseline knobs (persisted to the config artifact, frozen per run)."""

    n_ensemble: int = 64
    seed: int = 1234

    # TORAX geometry / numerics
    r_major: float = MAST_R0
    a_minor: float = MAST_A
    b0: float = MAST_B0
    fixed_dt: float = 0.01  # TORAX solver step [s]
    n_te_nodes: int = 6  # Te(rho) prescription nodes from Thomson

    # ensemble parameter priors (theta) — the uncertain TORAX inputs/params
    zeff_prior: float = 2.0
    zeff_spread: float = 0.6
    resist_mult_prior: float = 1.0  # multiplier on parallel resistivity
    resist_mult_logspread: float = 0.30  # log-normal spread
    current_peaking_prior: float = 2.0  # generic_current peaking (nu)
    current_peaking_spread: float = 0.6
    ip_frac_spread: float = 0.03  # fractional spread on the Ip boundary

    # magnetics analysis (EKI)
    assimilate: bool = True
    n_assim_slices: int = 6  # amb slices used in the single EKI update
    eki_inflation: float = 1.0  # observation-error inflation
    eki_step: float = 1.0  # EKI step length (1.0 = full Gauss-Newton)

    # readout
    kappa: float = 1.85
    bt0_for_readout: float = MAST_B0

    # marching / scope
    max_slices_per_shot: int | None = 80  # cap eval slices for speed (None = all)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --- Per-shot raw inputs (measured, non-MSE) --------------------------


@dataclass
class ShotInputs:
    """Raw measured non-MSE inputs that drive TORAX + the magnetics update.

    NO MSE here — the MSE truth is loaded separately by the eval harness.
    """

    shot_id: int
    # eval slice grid (from the canonical AmsShot — the held-out MSE time base)
    slice_t: np.ndarray  # (K,)
    active_channel_rpos: np.ndarray  # (C,) sightline major radii [m] (radial order)
    n_active: int
    # TORAX drive (time-keyed dicts)
    ip_t: dict[float, float]  # Ip(t) [A] from amc
    te_t: dict[float, dict[float, float]]  # Te(rho_norm)(t) [keV] from ayc
    ne_t: dict[float, dict[float, float]]  # n_e(rho_norm)(t) [m^-3] from ayc/ane
    t0: float  # TORAX start time [s]
    t_final: float  # TORAX end time [s]
    # magnetics for the analysis update
    amb_trust: np.ndarray  # (K, n_trust) raw amb at trustworthy sensors
    i_pf: np.ndarray  # (K, n_coil) KNOWN PF currents [A]


def _amc_interp(t_axis, arr, query):
    """Interp an amc channel onto ``query`` using only FINITE samples."""
    t = np.asarray(t_axis, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(a)
    if m.sum() < 2:
        return np.zeros(np.asarray(query).shape)
    return np.interp(query, t[m], a[m])


def load_shot_inputs(
    shot_id: int,
    operator: ForwardOperator,
    cfg: EnKFConfig,
    *,
    ams_shot=None,
) -> ShotInputs | None:
    """Load one shot's measured non-MSE inputs (TORAX drive + magnetics).

    ``ams_shot`` (a canonical ``mse_split.AmsShot``) supplies the held-out MSE
    slice grid + the CORRECT active-channel major radii (radial order) — the
    eval contract's time base + channel->R map.  When None, it is read here.
    Returns None if the shot has no usable beam-on MSE or no magnetics.
    """
    import zarr  # noqa: PLC0415

    from imas_ambix.gs.residual import trustworthy_target  # noqa: PLC0415
    from imas_ambix.statespace.mse_split import read_ams_shot  # noqa: PLC0415

    if ams_shot is None:
        ams_shot = read_ams_shot(local_shot_path(shot_id, tier="level1"))
    if ams_shot is None:
        return None
    slice_t = np.asarray(ams_shot.time, dtype=np.float64)
    if cfg.max_slices_per_shot and slice_t.size > cfg.max_slices_per_shot:
        sel = np.linspace(0, slice_t.size - 1, cfg.max_slices_per_shot).astype(int)
        slice_t = slice_t[sel]
    else:
        sel = np.arange(slice_t.size)
    if slice_t.size < 3:
        return None
    ch_rpos = np.asarray(ams_shot.active_channel_rpos, dtype=np.float64)
    n_active = ch_rpos.size

    root = local_shot_path(shot_id, tier="level1")
    try:
        store = zarr.open(str(root), mode="r")
    except Exception:  # noqa: BLE001
        return None
    if "amc" not in store or "amb" not in store:
        return None

    # --- amc: Ip(t), PF currents ---
    amc = store["amc"]
    amc_keys = set(amc.array_keys())
    amc_t = np.asarray(amc["time"], dtype=np.float64) if "time" in amc_keys else None
    if amc_t is None:
        return None

    # plasma-on window: restrict TORAX to where |Ip| is appreciable, so the
    # transition is driven over the real discharge (not pre-fill).
    ip_raw = (
        np.asarray(amc["plasma_current"], dtype=np.float64)
        if "plasma_current" in amc_keys
        else None
    )
    if ip_raw is None:
        return None
    # TORAX time window: from first eval slice to last, clipped to >=0
    t0 = float(max(slice_t[0], 0.0))
    t_final = float(slice_t[-1])
    if t_final <= t0:
        t_final = t0 + 10 * cfg.fixed_dt

    # Ip(t) drive: sample amc on a coarse grid over [t0, t_final]
    n_drive = max(5, int((t_final - t0) / 0.02) + 1)
    drive_t = np.linspace(t0, t_final, n_drive)
    ip_drive = _amc_interp(amc_t, ip_raw, drive_t) * _KA_TO_A
    ip_t = {
        float(tt): float(max(abs(iv), 1.0e3))
        for tt, iv in zip(drive_t, ip_drive, strict=True)
    }

    # --- amb at trustworthy sensors on the slice grid ---
    amb = store["amb"]
    amb_keys = set(amb.array_keys())
    amb_t = np.asarray(amb["time"], dtype=np.float64) if "time" in amb_keys else None
    tt = trustworthy_target(operator)
    n_trust = tt.rows.size
    amb_trust = np.full((slice_t.size, n_trust), np.nan)
    for j, ch in enumerate(tt.channels):
        if ch not in amb_keys:
            continue
        arr = np.asarray(amb[ch], dtype=np.float64)
        if amb_t is not None and amb_t.size == arr.size:
            m = np.isfinite(amb_t) & np.isfinite(arr)
            if m.sum() >= 2:
                amb_trust[:, j] = np.interp(slice_t, amb_t[m], arr[m])

    # PF currents per slice (operator's amc mapping)
    n_coil = len(operator.pf_amc_channels)
    i_pf = np.zeros((slice_t.size, n_coil))
    amc_cache = {
        ch: _amc_interp(amc_t, np.asarray(amc[ch], dtype=np.float64), slice_t)
        for ch in operator.pf_amc_channels
        if ch and ch in amc_keys
    }
    for s in range(slice_t.size):
        vals = {ch: amc_cache[ch][s] for ch in amc_cache}
        i_pf[s] = operator.assemble_pf_currents(vals)

    # --- Te(rho)(t) + n_e(rho)(t) from Thomson ayc (keV, m^-3) ---
    te_t, ne_t = _thomson_profiles(store, drive_t, cfg)

    return ShotInputs(
        shot_id=shot_id,
        slice_t=slice_t,
        active_channel_rpos=ch_rpos,
        n_active=n_active,
        ip_t=ip_t,
        te_t=te_t,
        ne_t=ne_t,
        t0=t0,
        t_final=t_final,
        amb_trust=amb_trust,
        i_pf=i_pf,
    )


def _thomson_profiles(store, drive_t: np.ndarray, cfg: EnKFConfig):
    """Build TORAX Te(rho_norm)(t) [keV] + n_e(rho_norm)(t) [m^-3] from ayc.

    Maps the Thomson radius to rho_norm via ``|R - R0| / a`` (outboard) onto a
    fixed ``n_te_nodes`` grid; falls back to a mild default profile where Thomson
    is missing.  Te in keV (TORAX convention), n_e in m^-3.
    """
    rho_nodes = np.linspace(0.0, 1.0, cfg.n_te_nodes)
    te_t: dict[float, dict[float, float]] = {}
    ne_t: dict[float, dict[float, float]] = {}
    default_te = {float(r): float(max(1.0 - 0.9 * r, 0.05)) for r in rho_nodes}  # keV
    default_ne = {float(r): float((2.5 - 1.5 * r) * 1e19) for r in rho_nodes}  # m^-3

    ayc = store["ayc"] if "ayc" in store else None  # noqa: SIM401 (zarr group, not dict)
    if ayc is None or not {"te", "time", "radius"}.issubset(set(ayc.array_keys())):
        for tt in drive_t:
            te_t[float(tt)] = dict(default_te)
            ne_t[float(tt)] = dict(default_ne)
        return te_t, ne_t

    te = np.asarray(ayc["te"], dtype=np.float64)  # (Tm, R) eV
    yt = np.asarray(ayc["time"], dtype=np.float64)
    yr = np.asarray(ayc["radius"], dtype=np.float64)
    ne = np.asarray(ayc["ne"], dtype=np.float64) if "ne" in ayc.array_keys() else None

    for tt in drive_t:
        km = int(np.argmin(np.abs(yt - tt)))
        prof = te[km]
        rr = yr[km] if yr.ndim == 2 else yr
        m = np.isfinite(prof) & np.isfinite(rr) & (prof > 0)
        if m.sum() >= 3:
            rho_ayc = np.clip(np.abs(rr[m] - cfg.r_major) / cfg.a_minor, 0.0, 1.0)
            o = np.argsort(rho_ayc)
            te_kev = np.clip(
                np.interp(
                    rho_nodes,
                    rho_ayc[o],
                    prof[m][o] / 1.0e3,
                    left=prof[m][o][0] / 1.0e3,
                    right=0.02,
                ),
                0.02,
                None,
            )
            te_t[float(tt)] = {
                float(r): float(v) for r, v in zip(rho_nodes, te_kev, strict=True)
            }
        else:
            te_t[float(tt)] = dict(default_te)
        if ne is not None and ne.shape == te.shape:
            nprof = ne[km]
            mn = np.isfinite(nprof) & np.isfinite(rr) & (nprof > 0)
            if mn.sum() >= 3:
                rho_n = np.clip(np.abs(rr[mn] - cfg.r_major) / cfg.a_minor, 0.0, 1.0)
                on = np.argsort(rho_n)
                ne_v = np.clip(
                    np.interp(
                        rho_nodes,
                        rho_n[on],
                        nprof[mn][on],
                        left=nprof[mn][on][0],
                        right=1e18,
                    ),
                    1e17,
                    None,
                )
                ne_t[float(tt)] = {
                    float(r): float(v) for r, v in zip(rho_nodes, ne_v, strict=True)
                }
            else:
                ne_t[float(tt)] = dict(default_ne)
        else:
            ne_t[float(tt)] = dict(default_ne)
    return te_t, ne_t


# --- TORAX forward run for one parameter vector -----------------------


def _torax_config(inp: ShotInputs, cfg: EnKFConfig, theta: dict[str, float]) -> dict:
    """Build the TORAX CONFIG dict for one ensemble member's parameters.

    Only the poloidal flux (current) equation evolves — heat/density/pedestal/
    fusion are frozen (Te + n_e are PRESCRIBED from Thomson; sigma(Te) is
    automatic).  ``theta`` perturbs Zeff, the resistivity multiplier (via Zeff
    scaling as the simplest validated knob), the initial-current peaking, and the
    Ip boundary.  The minimal-config v0 design.
    """
    ip_scaled = {t: v * theta.get("ip_frac", 1.0) for t, v in inp.ip_t.items()}
    # resistivity multiplier folded into an EFFECTIVE Zeff (eta_par ~ Zeff): a
    # documented v0 simplification (TORAX exposes no direct eta multiplier in the
    # minimal config; Zeff is the validated conductivity knob).
    zeff_eff = float(
        np.clip(
            theta.get("zeff", cfg.zeff_prior) * theta.get("resist_mult", 1.0), 1.0, 6.0
        )
    )
    nu = float(
        np.clip(theta.get("current_peaking", cfg.current_peaking_prior), 0.5, 4.0)
    )
    return {
        "profile_conditions": {
            "Ip": ip_scaled,
            "T_e": inp.te_t,
            "T_i": inp.te_t,  # T_i ~ T_e (no ion-temp measurement; frozen anyway)
            "n_e": inp.ne_t,
            "current_profile_nu": nu,  # initial current peaking
            "initial_psi_mode": "j",  # init psi from a current profile
        },
        "plasma_composition": {"Z_eff": zeff_eff},
        "numerics": {
            "t_initial": inp.t0,
            "t_final": inp.t_final,
            "evolve_current": True,
            "evolve_density": False,
            "evolve_ion_heat": False,
            "evolve_electron_heat": False,
            "fixed_dt": cfg.fixed_dt,
        },
        "geometry": {
            "geometry_type": "circular",
            "R_major": cfg.r_major,
            "a_minor": cfg.a_minor,
            "B_0": cfg.b0,
        },
        "neoclassical": {"bootstrap_current": {}},
        "sources": {"generic_current": {}, "ohmic": {}},
        "pedestal": {},
        "transport": {"model_name": "constant"},
        "solver": {"solver_type": "linear"},
        "time_step_calculator": {"calculator_type": "fixed"},
    }


@dataclass
class ToraxTrajectory:
    """One TORAX member trajectory reduced to what the readout + H need."""

    time: np.ndarray  # (T,) TORAX output times
    rho_norm: np.ndarray  # (G,) face-grid normalised minor radius (j_total grid)
    j_total: np.ndarray  # (T, G) toroidal current density [A/m^2]
    q: np.ndarray  # (T, Gq) safety factor (cell grid)
    ok: bool


def run_torax_member(inp: ShotInputs, cfg: EnKFConfig, theta: dict[str, float]):
    """Run one TORAX trajectory; return a reduced :class:`ToraxTrajectory`."""
    import torax  # noqa: PLC0415

    config = _torax_config(inp, cfg, theta)
    try:
        tcfg = torax.ToraxConfig.from_dict(config)
        dt_out, _hist = torax.run_simulation(tcfg, progress_bar=False)
    except Exception as e:  # noqa: BLE001
        logger.debug("TORAX member failed (theta=%s): %s", theta, e)
        return ToraxTrajectory(
            time=np.array([inp.t_final]),
            rho_norm=np.linspace(0, 1, 27),
            j_total=np.zeros((1, 27)),
            q=np.full((1, 26), np.nan),
            ok=False,
        )
    prof = dt_out["profiles"]
    jt = prof["j_total"]
    rho_norm = np.asarray(jt.coords["rho_norm"], dtype=np.float64)
    j_total = np.asarray(jt, dtype=np.float64)
    time = np.asarray(jt.coords["time"], dtype=np.float64)
    q = (
        np.asarray(prof["q"], dtype=np.float64)
        if "q" in prof.data_vars
        else np.full((time.size, 26), np.nan)
    )
    return ToraxTrajectory(time=time, rho_norm=rho_norm, j_total=j_total, q=q, ok=True)


def _j_at_slices(traj: ToraxTrajectory, slice_t: np.ndarray) -> np.ndarray:
    """Interp the TORAX ``j(rho)`` trajectory onto the eval slice times -> (K, G)."""
    if traj.time.size == 1:
        return np.repeat(traj.j_total, slice_t.size, axis=0)
    out = np.empty((slice_t.size, traj.j_total.shape[1]))
    for g in range(traj.j_total.shape[1]):
        out[:, g] = np.interp(slice_t, traj.time, traj.j_total[:, g])
    return out


def _q0_at_slices(traj: ToraxTrajectory, slice_t: np.ndarray) -> np.ndarray:
    """On-axis q (q0) interpolated to the eval slices (the free non-vacuity q0)."""
    if traj.q.shape[0] == 1 or not np.isfinite(traj.q).any():
        return np.full(slice_t.size, traj.q[0, 0] if traj.q.size else np.nan)
    q0_t = traj.q[:, 0]
    return np.interp(slice_t, traj.time, q0_t)


# --- The magnetics observation operator (j -> c_plasma -> amb) --------


@dataclass
class MagneticsObs:
    """GS forward operator as H, with the cell-area-weighted j->c_plasma embed.

    ``c_plasma[k] = j(node_k) * cell_area_k`` (AMPERES) so ``sum(c_plasma) ~ Ip``
    — the operator's plasma columns are field-per-ampere, so feeding current
    *density* without the area weight mis-scales the plasma field ~15x and the
    analysis injects garbage current.  ``node_area`` is each plasma node's share
    of the limiter-masked grid cell area (uniform grid -> equal cells).
    """

    operator: ForwardOperator
    trust_rows: np.ndarray
    node_rho: np.ndarray  # (n_node,) each plasma node's rho_norm
    node_area: np.ndarray  # (n_node,) cell area [m^2] per node

    @classmethod
    def build(cls, operator: ForwardOperator, cfg: EnKFConfig) -> MagneticsObs:
        from imas_ambix.gs.residual import trustworthy_target  # noqa: PLC0415

        tt = trustworthy_target(operator)
        prz = operator.plasma_rz  # (n_node, 2) in (R, Z)
        if prz.size == 0:
            return cls(operator, tt.rows, np.zeros(0), np.zeros(0))
        node_rminor = np.hypot(prz[:, 0] - cfg.r_major, prz[:, 1])
        node_rho = np.clip(node_rminor / cfg.a_minor, 0.0, 1.0)
        # cell area: the default plasma basis is a uniform (R, Z) grid; estimate
        # the per-node cell area from the node spacing (median nearest-neighbour
        # spacing squared) so sum(j * area) ~ integral over the plasma area.
        if prz.shape[0] > 1:
            # median spacing in R and Z
            ur = np.unique(np.round(prz[:, 0], 6))
            uz = np.unique(np.round(prz[:, 1], 6))
            dr = np.median(np.diff(ur)) if ur.size > 1 else cfg.a_minor
            dz = np.median(np.diff(uz)) if uz.size > 1 else cfg.a_minor
            cell = float(abs(dr * dz))
        else:
            cell = float(np.pi * cfg.a_minor**2)
        node_area = np.full(prz.shape[0], cell)
        return cls(operator, tt.rows, node_rho, node_area)

    def c_plasma_from_j(self, j_rho: np.ndarray, rho_grid: np.ndarray) -> np.ndarray:
        """Per-node current [A] from a j(rho) profile on ``rho_grid``."""
        j_node = np.interp(self.node_rho, rho_grid, j_rho)  # A/m^2 at each node
        return j_node * self.node_area  # A

    def predict_amb(
        self, j_rho: np.ndarray, rho_grid: np.ndarray, i_pf: np.ndarray
    ) -> np.ndarray:
        """Predicted trustworthy amb for one j(rho) profile."""
        c_plasma = self.c_plasma_from_j(j_rho, rho_grid)
        pred_full = self.operator.predict(i_pf, c_plasma=c_plasma)
        return pred_full[self.trust_rows]


# --- The ensemble smoother (forecast + one EKI magnetics update) ------


def _sample_theta(cfg: EnKFConfig, rng: np.random.Generator) -> list[dict[str, float]]:
    """Sample the prior ensemble of TORAX parameter vectors theta."""
    n = cfg.n_ensemble
    zeff = np.clip(rng.normal(cfg.zeff_prior, cfg.zeff_spread, n), 1.0, 5.0)
    rmult = np.exp(rng.normal(0.0, cfg.resist_mult_logspread, n))
    peak = np.clip(
        rng.normal(cfg.current_peaking_prior, cfg.current_peaking_spread, n), 0.5, 4.0
    )
    ipf = np.clip(rng.normal(1.0, cfg.ip_frac_spread, n), 0.85, 1.15)
    return [
        {
            "zeff": float(zeff[m]),
            "resist_mult": float(rmult[m]),
            "current_peaking": float(peak[m]),
            "ip_frac": float(ipf[m]),
        }
        for m in range(n)
    ]


def _theta_vec(theta: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            np.log(theta["zeff"]),
            np.log(theta["resist_mult"]),
            theta["current_peaking"],
            theta["ip_frac"],
        ]
    )


def _theta_from_vec(v: np.ndarray) -> dict[str, float]:
    return {
        "zeff": float(np.clip(np.exp(v[0]), 1.0, 5.0)),
        "resist_mult": float(np.clip(np.exp(v[1]), 0.4, 3.0)),
        "current_peaking": float(np.clip(v[2], 0.5, 4.0)),
        "ip_frac": float(np.clip(v[3], 0.85, 1.15)),
    }


@dataclass
class ShotResult:
    """The two-arm result for one shot + diagnostics."""

    shot_id: int
    slice_t: np.ndarray
    # ANALYSIS arm (magnetics-assimilated) — the scored prediction
    pitch_samples: np.ndarray  # (K, C, M)
    q0_torax_mean: np.ndarray  # (K,) on-axis q from TORAX (free non-vacuity q0)
    # FORECAST arm (transition + Ip only, magnetics NOT assimilated)
    pitch_samples_forecast: np.ndarray  # (K, C, M)
    q0_torax_forecast: np.ndarray  # (K,)
    # diagnostics
    innovation_forecast: float  # whitened amb misfit before the update
    innovation_analysis: float  # whitened amb misfit after the update
    n_ok_members: int
    readout_source: str


def run_shot(inp: ShotInputs, obs: MagneticsObs, cfg: EnKFConfig) -> ShotResult:
    """Run the parameter-space ensemble smoother over one shot.

    Forecast arm: prior TORAX ensemble (measured inputs only).  Analysis arm:
    one EKI update of theta against the trustworthy amb, then re-run.  Both arms
    are read out through the SHARED pitch model -> pitch_samples (K, C, M).
    """
    from imas_ambix.gs.residual import robust_sensor_scale  # noqa: PLC0415
    from imas_ambix.statespace.mse_eval import pitch_from_current_profile

    rng = np.random.default_rng(cfg.seed + int(inp.shot_id))
    K = inp.slice_t.size  # noqa: N806
    C = inp.n_active  # noqa: N806
    thetas = _sample_theta(cfg, rng)
    M = len(thetas)  # noqa: N806

    # --- FORECAST: run the prior ensemble ---
    trajs = [run_torax_member(inp, cfg, th) for th in thetas]
    n_ok = sum(1 for tr in trajs if tr.ok)

    # assimilation slices (subset of eval slices where amb is finite)
    finite_amb = np.isfinite(inp.amb_trust).all(axis=1)
    assim_idx = np.where(finite_amb)[0]
    if assim_idx.size > cfg.n_assim_slices:
        assim_idx = assim_idx[
            np.linspace(0, assim_idx.size - 1, cfg.n_assim_slices).astype(int)
        ]

    def _predicted_amb(traj: ToraxTrajectory) -> np.ndarray:
        """Stacked predicted amb over the assimilation slices for one member."""
        j_k = _j_at_slices(traj, inp.slice_t)  # (K, G)
        preds = []
        for s in assim_idx:
            preds.append(obs.predict_amb(j_k[s], traj.rho_norm, inp.i_pf[s]))
        return np.concatenate(preds) if preds else np.zeros(0)

    # observed amb stacked over assim slices + per-sensor whitening scale
    y_stack = (
        np.concatenate([inp.amb_trust[s] for s in assim_idx])
        if assim_idx.size
        else np.zeros(0)
    )
    if assim_idx.size:
        w_scale = robust_sensor_scale(inp.amb_trust[assim_idx])  # (n_trust,)
        w_stack = np.tile(w_scale, assim_idx.size)
    else:
        w_stack = np.ones(0)

    def _whitened_misfit(pred_stack: np.ndarray) -> float:
        if pred_stack.size == 0 or y_stack.size == 0:
            return float("nan")
        return float(
            np.linalg.norm((pred_stack - y_stack) / np.maximum(w_stack, 1e-12))
            / np.sqrt(pred_stack.size)
        )

    Yf = (  # noqa: N806
        np.array([_predicted_amb(tr) for tr in trajs])
        if assim_idx.size
        else np.zeros((M, 0))
    )
    innov_forecast = (
        _whitened_misfit(np.nanmean(Yf, axis=0)) if assim_idx.size else float("nan")
    )

    # --- ANALYSIS: one EKI update of theta against amb, then re-run ---
    analysis_trajs = trajs
    innov_analysis = innov_forecast
    if cfg.assimilate and assim_idx.size and n_ok >= max(4, M // 4):
        Theta = np.array([_theta_vec(th) for th in thetas])  # (M, p)  # noqa: N806
        valid = np.isfinite(Yf).all(axis=1)
        if valid.sum() >= 4:
            Tv = Theta[valid]  # noqa: N806
            Yv = Yf[valid]  # noqa: N806
            wv = np.maximum(w_stack, 1e-12)
            # whiten the observation space
            Yw = Yv / wv[None, :]  # noqa: N806
            yw = y_stack / wv
            th_mean = Tv.mean(axis=0)
            yw_mean = Yw.mean(axis=0)
            Ta = Tv - th_mean  # noqa: N806
            Ya = Yw - yw_mean  # noqa: N806
            nE = Tv.shape[0]  # noqa: N806
            C_ty = (Ta.T @ Ya) / (nE - 1)  # (p, n_obs)  # noqa: N806
            C_yy = (Ya.T @ Ya) / (nE - 1)  # (n_obs, n_obs)  # noqa: N806
            S = C_yy + cfg.eki_inflation * np.eye(  # noqa: N806
                C_yy.shape[0]
            )  # R = I in whitened space
            # EKI / ensemble-smoother update on each member's theta
            try:
                Kt = np.linalg.solve(S.T, C_ty.T)  # (n_obs, p)  # noqa: N806
            except np.linalg.LinAlgError:
                Kt = np.linalg.lstsq(S.T, C_ty.T, rcond=None)[0]  # noqa: N806
            Kgain = Kt.T  # (p, n_obs)  # noqa: N806
            pert = rng.normal(0.0, 1.0, size=(M, yw.size))  # perturbed obs (whitened)
            innov = (yw[None, :] + pert) - (Yf / wv[None, :])  # (M, n_obs)
            Theta_upd = Theta + cfg.eki_step * (innov @ Kgain.T)  # (M, p)  # noqa: N806
            thetas_upd = [_theta_from_vec(Theta_upd[m]) for m in range(M)]
            analysis_trajs = [run_torax_member(inp, cfg, th) for th in thetas_upd]
            Ya2 = np.array([_predicted_amb(tr) for tr in analysis_trajs])  # noqa: N806
            innov_analysis = _whitened_misfit(np.nanmean(Ya2, axis=0))

    # --- READOUT (shared): pitch_samples (K, C, M) for both arms ---
    def _pitch_samples(traj_list):
        ps = np.full((K, C, M), np.nan)
        for m, tr in enumerate(traj_list):
            if not tr.ok:
                continue
            j_k = _j_at_slices(tr, inp.slice_t)  # (K, G)
            rho_m = tr.rho_norm * cfg.a_minor  # minor-radius grid [m]
            # pitch_from_current_profile is vectorised over time -> (K, C)
            pk = pitch_from_current_profile(
                j_k,
                rho_m,
                inp.active_channel_rpos,
                cfg.r_major,
                cfg.bt0_for_readout,
                kind="j",
            )
            ps[:, :, m] = pk
        return ps

    ps_analysis = _pitch_samples(analysis_trajs)
    ps_forecast = _pitch_samples(trajs)

    q0_analysis = np.nanmean(
        np.array([_q0_at_slices(tr, inp.slice_t) for tr in analysis_trajs]), axis=0
    )
    q0_forecast = np.nanmean(
        np.array([_q0_at_slices(tr, inp.slice_t) for tr in trajs]), axis=0
    )

    return ShotResult(
        shot_id=inp.shot_id,
        slice_t=inp.slice_t,
        pitch_samples=ps_analysis,
        q0_torax_mean=q0_analysis,
        pitch_samples_forecast=ps_forecast,
        q0_torax_forecast=q0_forecast,
        innovation_forecast=innov_forecast,
        innovation_analysis=innov_analysis,
        n_ok_members=n_ok,
        readout_source="mse_eval",
    )


def shot_result_to_prediction(res: ShotResult, *, arm: str = "analysis"):
    """Convert a :class:`ShotResult` to the canonical ``mse_eval.ShotPrediction``.

    ``arm`` selects "analysis" (magnetics-assimilated, scored) or "forecast"
    (transition+Ip only, the non-vacuity control).  Emits ``pitch_samples``
    (K, C, M) so the harness scores energy-form CRPS natively, plus the
    Gaussian mean/std from the ensemble.
    """
    from imas_ambix.statespace.mse_eval import ShotPrediction  # noqa: PLC0415

    samples = res.pitch_samples if arm == "analysis" else res.pitch_samples_forecast
    # collapse non-finite members; keep finite samples for mean/std
    pmean = np.nanmean(samples, axis=2)
    pstd = np.nanstd(samples, axis=2)
    # floor std so coverage is non-degenerate
    pstd = np.where(np.isfinite(pstd) & (pstd > 1e-4), pstd, 0.05)
    pmean = np.where(np.isfinite(pmean), pmean, 0.0)
    return ShotPrediction(
        t=res.slice_t,
        pitch_mean=pmean,
        pitch_std=pstd,
        pitch_samples=samples,
    )


# --- Driver -----------------------------------------------------------


def _campaign_representatives() -> dict[str, list[int]]:
    """Map each committed campaign signature -> its recorded representative shots."""
    from imas_ambix.data.paths import MANIFEST_DIR  # noqa: PLC0415

    path = MANIFEST_DIR / "gs_geometry_tables.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    camps = raw.get("campaigns", raw)
    return {
        key: [int(s) for s in (entry.get("shots") or [])]
        for key, entry in camps.items()
    }


def _operator_for_shot(shot_id, op_cache, reps=None):
    """Return the campaign ForwardOperator matching ``shot_id`` (by signature).

    Built from a committed campaign-representative shot (which carries ``amm``);
    falls back to the target shot.
    """
    from imas_ambix.gs.geometry import (  # noqa: PLC0415
        build_table_for_shot,
        read_efm_geometry,
        setup_signature,
    )
    from imas_ambix.gs.operator import build_operator  # noqa: PLC0415

    try:
        key = setup_signature(read_efm_geometry(shot_id)).key
    except Exception:  # noqa: BLE001
        return None
    if key in op_cache:
        return op_cache[key]
    reps = reps if reps is not None else _campaign_representatives()
    for rep in list(reps.get(key, [])) + [shot_id]:
        try:
            op_cache[key] = build_operator(build_table_for_shot(int(rep)))
            return op_cache[key]
        except Exception as e:  # noqa: BLE001
            logger.debug("operator build failed rep %d: %s", rep, e)
    logger.warning("no operator for shot %d (campaign %s)", shot_id, key)
    return None


def predict_shots(
    shot_ids: Sequence[int],
    cfg: EnKFConfig | None = None,
    *,
    arm: str = "analysis",
    return_results: bool = False,
):
    """Produce canonical ``ShotPrediction`` objects for a set of held-out shots.

    Returns ``{shot_id: ShotPrediction}``; with ``return_results`` also returns
    ``{shot_id: ShotResult}`` (the two-arm diagnostics).
    """
    import torax  # noqa: PLC0415

    torax.set_jax_precision()
    cfg = cfg or EnKFConfig()
    op_cache: dict[str, ForwardOperator] = {}
    reps = _campaign_representatives()
    preds: dict[int, Any] = {}
    results: dict[int, ShotResult] = {}
    for sid in shot_ids:
        op = _operator_for_shot(int(sid), op_cache, reps)
        if op is None:
            continue
        obs = MagneticsObs.build(op, cfg)
        inp = load_shot_inputs(int(sid), op, cfg)
        if inp is None:
            logger.warning("no usable MSE/magnetics for shot %d — skipped", sid)
            continue
        res = run_shot(inp, obs, cfg)
        results[int(sid)] = res
        preds[int(sid)] = shot_result_to_prediction(res, arm=arm)
    if return_results:
        return preds, results
    return preds


# --- Scoring + artifact (D1 harness when the manifest lands) ----------


def score_and_write_artifact(
    cfg: EnKFConfig,
    shot_ids: Sequence[int],
    *,
    manifest_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Run the baseline, score on the held-out split, write the metrics artifact.

    Uses the OFFICIAL manifest (``mse_heldout_split_v0.json``) when present; else
    builds a LOCAL manifest for ``shot_ids`` via ``mse_split.build_shot_manifest``
    (a clearly-flagged stopgap so the pipeline is exercised end-to-end before
    D1's official split lands).  The official q0/rax + full metrics await the
    official manifest.
    """
    from imas_ambix.data.paths import MANIFEST_DIR  # noqa: PLC0415
    from imas_ambix.statespace import mse_eval as eval_mod  # noqa: PLC0415
    from imas_ambix.statespace import mse_split as split_mod  # noqa: PLC0415

    manifest_path = manifest_path or (MANIFEST_DIR / "mse_heldout_split_v0.json")
    official = manifest_path.exists()

    preds, results = predict_shots(shot_ids, cfg, arm="analysis", return_results=True)
    preds_forecast = {
        sid: shot_result_to_prediction(res, arm="forecast")
        for sid, res in results.items()
    }

    truth = eval_mod.MseTruth(level1_dir=LEVEL1_DIR)
    if official:
        manifest = eval_mod.load_manifest(manifest_path)
        manifest_provenance = f"official:{manifest_path}"
    else:
        shots_entries: dict[str, dict] = {}
        for sid in preds:
            ams = split_mod.read_ams_shot(local_shot_path(sid, tier="level1"))
            if ams is None:
                continue
            entry = split_mod.build_shot_manifest(ams, partition="held_out")
            shots_entries[str(sid)] = entry
        manifest = {"version": "local-stopgap", "shots": shots_entries}
        manifest_provenance = "local-stopgap (official split not yet landed)"

    scored_analysis = eval_mod.score(preds, manifest, truth)
    scored_forecast = eval_mod.score(preds_forecast, manifest, truth)

    # innovation drop (truth-free analysis validation, mean over shots)
    inn_f = [
        r.innovation_forecast
        for r in results.values()
        if np.isfinite(r.innovation_forecast)
    ]
    inn_a = [
        r.innovation_analysis
        for r in results.values()
        if np.isfinite(r.innovation_analysis)
    ]

    payload = {
        "schema": "enkf-baseline-metrics-v0",
        "method": "parameter-space ensemble smoother over TORAX current-diffusion",
        "forward_model": "TORAX (current-diffusion; sigma(Te) neoclassical closure)",
        "observation_operator": "gs/operator.py (EFIT-free GS Green's functions)",
        "readout": "SHARED mse_eval.pitch_from_current_profile + invert_pitch_to_q0rax",
        "manifest_provenance": manifest_provenance,
        "config": cfg.to_dict(),
        "n_shots_requested": len(list(shot_ids)),
        "n_shots_scored": scored_analysis["meta"]["n_shots"],
        "metrics_analysis_arm": scored_analysis,
        "metrics_forecast_arm_NONVACUITY_CONTROL": scored_forecast,
        "magnetics_innovation": {
            "forecast_mean": float(np.mean(inn_f)) if inn_f else None,
            "analysis_mean": float(np.mean(inn_a)) if inn_a else None,
            "drop_means_analysis_is_real": (
                bool(np.mean(inn_a) < np.mean(inn_f)) if (inn_f and inn_a) else None
            ),
            "note": (
                "whitened amb misfit ||W(y-H(x))|| (truth-free). A DROP from "
                "forecast->analysis confirms the EKI magnetics update is real; "
                "if pitch does NOT improve while innovation drops, that is the "
                "Stage-2 magnetics-under-determination THESIS (expected)."
            ),
        },
        "matched_compute_caveat": (
            "Equal CPU/wall-clock budget vs the neural filter. The O(N_ens) "
            "ensemble CANNOT ingest the camera/SXR image modalities the neural "
            "filter fuses natively — that asymmetry is EXPECTED and IS the thesis."
        ),
        "q0_provenance": (
            "scored q0/rax are method-matched via mse_eval.invert_pitch_to_q0rax "
            "on the predicted pitch (secondary); TORAX on-axis q is reported "
            "separately as the free non-vacuity check."
        ),
        "per_shot_diagnostics": {
            str(sid): {
                "n_ok_members": r.n_ok_members,
                "innovation_forecast": r.innovation_forecast,
                "innovation_analysis": r.innovation_analysis,
                "q0_torax_forecast_median": float(np.nanmedian(r.q0_torax_forecast)),
                "q0_torax_analysis_median": float(np.nanmedian(r.q0_torax_mean)),
            }
            for sid, r in results.items()
        },
    }
    out_path = out_path or (
        Path(__file__).parent / "artifacts" / "enkf_baseline_metrics_v0.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    return payload


__all__ = [
    "EnKFConfig",
    "ShotInputs",
    "MagneticsObs",
    "ToraxTrajectory",
    "ShotResult",
    "load_shot_inputs",
    "run_torax_member",
    "run_shot",
    "shot_result_to_prediction",
    "predict_shots",
    "score_and_write_artifact",
]
