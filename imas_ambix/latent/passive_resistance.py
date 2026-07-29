"""Data-led passive-resistance calibration from coil-only (vacuum) intervals.

The passive L matrix is exact from geometry (two-section flux linkage) but R
is not: the efm passive elements are bounding boxes of a complex 3-D vessel,
so the nominal ring resistances misstate the true conducting paths.  On a
coil-only interval there is no plasma: the measured magnetics minus the
static coil prediction is PURE eddy signal, the drives are measured, and the
only unknowns are a few bounded resistance multipliers — classical parameter
estimation, done standalone so every downstream consumer inherits a calibrated R
instead of re-learning it.

Case-current holdback (binding contract): the measured ``*_case_current``
channels are NEVER inputs here.  The measured-case circuits are moved into
the passive set (:func:`~imas_ambix.latent.temporal_operator.
build_passive_circuit_system` with ``hold_back_cases``) so their currents are
PREDICTED from the remaining drives through the mutual couplings, and the
measured case currents serve purely as held-back per-circuit fitting and
validation targets — the strongest per-circuit test of both L and R the
corpus offers.

Identifiability ladder (never per-shot, never per-slice): a global scale,
then vessel/case, then vessel regions + case pairs, then per-case — each
extra tier of bounded positive multipliers accepted only if the held-out
vacuum shots improve.  Vessel regions are assigned by a machine-agnostic
rule on NORMALISED centroid coordinates of the passive set (fractions of the
set's own radial span and vertical extent — no metre-level lock-in).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imas_ambix.latent.temporal_operator import PassiveCircuitSystem

logger = logging.getLogger(__name__)

#: normalised-radius thresholds of the vessel-region rule: r_norm < INBOARD is
#: the centre-column region, r_norm > OUTBOARD the outer cylinder; remaining
#: circuits with |z|/max|z| > ENDS are the end plates, the rest mid structures.
REGION_INBOARD_RFRAC = 0.25
REGION_OUTBOARD_RFRAC = 0.60
REGION_ENDS_ZFRAC = 0.70

#: the identifiability ladder, least → most DOF
LADDER_LEVELS = ("global", "vessel-case", "regions-casepairs", "regions-percase")

#: bounded positive multipliers (log-space optimisation bounds)
MULTIPLIER_BOUNDS = (0.2, 64.0)


def case_label_by_circuit() -> dict[int, str]:
    """Circuit id → case identity (e.g. ``"p4u"``) for every coil-case
    circuit, from the authoritative ``pfSystems.xml`` correspondence.  The
    identity comes from the case circuit's own NAME ("P4U case current"), not
    ``geometry_confusable_with`` (which names the enclosed WINDING — "p2iu"
    for the P2U case — and would mis-pair the P2 cases)."""
    from imas_ambix.gs import operator as op  # noqa: PLC0415

    return {
        circ: case.name.split()[0].lower()
        for circ, case in op._CASE_BY_CIRCUIT_ID.items()  # noqa: SLF001
    }


def resistance_group_labels(
    circuits: np.ndarray,
    centroid_r: np.ndarray,
    centroid_z: np.ndarray,
    level: str,
    *,
    case_of: dict[int, str] | None = None,
) -> list[str]:
    """Group label of each passive circuit at one ladder level.

    ``case_of`` maps a coil-case circuit id to its coil label (default: the
    machine metadata via :func:`case_label_by_circuit`).  Vessel regions use
    normalised centroid coordinates over the GIVEN circuit set, so the rule
    transfers across machines unchanged.
    """
    if level not in LADDER_LEVELS:
        raise ValueError(f"unknown ladder level {level!r} (want {LADDER_LEVELS})")
    if case_of is None:
        case_of = case_label_by_circuit()
    r = np.asarray(centroid_r, dtype=np.float64)
    z = np.asarray(centroid_z, dtype=np.float64)
    r_span = max(float(r.max() - r.min()), 1e-9)
    r_norm = (r - r.min()) / r_span
    z_frac = np.abs(z) / max(float(np.abs(z).max()), 1e-9)

    labels: list[str] = []
    for i, circ in enumerate(np.asarray(circuits, dtype=np.int64)):
        coil = case_of.get(int(circ))
        if level == "global":
            labels.append("all")
        elif coil is not None:
            if level == "vessel-case":
                labels.append("case")
            elif level == "regions-casepairs":
                labels.append(f"case:{coil[:-1]}")  # p4u/p4l share one build
            else:
                labels.append(f"case:{coil}")
        elif level == "vessel-case":
            labels.append("vessel")
        elif r_norm[i] < REGION_INBOARD_RFRAC:
            labels.append("vessel:inboard")
        elif r_norm[i] > REGION_OUTBOARD_RFRAC:
            labels.append("vessel:outboard")
        elif z_frac[i] > REGION_ENDS_ZFRAC:
            labels.append("vessel:ends")
        else:
            labels.append("vessel:mid")
    return labels


@dataclass
class ResistanceCalibration:
    """A fitted resistance model: multiplier per group at one ladder level."""

    level: str
    group_multipliers: dict[str, float]
    provenance: dict

    def per_circuit(
        self,
        circuits: np.ndarray,
        centroid_r: np.ndarray,
        centroid_z: np.ndarray,
        *,
        case_of: dict[int, str] | None = None,
    ) -> np.ndarray:
        """Per-circuit multipliers for an arbitrary passive circuit set.

        Fail-loud on an unknown group (a silent 1.0 would un-calibrate a
        region without anyone noticing); groups the calibration carries but
        the set does not use are simply unused.
        """
        labels = resistance_group_labels(
            circuits, centroid_r, centroid_z, self.level, case_of=case_of
        )
        missing = sorted({lb for lb in labels if lb not in self.group_multipliers})
        if missing:
            raise KeyError(
                f"calibration level {self.level!r} has no multiplier for "
                f"group(s) {missing}"
            )
        return np.array([self.group_multipliers[lb] for lb in labels])


def save_calibration(path: Path | str, cal: ResistanceCalibration) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "kind": "vacuum-passive-resistance-calibration",
                "level": cal.level,
                "group_multipliers": cal.group_multipliers,
                "region_rule": {
                    "inboard_rfrac": REGION_INBOARD_RFRAC,
                    "outboard_rfrac": REGION_OUTBOARD_RFRAC,
                    "ends_zfrac": REGION_ENDS_ZFRAC,
                },
                "provenance": cal.provenance,
            },
            indent=2,
        )
    )


def load_calibration(path: Path | str) -> ResistanceCalibration:
    obj = json.loads(Path(path).read_text())
    if obj.get("kind") != "vacuum-passive-resistance-calibration":
        raise ValueError(f"{path}: not a resistance calibration artifact")
    return ResistanceCalibration(
        level=str(obj["level"]),
        group_multipliers={k: float(v) for k, v in obj["group_multipliers"].items()},
        provenance=dict(obj.get("provenance", {})),
    )


# ---------------------------------------------------------------------------
# Vectorised ZOH mode response (uniform cadence)
# ---------------------------------------------------------------------------
def zoh_mode_response(
    tau: np.ndarray,
    dt: float,
    psi_m: np.ndarray,
    volt_m: np.ndarray | None = None,
) -> np.ndarray:
    """Exact-ZOH mode response on a UNIFORM time grid, vectorised.

    Same recurrence as :func:`~imas_ambix.latent.temporal_operator.
    integrate_eddy_ode` (``a_t = e^{−Δt/τ} a_{t−1} − (τ/Δt)(1 − e^{−Δt/τ})
    ΔΨ_t``, ``a_0 = 0``) — a linear constant-coefficient recurrence per mode,
    evaluated with :func:`scipy.signal.lfilter` so the per-sample Python loop
    disappears (the calibration objective integrates ~10⁵ samples × ~150
    modes per evaluation).  ``volt_m`` adds the voltage-type drive (step-
    midpoint value, response coefficient ``τ(1 − e^{−Δt/τ})``) — the galvanic
    case-wiring EMF that is not the derivative of a linked flux.
    """
    from scipy.signal import lfilter  # noqa: PLC0415

    tau = np.asarray(tau, dtype=np.float64)
    psi_m = np.asarray(psi_m, dtype=np.float64)
    dt = float(dt)
    u = np.zeros_like(psi_m)
    u[1:] = -(psi_m[1:] - psi_m[:-1])
    decay = np.exp(-dt / tau)
    coeff = tau / dt * (1.0 - decay)
    a = np.empty_like(psi_m)
    for m in range(psi_m.shape[1]):
        a[:, m] = lfilter([coeff[m]], [1.0, -decay[m]], u[:, m])
    if volt_m is not None:
        volt_m = np.asarray(volt_m, dtype=np.float64)
        vbar = np.zeros_like(volt_m)
        vbar[1:] = 0.5 * (volt_m[1:] + volt_m[:-1])
        vcoeff = tau * (1.0 - decay)
        for m in range(psi_m.shape[1]):
            a[:, m] += lfilter([vcoeff[m]], [1.0, -decay[m]], vbar[:, m])
    return a


# ---------------------------------------------------------------------------
# Fit data + per-shot loss terms
# ---------------------------------------------------------------------------
@dataclass
class VacuumShotData:
    """One shot's prepared coil-only arrays (leading coil-only run only)."""

    shot: int
    campaign: str
    stratum: str
    dt: float  # uniform raw cadence [s]
    psi_circ: np.ndarray  # (T, P) drive flux per passive circuit [Wb]
    meas_resid: np.ndarray  # (T, S) measured − non-case static prediction
    sigma: np.ndarray  # (S,) per-shot robust channel scale
    case_meas: np.ndarray  # (T, n_case) measured case currents [A]; NaN absent
    #: (T, C) drive-channel currents [A·turn] in the system's channels
    #: order — required by the structure discovery (drive-column edits and
    #: galvanic voltage terms are functions of the raw drives, not of the
    #: precomputed ``psi_circ``); None on data prepared before the field
    i_drive: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return int(self.psi_circ.shape[0])


@dataclass
class ModeMaps:
    """Per-campaign θ-dependent maps shared by every shot of the campaign."""

    tau: np.ndarray  # (P,)
    v: np.ndarray  # (P, P) L-orthonormal eigenvectors
    a_sens_modes: np.ndarray  # (S, P)
    case_v: np.ndarray  # (n_case, P) case-circuit rows of v


def campaign_mode_maps(
    system: PassiveCircuitSystem, multipliers: np.ndarray
) -> ModeMaps:
    """Read Nova's mode system for one candidate resistance model."""
    tau, v = system.mode_system(np.asarray(multipliers, dtype=np.float64))
    case_rows = np.array(
        [system.measured_channel_row[ch] for ch in sorted(system.measured_channel_row)],
        dtype=np.int64,
    )
    return ModeMaps(
        tau=tau,
        v=v,
        a_sens_modes=system.a_circuit @ v,
        case_v=v[case_rows] if case_rows.size else np.zeros((0, v.shape[0])),
    )


def shot_loss_terms(
    data: VacuumShotData,
    maps: ModeMaps,
    sigma_med: np.ndarray,
    sigma_case: np.ndarray,
) -> tuple[float, int, float, int]:
    """Whitened sum-of-squares terms of one shot under one resistance model.

    Magnetics: eddy-signal residual ``meas_resid − a_sens·i(t)``, per-channel
    mean removed over the interval (the offset-nuisance intercept used by the
    static vacuum fit), whitened by the pooled channel scale.
    Case currents: held-back measured vs predicted, per-channel mean removed
    (instrumental zero offset), whitened by the pooled case scale.
    Returns ``(ss_mag, n_mag, ss_case, n_case)``.
    """
    psi_m = data.psi_circ @ maps.v  # (T, P)
    a = zoh_mode_response(maps.tau, data.dt, psi_m)

    resid = data.meas_resid - a @ maps.a_sens_modes.T
    with np.errstate(invalid="ignore"):
        resid = resid - np.nanmean(resid, axis=0, keepdims=True)
    white = resid / sigma_med
    finite = np.isfinite(white)
    ss_mag = float(np.nansum(np.where(finite, white, 0.0) ** 2))
    n_mag = int(finite.sum())

    ss_case, n_case = 0.0, 0
    if data.case_meas.size and maps.case_v.size:
        rc = data.case_meas - a @ maps.case_v.T
        with np.errstate(invalid="ignore"):
            rc = rc - np.nanmean(rc, axis=0, keepdims=True)
        wc = rc / sigma_case
        fin = np.isfinite(wc)
        ss_case = float(np.nansum(np.where(fin, wc, 0.0) ** 2))
        n_case = int(fin.sum())
    return ss_mag, n_mag, ss_case, n_case


def pooled_loss(
    theta: np.ndarray,
    group_index: dict[str, np.ndarray],
    systems: dict[str, PassiveCircuitSystem],
    shots: list[VacuumShotData],
    sigma_med: dict[str, np.ndarray],
    sigma_case: dict[str, np.ndarray],
    *,
    w_case: float = 1.0,
) -> dict[str, float]:
    """Combined mean whitened square over a shot pool for one θ.

    ``theta`` is the per-GROUP multiplier vector; ``group_index[campaign]``
    maps it onto that campaign's circuits.  Returns the combined loss and its
    magnetics / case components (all means, so pools of different size
    compare directly).
    """
    maps = {
        key: campaign_mode_maps(systems[key], theta[group_index[key]])
        for key in systems
    }
    tot_mag = tot_case = 0.0
    n_mag = n_case = 0
    for d in shots:
        sm, nm, sc, nc = shot_loss_terms(
            d, maps[d.campaign], sigma_med[d.campaign], sigma_case[d.campaign]
        )
        tot_mag += sm
        n_mag += nm
        tot_case += sc
        n_case += nc
    mag = tot_mag / max(n_mag, 1)
    case = tot_case / max(n_case, 1)
    return {
        "combined": mag + w_case * case,
        "mag": mag,
        "case": case,
        "n_mag": float(n_mag),
        "n_case": float(n_case),
    }


# ---------------------------------------------------------------------------
# Structure discovery: wiring, constraint reductions, adjacency couplings
# ---------------------------------------------------------------------------
#
# The magnetic L is exact geometry and never touched; what the vacuum data can
# still falsify is the CONDUCTOR TOPOLOGY the diagonal-R isolated-ring model
# assumes. Three structured, physically constrained hypothesis families are
# supported:
#
# * case-coil galvanic wiring — the measured sibling identity
#   ``<coil>_current = <coil>_coil_current + <coil>_case_current`` (exact on
#   every pool shot) meters the case current inside the coil's supply
#   circuit: the case loop sees the winding's terminal voltage.  Candidate
#   case-row drive ``V_t = g_v·dΛ_w/dt + r_w·i_coil`` with Λ_w the winding's
#   geometric flux from the measured drives (a static edit of the case row's
#   drive-linkage columns) and the resistive term a voltage-type drive.
#   Approximation, recorded: the winding's linkage of PASSIVE-state currents
#   (including the case's own) is dropped from V_t — keeping it makes the
#   generalised eigenproblem asymmetric; the fitted g_v/r_w and the per-case
#   resistance multiplier absorb the diagonal part.
#
# * pair wiring as constraint REDUCTIONS — series (I_i = I_j) / anti-series
#   (I_i = −I_j) merges of measured case pairs that move as one circuit,
#   expressed as a reduction
#   map C with L→CᵀLC, R→CᵀRC, drives→Cᵀ·; and common/differential drive-
#   gain corrections for the un-separable up/down coil pairs.
#
# * adjacency-restricted galvanic couplings — the physical vessel is a
#   continuous welded shell, so ADJACENT elements share conductor.  Candidate
#   off-diagonal R stamps ``ρ·(e_i − e_j)(e_i − e_j)ᵀ`` (SPD-preserving, a
#   shared branch of resistance ρ) restricted to a nearest-neighbour graph
#   whose threshold is normalised by the circuits' own section scales — a
#   machine-agnostic rule, never a free P×P interaction fit.


def case_parent_coil_channels(case_channel: str, coil_channels: list[str]) -> list[str]:
    """Winding channels galvanically tied to a measured case channel.

    Machine-agnostic prefix rule: case ``p2l_case_current`` (family ``p2``,
    position ``l``) matches every ``*_coil_current`` whose coil label starts
    with the family and ends with the position — ``p2il`` and ``p2ol`` for
    the doubly-wound P2, exactly ``p4u`` for P4U.  The measured sibling
    identity (``plain = Σ coils + case``) is what grounds the mapping.
    """
    base = case_channel.split("_")[0]
    family, pos = base[:-1], base[-1]
    return sorted(
        ch
        for ch in coil_channels
        if ch.endswith("_coil_current")
        and (b := ch.split("_")[0]).startswith(family)
        and b.endswith(pos)
    )


def neighbour_edges(
    centroid_r: np.ndarray,
    centroid_z: np.ndarray,
    section_scale: np.ndarray,
    *,
    factor: float = 1.5,
    exclude_rows: set[int] | frozenset[int] = frozenset(),
) -> list[tuple[int, int]]:
    """Nearest-neighbour candidate edges of the passive set.

    An edge (i, j) is a candidate galvanic coupling when the centroid
    distance is within ``factor × (s_i + s_j)/2`` — touching or nearly
    touching cross-sections under each pair's OWN size scale (dimensionless
    rule; transfers across machines).  ``exclude_rows`` keeps the coil-case
    circuits out (their wiring is the dedicated hypothesis above).
    """
    r = np.asarray(centroid_r, dtype=np.float64)
    z = np.asarray(centroid_z, dtype=np.float64)
    s = np.asarray(section_scale, dtype=np.float64)
    edges: list[tuple[int, int]] = []
    for i in range(r.size):
        if i in exclude_rows:
            continue
        for j in range(i + 1, r.size):
            if j in exclude_rows:
                continue
            d = np.hypot(r[i] - r[j], z[i] - z[j])
            if d <= factor * 0.5 * (s[i] + s[j]):
                edges.append((i, j))
    return edges


def series_reduction(n: int, pairs: list[tuple[int, int, int]]) -> np.ndarray:
    """Constraint-reduction map C (n, n − len(pairs)) for wired pairs.

    Each pair (i, j, sign) imposes ``I_j = sign · I_i`` — one merged state
    carries both circuits.  With ``I = C q`` the reduced system is
    ``CᵀLC q̇ + CᵀRC q = Cᵀu``: series adds the loop EMFs
    (L_eff = L_ii + L_jj ± 2M, R_eff = R_i + R_j — the classical result),
    which is exactly what the congruence produces.  Pairs must be disjoint.
    """
    used: set[int] = set()
    for i, j, _s in pairs:
        if i == j or i in used or j in used:
            raise ValueError(f"pairs must be disjoint, got {pairs}")
        used.update((i, j))
    partner = {i: (j, s) for i, j, s in pairs}
    keep = [k for k in range(n) if k not in {j for _i, j, _s in pairs}]
    c = np.zeros((n, len(keep)))
    for q, k in enumerate(keep):
        c[k, q] = 1.0
        if k in partner:
            j, s = partner[k]
            c[j, q] = float(s)
    return c


@dataclass
class PassiveStructure:
    """Discovered conductor topology + its fitted bounded continuous DOF.

    ``case_series_pairs``: held-back case channels wired as one circuit
    (sign +1 series / −1 anti-series).  ``case_wiring``: per case channel the
    galvanic drive ``{parents, g_v, r_w}``.  ``pair_drive_gains``: per coil
    pair ``{channels: [u, l], common, differential}`` corrections on the
    drive columns.  ``adjacency``: per campaign the accepted neighbour
    couplings ``{i, j (circuit ids), r_couple}``.  ``r_level`` /
    ``r_group_multipliers``: the jointly-refit resistance calibration.
    """

    case_series_pairs: list[dict]
    case_wiring: dict[str, dict]
    pair_drive_gains: list[dict]
    adjacency: dict[str, list[dict]]
    neighbour_rule: dict
    r_level: str
    r_group_multipliers: dict[str, float]
    provenance: dict


def save_structure(path: Path | str, s: PassiveStructure) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "kind": "vacuum-passive-structure-calibration",
                "case_series_pairs": s.case_series_pairs,
                "case_wiring": s.case_wiring,
                "pair_drive_gains": s.pair_drive_gains,
                "adjacency": s.adjacency,
                "neighbour_rule": s.neighbour_rule,
                "r_level": s.r_level,
                "r_group_multipliers": s.r_group_multipliers,
                "provenance": s.provenance,
            },
            indent=2,
        )
    )


def load_structure(path: Path | str) -> PassiveStructure:
    obj = json.loads(Path(path).read_text())
    if obj.get("kind") != "vacuum-passive-structure-calibration":
        raise ValueError(f"{path}: not a passive-structure artifact")
    return PassiveStructure(
        case_series_pairs=list(obj["case_series_pairs"]),
        case_wiring={k: dict(v) for k, v in obj["case_wiring"].items()},
        pair_drive_gains=list(obj["pair_drive_gains"]),
        adjacency={k: list(v) for k, v in obj["adjacency"].items()},
        neighbour_rule=dict(obj.get("neighbour_rule", {})),
        r_level=str(obj["r_level"]),
        r_group_multipliers={
            k: float(v) for k, v in obj["r_group_multipliers"].items()
        },
        provenance=dict(obj.get("provenance", {})),
    )


@dataclass
class StructuredModeMaps:
    """θ-dependent maps of one campaign under one structure hypothesis set.

    ``v_phys`` (P, K) maps mode amplitudes to PHYSICAL circuit currents
    (constraint reduction folded: ``v_phys = C v_red``), so sensors, case
    targets and downstream consumers never see the reduced coordinates.
    Mode flux/voltage drives assemble from the raw drive currents:
    ``psi_m = i_drive @ drive_flux.T``, ``volt_m = i_drive @ drive_volt.T``.
    """

    tau: np.ndarray  # (K,)
    v_phys: np.ndarray  # (P, K)
    a_sens_modes: np.ndarray  # (S, K)
    case_map: np.ndarray  # (n_case, K) sorted-case-channel rows of v_phys
    drive_flux: np.ndarray  # (K, C) [Wb/A]
    drive_volt: np.ndarray | None  # (K, C) [Ω] or None when no galvanic term


@dataclass
class StructureHypothesis:
    """One campaign's frozen hypothesis set + precomputed geometry arrays.

    Everything θ-independent lives here so a candidate θ costs one cheap
    eigensolve: the reduction map, the adjacency edge rows, the wiring rows
    (case row, parent drive columns, parent winding-flux rows of the drive
    linkage), the coil-pair drive columns, and the R-multiplier group index.
    """

    system: PassiveCircuitSystem
    c_reduce: np.ndarray  # (P, Q)
    group_index: np.ndarray  # (P,) → R-multiplier θ slot per circuit
    edges: list[tuple[int, int]]  # adjacency candidate rows (θ slots in order)
    wiring_rows: np.ndarray  # (n_wire,) case circuit rows (θ slots in order)
    wiring_lam: np.ndarray  # (n_wire, C) Σ-parent winding flux rows [Wb/A]
    wiring_sel: np.ndarray  # (n_wire, C) parent column selectors (0/1)
    pair_cols: list[tuple[int, int]]  # coil-pair (u, l) drive columns


def coil_pair_channels(coil_channels: list[str]) -> list[tuple[str, str]]:
    """Up/down partner pairs among the drive channels (machine label rule).

    A channel pairs with the one whose coil label differs only in the final
    ``u``/``l`` position (``p4u_coil_current`` ↔ ``p4l_coil_current``,
    ``p6u_current`` ↔ ``p6l_current``) among the measured drive channels.
    Returned as (upper, lower), sorted by label.
    """
    bases = {ch.split("_")[0]: ch for ch in coil_channels}
    pairs: list[tuple[str, str]] = []
    for b in sorted(bases):
        if b.endswith("u") and (low := b[:-1] + "l") in bases:
            pairs.append((bases[b], bases[low]))
    return pairs


def build_structure_hypothesis(
    system: PassiveCircuitSystem,
    group_index: np.ndarray,
    *,
    case_series: list[tuple[str, str, int]] | None = None,
    wiring_cases: list[str] | None = None,
    drive_linkage: tuple[list[str], np.ndarray] | None = None,
    pair_channels: list[tuple[str, str]] | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> StructureHypothesis:
    """Freeze one hypothesis set into θ-independent arrays for one campaign.

    ``case_series``: (case_channel_i, case_channel_j, sign) wired pairs.
    ``wiring_cases``: held-back case channels given the galvanic drive (θ
    slots in this order); requires ``drive_linkage`` (channels, lam) from
    :func:`~imas_ambix.latent.temporal_operator.build_drive_linkage`.
    ``pair_channels``: coil pairs given common/differential drive gains.
    ``edges``: adjacency candidate row pairs (θ slots in list order).
    """
    n = system.n_circuits
    row_pairs: list[tuple[int, int, int]] = []
    for ch_i, ch_j, sign in case_series or []:
        row_pairs.append(
            (
                system.measured_channel_row[ch_i],
                system.measured_channel_row[ch_j],
                int(sign),
            )
        )
    c_reduce = series_reduction(n, row_pairs) if row_pairs else np.eye(n)

    wiring_cases = list(wiring_cases or [])
    n_wire = len(wiring_cases)
    n_drive = len(system.channels)
    wiring_rows = np.array(
        [system.measured_channel_row[ch] for ch in wiring_cases], dtype=np.int64
    )
    wiring_lam = np.zeros((n_wire, n_drive))
    wiring_sel = np.zeros((n_wire, n_drive))
    if n_wire:
        if drive_linkage is None:
            raise ValueError("wiring_cases needs the drive_linkage (channels, lam)")
        lam_channels, lam = drive_linkage
        lam_row = {ch: i for i, ch in enumerate(lam_channels)}
        col_of = {ch: i for i, ch in enumerate(system.channels)}
        for w_i, case_ch in enumerate(wiring_cases):
            parents = case_parent_coil_channels(case_ch, system.channels)
            if not parents:
                raise ValueError(f"no parent winding channels for {case_ch}")
            for p in parents:
                wiring_sel[w_i, col_of[p]] = 1.0
                # the winding's flux from EVERY measured drive (its own self
                # term included) — restricted to the system's drive columns
                for d_ch, d_col in col_of.items():
                    wiring_lam[w_i, d_col] += lam[lam_row[p], lam_row[d_ch]]

    pair_cols: list[tuple[int, int]] = []
    for ch_u, ch_l in pair_channels or []:
        col_of = {ch: i for i, ch in enumerate(system.channels)}
        pair_cols.append((col_of[ch_u], col_of[ch_l]))

    return StructureHypothesis(
        system=system,
        c_reduce=c_reduce,
        group_index=np.asarray(group_index, dtype=np.int64),
        edges=list(edges or []),
        wiring_rows=wiring_rows,
        wiring_lam=wiring_lam,
        wiring_sel=wiring_sel,
        pair_cols=pair_cols,
    )


def structured_mode_maps(
    hyp: StructureHypothesis,
    multipliers: np.ndarray,
    *,
    edge_r: np.ndarray | None = None,
    g_v: np.ndarray | None = None,
    r_w: np.ndarray | None = None,
    pair_gains: np.ndarray | None = None,
) -> StructuredModeMaps:
    """Solve one campaign's structured eigenproblem for one candidate θ.

    ``multipliers`` (P,) scale the diagonal ring resistances; ``edge_r``
    (per hyp.edges) adds the SPD adjacency stamps; ``g_v`` / ``r_w`` (per
    hyp.wiring_rows) apply the galvanic case wiring; ``pair_gains``
    (n_pairs, 2: common, differential) correct the pair drive columns.
    """
    from scipy.linalg import eigh  # noqa: PLC0415

    system = hyp.system
    n = system.n_circuits
    r_phys = np.diag(system.r_diag * np.asarray(multipliers, dtype=np.float64))
    if edge_r is not None and len(hyp.edges):
        for (i, j), rho in zip(hyp.edges, np.asarray(edge_r), strict=True):
            r_phys[i, i] += rho
            r_phys[j, j] += rho
            r_phys[i, j] -= rho
            r_phys[j, i] -= rho
    c = hyp.c_reduce
    l_red = c.T @ system.lmat @ c
    r_red = c.T @ r_phys @ c
    w, v_red = eigh(r_red, l_red)
    tau = 1.0 / np.clip(w, 1e-12, None)
    v_phys = c @ v_red

    m_eff = system.m_channel
    volt_cols = None
    if g_v is not None and hyp.wiring_rows.size:
        m_eff = m_eff.copy()
        # +g_v·dΛ_w/dt on the case row == −g_v·Λ_w in the linked-flux columns
        # (the mode drive is −dΨ/dt)
        m_eff[hyp.wiring_rows] -= np.asarray(g_v)[:, np.newaxis] * hyp.wiring_lam
    if r_w is not None and hyp.wiring_rows.size:
        volt_cols = np.zeros((n, len(system.channels)))
        volt_cols[hyp.wiring_rows] = np.asarray(r_w)[:, np.newaxis] * hyp.wiring_sel
    if pair_gains is not None and hyp.pair_cols:
        if m_eff is system.m_channel:
            m_eff = m_eff.copy()
        for (cu, cl), (gc, gd) in zip(
            hyp.pair_cols, np.asarray(pair_gains).reshape(-1, 2), strict=True
        ):
            col_u = system.m_channel[:, cu]
            col_l = system.m_channel[:, cl]
            common = 0.5 * (col_u + col_l)
            differ = 0.5 * (col_u - col_l)
            m_eff[:, cu] += gc * common + gd * differ
            m_eff[:, cl] += gc * common - gd * differ

    case_rows = np.array(
        [system.measured_channel_row[ch] for ch in sorted(system.measured_channel_row)],
        dtype=np.int64,
    )
    return StructuredModeMaps(
        tau=tau,
        v_phys=v_phys,
        a_sens_modes=system.a_circuit @ v_phys,
        case_map=(
            v_phys[case_rows] if case_rows.size else np.zeros((0, v_phys.shape[1]))
        ),
        drive_flux=v_phys.T @ m_eff,
        drive_volt=None if volt_cols is None else v_phys.T @ volt_cols,
    )


def structured_shot_loss(
    data: VacuumShotData,
    maps: StructuredModeMaps,
    sigma_med: np.ndarray,
    sigma_case: np.ndarray,
) -> tuple[float, int, float, int]:
    """Whitened loss terms of one shot under one structured model.

    Same contract as :func:`shot_loss_terms` (offset-nuisance magnetics +
    held-back case targets) with the drives assembled from the RAW drive
    currents so the structure's drive-column edits and galvanic voltage
    terms take effect.  Requires ``data.i_drive``.
    """
    if data.i_drive is None:
        raise ValueError("structured loss needs VacuumShotData.i_drive")
    psi_m = data.i_drive @ maps.drive_flux.T
    volt_m = None if maps.drive_volt is None else data.i_drive @ maps.drive_volt.T
    a = zoh_mode_response(maps.tau, data.dt, psi_m, volt_m=volt_m)

    resid = data.meas_resid - a @ maps.a_sens_modes.T
    with np.errstate(invalid="ignore"):
        resid = resid - np.nanmean(resid, axis=0, keepdims=True)
    white = resid / sigma_med
    finite = np.isfinite(white)
    ss_mag = float(np.nansum(np.where(finite, white, 0.0) ** 2))
    n_mag = int(finite.sum())

    ss_case, n_case = 0.0, 0
    if data.case_meas.size and maps.case_map.size:
        rc = data.case_meas - a @ maps.case_map.T
        with np.errstate(invalid="ignore"):
            rc = rc - np.nanmean(rc, axis=0, keepdims=True)
        wc = rc / sigma_case
        fin = np.isfinite(wc)
        ss_case = float(np.nansum(np.where(fin, wc, 0.0) ** 2))
        n_case = int(fin.sum())
    return ss_mag, n_mag, ss_case, n_case


def structure_hypothesis_parts(
    system: PassiveCircuitSystem,
    structure: PassiveStructure,
    *,
    campaign: str | None = None,
    drive_linkage: tuple[list[str], np.ndarray] | None = None,
) -> tuple[StructureHypothesis, dict]:
    """Instantiate a saved structure artifact on one circuit system.

    Elements that do not apply to the system are dropped automatically —
    the measured-cases-as-drives system (``hold_back_cases=False``) has no
    case circuits in its passive set, so the case wiring and series pairs
    apply only to the holdback form while the adjacency couplings, pair
    drive gains and resistance multipliers apply to both.  Returns the
    frozen hypothesis plus the fitted θ-part arrays ready for
    :func:`structured_mode_maps`.
    """
    case_series = [
        (p["channels"][0], p["channels"][1], int(p["sign"]))
        for p in structure.case_series_pairs
        if all(ch in system.measured_channel_row for ch in p["channels"])
    ]
    wiring_cases = sorted(
        ch for ch in structure.case_wiring if ch in system.measured_channel_row
    )
    if wiring_cases and drive_linkage is None:
        raise ValueError(
            "structure carries case wiring for this system — pass drive_linkage"
        )
    pair_channels = [
        (p["channels"][0], p["channels"][1])
        for p in structure.pair_drive_gains
        if all(ch in system.channels for ch in p["channels"])
    ]
    row_of = {int(c): i for i, c in enumerate(system.circuits)}
    edges: list[tuple[int, int]] = []
    edge_r: list[float] = []
    for rec in structure.adjacency.get(campaign or "", []):
        if int(rec["i"]) in row_of and int(rec["j"]) in row_of:
            edges.append((row_of[int(rec["i"])], row_of[int(rec["j"])]))
            edge_r.append(float(rec["r_couple"]))

    cal = ResistanceCalibration(
        level=structure.r_level,
        group_multipliers=structure.r_group_multipliers,
        provenance={},
    )
    multipliers = cal.per_circuit(system.circuits, system.centroid_r, system.centroid_z)

    hyp = build_structure_hypothesis(
        system,
        np.zeros(system.n_circuits, dtype=np.int64),
        case_series=case_series,
        wiring_cases=wiring_cases,
        drive_linkage=drive_linkage if wiring_cases else None,
        pair_channels=pair_channels,
        edges=edges,
    )
    parts: dict = {"multipliers": multipliers}
    if wiring_cases:
        parts["g_v"] = np.array(
            [structure.case_wiring[ch]["g_v"] for ch in wiring_cases]
        )
        parts["r_w"] = np.array(
            [structure.case_wiring[ch]["r_w"] for ch in wiring_cases]
        )
    if pair_channels:
        parts["pair_gains"] = np.array(
            [
                (p["common"], p["differential"])
                for p in structure.pair_drive_gains
                if all(ch in system.channels for ch in p["channels"])
            ]
        ).reshape(-1, 2)
    if edges:
        parts["edge_r"] = np.array(edge_r)
    return hyp, parts


def structured_reduced_basis(
    system: PassiveCircuitSystem,
    structure: PassiveStructure,
    *,
    sensor_scale: np.ndarray,
    k: int,
    cells: np.ndarray,
    campaign: str | None = None,
    drive_linkage: tuple[list[str], np.ndarray] | None = None,
):
    """Mode-reduced eigenbasis of a structured circuit system.

    Same k-mode history-relevance selection as
    :func:`~imas_ambix.latent.temporal_operator.reduce_passive_system`
    (``τ·‖a_sens/scale‖``, slowest-first), computed over the STRUCTURED
    eigenmodes; drive couplings carry the wiring flux edits and pair gains,
    and ``volt_channel`` carries the galvanic terms for the raw trajectory.
    """
    from imas_ambix.latent.temporal_operator import (  # noqa: PLC0415
        PassiveEigenbasis,
    )

    hyp, parts = structure_hypothesis_parts(
        system, structure, campaign=campaign, drive_linkage=drive_linkage
    )
    smaps = structured_mode_maps(hyp, **parts)
    scale = np.clip(np.asarray(sensor_scale, dtype=np.float64), 1e-12, None)
    relevance = smaps.tau * np.linalg.norm(
        smaps.a_sens_modes / scale[:, np.newaxis], axis=0
    )
    keep = np.argsort(relevance)[::-1][: int(k)]
    keep = keep[np.argsort(smaps.tau[keep])[::-1]]  # slowest-first
    g_modes = system.g_grid @ smaps.v_phys[:, keep]
    volt = None
    if smaps.drive_volt is not None and np.any(smaps.drive_volt):
        volt = smaps.drive_volt[keep]
    return PassiveEigenbasis(
        tau=smaps.tau[keep],
        v=smaps.v_phys[:, keep],
        a_sensor=smaps.a_sens_modes[:, keep],
        g_grid=g_modes,
        m_channel=smaps.drive_flux[keep],
        m_cell=g_modes[np.asarray(cells)].T,
        resistivity=float(system.resistivity),
        volt_channel=volt,
    )


__all__ = [
    "LADDER_LEVELS",
    "MULTIPLIER_BOUNDS",
    "ModeMaps",
    "PassiveStructure",
    "ResistanceCalibration",
    "StructureHypothesis",
    "StructuredModeMaps",
    "VacuumShotData",
    "build_structure_hypothesis",
    "campaign_mode_maps",
    "case_label_by_circuit",
    "case_parent_coil_channels",
    "coil_pair_channels",
    "load_calibration",
    "load_structure",
    "neighbour_edges",
    "pooled_loss",
    "resistance_group_labels",
    "save_calibration",
    "save_structure",
    "series_reduction",
    "shot_loss_terms",
    "structure_hypothesis_parts",
    "structured_mode_maps",
    "structured_reduced_basis",
    "structured_shot_loss",
    "zoh_mode_response",
]
