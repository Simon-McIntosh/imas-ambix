"""Data-led passive-resistance calibration from coil-only (vacuum) intervals.

The passive L matrix is exact from geometry (two-section flux linkage) but R
is not: the efm passive elements are bounding boxes of a complex 3-D vessel,
so the nominal ring resistances misstate the true conducting paths.  On a
coil-only interval there is no plasma: the measured magnetics minus the
static coil prediction is PURE eddy signal, the drives are measured, and the
only unknowns are a few bounded resistance multipliers — classical parameter
estimation, done standalone so every downstream rung inherits a calibrated R
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
def zoh_mode_response(tau: np.ndarray, dt: float, psi_m: np.ndarray) -> np.ndarray:
    """Exact-ZOH mode response on a UNIFORM time grid, vectorised.

    Same recurrence as :func:`~imas_ambix.latent.temporal_operator.
    integrate_eddy_ode` (``a_t = e^{−Δt/τ} a_{t−1} − (τ/Δt)(1 − e^{−Δt/τ})
    ΔΨ_t``, ``a_0 = 0``) — a linear constant-coefficient recurrence per mode,
    evaluated with :func:`scipy.signal.lfilter` so the per-sample Python loop
    disappears (the calibration objective integrates ~10⁵ samples × ~150
    modes per evaluation).
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
    """Solve the generalised eigenproblem for one candidate resistance model."""
    from scipy.linalg import eigh  # noqa: PLC0415

    r_diag = system.r_diag * np.asarray(multipliers, dtype=np.float64)
    w, v = eigh(np.diag(r_diag), system.lmat)
    tau = 1.0 / np.clip(w, 1e-12, None)
    case_rows = np.array(
        [system.case_channel_row[ch] for ch in sorted(system.case_channel_row)],
        dtype=np.int64,
    )
    return ModeMaps(
        tau=tau,
        v=v,
        a_sens_modes=system.a_circ @ v,
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
    mean removed over the interval (an offset-nuisance intercept, as the
    static vacuum audit fits), whitened by the pooled channel scale.
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


__all__ = [
    "LADDER_LEVELS",
    "MULTIPLIER_BOUNDS",
    "ModeMaps",
    "ResistanceCalibration",
    "VacuumShotData",
    "campaign_mode_maps",
    "case_label_by_circuit",
    "load_calibration",
    "pooled_loss",
    "resistance_group_labels",
    "save_calibration",
    "shot_loss_terms",
    "zoh_mode_response",
]
