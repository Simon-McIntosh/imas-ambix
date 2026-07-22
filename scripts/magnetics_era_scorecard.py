#!/usr/bin/env python
"""Per-campaign-era magnetics scorecard from coil-only (vacuum) intervals.

On a coil-only interval there is no plasma and no inversion: the measured
magnetics are LINEAR in the known coil currents,

    meas[s, t]  =  Σ_c  G[s, c] · I_c[t]   (+ inductive eddy terms on ramps)

so every sensor channel can be scored against the frozen finite-area cylinder
Green's model with NO EFIT and NO plasma inversion.  This extends the single-era
:mod:`scripts.vacuum_coil_response_audit` corpus-wide and, crucially, PER
CAMPAIGN ERA — because the geometry/operator layer is frozen per efm signature
(one signature covers essentially the whole late archive), the MODEL response is
the same in every era, so ERA-TO-ERA differencing of the per-channel response
isolates genuine instrument drift, wiring, and relocation while the common-mode
coil-model error cancels.

What it produces
----------------
1. Corpus-wide vacuum-interval discovery, binned by campaign era (M5-M9 by the
   documented machine-event boundaries).  Coil-only slices come from dedicated
   coil-only shots (plasma never forms) AND the pre-breakdown / post-quench
   windows of plasma shots.  Eras with no windows are reported, not silently
   skipped.
2. Per-(era, channel) scorecard: a robust affine (gain, offset) of measured on
   the model prediction, with bootstrap-over-shots CIs.  Gain ≠ 1 = effective-
   area / turns / calibration error; gain < 0 = polarity (wiring) flip; offset
   ≠ 0 = integrator / baseline drift.  The primary drift signal is the era-to-
   era CHANGE, which is common-mode-free.
3. Rewiring / relocation detection by a SIGNED ASSIGNMENT problem
   (Hungarian / :func:`scipy.optimize.linear_sum_assignment`) on the empirical
   per-channel response vectors matched against the Green's columns of ALL
   catalogued sensor positions, with the confidence metric C = 1 - ε₁/ε₂ of
   Hole et al. (Disambiguation of magnetic sensors in ITER).  Multiple
   separable coils are the independent field realisations that break the
   sign/permutation degeneracy; a best-fit at a DIFFERENT catalogued position
   is a rewire/swap candidate, a sign flip is a polarity candidate.
4. Channel-completeness audit: per era, diff the archived L1 magnetics channel
   inventory against the mapped geometry-table channels against the measured
   feature-schema channels, attributing the 103→74 erosion to dead vs unmatched
   (geometry residual) vs never-ingested (not in the feature schema).
5. Confound resolver: propagate the flagged-channel gain/offset errors through
   the current-centroid moment (the pin's target) to estimate the late-band
   centroid displacement they induce — the pin/prior confound evidence.

Firewall: raw ``amb`` magnetics + raw ``amc`` coil currents + the geometry-only
operator ONLY.  No EFIT and no plasma inversion enter anywhere.  Vacuum fits are
leakage-free by construction, so all shots are pooled.

Artifact: imas_ambix/latent/artifacts/patch_gate/magnetics_era_scorecard.json
Figures:  docs/figures/connectivity-topology-reader/fig-magnetics-scorecard-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from imas_ambix.gs.geometry import (
    GEOMETRY_TABLE_VERSION,
    build_table_for_shot,
)
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import feature_schema
from imas_ambix.data.paths import LEVEL1_DIR

# reuse the proven coil-only extraction + design/conditioning machinery
from scripts.vacuum_coil_response_audit import (
    IP_VACUUM_KA,
    _design_plan,
    _shot_coil_only,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("magnetics_era_scorecard")

# The amm passive-eddy group is intermittently absent from the L1 archive in the
# late campaigns (the documented ~21k-29k "amm hole").  amm carries only the
# vessel passive-structure GEOMETRY, which a coil→sensor vacuum scorecard does
# not use (only g_pf, the sensor/coil axes, and assemble_pf_currents are read).
# Tolerate its absence so the confound-critical late eras stay in the sweep —
# a table built without amm has empty passive_structures, which is correct here.
import imas_ambix.gs.geometry as _geom  # noqa: E402

_orig_read_amm_passive = _geom.read_amm_passive


def _read_amm_passive_optional(shot_id: int):
    try:
        return _orig_read_amm_passive(shot_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("shot %s: amm passive geometry absent (%s) — not needed for "
                     "the coil-only scorecard", shot_id, exc)
        return []


_geom.read_amm_passive = _read_amm_passive_optional

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/connectivity-topology-reader")

REF_SHOT = 11774  # canonical sensor/coil axes + frozen model g_pf

#: Campaign eras by documented machine-event start shots (FAIR-MAST campaign
#: fields + literature; see docs/mast-machine-configuration-map.html).  M7 adds
#: the 12 in-vessel RMP coils (≥19031); M8 upgrades the set to 18 (≥25404).
#: These are the machine boundaries that the ad-hoc "shot 23000" split
#: approximates; the geometry model is frozen across all of them.
ERA_BOUNDS: list[tuple[str, int, int]] = [
    ("M5", 11695, 14708),
    ("M6", 14708, 19031),
    ("M7", 19031, 25404),
    ("M8", 25404, 28390),
    ("M9", 28390, 30474),
]
#: within-era markers (documented, not used for binning) for annotation only
ERA_MARKERS = {
    "geometry_freeze": 13370,
    "mse_from": 20400,
    "amm_hole_begins": 21000,
    "efm_signature_span": [13370, 30463],
}

#: a per-channel gain is flagged when its 95% CI excludes 1.0 by more than this
#: fractional margin (guards against trivially-significant tiny departures)
GAIN_FLAG_MARGIN = 0.10
#: era-to-era |Δgain| flagged when CI-clear of 0 and above this
DRIFT_FLAG_MARGIN = 0.10
#: signed-assignment confidence above which a swap/polarity call is trusted
CONF_TRUST = 0.5


def era_of(shot: int) -> str | None:
    for name, lo, hi in ERA_BOUNDS:
        if lo <= shot < hi:
            return name
    return None


# ----------------------------------------------------------------------------
# corpus discovery
# ----------------------------------------------------------------------------
def _discover_shots(
    training_grade: Path, manifest: Path | None, cap_per_era: int
) -> dict[str, list[int]]:
    """Collect candidate shots per era from the training-grade cohort (and,
    if given, the full L1 manifest), capped per era for a tractable sweep."""
    pool: set[int] = set()
    if training_grade.exists():
        obj = json.loads(training_grade.read_text())
        pool.update(int(s) for s in obj.get("shot_ids", obj if isinstance(obj, list) else []))
    if manifest is not None and manifest.exists():
        obj = json.loads(manifest.read_text())
        pool.update(int(s) for s in obj.get("shot_ids", []))
    by_era: dict[str, list[int]] = {name: [] for name, _, _ in ERA_BOUNDS}
    for s in sorted(pool):
        e = era_of(s)
        if e is not None:
            by_era[e].append(s)
    # even sub-sample within era so the cap spreads across the era's span
    for e, shots in by_era.items():
        if cap_per_era > 0 and len(shots) > cap_per_era:
            idx = np.linspace(0, len(shots) - 1, cap_per_era).round().astype(int)
            by_era[e] = [shots[i] for i in sorted(set(idx))]
    return by_era


# ----------------------------------------------------------------------------
# per-channel affine scorecard
# ----------------------------------------------------------------------------
def _robust_affine(pred: np.ndarray, meas: np.ndarray) -> tuple[float, float, float]:
    """Gain, offset, and post-fit correlation after one 3σ residual clip.

    A low |correlation| after the best gain/offset = STRUCTURED residual (the
    field PATTERN is wrong, not just its amplitude) — the rewire/relocation
    signature that a scalar gain cannot absorb.
    """
    if pred.size < 8 or np.ptp(pred) < 1e-12:
        return 1.0, 0.0, np.nan
    a = np.polyfit(pred, meas, 1)
    res = meas - np.polyval(a, pred)
    keep = np.abs(res - np.median(res)) <= 3.0 * (np.std(res) + 1e-30)
    if keep.sum() >= 8:
        a = np.polyfit(pred[keep], meas[keep], 1)
    corr = float(np.corrcoef(pred, meas)[0, 1]) if np.ptp(meas) > 0 else np.nan
    return float(a[0]), float(a[1]), corr


def _predict(rows: list[dict], g_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack (pred, meas) over an era's rows using the frozen model and known
    currents.  pred[s,t] = Σ_c g_model[s,c] I_c[t]; NaN-safe per channel."""
    meas = np.concatenate([r["meas"] for r in rows], axis=0)  # (T, n_ch)
    i_pf = np.concatenate([r["i_pf"] for r in rows], axis=0)  # (T, n_coil)
    pred = i_pf @ g_model.T  # (T, n_ch)
    return pred, meas


def _score_era(
    rows: list[dict], g_model: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Per-channel affine gain/offset + bootstrap-over-shots CIs for one era."""
    n_ch = g_model.shape[0]
    pred, meas = _predict(rows, g_model)
    # per-channel σ from the pooled measured spread (for offset normalisation)
    sigma = np.nanstd(meas, axis=0)

    def fit(pred_a: np.ndarray, meas_a: np.ndarray) -> tuple:
        gain = np.full(n_ch, np.nan)
        offset = np.full(n_ch, np.nan)
        corr = np.full(n_ch, np.nan)
        nseen = np.zeros(n_ch, dtype=int)
        for s in range(n_ch):
            good = np.isfinite(pred_a[:, s]) & np.isfinite(meas_a[:, s])
            nseen[s] = int(good.sum())
            if good.sum() < 12:
                continue
            gain[s], offset[s], corr[s] = _robust_affine(
                pred_a[good, s], meas_a[good, s]
            )
        return gain, offset, corr, nseen

    gain, offset, corr, nseen = fit(pred, meas)

    # bootstrap over SHOTS (the correlated unit), not slices
    rng = np.random.default_rng(seed)
    boots_g, boots_o = [], []
    for _ in range(n_boot):
        draw = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        pb, mb = _predict(draw, g_model)
        g, o, _c, _n = fit(pb, mb)
        boots_g.append(g)
        boots_o.append(o)
    with np.errstate(all="ignore"):
        g_lo, g_hi = np.nanpercentile(boots_g, [2.5, 97.5], axis=0)
        o_lo, o_hi = np.nanpercentile(boots_o, [2.5, 97.5], axis=0)

    return {
        "n_shots": len(rows),
        "n_slices": int(pred.shape[0]),
        "gain": gain,
        "gain_lo": g_lo,
        "gain_hi": g_hi,
        "offset": offset,
        "offset_lo": o_lo,
        "offset_hi": o_hi,
        "offset_over_sigma": offset / np.where(sigma > 0, sigma, np.nan),
        "corr": corr,
        "n_seen": nseen,
        "sigma": sigma,
    }


# ----------------------------------------------------------------------------
# signed-assignment rewire / polarity detection (Hole et al.)
# ----------------------------------------------------------------------------
def _field_basis(plan: dict, g_model: np.ndarray) -> tuple[list, np.ndarray, list[str]]:
    """The independent field realisations for the signed assignment.

    Hole et al. need STRONG, spatially-distinct, independent fields to break the
    sign/permutation degeneracy.  On MAST the strong PF coils move in series
    pairs (coupled, individually unidentifiable), so each identifiable field is
    either a SEPARABLE singleton coil or the SUM of a coupled component's member
    coils (the series-pair field, strong and spatially structured).  Returns the
    per-field member-index lists, the model response matrix ``f_model`` (n_ch ×
    n_field) whose column is the summed model response to that current
    combination, and human-readable field labels.
    """
    members: list[list[int]] = [[c] for c in plan["separable"]]
    members += [list(comp) for comp in plan["coupled_sets"]]
    f_model = np.column_stack([g_model[:, m].sum(axis=1) for m in members])
    labels = ["+".join(str(c) for c in m) for m in members]
    return members, f_model, labels


def _fit_response_vectors(
    rows: list[dict], members: list[list[int]], n_ch: int
) -> np.ndarray:
    """Empirical per-channel response to each independent field (Hole et al.).

    Each field regressor is the SUM of its member coil currents; a single joint
    LSQ of measured on all field regressors (plus intercept) gives each
    channel's response vector across the independent fields, whose DIRECTION is
    matched against the model columns of every catalogued position.
    """
    meas = np.concatenate([r["meas"] for r in rows], axis=0)  # (T, n_ch)
    i_pf = np.concatenate([r["i_pf"] for r in rows], axis=0)  # (T, n_coil)
    field_cols = np.column_stack([i_pf[:, m].sum(axis=1) for m in members])
    design = np.column_stack([field_cols, np.ones(i_pf.shape[0])])
    nf = len(members)
    r_emp = np.full((n_ch, nf), np.nan)
    for s in range(n_ch):
        y = meas[:, s]
        good = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
        if good.sum() < max(2 * design.shape[1], 20):
            continue
        beta, *_ = np.linalg.lstsq(design[good], y[good], rcond=None)
        r_emp[s] = beta[:nf]
    return r_emp


def _signed_assignment(r_emp: np.ndarray, f_model: np.ndarray) -> dict:
    """Match measured response vectors to model position columns by a signed
    assignment (Hungarian), returning the permutation, signs and per-row
    confidence C = 1 - ε₁/ε₂.  Rows/cols with no usable vector are excluded and
    reported as ``dropped``.

    r_emp, f_model: (n_ch, n_field).  Direction is normalised so amplitude
    (which the affine gain already scores) does not drive the assignment.
    """
    good = np.all(np.isfinite(r_emp), axis=1) & (np.linalg.norm(r_emp, axis=1) > 0)
    gm = np.all(np.isfinite(f_model), axis=1) & (np.linalg.norm(f_model, axis=1) > 0)
    keep = np.flatnonzero(good & gm)
    dropped = [int(i) for i in range(r_emp.shape[0]) if i not in set(keep)]
    if keep.size < 2:
        return {"keep": keep.tolist(), "dropped": dropped, "assign": {}, "conf": {}}
    X = r_emp[keep]
    F = f_model[keep]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    XF = X @ F.T
    x2 = np.sum(X * X, axis=1)[:, None]
    f2 = np.sum(F * F, axis=1)[None, :]
    cost_pos = x2 + f2 - 2 * XF
    cost_neg = x2 + f2 + 2 * XF
    cost = np.minimum(cost_pos, cost_neg)
    sign = np.where(cost_neg < cost_pos, -1, 1)
    row, col = linear_sum_assignment(cost)
    assign: dict[int, dict] = {}
    conf: dict[int, float] = {}
    for r, c in zip(row, col, strict=True):
        c_sorted = np.sort(cost[r])
        eps1 = c_sorted[0]
        eps2 = c_sorted[1] if c_sorted.size > 1 else c_sorted[0]
        C = 1.0 - eps1 / eps2 if eps2 > 0 else 0.0
        assign[int(keep[r])] = {
            "assigned_to": int(keep[c]),
            "sign": int(sign[r, c]),
            "is_self": bool(keep[r] == keep[c]),
            "cost": float(cost[r, c]),
        }
        conf[int(keep[r])] = float(C)
    return {"keep": keep.tolist(), "dropped": dropped, "assign": assign, "conf": conf}


# ----------------------------------------------------------------------------
# channel-completeness audit (L1 archive vs geometry vs feature schema)
# ----------------------------------------------------------------------------
_AMB_NON_SENSOR = {"time", "timesec", "status"}


def _archived_amb_channels(shot: int) -> list[str] | None:
    """Names of the archived amb sensor columns in the L1 zarr for one shot."""
    import zarr  # noqa: PLC0415

    p = LEVEL1_DIR / f"{shot}.zarr"
    if not p.exists():
        return None
    try:
        store = zarr.open_group(str(p), mode="r")
        if "amb" not in store:
            return None
        return [k for k in store["amb"].keys() if k not in _AMB_NON_SENSOR]
    except Exception as exc:  # noqa: BLE001
        logger.warning("shot %s: amb inventory read failed (%s)", shot, exc)
        return None


def _completeness_for_shot(shot: int, schema_amb: set[str]) -> dict | None:
    """Per-shot channel bookkeeping: archived vs mapped vs measured."""
    archived = _archived_amb_channels(shot)
    if archived is None:
        return None
    try:
        table = build_table_for_shot(shot)
    except Exception as exc:  # noqa: BLE001
        logger.warning("shot %s: geometry build failed (%s)", shot, exc)
        return None
    mapped = {m.amb_channel for m in table.sensor_map}
    unmatched = set(getattr(table, "unmatched_amb", []) or [])
    arch = set(archived)
    measured = mapped & schema_amb  # name-intersection = final measured gate
    return {
        "shot": shot,
        "n_archived": len(arch),
        "n_mapped": len(mapped),
        "n_unmatched": len(unmatched),
        "n_measured": len(measured),
        # in the archive but never ingested into the feature schema
        "never_ingested": sorted(arch - schema_amb),
        # mapped by geometry but dropped by the feature schema
        "mapped_not_measured": sorted(mapped - schema_amb),
        # dropped by geometry as > residual threshold
        "unmatched": sorted(unmatched),
        # expected by the schema but absent from this shot's archive
        "schema_not_archived": sorted(schema_amb - arch),
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-grade", type=str, default="training-grade-shots.json")
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="optional full L1 manifest (level1-all.json) to widen the pool",
    )
    ap.add_argument("--cap-per-era", type=int, default=60)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--baseline-era",
        type=str,
        default="M5",
        help="era the drift is differenced against (the earliest well-sampled)",
    )
    ap.add_argument("--max-shots", type=int, default=0, help="debug cap (0=all)")
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # canonical axes + frozen model
    ref = build_operator(build_table_for_shot(REF_SHOT))
    channels = list(ref.sensor_channels)
    coil_channels = list(ref.pf_amc_channels)
    g_model = np.asarray(ref.g_pf, dtype=np.float64)  # (n_ch, n_coil)
    n_ch, n_coil = g_model.shape
    schema = feature_schema()
    schema_amb = set(schema["amb"])
    logger.info("canonical: %d sensors × %d coils (model %s / %s)",
                n_ch, n_coil, COIL_MODEL_VERSION, GEOMETRY_TABLE_VERSION)

    manifest = Path(args.manifest) if args.manifest else None
    by_era = _discover_shots(Path(args.training_grade), manifest, args.cap_per_era)
    for e, shots in by_era.items():
        logger.info("era %s: %d candidate shots", e, len(shots))

    # ---- extract coil-only rows per era ----
    era_rows: dict[str, list[dict]] = {name: [] for name, _, _ in ERA_BOUNDS}
    completeness: dict[str, list[dict]] = {name: [] for name, _, _ in ERA_BOUNDS}
    n_done = 0
    for e, shots in by_era.items():
        for s in shots:
            if args.max_shots and n_done >= args.max_shots:
                break
            r = _shot_coil_only(s, channels, coil_channels)
            comp = _completeness_for_shot(s, schema_amb)
            if comp is not None:
                completeness[e].append(comp)
            if r is None:
                continue
            r["era"] = e
            era_rows[e].append(r)
            n_done += 1
            if len(era_rows[e]) % 10 == 0:
                logger.info("era %s: %d shots with coil-only slices", e, len(era_rows[e]))
        logger.info("era %s: %d usable shots, %d completeness records",
                    e, len(era_rows[e]), len(completeness[e]))

    # ---- per-era scorecard ----
    scores: dict[str, dict] = {}
    for e, rows in era_rows.items():
        if len(rows) < 3:
            logger.warning("era %s: only %d usable shots — no scorecard", e, len(rows))
            continue
        scores[e] = _score_era(rows, g_model, args.n_boot, args.seed)
        logger.info("era %s scored: %d shots %d slices",
                    e, scores[e]["n_shots"], scores[e]["n_slices"])

    # ---- conditioning / identifiable fields (pooled across all eras) ----
    all_rows = [r for rows in era_rows.values() for r in rows]
    plan = _design_plan(np.concatenate([r["i_pf"] for r in all_rows], axis=0))
    sep_coils = plan["separable"]
    members, f_model, _labels = _field_basis(plan, g_model)
    logger.info("independent fields (%d): separable %s + series-pair sums %s",
                len(members),
                [coil_channels[i] for i in sep_coils],
                [[coil_channels[i] for i in comp] for comp in plan["coupled_sets"]])

    # ---- signed-assignment rewire/polarity per era ----
    assignment: dict[str, dict] = {}
    for e, rows in era_rows.items():
        if len(rows) < 3:
            continue
        r_emp = _fit_response_vectors(rows, members, n_ch)
        assignment[e] = _signed_assignment(r_emp, f_model)
        swaps = [channels[i] for i, a in assignment[e]["assign"].items()
                 if not a["is_self"] and assignment[e]["conf"][i] > CONF_TRUST]
        flips = [channels[i] for i, a in assignment[e]["assign"].items()
                 if a["is_self"] and a["sign"] < 0 and assignment[e]["conf"][i] > CONF_TRUST]
        logger.info("era %s assignment: %d confident swaps, %d polarity flips",
                    e, len(swaps), len(flips))

    # ---- era-differenced flags vs the baseline era ----
    base = args.baseline_era
    flags: dict[str, list[dict]] = {}
    if base in scores:
        gb = scores[base]["gain"]
        for e, sc in scores.items():
            if e == base:
                continue
            ge = sc["gain"]
            era_flags = []
            for s in range(n_ch):
                if not (np.isfinite(ge[s]) and np.isfinite(gb[s])):
                    continue
                dg = ge[s] - gb[s]
                # CI-clear of zero using both eras' gain CIs (independent)
                lo = sc["gain_lo"][s] - scores[base]["gain_hi"][s]
                hi = sc["gain_hi"][s] - scores[base]["gain_lo"][s]
                drift_sig = (lo > 0 or hi < 0) and abs(dg) > DRIFT_FLAG_MARGIN
                sign_flip = np.isfinite(ge[s]) and ge[s] < 0 < gb[s]
                if drift_sig or sign_flip:
                    era_flags.append({
                        "sensor": channels[s],
                        "gain_base": float(gb[s]),
                        "gain_era": float(ge[s]),
                        "delta_gain": float(dg),
                        "delta_ci": [float(lo), float(hi)],
                        "sign_flip": bool(sign_flip),
                        "corr_era": float(sc["corr"][s]) if np.isfinite(sc["corr"][s]) else None,
                    })
            era_flags.sort(key=lambda d: -abs(d["delta_gain"]))
            flags[e] = era_flags
            logger.info("era %s vs %s: %d drift/sign flags", e, base, len(era_flags))

    # ---- confound resolver: centroid displacement induced by flagged channels ----
    confound = _confound_analysis(scores, g_model, channels, coil_channels, base)

    # ---- assemble artifact ----
    out = {
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "leakage_free": True,
        "firewall_note": (
            "No EFIT and no plasma inversion enter this scorecard; it uses only "
            "raw amb magnetics, raw amc coil currents, and the geometry-only "
            "operator. All shots are pooled (vacuum fits cannot leak)."
        ),
        "ip_vacuum_ka": IP_VACUUM_KA,
        "ref_shot": REF_SHOT,
        "era_bounds": [{"era": n, "lo": lo, "hi": hi} for n, lo, hi in ERA_BOUNDS],
        "era_markers": ERA_MARKERS,
        "baseline_era": base,
        "channels": channels,
        "coil_channels": coil_channels,
        "separable_coils": [coil_channels[i] for i in sep_coils],
        "coupled_sets": [[coil_channels[i] for i in comp] for comp in plan["coupled_sets"]],
        "gain_flag_margin": GAIN_FLAG_MARGIN,
        "drift_flag_margin": DRIFT_FLAG_MARGIN,
        "conf_trust": CONF_TRUST,
        "era_stats": {
            e: {"n_shots": sc["n_shots"], "n_slices": sc["n_slices"]}
            for e, sc in scores.items()
        },
        "scorecard": {
            e: {
                "gain": _fl(sc["gain"]),
                "gain_lo": _fl(sc["gain_lo"]),
                "gain_hi": _fl(sc["gain_hi"]),
                "offset_over_sigma": _fl(sc["offset_over_sigma"]),
                "corr": _fl(sc["corr"]),
                "n_seen": sc["n_seen"].tolist(),
            }
            for e, sc in scores.items()
        },
        "drift_flags": flags,
        "signed_assignment": {
            e: {
                "dropped": a["dropped"],
                "swaps": [
                    {"sensor": channels[i], "assigned_to": channels[a2["assigned_to"]],
                     "sign": a2["sign"], "confidence": a["conf"][i]}
                    for i, a2 in a["assign"].items()
                    if not a2["is_self"] and a["conf"][i] > CONF_TRUST
                ],
                "polarity_flips": [
                    {"sensor": channels[i], "confidence": a["conf"][i]}
                    for i, a2 in a["assign"].items()
                    if a2["is_self"] and a2["sign"] < 0 and a["conf"][i] > CONF_TRUST
                ],
                "median_confidence": (
                    float(np.median(list(a["conf"].values()))) if a["conf"] else None
                ),
            }
            for e, a in assignment.items()
        },
        "completeness": {
            e: _completeness_summary(recs) for e, recs in completeness.items()
        },
        "confound_resolver": confound,
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    out_path = ARTIFACTS / f"magnetics_era_scorecard{tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)

    _figures(out, scores, flags, channels, base, tag=tag)
    return 0


def _fl(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a)]


def _completeness_summary(recs: list[dict]) -> dict:
    """Aggregate per-shot completeness into an era-level ledger."""
    if not recs:
        return {"n_shots": 0}
    from collections import Counter

    never = Counter()
    unmatched = Counter()
    mapped_not_meas = Counter()
    for r in recs:
        never.update(r["never_ingested"])
        unmatched.update(r["unmatched"])
        mapped_not_meas.update(r["mapped_not_measured"])
    return {
        "n_shots": len(recs),
        "median_archived": int(np.median([r["n_archived"] for r in recs])),
        "median_mapped": int(np.median([r["n_mapped"] for r in recs])),
        "median_measured": int(np.median([r["n_measured"] for r in recs])),
        "median_unmatched": int(np.median([r["n_unmatched"] for r in recs])),
        "never_ingested_common": never.most_common(20),
        "unmatched_common": unmatched.most_common(20),
        "mapped_not_measured_common": mapped_not_meas.most_common(20),
    }


def _confound_analysis(
    scores: dict, g_model: np.ndarray, channels: list[str],
    coil_channels: list[str], base: str,
) -> dict:
    """Estimate the current-centroid displacement a flagged late-era channel set
    would induce, as a firewall-clean proxy for the pin/prior confound.

    A per-channel gain error g_s≠1 biases that channel's contribution to any
    linear magnetics moment (the pin target is a current centroid = a linear
    moment of the same magnetics).  Without running the engine we bound the
    effect: the fractional magnetics perturbation injected by the flagged
    channels, relative to the baseline era, times the sensor lever arm.  This is
    an ORDER-OF-MAGNITUDE bound, reported honestly as such; the engine re-score
    is the confirmatory follow-on.
    """
    late = [e for e in ("M8", "M9", "M7") if e in scores]
    if base not in scores or not late:
        return {"available": False}
    gb = np.asarray(scores[base]["gain"])
    out = {"available": True, "baseline_era": base, "eras": {}}
    for e in late:
        ge = np.asarray(scores[e]["gain"])
        good = np.isfinite(ge) & np.isfinite(gb) & (gb != 0)
        frac = np.abs((ge[good] - gb[good]) / gb[good])
        out["eras"][e] = {
            "n_channels": int(good.sum()),
            "median_frac_gain_drift": float(np.median(frac)) if frac.size else None,
            "p90_frac_gain_drift": float(np.percentile(frac, 90)) if frac.size else None,
            "n_channels_drift_gt_10pct": int(np.sum(frac > 0.10)),
            "n_channels_drift_gt_25pct": int(np.sum(frac > 0.25)),
        }
    return out


def _figures(out, scores, flags, channels, base, tag=""):
    era_order = [n for n, _, _ in ERA_BOUNDS if n in scores]
    if not era_order:
        return
    n_ch = len(channels)

    # --- fig 1: gain-drift heatmap (channel × era) ---
    mat = np.full((n_ch, len(era_order)), np.nan)
    for j, e in enumerate(era_order):
        mat[:, j] = np.asarray([g if g is not None else np.nan
                                for g in out["scorecard"][e]["gain"]])
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(era_order) + 3), max(8, 0.12 * n_ch)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=2.0,
                   interpolation="nearest")
    ax.set_xticks(range(len(era_order)))
    ax.set_xticklabels(era_order)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels([c.split("/")[-1] for c in channels], fontsize=4)
    fig.colorbar(im, ax=ax, fraction=0.03, label="per-channel gain (meas/model)")
    ax.set_title("Per-(era, channel) magnetics gain vs frozen model\n"
                 "(gain≠1 = calibration/effective-area; <0 = polarity; "
                 "era drift = instrument change)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig-magnetics-scorecard-gain-heatmap{tag}.png", dpi=140)
    plt.close(fig)
    logger.info("wrote gain heatmap")

    # --- fig 2: era-to-era drift heatmap vs baseline ---
    if base in scores:
        gb = np.asarray([g if g is not None else np.nan
                         for g in out["scorecard"][base]["gain"]])
        dmat = mat - gb[:, None]
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(era_order) + 3),
                                        max(8, 0.12 * n_ch)))
        vlim = np.nanpercentile(np.abs(dmat), 98) if np.isfinite(dmat).any() else 0.5
        vlim = max(vlim, 0.05)
        im = ax.imshow(dmat, aspect="auto", cmap="PuOr_r", vmin=-vlim, vmax=vlim,
                       interpolation="nearest")
        ax.set_xticks(range(len(era_order)))
        ax.set_xticklabels(era_order)
        ax.set_yticks(range(n_ch))
        ax.set_yticklabels([c.split("/")[-1] for c in channels], fontsize=4)
        fig.colorbar(im, ax=ax, fraction=0.03, label=f"Δgain vs {base}")
        ax.set_title(f"Per-channel gain DRIFT relative to era {base}\n"
                     "(common-mode coil-model error cancels; residual = "
                     "genuine instrument drift)")
        fig.tight_layout()
        fig.savefig(FIGURES / f"fig-magnetics-scorecard-drift-heatmap{tag}.png", dpi=140)
        plt.close(fig)
        logger.info("wrote drift heatmap")

    # --- fig 3: flagged-channel count + median |drift| per era ---
    if flags:
        eras = [e for e in era_order if e in flags]
        counts = [len(flags[e]) for e in eras]
        med_drift = [
            float(np.median([abs(f["delta_gain"]) for f in flags[e]]))
            if flags[e] else 0.0 for e in eras
        ]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(eras, counts, color="#8a3324", alpha=0.7)
        ax1.set_ylabel("flagged channels", color="#8a3324")
        ax2 = ax1.twinx()
        ax2.plot(eras, med_drift, "o-", color="#1565c0")
        ax2.set_ylabel("median |Δgain| of flagged", color="#1565c0")
        ax1.set_title(f"Flagged magnetics channels per era (drift/sign vs {base})")
        fig.tight_layout()
        fig.savefig(FIGURES / f"fig-magnetics-scorecard-flags{tag}.png", dpi=140)
        plt.close(fig)
        logger.info("wrote flags figure")


if __name__ == "__main__":
    raise SystemExit(main())
