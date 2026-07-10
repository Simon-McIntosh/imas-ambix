#!/usr/bin/env python
"""Empirical coil-response (Green's-matrix) audit from coil-only intervals.

On a coil-only (vacuum) interval there is no plasma and no inversion: the
measured magnetics are LINEAR in the known coil currents,

    meas[s, t]  =  Σ_c  G[s, c] · I_c[t]   (+ inductive eddy terms on ramps)

so with ~81 sensor channels, 21 coil columns and thousands of coil-only slices
pooled across the full 48-shot fleet, the empirical Green's matrix ``G_emp`` is
heavily over-determined.  Fitting it channel-by-channel and comparing against
the geometry forward model ``G_model = fwd.g_pf`` turns a pure instrument
characterisation into a direct read on the coil model:

* a per-coil MULTIPLICATIVE error ``k_c`` (fit ``G_emp[:,c] ≈ k_c·G_model[:,c]``
  over sensors) that departs from 1 beyond its CI = a turns / ``xmult`` error on
  coil ``c``;
* a per-(coil, sensor) residual after the best scale = a GEOMETRY-error
  signature (the field pattern, not just its amplitude, is wrong);
* coil currents that never separate in the data (pairwise |corr| > 0.98) are
  UNSEPARABLE — their individual columns are not identifiable here and are
  reported as coupled sets (candidate physical couplings: a coil and its case
  circuit, series-connected pairs).

This is leakage-free by construction: it uses NO EFIT and NO plasma inversion,
only the raw ``amb`` magnetics, the raw ``amc`` coil currents, and the
geometry-only operator.  Because there is no plasma inversion to contaminate,
ALL 48 shots (train + held-out) are pooled.

Eddy handling — the pre-breakdown solenoid ramp induces vessel eddy currents
that add an inductive term ``∝ dI/dt`` to ``meas − G·I``.  Two fits are run and
compared: (a) a QUASI-STATIC-only fit (slices with |dI/dt| below a data-driven
threshold), and (b) a FULL-POOL fit augmented with one smoothed per-coil
``dI/dt`` regressor whose coefficient absorbs the inductive response.  The
static ``G_emp`` must agree between (a) and (b) within CIs; disagreement flags
residual eddy leakage in the quasi-static estimate.

Artifact: imas_ambix/latent/artifacts/patch_gate/vacuum_coil_response_audit.json
Figures:  docs/figures/force-balance-spine/fig-coil-response-*.png
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

from imas_ambix.gs.geometry import GEOMETRY_TABLE_VERSION, build_table_for_shot
from imas_ambix.gs.operator import COIL_MODEL_VERSION, build_operator
from imas_ambix.latent.data import (
    align_sensor_columns,
    anchored_columns,
    feature_schema,
    load_shot_slices_raw,
    read_split_shot_lists,
    robust_channel_scale,
    schema_group_offsets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vacuum_coil_response_audit")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/force-balance-spine")

#: a slice counts as coil-only when the loader's plasma-on flag is False AND
#: |Ip| is below this (kA) — a guard against ambiguous breakdown/halo edges.
IP_VACUUM_KA = 20.0

#: pairwise |correlation| above which two coil-current columns are declared
#: UNSEPARABLE in the pooled design (their individual G columns are not
#: identifiable — only their combined field is).
CORR_COUPLE = 0.98

#: a coil column is "unmeasurable" (dropped from the fit) when its pooled
#: current dynamic range ptp/σ_I is below this — it never moves relative to its
#: own noise, so it carries no leverage.
RANGE_FLOOR = 0.5


def _shot_coil_only(
    shot: int, channels: list[str], coil_channels: list[str]
) -> dict | None:
    """Assemble one shot's coil-only slices on canonical sensor/coil axes.

    Pure forward physics: reads level-1 raw, aligns the amb magnetics to the
    canonical sensor rows BY NAME, assembles the per-coil ``i_pf`` exactly as
    :func:`load_shot_windows` does, and maps it onto the canonical coil axis BY
    NAME.  Returns per-slice measured magnetics, per-coil currents, per-coil and
    per-slice ``dI/dt``, and the plasma-on σ — or ``None`` if the shot has no
    usable coil-only slices.
    """
    schema = feature_schema()
    try:
        table = build_table_for_shot(int(shot))
        fwd = build_operator(table)
    except Exception as exc:  # noqa: BLE001
        logger.warning("shot %s: operator build failed (%s)", shot, exc)
        return None
    loaded = load_shot_slices_raw(int(shot), schema)
    if loaded is None:
        return None
    x, times, plasma_on = loaded
    x = np.asarray(x, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if not np.any(plasma_on):
        return None

    offsets = schema_group_offsets(schema)
    amb_names = schema["amb"]
    amc_names = schema["amc"]
    op_rows, x_cols = align_sensor_columns(fwd.sensor_channels, amb_names)

    n_sensor = len(fwd.sensor_channels)
    t_n = x.shape[0]
    raw_mag = np.full((t_n, n_sensor), np.nan)
    if op_rows.size:
        raw_mag[:, op_rows] = x[:, offsets["amb"] + x_cols]

    # plasma-on σ on the operator sensor rows → robust, frozen-comparable units
    sigma_op = robust_channel_scale(
        np.nanstd(raw_mag[plasma_on], axis=0), fwd.sensor_channels
    )

    # i_pf per slice on the operator's coil axis, assembled as load_shot_windows
    n_coil_op = len(fwd.pf_amc_channels)
    i_pf = np.zeros((t_n, n_coil_op))
    amc_block = x[:, offsets["amc"] : offsets["amc"] + len(amc_names)]
    for t in range(t_n):
        amc_values = {
            ch: float(amc_block[t, j])
            for j, ch in enumerate(amc_names)
            if np.isfinite(amc_block[t, j])
        }
        i_pf[t] = fwd.assemble_pf_currents(amc_values)

    ip_col, _ = anchored_columns(schema)
    ip_ka = np.abs(x[:, ip_col])
    coil_only = (~plasma_on) & np.isfinite(ip_ka) & (ip_ka < IP_VACUUM_KA)
    if not coil_only.any():
        return None

    # per-coil dI/dt (the eddy proxy / inductive regressor)
    didt = np.gradient(i_pf, times, axis=0) if t_n >= 2 else np.zeros_like(i_pf)

    sel = np.flatnonzero(coil_only)

    # map operator sensor rows → canonical sensor axis (NaN where shot lacks it)
    row_of = {ch: r for r, ch in enumerate(fwd.sensor_channels)}
    can_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = can_rows >= 0
    safe_rows = np.clip(can_rows, 0, None)
    meas = np.where(present, raw_mag[sel][:, safe_rows], np.nan)
    sigma_can = np.where(present, sigma_op[safe_rows], np.nan)

    # map operator coil columns → canonical coil axis (0 where shot lacks it)
    col_of = {ch: c for c, ch in enumerate(fwd.pf_amc_channels)}
    can_cols = np.array([col_of.get(ch, -1) for ch in coil_channels])
    has_coil = can_cols >= 0
    safe_cols = np.clip(can_cols, 0, None)
    i_can = np.where(has_coil, i_pf[sel][:, safe_cols], 0.0)
    didt_can = np.where(has_coil, didt[sel][:, safe_cols], 0.0)
    didt_slice = np.max(np.abs(didt[sel]), axis=1)

    return {
        "shot": int(shot),
        "meas": meas,  # (n_slice, n_ch)
        "i_pf": i_can,  # (n_slice, n_coil)
        "didt": didt_can,  # (n_slice, n_coil)
        "didt_slice": didt_slice,  # (n_slice,)
        "sigma": sigma_can,  # (n_ch,)
    }


def _coupled_components(abs_corr: np.ndarray, thresh: float) -> list[list[int]]:
    """Connected components of the |corr| > thresh graph over coil columns."""
    n = abs_corr.shape[0]
    adj = (abs_corr > thresh) & ~np.eye(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp = []
        seen[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in np.flatnonzero(adj[j]):
                if not seen[k]:
                    seen[k] = True
                    stack.append(int(k))
        comps.append(sorted(comp))
    return comps


def _design_plan(i_pool: np.ndarray) -> dict:
    """Decide the identifiable regressor layout from the pooled coil currents.

    Returns the separable singleton coils (fit directly), the coupled
    components (each contributes ONE aggregate regressor = the component's
    first principal component, soaking up its shared 1-D variation without
    letting it corrupt the separable coils), and the unmeasurable columns
    (near-zero dynamic range, dropped).
    """
    n_coil = i_pool.shape[1]
    std = np.std(i_pool, axis=0)
    ptp = np.ptp(i_pool, axis=0)
    rng = ptp / np.where(std > 0, std, np.nan)
    measurable = np.isfinite(rng) & (rng >= RANGE_FLOOR) & (std > 0)

    # correlation only over measurable columns; degenerate cols excluded
    abs_corr = np.zeros((n_coil, n_coil))
    meas_idx = np.flatnonzero(measurable)
    if meas_idx.size >= 2:
        c = np.corrcoef(i_pool[:, meas_idx].T)
        abs_corr[np.ix_(meas_idx, meas_idx)] = np.abs(np.nan_to_num(c))

    comps = _coupled_components(abs_corr[np.ix_(meas_idx, meas_idx)], CORR_COUPLE)
    # remap component indices (into meas_idx) back to global coil indices
    comps_global = [[int(meas_idx[j]) for j in comp] for comp in comps]
    separable = [c[0] for c in comps_global if len(c) == 1]
    coupled = [c for c in comps_global if len(c) > 1]
    unmeasurable = [int(i) for i in range(n_coil) if not measurable[i]]
    return {
        "std": std,
        "ptp": ptp,
        "range": rng,
        "abs_corr": abs_corr,
        "separable": separable,
        "coupled_sets": coupled,
        "unmeasurable": unmeasurable,
    }


def _build_regressors(
    i_pf: np.ndarray,
    didt: np.ndarray,
    plan: dict,
    *,
    augment: bool,
) -> tuple[np.ndarray, list[int]]:
    """Assemble the design matrix and the map from its columns to separable coils.

    Columns: one per separable singleton coil (current), one aggregate PC1 per
    coupled component, and — when ``augment`` — one smoothed dI/dt regressor per
    separable coil (inductive-response nuisance).  Returns ``(design, coil_of)``
    where ``coil_of[k]`` is the global coil index for design column ``k`` if it
    is a separable-coil current column, else -1.
    """
    cols: list[np.ndarray] = []
    coil_of: list[int] = []
    for c in plan["separable"]:
        cols.append(i_pf[:, c])
        coil_of.append(c)
    for comp in plan["coupled_sets"]:
        block = i_pf[:, comp]
        mu = block.mean(axis=0)
        sd = block.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        z = (block - mu) / sd
        # PC1 score = shared 1-D variation of the coupled set
        _u, _s, vt = np.linalg.svd(z, full_matrices=False)
        pc1 = z @ vt[0]
        cols.append(pc1)
        coil_of.append(-1)
    if augment:
        for c in plan["separable"]:
            cols.append(didt[:, c])
            coil_of.append(-1)
    cols.append(np.ones(i_pf.shape[0]))  # intercept (residual channel offset)
    coil_of.append(-1)
    design = np.column_stack(cols)
    return design, coil_of


def _fit_g_emp(
    meas: np.ndarray,
    design: np.ndarray,
    coil_of: list[int],
    n_coil: int,
) -> np.ndarray:
    """Per-channel LSQ of measured on the design; return G_emp (n_ch, n_coil).

    Only separable-coil columns populate ``G_emp``; coupled / eddy / intercept
    columns are nuisance regressors whose coefficients are discarded.
    """
    n_ch = meas.shape[1]
    g_emp = np.full((n_ch, n_coil), np.nan)
    sep_cols = [k for k, c in enumerate(coil_of) if c >= 0]
    sep_coils = [coil_of[k] for k in sep_cols]
    for s in range(n_ch):
        y = meas[:, s]
        good = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
        if good.sum() < max(2 * design.shape[1], 20):
            continue
        beta, *_ = np.linalg.lstsq(design[good], y[good], rcond=None)
        for k, c in zip(sep_cols, sep_coils, strict=True):
            g_emp[s, c] = beta[k]
    return g_emp


def _scale_factors(
    g_emp: np.ndarray, g_model: np.ndarray, sigma: np.ndarray, coils: list[int]
) -> dict[int, float]:
    """Per-coil multiplicative scale k_c minimising Σ (1/σ²)(G_emp − k·G_model)²."""
    w = 1.0 / np.where(sigma > 0, sigma, np.nan) ** 2
    out: dict[int, float] = {}
    for c in coils:
        ge = g_emp[:, c]
        gm = g_model[:, c]
        good = np.isfinite(ge) & np.isfinite(gm) & np.isfinite(w)
        denom = np.sum(w[good] * gm[good] ** 2)
        if denom <= 0 or good.sum() < 3:
            continue
        out[c] = float(np.sum(w[good] * ge[good] * gm[good]) / denom)
    return out


def _bootstrap(
    rows: list[dict],
    plan: dict,
    g_model: np.ndarray,
    sigma: np.ndarray,
    n_coil: int,
    *,
    augment: bool,
    threshold: float,
    quasi_static: bool,
    n_boot: int,
    seed: int,
) -> dict:
    """Percentile CIs for G_emp and k_c, resampling SHOTS with replacement."""
    rng = np.random.default_rng(seed)
    coils = plan["separable"]
    n_ch = rows[0]["meas"].shape[1]
    boot_g = {c: [] for c in coils}
    boot_k = {c: [] for c in coils}
    for _ in range(n_boot):
        draw = rng.integers(0, len(rows), len(rows))
        meas_l, i_l, didt_l = [], [], []
        for i in draw:
            r = rows[i]
            keep = (
                r["didt_slice"] <= threshold
                if quasi_static
                else np.ones(r["meas"].shape[0], dtype=bool)
            )
            if not keep.any():
                continue
            meas_l.append(r["meas"][keep])
            i_l.append(r["i_pf"][keep])
            didt_l.append(r["didt"][keep])
        if not meas_l:
            continue
        meas = np.concatenate(meas_l)
        i_pf = np.concatenate(i_l)
        didt = np.concatenate(didt_l)
        design, coil_of = _build_regressors(i_pf, didt, plan, augment=augment)
        g_emp = _fit_g_emp(meas, design, coil_of, n_coil)
        k = _scale_factors(g_emp, g_model, sigma, coils)
        for c in coils:
            boot_g[c].append(g_emp[:, c])
            if c in k:
                boot_k[c].append(k[c])
    g_lo = np.full((n_ch, n_coil), np.nan)
    g_hi = np.full((n_ch, n_coil), np.nan)
    for c in coils:
        if boot_g[c]:
            arr = np.array(boot_g[c])
            g_lo[:, c], g_hi[:, c] = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    k_ci = {
        c: [
            float(np.percentile(boot_k[c], 2.5)),
            float(np.percentile(boot_k[c], 97.5)),
        ]
        for c in coils
        if len(boot_k[c]) >= 10
    }
    return {"g_lo": g_lo, "g_hi": g_hi, "k_ci": k_ci}


def _stack(rows: list[dict], *, quasi_static: bool, threshold: float):
    meas_l, i_l, didt_l = [], [], []
    n_slice = 0
    for r in rows:
        keep = (
            r["didt_slice"] <= threshold
            if quasi_static
            else np.ones(r["meas"].shape[0], dtype=bool)
        )
        if not keep.any():
            continue
        meas_l.append(r["meas"][keep])
        i_l.append(r["i_pf"][keep])
        didt_l.append(r["didt"][keep])
        n_slice += int(keep.sum())
    if not meas_l:
        return None
    return (
        np.concatenate(meas_l),
        np.concatenate(i_l),
        np.concatenate(didt_l),
        n_slice,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--didt-quantile", type=float, default=0.25)
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # canonical axes from a reference operator (channel/coil order + model G)
    ref = build_operator(build_table_for_shot(11774))
    channels = list(ref.sensor_channels)
    coil_channels = list(ref.pf_amc_channels)
    g_model = np.asarray(ref.g_pf, dtype=np.float64)  # (n_ch, n_coil)
    n_ch = len(channels)
    n_coil = len(coil_channels)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    all_shots = [int(s) for s in list(train_shots) + list(held_shots)]
    logger.info(
        "pooling %d shots (leakage-free: no EFIT, no inversion)", len(all_shots)
    )
    logger.info("canonical: %d sensors × %d coils", n_ch, n_coil)

    rows: list[dict] = []
    all_didt: list[np.ndarray] = []
    for s in all_shots:
        r = _shot_coil_only(s, channels, coil_channels)
        if r is None:
            logger.warning("shot %s: no usable coil-only slices", s)
            continue
        rows.append(r)
        all_didt.append(r["didt_slice"])
        logger.info("shot %s: %d coil-only slices", s, r["meas"].shape[0])
    if len(rows) < 3:
        logger.error("too few shots with coil-only slices (%d)", len(rows))
        return 1

    didt_pool = np.concatenate(all_didt)
    threshold = float(np.quantile(didt_pool, args.didt_quantile))
    logger.info(
        "|dI/dt| pool n=%d p25=%.3g median=%.3g p90=%.3g threshold=%.3g",
        didt_pool.size,
        np.quantile(didt_pool, 0.25),
        np.median(didt_pool),
        np.quantile(didt_pool, 0.90),
        threshold,
    )

    # σ = median plasma-on scale across shots (robust, frozen-comparable)
    all_sigma = np.stack([r["sigma"] for r in rows])
    with np.errstate(all="ignore"):
        sigma_med = np.nanmedian(all_sigma, axis=0)

    # ---- conditioning on the FULL pool ----
    full = _stack(rows, quasi_static=False, threshold=threshold)
    meas_full, i_full, didt_full, n_full = full
    plan = _design_plan(i_full)
    # SVD spectrum of the standardised design (measurable columns)
    meas_cols = [i for i in range(n_coil) if plan["range"][i] >= RANGE_FLOOR]
    z = i_full[:, meas_cols]
    z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1.0)
    sv = np.linalg.svd(z, compute_uv=False)
    eff_rank = int(np.sum(sv > sv[0] * 1e-3)) if sv.size else 0

    coupled_named = [[coil_channels[i] for i in comp] for comp in plan["coupled_sets"]]
    logger.info("separable coils: %s", [coil_channels[i] for i in plan["separable"]])
    logger.info("coupled sets: %s", coupled_named)
    logger.info("unmeasurable: %s", [coil_channels[i] for i in plan["unmeasurable"]])
    logger.info("effective rank %d of %d measurable columns", eff_rank, len(meas_cols))

    # ---- two fits: quasi-static, eddy-augmented ----
    def _run(quasi_static: bool, augment: bool) -> dict:
        st = _stack(rows, quasi_static=quasi_static, threshold=threshold)
        if st is None:
            return {}
        meas, i_pf, didt, n_slice = st
        design, coil_of = _build_regressors(i_pf, didt, plan, augment=augment)
        g_emp = _fit_g_emp(meas, design, coil_of, n_coil)
        k = _scale_factors(g_emp, g_model, sigma_med, plan["separable"])
        boot = _bootstrap(
            rows,
            plan,
            g_model,
            sigma_med,
            n_coil,
            augment=augment,
            threshold=threshold,
            quasi_static=quasi_static,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        return {
            "n_slices": int(n_slice),
            "g_emp": g_emp,
            "k": k,
            "boot": boot,
        }

    quasi = _run(quasi_static=True, augment=False)
    eddy = _run(quasi_static=False, augment=True)

    # ---- pattern residuals at p90 coil currents (quasi-static, primary) ----
    i_p90 = np.nanpercentile(np.abs(i_full), 90, axis=0)  # per coil
    g_emp_q = quasi["g_emp"]
    k_q = quasi["k"]
    pattern = []  # worst (coil, sensor) residuals in σ units
    resid_sigma = np.full((n_coil, n_ch), np.nan)
    for c in plan["separable"]:
        if c not in k_q:
            continue
        resid = g_emp_q[:, c] - k_q[c] * g_model[:, c]  # per-sensor [units/A]
        contrib = resid * i_p90[c]  # sensor reading contribution at p90 current
        rs = contrib / np.where(sigma_med > 0, sigma_med, np.nan)
        resid_sigma[c] = rs
        for s in range(n_ch):
            if np.isfinite(rs[s]):
                pattern.append(
                    {
                        "coil": coil_channels[c],
                        "sensor": channels[s],
                        "resid_over_sigma": float(rs[s]),
                    }
                )
    pattern.sort(key=lambda d: -abs(d["resid_over_sigma"]))

    # ---- quasi-static vs eddy-augmented agreement of static G ----
    agree = {}
    for c in plan["separable"]:
        gq = g_emp_q[:, c]
        ge = eddy["g_emp"][:, c]
        good = np.isfinite(gq) & np.isfinite(ge)
        if good.sum() < 3:
            continue
        denom = np.maximum(np.abs(gq[good]), np.abs(ge[good]))
        rel = np.abs(gq[good] - ge[good]) / np.where(denom > 0, denom, np.nan)
        agree[coil_channels[c]] = {
            "median_rel_diff": float(np.nanmedian(rel)),
            "k_quasistatic": k_q.get(c),
            "k_eddy": eddy["k"].get(c),
        }

    # ---- per-channel worst-case model error in σ at typical currents ----
    worst_ch = []
    for s in range(n_ch):
        col = resid_sigma[:, s]
        if np.isfinite(col).any():
            worst_ch.append(
                {
                    "sensor": channels[s],
                    "worst_resid_over_sigma": float(np.nanmax(np.abs(col))),
                }
            )
    worst_ch.sort(key=lambda d: -d["worst_resid_over_sigma"])

    def _k_block(res: dict) -> dict:
        k = res.get("k", {})
        k_ci = res.get("boot", {}).get("k_ci", {})
        return {
            coil_channels[c]: {
                "k": float(k[c]),
                "ci": k_ci.get(c),
                "diff_from_1_sig": (
                    None
                    if c not in k_ci
                    else bool(k_ci[c][0] > 1.0 or k_ci[c][1] < 1.0)
                ),
            }
            for c in k
        }

    out = {
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "leakage_free": True,
        "leakage_note": (
            "No EFIT and no plasma inversion enter this audit; it uses only raw "
            "amb magnetics, raw amc coil currents, and the geometry-only "
            "operator. All 48 fleet shots (train + held-out) are pooled — there "
            "is no train/test contamination to guard against."
        ),
        "ip_vacuum_ka": IP_VACUUM_KA,
        "corr_couple_threshold": CORR_COUPLE,
        "range_floor": RANGE_FLOOR,
        "didt_quantile": args.didt_quantile,
        "didt_threshold": threshold,
        "n_shots_used": len(rows),
        "shots_used": [r["shot"] for r in rows],
        "n_slices_full_pool": n_full,
        "channels": channels,
        "coil_channels": coil_channels,
        "sigma_median": sigma_med.tolist(),
        "conditioning": {
            "coil_current_std": plan["std"].tolist(),
            "coil_current_ptp": plan["ptp"].tolist(),
            "coil_dynamic_range": [
                None if not np.isfinite(v) else float(v) for v in plan["range"]
            ],
            "abs_corr": plan["abs_corr"].tolist(),
            "separable_coils": [coil_channels[i] for i in plan["separable"]],
            "coupled_sets": coupled_named,
            "unmeasurable_coils": [coil_channels[i] for i in plan["unmeasurable"]],
            "svd_spectrum": sv.tolist(),
            "effective_rank": eff_rank,
            "n_measurable_columns": len(meas_cols),
        },
        "scale_factors": {
            "quasistatic": _k_block(quasi),
            "eddy_augmented": _k_block(eddy),
        },
        "quasistatic_vs_eddy_agreement": agree,
        "pattern_residuals_top": pattern[:40],
        "per_channel_worst_model_error": worst_ch[:40],
        "n_slices": {
            "quasistatic": quasi.get("n_slices"),
            "eddy_full_pool": eddy.get("n_slices"),
        },
        "special_coils": {
            "solenoid": "sol_current",
            "p4_flux_offset": ["p4l_coil_current", "p4u_coil_current"],
            "p5_flux_offset": ["p5l_coil_current", "p5u_coil_current"],
            "p6_in_vessel": ["p6l_current", "p6u_current"],
        },
    }
    out_path = ARTIFACTS / "vacuum_coil_response_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)

    _figures(
        out,
        plan,
        g_emp_q,
        k_q,
        g_model,
        sigma_med,
        resid_sigma,
        quasi,
        channels,
        coil_channels,
    )
    return 0


def _figures(
    out,
    plan,
    g_emp_q,
    k_q,
    g_model,
    sigma_med,
    resid_sigma,
    quasi,
    channels,
    coil_channels,
):
    n_coil = len(coil_channels)

    # --- fig 1: per-coil scale factors k_c with CIs ---
    k_block = out["scale_factors"]["quasistatic"]
    k_e = out["scale_factors"]["eddy_augmented"]
    coils = [c for c in coil_channels if c in k_block]
    if coils:
        xs = np.arange(len(coils))
        kv = np.array([k_block[c]["k"] for c in coils])
        lo = np.array(
            [k_block[c]["ci"][0] if k_block[c]["ci"] else np.nan for c in coils]
        )
        hi = np.array(
            [k_block[c]["ci"][1] if k_block[c]["ci"] else np.nan for c in coils]
        )
        ke = np.array([k_e[c]["k"] if c in k_e else np.nan for c in coils])
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.errorbar(
            xs,
            kv,
            yerr=[kv - lo, hi - kv],
            fmt="o",
            ms=6,
            color="#1565c0",
            ecolor="#9bb8d9",
            elinewidth=1.2,
            capsize=3,
            label="quasi-static k_c ± 95% CI",
        )
        ax.plot(
            xs, ke, "s", ms=5, color="#8a3324", alpha=0.7, label="eddy-augmented k_c"
        )
        ax.axhline(1.0, color="k", lw=1.0, label="k = 1 (no turns/xmult error)")
        # flag k significantly ≠ 1
        for i, c in enumerate(coils):
            if k_block[c]["diff_from_1_sig"]:
                ax.annotate(
                    "≠1",
                    (xs[i], hi[i]),
                    fontsize=8,
                    color="#b00",
                    ha="center",
                    va="bottom",
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(coils, rotation=90, fontsize=7)
        ax.set_ylabel("multiplicative scale k_c")
        ax.set_title(
            "Empirical per-coil response scale vs geometry model "
            "(k_c ≠ 1 = turns / xmult error); coil-only pool, "
            f"{out['n_shots_used']} shots"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig-coil-response-scales.png", dpi=140)
        plt.close(fig)
        logger.info("wrote %s", FIGURES / "fig-coil-response-scales.png")

    # --- fig 2: coil-current correlation heatmap + coupled sets ---
    abs_corr = np.array(out["conditioning"]["abs_corr"])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(abs_corr, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_coil))
    ax.set_yticks(range(n_coil))
    ax.set_xticklabels(coil_channels, rotation=90, fontsize=6)
    ax.set_yticklabels(coil_channels, fontsize=6)
    # outline coupled sets
    for comp in plan["coupled_sets"]:
        for i in comp:
            for j in comp:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#39ff14",
                        lw=1.2,
                    )
                )
    fig.colorbar(im, ax=ax, fraction=0.045, label="|correlation|")
    sets_txt = (
        "; ".join(
            "+".join(coil_channels[i].replace("_current", "") for i in comp)
            for comp in plan["coupled_sets"]
        )
        or "none"
    )
    ax.set_title(
        f"Coil-current design correlation (coil-only pool)\n"
        f"UNSEPARABLE sets (|corr|>{CORR_COUPLE}, green): {sets_txt}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig-coil-response-corr.png", dpi=140)
    plt.close(fig)
    logger.info("wrote %s", FIGURES / "fig-coil-response-corr.png")

    # --- fig 3: coil×sensor pattern residual /σ at p90 currents ---
    sep = plan["separable"]
    if sep:
        mat = resid_sigma[sep]  # (n_sep, n_ch)
        fig, ax = plt.subplots(figsize=(15, max(3, 0.4 * len(sep))))
        vlim = np.nanpercentile(np.abs(mat), 98) if np.isfinite(mat).any() else 1.0
        vlim = max(vlim, 1e-3)
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(sep)))
        ax.set_yticklabels([coil_channels[i] for i in sep], fontsize=7)
        step = max(1, len(channels) // 50)
        ax.set_xticks(range(0, len(channels), step))
        ax.set_xticklabels(channels[::step], rotation=90, fontsize=5)
        ax.set_xlabel("sensor channel")
        fig.colorbar(im, ax=ax, fraction=0.02, label="residual / σ  @ p90 current")
        ax.set_title(
            "Geometry-signature residual  (G_emp − k_c·G_model)·I_p90 / σ  — "
            "structure = per-coil geometry error"
        )
        fig.tight_layout()
        fig.savefig(FIGURES / "fig-coil-response-residuals.png", dpi=140)
        plt.close(fig)
        logger.info("wrote %s", FIGURES / "fig-coil-response-residuals.png")


if __name__ == "__main__":
    raise SystemExit(main())
