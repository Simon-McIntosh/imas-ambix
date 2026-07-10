#!/usr/bin/env python
"""Plasma-free (vacuum-interval) validation of the static magnetics calibration.

The frozen offset-only static calibration
(``static_calibration_offset_only.json``) was fitted STATISTICALLY against a
plasma inversion arm: its per-channel offsets are whatever collapses the
free-arm halo on train shots.  That leaves an adjudication open — is the offset
a genuine instrument zero, and were the rejected full-affine gains (p10-p90
0.90-1.14) laundering real plasma signal scale?  The only place the truth is
pure forward physics is a COIL-ONLY (vacuum) interval: no plasma, no inversion.

This script measures, over the calibration fit shots AND the held-out gate
shots (validation only — uses NO EFIT and NO inversion, so it is firewall-clean
and leakage-free by construction):

* per-channel plasma-free OFFSET  — intercept of measured ≈ gain·vacuum + offset
  fitted over coil-only slices, on the trustworthy quasi-static stratum;
* per-channel plasma-free GAIN     — slope of the same fit (only meaningful on
  channels whose vacuum prediction carries real dynamic range; degenerate
  channels are pinned to identity, as the plasma fit does);
* agreement with the frozen plasma-fitted offsets — (offset_vacuum −
  offset_frozen)/σ, fraction of channels whose frozen offset falls inside the
  vacuum bootstrap CI;
* shot-to-shot offset STABILITY — between-shot vs within-shot variance of the
  per-shot offsets (the cross-shot filtering-proposal evidence).

Physics cautions baked in:

* Pre-breakdown the solenoid ramps fast → induced vessel eddy currents
  contaminate measured − vacuum.  Per slice we compute the coil-current time
  derivative and STRATIFY quasi-static (all |dI/dt| below a data-driven
  threshold) vs ramping; the quasi-static stratum is the trustworthy one and
  both are reported so the eddy contamination is itself measured.
* Two interval families are labelled separately: the pre-breakdown ramp (before
  the first plasma-on slice) and the post-plasma dwell tail (after the last),
  which carries halo/disruption ambiguity.

σ is the plasma-on robust channel scale (same units as the frozen
calibration's ``offset_over_sigma``).  Sign/units are taken straight from the
shared assembly code paths (``load_shot_slices_raw`` + the operator forward
model) — never re-derived.

Artifacts: imas_ambix/latent/artifacts/patch_gate/vacuum_interval_calibration.json
Figures:   docs/figures/force-balance-spine/
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
logger = logging.getLogger("vacuum_interval_calibration")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/force-balance-spine")

FROZEN_OFFSET = ARTIFACTS / "static_calibration_offset_only.json"
FROZEN_FREE_OFFSET = ARTIFACTS / "static_calibration_offset_only_free.json"
AFFINE_GAIN = ARTIFACTS / "static_calibration.json"

#: coil-only guard: a slice counts as vacuum only if |Ip| is below this (kA),
#: on top of the loader's own plasma-on flag being False.
IP_VACUUM_KA = 20.0

#: a channel's vacuum-fit gain is only meaningful where the vacuum prediction
#: spans real dynamic range: ptp(pred)/σ below this → pin to identity (gain=1),
#: mirroring static_calibration_audit's ``np.ptp(pred) < 1e-12`` degeneracy guard
#: but in σ units so a genuinely quiet channel is not over-interpreted.
GAIN_RANGE_FLOOR = 0.5


def _shot_vacuum_slices(shot: int, channels: list[str]) -> dict | None:
    """Assemble one shot's coil-only slices on the canonical ``channels`` axis.

    Pure forward physics: reads level-1 raw, aligns the amb magnetics to the
    campaign operator's sensor rows BY NAME, assembles ``i_pf`` exactly as
    :func:`imas_ambix.latent.data.load_shot_windows` does, and evaluates the
    operator's vacuum (PF-only) prediction.  Returns per-slice ``pred`` /
    ``meas`` / ``mask`` stacks on the canonical channel axis, the coil-current
    time derivative per slice, the plasma-on σ, and the family membership
    (pre-breakdown ramp vs post-plasma dwell) — or ``None`` if the shot has no
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
    on_idx = np.flatnonzero(plasma_on)
    first_on, last_on = int(on_idx[0]), int(on_idx[-1])

    offsets = schema_group_offsets(schema)
    amb_names = schema["amb"]
    amc_names = schema["amc"]
    op_rows, x_cols = align_sensor_columns(fwd.sensor_channels, amb_names)

    n_sensor = len(fwd.sensor_channels)
    t_n = x.shape[0]
    raw_mag = np.full((t_n, n_sensor), np.nan)
    if op_rows.size:
        raw_mag[:, op_rows] = x[:, offsets["amb"] + x_cols]
    mag_mask_full = np.isfinite(raw_mag)

    # plasma-on σ on the operator sensor rows → frozen-comparable units
    sigma_op = robust_channel_scale(
        np.nanstd(raw_mag[plasma_on], axis=0), fwd.sensor_channels
    )

    # i_pf per slice, assembled exactly as load_shot_windows
    n_coil = len(fwd.pf_amc_channels)
    i_pf = np.zeros((t_n, n_coil))
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

    # coil-only slices, split into the two interval families
    idx = np.arange(t_n)
    coil_only = (~plasma_on) & np.isfinite(ip_ka) & (ip_ka < IP_VACUUM_KA)
    prebreak = coil_only & (idx < first_on)
    postdwell = coil_only & (idx > last_on)
    if not (prebreak.any() or postdwell.any()):
        return None

    # per-slice coil-current derivative (max over coils) — the eddy proxy
    didt = np.gradient(i_pf, times, axis=0) if t_n >= 2 else np.zeros_like(i_pf)
    didt_slice = np.max(np.abs(didt), axis=1)

    # map operator sensor rows → canonical channel axis (NaN where absent)
    row_of = {ch: r for r, ch in enumerate(fwd.sensor_channels)}
    can_rows = np.array([row_of.get(ch, -1) for ch in channels])
    present = can_rows >= 0
    safe_rows = np.clip(can_rows, 0, None)

    def _gather(mat: np.ndarray, sel: np.ndarray) -> np.ndarray:
        # (n_sel, n_canonical) with NaN in channels the shot does not carry
        out = np.where(present, mat[sel][:, safe_rows], np.nan)
        return out

    pred_full = np.stack([fwd.vacuum_prediction(i_pf[t]) for t in range(t_n)])
    sigma_can = np.where(present, sigma_op[safe_rows], np.nan)

    fam = {}
    for name, sel_mask in (("prebreak", prebreak), ("postdwell", postdwell)):
        sel = np.flatnonzero(sel_mask)
        if sel.size == 0:
            continue
        fam[name] = {
            "pred": _gather(pred_full, sel),
            "meas": _gather(raw_mag, sel),
            "mask": _gather(mag_mask_full.astype(float), sel) > 0.5,
            "didt": didt_slice[sel],
        }
    if not fam:
        return None
    return {"shot": int(shot), "sigma": sigma_can, "present": present, "families": fam}


def _fit_offset_gain(
    pred: np.ndarray, meas: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel affine fit measured ≈ gain·pred + offset over pooled slices.

    ``pred`` / ``meas`` are ``(n_slice, n_channel)`` with NaN where absent.
    Returns ``(offset, gain, offset_only, range_over_sigma)`` — the affine fit's
    intercept/slope, the gain-pinned offset (median residual), and the vacuum
    prediction's dynamic range in σ units.  A channel with < 8 finite paired
    samples or sub-:data:`GAIN_RANGE_FLOOR` dynamic range keeps gain = 1 and
    reports only the offset (median residual).
    """
    n_ch = pred.shape[1]
    offset = np.full(n_ch, np.nan)
    gain = np.ones(n_ch)
    offset_only = np.full(n_ch, np.nan)
    range_sigma = np.zeros(n_ch)
    for s in range(n_ch):
        good = np.isfinite(pred[:, s]) & np.isfinite(meas[:, s])
        if good.sum() < 8:
            continue
        p = pred[good, s]
        m = meas[good, s]
        sig = sigma[s] if np.isfinite(sigma[s]) and sigma[s] > 0 else 1.0
        range_sigma[s] = float(np.ptp(p) / sig)
        offset_only[s] = float(np.median(m - p))
        if range_sigma[s] < GAIN_RANGE_FLOOR:
            offset[s] = offset_only[s]  # degenerate: identity gain
            continue
        a = np.polyfit(p, m, 1)
        res = m - np.polyval(a, p)
        keep = np.abs(res - np.median(res)) <= 3.0 * (np.std(res) + 1e-30)
        if keep.sum() >= 8:
            a = np.polyfit(p[keep], m[keep], 1)
        gain[s], offset[s] = float(a[0]), float(a[1])
    return offset, gain, offset_only, range_sigma


def _stack_family(
    shots: list[dict], family: str, *, quasi_static: bool, threshold: float
):
    """Pool one interval family's slices across shots into (pred, meas) stacks.

    ``quasi_static`` selects the stratum: True keeps slices with |dI/dt| ≤
    ``threshold``, False keeps the ramping complement.  Returns pooled
    ``(pred, meas)`` plus the per-shot pooled slice lists (for the stability
    decomposition and bootstrap-over-shots).
    """
    preds, meass, per_shot = [], [], []
    for sh in shots:
        fam = sh["families"].get(family)
        if fam is None:
            continue
        keep = fam["didt"] <= threshold if quasi_static else fam["didt"] > threshold
        if not keep.any():
            continue
        pred = np.where(fam["mask"][keep], fam["pred"][keep], np.nan)
        meas = np.where(fam["mask"][keep], fam["meas"][keep], np.nan)
        preds.append(pred)
        meass.append(meas)
        per_shot.append({"shot": sh["shot"], "pred": pred, "meas": meas})
    if not preds:
        return np.empty((0, 0)), np.empty((0, 0)), []
    return np.concatenate(preds), np.concatenate(meass), per_shot


def _bootstrap_over_shots(
    per_shot: list[dict], sigma: np.ndarray, *, n_boot: int, seed: int
):
    """Percentile CIs of the affine offset/gain, resampling SHOTS with
    replacement (a shot's slices always move together)."""
    if len(per_shot) < 2:
        return None
    rng = np.random.default_rng(seed)
    boots_o, boots_g = [], []
    for _ in range(n_boot):
        draw = rng.integers(0, len(per_shot), len(per_shot))
        pred = np.concatenate([per_shot[i]["pred"] for i in draw])
        meas = np.concatenate([per_shot[i]["meas"] for i in draw])
        o, g, _oo, _r = _fit_offset_gain(pred, meas, sigma)
        boots_o.append(o)
        boots_g.append(g)
    boots_o = np.array(boots_o)
    boots_g = np.array(boots_g)
    o_lo, o_hi = np.nanpercentile(boots_o, [2.5, 97.5], axis=0)
    g_lo, g_hi = np.nanpercentile(boots_g, [2.5, 97.5], axis=0)
    return o_lo, o_hi, g_lo, g_hi


def _per_shot_offset_matrix(per_shot: list[dict], n_ch: int):
    """(n_shot, n_channel) per-shot offset (median residual) + within-shot
    variance, for the stability decomposition."""
    shots = [d["shot"] for d in per_shot]
    off = np.full((len(per_shot), n_ch), np.nan)
    within_var = np.full((len(per_shot), n_ch), np.nan)
    for i, d in enumerate(per_shot):
        resid = d["meas"] - d["pred"]  # (n_slice, n_ch)
        with np.errstate(all="ignore"):
            off[i] = np.nanmedian(resid, axis=0)
            within_var[i] = np.nanvar(resid, axis=0)
    return shots, off, within_var


def _variance_decomposition(off: np.ndarray, within_var: np.ndarray):
    """Between-shot vs within-shot variance of the per-shot offsets, per channel.

    between = var across shots of the per-shot offset; within = mean across
    shots of the within-shot residual variance.  A between/within ratio ≪ 1
    means the offset is near-constant shot-to-shot → a cross-shot averaging
    pass collapses the nuisance space without losing per-shot signal.
    """
    with np.errstate(all="ignore"):
        between = np.nanvar(off, axis=0)
        within = np.nanmean(within_var, axis=0)
    ratio = between / np.where(within > 0, within, np.nan)
    return between, within, ratio


def _load_frozen(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    cal = json.loads(path.read_text())
    return dict(zip(cal["channels"], cal["offset"], strict=True))


def _load_frozen_gain(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    cal = json.loads(path.read_text())
    return dict(zip(cal["channels"], cal["gain"], strict=True))


def _median_abs(v: np.ndarray) -> float | None:
    finite = v[np.isfinite(v)]
    return float(np.median(np.abs(finite))) if finite.size else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--n-fit-shots", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="artifact name suffix (e.g. a coil-model version tag)",
    )
    ap.add_argument(
        "--didt-quantile",
        type=float,
        default=0.25,
        help="quasi-static threshold = this quantile of the pooled |dI/dt|",
    )
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # canonical channel axis = the frozen calibration's channels (directly
    # comparable); fall back to the first shot's operator rows if absent
    frozen_off = _load_frozen(FROZEN_OFFSET)
    frozen_free_off = _load_frozen(FROZEN_FREE_OFFSET)
    frozen_gain = _load_frozen_gain(AFFINE_GAIN)
    if frozen_off is not None:
        channels = list(json.loads(FROZEN_OFFSET.read_text())["channels"])
    else:
        channels = list(build_operator(build_table_for_shot(11774)).sensor_channels)
    n_ch = len(channels)

    train_shots, held_shots = read_split_shot_lists(args.n_train, args.n_heldout)
    fit_shots = train_shots[
        args.n_baseline_shots : args.n_baseline_shots + args.n_fit_shots
    ]
    logger.info("canonical channels: %d", n_ch)
    logger.info("fit shots: %s", fit_shots)
    logger.info("held-out shots: %s", held_shots)

    cohorts = {"fit": fit_shots, "heldout": held_shots}
    loaded: dict[str, list[dict]] = {}
    all_didt: list[np.ndarray] = []
    for cohort, shots in cohorts.items():
        rows = []
        for s in shots:
            r = _shot_vacuum_slices(int(s), channels)
            if r is None:
                logger.warning("[%s] shot %s: no usable coil-only slices", cohort, s)
                continue
            rows.append(r)
            for fam in r["families"].values():
                all_didt.append(fam["didt"])
            logger.info(
                "[%s] shot %s: %s",
                cohort,
                s,
                {k: int(v["pred"].shape[0]) for k, v in r["families"].items()},
            )
        loaded[cohort] = rows

    if not all_didt:
        logger.error("no coil-only slices found in any shot")
        return 1
    didt_pool = np.concatenate(all_didt)
    threshold = float(np.quantile(didt_pool, args.didt_quantile))
    logger.info(
        "|dI/dt| pool: n=%d  p25=%.3g  median=%.3g  p90=%.3g  threshold=%.3g",
        didt_pool.size,
        np.quantile(didt_pool, 0.25),
        np.median(didt_pool),
        np.quantile(didt_pool, 0.90),
        threshold,
    )

    # σ = median plasma-on scale across all loaded shots (frozen-comparable)
    all_sigma = np.stack(
        [r["sigma"] for rows in loaded.values() for r in rows if r is not None]
    )
    with np.errstate(all="ignore"):
        sigma_med = np.nanmedian(all_sigma, axis=0)

    def _run(cohort: str, family: str, quasi_static: bool) -> dict:
        pred, meas, per_shot = _stack_family(
            loaded[cohort], family, quasi_static=quasi_static, threshold=threshold
        )
        block: dict = {
            "n_slices": int(pred.shape[0]),
            "n_shots": len(per_shot),
        }
        if pred.shape[0] == 0:
            return block
        offset, gain, offset_only, range_sigma = _fit_offset_gain(pred, meas, sigma_med)
        block.update(
            offset=offset.tolist(),
            gain=gain.tolist(),
            offset_only=offset_only.tolist(),
            offset_over_sigma=(offset / (sigma_med + 1e-30)).tolist(),
            range_over_sigma=range_sigma.tolist(),
            gain_median=float(np.nanmedian(gain)),
            gain_p10_p90=[
                float(np.nanpercentile(gain, 10)),
                float(np.nanpercentile(gain, 90)),
            ],
        )
        boot = _bootstrap_over_shots(
            per_shot, sigma_med, n_boot=args.n_boot, seed=args.seed
        )
        if boot is not None:
            o_lo, o_hi, g_lo, g_hi = boot
            block.update(
                offset_ci_lo=o_lo.tolist(),
                offset_ci_hi=o_hi.tolist(),
                gain_ci_lo=g_lo.tolist(),
                gain_ci_hi=g_hi.tolist(),
            )
        shots, off_mat, within_var = _per_shot_offset_matrix(per_shot, n_ch)
        between, within, ratio = _variance_decomposition(off_mat, within_var)
        block.update(
            per_shot_ids=shots,
            per_shot_offset=off_mat.tolist(),
            between_shot_var=between.tolist(),
            within_shot_var=within.tolist(),
            between_over_within=ratio.tolist(),
            between_over_within_median=(
                float(np.nanmedian(ratio[np.isfinite(ratio)]))
                if np.isfinite(ratio).any()
                else None
            ),
        )
        return block

    results: dict[str, dict] = {}
    for cohort in cohorts:
        for family in ("prebreak", "postdwell"):
            for qs, label in ((True, "quasistatic"), (False, "ramping")):
                key = f"{cohort}__{family}__{label}"
                results[key] = _run(cohort, family, qs)

    # ---- agreement with the frozen plasma-fitted offset (primary stratum) ----
    primary = results["fit__prebreak__quasistatic"]
    agreement: dict = {}
    if "offset" in primary and frozen_off is not None:
        vac_off = np.array(primary["offset"])
        vac_off_sig = np.array(primary["offset_over_sigma"])
        froz = np.array([frozen_off.get(c, np.nan) for c in channels])
        froz_sig = froz / (sigma_med + 1e-30)
        delta_sig = vac_off_sig - froz_sig
        in_ci = np.zeros(n_ch, dtype=bool)
        if "offset_ci_lo" in primary:
            lo = np.array(primary["offset_ci_lo"])
            hi = np.array(primary["offset_ci_hi"])
            in_ci = (froz >= lo) & (froz <= hi)
        # channels the vacuum fit actually constrains
        constrained = np.isfinite(vac_off) & np.isfinite(froz)
        agreement = {
            "delta_over_sigma": delta_sig.tolist(),
            "median_abs_delta_over_sigma": _median_abs(delta_sig[constrained]),
            "frozen_in_vacuum_ci_fraction": (
                float(in_ci[constrained].sum() / max(1, constrained.sum()))
            ),
            "n_constrained_channels": int(constrained.sum()),
        }

    # ---- cross-arm offset consistency table (frozen priors vs free vs vacuum)
    consistency: dict = {}
    vac = np.array(primary.get("offset", [np.nan] * n_ch))
    priors_off = np.array([(frozen_off or {}).get(c, np.nan) for c in channels])
    free_off = np.array([(frozen_free_off or {}).get(c, np.nan) for c in channels])
    for a_name, a, b_name, b in (
        ("vacuum", vac, "priors", priors_off),
        ("vacuum", vac, "free", free_off),
        ("priors", priors_off, "free", free_off),
    ):
        d = (a - b) / (sigma_med + 1e-30)
        consistency[f"{a_name}_vs_{b_name}_median_abs_delta_over_sigma"] = _median_abs(
            d
        )
    consistency["free_offset_available"] = frozen_free_off is not None

    out = {
        "coil_model_version": COIL_MODEL_VERSION,
        "geometry_table_version": GEOMETRY_TABLE_VERSION,
        "ip_vacuum_ka": IP_VACUUM_KA,
        "gain_range_floor": GAIN_RANGE_FLOOR,
        "didt_quantile": args.didt_quantile,
        "didt_threshold": threshold,
        "fit_shots": [int(s) for s in fit_shots],
        "held_shots": [int(s) for s in held_shots],
        "channels": channels,
        "sigma_median": sigma_med.tolist(),
        "results": results,
        "agreement_with_frozen": agreement,
        "cross_arm_consistency": consistency,
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    (ARTIFACTS / f"vacuum_interval_calibration{tag}.json").write_text(
        json.dumps(out, indent=2)
    )
    logger.info("wrote %s", ARTIFACTS / "vacuum_interval_calibration.json")

    _figures(out, sigma_med, channels, frozen_off, frozen_gain)
    return 0


def _figures(out, sigma_med, channels, frozen_off, frozen_gain):
    n_ch = len(channels)
    x = np.arange(n_ch)
    step = max(1, n_ch // 40)
    tick_lbl = [c.split("/")[-1] for c in channels[::step]]
    primary = out["results"].get("fit__prebreak__quasistatic", {})

    # --- fig 1: vacuum offset validation vs frozen ---
    if "offset" in primary:
        vac_sig = np.array(primary["offset_over_sigma"])
        froz = np.array([(frozen_off or {}).get(c, np.nan) for c in channels])
        froz_sig = froz / (sigma_med + 1e-30)
        fig, axes = plt.subplots(2, 1, figsize=(13, 7))
        ax = axes[0]
        if "offset_ci_lo" in primary:
            lo = np.array(primary["offset_ci_lo"]) / (sigma_med + 1e-30)
            hi = np.array(primary["offset_ci_hi"]) / (sigma_med + 1e-30)
            ax.errorbar(
                x,
                vac_sig,
                yerr=[vac_sig - lo, hi - vac_sig],
                fmt="o",
                ms=3,
                color="#1565c0",
                ecolor="#9bb8d9",
                elinewidth=0.8,
                label="vacuum (quasi-static) offset ± 95% CI",
            )
        else:
            ax.plot(x, vac_sig, "o", ms=3, color="#1565c0", label="vacuum offset")
        ax.plot(x, froz_sig, "x", ms=5, color="#8a3324", label="frozen plasma-fitted")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("offset / channel σ")
        ax.set_title(
            "Plasma-free (vacuum-interval) offsets vs the frozen plasma-fitted "
            f"calibration — pre-breakdown quasi-static, {primary['n_shots']} fit shots"
        )
        ax.legend(fontsize=8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(tick_lbl, rotation=90, fontsize=6)
        ax2 = axes[1]
        agr = out["agreement_with_frozen"]
        if "delta_over_sigma" in agr:
            dsig = np.array(agr["delta_over_sigma"])
            dsig = dsig[np.isfinite(dsig)]
            ax2.hist(dsig, bins=30, color="#4a7", edgecolor="k", alpha=0.8)
            ax2.axvline(0, color="k", lw=0.8)
            ax2.set_xlabel("(offset_vacuum − offset_frozen) / σ")
            ax2.set_ylabel("channels")
            med_d = agr.get("median_abs_delta_over_sigma")
            ax2.set_title(
                f"agreement: median |Δ|/σ = "
                f"{f'{med_d:.3f}' if med_d is not None else 'n/a'}  •  frozen "
                f"inside vacuum CI = "
                f"{100.0 * agr.get('frozen_in_vacuum_ci_fraction', 0.0):.0f}%"
            )
        fig.tight_layout()
        fig.savefig(FIGURES / "fig-vacuum-offset-validation.png", dpi=140)
        plt.close(fig)
        logger.info("wrote %s", FIGURES / "fig-vacuum-offset-validation.png")

    # --- fig 2: shot-to-shot offset stability ---
    if "per_shot_offset" in primary:
        off_mat = np.array(primary["per_shot_offset"]) / (sigma_med + 1e-30)
        shots = primary["per_shot_ids"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[3, 1])
        vlim = np.nanpercentile(np.abs(off_mat), 95) if off_mat.size else 1.0
        im = axes[0].imshow(
            off_mat,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        axes[0].set_yticks(range(len(shots)))
        axes[0].set_yticklabels(shots, fontsize=7)
        axes[0].set_xticks(x[::step])
        axes[0].set_xticklabels(tick_lbl, rotation=90, fontsize=6)
        axes[0].set_ylabel("fit shot")
        axes[0].set_title("per-shot vacuum offset / σ (pre-breakdown quasi-static)")
        fig.colorbar(im, ax=axes[0], fraction=0.03, label="offset / σ")
        ratio = np.array(primary["between_over_within"])
        ratio = ratio[np.isfinite(ratio)]
        axes[1].hist(np.log10(ratio + 1e-6), bins=25, color="#a67", edgecolor="k")
        axes[1].axvline(0, color="k", lw=0.8, label="between = within")
        med = primary.get("between_over_within_median")
        axes[1].set_xlabel("log10(between-shot / within-shot var)")
        axes[1].set_ylabel("channels")
        axes[1].set_title(
            f"stability: median between/within = "
            f"{f'{med:.3f}' if med is not None else 'n/a'}"
        )
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig-offset-stability-shots.png", dpi=140)
        plt.close(fig)
        logger.info("wrote %s", FIGURES / "fig-offset-stability-shots.png")

    # --- fig 3: gain adjudication ---
    # The quasi-static stratum has (by definition) no coil-current dynamic
    # range, so a per-channel gain is unidentifiable there (pinned to 1).  The
    # ONLY vacuum stratum with leverage to fit a gain is the RAMPING stratum —
    # but that is exactly where induced vessel eddy currents contaminate
    # measured − predicted, so its gain is an eddy artifact, not a static
    # instrument scale.  Plotting it against the rejected plasma-fitted affine
    # gains and gain=1 is the adjudication: is there ANY clean support for a
    # real per-channel scale error?
    ramp = out["results"].get("fit__prebreak__ramping", {})
    if "gain" in ramp:
        gain = np.array(ramp["gain"])
        rng_sig = np.array(ramp["range_over_sigma"])
        meaningful = rng_sig >= GAIN_RANGE_FLOOR
        fig, ax = plt.subplots(figsize=(13, 5))
        if "gain_ci_lo" in ramp:
            glo = np.array(ramp["gain_ci_lo"])
            ghi = np.array(ramp["gain_ci_hi"])
            ax.errorbar(
                x[meaningful],
                gain[meaningful],
                yerr=[(gain - glo)[meaningful], (ghi - gain)[meaningful]],
                fmt="o",
                ms=4,
                color="#1565c0",
                ecolor="#9bb8d9",
                elinewidth=0.8,
                label="vacuum ramping-stratum gain ± 95% CI (EDDY-contaminated)",
            )
        else:
            ax.plot(
                x[meaningful],
                gain[meaningful],
                "o",
                ms=4,
                color="#1565c0",
                label="vacuum ramping-stratum gain (eddy-contaminated)",
            )
        if frozen_gain is not None:
            aff = np.array([frozen_gain.get(c, np.nan) for c in channels])
            if np.isfinite(aff).any() and not np.allclose(aff[np.isfinite(aff)], 1.0):
                ax.plot(
                    x,
                    aff,
                    "s",
                    ms=4,
                    color="#8a3324",
                    alpha=0.7,
                    label="rejected plasma-fitted affine gain",
                )
        ax.axhline(1.0, color="k", lw=0.9, label="gain = 1 (no scale error)")
        ax.set_ylabel("gain")
        ax.set_xlabel("channel index")
        gm = ramp.get("gain_median")
        p10, p90 = ramp.get("gain_p10_p90", [None, None])
        gm_s = f"{gm:.3f}" if gm is not None else "n/a"
        p10_s = f"{p10:.3f}" if p10 is not None else "n/a"
        p90_s = f"{p90:.3f}" if p90 is not None else "n/a"
        ax.set_title(
            "Gain adjudication — quasi-static has NO coil range (gain "
            f"unidentifiable); ramping gain is eddy-driven: median {gm_s} "
            f"(p10-p90 {p10_s}-{p90_s}), {int(meaningful.sum())} ch with range"
        )
        ax.legend(fontsize=8)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(tick_lbl, rotation=90, fontsize=6)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig-gain-adjudication-vacuum.png", dpi=140)
        plt.close(fig)
        logger.info("wrote %s", FIGURES / "fig-gain-adjudication-vacuum.png")


if __name__ == "__main__":
    raise SystemExit(main())
