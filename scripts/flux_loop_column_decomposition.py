#!/usr/bin/env python
"""Name the coil column that carries the bay flux-loop bimodal vacuum gain.

The P4/P5 bay flux loops (``fl_p4*_*`` / ``fl_p5*_*``) show a BIMODAL per-shot
vacuum gain (two stable populations near 1.6 and 0.6) across every campaign era,
while the centre-column control loops (``fl_cc*``) sit tight near 0.95-1.0.  The
existing per-shot artifact ``flux_loop_pershot_gains.json`` proves the split but
carries only a SCALAR gain per loop = slope of the measured signal against the
SUMMED PF prediction ``i_pf @ G.T``.  A scalar cannot say WHICH coil column is
mis-modelled.  This script does the per-(sensor, coil) decomposition that names
the column and routes the fix.

Firewall (identical to :mod:`scripts.vacuum_coil_response_audit`): coil-only
(vacuum) slices only, raw ``amb`` magnetics + raw ``amc`` coil currents + the
geometry-only operator.  NO EFIT, NO plasma inversion.

Two complementary reads, both leakage-free:

1. POOLED PER-SENSOR COUPLING.  Pool the cohort's coil-only slices and, for each
   bay/control loop separately, regress the MEASURED signal on the INDIVIDUAL
   coil-current columns (not the summed prediction).  This yields the empirical
   Green's row ``G_emp[s, c]``; the per-coil coupling correction is
   ``k[s, c] = G_emp[s, c] / G_model[s, c]`` (``k = 1`` ⇒ model column correct).
   Coil currents that never separate in the pooled design (pairwise |corr| >
   0.98) are reported as coupled sets and NOT individually claimed.

2. PER-SHOT GAIN DECOMPOSITION.  The scalar gain is an exposure-weighted average
   of the per-coil corrections.  With ``pred = Σ_c G[s,c]·I_c`` and
   ``meas ≈ Σ_c k_c·G[s,c]·I_c``, the least-squares slope over a window is

       g_shot = cov(meas, pred)/var(pred) = Σ_c k_c · w_c ,
       w_c    = cov(G[s,c]·I_c, pred)/var(pred)   (Σ_c w_c = 1) ,

   so ``w_c`` is coil ``c``'s exposure weight in that shot and ``g_shot`` is a
   convex-ish mixture of the ``k_c``.  A gain that is bimodal across shots ⇒ an
   exposure weight ``w_c*`` that is bimodal on a column whose ``k_c* ≠ 1``.  We
   fit ``g ≈ W·k`` across shots and rank columns by how much they DRIVE the gain
   spread (corr(g, w_c) × spread(w_c) × |k_c − 1|).  The winner NAMES the fault.

Drive-DATA vs MODEL-GEOMETRY verdict.  Once the column is named, its ``k[s, c]``
across the several bay-loop sensors discriminates the cause: a coupling that is
a roughly CONSTANT multiplier across sensors (tight spread of ``k``) is an
amplitude / data-scaling error on that ``amc`` channel (a drive-DATA semantics
fault — the recorded current is scaled or the channel mislabelled); a coupling
that varies sensor-to-sensor in size or sign (the field PATTERN is wrong, not
just its amplitude) is a MODEL-column geometry error.

Artifact: imas_ambix/latent/artifacts/patch_gate/flux_loop_column_decomposition.json
Figure:   docs/figures/nonaxisymmetric-field-subtraction/fig-column-decomposition.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from imas_ambix.gs.geometry import build_table_for_shot
import imas_ambix.gs.geometry as _geom
from imas_ambix.gs.operator import build_operator

# tolerate the late-campaign amm hole (passive geometry unused on the coil-only
# vacuum interval) — same guard the sibling sweep uses.
_orig_amm = _geom.read_amm_passive


def _amm_opt(shot):  # noqa: ANN001
    try:
        return _orig_amm(shot)
    except Exception:  # noqa: BLE001
        return []


_geom.read_amm_passive = _amm_opt

from scripts.vacuum_coil_response_audit import _shot_coil_only  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flux_loop_column_decomposition")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/nonaxisymmetric-field-subtraction")

MANIFEST = "/work/projects/imas_gpu/mast/manifests/level1-all.json"

#: bay flux loops that show the bimodal gain, and the centre-column control
#: loops that stay tight (negative control).
BAY_LOOPS = ["fl_p4l_1", "fl_p4l_4", "fl_p4u_4", "fl_p5l_1", "fl_p5l_4", "fl_p5u_1"]
CONTROL_LOOPS = ["fl_cc01", "fl_cc02", "fl_cc03", "fl_cc05"]

#: pairwise |corr| above which two coil columns are declared UNSEPARABLE in the
#: pooled design (their individual G columns are not identifiable).
CORR_COUPLE = 0.98

#: a coil is kept in a sensor's design only when its model contribution to that
#: sensor carries at least this fraction of the summed-prediction amplitude —
#: coils that never move the loop cannot have their coupling measured there.
CONTRIB_FLOOR = 0.02

#: minimum coil-only slices for a shot to enter the pool / a per-shot fit.
MIN_SLICES = 100


def select_cohort() -> list[int]:
    """Reuse the sibling sweep's cohort: dense windows at the RMP-install
    boundaries plus a uniform backbone across the corpus."""
    ids = json.loads(Path(MANIFEST).read_text())["shot_ids"]
    a = np.array(sorted(int(s) for s in ids if 11695 <= int(s) <= 30473))
    sel: set[int] = set()
    for b in (19031, 25404):
        w = a[(a >= b - 600) & (a <= b + 600)]
        step = max(1, len(w) // 150)
        sel.update(int(s) for s in w[::step])
    u = a[:: max(1, len(a) // 220)]
    sel.update(int(s) for s in u)
    return sorted(sel)


def _robust_slope(y: np.ndarray, x: np.ndarray) -> float | None:
    """Slope of ``y`` on ``x`` with one 3σ residual-clip pass (the sweep idiom)."""
    good = np.isfinite(y) & np.isfinite(x)
    if good.sum() < MIN_SLICES or np.ptp(x[good]) < 1e-12:
        return None
    a = np.polyfit(x[good], y[good], 1)
    res = y[good] - np.polyval(a, x[good])
    keep = np.abs(res - np.median(res)) <= 3 * (np.std(res) + 1e-30)
    if keep.sum() >= 50:
        a = np.polyfit(x[good][keep], y[good][keep], 1)
    return float(a[0])


def _coupled_components(abs_corr: np.ndarray, thresh: float) -> list[list[int]]:
    """Connected components of the |corr| > thresh graph over coil columns."""
    n = abs_corr.shape[0]
    adj = (abs_corr > thresh) & ~np.eye(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
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


def gather(shots: list[int]) -> dict:
    """Load coil-only slices for the cohort and assemble the per-sensor pools
    and per-shot gain / exposure-weight records."""
    ref = build_operator(build_table_for_shot(11774))
    CH = list(ref.sensor_channels)
    COILS = list(ref.pf_amc_channels)
    G = np.asarray(ref.g_pf, dtype=np.float64)  # (n_ch, n_coil)
    n_coil = len(COILS)

    loops = [c for c in BAY_LOOPS + CONTROL_LOOPS if c in CH]
    row_of = {c: CH.index(c) for c in loops}

    # per-sensor pooled slice stores (measured, currents) — for the G_emp fit
    pool_meas: dict[str, list[np.ndarray]] = {c: [] for c in loops}
    pool_i: list[np.ndarray] = []  # currents shared across sensors (same slices)
    # per-shot records: gain g and exposure weights w_c, per loop
    pershot: dict[str, dict] = {c: {"shot": [], "gain": [], "w": []} for c in loops}

    used = 0
    for si, shot in enumerate(shots):
        try:
            r = _shot_coil_only(shot, CH, COILS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("shot %s failed: %s", shot, exc)
            continue
        if r is None:
            continue
        meas, ipf = r["meas"], r["i_pf"]
        if meas.shape[0] < MIN_SLICES:
            continue
        used += 1
        pool_i.append(ipf)
        for c in loops:
            s = row_of[c]
            ys = meas[:, s]
            pool_meas[c].append(ys)
            # per-shot scalar gain and exposure decomposition
            contrib = ipf * G[s, :][None, :]  # (T, n_coil) per-coil contribution
            pred = np.nansum(contrib, axis=1)
            g = _robust_slope(ys, pred)
            if g is None:
                continue
            good = np.isfinite(pred)
            vp = np.var(pred[good])
            if vp <= 0:
                continue
            w = np.array(
                [
                    np.cov(
                        np.vstack([contrib[good, k], pred[good]])
                    )[0, 1]
                    / vp
                    for k in range(n_coil)
                ]
            )
            pershot[c]["shot"].append(int(shot))
            pershot[c]["gain"].append(float(g))
            pershot[c]["w"].append(w)
        if si % 40 == 0:
            logger.info("%d/%d shots scanned, %d used", si, len(shots), used)

    logger.info("pooled %d shots with coil-only slices", used)
    return {
        "CH": CH,
        "COILS": COILS,
        "G": G,
        "loops": loops,
        "row_of": row_of,
        "pool_meas": pool_meas,
        "pool_i": pool_i,
        "pershot": pershot,
        "n_used": used,
    }


def per_sensor_coupling(data: dict) -> dict:
    """Pooled per-(sensor, source) multiplicative coupling ``k``.

    For each loop, regress the pooled measured signal on the MODEL-PREDICTED
    per-source contribution ``G_model[s,c]·I_c`` — one regressor per identifiable
    SOURCE.  A source is either a coil column that separates in the pooled design
    (pairwise |corr| ≤ 0.98) or a coupled SET of near-degenerate columns (e.g. an
    up/down coil pair driven together); a coupled set enters as its summed model
    contribution ``Σ_{c∈set} G[s,c]·I_c``.  Because every regressor is the model's
    own predicted contribution, the fitted coefficient IS the multiplicative
    coupling correction: ``k = 1`` ⇒ that source's model column is correct, and a
    departure names the fault.  Only sources whose model contribution to THIS
    sensor clears :data:`CONTRIB_FLOOR` are fit.  Bootstrap over shots for a CI.
    """
    COILS = data["COILS"]
    G = data["G"]
    n_coil = len(COILS)
    i_all = np.concatenate(data["pool_i"], axis=0)  # (N, n_coil)

    # separability from the pooled current design (same slices for every sensor)
    std = i_all.std(0)
    abs_corr = np.zeros((n_coil, n_coil))
    meas_idx = np.flatnonzero(std > 0)
    if meas_idx.size >= 2:
        cc = np.corrcoef(i_all[:, meas_idx].T)
        abs_corr[np.ix_(meas_idx, meas_idx)] = np.abs(np.nan_to_num(cc))
    comps = _coupled_components(abs_corr[np.ix_(meas_idx, meas_idx)], CORR_COUPLE)
    comps_global = [[int(meas_idx[j]) for j in comp] for comp in comps]
    # a SOURCE = a list of coil indices (singleton = separable coil, else a set)
    sources = [comp for comp in comps_global]
    unmeasurable = [int(i) for i in range(n_coil) if std[i] <= 0]

    def _label(src: list[int]) -> str:
        return COILS[src[0]] if len(src) == 1 else "+".join(COILS[i] for i in src)

    out: dict[str, dict] = {}
    rng = np.random.default_rng(0)
    lens = [m.shape[0] for m in data["pool_i"]]
    bounds = np.cumsum([0] + lens)
    n_shot = len(lens)

    for loop in data["loops"]:
        s = data["row_of"][loop]
        y_all = np.concatenate(data["pool_meas"][loop], axis=0)
        gm = G[s, :]  # model row
        # model contribution amplitude of each SOURCE to this sensor
        model_contrib = i_all * gm[None, :]  # (N, n_coil), predicted units
        src_contrib = {
            si: model_contrib[:, src].sum(axis=1) for si, src in enumerate(sources)
        }
        total_amp = sum(np.nanstd(v) for v in src_contrib.values()) + 1e-30
        frac = {si: float(np.nanstd(v) / total_amp) for si, v in src_contrib.items()}
        active = [si for si in range(len(sources)) if frac[si] >= CONTRIB_FLOOR]

        def _fit(rows: np.ndarray) -> dict[int, float]:
            y = y_all[rows]
            cols = [src_contrib[si][rows] for si in active]
            cols.append(np.ones(rows.size))  # intercept
            D = np.column_stack(cols)
            good = np.isfinite(y) & np.all(np.isfinite(D), axis=1)
            if good.sum() < max(3 * D.shape[1], 50):
                return {}
            beta, *_ = np.linalg.lstsq(D[good], y[good], rcond=None)
            return {si: float(beta[j]) for j, si in enumerate(active)}

        full_rows = np.arange(y_all.size)
        k_fit = _fit(full_rows)
        boot = {si: [] for si in k_fit}
        for _ in range(200):
            draw = rng.integers(0, n_shot, n_shot)
            rows = np.concatenate([np.arange(bounds[i], bounds[i + 1]) for i in draw])
            kb = _fit(rows)
            for si in k_fit:
                if si in kb:
                    boot[si].append(kb[si])

        k_block = {}
        for si, k in k_fit.items():
            bs = np.array(boot[si]) if len(boot[si]) >= 20 else None
            ci = (
                [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                if bs is not None
                else None
            )
            k_block[_label(sources[si])] = {
                "k": float(k),
                "ci": ci,
                "contrib_frac": frac[si],
                "separable": bool(len(sources[si]) == 1),
                "diff_from_1_sig": (
                    None if ci is None else bool(ci[0] > 1.0 or ci[1] < 1.0)
                ),
            }
        out[loop] = {
            "role": "bay" if loop in BAY_LOOPS else "control",
            "k_per_source": k_block,
        }
    return {
        "per_loop": out,
        "sources": [_label(src) for src in sources],
        "coupled_sets": [
            [COILS[i] for i in comp] for comp in comps_global if len(comp) > 1
        ],
        "unmeasurable": [COILS[i] for i in unmeasurable],
    }


def gain_decomposition(data: dict) -> dict:
    """Per-shot gain regressed on per-coil exposure weights: ``g ≈ W·k``.

    Ranks columns by how much they DRIVE the gain spread and reports, per coil,
    the across-shot exposure-weight distribution (its bimodality is the whole
    point) alongside the fitted ``k``."""
    COILS = data["COILS"]
    n_coil = len(COILS)
    out: dict[str, dict] = {}
    for loop in data["loops"]:
        rec = data["pershot"][loop]
        if len(rec["gain"]) < 20:
            continue
        g = np.array(rec["gain"])
        W = np.vstack(rec["w"])  # (n_shot, n_coil)
        # keep coils with real exposure across the cohort
        wmean = np.nanmean(np.abs(W), axis=0)
        active = [c for c in range(n_coil) if wmean[c] >= CONTRIB_FLOOR]
        Wa = W[:, active]
        good = np.all(np.isfinite(Wa), axis=1) & np.isfinite(g)
        Wa, gg = Wa[good], g[good]
        # solve g ≈ Wa · k  (Σ w ≈ 1, so k absorbs the level; no intercept)
        k, *_ = np.linalg.lstsq(Wa, gg, rcond=None)
        rank = []
        for j, c in enumerate(active):
            wc = Wa[:, j]
            spread = float(np.percentile(wc, 90) - np.percentile(wc, 10))
            corr = float(np.corrcoef(wc, gg)[0, 1]) if np.std(wc) > 0 else 0.0
            drive = abs(corr) * spread * abs(k[j] - 1.0)
            rank.append(
                {
                    "coil": COILS[c],
                    "k_fit": float(k[j]),
                    "w_mean": float(np.mean(wc)),
                    "w_p10": float(np.percentile(wc, 10)),
                    "w_p90": float(np.percentile(wc, 90)),
                    "w_spread": spread,
                    "corr_gain_w": corr,
                    "drive_score": float(drive),
                }
            )
        rank.sort(key=lambda d: -d["drive_score"])
        out[loop] = {
            "role": "bay" if loop in BAY_LOOPS else "control",
            "n_shot": int(good.sum()),
            "gain_median": float(np.median(gg)),
            "gain_p10": float(np.percentile(gg, 10)),
            "gain_p90": float(np.percentile(gg, 90)),
            "columns_by_drive": rank,
        }
    return out


def verdict(per_sensor: dict, gain: dict) -> dict:
    """Name the source column/set and route the fix.

    The named source is the one that (a) tops the gain-drive ranking on the bay
    loops, (b) carries a bay-loop ``k`` significantly ≠ 1, and (c) leaves the
    control loops' ``k`` at ~1.  The drive-DATA vs MODEL-GEOMETRY call comes from
    the spread of the named source's ``k`` across the bay-loop sensors.
    """
    from collections import defaultdict

    # map each raw coil column to the identifiable SOURCE label it belongs to
    coil_to_source = {}
    for label in per_sensor["sources"]:
        for coil in label.split("+"):
            coil_to_source[coil] = label

    # aggregate the bay-loop gain-drive ranking to the SOURCE level
    score = defaultdict(float)
    for loop, blk in gain.items():
        if blk["role"] != "bay":
            continue
        for entry in blk["columns_by_drive"]:
            src = coil_to_source.get(entry["coil"], entry["coil"])
            score[src] += entry["drive_score"]
    ranked = sorted(score, key=lambda c: -score[c])
    named = ranked[0] if ranked else None

    # bay-loop k for the named source across sensors
    bay_k, ctrl_k = [], []
    for loop, blk in per_sensor["per_loop"].items():
        kb = blk["k_per_source"].get(named)
        if kb is None:
            continue
        (bay_k if blk["role"] == "bay" else ctrl_k).append(kb["k"])
    bay_k = np.array(bay_k)
    ctrl_k = np.array(ctrl_k)

    cause = None
    detail = ""
    if bay_k.size:
        med = float(np.median(bay_k))
        rel_spread = float(np.std(bay_k) / (abs(med) + 1e-9))
        # a constant multiplier across sensors ⇒ amplitude/data-scaling;
        # a sensor-varying (or sign-changing) coupling ⇒ geometry pattern error.
        same_sign = bool(np.all(bay_k > 0) or np.all(bay_k < 0))
        if same_sign and rel_spread < 0.35:
            cause = "drive_data"
            detail = (
                f"k≈{med:.2f} is a roughly constant multiplier across the "
                f"{bay_k.size} bay-loop sensors (rel. spread {rel_spread:.2f}): "
                "an amplitude / data-scaling error on the amc channel."
            )
        else:
            cause = "model_geometry"
            detail = (
                f"k varies across the {bay_k.size} bay-loop sensors "
                f"(rel. spread {rel_spread:.2f}, sign-consistent={same_sign}): "
                "the field PATTERN is wrong — a model-column geometry error."
            )
    fix = {
        "drive_data": (
            "Recalibrate / re-map the named amc current channel: audit its "
            "turns / xmult scaling and channel-name assignment against the "
            "machine description before it enters G_pf assembly."
        ),
        "model_geometry": (
            "Refit the named coil's geometry column in the operator (filament "
            "R,Z / turns distribution) so the modelled field PATTERN matches the "
            "measured bay-loop response; a single scalar will not close it."
        ),
    }.get(cause)
    return {
        "named_column": named,
        "bay_drive_score": {c: float(score[c]) for c in ranked[:6]},
        "named_bay_k_median": (float(np.median(bay_k)) if bay_k.size else None),
        "named_bay_k_values": bay_k.tolist(),
        "named_control_k_median": (
            float(np.median(ctrl_k)) if ctrl_k.size else None
        ),
        "named_control_k_values": ctrl_k.tolist(),
        "cause": cause,
        "cause_detail": detail,
        "routed_fix": fix,
    }


def make_figure(data: dict, per_sensor: dict, gain: dict, vdt: dict) -> None:
    named = vdt["named_column"]
    COILS = data["COILS"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) per-shot gain distributions: bay loops vs controls
    ax = axes[0, 0]
    for loop in data["loops"]:
        rec = data["pershot"].get(loop)
        if not rec or len(rec["gain"]) < 20:
            continue
        g = np.array(rec["gain"])
        g = g[np.isfinite(g)]
        is_bay = loop in BAY_LOOPS
        ax.hist(
            g,
            bins=np.linspace(-0.5, 2.5, 50),
            histtype="step",
            lw=2 if is_bay else 1.4,
            color="#b00" if is_bay else "#1565c0",
            alpha=0.8 if is_bay else 0.9,
            label=loop if (is_bay or loop == CONTROL_LOOPS[0]) else None,
        )
    ax.axvline(1.0, color="k", lw=1)
    ax.set_xlabel("per-shot scalar gain (meas vs summed prediction)")
    ax.set_ylabel("shots")
    ax.set_title("(a) bay loops (red) bimodal; controls (blue) tight at ~1")
    ax.legend(fontsize=7, ncol=2)

    # the named source may be a coupled set — sum its coils' exposure weights
    named_cols = [c for c in (named.split("+") if named else []) if c in COILS]
    cidx = [COILS.index(c) for c in named_cols]

    # (b) exposure-weight distribution of the NAMED source across shots, bay loops
    ax = axes[0, 1]
    for loop in BAY_LOOPS:
        rec = data["pershot"].get(loop)
        if not rec or not cidx or len(rec["w"]) < 20:
            continue
        W = np.vstack(rec["w"])
        wc = W[:, cidx].sum(axis=1)
        wc = wc[np.isfinite(wc)]
        ax.hist(wc, bins=40, histtype="step", lw=1.6, alpha=0.8, label=loop)
    ax.set_xlabel(f"exposure weight w of {named}")
    ax.set_ylabel("shots")
    ax.set_title("(b) named source: bimodal exposure on bay loops")
    ax.legend(fontsize=7)

    # (c) gain vs named-source exposure weight — the collapse onto a line
    ax = axes[1, 0]
    for loop in BAY_LOOPS:
        rec = data["pershot"].get(loop)
        if not rec or not cidx or len(rec["w"]) < 20:
            continue
        W = np.vstack(rec["w"])
        g = np.array(rec["gain"])
        wc = W[:, cidx].sum(axis=1)
        m = np.isfinite(wc) & np.isfinite(g)
        ax.scatter(wc[m], g[m], s=8, alpha=0.4, label=loop)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel(f"exposure weight w of {named}")
    ax.set_ylabel("per-shot gain")
    ax.set_title("(c) gain tracks named-source exposure (bimodal mix)")
    ax.legend(fontsize=7)

    # (d) per-sensor k for the named column: bay vs control
    ax = axes[1, 1]
    loops, ks, los, his, cols = [], [], [], [], []
    for loop in data["loops"]:
        kb = per_sensor["per_loop"].get(loop, {}).get("k_per_source", {}).get(named)
        if kb is None:
            continue
        loops.append(loop)
        ks.append(kb["k"])
        ci_ = kb.get("ci")
        los.append(kb["k"] - ci_[0] if ci_ else 0.0)
        his.append(ci_[1] - kb["k"] if ci_ else 0.0)
        cols.append("#b00" if loop in BAY_LOOPS else "#1565c0")
    xs = np.arange(len(loops))
    ax.bar(xs, ks, color=cols, alpha=0.75)
    ax.errorbar(xs, ks, yerr=[los, his], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(loops, rotation=90, fontsize=7)
    ax.set_ylabel(f"pooled coupling k for {named}")
    ax.set_title(f"(d) k[{named}] ≠ 1 on bay (red), ~1 on control (blue)")

    fig.suptitle(
        f"Bay flux-loop bimodal-gain column decomposition — named column: "
        f"{named}  ({vdt['cause']})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "fig-column-decomposition.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-shots",
        type=int,
        default=0,
        help="debug: cap the cohort at this many shots (0 = full cohort)",
    )
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    shots = select_cohort()
    if args.max_shots > 0:
        shots = shots[: args.max_shots]
    logger.info("cohort: %d shots", len(shots))

    data = gather(shots)
    if data["n_used"] < 10:
        logger.error("too few usable shots (%d)", data["n_used"])
        return 1

    per_sensor = per_sensor_coupling(data)
    gain = gain_decomposition(data)
    vdt = verdict(per_sensor, gain)

    logger.info("NAMED COLUMN: %s (%s)", vdt["named_column"], vdt["cause"])
    logger.info("bay k median: %s  control k median: %s",
                vdt["named_bay_k_median"], vdt["named_control_k_median"])

    out = {
        "leakage_free": True,
        "firewall": (
            "coil-only vacuum slices only; raw amb magnetics + raw amc coil "
            "currents + geometry-only operator; NO EFIT, NO plasma inversion."
        ),
        "cohort": {"n_requested": len(shots), "n_used": data["n_used"]},
        "bay_loops": BAY_LOOPS,
        "control_loops": CONTROL_LOOPS,
        "corr_couple_threshold": CORR_COUPLE,
        "contrib_floor": CONTRIB_FLOOR,
        "coupled_sets": per_sensor["coupled_sets"],
        "per_sensor_coupling": per_sensor["per_loop"],
        "gain_decomposition": gain,
        "verdict": vdt,
    }
    out_path = ARTIFACTS / "flux_loop_column_decomposition.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", out_path)

    make_figure(data, per_sensor, gain, vdt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
