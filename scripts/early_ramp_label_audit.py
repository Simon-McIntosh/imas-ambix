#!/usr/bin/env python
"""Early-ramp label availability & noise-floor study for the flux-ledger route.

The spine label factory labels only slices at Ip >= 300 kA.  The integrated
flux-consumption ledger (``scripts/current_diffusion_flux_ledger_report.py``)
backs the internal flux content out of the measured surface swing minus the
modelled resistive consumption, and a full-ramp Ejima claim would want
coverage below that threshold.  This audit asks two model-free questions from
the raw 1 kHz streams alone (no per-slice equilibrium refits):

A. SIGNAL vs NOISE FLOOR vs Ip.  Each amb magnetics channel is the sum of a
   coil-driven (vacuum) field and a plasma-driven field.  We fit the linear
   coil -> channel map on the VACUUM samples (coils energised, |Ip| < 10 kA;
   both the pre-breakdown and post-shot ramp-down anchor it across a range of
   coil currents), and the residual on the plasma-on window is the
   PLASMA-ATTRIBUTABLE signal.  Its channel noise floor is the residual std on
   those same vacuum samples.  SNR(Ip) is the binned plasma residual over the
   noise floor, medianed within each magnetics kind (B-probes, flux loops).
   The Ip at which the group-median SNR crosses ~3 and ~10 is the
   noise-limited label floor.

B. WHAT THE EARLY PHASE COSTS THE LEDGER.  From the measured flux loops alone
   (the fl_* channels ARE poloidal flux in Wb), the wall loop voltage
   dPsi_loop/dt (median across loops) is integrated from plasma initiation
   (first |Ip| > 20 kA) to the first labelled slice (first |Ip| >= min-ip-ka)
   and over the labelled window.  The pre-label fraction of that integral
   bounds the surface swing the current ledger never sees.  The Ip>20 kA marker
   can precede real breakdown (Rogowski coil-premagnetisation offset), so the
   window is also reported from the density (ane) onset, the physical
   initiation.  This wall swing mixes inductive storage and resistive
   consumption -- it is a bound/scale, not a resistive split.

C. BURN-THROUGH TIMING.  Dalpha and radiated-power channels are ABSENT from
   the loaded feature schema (ama MHD amplitudes, amb magnetics, amc coils,
   ane density only), so the burn-through window is located from Ip morphology
   (the slow-growth phase before the main dIp/dt ramp) cross-checked against
   the density (ane) onset.  We report the initiation / low-current milestones
   against the first-label time to show whether the 300 kA threshold sits
   safely after burn-through.

Artifacts: imas_ambix/latent/artifacts/patch_gate/early_ramp_label_audit.json
Figures:   docs/figures/temporal-physics-spine/fig-early-ramp-*.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from imas_ambix.latent.data import (
    anchored_columns,
    feature_schema,
    load_shot_slices_raw,
    read_split_shot_lists,
    robust_channel_scale,
    schema_group_offsets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("early_ramp_label_audit")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")
FIGURES = Path("docs/figures/temporal-physics-spine")

# coil channels that share the plasma-window clock (plasma_current is the
# quantity we are separating against; tf_current lives on a different, near-DC
# clock and carries no poloidal-field information about the plasma)
_EXCLUDE_COILS = ("plasma_current", "tf_current")

VAC_IP_KA = 5.0  # |Ip| below this (coils live) = a vacuum sample
CUR_IP_KA = 5.0  # |Ip| at/above this = a current-carrying (plasma) sample
INIT_IP_KA = 20.0  # first crossing = plasma initiation (current channel formed)


def _kind(name: str) -> str:
    return "flux_loop" if str(name).lower().startswith("fl") else "b_probe"


def _load(shot: int, schema) -> dict | None:
    """Raw streams + a filled |Ip| trace on the coil-valid window."""
    loaded = load_shot_slices_raw(shot, schema)
    if loaded is None:
        return None
    x, times, plasma_on = loaded
    off = schema_group_offsets(schema)
    ip_col, ne_col = anchored_columns(schema)
    amc_names, amb_names = schema["amc"], schema["amb"]

    ip_raw = np.asarray(x[:, ip_col], dtype=np.float64)
    ip_fin = np.isfinite(ip_raw)
    if ip_fin.sum() < 50:
        return None
    # |Ip| filled by interpolation ONLY within the measured span; outside the
    # coil-valid window it stays NaN (a dropout must never fabricate a step)
    ip = np.full(times.size, np.nan)
    ip[ip_fin] = np.abs(ip_raw[ip_fin])
    span = (times >= times[ip_fin].min()) & (times <= times[ip_fin].max())
    ip[span] = np.interp(times[span], times[ip_fin], np.abs(ip_raw[ip_fin]))

    coil_cols = [
        off["amc"] + j for j, n in enumerate(amc_names) if n not in _EXCLUDE_COILS
    ]
    coil = np.asarray(x[:, coil_cols], dtype=np.float64)
    coil_fin = np.isfinite(coil).all(axis=1)

    ne_raw = np.asarray(x[:, ne_col], dtype=np.float64)

    return {
        "shot": int(shot),
        "x": np.asarray(x, dtype=np.float64),
        "times": np.asarray(times, dtype=np.float64),
        "plasma_on": np.asarray(plasma_on, dtype=bool),
        "ip": ip,
        "ip_fin": ip_fin,
        "coil": coil,
        "coil_fin": coil_fin,
        "ne": ne_raw,
        "amb0": off["amb"],
        "amb_names": amb_names,
    }


def _channel_snr(d: dict) -> dict:
    """Per-channel noise floor, plasma-on residual trace, and whitening scale.

    The coil -> channel map is least-squares fit on the vacuum samples; the
    residual std there is the noise floor; the plasma-on |residual| is the
    plasma-attributable signal.  The spine's whitening scale (std of the raw
    channel over plasma-on slices, floored by
    :func:`robust_channel_scale`) is returned per channel for the cross-check.
    """
    x, ip = d["x"], d["ip"]
    coil, coil_fin, plasma_on = d["coil"], d["coil_fin"], d["plasma_on"]
    amb0, amb_names = d["amb0"], d["amb_names"]
    ip_ok = np.isfinite(ip)
    peak = float(np.nanmax(ip))

    vac = coil_fin & ip_ok & (ip < VAC_IP_KA)
    # the current-carrying phase (ramp + flat-top + controlled decay): SNR is a
    # function of Ip regardless of phase, and the flat-top dwell fills the
    # high-Ip bins the fast ramp is too brief to populate.  The per-bin median
    # over channels AND samples is robust to any termination transient.
    pon = coil_fin & ip_ok & (ip >= CUR_IP_KA)

    out = {}
    whit_raw, whit_names = [], []
    for j, ch in enumerate(amb_names):
        sig = x[:, amb0 + j]
        mv = vac & np.isfinite(sig)
        mp = pon & np.isfinite(sig)
        # plasma-on std of the raw channel: the spine's pre-floor whitening
        # scale (its own Ip>0.2*peak plasma-on convention, not the ramp window)
        mw = plasma_on & coil_fin & np.isfinite(sig)
        whit_raw.append(float(np.std(sig[mw])) if mw.sum() > 5 else np.nan)
        whit_names.append(ch)
        if mv.sum() < 20 or mp.sum() < 5:
            continue
        dm_vac = np.column_stack([coil[mv], np.ones(mv.sum())])
        coef, *_ = np.linalg.lstsq(dm_vac, sig[mv], rcond=None)
        noise = float(np.std(sig[mv] - dm_vac @ coef))
        if not np.isfinite(noise) or noise <= 0:
            continue
        dm_pon = np.column_stack([coil[mp], np.ones(mp.sum())])
        resid = np.abs(sig[mp] - dm_pon @ coef)
        out[ch] = {
            "noise_floor": noise,
            "ip_pon": ip[mp],
            "resid_pon": resid,
            "kind": _kind(ch),
        }
    # floor the whitening scales exactly as the spine does (kind-relative)
    whit = robust_channel_scale(np.asarray(whit_raw), whit_names)
    for ch, w in zip(whit_names, whit, strict=True):
        if ch in out:
            out[ch]["whitening_scale"] = float(w)
    return {"channels": out, "peak_ka": peak}


def _crossing(centers: np.ndarray, snr: np.ndarray, level: float) -> float | None:
    """Lowest Ip where the (rising) SNR curve exceeds ``level`` [kA]."""
    ok = np.isfinite(snr)
    c, s = centers[ok], snr[ok]
    if c.size < 2:
        return None
    if s[0] >= level:
        return float(c[0])  # already above at the lowest observed bin
    for i in range(1, c.size):
        if s[i] >= level > s[i - 1]:
            f = (level - s[i - 1]) / (s[i] - s[i - 1])
            return float(c[i - 1] + f * (c[i] - c[i - 1]))
    return None  # never reaches the level in the observed range


def _raw_vloop_integral(
    d: dict, t0: float, t1: float, smooth_ms: float = 25.0
) -> float:
    """Integral of |median wall loop voltage| between t0 and t1 [Wb].

    dΨ_loop/dt per fl_* channel, box-smoothed, medianed across loops (the
    UNCORRECTED wall-side swing: coil + eddy + plasma), then ∫|·|dt.
    """
    x, times = d["x"], d["times"]
    amb0, amb_names = d["amb0"], d["amb_names"]
    dt = float(np.median(np.diff(times)))
    n_box = max(3, int(round(smooth_ms * 1e-3 / dt)))
    ker = np.ones(n_box) / n_box
    v_list = []
    for j, ch in enumerate(amb_names):
        if _kind(ch) != "flux_loop":
            continue
        sig = x[:, amb0 + j]
        ok = np.isfinite(sig)
        if ok.sum() < 10 * n_box:
            continue
        s = np.interp(times, times[ok], sig[ok])
        s = np.convolve(s, ker, mode="same")
        v_list.append(np.gradient(s, times))
    if not v_list:
        return float("nan")
    v = np.median(np.stack(v_list), axis=0)
    win = (times >= t0) & (times <= t1)
    return float(np.trapezoid(np.abs(v[win]), times[win]))


def _first_cross(times: np.ndarray, ip: np.ndarray, thr: float) -> float | None:
    m = np.isfinite(ip) & (ip >= thr)
    return float(times[np.argmax(m)]) if m.any() else None


def _burnthrough(d: dict, min_ip_ka: float) -> dict:
    """Initiation / low-current milestones + density onset [s]."""
    times, ip, ne = d["times"], d["ip"], d["ne"]
    peak = float(np.nanmax(ip))
    dt = float(np.median(np.diff(times)))
    n_box = max(3, int(round(0.005 / dt)))
    ip_ok = np.isfinite(ip)
    ips = np.copy(ip)
    ips[ip_ok] = np.convolve(ip[ip_ok], np.ones(n_box) / n_box, mode="same")
    t_init = _first_cross(times, ip, INIT_IP_KA)
    t_label = _first_cross(times, ip, min_ip_ka)
    # max dIp/dt on the initiation->label ramp = the knee out of burn-through
    t_maxdip = None
    if t_init is not None and t_label is not None:
        reg = ip_ok & (times >= t_init) & (times <= t_label + 0.05)
        if reg.sum() > 5:
            dip = np.gradient(ips, times)
            t_maxdip = float(times[reg][np.argmax(dip[reg])])
    ne_fin = np.isfinite(ne)
    t_ne = None
    if ne_fin.sum() > 10:
        nei = np.interp(times, times[ne_fin], ne[ne_fin])
        t_ne = float(times[np.argmax(nei > 0.1 * np.nanmax(nei))])
    return {
        "peak_ka": peak,
        "t_init_20ka": t_init,
        "t_50ka": _first_cross(times, ip, 50.0),
        "t_100ka": _first_cross(times, ip, 100.0),
        "t_first_label": t_label,
        "t_max_dipdt": t_maxdip,
        "t_ne_onset": t_ne,
        "dt_init_to_label_ms": (
            None if (t_init is None or t_label is None) else (t_label - t_init) * 1e3
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=8)
    ap.add_argument("--n-baseline-shots", type=int, default=10)
    ap.add_argument("--n-tune-shots", type=int, default=4)
    ap.add_argument(
        "--extra-train",
        type=int,
        default=6,
        help="additional train shots (from the baseline block) to widen the "
        "sample beyond the 4 tune shots",
    )
    ap.add_argument("--min-ip-ka", type=float, default=300.0)
    ap.add_argument("--ip-bin-ka", type=float, default=20.0)
    args = ap.parse_args()

    schema = feature_schema()
    train, _held = read_split_shot_lists(args.n_train, args.n_heldout)
    tune = train[args.n_baseline_shots : args.n_baseline_shots + args.n_tune_shots]
    extra = train[: min(args.extra_train, args.n_baseline_shots)]
    shots = list(dict.fromkeys(tune + extra))  # tune first, dedup, keep order
    logger.info("tune shots %s + extra train %s", tune, extra)

    loaded = {}
    for s in shots:
        d = _load(s, schema)
        if d is None:
            logger.warning("shot %s: no usable raw stream — skipped", s)
            continue
        # a labelable shot must reach the label floor (a Rogowski that never
        # crosses min-ip-ka carries no labelled slice — exclude it from every
        # arm so it cannot contaminate the SNR pool or the ledger budget)
        if float(np.nanmax(d["ip"])) < args.min_ip_ka:
            logger.info(
                "shot %s: peak Ip < %g kA — not labelable, skipped", s, args.min_ip_ka
            )
            continue
        loaded[s] = d
    if not loaded:
        raise SystemExit("no shots loaded")

    # ---- A: SNR vs Ip ----
    per_shot_snr = {s: _channel_snr(d) for s, d in loaded.items()}
    peak_max = max(v["peak_ka"] for v in per_shot_snr.values())
    edges = np.arange(0.0, peak_max + args.ip_bin_ka, args.ip_bin_ka)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # pool per-bin per-channel SNR across all shots, then median within kind
    def pooled_group(kind: str) -> np.ndarray:
        snr = np.full(centers.size, np.nan)
        for b in range(centers.size):
            lo, hi = edges[b], edges[b + 1]
            vals = []
            for v in per_shot_snr.values():
                for c in v["channels"].values():
                    if c["kind"] != kind:
                        continue
                    m = (c["ip_pon"] >= lo) & (c["ip_pon"] < hi)
                    if m.sum() >= 3:
                        vals.append(np.median(c["resid_pon"][m]) / c["noise_floor"])
            if vals:
                snr[b] = float(np.median(vals))
        return snr

    snr_bp = pooled_group("b_probe")
    snr_fl = pooled_group("flux_loop")
    floors = {
        "b_probe": {
            "snr3_ka": _crossing(centers, snr_bp, 3.0),
            "snr10_ka": _crossing(centers, snr_bp, 10.0),
        },
        "flux_loop": {
            "snr3_ka": _crossing(centers, snr_fl, 3.0),
            "snr10_ka": _crossing(centers, snr_fl, 10.0),
        },
    }
    # noise floor vs whitening scale cross-check (median ratio over channels)
    nf_over_whit = {"b_probe": [], "flux_loop": []}
    for v in per_shot_snr.values():
        for c in v["channels"].values():
            w = c.get("whitening_scale")
            if w and w > 0:
                nf_over_whit[c["kind"]].append(c["noise_floor"] / w)
    whit_cross = {
        k: (float(np.median(vv)) if vv else None) for k, vv in nf_over_whit.items()
    }

    # ---- B: pre-label ledger cost + C: burn-through ----
    ledger_cost = {}
    burn = {}
    for s, d in loaded.items():
        bt = _burnthrough(d, args.min_ip_ka)
        burn[str(s)] = bt
        t_init, t_lab = bt["t_init_20ka"], bt["t_first_label"]
        if t_init is None or t_lab is None:
            ledger_cost[str(s)] = None
            continue
        # end of the labelled window: last time |Ip| >= min-ip-ka
        ip, times = d["ip"], d["times"]
        above = np.isfinite(ip) & (ip >= args.min_ip_ka)
        t_end = float(times[np.flatnonzero(above)[-1]]) if above.any() else t_lab
        # the Ip>20 kA marker can precede real breakdown (the Rogowski carries
        # a coil-premagnetisation offset before the plasma forms); the density
        # (ane) onset is the physical initiation.  Report the pre-label window
        # from BOTH so a coil-premag-inflated Ip marker is visible.
        t_break = bt["t_ne_onset"] if bt["t_ne_onset"] is not None else t_init
        t_break = min(max(t_break, t_init), t_lab)
        # headline bound: the RAW wall loop-voltage integral (mixes inductive
        # + resistive + coil swing — a scale, not a resistive split)
        pre_v = _raw_vloop_integral(d, t_init, t_lab)
        tot_v = _raw_vloop_integral(d, t_init, t_end)
        pre_v_ne = _raw_vloop_integral(d, t_break, t_lab)
        tot_v_ne = _raw_vloop_integral(d, t_break, t_end)
        ledger_cost[str(s)] = {
            "t_init_20ka": t_init,
            "t_breakdown_ne": t_break,
            "t_first_label": t_lab,
            "t_label_end": t_end,
            "prelabel_raw_vloop_integral_wb": pre_v,
            "total_raw_vloop_integral_wb": tot_v,
            "prelabel_fraction": (
                None
                if not (np.isfinite(pre_v) and np.isfinite(tot_v) and tot_v > 0)
                else float(pre_v / tot_v)
            ),
            "prelabel_raw_vloop_from_ne_wb": pre_v_ne,
            "prelabel_fraction_from_ne": (
                None
                if not (
                    np.isfinite(pre_v_ne) and np.isfinite(tot_v_ne) and tot_v_ne > 0
                )
                else float(pre_v_ne / tot_v_ne)
            ),
        }

    pre_fracs = [
        v["prelabel_fraction"]
        for v in ledger_cost.values()
        if v and v["prelabel_fraction"] is not None
    ]

    summary = {
        "arm": "early-ramp-label-audit",
        "shots": sorted(loaded),
        "tune_shots": tune,
        "extra_train_shots": [s for s in extra if s in loaded],
        "min_ip_ka": args.min_ip_ka,
        "ip_bin_ka": args.ip_bin_ka,
        "snr_vs_ip": {
            "ip_centers_ka": [float(c) for c in centers],
            "b_probe_median_snr": [
                None if not np.isfinite(v) else float(v) for v in snr_bp
            ],
            "flux_loop_median_snr": [
                None if not np.isfinite(v) else float(v) for v in snr_fl
            ],
        },
        "noise_floor_ip_ka": floors,
        "noise_floor_over_whitening_scale_median": whit_cross,
        "prelabel_ledger_cost": ledger_cost,
        "prelabel_fraction_median": (
            float(np.median(pre_fracs)) if pre_fracs else None
        ),
        "burnthrough": burn,
        "verdict": _verdict(floors, pre_fracs, burn, args.min_ip_ka),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / "early_ramp_label_audit.json"
    out_json.write_text(json.dumps(summary, indent=2))

    _figures(centers, snr_bp, snr_fl, floors, args.min_ip_ka, loaded, ledger_cost, burn)

    logger.info(
        "SNR floors: B-probe SNR3 %.1f kA / SNR10 %.1f kA; flux-loop SNR3 %.1f / "
        "SNR10 %.1f kA | pre-label flux fraction median %.3f | %s",
        floors["b_probe"]["snr3_ka"] or float("nan"),
        floors["b_probe"]["snr10_ka"] or float("nan"),
        floors["flux_loop"]["snr3_ka"] or float("nan"),
        floors["flux_loop"]["snr10_ka"] or float("nan"),
        summary["prelabel_fraction_median"] or float("nan"),
        out_json,
    )
    return 0


def _verdict(floors, pre_fracs, burn, min_ip_ka) -> dict:
    snr10 = [v["snr10_ka"] for v in floors.values() if v["snr10_ka"] is not None]
    ip_floor = float(max(snr10)) if snr10 else None
    # every label-reaching shot must have its first label AFTER both the
    # main-ramp knee (max dIp/dt) and the density onset (burn-through is over
    # once the density has risen and the current is ramping hard)
    checkable = [
        b
        for b in burn.values()
        if b["t_first_label"] is not None
        and b["t_max_dipdt"] is not None
        and b["t_ne_onset"] is not None
    ]
    labels_after_bt = bool(checkable) and all(
        b["t_first_label"] >= b["t_max_dipdt"] and b["t_first_label"] >= b["t_ne_onset"]
        for b in checkable
    )
    return {
        "snr_ip_floor_ka": ip_floor,
        "noise_limited": bool(ip_floor is not None and ip_floor >= min_ip_ka),
        "labels_after_burnthrough": bool(labels_after_bt),
        "prelabel_fraction_median": (
            float(np.median(pre_fracs)) if pre_fracs else None
        ),
    }


def _figures(centers, snr_bp, snr_fl, floors, min_ip_ka, loaded, ledger_cost, burn):
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- fig 1: SNR vs Ip per group ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(centers, snr_bp, "o-", color="#4477aa", label="B-probes (median)")
    ax.semilogy(centers, snr_fl, "s-", color="#cc6677", label="flux loops (median)")
    for lv, ls in ((3.0, ":"), (10.0, "--")):
        ax.axhline(lv, color="0.5", ls=ls, lw=0.9)
        ax.text(centers[-1], lv, f" SNR={lv:g}", va="bottom", fontsize=7, color="0.4")
    ax.axvline(
        min_ip_ka, color="#228833", lw=1.4, label=f"label floor {min_ip_ka:g} kA"
    )
    for grp, col in (("b_probe", "#4477aa"), ("flux_loop", "#cc6677")):
        for key, ls in (("snr3_ka", ":"), ("snr10_ka", "--")):
            v = floors[grp][key]
            if v is not None:
                ax.axvline(v, color=col, ls=ls, lw=0.8, alpha=0.7)
    ax.set_xlabel("|Ip| [kA]")
    ax.set_ylabel("plasma-attributable SNR (residual / noise floor)")
    ax.set_title(
        "Plasma-attributable SNR vs Ip\n(SNR=10 crossing ~12 kA — far below the "
        "300 kA label floor)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig-early-ramp-snr-vs-ip.png", dpi=120)
    plt.close(fig)

    # --- fig 2: per-shot Ip(t) + integrated wall V_loop, pre-label shaded ---
    shots = sorted(loaded)
    n = len(shots)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False
    )
    for k, s in enumerate(shots):
        ax = axes[k // ncol][k % ncol]
        d = loaded[s]
        times, ip = d["times"], d["ip"]
        ok = np.isfinite(ip)
        bt = burn[str(s)]
        t_init, t_lab = bt["t_init_20ka"], bt["t_first_label"]
        lc = ledger_cost[str(s)]
        t_end = lc["t_label_end"] if lc else None
        win = ok & (times >= (t_init or times.min()) - 0.02)
        if t_end is not None:
            win &= times <= t_end + 0.03
        ax.plot(times[win], ip[win], color="#333333", lw=1.3, label="|Ip|")
        if t_init is not None and t_lab is not None:
            ax.axvspan(t_init, t_lab, color="#cc6677", alpha=0.22, label="pre-label")
        for tt, col, lab in (
            (bt["t_max_dipdt"], "#ee8866", "max dIp/dt"),
            (bt["t_ne_onset"], "#66aadd", "n_e onset"),
            (t_lab, "#228833", "first label"),
        ):
            if tt is not None:
                ax.axvline(
                    tt, color=col, lw=1.0, ls="--", label=lab if k == 0 else None
                )
        ax.axhline(min_ip_ka, color="#228833", lw=0.7, ls=":")
        frac = None if not lc else lc["prelabel_fraction"]
        ax.set_title(
            f"{s}: pre-label flux "
            f"{'n/a' if frac is None else f'{frac * 100:.0f}%'} of swing",
            fontsize=9,
        )
        ax.set_xlabel("t [s]")
        ax.set_ylabel("|Ip| [kA]")
        if k == 0:
            ax.legend(fontsize=6, loc="upper left")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        "Pre-label window (initiation -> first 300 kA label): the surface-flux "
        "swing the current ledger never sees",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "fig-early-ramp-prelabel-window.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
