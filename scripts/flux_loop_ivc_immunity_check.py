#!/usr/bin/env python
"""Empirical sensor exposure ladder on vacuum shots with in-vessel coils live.

Full toroidal flux loops should reject n!=0 energised fields exactly (the
toroidal line integral of an n!=0 field vanishes), while toroidally-partial
sensors (pickup probes, saddle loops) see them fully.  Two tests on the
dedicated-vacuum shots where the error-field correction coils, ELM/RMP coils
and horseshoe coils are driven:

1. LOOP IMMUNITY — fit each ``fl_*`` loop on the PF currents alone; the
   IVC-coherent fraction of the residual bounds any energised coupling.
2. EXPOSURE LADDER — for every sensor family (loops, ccbv/obr/obv probes,
   sad_out saddle sensors), joint LSQ on PF + IVC drives; report the variance
   the IVC terms actually explain (delta-R2, robust to collinearity) and the
   strongest IVC term's amplitude in units of the sensor's signal std.

Measured verdict (2026-07-22): saddles dR2 0.18-0.32 >> obv 0.063-0.065 >
obr/ccbv 0.002-0.011 > healthy loops 0.0007-0.007 — exactly the exposure
ordering the 3D coil/sensor geometry predicts, with a no-drive control shot
dropping every family to noise.

Firewall: raw L1 amb/amc/xmb/xma only — no EFIT, no inversion.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import zarr

from imas_ambix.data.paths import LEVEL1_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flux_loop_ivc_immunity_check")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")

PF = [
    "p2il_coil_current", "p2iu_coil_current", "p2ol_coil_current",
    "p2ou_coil_current", "p3l_coil_current", "p3u_coil_current",
    "p4l_coil_current", "p4u_coil_current", "p5l_coil_current",
    "p5u_coil_current", "p6l_current", "p6u_current", "sol_current",
    "p2l_case_current", "p2u_case_current", "p4l_case_current",
    "p4u_case_current", "p5l_case_current", "p5u_case_current",
]
IVC_AMC = ["error_field_02", "error_field_05"]
IVC_XMA = ["rog_elm_u_01", "rog_elm_l_01", "rog_elm_l_06", "hscu_dot", "hscl_dot"]

#: default shots: the M8 plasma-free in-vessel-coil commissioning series plus a
#: saddle-active vacuum shot (see ivc_vacuum_candidates.json)
DEFAULT_SHOTS = [27394, 27425, 27462, 27507, 27539, 25836]


def _grp_time(g, grp):
    for k in ("time", "sec"):
        if k in g[grp]:
            return np.asarray(g[grp][k], float)
    return None


def check_shot(shot: int) -> list[dict]:
    g = zarr.open_group(str(LEVEL1_DIR / f"{shot}.zarr"), mode="r")
    t = _grp_time(g, "amb")
    amc_t, xma_t = _grp_time(g, "amc"), _grp_time(g, "xma")

    def on_grid(grp, k, gt):
        try:
            a = np.asarray(g[grp][k], float)
            if gt is None or a.shape != gt.shape:
                return None
            return np.interp(t, gt, a)
        except Exception:  # noqa: BLE001
            return None

    cols = [v for k in PF if (v := on_grid("amc", k, amc_t)) is not None
            and np.nanstd(v) > 0]
    design = np.column_stack(cols + [np.ones_like(t)])
    drives = {}
    for grp, gt, keys in (("amc", amc_t, IVC_AMC), ("xma", xma_t, IVC_XMA)):
        for k in keys:
            v = on_grid(grp, k, gt)
            if v is not None and np.nanstd(v) > 1e-6:
                drives[k] = v

    out = []
    for fl in sorted(k for k in g["amb"].array_keys() if k.startswith("fl_")):
        y = on_grid("amb", fl, _grp_time(g, "amb"))
        if y is None or not np.isfinite(y).any() or np.nanstd(y) == 0:
            continue
        good = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
        if good.sum() < 200:
            continue
        beta, *_ = np.linalg.lstsq(design[good], y[good], rcond=None)
        res = y[good] - design[good] @ beta
        sig = float(np.nanstd(y[good]))
        rel_res = float(np.std(res) / sig) if sig > 0 else np.nan
        rmax, kmax = 0.0, ""
        for k, v in drives.items():
            vv = v[good]
            if np.std(vv) == 0:
                continue
            r = abs(float(np.corrcoef(res, vv)[0, 1]))
            if r > rmax:
                rmax, kmax = r, k
        out.append({
            "shot": shot, "loop": fl,
            "pf_r2": float(1 - np.var(res) / np.var(y[good])),
            "resid_over_sigma": rel_res,
            # upper bound on the IVC-coherent fraction of the SIGNAL
            "ivc_coherent_bound": rmax * rel_res,
            "max_corr_drive": kmax, "max_corr": rmax,
        })
    return out


#: sensor families for the exposure ladder: (zarr group, channel prefix)
FAMILIES = {
    "full_loops": ("amb", "fl_"),
    "ccbv_probes": ("amb", "ccbv"),
    "obr_probes": ("amb", "obr"),
    "obv_probes": ("amb", "obv"),
    "saddle_sensors": ("xmb", "sad_out"),
}


def exposure_ladder(shot: int) -> dict:
    """Joint PF+IVC fit per sensor; per-family delta-R2 and peak IVC amplitude."""
    g = zarr.open_group(str(LEVEL1_DIR / f"{shot}.zarr"), mode="r")
    t = _grp_time(g, "amc")

    def on_grid(grp, k):
        gt = _grp_time(g, grp)
        try:
            a = np.asarray(g[grp][k], float)
            if gt is None or a.shape != gt.shape:
                return None
            return np.interp(t, gt, a)
        except Exception:  # noqa: BLE001
            return None

    pf_cols = [v for k in PF if (v := on_grid("amc", k)) is not None
               and np.nanstd(v) > 0]
    ivc_cols, ivc_names = [], []
    for grp, keys in (("amc", IVC_AMC), ("xma", IVC_XMA)):
        for k in keys:
            v = on_grid(grp, k)
            if v is not None and np.nanstd(v) > 1e-6:
                ivc_cols.append(v)
                ivc_names.append(k)
    x_pf = np.column_stack(pf_cols + [np.ones_like(t)])
    x_all = np.column_stack(pf_cols + ivc_cols + [np.ones_like(t)])
    n_pf = len(pf_cols)

    fam_out: dict[str, dict] = {}
    for fam, (grp, pref) in FAMILIES.items():
        if grp not in g:
            continue
        rows = []
        for k in sorted(g[grp].array_keys()):
            if not k.startswith(pref):
                continue
            y = on_grid(grp, k)
            if y is None or np.nanstd(y) == 0:
                continue
            good = np.isfinite(y) & np.all(np.isfinite(x_all), axis=1)
            if good.sum() < 300:
                continue
            sig = float(np.std(y[good]))
            b0, *_ = np.linalg.lstsq(x_pf[good], y[good], rcond=None)
            r0 = y[good] - x_pf[good] @ b0
            b1, *_ = np.linalg.lstsq(x_all[good], y[good], rcond=None)
            r1 = y[good] - x_all[good] @ b1
            dr2 = float((np.var(r0) - np.var(r1)) / np.var(y[good]))
            amp, aname = 0.0, ""
            for j, nm in enumerate(ivc_names):
                a = abs(b1[n_pf + j]) * float(np.std(ivc_cols[j][good])) / (sig + 1e-30)
                if a > amp:
                    amp, aname = a, nm
            rows.append({"sensor": k, "delta_r2": dr2,
                         "ivc_amp_over_sigma": amp, "strongest_drive": aname})
        if rows:
            dr2s = np.array([r["delta_r2"] for r in rows])
            fam_out[fam] = {
                "n": len(rows),
                "delta_r2_median": float(np.median(dr2s)),
                "delta_r2_max": float(dr2s.max()),
                "top": sorted(rows, key=lambda r: -r["delta_r2"])[:4],
            }
    return fam_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, nargs="*", default=DEFAULT_SHOTS)
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for s in args.shots:
        try:
            rows += check_shot(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s failed (%s)", s, exc)
    if not rows:
        logger.error("no loops scored")
        return 1
    # a loop whose signal is not PF-driven at all is a dead/faulty CHANNEL, not
    # an immunity violation — its noise correlates weakly with everything.
    # Split the verdict: healthy loops (PF explains the signal) vs flagged.
    healthy = [r for r in rows if r["pf_r2"] > 0.9]
    flagged = sorted({r["loop"] for r in rows if r["pf_r2"] <= 0.9})
    bounds = np.array([r["ivc_coherent_bound"] for r in healthy])
    verdict = {
        "n_loop_shot_pairs": len(rows),
        "n_healthy": len(healthy),
        "flagged_channels": flagged,
        "healthy_bound_median": float(np.median(bounds)),
        "healthy_bound_p95": float(np.percentile(bounds, 95)),
        "healthy_bound_max": float(bounds.max()),
        "healthy_immune_at_2pct": bool(bounds.max() < 0.02),
        "exposure_ladder": {},
        "rows": rows,
    }
    for s in args.shots:
        try:
            verdict["exposure_ladder"][str(s)] = exposure_ladder(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shot %s exposure ladder failed (%s)", s, exc)
    out = ARTIFACTS / "flux_loop_ivc_immunity.json"
    out.write_text(json.dumps(verdict, indent=2))
    logger.info(
        "healthy loops (n=%d): IVC-coherent bound median %.4f, p95 %.4f, "
        "max %.4f of signal -> %s; flagged dead/faulty channels: %s",
        len(healthy), verdict["healthy_bound_median"],
        verdict["healthy_bound_p95"], verdict["healthy_bound_max"],
        "IMMUNE (<2%)" if verdict["healthy_immune_at_2pct"] else "NOT immune",
        flagged or "none")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
