#!/usr/bin/env python
"""Find plasma-less intervals with in-vessel coils energised.

Toroidally-localised active coils (the in-vessel RMP/ELM coils, outer saddle
coils, error-field correction coils) give each magnetic sensor a spatially
distinct signature — the field diversity that breaks the sign/permutation
degeneracy in sensor disambiguation (Hole et al., Disambiguation of magnetic
sensors in ITER).  The archived axisymmetric-PF vacuum shots cannot do this, but
MAST does record the in-vessel coil currents (``xma/rog_elm_*`` ELM/RMP
Rogowskis, ``xmb/sad_out_*`` outer saddle, ``xmb/halo_elm_*``, ``xma/hsc*``
horseshoe, ``amc/error_field_02``/``_05``).

This scans every RMP-era L1 shot for the strongest seam: dedicated-vacuum shots
(peak |Ip| < 50 kA, plasma never forms) with an in-vessel coil driven.  For such
a shot the whole record is a clean plasma-less interval, so the per-channel peak
is the energised-in-vacuum value.  It also reports the pervasive n=5 error-field
correction current (energised in the pre-breakdown window of ordinary shots).

Firewall: raw L1 amc/xma/xmb only — no EFIT, no inversion.
Artifact: imas_ambix/latent/artifacts/patch_gate/ivc_vacuum_candidates.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import zarr

from imas_ambix.data.paths import LEVEL1_DIR, MANIFEST_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scan_invessel_coil_vacuum_activity")

ARTIFACTS = Path("imas_ambix/latent/artifacts/patch_gate")

VAC_PEAK_KA = 50.0
RMP_LO, RMP_HI = 19031, 30474  # M7 start .. end of archive
XMA_ELM = (
    [f"rog_elm_l_{i:02d}" for i in range(1, 13)]
    + [f"rog_elm_u_{i:02d}" for i in range(1, 13, 2)]
    + ["hscu_dot", "hscl_dot"]
)
XMB_ELM = (
    [f"sad_out_u{i:02d}" for i in range(1, 13)]
    + [f"sad_out_l{i:02d}" for i in range(1, 13)]
    + [f"sad_out_m{i:02d}" for i in range(1, 13)]
    + [f"halo_elm_u_{i}" for i in range(1, 5)]
    + [f"halo_elm_l_{i}" for i in range(1, 5)]
)


def _era(shot: int) -> str:
    for name, lo, hi in (("M7", 19031, 25404), ("M8", 25404, 28390), ("M9", 28390, 30474)):
        if lo <= shot < hi:
            return name
    return "?"


def _peak(g, grp: str, k: str) -> float | None:
    try:
        a = np.asarray(g[grp][k], float)
        return float(np.nanmax(np.abs(a))) if a.size else 0.0
    except Exception:  # noqa: BLE001
        return None


def scan(shot: int) -> dict | None:
    p = LEVEL1_DIR / f"{shot}.zarr"
    if not p.exists():
        return None
    try:
        g = zarr.open_group(str(p), mode="r")
        if "amc" not in g or "plasma_current" not in g["amc"]:
            return None
        ip = np.abs(np.asarray(g["amc"]["plasma_current"], float))
    except Exception:  # noqa: BLE001
        return None
    if ip.size < 4:
        return None
    peak = float(np.nanmax(ip))
    rec = {"shot": shot, "peak_ip_ka": peak, "ef2": _peak(g, "amc", "error_field_02"),
           "ef5": _peak(g, "amc", "error_field_05"), "vacuum": peak < VAC_PEAK_KA}
    if peak < VAC_PEAK_KA:
        elm = {}
        for grp, keys in (("xma", XMA_ELM), ("xmb", XMB_ELM)):
            if grp not in g:
                continue
            for k in keys:
                m = _peak(g, grp, k)
                if m:
                    elm[f"{grp}/{k}"] = m
        rec["ivc_top"] = sorted(elm.items(), key=lambda kv: -kv[1])[:8]
        rec["n_ivc_present"] = len(elm)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=str,
                    default=str(MANIFEST_DIR / "level1-all.json"))
    ap.add_argument("--drive-floor", type=float, default=1.0,
                    help="raw-unit threshold for 'in-vessel coil driven'")
    ap.add_argument("--max-candidates", type=int, default=120)
    args = ap.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    ids = json.loads(Path(args.manifest).read_text()).get("shot_ids", [])
    shots = sorted(s for s in (int(x) for x in ids) if RMP_LO <= s <= RMP_HI)
    logger.info("scanning %d RMP-era shots", len(shots))

    recs = []
    for i, s in enumerate(shots):
        r = scan(s)
        if r is not None:
            recs.append(r)
        if i % 500 == 0:
            n_vac = sum(1 for r in recs if r["vacuum"])
            logger.info("%d/%d scanned, %d valid, %d vacuum", i, len(shots), len(recs), n_vac)

    def topval(r):
        t = r.get("ivc_top", [])
        return t[0][1] if t else 0.0

    vac_ivc = [r for r in recs if r["vacuum"] and r.get("n_ivc_present", 0) > 0]
    strong = sorted((r for r in vac_ivc if topval(r) > args.drive_floor),
                    key=topval, reverse=True)
    from collections import Counter

    ledger = {
        "description": (
            "Dedicated-vacuum (peak|Ip|<50kA) RMP-era shots with an in-vessel coil "
            "(RMP/ELM Rogowski, outer saddle, ELM-halo, horseshoe) driven above the "
            "floor in raw pickup/Rogowski volts (some channels rail near 10). The "
            "plasma-less toroidally-asymmetric-field intervals for high-confidence "
            "sensor disambiguation. Firewall: L1 amc/xma/xmb only."),
        "n_scanned_valid": len(recs),
        "n_vacuum_shots_total": sum(1 for r in recs if r["vacuum"]),
        "n_vacuum_with_ivc": len(vac_ivc),
        "drive_floor": args.drive_floor,
        "n_strong": len(strong),
        "era_dist_strong": dict(Counter(_era(r["shot"]) for r in strong)),
        "error_field_05_median": float(np.median(
            [r["ef5"] for r in recs if r["ef5"] is not None])),
        "candidates": [
            {"shot": r["shot"], "era": _era(r["shot"]),
             "peak_ip_ka": round(r["peak_ip_ka"], 2), "ef5": r["ef5"],
             "n_ivc": r["n_ivc_present"],
             "top": [(k, round(v, 3)) for k, v in r.get("ivc_top", [])[:4]]}
            for r in strong[: args.max_candidates]
        ],
    }
    out = ARTIFACTS / "ivc_vacuum_candidates.json"
    out.write_text(json.dumps(ledger, indent=2))
    logger.info("wrote %s: %d strong candidates %s",
                out, ledger["n_strong"], ledger["era_dist_strong"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
