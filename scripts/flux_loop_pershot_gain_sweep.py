"""Per-shot vacuum gain trajectory for the bay flux loops + EF current stats.

Tests step-vs-ramp at the RMP install boundaries (19031, 25404): a passive-
structure mechanism steps there; instrument aging / processing drift ramps
smoothly.  Also characterises the error-field correction currents corpus-wide
(availability, amplitude, within-era variation, vacuum-window presence)."""
import json
import sys

import numpy as np
import zarr

import imas_ambix.gs.geometry as _geom
from imas_ambix.data.description_reader import read_geometry_table
from imas_ambix.data.paths import LEVEL1_DIR
from imas_ambix.gs.operator import build_operator

# tolerate the late-campaign amm hole (passive geometry unused here)
_orig = _geom.read_amm_passive
def _amm_opt(shot):
    try: return _orig(shot)
    except Exception: return []
_geom.read_amm_passive = _amm_opt

from scripts.vacuum_coil_response_audit import _shot_coil_only

TRACK = ["fl_p4l_1","fl_p4l_4","fl_p4u_4","fl_p5l_1","fl_p5l_4","fl_p5u_1",
         "fl_p3u_4","fl_cc03","fl_cc05","obr10","obv09"]

ref = build_operator(read_geometry_table(11774))
CH = list(ref.sensor_channels); COILS = list(ref.pf_amc_channels)
G = np.asarray(ref.g_pf, float)
IDX = {c: CH.index(c) for c in TRACK if c in CH}

def ef_stats(shot):
    try:
        g = zarr.open_group(str(LEVEL1_DIR/f"{shot}.zarr"), mode="r")
        amc = g["amc"]
        out = {}
        for names,tag in ((("error_field_a","error_field_02"),"ef_a"),
                          (("error_field_b","error_field_05"),"ef_b")):
            for k in names:
                if k in amc:
                    a = np.asarray(amc[k], float)
                    out[tag] = {"max": float(np.nanmax(np.abs(a))),
                                "std": float(np.nanstd(a)), "channel": k}
                    break
        return out
    except Exception:
        return {}

def per_shot(shot):
    r = _shot_coil_only(shot, CH, COILS)
    if r is None: return None
    meas, ipf = r["meas"], r["i_pf"]
    if meas.shape[0] < 100: return None
    pred = ipf @ G.T
    gains = {}
    for name,i in IDX.items():
        y, x = meas[:, i], pred[:, i]
        good = np.isfinite(y) & np.isfinite(x)
        if good.sum() < 100 or np.ptp(x[good]) < 1e-12: continue
        a = np.polyfit(x[good], y[good], 1)
        res = y[good] - np.polyval(a, x[good])
        keep = np.abs(res - np.median(res)) <= 3*(np.std(res)+1e-30)
        if keep.sum() >= 50: a = np.polyfit(x[good][keep], y[good][keep], 1)
        gains[name] = float(a[0])
    return {"shot": shot, "n_slices": int(meas.shape[0]), "gains": gains,
            "ef": ef_stats(shot)}

def main():
    ids = json.loads(open("/work/projects/imas_gpu/mast/manifests/level1-all.json").read())["shot_ids"]
    ids = sorted(int(s) for s in ids if 11695 <= int(s) <= 30473)
    a = np.array(ids)
    sel = set()
    for b in (19031, 25404):                      # dense boundary windows
        w = a[(a >= b-600) & (a <= b+600)]
        step = max(1, len(w)//150)
        sel.update(int(s) for s in w[::step])
    u = a[::max(1, len(a)//220)]                  # uniform backbone
    sel.update(int(s) for s in u)
    shots = sorted(sel)
    print(f"sweeping {len(shots)} shots", file=sys.stderr, flush=True)
    out = []
    for i,s in enumerate(shots):
        try:
            r = per_shot(s)
            if r: out.append(r)
        except Exception:
            pass
        if i % 40 == 0:
            print(f"{i}/{len(shots)} done, {len(out)} valid", file=sys.stderr, flush=True)
    json.dump(out, open("imas_ambix/latent/artifacts/patch_gate/flux_loop_pershot_gains.json","w"))
    print(f"DONE {len(out)}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
