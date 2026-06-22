"""Stage the ait divertor heat-flux as a SIGNAL stream (PREPARED, held).

The ait source is the divertor IR heat-flux ANALYSIS (not raw IR camera frames —
those are ~13 real frames and are skipped).  Per shot it carries, on its own
``time`` axis (~4.6k samples), the strike-point divertor traces the lead named:

  * ``etot_{isp,osp}`` / ``etotsum_{isp,osp}`` — energy to the inner / outer
    strike point (and ELM-resolved ``*_elm`` variants),
  * ``lampowpp_{isp,osp}`` / ``lampowsol_{isp,osp}`` — power-balance / SOL lambda
    power-decay widths,
  * ``ptot_{isp,osp}`` / ``pkpower_density_{isp,osp}`` /
    ``peakpower_pos_{isp,osp}`` — total + peak heat-flux density and its position,
  * ``temperature_{isp,osp}`` — strike-point surface temperature,
  * ``qprofile_{isp,osp}`` / ``tprofile_{isp,osp}`` — the (time, 186) heat-flux /
    temperature PROFILES along ``rcoord_{isp,osp}``.

These are time-resolved at a moderate cadence (NOT the MHz raw rate the HF
phase-tokeniser targets), so ait fits the L2 measured-signal pattern (like
``summary_l2`` / ``pf_active_l2`` the world-model v2 signal loader already
conditions on), NOT ``signal_hf_encode``.  This script extracts the ait traces
into a per-shot signal Zarr keyed by the ait ``time`` axis — the staging the
downstream uniform-quantiser tokeniser / conditioning loader consumes.

This is CPU work (a read + reshape, no model), so it can run on ``sun`` — but it
is PREPARED and HELD per the lead until the encode slot is signalled (kept here
beside the GPU encodes for a single coordinated launch).  The leakage stance:
ait is a DIAGNOSTIC measurement (divertor heat flux), an admissible conditioning
/ probe stream — NOT a reconstruction (efm/esm/xdc), so it does not trip the
leakage ban.

Per-shot 1-D traces kept (scalar-per-time-step); the 2-D ``qprofile``/
``tprofile`` are kept whole (time, 186) so a later loader can sub-sample chords.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr

from imas_ambix.camdyn.dataset import level1_shot_path

#: ait 1-D divertor traces (scalar per time step) — the strike-point signal set.
AIT_TRACE_KEYS: tuple[str, ...] = (
    "etot_isp",
    "etot_osp",
    "etotsum_isp",
    "etotsum_osp",
    "etot_isp_elm",
    "etot_osp_elm",
    "lampowpp_isp",
    "lampowpp_osp",
    "lampowsol_isp",
    "lampowsol_osp",
    "ptot_isp",
    "ptot_osp",
    "pkpower_density_isp",
    "pkpower_density_osp",
    "peakpower_pos_isp",
    "peakpower_pos_osp",
    "temperature_isp",
    "temperature_osp",
    "satpixels_isp",
    "satpixels_osp",
)
#: ait 2-D profiles (time, R) + their R axes.
AIT_PROFILE_KEYS: tuple[str, ...] = (
    "qprofile_isp",
    "qprofile_osp",
    "tprofile_isp",
    "tprofile_osp",
)
AIT_RCOORD_KEYS: tuple[str, ...] = ("rcoord_isp", "rcoord_osp")

#: Where staged ait signal streams land (sibling of the other signal stores,
#: NOT under the camera token root, NOT under the eval-only target root).
DEFAULT_AIT_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/signals-ait")


def stage_ait_shot(shot_id: int, ait_root: Path) -> tuple[bool, str]:
    """Extract one shot's ait divertor traces into a per-shot signal Zarr.

    Returns ``(ok, reason)``.  Skips shots with no ait group or no time axis.
    """
    try:
        g = zarr.open_group(str(level1_shot_path(int(shot_id))), mode="r")
    except Exception as exc:  # noqa: BLE001
        return False, f"open: {exc}"
    if "ait" not in set(g.group_keys()):
        return False, "no ait"
    ait = g["ait"]
    ak = set(ait.array_keys())
    if "time" not in ak:
        return False, "no time axis"
    t = np.asarray(ait["time"], dtype=np.float64)
    if t.size < 2:
        return False, "degenerate time"

    out_path = ait_root / "ait" / str(shot_id) / "ait.zarr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(out_path), mode="w")
    store.create_array("time", data=t.astype(np.float32))

    kept_traces = []
    for k in AIT_TRACE_KEYS:
        if k in ak:
            v = np.asarray(ait[k], dtype=np.float32)
            if v.ndim == 1 and v.shape[0] == t.shape[0]:
                store.create_array(k, data=v)
                kept_traces.append(k)
    kept_profiles = []
    for k in AIT_PROFILE_KEYS:
        if k in ak:
            v = np.asarray(ait[k], dtype=np.float32)
            if v.ndim == 2 and v.shape[0] == t.shape[0]:
                store.create_array(k, data=v)
                kept_profiles.append(k)
    for k in AIT_RCOORD_KEYS:
        if k in ak:
            store.create_array(k, data=np.asarray(ait[k], dtype=np.float32))

    store.attrs.update(
        {
            "shot_id": int(shot_id),
            "source": "ait",
            "kind": "divertor_heat_flux_signal",
            "n_time": int(t.shape[0]),
            "traces": kept_traces,
            "profiles": kept_profiles,
        }
    )
    if not kept_traces and not kept_profiles:
        return False, "no usable traces"
    return True, ""


def _ait_candidate_shots() -> list[int]:
    from imas_ambix.data.paths import LEVEL1_DIR

    return sorted(
        int(p.name[:-5])
        for p in Path(LEVEL1_DIR).glob("*.zarr")
        if p.name[:-5].isdigit()
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ait-root", default=str(DEFAULT_AIT_ROOT))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-shots", type=int, default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    shots = _ait_candidate_shots()
    n_shards = max(1, int(args.n_shards))
    shard = int(args.shard) % n_shards
    if n_shards > 1:
        shots = shots[shard::n_shards]
    if args.max_shots:
        shots = shots[: args.max_shots]
    print(
        f"[ait shard {shard}/{n_shards}] staging ait for {len(shots)} candidate shots",
        flush=True,
    )

    ait_root = Path(args.ait_root)
    n_ok = 0
    skips: dict[str, int] = {}
    t0 = time.monotonic()
    for i, sid in enumerate(shots):
        ok, reason = stage_ait_shot(sid, ait_root)
        if ok:
            n_ok += 1
        else:
            skips[reason] = skips.get(reason, 0) + 1
        if (i + 1) % 500 == 0:
            print(
                f"[ait]   {i + 1}/{len(shots)} done, {n_ok} staged, "
                f"{time.monotonic() - t0:.0f}s",
                flush=True,
            )
    summary = {
        "shard": shard,
        "n_shards": n_shards,
        "n_candidates": len(shots),
        "staged": n_ok,
        "skips": skips,
        "ait_root": str(ait_root),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
