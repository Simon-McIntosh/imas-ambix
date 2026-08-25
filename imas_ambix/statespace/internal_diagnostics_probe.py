"""Evidence probes for `internal-current-diagnostics-v0`.

This module turns two plan claims into reproducible artifacts:

1. the decisive held-out Thompson cohort
   ``held_out_112 ∩ usable-near-axis atm Thomson == 45 shots``; and
2. the CXRS (`act`) assessment needed for the secondary-constraint note:
   Ti is available, Zeff is not obviously exposed in level-1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.data.paths import MANIFEST_DIR, local_shot_path
from imas_ambix.statespace.fast_loader import read_act_shot

logger = logging.getLogger(__name__)

MAST_R0 = 0.85
DEFAULT_ATM_RADIUS_TOL = 0.10


def _open_group(shot_zarr_path: Path, group: str):
    import zarr  # noqa: PLC0415

    grp_path = shot_zarr_path / group
    if not grp_path.exists():
        return None
    try:
        return zarr.open_group(str(grp_path), mode="r")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cannot open %s/%s: %s", shot_zarr_path.name, group, exc)
        return None


def atm_has_usable_near_axis(
    pe: np.ndarray,
    radius: np.ndarray,
    *,
    r0: float = MAST_R0,
    radius_tol: float = DEFAULT_ATM_RADIUS_TOL,
) -> bool:
    """True when `atm` carries any finite pressure sample near the magnetic axis."""

    pe_arr = np.asarray(pe, dtype=np.float64)
    rad_arr = np.asarray(radius, dtype=np.float64)
    if pe_arr.ndim != 2:
        return False
    if rad_arr.ndim == 1:
        if rad_arr.size != pe_arr.shape[1]:
            return False
        near = np.abs(rad_arr - float(r0)) <= float(radius_tol)
        return bool(near.any() and np.isfinite(pe_arr[:, near]).any())
    if rad_arr.ndim == 2 and rad_arr.shape == pe_arr.shape:
        near = np.abs(rad_arr - float(r0)) <= float(radius_tol)
        return bool(np.isfinite(pe_arr[near]).any())
    return False


def act_key_assessment(keys: set[str]) -> dict[str, object]:
    """Assess whether `act` exposes Ti and/or Zeff-like quantities."""

    lower = {k.lower() for k in keys}
    zeff = sorted(k for k in keys if "zeff" in k.lower() or "effective" in k.lower())
    ti = any("temperature" in k for k in lower)
    vel = any("velocity" in k for k in lower)
    counts = any("cx_counts" in k for k in lower)
    return {
        "ti_available": bool(ti),
        "velocity_available": bool(vel),
        "cx_counts_available": bool(counts),
        "zeff_available": bool(zeff),
        "zeff_key_candidates": zeff,
    }


@dataclass
class InternalDiagnosticsProbeResult:
    held_out_shots: list[int]
    atm_near_axis_shots: list[int]
    act_present_shots: list[int]
    act_present_on_atm_cohort: list[int]
    act_key_assessment: dict[str, object]
    atm_radius_tol_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "internal-diagnostics-probe-v0",
            "held_out_total": len(self.held_out_shots),
            "atm_near_axis_radius_tol_m": self.atm_radius_tol_m,
            "atm_near_axis_count": len(self.atm_near_axis_shots),
            "atm_near_axis_shot_ids": self.atm_near_axis_shots,
            "act_present_count": len(self.act_present_shots),
            "act_present_shot_ids": self.act_present_shots,
            "act_present_on_atm_cohort_count": len(self.act_present_on_atm_cohort),
            "act_present_on_atm_cohort_shot_ids": self.act_present_on_atm_cohort,
            "act_assessment": self.act_key_assessment,
        }


def probe_internal_diagnostics(
    *,
    manifest_path: Path | None = None,
    out_path: Path | None = None,
    radius_tol: float = DEFAULT_ATM_RADIUS_TOL,
    act_system: str = "c_pla",
) -> dict[str, object]:
    """Probe the internal-diagnostics corpus claims and optionally persist them."""

    manifest_path = manifest_path or (MANIFEST_DIR / "mse_heldout_split_v0.json")
    manifest = json.loads(manifest_path.read_text())
    held_out = [
        int(k) for k, v in manifest["shots"].items() if v.get("partition") == "held_out"
    ]

    atm_hits: list[int] = []
    act_hits: list[int] = []
    act_hits_on_atm: list[int] = []
    act_keys_union: set[str] = set()

    for sid in held_out:
        shot_path = local_shot_path(int(sid), tier="level1")
        store = _open_group(shot_path, "atm")
        atm_ok = False
        if store is not None and {"pe", "radius"}.issubset(set(store.array_keys())):
            pe = np.asarray(store["pe"])
            radius = np.asarray(store["radius"])
            atm_ok = atm_has_usable_near_axis(pe, radius, radius_tol=radius_tol)
            if atm_ok:
                atm_hits.append(int(sid))

        act_grp = _open_group(shot_path, "act")
        act = read_act_shot(shot_path, system=act_system)
        if act_grp is not None:
            act_keys_union.update(set(act_grp.array_keys()))
        if act is not None and act.avail_mask.get("temperature", False):
            act_hits.append(int(sid))
            if atm_ok:
                act_hits_on_atm.append(int(sid))

    result = InternalDiagnosticsProbeResult(
        held_out_shots=held_out,
        atm_near_axis_shots=atm_hits,
        act_present_shots=act_hits,
        act_present_on_atm_cohort=act_hits_on_atm,
        act_key_assessment=act_key_assessment(act_keys_union),
        atm_radius_tol_m=float(radius_tol),
    )
    payload = result.to_dict()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=float))
    return payload


def main() -> None:
    import argparse  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(
        description="Probe internal-current-diagnostics evidence"
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DIR / "mse_heldout_split_v0.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent
        / "artifacts"
        / "internal_diagnostics_probe_v0.json",
    )
    ap.add_argument("--radius-tol", type=float, default=DEFAULT_ATM_RADIUS_TOL)
    ap.add_argument("--act-system", default="c_pla")
    args = ap.parse_args()
    payload = probe_internal_diagnostics(
        manifest_path=args.manifest,
        out_path=args.out,
        radius_tol=args.radius_tol,
        act_system=args.act_system,
    )
    print(
        f"INTERNAL_DIAGNOSTICS_DONE atm_count={payload['atm_near_axis_count']} "
        f"act_count={payload['act_present_count']} out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
