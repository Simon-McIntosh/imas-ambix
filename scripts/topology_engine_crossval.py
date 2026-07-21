"""Cross-validate the disc-field ENGINE read against EFIT, per topology class.

The census (:mod:`scripts.topology_census`) stratifies the corpus by EFIT-read
topology class; here the ENGINE's staged-disc inversion runs on the selected
slices — measured magnetics only, firewall intact, no EFIT inputs — and its
connectivity boundary read is scored against EFIT's reported reconstruction
per class.  Unlike the EFIT-ψ leg (same field, isolates the READ), this leg
carries the full engine solve, so residuals fold in the disc current model and
calibration — it is a CHARACTERISATION of the engine on real exotic
topologies, not a read-reproduction gate.

Verdict key (module glossary — code below is named by mechanism):
  T-F3  engine cross-validation.  Pre-declared split:
        * MUST PASS — the routine operating classes ``limited``, ``sn-lower``,
          ``connected-dn``: median LCFS-radii residual vs EFIT ≤ 5.0 cm and
          diverted/limited class agreement ≥ 0.80 (the shipped engine's
          flat-top skill leaves headroom to 5 cm; class flips near the
          marginal band are tolerated by the 0.80 floor).
        * RECORDED AS FINDINGS — ``sn-upper`` (rarer, known up/down solve
          asymmetry candidates), ``marginal-dn`` (sits inside the read's
          continuous limited↔diverted blend), axis and X-set residuals on all
          classes (the engine axis is an interior-null read with its own
          characterised offset).  Findings are reported per class, never
          averaged into a corpus mean.
        A class below the support floor after engine-side attrition (missing
        level-1 sensors / geometry) is reported with its achieved support.

Scoring frame: the engine boundary is read on the ENGINE grid (its own solve
domain); EFIT's LCFS polygon is rendered to radii about the ENGINE's read axis
so both sides describe the boundary from the same origin.

Usage:
    uv run python -m scripts.topology_engine_crossval \
        --census imas_ambix/latent/artifacts/patch_gate/topology_census-v0.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("topology_engine_crossval")

from scripts.topology_census import LEVEL2_SHOTS  # noqa: E402
from scripts.topology_efit_read_eval import (  # noqa: E402
    SUPPORT_FLOOR,
    _agg,
    _xset_match_cm,
    polygon_ray_radii,
    stratified_selection,
)

PASS_CLASSES = ("limited", "sn-lower", "connected-dn")
FINDING_CLASSES = ("sn-upper", "marginal-dn", "snowflake-candidate")

ENGINE_LCFS_TOL_CM = 5.0
ENGINE_CLASS_ACC_TOL = 0.80
TIME_MATCH_S = 0.02  # census slice ↔ engine payload time pairing window
MAX_PAYLOAD_SLICES = 200  # per-shot payload build ceiling (dense time cover)
MIN_IP_KA = 100.0


def engine_rows_for_shot(shot: int, recs, *, nr: int, nz: int) -> list[dict]:
    """Run the disc engine on one shot and score its read at the census times."""
    import zarr  # noqa: PLC0415

    from imas_ambix.latent.boundary_disc import disc_read  # noqa: PLC0415
    from imas_ambix.latent.connectivity_boundary import boundary_read  # noqa: PLC0415
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES  # noqa: PLC0415
    from scripts.heldout_mse_gate_eval import _campaign_table  # noqa: PLC0415
    from scripts.spine_label_factory import factory_shot_payloads  # noqa: PLC0415

    table = _campaign_table(int(shot))
    if table is None:
        return []
    payload = factory_shot_payloads(
        int(shot),
        nr=nr,
        nz=nz,
        max_slices=MAX_PAYLOAD_SLICES,
        min_ip_ka=MIN_IP_KA,
        table=table,
    )
    if payload is None:
        return []
    grid, tbl, basis = payload["grid"], payload["table"], payload["basis"]
    times = np.array([float(p.time_s) for p in payload["payloads"]])

    g = zarr.open_group(str(LEVEL2_SHOTS / f"{shot}.zarr"), mode="r")
    eq = g["equilibrium"]

    rows = []
    for rec in recs:
        t_ref = float(rec["time_s"])
        j = int(np.argmin(np.abs(times - t_ref)))
        if abs(times[j] - t_ref) > TIME_MATCH_S:
            continue
        p = payload["payloads"][j]
        try:
            inv = disc_read(p, grid, tbl, basis)
        except Exception:  # noqa: BLE001 — sweep on
            inv = None
        if inv is None or inv.ring is None:
            continue
        psi = np.asarray(inv.psi_tot, dtype=np.float64)
        centroid = (float(inv.centroid_r), float(inv.centroid_z))
        if not (np.isfinite(centroid[0]) and centroid[0] <= 1.4):
            continue
        try:
            eng = boundary_read(psi, grid, centroid, lcfs_norm=1.0)
        except ValueError:
            continue
        if not eng.found:
            continue
        k = int(rec["k"])
        lcfs = np.c_[
            np.asarray(eq["lcfs_r"][:, k], dtype=np.float64),
            np.asarray(eq["lcfs_z"][:, k], dtype=np.float64),
        ]
        lcfs = lcfs[np.isfinite(lcfs).all(axis=1) & (lcfs[:, 0] > 0)]
        if lcfs.shape[0] < 8:
            continue
        efit_radii = polygon_ray_radii(lcfs, eng.axis, LCFS_ANGLES)
        ok = np.isfinite(eng.radii) & np.isfinite(efit_radii)
        if not ok.any():
            continue
        dr_cm = 100.0 * np.abs(eng.radii[ok] - efit_radii[ok])
        efit_axis = (float(eq["magnetic_axis_r"][k]), float(eq["magnetic_axis_z"][k]))
        u_x = np.array(
            [
                (rec["u_x_lo"], rec["x_lo_r"], rec["x_lo_z"]),
                (rec["u_x_hi"], rec["x_hi_r"], rec["x_hi_z"]),
            ]
        )
        from scripts.topology_census import X_BIND_U  # noqa: PLC0415

        binding = np.abs(u_x[:, 0] - 1.0) <= X_BIND_U
        rows.append(
            {
                "shot": int(shot),
                "k": k,
                "time_s": t_ref,
                "ip_ka": float(rec["ip_ka"]),
                "radii_dmed_cm": float(np.median(dr_cm)),
                "radii_dmax_cm": float(np.max(dr_cm)),
                "axis_d_cm": 100.0
                * float(
                    np.hypot(eng.axis[0] - efit_axis[0], eng.axis[1] - efit_axis[1])
                ),
                "xset_d_cm": _xset_match_cm(eng.xset, u_x[binding][:, 1:]),
                "dev_is_diverted": bool(eng.is_diverted),
                "class_margin": float(np.clip(eng.class_margin, -1.0, 1.0)),
            }
        )
    return rows


def score_class(cname: str, recs: np.ndarray, *, nr: int, nz: int) -> dict:
    rows: list[dict] = []
    by_shot: dict[int, list] = {}
    for rec in recs:
        by_shot.setdefault(int(rec["shot"]), []).append(rec)
    for i, (shot, srecs) in enumerate(sorted(by_shot.items())):
        try:
            rows += engine_rows_for_shot(shot, srecs, nr=nr, nz=nz)
        except Exception as exc:  # noqa: BLE001 — engine attrition is reported
            logger.warning("  %s shot %d failed: %s", cname, shot, exc)
        if (i + 1) % 10 == 0:
            logger.info(
                "  %s: %d/%d shots, %d rows", cname, i + 1, len(by_shot), len(rows)
            )
    diverted_expected = cname != "limited"
    class_hits = [r["dev_is_diverted"] == diverted_expected for r in rows]
    return {
        "class": cname,
        "n_selected": int(recs.size),
        "n_scored": len(rows),
        "radii_dmed_cm": _agg([r["radii_dmed_cm"] for r in rows]),
        "axis_d_cm": _agg([r["axis_d_cm"] for r in rows]),
        "xset_d_cm": _agg([r["xset_d_cm"] for r in rows]),
        "class_agreement": float(np.mean(class_hits)) if class_hits else None,
        "rows": rows,
    }


def class_verdict(res: dict) -> dict:
    supported = res["n_scored"] >= SUPPORT_FLOOR
    if res["class"] not in PASS_CLASSES:
        return {"supported": supported, "gated": False, "pass": None}
    if not supported:
        return {"supported": False, "gated": True, "pass": None}
    checks = {
        "radii": res["radii_dmed_cm"]["med"] is not None
        and res["radii_dmed_cm"]["med"] <= ENGINE_LCFS_TOL_CM,
        "class": res["class_agreement"] is not None
        and res["class_agreement"] >= ENGINE_CLASS_ACC_TOL,
    }
    return {
        "supported": True,
        "gated": True,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--census",
        type=Path,
        default=Path("imas_ambix/latent/artifacts/patch_gate/topology_census-v0.npz"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "imas_ambix/latent/artifacts/patch_gate/topology_engine_crossval-v0.json"
        ),
    )
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--per-class", type=int, default=None, help="cap shots per class")
    args = ap.parse_args()
    rows = np.load(args.census)["rows"]
    sel = stratified_selection(rows)
    results, verdicts = {}, {}
    for cname, recs in sel.items():
        if args.classes and cname not in args.classes:
            continue
        if args.per_class and recs.size > args.per_class:
            idx = np.linspace(0, recs.size - 1, args.per_class).astype(int)
            recs = recs[idx]
        logger.info("class %s: %d selected", cname, recs.size)
        res = score_class(cname, recs, nr=args.nr, nz=args.nz)
        results[cname] = res
        verdicts[cname] = class_verdict(res)
        logger.info(
            "  scored %d — radii med %s cm, axis med %s cm, xset med %s cm, acc %s",
            res["n_scored"],
            res["radii_dmed_cm"]["med"],
            res["axis_d_cm"]["med"],
            res["xset_d_cm"]["med"],
            res["class_agreement"],
        )
    gated = [v for v in verdicts.values() if v.get("gated")]
    gate_pass = (
        bool(gated)
        and all(v["pass"] for v in gated if v["supported"])
        and any(v["supported"] for v in gated)
    )
    artifact = {
        "verdict_keys": {"T-F3": bool(gate_pass)},
        "tolerances": {
            "pass_classes": list(PASS_CLASSES),
            "lcfs_med_cm": ENGINE_LCFS_TOL_CM,
            "class_acc": ENGINE_CLASS_ACC_TOL,
            "support_floor": SUPPORT_FLOOR,
        },
        "verdicts": verdicts,
        "classes": {
            c: {k: v for k, v in r.items() if k != "rows"} for c, r in results.items()
        },
        "rows": {c: r["rows"] for c, r in results.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))
    logger.info("T-F3 %s → %s", "PASS" if gate_pass else "FAIL", args.out)


if __name__ == "__main__":
    main()
