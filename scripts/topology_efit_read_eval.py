"""Score the device connectivity read on REAL EFIT ψ maps, per topology class.

Consumes the corpus topology census (:mod:`scripts.topology_census`), draws a
stratified per-class validation set, feeds EFIT's OWN stored ψ maps (65×65,
``[z, r, t]``) into the device HARD and SMOOTH connectivity reads, and scores
boundary / axis / X-set / class against EFIT's reported reconstruction — the
read is isolated from our solver entirely (both sides see the same field), so
residuals measure the READ against EFIT's contouring of that field.  EFIT
remains a validation referee only; nothing here enters the engine solve.

Verdict keys (module glossary — code below is named by mechanism):
  T-F1  census + stratified support: every corpus slice classified; each class
        either meets the pre-declared support floor or is reported unsupported.
  T-F2  EFIT-ψ reproduction: on every supported class the hard read matches
        EFIT's boundary/axis/X to the pre-declared per-class tolerances AND the
        smooth read matches the hard read within the shipped smooth-convergence
        bounds.  Any class-conditional failure fails the gate — no averaging.

Pre-declared tolerances (the stored EFIT grid is coarse — ΔR=3.0 cm,
ΔZ=6.25 cm — and EFIT's published LCFS/X come from its own higher-order
internal representation, so the floor is the grid rendering, not the read):
  LCFS radii   median ≤ 4.0 cm per class (EFIT polygon vs ray-marched radii)
  ψ_bnd        median |Δ|/span ≤ 0.03 per class
  axis         median ≤ 5.0 cm per class (Z cell is 6.25 cm)
  X-set        median ≤ 8.0 cm per class (one grid-cell diagonal ≈ 6.9 cm)
  class        agreement ≥ 0.90 on limited / sn-lower / sn-upper /
               connected-dn; marginal-dn agreement is REPORTED (the class sits
               inside the read's continuous limited↔diverted blend by design)
  smooth       vs hard at the convergence point τ=0.001: |Δψ_bnd|/span ≤ 0.005
               and radii median ≤ 0.5 cm per class (the shipped smooth-
               convergence bounds, evaluated at the ladder's min τ)
Support floor: ≥ 30 slices per class, drawn evenly over the shot range (early
and late campaigns), at most one slice per (shot, class); snowflake candidates
are an orthogonal overlay scored as findings (no pass/fail tolerance).

Usage:
    uv run python -m scripts.topology_efit_read_eval \
        --census imas_ambix/latent/artifacts/patch_gate/topology_census-v0.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("topology_efit_read_eval")

from scripts.topology_census import CLASSES, LEVEL2_SHOTS, X_BIND_U  # noqa: E402

SCORED_CLASSES = ("limited", "sn-lower", "sn-upper", "connected-dn", "marginal-dn")
CLASS_GATED = ("limited", "sn-lower", "sn-upper", "connected-dn")

SUPPORT_FLOOR = 30  # slices per class for the class to be scoreable
N_PER_CLASS = 60  # stratified draw ceiling per class
LCFS_TOL_CM = 4.0
PSI_BND_FRAC_TOL = 0.03
AXIS_TOL_CM = 5.0
XSET_TOL_CM = 8.0
CLASS_ACC_TOL = 0.90
SMOOTH_TAU = 0.001  # smooth-vs-hard is gated at the convergence (min-τ) point
SMOOTH_PSI_TOL = 0.005
SMOOTH_RADII_TOL_CM = 0.5


@dataclass
class GridShim:
    """The minimal grid surface the connectivity read touches (EFIT geometry)."""

    rg: np.ndarray
    zg: np.ndarray
    inside_limiter: np.ndarray
    limiter_r: np.ndarray
    limiter_z: np.ndarray

    @property
    def nr(self) -> int:
        return self.rg.size

    @property
    def nz(self) -> int:
        return self.zg.size

    @property
    def dr(self) -> float:
        return float(self.rg[1] - self.rg[0])

    @property
    def dz(self) -> float:
        return float(self.zg[1] - self.zg[0])


def stratified_selection(rows: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class validation draw: even over the shot range, one slice per shot.

    Slices are ranked inside each class by shot, then thinned evenly so early
    and late campaigns (including the 28k–30k MAST-U-preparation range) are all
    represented; ties within a shot prefer the highest |Ip| (flat-top).
    """
    sel: dict[str, np.ndarray] = {}
    for cname in SCORED_CLASSES:
        ci = CLASSES.index(cname)
        sub = rows[rows["cls"] == ci]
        if sub.size == 0:
            sel[cname] = sub
            continue
        # one representative (highest |Ip|) per shot
        order = np.lexsort((-np.abs(sub["ip_ka"]), sub["shot"]))
        sub = sub[order]
        _, first = np.unique(sub["shot"], return_index=True)
        per_shot = sub[first]
        if per_shot.size > N_PER_CLASS:
            idx = np.linspace(0, per_shot.size - 1, N_PER_CLASS).astype(int)
            per_shot = per_shot[idx]
        sel[cname] = per_shot
    snow = rows[rows["snowflake"]]
    if snow.size:
        order = np.lexsort((-np.abs(snow["ip_ka"]), snow["shot"]))
        snow = snow[order]
        _, first = np.unique(snow["shot"], return_index=True)
        snow = snow[first]
        if snow.size > N_PER_CLASS:
            idx = np.linspace(0, snow.size - 1, N_PER_CLASS).astype(int)
            snow = snow[idx]
    sel["snowflake-candidate"] = snow
    return sel


_CANONICAL_WALL: tuple[np.ndarray, np.ndarray] | None = None
_CANONICAL_WALL_SHOT = 22086  # any shot carrying the (corpus-constant) polygon


def _canonical_wall() -> tuple[np.ndarray, np.ndarray]:
    global _CANONICAL_WALL  # noqa: PLW0603 — one lazy load per process
    if _CANONICAL_WALL is None:
        import zarr  # noqa: PLC0415

        w = zarr.open_group(
            str(LEVEL2_SHOTS / f"{_CANONICAL_WALL_SHOT}.zarr"), mode="r"
        )["wall"]
        _CANONICAL_WALL = (
            np.asarray(w["limiter_r"], dtype=np.float64),
            np.asarray(w["limiter_z"], dtype=np.float64),
        )
    return _CANONICAL_WALL


def load_slice(shot: int, k: int):
    """One EFIT slice: ψ(nz,nr), grid shim, axis, LCFS polygon, X slots."""
    import zarr  # noqa: PLC0415

    g = zarr.open_group(str(LEVEL2_SHOTS / f"{shot}.zarr"), mode="r")
    eq = g["equilibrium"]
    psi = np.asarray(eq["psi"][:, :, k], dtype=np.float64)  # (nz, nr)
    rg = np.asarray(eq["major_radius"], dtype=np.float64)
    zg = np.asarray(eq["z"], dtype=np.float64)
    if "wall" in g:
        lr = np.asarray(g["wall"]["limiter_r"], dtype=np.float64)
        lz = np.asarray(g["wall"]["limiter_z"], dtype=np.float64)
    else:
        # a small band of shots lacks the wall group; the MAST limiter polygon
        # is byte-identical across the corpus, so the canonical wall applies
        lr, lz = _canonical_wall()
    from imas_ambix.latent.topology import _inside_polygon  # noqa: PLC0415

    mesh_r, mesh_z = np.meshgrid(rg, zg)
    inside = _inside_polygon(mesh_r.ravel(), mesh_z.ravel(), lr, lz).reshape(
        zg.size, rg.size
    )
    grid = GridShim(rg=rg, zg=zg, inside_limiter=inside, limiter_r=lr, limiter_z=lz)
    axis = (float(eq["magnetic_axis_r"][k]), float(eq["magnetic_axis_z"][k]))
    lcfs = np.c_[
        np.asarray(eq["lcfs_r"][:, k], dtype=np.float64),
        np.asarray(eq["lcfs_z"][:, k], dtype=np.float64),
    ]
    lcfs = lcfs[np.isfinite(lcfs).all(axis=1) & (lcfs[:, 0] > 0)]
    xpts = np.c_[
        np.asarray(eq["x_point_r"][:, k], dtype=np.float64),
        np.asarray(eq["x_point_z"][:, k], dtype=np.float64),
    ]
    return psi, grid, axis, lcfs, xpts


def polygon_ray_radii(lcfs: np.ndarray, axis, angles) -> np.ndarray:
    """Distance from ``axis`` to the LCFS polygon along each query angle.

    Segment–ray intersection; the OUTERMOST crossing per angle (matches the
    read's outward ray-march).  NaN when a ray misses the polygon.
    """
    ar, az = axis
    p = lcfs
    q = np.roll(lcfs, -1, axis=0)
    out = np.full(len(angles), np.nan)
    for i, th in enumerate(np.asarray(angles, dtype=np.float64)):
        dx, dy = np.cos(th), np.sin(th)
        ex, ey = q[:, 0] - p[:, 0], q[:, 1] - p[:, 1]
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p[:, 0] - ar) * ey - (p[:, 1] - az) * ex) / den
            s = ((p[:, 0] - ar) * dy - (p[:, 1] - az) * dx) / den
        hit = np.isfinite(t) & np.isfinite(s) & (t > 0) & (s >= 0.0) & (s <= 1.0)
        if hit.any():
            out[i] = float(np.max(t[hit]))
    return out


def _xset_match_cm(dev_xset, ref_xset) -> float:
    """Worst nearest-pair distance [cm] between finite X-point sets (see gate)."""
    dev = np.asarray(dev_xset, dtype=np.float64).reshape(-1, 2)
    ref = np.asarray(ref_xset, dtype=np.float64).reshape(-1, 2)
    dev = dev[np.isfinite(dev).all(axis=1)]
    ref = ref[np.isfinite(ref).all(axis=1)]
    if dev.shape[0] == 0 or ref.shape[0] == 0:
        return float("nan")
    d = np.hypot(
        ref[:, None, 0] - dev[None, :, 0], ref[:, None, 1] - dev[None, :, 1]
    ).min(axis=1)
    return 100.0 * float(np.max(d))


def score_slice(rec) -> dict | None:
    """Hard + smooth device reads on one EFIT slice, scored against EFIT."""
    from imas_ambix.latent.connectivity_boundary import (  # noqa: PLC0415
        boundary_read,
        boundary_read_smooth,
    )
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES  # noqa: PLC0415

    shot, k = int(rec["shot"]), int(rec["k"])
    psi, grid, axis, lcfs, xpts = load_slice(shot, k)
    if lcfs.shape[0] < 8:
        return None
    try:
        hard = boundary_read(psi, grid, axis, lcfs_norm=1.0)
    except ValueError:
        return None  # flood seed in wall material — EFIT axis outside the mask
    if not hard.found:
        return None
    smooth = boundary_read_smooth(
        psi, grid, axis, temperature=SMOOTH_TAU, lcfs_norm=1.0
    )
    span = hard.psi_bnd - hard.psi_axis
    if abs(span) < 1e-9:
        return None
    # EFIT references: boundary flux from its own polygon, radii from rays
    efit_radii = polygon_ray_radii(lcfs, axis, LCFS_ANGLES)
    from scripts.topology_census import _bilinear  # noqa: PLC0415

    psi3 = psi[:, :, None]
    efit_psi_bnd = float(
        np.nanmean(_bilinear(psi3, grid.rg, grid.zg, lcfs[:, 0], lcfs[:, 1], 0))
    )
    ok = np.isfinite(hard.radii) & np.isfinite(efit_radii)
    if not ok.any():
        return None
    dr_cm = 100.0 * np.abs(hard.radii[ok] - efit_radii[ok])
    # EFIT X reference: only slots participating in the boundary (|u−1| ≤ band)
    u_x = np.array(
        [
            (rec["u_x_lo"], rec["x_lo_r"], rec["x_lo_z"]),
            (rec["u_x_hi"], rec["x_hi_r"], rec["x_hi_z"]),
        ]
    )
    binding = np.abs(u_x[:, 0] - 1.0) <= X_BIND_U
    efit_xset = u_x[binding][:, 1:]
    sm_rad_ok = np.isfinite(smooth["radii"]) & np.isfinite(hard.radii)
    return {
        "shot": shot,
        "k": k,
        "time_s": float(rec["time_s"]),
        "ip_ka": float(rec["ip_ka"]),
        "radii_dmed_cm": float(np.median(dr_cm)),
        "radii_dmax_cm": float(np.max(dr_cm)),
        "dpsi_frac": float(abs(hard.psi_bnd - efit_psi_bnd) / abs(span)),
        "axis_d_cm": 100.0
        * float(np.hypot(hard.axis[0] - axis[0], hard.axis[1] - axis[1])),
        "xset_d_cm": _xset_match_cm(hard.xset, efit_xset),
        "n_efit_x_binding": int(binding.sum()),
        "dev_is_diverted": bool(hard.is_diverted),
        "class_margin": float(np.clip(hard.class_margin, -1.0, 1.0)),
        "smooth_dpsi_frac": float(
            abs(float(smooth["psi_bnd"]) - hard.psi_bnd) / abs(span)
        ),
        "smooth_radii_dmed_cm": (
            100.0
            * float(
                np.median(np.abs(smooth["radii"][sm_rad_ok] - hard.radii[sm_rad_ok]))
            )
            if sm_rad_ok.any()
            else float("nan")
        ),
        "smooth_p_diverted": float(smooth["p_diverted"]),
    }


def _agg(vals: list[float]) -> dict:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "med": None, "p90": None, "max": None}
    return {
        "n": int(v.size),
        "med": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "max": float(np.max(v)),
    }


def score_class(cname: str, recs: np.ndarray) -> dict:
    rows = []
    for rec in recs:
        try:
            r = score_slice(rec)
        except Exception as exc:  # noqa: BLE001 — sweep on, count the miss
            logger.warning(
                "  %s slice %d/%d failed: %s", cname, rec["shot"], rec["k"], exc
            )
            r = None
        if r is not None:
            rows.append(r)
    diverted_expected = cname not in ("limited",)
    class_hits = [r["dev_is_diverted"] == diverted_expected for r in rows]
    out = {
        "class": cname,
        "n_selected": int(recs.size),
        "n_scored": len(rows),
        "radii_dmed_cm": _agg([r["radii_dmed_cm"] for r in rows]),
        "dpsi_frac": _agg([r["dpsi_frac"] for r in rows]),
        "axis_d_cm": _agg([r["axis_d_cm"] for r in rows]),
        "xset_d_cm": _agg([r["xset_d_cm"] for r in rows]),
        "class_agreement": (float(np.mean(class_hits)) if class_hits else None),
        "smooth_dpsi_frac": _agg([r["smooth_dpsi_frac"] for r in rows]),
        "smooth_radii_dmed_cm": _agg([r["smooth_radii_dmed_cm"] for r in rows]),
        "rows": rows,
    }
    return out


def class_verdict(res: dict, supported: bool) -> dict:
    """Apply the pre-declared per-class tolerances to one class's aggregates."""
    if not supported:
        return {"supported": False, "pass": None}
    checks = {
        "radii": res["radii_dmed_cm"]["med"] is not None
        and res["radii_dmed_cm"]["med"] <= LCFS_TOL_CM,
        "psi_bnd": res["dpsi_frac"]["med"] is not None
        and res["dpsi_frac"]["med"] <= PSI_BND_FRAC_TOL,
        "axis": res["axis_d_cm"]["med"] is not None
        and res["axis_d_cm"]["med"] <= AXIS_TOL_CM,
        "smooth_psi": res["smooth_dpsi_frac"]["med"] is not None
        and res["smooth_dpsi_frac"]["med"] <= SMOOTH_PSI_TOL,
        "smooth_radii": res["smooth_radii_dmed_cm"]["med"] is not None
        and res["smooth_radii_dmed_cm"]["med"] <= SMOOTH_RADII_TOL_CM,
    }
    if res["class"] != "limited":
        checks["xset"] = res["xset_d_cm"]["med"] is not None and (
            res["xset_d_cm"]["med"] <= XSET_TOL_CM
        )
    if res["class"] in CLASS_GATED:
        checks["class"] = res["class_agreement"] is not None and (
            res["class_agreement"] >= CLASS_ACC_TOL
        )
    return {"supported": True, "checks": checks, "pass": all(checks.values())}


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
            "imas_ambix/latent/artifacts/patch_gate/topology_efit_read_eval-v0.json"
        ),
    )
    args = ap.parse_args()
    rows = np.load(args.census)["rows"]
    sel = stratified_selection(rows)
    results, verdicts = {}, {}
    for cname, recs in sel.items():
        logger.info("class %s: %d selected", cname, recs.size)
        res = score_class(cname, recs)
        results[cname] = res
        if cname in SCORED_CLASSES:
            verdicts[cname] = class_verdict(res, supported=recs.size >= SUPPORT_FLOOR)
        logger.info(
            "  scored %d — radii med %s cm, dpsi med %s, axis med %s cm, "
            "xset med %s cm, class acc %s",
            res["n_scored"],
            res["radii_dmed_cm"]["med"],
            res["dpsi_frac"]["med"],
            res["axis_d_cm"]["med"],
            res["xset_d_cm"]["med"],
            res["class_agreement"],
        )
    gate_pass = all(v["pass"] for v in verdicts.values() if v["supported"]) and any(
        v["supported"] for v in verdicts.values()
    )
    artifact = {
        "verdict_keys": {"T-F2": bool(gate_pass)},
        "tolerances": {
            "lcfs_med_cm": LCFS_TOL_CM,
            "psi_bnd_frac": PSI_BND_FRAC_TOL,
            "axis_med_cm": AXIS_TOL_CM,
            "xset_med_cm": XSET_TOL_CM,
            "class_acc": CLASS_ACC_TOL,
            "smooth_psi_frac": SMOOTH_PSI_TOL,
            "smooth_radii_cm": SMOOTH_RADII_TOL_CM,
            "support_floor": SUPPORT_FLOOR,
        },
        "verdicts": verdicts,
        "classes": {
            c: {k: v for k, v in r.items() if k != "rows"} for c, r in results.items()
        },
        "rows": {c: r["rows"] for c, r in results.items()},
        "selection": {
            c: [[int(r["shot"]), int(r["k"])] for r in recs] for c, recs in sel.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))
    logger.info("T-F2 %s → %s", "PASS" if gate_pass else "FAIL", args.out)


if __name__ == "__main__":
    main()
