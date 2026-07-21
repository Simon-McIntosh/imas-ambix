"""Corpus-scale success rate of the connectivity read on real EFIT ψ maps.

The stratified per-class validation scores 60 slices per class; a robustness
claim for world-model label production needs the SUCCESS RATE measured at
corpus scale.  This harness draws a seeded uniform random sample of VALID
census slices (all classes, whole campaign range), runs the device hard read
on each EFIT ψ map, and accounts for every slice in exactly one bucket:

  ok            — read completed AND matches EFIT within the per-class
                  validation tolerances (LCFS radii median ≤ 4 cm, ψ_bnd
                  ≤ 0.03 span, axis ≤ 5 cm)
  tol-radii / tol-psi / tol-axis — read completed but out of tolerance
                  (reported per metric, worst first)
  not-found     — read returned found=False
  seed-in-wall  — flood-seed precondition rejected (EFIT axis cell in
                  material / outside the vessel)
  degenerate    — flux span below floor / no finite radii to score
  error         — exception (recorded with its message for RCA)

Every non-ok slice is listed (shot, k, class, bucket, detail) in the artifact
so failures are individually reproducible — nothing is averaged away.

Usage (chunked; merge with --merge):
    uv run python -m scripts.topology_read_success_rate --n 20000 --chunk 0 --n-chunks 4
    uv run python -m scripts.topology_read_success_rate --merge
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("topology_read_success_rate")

from scripts.topology_census import CLASSES  # noqa: E402
from scripts.topology_efit_read_eval import (  # noqa: E402
    AXIS_TOL_CM,
    LCFS_TOL_CM,
    PSI_BND_FRAC_TOL,
    load_slice,
    polygon_ray_radii,
)

ART_DIR = Path("imas_ambix/latent/artifacts/patch_gate")
SAMPLE_SEED = 20260721  # fixed draw — chunks partition one deterministic sample


def read_one(rec) -> tuple[str, str]:
    """Score one slice; returns (bucket, detail)."""
    from imas_ambix.latent.connectivity_boundary import boundary_read  # noqa: PLC0415
    from imas_ambix.worldmodel.equilibrium_labels import LCFS_ANGLES  # noqa: PLC0415

    shot, k = int(rec["shot"]), int(rec["k"])
    try:
        psi, grid, axis, lcfs, _xpts = load_slice(shot, k)
        if lcfs.shape[0] < 8:
            return "degenerate", "fewer than 8 finite LCFS vertices"
        try:
            hard = boundary_read(psi, grid, axis, lcfs_norm=1.0)
        except ValueError as exc:
            return "seed-in-wall", str(exc)[:120]
        if not hard.found:
            return "not-found", ""
        span = hard.psi_bnd - hard.psi_axis
        if abs(span) < 1e-9:
            return "degenerate", "zero axis-boundary flux span"
        efit_radii = polygon_ray_radii(lcfs, axis, LCFS_ANGLES)
        ok = np.isfinite(hard.radii) & np.isfinite(efit_radii)
        if not ok.any():
            return "degenerate", "no finite radii pair"
        dr_med = 100.0 * float(np.median(np.abs(hard.radii[ok] - efit_radii[ok])))
        from scripts.topology_census import _bilinear  # noqa: PLC0415

        efit_psi_bnd = float(
            np.nanmean(
                _bilinear(psi[:, :, None], grid.rg, grid.zg, lcfs[:, 0], lcfs[:, 1], 0)
            )
        )
        dpsi = abs(hard.psi_bnd - efit_psi_bnd) / abs(span)
        axis_d = 100.0 * float(np.hypot(hard.axis[0] - axis[0], hard.axis[1] - axis[1]))
        if dr_med > LCFS_TOL_CM:
            return "tol-radii", f"{dr_med:.2f} cm"
        if dpsi > PSI_BND_FRAC_TOL:
            return "tol-psi", f"{dpsi:.4f} span"
        if axis_d > AXIS_TOL_CM:
            return "tol-axis", f"{axis_d:.2f} cm"
        return "ok", ""
    except Exception as exc:  # noqa: BLE001 — every failure is a datum
        return "error", f"{type(exc).__name__}: {exc}"[:160]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", type=Path, default=ART_DIR / "topology_census-v0.npz")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        parts = sorted(glob.glob(str(ART_DIR / "topology_read_success_chunk*.json")))
        rows = [r for p in parts for r in json.loads(Path(p).read_text())["rows"]]
        buckets: dict[str, int] = {}
        for r in rows:
            buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
        n = len(rows)
        ok = buckets.get("ok", 0)
        artifact = {
            "n_scored": n,
            "success_rate": ok / n if n else None,
            "buckets": dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
            "failures": [r for r in rows if r["bucket"] != "ok"],
            "tolerances": {
                "lcfs_med_cm": LCFS_TOL_CM,
                "psi_bnd_frac": PSI_BND_FRAC_TOL,
                "axis_med_cm": AXIS_TOL_CM,
            },
            "seed": SAMPLE_SEED,
        }
        out = ART_DIR / "topology_read_success-v0.json"
        out.write_text(json.dumps(artifact, indent=2))
        logger.info(
            "merged %d chunks: %d slices, success %.4f\n%s",
            len(parts),
            n,
            artifact["success_rate"],
            json.dumps(artifact["buckets"], indent=2),
        )
        return

    census = np.load(args.census)["rows"]
    valid = census[census["cls"] != CLASSES.index("invalid")]
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(valid.size, size=min(args.n, valid.size), replace=False)
    sample = valid[np.sort(idx)]
    part = sample[args.chunk :: args.n_chunks]
    logger.info("chunk %d/%d: %d slices", args.chunk, args.n_chunks, part.size)
    rows = []
    for i, rec in enumerate(part):
        bucket, detail = read_one(rec)
        rows.append(
            {
                "shot": int(rec["shot"]),
                "k": int(rec["k"]),
                "class": CLASSES[int(rec["cls"])],
                "bucket": bucket,
                "detail": detail,
            }
        )
        if (i + 1) % 250 == 0:
            n_ok = sum(1 for r in rows if r["bucket"] == "ok")
            logger.info("  %d/%d — ok %.4f", i + 1, part.size, n_ok / len(rows))
    out = ART_DIR / f"topology_read_success_chunk{args.chunk}.json"
    out.write_text(json.dumps({"rows": rows}))
    n_ok = sum(1 for r in rows if r["bucket"] == "ok")
    logger.info("chunk done: %d slices, ok %.4f → %s", len(rows), n_ok / len(rows), out)


if __name__ == "__main__":
    main()
