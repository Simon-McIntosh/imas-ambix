"""EFIT-derived topology census of the MAST level-2 corpus.

Classifies EVERY equilibrium slice in the level-2 mirror from EFIT's OWN
reconstruction data — the stored ψ map (65×65, ordered ``[z, r, t]``), the LCFS
polygon, the two X-point slots, and the magnetic axis — into a topology class,
so the corpus can be stratified for validating the connectivity read on REAL
exotic topologies.  EFIT is a referee here: nothing in this module feeds the
engine solve.

Per-slice classification (all flux comparisons in normalised units
``u = (ψ − ψ_axis)/(ψ_bnd − ψ_axis)``, sign-agnostic; ψ_axis is the bilinear
read at EFIT's axis, ψ_bnd the mean bilinear read over EFIT's LCFS polygon —
the polygon is an EFIT iso-flux line, verified constant to ~1e-4 of the span):

* An X-point slot PARTICIPATES in the boundary when ``|u_x − 1| ≤ 0.05``
  (the bilinear-off-65×65 flux floor at a saddle).
* both participate and ``|u_x1 − u_x2| ≤ 0.01``      → ``connected-DN``
* both participate and ``0.01 < |u_x1 − u_x2| ≤ 0.05`` → ``marginal-DN``
* exactly one participates (or the DN gap exceeds 0.05): the participating
  (closest-to-binding) X-point's Z sign splits ``SN-lower`` / ``SN-upper``.
* no slot participates (or both slots empty)          → ``limited``
* ``snowflake-candidate`` is an ORTHOGONAL flag, not a class: two finite
  X-points within 0.25 m of each other at near-equal flux (gap ≤ 0.02) — the
  merged-saddle geometry a snowflake experiment approaches.

Slice validity: finite axis + ≥ 8 finite LCFS vertices + |Ip| ≥ 100 kA
(``summary/ip`` interpolated onto the equilibrium time base) + flux span
> 1e-4 Wb.  Invalid slices are counted (``invalid`` class) but never scored.

Output: one compressed ``.npz`` of per-slice records per shot-range chunk plus
a JSON summary; ``--merge`` folds chunk files into the census artifact used by
the stratified-selection and scoring stages.

Usage:
    uv run python -m scripts.topology_census --start 0 --end 3000 \
        --out imas_ambix/latent/artifacts/patch_gate/topology_census_chunk0.npz
    uv run python -m scripts.topology_census --merge \
        imas_ambix/latent/artifacts/patch_gate/topology_census_chunk*.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("topology_census")

LEVEL2_SHOTS = Path("/work/projects/imas_gpu/mast/level2/shots")

# pre-declared classification thresholds (normalised-flux units u)
X_BIND_U = 0.05  # |u_x − 1| for an X-point to participate in the boundary
DN_CONNECTED_U = 0.01  # inter-X flux gap for a connected double-null
DN_MARGINAL_U = 0.05  # inter-X flux gap ceiling for a marginal double-null
SNOWFLAKE_DIST_M = 0.25  # inter-X spatial distance for a snowflake candidate
SNOWFLAKE_GAP_U = 0.02  # inter-X flux gap for a snowflake candidate
MIN_IP_KA = 100.0  # slice validity: plasma current floor
MIN_SPAN_WB = 1.0e-4  # slice validity: axis→boundary flux span floor
MIN_LCFS_PTS = 8  # slice validity: finite LCFS vertices

CLASSES = (
    "limited",
    "sn-lower",
    "sn-upper",
    "connected-dn",
    "marginal-dn",
    "invalid",
)

RECORD_DTYPE = np.dtype(
    [
        ("shot", np.int32),
        ("k", np.int16),  # slice index within the shot
        ("time_s", np.float32),
        ("ip_ka", np.float32),
        ("cls", np.int8),  # index into CLASSES
        ("snowflake", np.bool_),
        ("u_x_lo", np.float32),  # normalised flux of the lower-Z X slot
        ("u_x_hi", np.float32),
        ("x_lo_r", np.float32),
        ("x_lo_z", np.float32),
        ("x_hi_r", np.float32),
        ("x_hi_z", np.float32),
        ("dn_gap_u", np.float32),
        ("x_dist_m", np.float32),
        ("axis_r", np.float32),
        ("axis_z", np.float32),
        ("triangularity_upper", np.float32),
        ("triangularity_lower", np.float32),
        ("elongation", np.float32),
    ]
)


def _bilinear(psi_zrt: np.ndarray, rg, zg, r, z, k) -> np.ndarray:
    """Bilinear ψ at points (r, z) for slice ``k``; NaN outside the grid.

    ``psi_zrt`` is the stored ``[z, r, t]`` map; ``r``/``z``/``k`` broadcast.
    """
    r = np.asarray(r, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    dr = rg[1] - rg[0]
    dz = zg[1] - zg[0]
    fr = (r - rg[0]) / dr
    fz = (z - zg[0]) / dz
    ok = (fr >= 0) & (fr <= rg.size - 1) & (fz >= 0) & (fz <= zg.size - 1)
    fr = np.clip(np.nan_to_num(fr), 0, rg.size - 1 - 1e-9)
    fz = np.clip(np.nan_to_num(fz), 0, zg.size - 1 - 1e-9)
    j = fr.astype(int)
    i = fz.astype(int)
    tr = fr - j
    tz = fz - i
    p00 = psi_zrt[i, j, k]
    p01 = psi_zrt[i, j + 1, k]
    p10 = psi_zrt[i + 1, j, k]
    p11 = psi_zrt[i + 1, j + 1, k]
    val = (
        p00 * (1 - tz) * (1 - tr)
        + p01 * (1 - tz) * tr
        + p10 * tz * (1 - tr)
        + p11 * tz * tr
    )
    return np.where(ok, val, np.nan)


def census_shot(shot: int) -> np.ndarray:
    """Classify every equilibrium slice of one shot; returns RECORD_DTYPE rows."""
    import zarr  # noqa: PLC0415 — worker import

    g = zarr.open_group(str(LEVEL2_SHOTS / f"{shot}.zarr"), mode="r")
    if "equilibrium" not in g:
        return np.empty(0, dtype=RECORD_DTYPE)
    eq = g["equilibrium"]
    t = np.asarray(eq["time"], dtype=np.float64)
    n = t.size
    if n == 0:
        return np.empty(0, dtype=RECORD_DTYPE)
    psi = np.asarray(eq["psi"], dtype=np.float64)  # (nz, nr, t)
    rg = np.asarray(eq["major_radius"], dtype=np.float64)
    zg = np.asarray(eq["z"], dtype=np.float64)
    ax_r = np.asarray(eq["magnetic_axis_r"], dtype=np.float64)
    ax_z = np.asarray(eq["magnetic_axis_z"], dtype=np.float64)
    lcfs_r = np.asarray(eq["lcfs_r"], dtype=np.float64)  # (157, t)
    lcfs_z = np.asarray(eq["lcfs_z"], dtype=np.float64)
    x_r = np.asarray(eq["x_point_r"], dtype=np.float64)  # (2, t)
    x_z = np.asarray(eq["x_point_z"], dtype=np.float64)
    tri_u = np.asarray(eq["triangularity_upper"], dtype=np.float64)
    tri_l = np.asarray(eq["triangularity_lower"], dtype=np.float64)
    elon = np.asarray(eq["elongation"], dtype=np.float64)

    # plasma current on the equilibrium time base (kA)
    ip_ka = np.full(n, np.nan)
    if "summary" in g and "ip" in g["summary"]:
        st = np.asarray(g["summary"]["time"], dtype=np.float64)
        sip = np.asarray(g["summary"]["ip"], dtype=np.float64)
        m = np.isfinite(st) & np.isfinite(sip)
        if m.sum() >= 2:
            ip_ka = np.interp(t, st[m], sip[m]) / 1.0e3

    rows = np.zeros(n, dtype=RECORD_DTYPE)
    rows["shot"] = shot
    rows["k"] = np.arange(n)
    rows["time_s"] = t
    rows["ip_ka"] = ip_ka
    rows["triangularity_upper"] = tri_u
    rows["triangularity_lower"] = tri_l
    rows["elongation"] = elon
    rows["axis_r"] = ax_r
    rows["axis_z"] = ax_z

    ks = np.arange(n)
    psi_axis = _bilinear(psi, rg, zg, ax_r, ax_z, ks)
    # ψ_bnd = mean interpolated flux over the finite LCFS vertices
    lp = _bilinear(
        psi, rg, zg, lcfs_r, lcfs_z, np.broadcast_to(ks, lcfs_r.shape)
    )  # (157, t)
    n_lcfs = np.isfinite(lp).sum(axis=0)
    with np.errstate(invalid="ignore"):
        psi_bnd = np.nanmean(lp, axis=0)
    span = psi_bnd - psi_axis

    # X-slot normalised flux; order slots by Z so lo/hi are geometric
    span_safe = np.where(np.abs(span) > MIN_SPAN_WB, span, np.nan)
    u_x = (
        _bilinear(psi, rg, zg, x_r, x_z, np.broadcast_to(ks, x_r.shape)) - psi_axis
    ) / span_safe
    lo_first = np.where(np.isfinite(x_z).all(axis=0), np.argmin(x_z, axis=0), 0).astype(
        int
    )
    hi_first = 1 - lo_first
    cols = np.arange(n)
    u_lo, u_hi = u_x[lo_first, cols], u_x[hi_first, cols]
    xr_lo, xz_lo = x_r[lo_first, cols], x_z[lo_first, cols]
    xr_hi, xz_hi = x_r[hi_first, cols], x_z[hi_first, cols]
    rows["u_x_lo"], rows["u_x_hi"] = u_lo, u_hi
    rows["x_lo_r"], rows["x_lo_z"] = xr_lo, xz_lo
    rows["x_hi_r"], rows["x_hi_z"] = xr_hi, xz_hi
    dn_gap = np.abs(u_lo - u_hi)
    x_dist = np.hypot(xr_lo - xr_hi, xz_lo - xz_hi)
    rows["dn_gap_u"] = dn_gap
    rows["x_dist_m"] = x_dist

    valid = (
        np.isfinite(psi_axis)
        & np.isfinite(psi_bnd)
        & (np.abs(span) > MIN_SPAN_WB)
        & (n_lcfs >= MIN_LCFS_PTS)
        & np.isfinite(ip_ka)
        & (np.abs(ip_ka) >= MIN_IP_KA)
    )
    bind_lo = np.abs(u_lo - 1.0) <= X_BIND_U
    bind_hi = np.abs(u_hi - 1.0) <= X_BIND_U
    both = bind_lo & bind_hi
    cls = np.full(n, CLASSES.index("invalid"), dtype=np.int8)
    cls[valid] = CLASSES.index("limited")
    sn_lo = valid & bind_lo & ~(both & (dn_gap <= DN_MARGINAL_U))
    sn_hi = valid & bind_hi & ~bind_lo & ~(both & (dn_gap <= DN_MARGINAL_U))
    # when both bind but the gap exceeds the marginal ceiling, the closer slot wins
    far_dn = valid & both & (dn_gap > DN_MARGINAL_U)
    lo_closer = np.abs(u_lo - 1.0) <= np.abs(u_hi - 1.0)
    cls[sn_lo & ~far_dn] = CLASSES.index("sn-lower")
    cls[sn_hi & ~far_dn] = CLASSES.index("sn-upper")
    cls[far_dn & lo_closer] = CLASSES.index("sn-lower")
    cls[far_dn & ~lo_closer] = CLASSES.index("sn-upper")
    cls[valid & both & (dn_gap <= DN_MARGINAL_U) & (dn_gap > DN_CONNECTED_U)] = (
        CLASSES.index("marginal-dn")
    )
    cls[valid & both & (dn_gap <= DN_CONNECTED_U)] = CLASSES.index("connected-dn")
    rows["cls"] = cls
    rows["snowflake"] = (
        valid
        & np.isfinite(x_dist)
        & (x_dist <= SNOWFLAKE_DIST_M)
        & (dn_gap <= SNOWFLAKE_GAP_U)
        & (bind_lo | bind_hi)
    )
    return rows


def _shot_list() -> list[int]:
    return sorted(int(p.name.split(".")[0]) for p in LEVEL2_SHOTS.glob("*.zarr"))


def run_chunk(start: int, end: int, out: Path, workers: int) -> None:
    shots = _shot_list()[start:end]
    logger.info("census: %d shots [%d..%d]", len(shots), shots[0], shots[-1])
    parts: list[np.ndarray] = []
    failed: list[int] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(census_shot, s): s for s in shots}
        for i, fut in enumerate(as_completed(futs)):
            s = futs[fut]
            try:
                parts.append(fut.result())
            except Exception as exc:  # noqa: BLE001 — census sweeps on
                failed.append(s)
                logger.warning("shot %d failed: %s", s, exc)
            if (i + 1) % 250 == 0:
                logger.info("  %d/%d shots done", i + 1, len(shots))
    rows = (
        np.concatenate([p for p in parts if p.size])
        if parts
        else np.empty(0, dtype=RECORD_DTYPE)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=rows, failed=np.asarray(failed, dtype=np.int32))
    counts = {c: int((rows["cls"] == i).sum()) for i, c in enumerate(CLASSES)}
    counts["snowflake-candidate"] = int(rows["snowflake"].sum())
    logger.info("chunk done: %d rows → %s\n%s", rows.size, out, json.dumps(counts))


def merge(paths: list[str], out: Path) -> None:
    rows = np.concatenate([np.load(p)["rows"] for p in sorted(paths)])
    failed = np.concatenate([np.load(p)["failed"] for p in sorted(paths)])
    np.savez_compressed(out, rows=rows, failed=failed)
    counts = {c: int((rows["cls"] == i).sum()) for i, c in enumerate(CLASSES)}
    counts["snowflake-candidate"] = int(rows["snowflake"].sum())
    summary = {
        "n_slices": int(rows.size),
        "n_shots": int(np.unique(rows["shot"]).size),
        "n_failed_shots": int(failed.size),
        "counts": counts,
        "thresholds": {
            "x_bind_u": X_BIND_U,
            "dn_connected_u": DN_CONNECTED_U,
            "dn_marginal_u": DN_MARGINAL_U,
            "snowflake_dist_m": SNOWFLAKE_DIST_M,
            "snowflake_gap_u": SNOWFLAKE_GAP_U,
            "min_ip_ka": MIN_IP_KA,
        },
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "merged %d chunks → %s\n%s", len(paths), out, json.dumps(summary, indent=2)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0, help="shot-list start index")
    ap.add_argument("--end", type=int, default=None, help="shot-list end index")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--merge", nargs="*", default=None, help="chunk npz globs to merge")
    args = ap.parse_args()
    if args.merge is not None:
        paths = [p for pat in args.merge for p in glob.glob(pat)]
        out = args.out or Path(
            "imas_ambix/latent/artifacts/patch_gate/topology_census-v0.npz"
        )
        merge(paths, out)
        return
    end = args.end if args.end is not None else len(_shot_list())
    out = args.out or Path(
        f"imas_ambix/latent/artifacts/patch_gate/topology_census_chunk_{args.start}_{end}.npz"
    )
    run_chunk(args.start, end, out, args.workers)


if __name__ == "__main__":
    main()
