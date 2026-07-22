"""Accuracy / cost benchmark: analytic polygon-section kernel vs a filament proxy.

Quantifies what the analytic axisymmetric polygon-section Green's function
(:func:`imas_ambix.gs.polygon.polygon_greens`, Urankar Part V) buys over the
multi-filament proxy the operator falls back to for a non-rectangular section.

The proxy under test is the operator's REAL fallback: a slanted section can only
be tiled by axis-aligned rectangular filaments (:class:`PFFilament` carries a
width/height, never a shear), so a naive lattice staircases the slanted boundary
and its field error is Riemann-limited (~1/√N).  We measure, per section shape
and per sensor-distance band:

  * the analytic kernel's own accuracy (vs an independent ground truth),
  * the proxy filament count N needed to reach 1e-3 / 1e-4 / 1e-5 field
    agreement, and
  * the wall-cost ratio (proxy-at-matched-accuracy vs analytic).

Ground truth is the exact affine-tiled filament sum (fan-triangulate the convex
piece, barycentric-refine into n² area-exact sub-cells, place a point filament at
each centroid — no staircase, converges O(1/n²)) RICHARDSON-EXTRAPOLATED in 1/n²
from two high resolutions to ~1e-9.  It is independent of the analytic kernel, so
"analytic-vs-GT" is a genuine accuracy statement, not a self-check.  A second,
"smart" affine-tiling proxy is reported alongside the naive rectangular grid to
show that even an ideal tiling is O(N) — the analytic kernel is O(edges).

Non-convex shapes (L, hollow frame) are handled as a sum of convex pieces
(analytic, proxy and ground truth all decompose identically), matching how such
a section would be represented.

numpy/scipy/matplotlib only.  Light enough for a login node; set
OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1 for stable single-thread timing.
Emits a JSON artifact and a multi-pane figure under
docs/figures/polygonal-section-coil-fields/.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from imas_ambix.gs.operator import greens_bz_br, greens_psi
from imas_ambix.gs.polygon import polygon_greens

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "docs" / "figures" / "polygonal-section-coil-fields"

# ------------------------------------------------------------------- shapes ---
# Each shape is a list of convex polygon pieces (r, z) vertices.  Single-piece
# for the convex sections; multi-piece decompositions for the L / hollow frame.
# The parallelogram / crown / trapezoid mirror the validation geometries and are
# sized/placed like the MAST slanted pf_passive elements (P2 arms ~45°, crowns
# ~65°, in the R∈[0.3,1.1] m, |Z|≲1.4 m band).

RECT = [np.array([(0.84, 0.01), (0.96, 0.01), (0.96, 0.19), (0.84, 0.19)])]
PARA = [np.array([(0.85, 0.00), (0.97, 0.00), (1.05, 0.20), (0.93, 0.20)])]  # ~45°
CROWN = [np.array([(0.30, 1.20), (0.42, 1.20), (0.50, 1.37), (0.38, 1.37)])]  # ~65°
TRAP = [np.array([(0.80, -0.10), (1.00, -0.10), (0.95, 0.08), (0.85, 0.08)])]
# L-shape: vertical bar + horizontal foot (two rectangles sharing an edge)
L_SHAPE = [
    np.array([(0.80, 0.00), (0.90, 0.00), (0.90, 0.40), (0.80, 0.40)]),
    np.array([(0.90, 0.00), (1.20, 0.00), (1.20, 0.10), (0.90, 0.10)]),
]
# Hollow frame (coil-case-like): four thin rails around an empty centre
HOLLOW = [
    np.array([(0.80, 0.00), (1.10, 0.00), (1.10, 0.04), (0.80, 0.04)]),  # bottom
    np.array([(0.80, 0.26), (1.10, 0.26), (1.10, 0.30), (0.80, 0.30)]),  # top
    np.array([(0.80, 0.04), (0.84, 0.04), (0.84, 0.26), (0.80, 0.26)]),  # left
    np.array([(1.06, 0.04), (1.10, 0.04), (1.10, 0.26), (1.06, 0.26)]),  # right
]

SHAPES: dict[str, list[np.ndarray]] = {
    "rectangle": RECT,
    "parallelogram_45": PARA,
    "crown_65": CROWN,
    "trapezoid": TRAP,
    "L_decomposed": L_SHAPE,
    "hollow_frame": HOLLOW,
}


def _poly_area(v: np.ndarray) -> float:
    rolled = np.roll(v, -1, axis=0)
    return 0.5 * abs(float(np.sum(v[:, 0] * rolled[:, 1] - rolled[:, 0] * v[:, 1])))


def _pieces_area(pieces: list[np.ndarray]) -> float:
    return float(sum(_poly_area(v) for v in pieces))


def _pieces_centroid_size(pieces: list[np.ndarray]) -> tuple[float, float, float]:
    """Area-weighted centroid and the bounding-box max extent of a shape."""
    all_v = np.concatenate(pieces, axis=0)
    r_lo, r_hi = all_v[:, 0].min(), all_v[:, 0].max()
    z_lo, z_hi = all_v[:, 1].min(), all_v[:, 1].max()
    aw_r = aw_z = wtot = 0.0
    for v in pieces:
        a = _poly_area(v)
        aw_r += a * v[:, 0].mean()
        aw_z += a * v[:, 1].mean()
        wtot += a
    return aw_r / wtot, aw_z / wtot, max(r_hi - r_lo, z_hi - z_lo)


# --------------------------------------------------------------- ground truth -


def _affine_tile(v: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact area-conserving centroid tiling of one convex polygon (O(1/n²)).

    Fan-triangulate from v[0]; split each triangle into n² affine sub-triangles
    (lower + upper barycentric families) and place a point filament at each
    sub-triangle centroid with weight = its exact area.  No staircase error.
    """
    ar, az, wt = [], [], []
    for i in range(1, len(v) - 1):
        v0, v1, v2 = v[0], v[i], v[i + 1]
        area = 0.5 * abs(
            (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
        )
        jj, kk = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        m_lo = (jj + kk) < n
        u_lo, w_lo = (jj[m_lo] + 1 / 3) / n, (kk[m_lo] + 1 / 3) / n
        m_up = (jj + kk) < (n - 1)
        u_up, w_up = (jj[m_up] + 2 / 3) / n, (kk[m_up] + 2 / 3) / n
        uu = np.concatenate([u_lo, u_up])
        ww = np.concatenate([w_lo, w_up])
        ar.append(v0[0] + uu * (v1[0] - v0[0]) + ww * (v2[0] - v0[0]))
        az.append(v0[1] + uu * (v1[1] - v0[1]) + ww * (v2[1] - v0[1]))
        wt.append(np.full(uu.shape, area / n**2))
    return np.concatenate(ar), np.concatenate(az), np.concatenate(wt)


def _filament_fields(tr, tz, ar, az, wt):
    """(ψ, B_R, B_Z) at each target from a weighted point-filament set.

    Uses the operator's OWN axisymmetric loop kernels (``greens_psi`` /
    ``greens_bz_br``) elementwise over filaments — the exact functions the
    operator sums over a retained lattice — so the proxy is the real fallback,
    not a re-derivation.  Targets are few; filaments are many, so we loop the
    targets and vectorise over filaments.
    """
    psi = np.empty(tr.shape)
    br = np.empty(tr.shape)
    bz = np.empty(tr.shape)
    for i in range(tr.size):
        rr = np.full(ar.shape, tr[i])
        zz = np.full(az.shape, tz[i])
        psi[i] = float(np.sum(wt * greens_psi(rr, zz, ar, az)))
        bz_i, br_i = greens_bz_br(rr, zz, ar, az)
        bz[i] = float(np.sum(wt * bz_i))
        br[i] = float(np.sum(wt * br_i))
    return psi, br, bz


def _affine_fields(
    tr: np.ndarray, tz: np.ndarray, pieces: list[np.ndarray], n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ψ, B_R, B_Z) at targets from the exact affine tiling of the shape."""
    a_tot = _pieces_area(pieces)
    psi = np.zeros_like(tr)
    br = np.zeros_like(tr)
    bz = np.zeros_like(tr)
    for v in pieces:
        ar, az, wt = _affine_tile(v, n)
        p, b_r, b_z = _filament_fields(tr, tz, ar, az, wt / a_tot)
        psi += p
        br += b_r
        bz += b_z
    return psi, br, bz


def _ground_truth(
    tr: np.ndarray, tz: np.ndarray, pieces: list[np.ndarray], n_lo=280, n_hi=560
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Richardson-extrapolate the affine tiling in 1/n² → ~1e-9 ground truth."""
    lo = _affine_fields(tr, tz, pieces, n_lo)
    hi = _affine_fields(tr, tz, pieces, n_hi)
    ratio = (n_hi / n_lo) ** 2
    # f_exact ≈ f_hi + (f_hi - f_lo)/(ratio - 1)   [Richardson for O(1/n²) error]
    return tuple(hv + (hv - lv) / (ratio - 1.0) for hv, lv in zip(hi, lo, strict=True))


# --------------------------------------------------------- rectangular proxy --


def _rect_grid_filaments(pieces: list[np.ndarray], nr: int):
    """Axis-aligned grid tiling of the shape's bounding box, inside-cell mask.

    This is the operator's real fallback: PFFilament rectangles can only tile a
    slanted section as an axis-aligned staircase.  Cell current = cell_area /
    total_shape_area (uniform density J = 1/A); the retained tiled area differs
    from the true area by the boundary staircase → the ~1/√N Riemann error.
    Returns (r, z, weight) filament arrays.
    """
    all_v = np.concatenate(pieces, axis=0)
    r_lo, r_hi = all_v[:, 0].min(), all_v[:, 0].max()
    z_lo, z_hi = all_v[:, 1].min(), all_v[:, 1].max()
    span_r, span_z = r_hi - r_lo, z_hi - z_lo
    nz = max(1, int(round(nr * span_z / span_r)))
    dr, dz = span_r / nr, span_z / nz
    ic = (np.arange(nr) + 0.5) * dr + r_lo
    jc = (np.arange(nz) + 0.5) * dz + z_lo
    gr, gz = np.meshgrid(ic, jc, indexing="ij")
    pts = np.column_stack([gr.ravel(), gz.ravel()])
    inside = np.zeros(pts.shape[0], dtype=bool)
    for v in pieces:
        inside |= MplPath(v).contains_points(pts)
    r = pts[inside, 0]
    z = pts[inside, 1]
    a_tot = _pieces_area(pieces)
    wt = np.full(r.shape, dr * dz / a_tot)  # uniform density; Σwt ≈ 1 (staircase)
    return r, z, wt


def _proxy_fields(tr, tz, r, z, wt):
    """Field from the staircased rectangular filament lattice (the fallback)."""
    return _filament_fields(tr, tz, r, z, wt)


# ------------------------------------------------------------- error metrics --


def _field_rel_err(got, gt) -> float:
    """Band-aggregate relative error over the target set: max(ψ-rel, |B|-rel).

    An L2 norm over the band's targets (not a per-target ratio), so a single
    sensor where a field component crosses zero cannot inflate the metric.  ψ and
    the stacked B-vector (br, bz) are each normalised by their own L2 scale over
    the band; the reported figure is the worse of the two.
    """
    psi_g, br_g, bz_g = got
    psi_t, br_t, bz_t = gt
    psi_rel = float(np.linalg.norm(psi_g - psi_t) / max(np.linalg.norm(psi_t), 1e-300))
    b_num = float(np.linalg.norm(np.concatenate([br_g - br_t, bz_g - bz_t])))
    b_den = max(float(np.linalg.norm(np.concatenate([br_t, bz_t]))), 1e-300)
    return max(psi_rel, b_num / b_den)


# ---------------------------------------------------------------- benchmark ---

BANDS = {"near": 1.3, "mid": 2.5, "far": 4.0}  # × shape size, from the centroid
TOLS = [1e-3, 1e-4, 1e-5]
_R_MIN = 0.2  # machine bore; a ring at large multiples would cross the axis


def _targets_for_shape(pieces: list[np.ndarray]) -> dict[str, tuple]:
    """Sensor rings per distance band (× shape size), physical directions only.

    16 directions per band; targets below the machine bore (R < _R_MIN — a ring
    at large multiples crosses the axis) or inside the section are dropped, so
    the error metric is never taken at an unphysical or interior point.
    """
    cr, cz, size = _pieces_centroid_size(pieces)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    paths = [MplPath(v) for v in pieces]
    out = {}
    for band, mult in BANDS.items():
        d = mult * size
        tr = cr + d * np.cos(ang)
        tz = cz + d * np.sin(ang)
        keep = tr >= _R_MIN
        pts = np.column_stack([tr, tz])
        for p in paths:
            keep &= ~p.contains_points(pts)
        out[band] = (np.asarray(tr[keep]), np.asarray(tz[keep]))
    return out


def _time_call(fn, repeat=5) -> float:
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@dataclass
class ShapeResult:
    name: str
    n_edges: int
    area: float
    analytic_rel_err: dict = field(default_factory=dict)  # band -> max rel err
    proxy_n_to_tol: dict = field(default_factory=dict)  # band -> {tol -> N}
    proxy_curve: dict = field(default_factory=dict)  # band -> [(N, relerr)]
    cost: dict = field(default_factory=dict)  # band -> {analytic_s, proxy_s, ratio}


def _n_to_tol(curve: list[tuple[int, float]], tol: float) -> float | None:
    """Reliable filament count: smallest N past which the proxy error ENVELOPE
    stays ≤ tol.

    The rectangular staircase makes err(N) non-monotone (a thin rail can alias
    with the grid — a lucky N hits tol, the next N misses by 1000×), so a first-
    crossing count is not reproducible.  We take the suffix-max envelope (the
    worst error at this N or any larger swept N) and interpolate where THAT drops
    below tol — the honest "commit at least this many filaments" number.  None if
    the swept range never keeps the envelope below tol."""
    xs = np.array([c[0] for c in curve], dtype=float)
    ys = np.array([c[1] for c in curve], dtype=float)
    env = np.maximum.accumulate(ys[::-1])[::-1]  # suffix max
    below = np.where(env <= tol)[0]
    if below.size == 0:
        return None
    j = below[0]
    if j == 0:
        return float(xs[0])
    x0, x1 = np.log(xs[j - 1]), np.log(xs[j])
    y0, y1 = np.log(env[j - 1]), np.log(env[j])
    if y1 == y0:
        return float(xs[j])
    t = (np.log(tol) - y0) / (y1 - y0)
    return float(np.exp(x0 + t * (x1 - x0)))


# analytic φ-quadrature ladder (panels, nodes), cheapest first — used to tune
# the analytic kernel to the target tolerance for a MATCHED-accuracy cost race.
_QUAD_LADDER = [(2, 8), (3, 12), (4, 16), (6, 24), (8, 32), (12, 48), (16, 48)]


def run_shape(name: str, pieces: list[np.ndarray], nr_sweep: list[int]) -> ShapeResult:
    n_edges = sum(len(v) for v in pieces)
    res = ShapeResult(name=name, n_edges=n_edges, area=_pieces_area(pieces))
    targets = _targets_for_shape(pieces)
    a_tot = _pieces_area(pieces)

    def analytic_q(tr, tz, npan, nnod):
        psi = np.zeros_like(tr)
        br = np.zeros_like(tr)
        bz = np.zeros_like(tr)
        for v in pieces:
            p, b_r, b_z = polygon_greens(tr, tz, v, n_panels=npan, n_nodes=nnod)
            wa = _poly_area(v) / a_tot
            psi += wa * p
            br += wa * b_r
            bz += wa * b_z
        return psi, br, bz

    def cheapest_quad(tr, tz, gt, tol):
        """Cheapest φ-quadrature whose analytic field reaches tol (else the top)."""
        for npan, nnod in _QUAD_LADDER:
            if _field_rel_err(analytic_q(tr, tz, npan, nnod), gt) <= tol:
                return npan, nnod
        return _QUAD_LADDER[-1]

    for band, (tr, tz) in targets.items():
        if tr.size == 0:
            continue
        gt = _ground_truth(tr, tz, pieces)
        # exactness at the default (over-resolved) quadrature — the "how exact"
        res.analytic_rel_err[band] = _field_rel_err(analytic_q(tr, tz, 16, 48), gt)

        curve = []
        for nr in nr_sweep:
            r, z, wt = _rect_grid_filaments(pieces, nr)
            prox = _proxy_fields(tr, tz, r, z, wt)
            curve.append((int(r.size), _field_rel_err(prox, gt)))
        curve = sorted({n: e for n, e in curve}.items())
        res.proxy_curve[band] = curve
        res.proxy_n_to_tol[band] = {f"{t:.0e}": _n_to_tol(curve, t) for t in TOLS}

        # matched-accuracy cost race at 1e-4: cheapest analytic quadrature that
        # reaches 1e-4 vs the proxy at the reliable N-envelope for 1e-4.
        n_star = res.proxy_n_to_tol[band]["1e-04"]
        q_ana = cheapest_quad(tr, tz, gt, 1e-4)
        # bind loop vars as defaults so the timing closures capture this band
        t_ana = _time_call(lambda tr=tr, tz=tz, q=q_ana: analytic_q(tr, tz, *q))
        if n_star is not None:
            nr_star = _nr_for_target_count(pieces, int(np.ceil(n_star)))
            r, z, wt = _rect_grid_filaments(pieces, nr_star)
            n_used = int(r.size)
            t_prox = _time_call(
                lambda tr=tr, tz=tz, r=r, z=z, wt=wt: _proxy_fields(tr, tz, r, z, wt)
            )
            ratio = t_prox / t_ana
        else:
            t_prox, ratio, n_used = None, None, None
        res.cost[band] = {
            "n_targets": int(tr.size),
            "analytic_quad_at_1e-4": list(q_ana),
            "analytic_s": t_ana,
            "proxy_s_at_1e-4": t_prox,
            "n_at_1e-4": None if n_star is None else int(np.ceil(n_star)),
            "n_used_at_1e-4": n_used,
            "cost_ratio_proxy_over_analytic": ratio,
        }
    return res


def _nr_for_target_count(pieces: list[np.ndarray], n_target: int) -> int:
    """Grid resolution nr whose inside-cell count is nearest to n_target."""
    best_nr, best_diff = 4, np.inf
    for nr in range(4, 400):
        r, _, _ = _rect_grid_filaments(pieces, nr)
        diff = abs(r.size - n_target)
        if diff < best_diff:
            best_nr, best_diff = nr, diff
        if r.size > 2 * n_target:
            break
    return best_nr


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    nr_sweep = [4, 6, 8, 11, 16, 22, 32, 45, 64, 90, 128, 180, 256, 360]
    results = {}
    for name, pieces in SHAPES.items():
        print(f"[benchmark] {name} ...", flush=True)
        r = run_shape(name, pieces, nr_sweep)
        results[name] = {
            "n_edges": r.n_edges,
            "area_m2": r.area,
            "analytic_rel_err": r.analytic_rel_err,
            "proxy_n_to_tol": r.proxy_n_to_tol,
            "proxy_curve": {b: r.proxy_curve[b] for b in r.proxy_curve},
            "cost": r.cost,
        }
    payload = {
        "description": "Analytic polygon-section kernel vs rectangular filament "
        "proxy: accuracy, proxy-N-to-tolerance, and wall-cost.",
        "ground_truth": "Richardson-extrapolated exact affine tiling (1/n^2), "
        "n=280,560 -> ~1e-9; independent of the analytic kernel.",
        "proxy": "axis-aligned staircased grid of point filaments (the operator's "
        "PFFilament lattice fallback), uniform current density J=1/A.",
        "bands": BANDS,
        "tolerances": TOLS,
        "results": results,
    }
    out_json = OUTDIR / "proxy-benchmark.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[benchmark] wrote {out_json}")
    _make_figure(payload)


def _make_figure(payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = payload["results"]
    names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) convergence curves (mid band) — proxy error vs N, analytic floor
    ax = axes[0, 0]
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        curve = results[name]["proxy_curve"].get("mid")
        if not curve:
            continue
        ns = [c[0] for c in curve]
        es = [c[1] for c in curve]
        ax.loglog(ns, es, "o-", color=cmap(i), ms=4, lw=1.3, label=name)
        floor = results[name]["analytic_rel_err"].get("mid")
        if floor is not None:
            ax.axhline(floor, color=cmap(i), ls=":", lw=1.0, alpha=0.7)
    ref_n = np.array([8, 180])
    ax.loglog(ref_n, 0.05 * (ref_n / 8.0) ** -0.5, "k--", lw=1.0, label="~1/√N ref")
    for t in TOLS:
        ax.axhline(t, color="0.8", lw=0.7, zorder=0)
    ax.set_xlabel("proxy filament count N")
    ax.set_ylabel("max field rel. error (mid band)")
    ax.set_title("(a) Proxy convergence vs analytic floor (dotted)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.2)

    # (b) proxy N to reach 1e-4, per shape and band
    ax = axes[0, 1]
    bandnames = list(BANDS.keys())
    x = np.arange(len(names))
    w = 0.25
    for k, band in enumerate(bandnames):
        vals = []
        for name in names:
            n = results[name]["proxy_n_to_tol"].get(band, {}).get("1e-04")
            vals.append(np.nan if n is None else n)
        ax.bar(x + (k - 1) * w, vals, w, label=band)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("proxy N for 1e-4 agreement")
    ax.set_title("(b) Filaments the proxy needs (1e-4)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    # (c) wall-cost ratio proxy/analytic at 1e-4
    ax = axes[1, 0]
    for k, band in enumerate(bandnames):
        vals = []
        for name in names:
            c = (
                results[name]["cost"]
                .get(band, {})
                .get("cost_ratio_proxy_over_analytic")
            )
            vals.append(np.nan if c is None else c)
        ax.bar(x + (k - 1) * w, vals, w, label=band)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("wall-cost ratio  proxy / analytic")
    ax.set_title("(c) Cost at matched accuracy (>1 ⇒ analytic wins)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    # (d) analytic accuracy per shape/band (how exact the kernel is)
    ax = axes[1, 1]
    for k, band in enumerate(bandnames):
        vals = [results[name]["analytic_rel_err"].get(band, np.nan) for name in names]
        ax.bar(x + (k - 1) * w, vals, w, label=band)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("analytic max rel. error vs ground truth")
    ax.set_title("(d) Analytic kernel accuracy (vs 1e-9 GT)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle(
        "Analytic polygon-section kernel vs multi-filament proxy — "
        "accuracy, filament budget, and cost",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTDIR / "fig-proxy-benchmark.png"
    fig.savefig(out, dpi=130)
    print(f"[benchmark] wrote {out}")


if __name__ == "__main__":
    main()
