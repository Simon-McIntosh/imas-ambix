"""Convergence-accelerator probe on the temperature-smoothed pinned map.

The single-shot device rollout fixed two things the CPU accelerator study
asked for — the current-centroid pin lives INSIDE the map (the per-sweep
closed-form K=2 scaffold LSQ) and the whole solve is a fixed-shape jax graph —
but its topology read is still the HARD kernel (exact-min binding + boolean
flood core mask), so residual cell-flips survive in the on-device map and
safeguarded Anderson merely ties relaxed Picard (both ~49% converged at a
20-sweep budget).  The smooth read (softmin binding + retracted-gate sigmoid
core weight at temperature τ) has since landed and is end-to-end
differentiable: gradients flow from every read scalar back through ψ to the
currents.

This probe measures, on the real staged shot (the rollout's own input arrays),
what that buys the fixed-point iteration:

* **derivative health** — jax.jvp of the full pinned ψ-map against central
  finite differences, hard vs smooth map (is the analytic tangent clean, and
  does the hard map's tangent disagree with its own finite differences?);
* **accelerator A/B at a fixed evaluation budget** — relaxed Picard, the
  rollout's safeguarded Anderson, Jacobian-free Newton–Krylov with EXACT
  jvp tangents (fixed-shape GMRES, no finite differences), and a REDUCED
  Newton on the 2-coefficient scaffold map (the fixed point lives in the
  K=2 coefficient space; its 2×2 Jacobian costs two tangent passes) — all
  pure fixed-shape jax, i.e. batchable by construction;
* **temperature sensitivity** — convergence vs τ (does smoothing trade
  accuracy for contraction?).

Cost accounting is in MAP EVALUATIONS (one read + one Green's GEMM), the unit
that survives batching: a vmapped GEMM costs the same per column whether the
column belongs to a Picard sweep, a GMRES tangent, or a Jacobian column.

Usage:
    uv run python -m scripts.tau_map_accelerator_probe \
        --inputs imas_ambix/latent/artifacts/patch_gate/\\
            device_rollout_single_shot-inputs.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tau_map_accelerator_probe")

# the rollout's frozen solver constants (shared so the A/B is like-for-like)
BETA0 = 0.5
ALPHA = 1.0
CENTROID_SIGMA_M = 0.02
READ_N_LEVELS = 48
READ_N_BISECT = 12
READ_N_RAY = 96
RELAX = 0.5
ANDERSON_M = 3
ANDERSON_WARMUP = 6
ANDERSON_CAP = 2.0

TOLERANCE = 3e-4
NEWTON_WARMUP = 8  # pinned Picard sweeps before any Newton-type iteration


def build_step(arrs, *, read: str, tau: float = 1e-3):
    """(jphi_from_psi, psi_from_jphi) for the pinned K=2 map under one read.

    ``read='hard'`` replicates the rollout map (exact-min binding, boolean
    flood core).  ``read='smooth'`` swaps in the softmin binding + sigmoid
    core weight at temperature ``tau`` and takes the axis from the stencil
    O-point (biquadratic sub-grid refine — differentiable through the
    surface fit).  Both are pure fixed-shape fp64 functions.
    """
    import jax.numpy as jnp

    from imas_ambix.latent.connectivity_boundary import (
        _DEFAULT_ANGLES,
        _NO_WALL_PSI,
        boundary_read_jax,
        boundary_read_smooth_jax,
    )
    from imas_ambix.latent.flux_surface_connectivity import flood_fill_core
    from imas_ambix.latent.stencil_nulls import magnetic_axis_subgrid

    rg = jnp.asarray(arrs["rg"])
    zg = jnp.asarray(arrs["zg"])
    inside = jnp.asarray(arrs["inside"])
    flat_r = jnp.asarray(arrs["flat_r"])
    flat_z = jnp.asarray(arrs["flat_z"])
    cells = jnp.asarray(arrs["cells"])
    wall_r = jnp.asarray(arrs["wall_r"])
    wall_z = jnp.asarray(arrs["wall_z"])
    r0 = float(arrs["r0"])
    cell_area = float(arrs["dr"]) * float(arrs["dz"])
    nz, nr = int(arrs["nz"]), int(arrs["nr"])
    n_flood = nr + nz
    cell_r = flat_r[cells]
    cell_z = flat_z[cells]
    img_r_ratio = flat_r / r0
    img_r_inv = r0 / jnp.maximum(flat_r, 1e-3)
    base = BETA0 * img_r_ratio + (1.0 - BETA0) * img_r_inv
    g_mat = jnp.asarray(arrs["G"])

    def family_images(psi_n, core_weight):
        """The two scaffold family images on a read's support (jφ is linear
        in the coefficient pair on a frozen support: jφ = c₀·img_p + c₁·img_f)."""
        e = jnp.clip(1.0 - psi_n, 0.0, None) ** ALPHA * core_weight
        return img_r_ratio * e, img_r_inv * e

    def coeffs_from_images(img_p, img_f, pin, ip):
        """Closed-form pinned K=2 LSQ (Ip row + centroid tether rows) → c."""
        u = jnp.stack([img_p[cells], img_f[cells]], axis=1) * cell_area
        s_row = jnp.sum(u, axis=0) / ip
        mr_row = jnp.sum(u * (cell_r - pin[0])[:, None], axis=0) / (
            ip * CENTROID_SIGMA_M
        )
        mz_row = jnp.sum(u * (cell_z - pin[1])[:, None], axis=0) / (
            ip * CENTROID_SIGMA_M
        )
        a_mat = jnp.stack([s_row, mr_row, mz_row], axis=0)
        b_vec = jnp.array([1.0, 0.0, 0.0])
        n_mat = a_mat.T @ a_mat
        n_mat = n_mat + 1e-12 * (jnp.trace(n_mat) + 1e-30) * jnp.eye(2)
        return jnp.linalg.solve(n_mat, a_mat.T @ b_vec)

    def _scaffold(psi_n, core_weight, pin, ip):
        img_p, img_f = family_images(psi_n, core_weight)
        c = coeffs_from_images(img_p, img_f, pin, ip)
        jphi = c[0] * img_p + c[1] * img_f
        bad = ~jnp.all(jnp.isfinite(jphi)) | (jnp.sum(jnp.abs(jphi)) < 1e-12)
        e = jnp.clip(img_p / jnp.maximum(img_r_ratio, 1e-12), 0.0, None)
        jphi = jnp.where(bad, base * e, jphi)
        return jphi, c

    if read == "hard":

        def support(psi, axis):
            psi2d = psi.reshape(nz, nr)
            rd = boundary_read_jax(
                psi2d,
                rg,
                zg,
                inside,
                axis[0],
                axis[1],
                READ_N_LEVELS,
                READ_N_BISECT,
                READ_N_RAY,
                _DEFAULT_ANGLES,
                0.999,
                wall_r,
                wall_z,
                _NO_WALL_PSI,
            )
            ax_r = jnp.where(jnp.isfinite(rd["axis_r"]), rd["axis_r"], axis[0])
            ax_z = jnp.where(jnp.isfinite(rd["axis_z"]), rd["axis_z"], axis[1])
            axis = jnp.array([ax_r, ax_z])
            psi_axis = jnp.where(
                jnp.isfinite(rd["axis_psi_sub"]), rd["axis_psi_sub"], rd["psi_axis"]
            )
            psi_bnd = jnp.where(
                jnp.isfinite(rd["psi_bnd"]), rd["psi_bnd"], psi_axis + 1.0
            )
            span = psi_bnd - psi_axis
            span = jnp.where(jnp.abs(span) < 1e-12, 1e-12, span)
            psi_n = (psi - psi_axis) / span
            ja = jnp.argmin(jnp.abs(rg - axis[0]))
            ia = jnp.argmin(jnp.abs(zg - axis[1]))
            seed2d = jnp.zeros((nz, nr), dtype=bool).at[ia, ja].set(True)
            confined = ((psi_n < 1.0).reshape(nz, nr)) & inside
            core = flood_fill_core(confined, seed2d, n_flood).reshape(-1)
            return psi_n, core, axis

    elif read == "smooth":

        def support(psi, axis):
            psi2d = psi.reshape(nz, nr)
            ax = magnetic_axis_subgrid(psi2d, rg, zg, inside)
            ax_r = jnp.where(ax["found"], ax["r"], axis[0])
            ax_z = jnp.where(ax["found"], ax["z"], axis[1])
            axis = jnp.array([ax_r, ax_z])
            rd = boundary_read_smooth_jax(
                psi2d,
                rg,
                zg,
                inside,
                axis[0],
                axis[1],
                READ_N_LEVELS,
                READ_N_BISECT,
                READ_N_RAY,
                _DEFAULT_ANGLES,
                0.999,
                wall_r,
                wall_z,
                _NO_WALL_PSI,
                tau,
            )
            psi_axis = jnp.where(ax["found"], ax["psi"], rd["psi_axis"])
            psi_bnd = jnp.where(
                jnp.isfinite(rd["psi_bnd"]), rd["psi_bnd"], psi_axis + 1.0
            )
            span = psi_bnd - psi_axis
            span = jnp.where(jnp.abs(span) < 1e-12, 1e-12, span)
            psi_n = (psi - psi_axis) / span
            weight = rd["core_weight"].reshape(-1)
            return psi_n, weight, axis

    else:
        raise ValueError(f"read must be 'hard' or 'smooth', got {read!r}")

    def jphi_from_psi(psi, axis, pin, ip):
        psi_n, weight, axis = support(psi, axis)
        jphi, c = _scaffold(psi_n, weight, pin, ip)
        return jphi, axis, c

    def psi_from_jphi(jphi, psi_coil, ip):
        i_cell = jphi[cells] * cell_area
        total = jnp.sum(i_cell)
        scale = jnp.where(jnp.abs(total) > 1e-12, ip / total, 0.0)
        return g_mat @ (i_cell * scale) + psi_coil

    def psi_map(psi, axis, pin, ip, psi_coil):
        """One full application g(ψ) of the pinned fixed-point map."""
        jphi, axis, c = jphi_from_psi(psi, axis, pin, ip)
        return psi_from_jphi(jphi, psi_coil, ip), axis, c

    return {
        "support": support,
        "family_images": family_images,
        "coeffs_from_images": coeffs_from_images,
        "jphi_from_psi": jphi_from_psi,
        "psi_from_jphi": psi_from_jphi,
        "psi_map": psi_map,
    }


def _residual(g, psi):
    import jax.numpy as jnp

    return jnp.max(jnp.abs(g - psi)) / jnp.maximum(jnp.max(jnp.abs(g)), 1e-12)


def run_fixed_point(psi_map, psi0, axis0, pin, ip, psi_coil, n_evals, accelerator):
    """Picard / safeguarded-Anderson iteration; per-eval residual trace."""
    import jax
    import jax.numpy as jnp

    n_flat = psi0.shape[0]
    m = ANDERSON_M

    def body(i, carry):
        psi, axis, dx, df, f_prev, x_prev, norm_prev, trace = carry
        g, axis, _c = psi_map(psi, axis, pin, ip, psi_coil)
        f = g - psi
        trace = trace.at[i].set(_residual(g, psi))
        psi_pic = psi + RELAX * f
        if accelerator == "anderson":
            norm_f = jnp.max(jnp.abs(f))
            grew = norm_f > norm_prev
            dx = jnp.where(grew, jnp.zeros_like(dx), dx)
            df = jnp.where(grew, jnp.zeros_like(df), df)
            col = jnp.mod(i, m)
            dx_new = jax.lax.dynamic_update_index_in_dim(dx, psi - x_prev, col, axis=1)
            df_new = jax.lax.dynamic_update_index_in_dim(df, f - f_prev, col, axis=1)
            have = (i >= 1) & ~grew
            dx = jnp.where(have, dx_new, dx)
            df = jnp.where(have, df_new, df)
            a = df.T @ df
            a = a + 1e-10 * (jnp.trace(a) + 1e-30) * jnp.eye(m)
            gam = jnp.linalg.solve(a, df.T @ f)
            psi_and = psi + RELAX * f - (dx + RELAX * df) @ gam
            step_pic = jnp.max(jnp.abs(psi_pic - psi))
            step_and = jnp.max(jnp.abs(psi_and - psi))
            use = (
                (i >= ANDERSON_WARMUP)
                & ~grew
                & jnp.all(jnp.isfinite(psi_and))
                & (step_and <= ANDERSON_CAP * jnp.maximum(step_pic, 1e-300))
            )
            psi_next = jnp.where(use, psi_and, psi_pic)
            norm_prev = norm_f
        else:
            psi_next = psi_pic
        return psi_next, axis, dx, df, f, psi, norm_prev, trace

    init = (
        psi0,
        axis0,
        jnp.zeros((n_flat, m)),
        jnp.zeros((n_flat, m)),
        jnp.zeros(n_flat),
        psi0,
        jnp.asarray(jnp.inf, dtype=jnp.float64),
        jnp.full(n_evals, jnp.nan),
    )
    psi, axis, *_r, trace = jax.lax.fori_loop(0, n_evals, body, init)
    return psi, axis, trace


def run_newton_krylov(psi_map, psi0, axis0, pin, ip, psi_coil, n_newton, gmres_m):
    """Exact-tangent Jacobian-free Newton–Krylov on the pinned ψ-map.

    A short pinned Picard warmup, then ``n_newton`` damped Newton steps: each
    linearises the map ONCE (``jax.linearize`` — exact tangents, no finite
    differences) and solves (I − J)s = f with a fixed-shape ``gmres_m``-step
    GMRES (no early exit — vmap-safe by construction).  The axis is FROZEN
    inside each linearisation (re-read between steps): the tangent flows
    through the read scalars and the scaffold LSQ, not the integer vertex
    picks.  Cost per Newton step ≈ 1 + gmres_m map evaluations.
    Returns the per-eval residual trace on the same accounting as Picard.
    """
    import jax
    import jax.numpy as jnp

    psi, axis = psi0, axis0
    traces = []
    # warmup (counts toward the evaluation budget)
    for _ in range(NEWTON_WARMUP):
        g, axis, _c = psi_map(psi, axis, pin, ip, psi_coil)
        traces.append(float(_residual(g, psi)))
        psi = psi + RELAX * (g - psi)

    for _ in range(n_newton):
        ax_frozen = axis

        def g_of_psi(p, _ax=ax_frozen):
            out, _a2, _c = psi_map(p, _ax, pin, ip, psi_coil)
            return out

        g, jvp = jax.linearize(g_of_psi, psi)
        f = g - psi
        traces.append(float(_residual(g, psi)))

        def amat(v, _jvp=jvp):
            return v - _jvp(v)  # (I − J) v, exact tangent

        s, _info = jax.scipy.sparse.linalg.gmres(
            amat, f, maxiter=gmres_m, restart=gmres_m, solve_method="batched"
        )
        s = jnp.where(jnp.all(jnp.isfinite(s)), s, RELAX * f)
        # damping: cap the Newton step against the relaxed Picard step
        cap = 10.0 * jnp.max(jnp.abs(RELAX * f))
        norm_s = jnp.max(jnp.abs(s))
        s = jnp.where(norm_s > cap, s * (cap / jnp.maximum(norm_s, 1e-300)), s)
        psi = psi + s
        for _ in range(gmres_m):  # tangent passes on the shared accounting
            traces.append(np.nan)
        # refresh the axis on the new iterate
        g2, axis, _c = psi_map(psi, ax_frozen, pin, ip, psi_coil)
        traces.append(float(_residual(g2, psi)))

    return psi, axis, np.asarray(traces, dtype=np.float64)


def run_reduced_newton(step, psi0, axis0, pin, ip, psi_coil, n_outer, n_inner):
    """Damped Newton on the 2-coefficient scaffold map (support-lagged).

    On a FROZEN read support (ψ_N + core weight of the current iterate), jφ is
    exactly linear in the coefficient pair, so ψ(c) = c₀·(G·i_p) + c₁·(G·i_f)
    + ψ_coil is explicit and the reduced map C(c) — read ψ(c), re-fit the
    scaffold — has its fixed point in R².  Each inner Newton step costs one
    value + two tangent passes of C (jax.jacfwd), i.e. ~3 map evaluations, and
    every operation (tangents, 2×2 solve) is fixed-shape vmap-safe arithmetic.
    The support is refreshed on the outer loop (like the outer read of any
    quasi-Newton free-boundary scheme).  Returns the shared-accounting trace.
    """
    import jax
    import jax.numpy as jnp

    psi, axis = psi0, axis0
    traces = []
    for _ in range(NEWTON_WARMUP):
        g, axis, _c = step["psi_map"](psi, axis, pin, ip, psi_coil)
        traces.append(float(_residual(g, psi)))
        psi = psi + RELAX * (g - psi)

    c = None
    for _outer in range(n_outer):
        # freeze the support at the current iterate
        psi_n, weight, axis = step["support"](psi, axis)
        img_p, img_f = step["family_images"](psi_n, weight)
        if c is None:
            c = step["coeffs_from_images"](img_p, img_f, pin, ip)

        # ψ(c) is explicit on the frozen support (jφ linear in c; the Ip
        # renormalisation inside psi_from_jphi is part of C, as in the map)
        def c_map(cc, _img_p=img_p, _img_f=img_f, _axis=axis):
            jphi_c = cc[0] * _img_p + cc[1] * _img_f
            psi_c = step["psi_from_jphi"](jphi_c, psi_coil, ip)
            psi_n2, w2, _ax2 = step["support"](psi_c, _axis)
            i2p, i2f = step["family_images"](psi_n2, w2)
            return step["coeffs_from_images"](i2p, i2f, pin, ip)

        for _inner in range(n_inner):
            jac = jax.jacfwd(c_map)(c)
            r = c - c_map(c)
            traces.append(np.nan)  # value pass
            traces.append(np.nan)  # tangent pass 1
            traces.append(np.nan)  # tangent pass 2
            jr = jnp.eye(2) - jac  # d r / d c
            jr = jr + 1e-12 * jnp.eye(2)
            dc = jnp.linalg.solve(jr, r)
            dc = jnp.where(jnp.all(jnp.isfinite(dc)), dc, r)
            # damping: never step more than 2x the plain c-Picard move
            cap = 2.0 * jnp.linalg.norm(r)
            nrm = jnp.linalg.norm(dc)
            dc = jnp.where(nrm > cap, dc * (cap / jnp.maximum(nrm, 1e-300)), dc)
            c = c - dc
        # promote the converged-on-support c to a full ψ iterate + measure
        jphi_c = c[0] * img_p + c[1] * img_f
        psi = step["psi_from_jphi"](jphi_c, psi_coil, ip)
        g, axis, c_read = step["psi_map"](psi, axis, pin, ip, psi_coil)
        traces[-1] = float(_residual(g, psi))  # charge the measure to a tangent slot
        c = c_read

    return psi, axis, np.asarray(traces, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        type=str,
        default=(
            "imas_ambix/latent/artifacts/patch_gate/"
            "device_rollout_single_shot-inputs.npz"
        ),
    )
    ap.add_argument("--arms", type=str, default="picard,anderson,nk,newton2")
    ap.add_argument("--reads", type=str, default="hard,smooth")
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--tau-sweep", type=str, default="")
    ap.add_argument("--n-evals", type=int, default=40)
    ap.add_argument("--n-newton", type=int, default=4)
    ap.add_argument("--gmres-m", type=int, default=6)
    ap.add_argument("--max-slices", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--out",
        type=str,
        default=(
            "imas_ambix/latent/artifacts/patch_gate/tau_map_accelerator_probe-v0.json"
        ),
    )
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    arrs = dict(np.load(args.inputs, allow_pickle=True))
    n_slices = int(arrs["ip"].shape[0])
    idx = list(range(n_slices))
    if args.max_slices:
        idx = idx[: args.max_slices]
    logger.info(
        "probe on shot %d: %d slices, arms=%s reads=%s tau=%g",
        int(arrs["shot"]),
        len(idx),
        args.arms,
        args.reads,
        args.tau,
    )

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    reads = [r.strip() for r in args.reads.split(",") if r.strip()]
    out: dict = {
        "schema": "tau-map-accelerator-probe-v0",
        "shot": int(arrs["shot"]),
        "n_slices": len(idx),
        "tolerance": TOLERANCE,
        "constants": {
            "relax": RELAX,
            "anderson_m": ANDERSON_M,
            "newton_warmup": NEWTON_WARMUP,
            "gmres_m": args.gmres_m,
            "n_newton": args.n_newton,
            "tau": args.tau,
        },
        "legs": {},
    }

    # --- derivative health: jvp vs central FD through the full pinned map ----
    k0 = idx[len(idx) // 2]
    pin = jnp.asarray(arrs["axis_seed"][k0])
    ip0 = float(arrs["ip"][k0])
    coil0 = jnp.asarray(arrs["psi_coil"][k0])
    seed0 = jnp.asarray(arrs["disc_seed"][k0])
    deriv = {}
    for read in reads:
        step = build_step(arrs, read=read, tau=args.tau)
        pm = step["psi_map"]
        psi = step["psi_from_jphi"](seed0, coil0, ip0)
        # settle a few sweeps so the derivative is probed near the attractor
        axis = pin
        for _ in range(NEWTON_WARMUP):
            g, axis, _c = pm(psi, axis, pin, ip0, coil0)
            psi = psi + RELAX * (g - psi)
        ax_frozen = axis

        def g_scalar(p, _pm=pm, _ax=ax_frozen):
            gg, _a, _cc = _pm(p, _ax, pin, ip0, coil0)
            return gg

        rng = np.random.default_rng(0)
        v = jnp.asarray(rng.standard_normal(psi.shape[0]))
        v = v / jnp.linalg.norm(v)
        _, jv = jax.jvp(g_scalar, (psi,), (v,))
        rows = {}
        for eps_scale in (1e-4, 1e-5, 1e-6):
            eps = eps_scale * float(jnp.max(jnp.abs(psi)))
            fd = (g_scalar(psi + eps * v) - g_scalar(psi - eps * v)) / (2 * eps)
            rel = float(
                jnp.linalg.norm(jv - fd) / jnp.maximum(jnp.linalg.norm(jv), 1e-30)
            )
            rows[f"fd_rel_err_eps{eps_scale:g}"] = rel
        rows["jvp_finite"] = bool(jnp.all(jnp.isfinite(jv)))
        rows["jvp_norm"] = float(jnp.linalg.norm(jv))
        deriv[read] = rows
        logger.info("derivative health [%s]: %s", read, rows)
    out["derivative_health"] = deriv

    # --- accelerator A/B ------------------------------------------------------
    for read in reads:
        step = build_step(arrs, read=read, tau=args.tau)
        pm = step["psi_map"]
        for arm in arms:
            key = f"{read}:{arm}"
            rows = []
            t_leg = time.perf_counter()
            for k in idx:
                pin = jnp.asarray(arrs["axis_seed"][k])
                ipk = float(arrs["ip"][k])
                coil = jnp.asarray(arrs["psi_coil"][k])
                seed = jnp.asarray(arrs["disc_seed"][k])
                psi0 = step["psi_from_jphi"](seed, coil, ipk)
                if arm in ("picard", "anderson"):
                    _psi, _axis, trace = run_fixed_point(
                        pm, psi0, pin, pin, ipk, coil, args.n_evals, arm
                    )
                    tr = np.asarray(trace, dtype=np.float64)
                elif arm == "nk":
                    _psi, _axis, tr = run_newton_krylov(
                        pm, psi0, pin, pin, ipk, coil, args.n_newton, args.gmres_m
                    )
                elif arm == "newton2":
                    _psi, _axis, tr = run_reduced_newton(
                        step, psi0, pin, pin, ipk, coil, args.n_newton, 3
                    )
                else:
                    raise ValueError(f"unknown arm {arm!r}")
                finite = tr[np.isfinite(tr)]
                below = np.where(finite <= TOLERANCE)[0]
                # evals-to-tolerance on the SHARED accounting: index into the
                # full trace (NaN rows are tangent passes, they cost an eval)
                if below.size:
                    hit = np.where(np.isfinite(tr))[0][below[0]]
                    evals_to_tol = int(hit + 1)
                else:
                    evals_to_tol = None
                rows.append(
                    {
                        "k": int(k),
                        "final_residual": float(finite[-1]) if finite.size else None,
                        "evals_to_tol": evals_to_tol,
                        "n_evals": int(tr.size),
                    }
                )
            wall = time.perf_counter() - t_leg
            conv = [r for r in rows if r["evals_to_tol"] is not None]
            conv20 = [r for r in conv if r["evals_to_tol"] <= 20]
            leg = {
                "converged_frac": len(conv) / max(len(rows), 1),
                "converged_frac_at_20_evals": len(conv20) / max(len(rows), 1),
                "evals_to_tol_median": float(
                    np.median([r["evals_to_tol"] for r in conv])
                )
                if conv
                else None,
                "final_residual_median": float(
                    np.median(
                        [
                            r["final_residual"]
                            for r in rows
                            if r["final_residual"] is not None
                        ]
                    )
                ),
                "wall_s": wall,
                "rows": rows,
            }
            out["legs"][key] = leg
            logger.info(
                "%-16s conv %4.0f%% (@20: %4.0f%%) evals med %s resid med %.2e %.0fs",
                key,
                100 * leg["converged_frac"],
                100 * leg["converged_frac_at_20_evals"],
                leg["evals_to_tol_median"],
                leg["final_residual_median"],
                wall,
            )

    # --- temperature sensitivity (picard on the smooth map) ------------------
    if args.tau_sweep:
        taus = [float(t) for t in args.tau_sweep.split(",") if t.strip()]
        sweep = {}
        for tau in taus:
            step = build_step(arrs, read="smooth", tau=tau)
            pm = step["psi_map"]
            hits = []
            for k in idx[:: max(1, len(idx) // 12)]:
                pin = jnp.asarray(arrs["axis_seed"][k])
                ipk = float(arrs["ip"][k])
                coil = jnp.asarray(arrs["psi_coil"][k])
                seed_k = jnp.asarray(arrs["disc_seed"][k])
                psi0 = step["psi_from_jphi"](seed_k, coil, ipk)
                _p, _a, trace = run_fixed_point(
                    pm, psi0, pin, pin, ipk, coil, args.n_evals, "picard"
                )
                tr = np.asarray(trace)
                below = np.where(tr <= TOLERANCE)[0]
                hits.append(int(below[0] + 1) if below.size else None)
            ok = [h for h in hits if h is not None]
            sweep[f"{tau:g}"] = {
                "converged_frac": len(ok) / max(len(hits), 1),
                "evals_to_tol_median": float(np.median(ok)) if ok else None,
            }
            logger.info("tau sweep %g: %s", tau, sweep[f"{tau:g}"])
        out["tau_sweep"] = sweep

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
