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


def build_step_rollout(arrs, *, read: str, tau: float = 1e-3):
    """The production device-rollout slice step, adapted to the probe interface.

    Wraps :func:`scripts.device_rollout_single_shot._build_slice_step` (the
    corpus engine's actual per-sweep map, including its ``read=`` switch) into
    the probe's step dict so the SAME runners race the production consumer —
    the reproduction check that the re-pointed rollout map matches the probe's
    measured smooth-map convergence.  Fixed-point arms only (no family-image
    hooks, so the reduced-Newton arm is unavailable on this step source).
    """
    import jax.numpy as jnp

    from scripts.device_rollout_single_shot import _build_slice_step

    jphi_from_psi, psi_from_jphi_g = _build_slice_step(arrs, "fp64", read, tau)
    g_mat = jnp.asarray(arrs["G"])

    def psi_from_jphi(jphi, psi_coil, ip):
        psi, _i_cell = psi_from_jphi_g(g_mat, jphi, psi_coil, ip)
        return psi

    def psi_map(psi, axis, pin, ip, psi_coil):
        jphi, axis = jphi_from_psi(psi, axis, pin, ip)
        return psi_from_jphi(jphi, psi_coil, ip), axis, jnp.zeros(2)

    return {"psi_map": psi_map, "psi_from_jphi": psi_from_jphi}


def _residual(g, psi):
    import jax.numpy as jnp

    return jnp.max(jnp.abs(g - psi)) / jnp.maximum(jnp.max(jnp.abs(g)), 1e-12)


def make_fixed_point_runner(psi_map, n_evals, accelerator):
    """Jitted-once Picard / safeguarded-Anderson runner (reused across slices;
    ip/pin/coil are traced arguments so no per-slice retrace)."""
    import jax
    import jax.numpy as jnp

    m = ANDERSON_M

    @jax.jit
    def run(psi0, axis0, pin, ip, psi_coil):
        n_flat = psi0.shape[0]

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
                upd = jax.lax.dynamic_update_index_in_dim
                dx_new = upd(dx, psi - x_prev, col, axis=1)
                df_new = upd(df, f - f_prev, col, axis=1)
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

    return run


def make_warmup_runner(psi_map):
    """Jitted-once pinned-Picard warmup (shared by the Newton-type arms)."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def warmup(psi0, axis0, pin, ip, psi_coil):
        def body(i, carry):
            psi, axis, trace = carry
            g, axis, _c = psi_map(psi, axis, pin, ip, psi_coil)
            trace = trace.at[i].set(_residual(g, psi))
            return psi + RELAX * (g - psi), axis, trace

        init = (psi0, axis0, jnp.full(NEWTON_WARMUP, jnp.nan))
        return jax.lax.fori_loop(0, NEWTON_WARMUP, body, init)

    return warmup


def make_nk_runner(psi_map, gmres_m):
    """Exact-tangent Jacobian-free Newton–Krylov, jitted once per leg.

    Each Newton step linearises the map ONCE (``jax.linearize`` — exact
    tangents, no finite differences) with the axis frozen inside the
    linearisation (re-read between steps: the tangent flows through the read
    scalars and the scaffold LSQ, not the integer vertex picks), and solves
    (I − J)s = f with a fixed-shape ``gmres_m``-step GMRES — no early exit,
    vmap-safe by construction.  Cost per step ≈ 2 + gmres_m map evaluations.
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def newton_step(psi, axis, pin, ip, psi_coil):
        def g_of_psi(p):
            out, _a, _c = psi_map(p, axis, pin, ip, psi_coil)
            return out

        g, jvp = jax.linearize(g_of_psi, psi)
        f = g - psi
        resid_pre = _residual(g, psi)

        def amat(v):
            return v - jvp(v)  # (I − J) v, exact tangent

        s, _info = jax.scipy.sparse.linalg.gmres(
            amat, f, maxiter=gmres_m, restart=gmres_m, solve_method="batched"
        )
        s = jnp.where(jnp.all(jnp.isfinite(s)), s, RELAX * f)
        # damping: cap the Newton step against the relaxed Picard step
        cap = 10.0 * jnp.max(jnp.abs(RELAX * f))
        norm_s = jnp.max(jnp.abs(s))
        s = jnp.where(norm_s > cap, s * (cap / jnp.maximum(norm_s, 1e-300)), s)
        psi2 = psi + s
        g2, axis2, _c = psi_map(psi2, axis, pin, ip, psi_coil)
        return psi2, axis2, resid_pre, _residual(g2, psi2)

    warmup = make_warmup_runner(psi_map)

    def run(psi0, axis0, pin, ip, psi_coil, n_newton):
        psi, axis, wtrace = warmup(psi0, axis0, pin, ip, psi_coil)
        traces = [float(x) for x in np.asarray(wtrace)]
        for _ in range(n_newton):
            psi, axis, r_pre, r_post = newton_step(psi, axis, pin, ip, psi_coil)
            traces.append(float(r_pre))
            traces.extend([np.nan] * gmres_m)  # tangent passes, shared accounting
            traces.append(float(r_post))
        return psi, axis, np.asarray(traces, dtype=np.float64)

    return run


def make_reduced_newton_runner(step, n_inner):
    """Damped Newton on the 2-coefficient scaffold map, jitted once per leg.

    On a FROZEN read support (ψ_N + core weight of the current iterate), jφ is
    exactly linear in the coefficient pair, so ψ(c) is explicit and the reduced
    map C(c) — read ψ(c), re-fit the scaffold — has its fixed point in R².
    Each inner Newton step costs one value + two tangent passes of C
    (``jax.jacfwd``), ~3 map evaluations, all fixed-shape vmap-safe arithmetic;
    the support refreshes on the outer loop.
    """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def outer_step(psi, axis, pin, ip, psi_coil):
        psi_n, weight, axis = step["support"](psi, axis)
        img_p, img_f = step["family_images"](psi_n, weight)
        c0 = step["coeffs_from_images"](img_p, img_f, pin, ip)

        def c_map(cc):
            jphi_c = cc[0] * img_p + cc[1] * img_f
            psi_c = step["psi_from_jphi"](jphi_c, psi_coil, ip)
            psi_n2, w2, _ax2 = step["support"](psi_c, axis)
            i2p, i2f = step["family_images"](psi_n2, w2)
            return step["coeffs_from_images"](i2p, i2f, pin, ip)

        def inner(_i, c):
            jac = jax.jacfwd(c_map)(c)
            r = c - c_map(c)
            jr = jnp.eye(2) - jac + 1e-12 * jnp.eye(2)
            dc = jnp.linalg.solve(jr, r)
            dc = jnp.where(jnp.all(jnp.isfinite(dc)), dc, r)
            cap = 2.0 * jnp.linalg.norm(r)
            nrm = jnp.linalg.norm(dc)
            return c - jnp.where(nrm > cap, dc * (cap / jnp.maximum(nrm, 1e-300)), dc)

        c = jax.lax.fori_loop(0, n_inner, inner, c0)
        jphi_c = c[0] * img_p + c[1] * img_f
        psi2 = step["psi_from_jphi"](jphi_c, psi_coil, ip)
        g, axis2, _c = step["psi_map"](psi2, axis, pin, ip, psi_coil)
        return psi2, axis2, _residual(g, psi2)

    warmup = make_warmup_runner(step["psi_map"])

    def run(psi0, axis0, pin, ip, psi_coil, n_outer):
        psi, axis = psi0, axis0
        psi, axis, wtrace = warmup(psi0, axis0, pin, ip, psi_coil)
        traces = [float(x) for x in np.asarray(wtrace)]
        for _ in range(n_outer):
            psi, axis, resid = outer_step(psi, axis, pin, ip, psi_coil)
            # 1 read + n_inner*(1 value + 2 tangents) + 1 promote/measure
            traces.extend([np.nan] * (1 + 3 * n_inner))
            traces.append(float(resid))
        return psi, axis, np.asarray(traces, dtype=np.float64)

    return run


def _tau_figure(out: dict, path: Path) -> None:
    """The τ-calibration figure: sweep, continuation-vs-single, emit residual."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = out.get("tau_sweep", {})
    sched = out.get("tau_schedule", {})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    a0 = axes[0]
    if sweep:
        taus = sorted(sweep, key=float)
        cf = [100 * sweep[t]["converged_frac"] for t in taus]
        a0.semilogx([float(t) for t in taus], cf, "o-", color="#268")
        a0.axvline(1e-3, color="k", ls="--", lw=0.8, label="read accuracy point")
        a0.legend(fontsize=8)
    a0.set_title("Picard converged fraction vs τ (single temperature)")
    a0.set_xlabel("smoothing temperature τ")
    a0.set_ylabel("converged [%]")

    a1 = axes[1]
    bars = {f"τ={t}": sweep[t]["converged_frac"] for t in sorted(sweep, key=float)}
    bars.update({f"anneal {k}": v["converged_frac"] for k, v in sched.items()})
    if bars:
        x = np.arange(len(bars))
        cols = ["#268"] * len(sweep) + ["#2a7"] * len(sched)
        a1.bar(x, [100 * v for v in bars.values()], color=cols)
        a1.set_xticks(x)
        a1.set_xticklabels(list(bars), fontsize=7, rotation=20, ha="right")
    a1.set_title(
        f"single τ vs continuation at equal budget "
        f"({out['constants'].get('n_evals', '?')} evals)"
    )
    a1.set_ylabel("converged [%]")

    a2 = axes[2]
    rows = {
        f"τ={t}": sweep[t]["final_residual_median"] for t in sorted(sweep, key=float)
    }
    rows.update({f"anneal {k}": v["final_residual_median"] for k, v in sched.items()})
    if rows:
        x = np.arange(len(rows))
        cols = ["#268"] * len(sweep) + ["#2a7"] * len(sched)
        a2.bar(x, list(rows.values()), color=cols)
        a2.set_yscale("log")
        a2.axhline(TOLERANCE, color="k", ls="--", lw=0.8, label="tolerance")
        a2.set_xticks(x)
        a2.set_xticklabels(list(rows), fontsize=7, rotation=20, ha="right")
        a2.legend(fontsize=8)
    a2.set_title("median final residual (emit map for schedules)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


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
    ap.add_argument(
        "--tau-sweep-full",
        action="store_true",
        help="run the tau sweep on EVERY slice (default: ~12-slice subsample)",
    )
    ap.add_argument(
        "--tau-schedule",
        type=str,
        default="",
        help="tau-continuation study: ';'-separated schedules, each a comma "
        "list hot->cold (e.g. '1e-2,3e-3,1e-3;3e-3,1e-3'); each schedule "
        "splits --n-evals equally across its stages, so it races the "
        "single-temperature arms at EQUAL total evaluations; convergence "
        "is scored on the FINAL stage only (labels emit at the cold read)",
    )
    ap.add_argument(
        "--step-source",
        choices=["probe", "rollout"],
        default="probe",
        help="build the map from the probe's own step or from the production "
        "device-rollout slice step (fixed-point arms only)",
    )
    ap.add_argument(
        "--figure",
        type=str,
        default="",
        help="write the tau-calibration figure (sweep + schedule panels) here",
    )
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
    if args.step_source == "rollout":
        builder = build_step_rollout
        bad = [a for a in arms if a not in ("picard", "anderson")]
        if bad:
            raise SystemExit(
                f"--step-source rollout supports fixed-point arms only, got {bad}"
            )
    else:
        builder = build_step
    out: dict = {
        "schema": "tau-map-accelerator-probe-v0",
        "step_source": args.step_source,
        "shot": int(arrs["shot"]),
        "n_slices": len(idx),
        "tolerance": TOLERANCE,
        "constants": {
            "n_evals": args.n_evals,
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
        step = builder(arrs, read=read, tau=args.tau)
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
        step = builder(arrs, read=read, tau=args.tau)
        pm = step["psi_map"]
        for arm in arms:
            key = f"{read}:{arm}"
            rows = []
            t_leg = time.perf_counter()
            if arm in ("picard", "anderson"):
                runner = make_fixed_point_runner(pm, args.n_evals, arm)
            elif arm == "nk":
                runner = make_nk_runner(pm, args.gmres_m)
            elif arm == "newton2":
                runner = make_reduced_newton_runner(step, 3)
            else:
                raise ValueError(f"unknown arm {arm!r}")
            for k in idx:
                pin = jnp.asarray(arrs["axis_seed"][k])
                ipk = jnp.asarray(float(arrs["ip"][k]))
                coil = jnp.asarray(arrs["psi_coil"][k])
                seed = jnp.asarray(arrs["disc_seed"][k])
                psi0 = step["psi_from_jphi"](seed, coil, ipk)
                if arm in ("picard", "anderson"):
                    _psi, _axis, trace = runner(psi0, pin, pin, ipk, coil)
                    tr = np.asarray(trace, dtype=np.float64)
                else:
                    _psi, _axis, tr = runner(psi0, pin, pin, ipk, coil, args.n_newton)
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
        sweep_idx = idx if args.tau_sweep_full else idx[:: max(1, len(idx) // 12)]
        sweep = {}
        for tau in taus:
            step = builder(arrs, read="smooth", tau=tau)
            runner = make_fixed_point_runner(step["psi_map"], args.n_evals, "picard")
            hits = []
            finals = []
            for k in sweep_idx:
                pin = jnp.asarray(arrs["axis_seed"][k])
                ipk = jnp.asarray(float(arrs["ip"][k]))
                coil = jnp.asarray(arrs["psi_coil"][k])
                seed_k = jnp.asarray(arrs["disc_seed"][k])
                psi0 = step["psi_from_jphi"](seed_k, coil, ipk)
                _p, _a, trace = runner(psi0, pin, pin, ipk, coil)
                tr = np.asarray(trace)
                below = np.where(tr <= TOLERANCE)[0]
                hits.append(int(below[0] + 1) if below.size else None)
                finals.append(float(tr[np.isfinite(tr)][-1]))
            ok = [h for h in hits if h is not None]
            sweep[f"{tau:g}"] = {
                "n_slices": len(sweep_idx),
                "converged_frac": len(ok) / max(len(hits), 1),
                "evals_to_tol_median": float(np.median(ok)) if ok else None,
                "final_residual_median": float(np.median(finals)),
            }
            logger.info("tau sweep %g: %s", tau, sweep[f"{tau:g}"])
        out["tau_sweep"] = sweep

    # --- tau continuation (anneal hot -> cold at equal total evaluations) ----
    if args.tau_schedule:
        schedules = [
            [float(t) for t in sched.split(",") if t.strip()]
            for sched in args.tau_schedule.split(";")
            if sched.strip()
        ]
        out["tau_schedule"] = {}
        for taus in schedules:
            key = ",".join(f"{t:g}" for t in taus)
            n_stage = [args.n_evals // len(taus)] * len(taus)
            n_stage[-1] += args.n_evals - sum(n_stage)
            stages = [
                (
                    make_fixed_point_runner(
                        builder(arrs, read="smooth", tau=t)["psi_map"], n, "picard"
                    ),
                    n,
                )
                for t, n in zip(taus, n_stage, strict=True)
            ]
            step_cold = builder(arrs, read="smooth", tau=taus[-1])
            hits = []
            finals = []
            for k in idx:
                pin = jnp.asarray(arrs["axis_seed"][k])
                ipk = jnp.asarray(float(arrs["ip"][k]))
                coil = jnp.asarray(arrs["psi_coil"][k])
                seed_k = jnp.asarray(arrs["disc_seed"][k])
                psi = step_cold["psi_from_jphi"](seed_k, coil, ipk)
                axis = pin
                spent = 0
                hit = None
                tr_last = np.empty(0)
                for runner, n in stages:
                    psi, axis, trace = runner(psi, axis, pin, ipk, coil)
                    tr_last = np.asarray(trace)
                    spent += n
                # convergence is scored on the FINAL (cold) stage only — a
                # tolerance hit on a hot map is not an emit-grade residual
                below = np.where(tr_last <= TOLERANCE)[0]
                if below.size:
                    hit = spent - n_stage[-1] + int(below[0]) + 1
                hits.append(hit)
                finals.append(float(tr_last[np.isfinite(tr_last)][-1]))
            ok = [h for h in hits if h is not None]
            out["tau_schedule"][key] = {
                "stage_evals": n_stage,
                "n_slices": len(idx),
                "converged_frac": len(ok) / max(len(hits), 1),
                "evals_to_tol_median": float(np.median(ok)) if ok else None,
                "final_residual_median": float(np.median(finals)),
            }
            logger.info("tau schedule [%s]: %s", key, out["tau_schedule"][key])

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", path)
    if args.figure:
        _tau_figure(out, Path(args.figure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
