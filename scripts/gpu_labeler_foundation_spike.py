"""Foundation spike for the batched-over-shots GPU equilibrium labeller.

The per-solve GPU port is a confirmed dead end (the grid-free Green's matvec is
~1.7x slower than the cached grid solve at batch 1).  The cure the expert
reviews converged on is to batch over the independent axis (many equilibria at
once), keep the whole Picard rollout on device, and read the topology on device
too.  This spike de-risks the substrate that every deployment dimension shares,
BEFORE any rollout is built, in three measured legs:

  Leg A1 (framework) — the inner GS step (psi = G @ i_cell + psi_coil, a
    force-balanced two-term jphi(psi_N), Ip renormalisation) prototyped in BOTH
    jax and torch, batched, same dtype, so the framework choice rests on
    measured step time plus the qualitative factors (the device topology read
    exists only in jax; autograd through the rollout is a strategic need).

  Leg A2 (GEMM crossover) — psi = G @ I microbenched with the REAL campaign
    Green's matrix over a batch sweep in fp64 / fp32 / tf32 / bf16, to find the
    compute-bound crossover and the achievable slices/s per dtype.

  Leg A3 (batched solve + on-device read + precision) — a fixed-shape Picard
    equilibrium solve fully on device (vmap over the batch), consuming the
    connectivity boundary read for the axis / binding flux / core region, scored
    for axis reproduction against the host reference solve, then re-run with the
    matvec at tf32 and bf16 (fp32 accumulate) to measure the precision cost.  A
    GPU/CPU parity check on the fp64 batched solve mirrors the FSA capability
    demo.

Two stages so data prep (needs filesystem + a resolved venv) is separable from
the device run:

    # login node — stage the held-out payloads + host reference into an NPZ
    uv run python -m scripts.gpu_labeler_foundation_spike --stage prep

    # H200 (reservation mandatory) — run the three legs against the staged NPZ
    srun --partition=betelgeuse --reservation=gpu_0003_grpA --gres=gpu:1 \
         --cpus-per-task=4 --mem=64G --time=00:45:00 \
      bash -lc 'export TMPDIR=/tmp; cd <repo>; \
        uv run python -m scripts.gpu_labeler_foundation_spike --stage gpu'

``--stage all`` (default) runs prep then the legs — a CPU smoke everywhere, a
genuine GPU demonstration where a CUDA jaxlib is present.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

ARTIFACT = Path("imas_ambix/latent/artifacts/patch_gate/gpu_labeler_foundation-v0.json")
STAGE_NPZ = Path(
    "imas_ambix/latent/artifacts/patch_gate/gpu_labeler_foundation-inputs.npz"
)
FIG_DIR = Path("docs/figures/gpu-accelerated-labeler")

# fixed profile parameters θ for the spike's inner GS step (the two-term shape)
BETA0 = 0.5
ALPHA = 1.0
# reproduction bar for the batched fp64 on-device solve vs the host reference
AXIS_TOL_CM = 2.0
# in-loop connectivity-read resolution (lighter than the host default so the
# fixed-iteration rollout stays affordable batched; the physical fixed point is
# unchanged — this only coarsens the per-iterate binding sweep)
READ_N_LEVELS = 48
READ_N_BISECT = 12
READ_N_RAY = 96
PICARD_ITERS = 60
RELAX = 0.5
SEED_WIDTH = (0.35, 0.5)


def _run(cmd) -> str:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# stage 1 — payload + host-reference preparation (login node)
# ---------------------------------------------------------------------------


def stage_prep(nr: int, nz: int, max_slices: int, min_ip_ka: float, max_shots: int):
    """Assemble held-out slices sharing one campaign grid into an NPZ.

    Builds the shared :class:`EquilibriumGrid` from the first held-out shot, the
    plasma Green's matrix and per-slice vacuum coil flux on that grid, and the
    host reference equilibrium (grid-free Picard) axis for each unique slice —
    everything the device legs consume with no further data access.
    """
    from imas_ambix.eval import prediction_bar as pbar
    from imas_ambix.latent.connectivity_boundary import _densify_wall
    from imas_ambix.latent.gs_solve import (
        SUBSTRATE_GREENS,
        solve_equilibrium,
    )
    from scripts.heldout_mse_gate_eval import _campaign_table
    from scripts.spine_label_factory import factory_shot_payloads

    manifest = pbar.load_locked_manifest()
    shots = list(pbar.held_out_shot_ids(manifest))
    print(f"held-out shots: {shots}")

    shared_grid = None
    shared_campaign = None
    G = None
    psi_coil_list: list[np.ndarray] = []
    ip_list: list[float] = []
    ref_axis_list: list[list[float]] = []
    ref_axis_psi: list[float] = []
    ref_bnd: list[float] = []
    ref_converged: list[bool] = []
    slice_tag: list[str] = []

    used_shots = 0
    for shot in shots:
        if used_shots >= max_shots:
            break
        table = _campaign_table(int(shot))
        if table is None:
            print(f"  shot {shot}: no campaign table, skip")
            continue
        payload = factory_shot_payloads(
            int(shot),
            nr=nr,
            nz=nz,
            max_slices=max_slices,
            min_ip_ka=min_ip_ka,
            table=table,
        )
        if payload is None:
            print(f"  shot {shot}: no payloads, skip")
            continue
        campaign = payload["campaign"]
        if shared_grid is None:
            shared_grid = payload["grid"]
            shared_campaign = campaign
            G = np.asarray(shared_grid.plasma_grid_psi_columns(), dtype=np.float64)
            print(
                f"  shared grid from shot {shot} campaign {campaign}: "
                f"G {G.shape} n_cells={shared_grid.cells.size}"
            )
        elif campaign != shared_campaign:
            print(f"  shot {shot}: campaign {campaign} != {shared_campaign}, skip")
            continue

        grid = shared_grid
        for p in payload["payloads"]:
            i_pf = np.asarray(p.i_pf, dtype=np.float64)
            ip = float(p.ip_amperes)
            psi_coil = np.asarray(grid.coil_psi(i_pf), dtype=np.float64)
            res = solve_equilibrium(
                grid,
                i_pf,
                ip,
                beta0=BETA0,
                alpha=ALPHA,
                max_iterations=200,
                coil_field_mode="analytic-add",
                substrate=SUBSTRATE_GREENS,
            )
            psi_coil_list.append(psi_coil)
            ip_list.append(ip)
            ref_axis_list.append([float(res.axis[0]), float(res.axis[1])])
            ref_axis_psi.append(float(res.axis_psi))
            ref_bnd.append(float(res.boundary_psi))
            ref_converged.append(bool(res.converged))
            slice_tag.append(f"{shot}:{p.t_index}")
        used_shots += 1
        print(f"  shot {shot}: {len(payload['payloads'])} slices staged")

    if shared_grid is None or not ip_list:
        raise SystemExit("no usable held-out slices staged")

    wall_r, wall_z = _densify_wall(shared_grid)
    STAGE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        STAGE_NPZ,
        rg=np.asarray(shared_grid.rg, dtype=np.float64),
        zg=np.asarray(shared_grid.zg, dtype=np.float64),
        inside=np.asarray(shared_grid.inside_limiter, dtype=bool),
        flat_r=np.asarray(shared_grid.flat_r, dtype=np.float64),
        flat_z=np.asarray(shared_grid.flat_z, dtype=np.float64),
        cells=np.asarray(shared_grid.cells, dtype=np.int64),
        limiter_r=np.asarray(shared_grid.limiter_r, dtype=np.float64),
        limiter_z=np.asarray(shared_grid.limiter_z, dtype=np.float64),
        wall_r=np.asarray(wall_r, dtype=np.float64),
        wall_z=np.asarray(wall_z, dtype=np.float64),
        G=G,
        r0=np.float64(shared_grid.r0),
        dr=np.float64(shared_grid.dr),
        dz=np.float64(shared_grid.dz),
        psi_coil=np.asarray(psi_coil_list, dtype=np.float64),
        ip=np.asarray(ip_list, dtype=np.float64),
        ref_axis=np.asarray(ref_axis_list, dtype=np.float64),
        ref_axis_psi=np.asarray(ref_axis_psi, dtype=np.float64),
        ref_bnd=np.asarray(ref_bnd, dtype=np.float64),
        ref_converged=np.asarray(ref_converged, dtype=bool),
        slice_tag=np.asarray(slice_tag),
        campaign=np.asarray(str(shared_campaign)),
        nr=np.int64(nr),
        nz=np.int64(nz),
    )
    print(f"staged {len(ip_list)} unique slices from {used_shots} shots → {STAGE_NPZ}")


# ---------------------------------------------------------------------------
# device kernels (jax) — the inner GS step, the batched Picard solve
# ---------------------------------------------------------------------------


def _matvec(G, x, mode):
    """psi_plasma = G @ x under a precision ``mode`` (result always cast to f64).

    ``mode`` is a host-level string, so the branch is resolved at trace time —
    only the matmul precision changes.  fp32/tf32/bf16 touch ONLY this matvec;
    the caller keeps every cross-iteration state and the topology read in fp64.
    """
    import jax
    import jax.numpy as jnp

    if mode == "fp64":
        return G @ x
    if mode == "fp32":
        return jnp.matmul(
            G.astype(jnp.float32),
            x.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ).astype(jnp.float64)
    if mode == "tf32":
        return jnp.matmul(
            G.astype(jnp.float32),
            x.astype(jnp.float32),
            precision=jax.lax.Precision.DEFAULT,
        ).astype(jnp.float64)
    if mode == "bf16":
        return jnp.dot(
            G.astype(jnp.bfloat16),
            x.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.float64)
    raise ValueError(f"unknown matvec mode {mode!r}")


def _build_batched_solver(arrs, mode: str):
    """Return a jitted vmapped fixed-shape Picard solver for one precision mode.

    The solver reproduces the host grid-free Picard (analytic-add, greens-matvec
    substrate): each iterate renormalises the force-balanced filament currents to
    Ip, evaluates ψ by the Green's matvec + vacuum coil flux, reads the
    connectivity boundary (axis, binding flux, axis-connected core), and applies
    the two-term jφ(ψ_N) inside the core.  Only the matvec precision varies.
    """
    import jax
    import jax.numpy as jnp

    from imas_ambix.latent.connectivity_boundary import (
        _DEFAULT_ANGLES,
        boundary_read_jax,
    )
    from imas_ambix.latent.flux_surface_connectivity import flood_fill_core

    # only the small geometry arrays are closed over (XLA may constant-fold them
    # harmlessly); the large Green's matrix G is a TRACED argument so it is never
    # baked in as a folded constant — that would poison both compile time and the
    # matvec timing.
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
    nz = int(arrs["nz"])
    nr = int(arrs["nr"])
    n_flood = nr + nz

    def one(G, psi_coil, ip, seed_axis):
        base = BETA0 * flat_r / r0 + (1.0 - BETA0) * r0 / jnp.maximum(flat_r, 1e-3)

        # cold Gaussian seed jφ at the geometric centre (matches the host seed)
        gseed = jnp.exp(
            -(
                ((flat_r - r0) / SEED_WIDTH[0]) ** 2
                + ((flat_z - 0.0) / SEED_WIDTH[1]) ** 2
            )
        )
        jphi0 = jnp.where(inside.reshape(-1), gseed, 0.0)

        def body(i, carry):
            psi, jphi, axis = carry
            i_cell = jphi[cells] * cell_area
            total = jnp.sum(i_cell)
            scale = jnp.where(jnp.abs(total) > 1e-12, ip / total, 0.0)
            i_cell = i_cell * scale
            psi_new = _matvec(G, i_cell, mode) + psi_coil
            psi = jnp.where(i == 0, psi_new, RELAX * psi_new + (1.0 - RELAX) * psi)

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
            core = flood_fill_core(confined, seed2d, n_flood)
            shape = base * jnp.clip(1.0 - psi_n, 0.0, None) ** ALPHA
            jphi = shape * core.reshape(-1)
            return psi, jphi, axis

        psi_init = jnp.zeros_like(psi_coil)
        psi, jphi, axis = jax.lax.fori_loop(
            0, PICARD_ITERS, body, (psi_init, jphi0, seed_axis)
        )
        # one final force-balance residual on the greens-matvec map
        i_cell = jphi[cells] * cell_area
        total = jnp.sum(i_cell)
        scale = jnp.where(jnp.abs(total) > 1e-12, ip / total, 0.0)
        i_cell = i_cell * scale
        psi_map = _matvec(G, i_cell, "fp64") + psi_coil
        resid = jnp.max(jnp.abs(psi_map - psi)) / jnp.maximum(
            jnp.max(jnp.abs(psi_map)), 1e-12
        )
        return {"axis": axis, "residual": resid, "psi": psi}

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))


# ---------------------------------------------------------------------------
# stage 2 — the three device legs
# ---------------------------------------------------------------------------


def framework_step_compare(arrs, batch: int) -> dict:
    """Inner GS step timed in jax and torch at the same batch and dtype (fp64).

    The step is the arithmetic core of one Picard iterate — matvec, two-term
    jφ(ψ_N) with a GIVEN normalisation, Ip renormalisation — WITHOUT the topology
    read (torch has no device topology read; that asymmetry is the qualitative
    factor reported alongside the numbers, not something to time here).
    """
    import jax
    import jax.numpy as jnp

    G = arrs["G"]
    n_grid, n_cells = G.shape
    flat_r = arrs["flat_r"]
    r0 = float(arrs["r0"])
    cell_area = float(arrs["dr"]) * float(arrs["dz"])
    rng = np.random.default_rng(0)
    i_cell0 = np.abs(rng.standard_normal((n_cells, batch))).astype(np.float64)
    psi_coil = arrs["psi_coil"][0].astype(np.float64)
    axis_psi = np.float64(-0.05)
    bnd_psi = np.float64(0.0)
    base = BETA0 * flat_r / r0 + (1.0 - BETA0) * r0 / np.maximum(flat_r, 1e-3)

    out = {"batch": batch, "dtype": "fp64"}

    # --- jax --- (G traced, not folded — see leg_a2)
    Gj = jnp.asarray(G)
    icj = jnp.asarray(i_cell0)
    pcj = jnp.asarray(psi_coil)[:, None]
    basej = jnp.asarray(base)[:, None]
    cells = jnp.asarray(arrs["cells"])

    def jstep(G, i_cell):
        total = jnp.sum(i_cell, axis=0, keepdims=True)
        i_cell = i_cell * jnp.where(jnp.abs(total) > 1e-12, 1.0e5 / total, 0.0)
        psi = G @ i_cell + pcj
        psi_n = (psi - axis_psi) / (bnd_psi - axis_psi)
        shape = basej * jnp.clip(1.0 - psi_n, 0.0, None)
        jphi = jnp.where(psi_n < 1.0, shape, 0.0)
        ic = jphi[cells] * cell_area
        return ic

    jf = jax.jit(jstep)
    r = jf(Gj, icj)
    jax.block_until_ready(r)
    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        r = jf(Gj, icj)
    jax.block_until_ready(r)
    jax_wall = (time.perf_counter() - t0) / reps
    devs = jax.devices()
    out["jax_devices"] = [f"{d.platform}:{d.id}" for d in devs]
    out["jax_on_gpu"] = any(d.platform == "gpu" for d in devs)
    out["jax_step_ms"] = round(jax_wall * 1e3, 4)
    out["jax_slices_per_s"] = round(batch / jax_wall, 1)

    # --- torch ---
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        dev = torch.device("cuda" if cuda_ok else "cpu")
        Gt = torch.as_tensor(G, dtype=torch.float64, device=dev)
        ict = torch.as_tensor(i_cell0, dtype=torch.float64, device=dev)
        pct = torch.as_tensor(psi_coil, dtype=torch.float64, device=dev)[:, None]
        baset = torch.as_tensor(base, dtype=torch.float64, device=dev)[:, None]
        cells_t = torch.as_tensor(np.asarray(arrs["cells"]), device=dev)

        def tstep(i_cell):
            total = i_cell.sum(0, keepdim=True)
            i_cell = i_cell * torch.where(
                total.abs() > 1e-12, 1.0e5 / total, torch.zeros_like(total)
            )
            psi = Gt @ i_cell + pct
            psi_n = (psi - float(axis_psi)) / (float(bnd_psi) - float(axis_psi))
            shape = baset * torch.clamp(1.0 - psi_n, min=0.0)
            jphi = torch.where(psi_n < 1.0, shape, torch.zeros_like(shape))
            return jphi[cells_t] * cell_area

        r = tstep(ict)
        if cuda_ok:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            r = tstep(ict)
        if cuda_ok:
            torch.cuda.synchronize()
        torch_wall = (time.perf_counter() - t0) / reps
        out["torch_cuda_available"] = bool(cuda_ok)
        out["torch_device"] = str(dev)
        out["torch_step_ms"] = round(torch_wall * 1e3, 4)
        out["torch_slices_per_s"] = round(batch / torch_wall, 1)
    except Exception as exc:  # noqa: BLE001
        out["torch_error"] = str(exc)

    out["qualitative"] = {
        "device_topology_read": "jax-only (connectivity_boundary is JAX; torch "
        "has no device port of the boundary read / flood-fill / stencil nulls)",
        "autograd_through_rollout": "jax vmap+grad through the fixed-point map is "
        "a strategic need (learned operator handoff); torch would need a rewrite",
        "temporal_stack": "the temporal world-model stack is torch-fp64",
    }
    return out


def gemm_crossover_bench(arrs, batches: list[int]) -> dict:
    """psi = G @ I microbench over a batch sweep in fp64/fp32/tf32/bf16."""
    import jax
    import jax.numpy as jnp

    G = arrs["G"]
    n_grid, n_cells = G.shape
    rng = np.random.default_rng(1)
    devs = jax.devices()
    on_gpu = any(d.platform == "gpu" for d in devs)
    # G is a TRACED argument (not closed over) so XLA never constant-folds the
    # 6305×n_cells matrix — that would both stall compilation and fold away the
    # matmul we are timing.
    G_dev = jnp.asarray(G)

    def mk(mode):
        def f(G, I):
            return _matvec(G, I, mode)

        return jax.jit(f)

    modes = ["fp64", "fp32", "tf32", "bf16"]
    fns = {m: mk(m) for m in modes}
    results: dict[str, list[dict]] = {m: [] for m in modes}
    for B in batches:
        I = jnp.asarray(np.abs(rng.standard_normal((n_cells, B))).astype(np.float64))
        flop = 2.0 * n_grid * n_cells * B
        for m in modes:
            f = fns[m]
            r = f(G_dev, I)
            jax.block_until_ready(r)
            reps = 20
            t0 = time.perf_counter()
            for _ in range(reps):
                r = f(G_dev, I)
            jax.block_until_ready(r)
            wall = (time.perf_counter() - t0) / reps
            results[m].append(
                {
                    "batch": B,
                    "wall_ms": round(wall * 1e3, 5),
                    "slices_per_s": round(B / wall, 1),
                    "tflop_s": round(flop / wall / 1e12, 3),
                }
            )
            print(
                f"  GEMM {m:5s} B={B:5d}  {wall * 1e3:8.3f} ms  "
                f"{B / wall:10.1f} sl/s  {flop / wall / 1e12:7.3f} TFLOP/s"
            )

    # crossover: the largest B where slices/s is still within 15% of a linear
    # extrapolation from the smallest batch (i.e. throughput still ~scaling)
    crossover: dict[str, int | None] = {}
    for m in modes:
        rows = results[m]
        base_rate = rows[0]["slices_per_s"] / rows[0]["batch"]
        cx = None
        for row in rows:
            ideal = base_rate * row["batch"]
            if row["slices_per_s"] >= 0.85 * ideal:
                cx = row["batch"]
        crossover[m] = cx
    return {
        "on_gpu": on_gpu,
        "devices": [f"{d.platform}:{d.id}" for d in devs],
        "G_shape": [int(n_grid), int(n_cells)],
        "batches": batches,
        "by_dtype": results,
        "linear_scaling_ceiling": crossover,
    }


def _axis_cm(a, b) -> float:
    return float(100.0 * np.hypot(a[0] - b[0], a[1] - b[1]))


def batched_solve_reproduction(arrs, batch: int) -> dict:
    """Batched fixed-shape on-device Picard: reproduction + precision + parity."""
    import jax
    import jax.numpy as jnp

    n_unique = int(arrs["ip"].shape[0])
    devs = jax.devices()
    on_gpu = any(d.platform == "gpu" for d in devs)

    # tile the unique slices to fill the batch; score only the unique ones
    reps = int(np.ceil(batch / n_unique))
    idx = np.tile(np.arange(n_unique), reps)[:batch]
    psi_coil_b = jnp.asarray(arrs["psi_coil"][idx])
    ip_b = jnp.asarray(arrs["ip"][idx])
    # seed axis = geometric centre for every slice (the read refines it)
    r0 = float(arrs["r0"])
    seed_axis_b = jnp.asarray(
        np.tile(np.array([r0, 0.0]), (batch, 1)).astype(np.float64)
    )

    ref_axis = arrs["ref_axis"]
    ref_conv = arrs["ref_converged"]
    G_dev = jnp.asarray(arrs["G"])

    out: dict = {
        "on_gpu": on_gpu,
        "devices": [f"{d.platform}:{d.id}" for d in devs],
        "batch": batch,
        "n_unique": n_unique,
        "picard_iters": PICARD_ITERS,
        "read_resolution": {
            "n_levels": READ_N_LEVELS,
            "n_bisect": READ_N_BISECT,
            "n_ray": READ_N_RAY,
        },
        "axis_tol_cm": AXIS_TOL_CM,
    }

    # --- fp64 batched solve on the active device ---
    solver = _build_batched_solver(arrs, "fp64")
    t0 = time.perf_counter()
    res = solver(G_dev, psi_coil_b, ip_b, seed_axis_b)
    jax.block_until_ready(res["axis"])
    wall_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = solver(G_dev, psi_coil_b, ip_b, seed_axis_b)
    jax.block_until_ready(res["axis"])
    wall = time.perf_counter() - t0
    axis_dev = np.asarray(res["axis"])
    resid_dev = np.asarray(res["residual"])
    out["fp64_wall_s"] = round(wall, 4)
    out["fp64_compile_s"] = round(wall_compile, 2)
    out["fp64_solves_per_s"] = round(batch / wall, 1)
    out["fp64_is_f64"] = bool(axis_dev.dtype == np.float64)

    # reproduction: device axis vs host reference axis, unique converged slices
    dev_axis_unique = axis_dev[:n_unique]
    dcm = []
    pair_dev_r, pair_dev_z, pair_ref_r, pair_ref_z = [], [], [], []
    for j in range(n_unique):
        if not bool(ref_conv[j]):
            continue
        a = dev_axis_unique[j]
        if not np.all(np.isfinite(a)):
            continue
        dcm.append(_axis_cm(a, ref_axis[j]))
        pair_dev_r.append(float(a[0]))
        pair_dev_z.append(float(a[1]))
        pair_ref_r.append(float(ref_axis[j][0]))
        pair_ref_z.append(float(ref_axis[j][1]))
    dcm = np.asarray(dcm)
    axis_med = float(np.median(dcm)) if dcm.size else float("nan")
    axis_p90 = float(np.percentile(dcm, 90)) if dcm.size else float("nan")
    out["reproduction"] = {
        "n_scored": int(dcm.size),
        "n_ref_converged": int(np.sum(ref_conv)),
        "axis_median_cm": axis_med,
        "axis_p90_cm": axis_p90,
        "residual_median": float(np.median(resid_dev)),
        "pass": bool(np.isfinite(axis_med) and axis_med <= AXIS_TOL_CM),
        "pairs": {
            "dev_r": pair_dev_r,
            "dev_z": pair_dev_z,
            "ref_r": pair_ref_r,
            "ref_z": pair_ref_z,
        },
    }

    # --- precision tiers vs the fp64 on-device baseline ---
    precision = {}
    for mode in ("tf32", "bf16"):
        try:
            sv = _build_batched_solver(arrs, mode)
            r = sv(G_dev, psi_coil_b, ip_b, seed_axis_b)
            jax.block_until_ready(r["axis"])
            t0 = time.perf_counter()
            r = sv(G_dev, psi_coil_b, ip_b, seed_axis_b)
            jax.block_until_ready(r["axis"])
            w = time.perf_counter() - t0
            a_unique = np.asarray(r["axis"])[:n_unique]
            deltas = []
            for j in range(n_unique):
                a0, a1 = dev_axis_unique[j], a_unique[j]
                if np.all(np.isfinite(a0)) and np.all(np.isfinite(a1)):
                    deltas.append(_axis_cm(a0, a1))
            deltas = np.asarray(deltas)
            precision[mode] = {
                "wall_s": round(w, 4),
                "solves_per_s": round(batch / w, 1),
                "axis_delta_vs_fp64_median_cm": float(np.median(deltas))
                if deltas.size
                else float("nan"),
                "axis_delta_vs_fp64_p90_cm": float(np.percentile(deltas, 90))
                if deltas.size
                else float("nan"),
            }
            print(
                f"  precision {mode}: median Δaxis vs fp64 = "
                f"{precision[mode]['axis_delta_vs_fp64_median_cm']:.3f} cm, "
                f"{precision[mode]['solves_per_s']:.1f} solves/s"
            )
        except Exception as exc:  # noqa: BLE001
            precision[mode] = {"error": str(exc)}
    out["precision"] = precision

    # --- GPU/CPU parity of the fp64 batched solve (small batch) ---
    small = min(8, batch)
    parity_batch = (psi_coil_b[:small], ip_b[:small], seed_axis_b[:small])
    with jax.default_device(jax.devices("cpu")[0]):
        cpu_solver = _build_batched_solver(arrs, "fp64")
        rc = cpu_solver(
            jnp.asarray(np.asarray(arrs["G"])),
            jnp.asarray(np.asarray(parity_batch[0])),
            jnp.asarray(np.asarray(parity_batch[1])),
            jnp.asarray(np.asarray(parity_batch[2])),
        )
        jax.block_until_ready(rc["axis"])
    axis_cpu = np.asarray(rc["axis"])
    axis_dev_small = axis_dev[:small]
    fin = np.all(np.isfinite(axis_cpu), axis=1) & np.all(
        np.isfinite(axis_dev_small), axis=1
    )
    parity_max = (
        float(np.max(np.abs(axis_cpu[fin] - axis_dev_small[fin])))
        if fin.any()
        else float("nan")
    )
    out["parity"] = {
        "small_batch": small,
        "max_abs_axis_diff_m": parity_max,
        "pass": bool(np.isfinite(parity_max) and parity_max < 1e-9) if on_gpu else None,
    }
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _figures(a2: dict, a3: dict, arrs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # fig 1: GEMM crossover — slices/s and TFLOP/s vs B per dtype
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"fp64": "#268", "fp32": "#484", "tf32": "#c73", "bf16": "#a3a"}
    for m, rows in a2["by_dtype"].items():
        B = [r["batch"] for r in rows]
        sl = [r["slices_per_s"] for r in rows]
        tf = [r["tflop_s"] for r in rows]
        ax1.loglog(B, sl, "o-", color=colors.get(m), label=m, ms=4)
        ax2.loglog(B, tf, "o-", color=colors.get(m), label=m, ms=4)
        cx = a2["linear_scaling_ceiling"].get(m)
        if cx:
            ax1.axvline(cx, color=colors.get(m), ls=":", lw=0.8, alpha=0.6)
    ax1.set_xlabel("batch B")
    ax1.set_ylabel("slices / s")
    ax1.set_title("GEMM throughput (crossover = dotted)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.2)
    ax2.set_xlabel("batch B")
    ax2.set_ylabel("effective TFLOP/s")
    ax2.set_title(f"GEMM compute rate — {'GPU' if a2['on_gpu'] else 'CPU'}")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.2)
    fig.suptitle(f"Green's matvec ψ = G @ I  —  G {a2['G_shape']}  ({a2['devices']})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-s2-gemm-crossover.png", dpi=130)
    plt.close(fig)

    # fig 2: batched-solve axis vs host-reference axis scatter (y=x) + precision
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    rep = a3["reproduction"]
    pairs = rep.get("pairs", {})
    dev_r = np.asarray(pairs.get("dev_r", []))
    dev_z = np.asarray(pairs.get("dev_z", []))
    ref_r = np.asarray(pairs.get("ref_r", []))
    ref_z = np.asarray(pairs.get("ref_z", []))
    if ref_r.size:
        lo = float(min(ref_r.min(), dev_r.min(), ref_z.min(), dev_z.min()))
        hi = float(max(ref_r.max(), dev_r.max(), ref_z.max(), dev_z.max()))
        ax1.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax1.scatter(ref_r, dev_r, s=22, color="#268", label="axis R [m]")
        ax1.scatter(ref_z, dev_z, s=22, color="#c73", label="axis Z [m]")
    ax1.set_xlabel("host reference axis [m]")
    ax1.set_ylabel("device batched-solve axis [m]")
    ax1.set_title(
        f"A3 reproduction — {'PASS' if rep['pass'] else 'FAIL'} "
        f"(n={rep['n_scored']}, median {rep['axis_median_cm']:.2f} cm ≤ "
        f"{a3['axis_tol_cm']} cm)"
    )
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal", adjustable="datalim")
    modes = [
        m
        for m in ("tf32", "bf16")
        if "axis_delta_vs_fp64_median_cm" in a3["precision"].get(m, {})
    ]
    med = [a3["precision"][m]["axis_delta_vs_fp64_median_cm"] for m in modes]
    p90 = [a3["precision"][m]["axis_delta_vs_fp64_p90_cm"] for m in modes]
    x = np.arange(len(modes))
    ax2.bar(x - 0.2, med, 0.4, label="median", color="#c73")
    ax2.bar(x + 0.2, p90, 0.4, label="p90", color="#e9b")
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes)
    ax2.set_ylabel("Δaxis vs fp64 on-device [cm]")
    ax2.set_title("A3 precision-tier axis cost")
    ax2.legend(fontsize=8)
    fig.suptitle(
        f"Batched on-device fp64 solve: {a3.get('fp64_solves_per_s', 0)} solves/s "
        f"@ B={a3['batch']}  ({'GPU' if a3['on_gpu'] else 'CPU'})"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-s2-batched-solve-parity.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def stage_gpu(a1_batch: int, a2_batches: list[int], a3_batch: int, no_figures: bool):
    if not STAGE_NPZ.exists():
        raise SystemExit(f"missing staged inputs {STAGE_NPZ}; run --stage prep first")
    arrs = dict(np.load(STAGE_NPZ, allow_pickle=True))
    import jax

    # fp64 everywhere: the binding flux is a small difference of grid fluxes and
    # the whole solve state is fp64.  Enable BEFORE any array is created so every
    # leg (not just the topology read, which enables it on import) is genuinely
    # fp64 — otherwise "fp64" is silently truncated to fp32 and the loop carry
    # dtypes disagree.
    jax.config.update("jax_enable_x64", True)

    print(
        f"jax {jax.__version__} | backend={jax.default_backend()} | "
        f"devices={jax.devices()}"
    )
    print(f"staged: {int(arrs['ip'].shape[0])} unique slices, G {arrs['G'].shape}")

    print("\n[A1] framework — inner GS step (jax vs torch)")
    a1 = framework_step_compare(arrs, a1_batch)
    print(
        f"  jax: {a1.get('jax_step_ms')} ms/step "
        f"({a1.get('jax_slices_per_s')} sl/s)  "
        f"torch: {a1.get('torch_step_ms')} ms/step "
        f"({a1.get('torch_slices_per_s')} sl/s)"
    )

    print("\n[A2] GEMM crossover")
    a2 = gemm_crossover_bench(arrs, a2_batches)

    print("\n[A3] batched solve + on-device read + precision")
    a3 = batched_solve_reproduction(arrs, a3_batch)
    rep = a3["reproduction"]
    print(
        f"  reproduction axis median {rep['axis_median_cm']:.3f} cm "
        f"({'PASS' if rep['pass'] else 'FAIL'}), parity {a3['parity']}"
    )

    if not no_figures:
        _figures(a2, a3, arrs)

    devs = jax.devices()
    stamp = {
        "kind": "gpu-labeler-foundation-spike",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "on_gpu": any(d.platform == "gpu" for d in devs),
        "devices": [f"{d.platform}:{d.id}" for d in devs],
        "campaign": str(arrs["campaign"]),
        "n_unique_slices": int(arrs["ip"].shape[0]),
        "G_shape": [int(arrs["G"].shape[0]), int(arrs["G"].shape[1])],
        "leg_a1_framework": a1,
        "leg_a2_gemm_crossover": a2,
        "leg_a3_batched_solve": a3,
        "git_commit": _run(["git", "rev-parse", "HEAD"]) or "unknown",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(stamp, indent=2))
    print(f"\nwrote {ARTIFACT}")

    ok = stamp["on_gpu"]
    print(
        f"\nDEVICE={'GPU' if ok else 'CPU'}  "
        f"A3 reproduction={'PASS' if rep['pass'] else 'FAIL'} "
        f"(axis median {rep['axis_median_cm']:.3f} cm ≤ {AXIS_TOL_CM} cm)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prep", "gpu", "all"], default="all")
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices", type=int, default=6)
    ap.add_argument("--max-shots", type=int, default=6)
    ap.add_argument("--min-ip-ka", type=float, default=200.0)
    ap.add_argument("--a1-batch", type=int, default=512)
    ap.add_argument(
        "--a2-batches", type=str, default="1,8,32,128,256,512,600,1024,2048,4096"
    )
    ap.add_argument("--a3-batch", type=int, default=512)
    ap.add_argument(
        "--picard-iters",
        type=int,
        default=PICARD_ITERS,
        help="fixed Picard iterations in the batched on-device solve",
    )
    ap.add_argument(
        "--read-n-levels",
        type=int,
        default=READ_N_LEVELS,
        help="in-loop connectivity-read binding-sweep resolution",
    )
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    global PICARD_ITERS, READ_N_LEVELS
    PICARD_ITERS = int(args.picard_iters)
    READ_N_LEVELS = int(args.read_n_levels)
    if args.stage in ("prep", "all"):
        stage_prep(args.nr, args.nz, args.max_slices, args.min_ip_ka, args.max_shots)
    if args.stage in ("gpu", "all"):
        batches = [int(b) for b in args.a2_batches.split(",") if b.strip()]
        return stage_gpu(args.a1_batch, batches, args.a3_batch, args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
