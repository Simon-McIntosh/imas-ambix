"""Single-shot on-device causal rollout — the inner object every deployment wraps.

The foundation spike demonstrated the substrate (batched GEMM fast lanes,
on-device connectivity read at GPU/CPU parity, tiered precision safe) and
localised the one missing piece: a naive cold-Gaussian fixed-iteration Picard
corner-locks toward the coil-side attractor.  This rollout carries the cure the
host spine uses — the measured-moment DISC SEED (basin insurance) plus a
warm-started jφ chain along time — and assembles the whole single-shot causal
rollout on device:

  Leg B1 (sequential march — the correctness anchor) — one ``lax.scan`` over
    the shot's time-ordered slices: slice 0 solved from its disc seed with a
    first-solve sweep budget, every later slice warm-started from the previous
    converged jφ with a short budget.  Scored for axis reproduction against a
    host reference march (grid-free Picard, same disc seed + warm chain), with
    a GPU/CPU backend-parity check on the device march itself.  The inner
    fixed point runs safeguarded Anderson-Picard on the smooth connectivity
    read (the accelerator the differentiable-map study selected); a plain
    Picard arm is timed alongside as the control.

  Leg B2 (windowed parallel-in-time) — Jacobi waveform relaxation: all slices
    of a causal window solved together (vmap over the window), seeds shifted
    one slice per outer iteration so warm information propagates along time,
    exact jφ carry between windows.  Gate: the PinT trajectory converges to
    the march's (axis) within a bounded outer count, at a measured wall-clock
    speedup vs the sequential march.

  Leg B3 (fp64 temporal carry) — the cross-slice recurrent state the full
    engine threads through the scan: a batched Thomas tridiagonal solve of the
    resistive ψ-diffusion step (vertex-centred finite volume, θ=1, frozen
    per-interval geometry) and the exact modal-ZOH passive recurrence — each
    validated at fp64 against the host references (``diffuse_psi``,
    ``zoh_mode_response``) on the shot's real time grid, then threaded
    together through one ``lax.scan`` over the slice intervals.  The fixed-θ
    profile used by B1/B2 takes no feedback from this state; coupling the
    diffusion prior into the on-device profile update is the corpus-engine
    integration step, not this rung.

Two stages, as for the spike:

    # login node — stage one held-out shot + host references into an NPZ
    uv run python -m scripts.device_rollout_single_shot --stage prep

    # H200 (reservation mandatory) — run the three legs against the staged NPZ
    srun --partition=betelgeuse --reservation=gpu_0003_grpA --gres=gpu:1 \
         --cpus-per-task=4 --mem=64G --time=00:45:00 \
      bash -lc 'export TMPDIR=/tmp; cd <repo>; \
        uv run python -m scripts.device_rollout_single_shot --stage gpu'

Firewall unchanged: GS force balance, physics + measured channels only, no
EFIT anywhere in the loop.
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

ARTIFACT = Path(
    "imas_ambix/latent/artifacts/patch_gate/device_rollout_single_shot-v0.json"
)
STAGE_NPZ = Path(
    "imas_ambix/latent/artifacts/patch_gate/device_rollout_single_shot-inputs.npz"
)
FIG_DIR = Path("docs/figures/gpu-accelerated-labeler")

# fixed profile parameters θ for the SEED shape (the spike's two-term form);
# the in-loop profile is the pinned K=2 LSQ, not this fixed shape
BETA0 = 0.5
ALPHA = 1.0
# disc-centroid soft-tether width [m] — the spine scaffold's default
CENTROID_SIGMA_M = 0.02
# confined-axis bound: beyond this the read is the outboard attractor [m]
CONFINED_AXIS_R_MAX = 1.4
# reproduction bar for the device march vs the host reference march
AXIS_TOL_CM = 2.0
# PinT-vs-march agreement bar (same algorithm both sides — much tighter)
PINT_TOL_CM = 0.5
# in-loop connectivity-read resolution (as the spike: coarser per-iterate
# binding sweep, unchanged physical fixed point)
READ_N_LEVELS = 48
READ_N_BISECT = 12
READ_N_RAY = 96
RELAX = 0.5
# Anderson depth + warmup (sweeps of plain relaxed Picard before mixing —
# the shape transient the accelerator study also skips)
ANDERSON_M = 3
ANDERSON_WARMUP = 6
# Anderson step-size safeguard: reject a mixed step this many times longer
# than the relaxed Picard step (falls back to Picard for that sweep)
ANDERSON_CAP = 2.0
# temporal-carry references
DIFFUSION_DT_S = 2.0e-3
ZOH_N_MODES = 16
ZOH_TAU_RANGE = (1.0e-3, 0.3)


def _run(cmd) -> str:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def _axis_cm(a, b) -> float:
    return float(100.0 * np.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------------------------------------------------------
# stage 1 — payload + host-reference preparation (login node)
# ---------------------------------------------------------------------------


def _host_reference_march(
    grid, payloads, disc_seeds, centroids, tbl, basis, topology_read: str
) -> dict:
    """Sequential host K=2 pinned-scaffold march — the production anchor.

    The unpinned fixed-θ map corner-locks toward the outboard attractor on
    this shot even from a disc seed (measured: the host fixed-θ chain sits at
    R ≈ 1.77 m from the ramp on) — basin control lives in the centroid-PINNED
    map, exactly the accelerator study's conclusion.  The anchor is therefore
    the spine's own K=2 position scaffold (free-sign n_p=n_f=1 fit, all
    magnetics masked, disc-centroid soft tether), warm-chained along time.
    ``topology_read`` selects the host arm: ``"hard"`` is the production label
    read; ``"connectivity"`` shares the device read definition, so the
    device-vs-host gap under it isolates solver formulation from the
    already-quantified hard-vs-smooth read difference.
    """
    from scripts.differentiable_solve_gate_eval import _fit_slice
    from scripts.spine_label_factory import frozen_spine_config

    spine, _sha = frozen_spine_config()
    iso = spine["interior_solve"]
    ref_axis, ref_resid, ref_scored, ref_confined = [], [], [], []
    ref_jphi, ref_psi = [], []
    warm = None
    for k, p in enumerate(payloads):
        seed = disc_seeds[k] if warm is None else warm
        f = _fit_slice(
            grid,
            tbl,
            basis,
            p,
            n_p=1,
            n_f=1,
            nonneg=False,
            smoothness=float(iso["smoothness"]),
            boundary_read=iso["boundary_read_scoring"],
            centroid=(float(centroids[k][0]), float(centroids[k][1])),
            warm=seed,
            sigma=CENTROID_SIGMA_M,
            topology_read=topology_read,
        )
        ok = bool(f.scored) and f.target is not None and np.isfinite(f.target[0])
        axis = (
            [float(f.target[0]), float(f.target[1])]
            if ok
            else [float("nan"), float("nan")]
        )
        confined = ok and float(f.target[0]) <= CONFINED_AXIS_R_MAX
        if confined and f.jphi_flat is not None and np.isfinite(f.jphi_flat).all():
            warm = np.asarray(f.jphi_flat, dtype=np.float64)
        ref_axis.append(axis)
        ref_resid.append(float(f.residual) if f.residual is not None else np.nan)
        ref_scored.append(ok)
        ref_confined.append(bool(confined))
        ref_jphi.append(
            np.asarray(f.jphi_flat, dtype=np.float64)
            if f.jphi_flat is not None
            else np.zeros(grid.flat_r.size)
        )
        ref_psi.append(
            np.asarray(f.psi, dtype=np.float64).ravel()
            if f.psi is not None
            else np.zeros(grid.flat_r.size)
        )
    return {
        "axis": np.asarray(ref_axis),
        "resid": np.asarray(ref_resid),
        "conv": np.asarray(ref_scored),
        "confined": np.asarray(ref_confined),
        "jphi": np.asarray(ref_jphi),
        "psi": np.asarray(ref_psi),
    }


def _diffusion_reference(grid, ref, ip_t, time_s) -> dict | None:
    """Host ψ-diffusion reference on frozen flat-top geometry over the shot.

    Geometry from the highest-current CONVERGED reference slice; the fixed-θ
    two-term profile is exactly the n_p=1/n_f=1 legendre family at α=1, so the
    equivalent normalised coefficients come from an in-family LSQ of the
    reference jφ onto the basis images.
    """
    from imas_ambix.latent.current_diffusion import (
        EtaProfile,
        diffuse_psi,
        flux_surface_geometry,
        reconstruct_profile_scales,
    )
    from imas_ambix.latent.gs_solve import profile_basis

    conv = np.asarray(ref["conv"]) & np.asarray(ref["confined"])
    if not conv.any():
        return None
    order = np.argsort(-np.abs(ip_t) * conv)
    for j in order:
        if not conv[j]:
            continue
        psi2d = ref["psi"][j].reshape(grid.nz, grid.nr)
        ip = float(ip_t[j])
        rec = reconstruct_profile_scales(psi2d, grid, ip, n_p=1, n_f=1, nonneg=False)
        images = profile_basis(
            rec["psi_n"], grid.flat_r, r0=grid.r0, n_p=1, n_f=1, kind="legendre"
        )
        core = rec["core"].ravel()
        u = images[core] * rec["s_k"][np.newaxis, :]
        y = ref["jphi"][j][core]
        if u.shape[0] < 16 or not np.isfinite(u).all():
            continue
        coeffs, *_ = np.linalg.lstsq(u, y, rcond=None)
        geo = flux_surface_geometry(
            psi2d, grid, coeffs=coeffs, ip_amperes=ip, n_p=1, n_f=1, nonneg=False
        )
        if geo is None:
            continue
        t0, t1 = float(time_s[0]), float(time_s[-1])
        n_sub = min(2000, max(8, int(np.ceil((t1 - t0) / DIFFUSION_DT_S)) + 1))
        t_sub = np.linspace(t0, t1, n_sub)
        ip_sub = np.interp(t_sub, time_s, np.abs(ip_t))
        eta = EtaProfile()
        step = diffuse_psi(geo, eta, t_grid=t_sub, ip_of_t=ip_sub)
        toc = _toc_face(geo, eta)
        return {
            "geo_slice": int(j),
            "t_sub": t_sub,
            "ip_sub": ip_sub,
            "psi_face_ref": step["psi_face"],
            "psi_face0": np.asarray(geo.psi_face, dtype=np.float64),
            "rho_face": np.asarray(geo.rho_face, dtype=np.float64),
            "d_face": np.asarray(geo.d_face, dtype=np.float64),
            "toc_face": toc,
            "k_edge": float(geo.ip_edge_gradient(1.0)),  # ∂ψ/∂ρ̂ at the edge per ampere
        }
    return None


def _toc_face(geo, eta) -> np.ndarray:
    """The host diffusion's face-centred capacitance coefficient, replicated."""
    from imas_ambix.latent.current_diffusion import _16PI2, MU0

    sigma_cell = 1.0 / eta(geo.psi_n_cell)
    toc_cell = sigma_cell * MU0 * _16PI2 * geo.phi_b**2 * geo.rho_cell / geo.f_cell**2
    toc = np.zeros(geo.rho_face.size)
    toc[1:-1] = 0.5 * (toc_cell[:-1] + toc_cell[1:])
    toc[0] = toc_cell[0]
    toc[-1] = toc_cell[-1]
    return toc


def _zoh_reference(t_sub: np.ndarray, ip_sub: np.ndarray) -> dict:
    """Host exact-ZOH modal reference on the shot's uniform sub-step grid.

    The recurrence is linear per mode, so kernel parity is drive-independent;
    the drive used is a per-mode weighting of the measured Ip trace (a
    representative linked-flux magnitude), never a fabricated measurement.
    """
    from imas_ambix.latent.passive_resistance import zoh_mode_response

    rng = np.random.default_rng(3)
    tau = np.geomspace(ZOH_TAU_RANGE[0], ZOH_TAU_RANGE[1], ZOH_N_MODES)
    w = rng.uniform(0.5, 1.5, ZOH_N_MODES) * np.where(
        np.arange(ZOH_N_MODES) % 2 == 0, 1.0, -1.0
    )
    # linked flux per mode ~ mWb-scale for a MA-scale Ip
    psi_m = np.outer(ip_sub, w) * 1.0e-9
    dt = float(t_sub[1] - t_sub[0])
    a_ref = zoh_mode_response(tau, dt, psi_m)
    return {"tau": tau, "psi_m": psi_m, "a_ref": a_ref, "dt": dt}


def stage_prep(
    nr: int,
    nz: int,
    max_slices: int,
    min_ip_ka: float,
    shot_arg: int,
    host_reads: str = "hard",
):
    """Stage ONE held-out shot's time-ordered slices + host references."""
    import os

    # the host connectivity-read arm runs the jax kernels on CPU; keep prep off
    # any GPU so a concurrent device run is not starved of memory
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from imas_ambix.eval import prediction_bar as pbar
    from imas_ambix.latent.boundary_disc import disc_read
    from imas_ambix.latent.connectivity_boundary import _densify_wall
    from scripts.heldout_mse_gate_eval import _campaign_table
    from scripts.position_controlled_solve_gate import _disc_seed_flat
    from scripts.spine_label_factory import factory_shot_payloads

    if shot_arg > 0:
        shots = [shot_arg]
    else:
        manifest = pbar.load_locked_manifest()
        shots = list(pbar.held_out_shot_ids(manifest))
    print(f"candidate shots: {shots}")

    for shot in shots:
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
        grid = payload["grid"]
        tbl, basis = payload["table"], payload["basis"]
        order = np.argsort([p.time_s for p in payload["payloads"]])

        kept, seeds, axis_seeds, n_no_disc = [], [], [], 0
        for k in order:
            p = payload["payloads"][int(k)]
            inv = disc_read(p, grid, tbl, basis)
            if inv is None or inv.ring is None:
                n_no_disc += 1
                continue
            kept.append(p)
            seeds.append(_disc_seed_flat(grid, inv))
            axis_seeds.append([float(inv.centroid_r), float(inv.centroid_z)])
        if len(kept) < 12:
            print(f"  shot {shot}: only {len(kept)} disc-readable slices, skip")
            continue

        print(
            f"  shot {shot}: {len(kept)} disc-readable slices "
            f"({n_no_disc} without a disc read — dropped, recorded)"
        )
        G = np.asarray(grid.plasma_grid_psi_columns(), dtype=np.float64)
        psi_coil = np.asarray(
            [grid.coil_psi(np.asarray(p.i_pf, dtype=np.float64)) for p in kept]
        )
        ip_t = np.asarray([float(p.ip_amperes) for p in kept])
        time_s = np.asarray([float(p.time_s) for p in kept])
        disc_seeds = np.asarray(seeds)

        print("  host reference march (pinned K=2 scaffold, warm chain) ...")
        refs: dict[str, dict] = {}
        host_walls: dict[str, float] = {}
        reads = ("hard", "connectivity") if host_reads == "both" else ("hard",)
        for read in reads:
            t0 = time.perf_counter()
            refs[read] = _host_reference_march(
                grid, kept, disc_seeds, np.asarray(axis_seeds), tbl, basis, read
            )
            host_walls[read] = time.perf_counter() - t0
            print(
                f"    [{read}] {len(kept)} slices in {host_walls[read]:.1f}s, "
                f"{int(refs[read]['conv'].sum())}/{len(kept)} scored, "
                f"{int(refs[read]['confined'].sum())} confined"
            )
        ref = refs["hard"]
        host_wall = host_walls["hard"]

        diff = _diffusion_reference(grid, ref, ip_t, time_s)
        if diff is None:
            print("  WARNING: no diffusion geometry — carry leg will be skipped")
        zoh = _zoh_reference(diff["t_sub"], diff["ip_sub"]) if diff else None

        wall_r, wall_z = _densify_wall(grid)
        STAGE_NPZ.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if "connectivity" in refs:
            extra.update(
                ref_c_axis=refs["connectivity"]["axis"],
                ref_c_converged=refs["connectivity"]["conv"],
                ref_c_confined=refs["connectivity"]["confined"],
                host_march_connectivity_wall_s=np.float64(host_walls["connectivity"]),
            )
        if diff is not None:
            extra = {f"diff_{k}": v for k, v in diff.items()}
            extra.update({f"zoh_{k}": v for k, v in zoh.items()})
        np.savez_compressed(
            STAGE_NPZ,
            rg=np.asarray(grid.rg, dtype=np.float64),
            zg=np.asarray(grid.zg, dtype=np.float64),
            inside=np.asarray(grid.inside_limiter, dtype=bool),
            flat_r=np.asarray(grid.flat_r, dtype=np.float64),
            flat_z=np.asarray(grid.flat_z, dtype=np.float64),
            cells=np.asarray(grid.cells, dtype=np.int64),
            wall_r=np.asarray(wall_r, dtype=np.float64),
            wall_z=np.asarray(wall_z, dtype=np.float64),
            G=G,
            r0=np.float64(grid.r0),
            dr=np.float64(grid.dr),
            dz=np.float64(grid.dz),
            nr=np.int64(nr),
            nz=np.int64(nz),
            shot=np.int64(shot),
            n_no_disc=np.int64(n_no_disc),
            psi_coil=psi_coil,
            ip=ip_t,
            time_s=time_s,
            disc_seed=disc_seeds,
            axis_seed=np.asarray(axis_seeds),
            ref_axis=ref["axis"],
            ref_resid=ref["resid"],
            ref_converged=ref["conv"],
            ref_confined=ref["confined"],
            host_march_wall_s=np.float64(host_wall),
            **extra,
        )
        print(f"staged shot {shot}: {len(kept)} slices → {STAGE_NPZ}")
        return
    raise SystemExit("no usable held-out shot staged")


# ---------------------------------------------------------------------------
# device kernels (jax)
# ---------------------------------------------------------------------------


def _matvec(G, x, mode):
    """psi_plasma = G @ x under a precision ``mode`` (result cast to f64)."""
    import jax
    import jax.numpy as jnp

    if mode == "fp64":
        return G @ x
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


def _build_slice_step(arrs, mode: str, read: str = "hard", tau: float = 1e-3):
    """Return (jphi_from_psi, psi_from_jphi): the two halves of the Picard map.

    ``jphi_from_psi`` runs the on-device topology read and updates jφ by the
    pinned K=2 scaffold: a closed-form 2-coefficient LSQ (free-sign p/f pair
    on the (1−ψ_N) edge family) whose rows are the Ip normalisation and the
    disc-centroid soft tether — the basin insurance the accelerator study
    showed the unpinned map lacks.  This is the "tiny per-sweep profile LSQ"
    the matrix-freeze decision allows; the interaction matrices are never
    touched.  ``read`` selects the topology read: ``'hard'`` (default,
    byte-identical to the shipped rollout — exact-min binding + boolean flood
    core) or ``'smooth'`` (the temperature-smoothed kernel at temperature
    ``tau``: softmin binding + retracted-gate sigmoid core weight, stencil
    O-point axis — the end-to-end differentiable map the accelerator probe
    measured).  ``psi_from_jphi`` renormalises to Ip and evaluates the
    Green's matvec plus the per-slice vacuum coil flux.  Both are pure
    fixed-shape functions of fp64 arrays, safe under jit/vmap/scan.
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

    if read not in ("hard", "smooth"):
        raise ValueError(f"read must be 'hard' or 'smooth', got {read!r}")

    rg = jnp.asarray(arrs["rg"])
    zg = jnp.asarray(arrs["zg"])
    inside = jnp.asarray(arrs["inside"])
    flat_r = jnp.asarray(arrs["flat_r"])
    cells = jnp.asarray(arrs["cells"])
    wall_r = jnp.asarray(arrs["wall_r"])
    wall_z = jnp.asarray(arrs["wall_z"])
    r0 = float(arrs["r0"])
    cell_area = float(arrs["dr"]) * float(arrs["dz"])
    nz = int(arrs["nz"])
    nr = int(arrs["nr"])
    n_flood = nr + nz
    base = BETA0 * flat_r / r0 + (1.0 - BETA0) * r0 / jnp.maximum(flat_r, 1e-3)
    cell_r = flat_r[cells]
    cell_z = jnp.asarray(arrs["flat_z"])[cells]
    img_r_ratio = flat_r / r0
    img_r_inv = r0 / jnp.maximum(flat_r, 1e-3)

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
            weight = flood_fill_core(confined, seed2d, n_flood).reshape(-1)
            return psi_n, weight, axis

    else:  # 'smooth' — stencil axis first, smooth read seeded at it

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

    def jphi_from_psi(psi, axis, pin, ip):
        psi_n, weight, axis = support(psi, axis)

        # pinned K=2 scaffold: jφ = c_p·(R/R0)·e + c_f·(R0/R)·e on the core,
        # c from the closed-form LSQ of the Ip row + the centroid tether rows
        e = jnp.clip(1.0 - psi_n, 0.0, None) ** ALPHA * weight
        img_p = img_r_ratio * e
        img_f = img_r_inv * e
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
        c = jnp.linalg.solve(n_mat, a_mat.T @ b_vec)
        jphi = c[0] * img_p + c[1] * img_f
        # fallback: an unusable fit (empty core / singular rows) drops to the
        # fixed two-term shape rather than emitting a fabricated profile
        bad = ~jnp.all(jnp.isfinite(jphi)) | (jnp.sum(jnp.abs(jphi)) < 1e-12)
        jphi = jnp.where(bad, base * e, jphi)
        return jphi, axis

    def psi_from_jphi(G, jphi, psi_coil, ip):
        i_cell = jphi[cells] * cell_area
        total = jnp.sum(i_cell)
        scale = jnp.where(jnp.abs(total) > 1e-12, ip / total, 0.0)
        return _matvec(G, i_cell * scale, mode) + psi_coil, i_cell * scale

    return jphi_from_psi, psi_from_jphi


def _build_slice_solver(
    arrs,
    mode: str,
    n_sweeps: int,
    accelerator: str,
    read: str = "hard",
    tau: float = 1e-3,
):
    """One slice's fixed-sweep solve of the ψ fixed-point map.

    ``accelerator='picard'`` is the relaxed Picard control;
    ``'anderson'`` mixes the last :data:`ANDERSON_M` residual differences
    (type-II Anderson on ψ) with two safeguards — a warmup window and a
    step-size cap — falling back to the relaxed Picard step whenever the
    mixed step is unavailable or oversized.  Returns the solve as a pure
    function (NOT jitted) so callers embed it in scan / vmap graphs.
    """
    import jax
    import jax.numpy as jnp

    jphi_from_psi, psi_from_jphi = _build_slice_step(arrs, mode, read, tau)
    n_flat = int(arrs["flat_r"].size)
    m = ANDERSON_M

    def solve(G, psi_coil, ip, jphi_seed, axis_seed, pin):
        psi0, _ = psi_from_jphi(G, jphi_seed, psi_coil, ip)

        def body(i, carry):
            psi, axis, dx, df, f_prev, x_prev, norm_prev, trace = carry
            jphi, axis = jphi_from_psi(psi, axis, pin, ip)
            g, _ = psi_from_jphi(G, jphi, psi_coil, ip)
            f = g - psi
            resid = jnp.max(jnp.abs(f)) / jnp.maximum(jnp.max(jnp.abs(g)), 1e-12)
            trace = trace.at[i].set(resid)
            psi_pic = psi + RELAX * f

            if accelerator == "anderson":
                # residual-growth safeguard: when the map residual grows, the
                # mixed history has crossed a core-mask flip and is stale —
                # zero the buffers and take relaxed Picard until it rebuilds
                norm_f = jnp.max(jnp.abs(f))
                grew = norm_f > norm_prev
                dx = jnp.where(grew, jnp.zeros_like(dx), dx)
                df = jnp.where(grew, jnp.zeros_like(df), df)
                col = jnp.mod(i, m)
                dx_new = jax.lax.dynamic_update_index_in_dim(
                    dx, psi - x_prev, col, axis=1
                )
                df_new = jax.lax.dynamic_update_index_in_dim(
                    df, f - f_prev, col, axis=1
                )
                have_hist = (i >= 1) & ~grew
                dx = jnp.where(have_hist, dx_new, dx)
                df = jnp.where(have_hist, df_new, df)
                # γ from the regularised m×m normal equations
                a = df.T @ df
                a = a + 1e-10 * (jnp.trace(a) + 1e-30) * jnp.eye(m)
                gam = jnp.linalg.solve(a, df.T @ f)
                psi_and = psi + RELAX * f - (dx + RELAX * df) @ gam
                step_pic = jnp.max(jnp.abs(psi_pic - psi))
                step_and = jnp.max(jnp.abs(psi_and - psi))
                use_and = (
                    (i >= ANDERSON_WARMUP)
                    & ~grew
                    & jnp.all(jnp.isfinite(psi_and))
                    & (step_and <= ANDERSON_CAP * jnp.maximum(step_pic, 1e-300))
                )
                psi_next = jnp.where(use_and, psi_and, psi_pic)
                norm_prev = norm_f
            else:
                psi_next = psi_pic
            return psi_next, axis, dx, df, f, psi, norm_prev, trace

        init = (
            psi0,
            axis_seed,
            jnp.zeros((n_flat, m)),
            jnp.zeros((n_flat, m)),
            jnp.zeros(n_flat),
            psi0,
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            jnp.zeros(n_sweeps),
        )
        psi, axis, *_rest, trace = jax.lax.fori_loop(0, n_sweeps, body, init)
        # emit the converged current + a final residual on the unrelaxed map
        jphi, axis = jphi_from_psi(psi, axis, pin, ip)
        g, _ = psi_from_jphi(G, jphi, psi_coil, ip)
        resid = jnp.max(jnp.abs(g - psi)) / jnp.maximum(jnp.max(jnp.abs(g)), 1e-12)
        return {
            "psi": psi,
            "jphi": jphi,
            "axis": axis,
            "residual": resid,
            "trace": trace,
        }

    return solve


def _build_march(
    arrs,
    mode: str,
    n_first: int,
    n_warm: int,
    accelerator: str,
    read: str = "hard",
    tau: float = 1e-3,
):
    """The sequential warm-started device march: one scan over the slices."""
    import jax
    import jax.numpy as jnp

    solve_first = _build_slice_solver(arrs, mode, n_first, accelerator, read, tau)
    solve_warm = _build_slice_solver(arrs, mode, n_warm, accelerator, read, tau)

    def march(G, psi_coil, ip, disc_seed, axis_seed, pins):
        first = solve_first(G, psi_coil[0], ip[0], disc_seed[0], axis_seed[0], pins[0])

        def step(carry, xs):
            jphi_prev, axis_prev = carry
            pc, ipk, pin = xs
            out = solve_warm(G, pc, ipk, jphi_prev, axis_prev, pin)
            ok = jnp.all(jnp.isfinite(out["jphi"])) & (
                jnp.abs(jnp.sum(out["jphi"])) > 1e-12
            )
            jphi_carry = jnp.where(ok, out["jphi"], jphi_prev)
            axis_carry = jnp.where(
                jnp.all(jnp.isfinite(out["axis"])), out["axis"], axis_prev
            )
            return (jphi_carry, axis_carry), (
                out["axis"],
                out["residual"],
                out["trace"],
                out["jphi"],
            )

        (_, _), (axes, resids, traces, jphis) = jax.lax.scan(
            step,
            (first["jphi"], first["axis"]),
            (psi_coil[1:], ip[1:], pins[1:]),
        )
        axes = jnp.concatenate([first["axis"][None], axes], axis=0)
        resids = jnp.concatenate([first["residual"][None], resids], axis=0)
        jphis = jnp.concatenate([first["jphi"][None], jphis], axis=0)
        return {
            "axis": axes,
            "residual": resids,
            "warm_traces": traces,
            "first_trace": first["trace"],
            "jphi": jphis,
        }

    return jax.jit(march)


def _build_window_solver(
    arrs,
    mode: str,
    n_pint: int,
    accelerator: str,
    read: str = "hard",
    tau: float = 1e-3,
):
    """A batched (vmap over the window) fixed-sweep solve — one PinT outer."""
    import jax

    solve = _build_slice_solver(arrs, mode, n_pint, accelerator, read, tau)
    return jax.jit(jax.vmap(solve, in_axes=(None, 0, 0, 0, 0, 0)))


def _thomas(a, b, c, r):
    """Tridiagonal solve (Thomas), sequential scan over the ~25 face rows."""
    import jax

    def fwd(carry, xs):
        cp_prev, rp_prev = carry
        ai, bi, ci, ri = xs
        den = bi - ai * cp_prev
        cp = ci / den
        rp = (ri - ai * rp_prev) / den
        return (cp, rp), (cp, rp)

    (_, _), (cp, rp) = jax.lax.scan(fwd, (0.0, 0.0), (a, b, c, r))

    def bwd(x_next, xs):
        cpi, rpi = xs
        x = rpi - cpi * x_next
        return x, x

    _, x_rev = jax.lax.scan(bwd, 0.0, (cp[::-1], rp[::-1]))
    return x_rev[::-1]


def _build_diffusion_scan(arrs):
    """Device ψ-diffusion (θ=1) over the staged uniform sub-step grid."""
    import jax
    import jax.numpy as jnp

    toc = jnp.asarray(arrs["diff_toc_face"])
    d_face = jnp.asarray(arrs["diff_d_face"])
    rho = np.asarray(arrs["diff_rho_face"])
    drho = float(rho[1] - rho[0])
    k_edge = float(arrs["diff_k_edge"])
    d_mid = 0.5 * (d_face[:-1] + d_face[1:])
    n = int(rho.size)

    def step(psi, dt, ip_k):
        lam = dt / (toc * drho * drho)
        a = (
            jnp.zeros(n)
            .at[1:-1]
            .set(-lam[1:-1] * d_mid[:-1])
            .at[-1]
            .set(-2.0 * lam[-1] * d_mid[-1])
        )
        c = (
            jnp.zeros(n)
            .at[1:-1]
            .set(-lam[1:-1] * d_mid[1:])
            .at[0]
            .set(-2.0 * lam[0] * d_mid[0])
        )
        b = (
            jnp.ones(n)
            .at[1:-1]
            .set(1.0 + lam[1:-1] * (d_mid[:-1] + d_mid[1:]))
            .at[0]
            .set(1.0 + 2.0 * lam[0] * d_mid[0])
            .at[-1]
            .set(1.0 + 2.0 * lam[-1] * d_mid[-1])
        )
        grad_edge = k_edge * ip_k
        r = psi.at[-1].add(2.0 * lam[-1] * d_face[-1] * grad_edge * drho)
        return _thomas(a, b, c, r)

    def rollout(psi0, t_sub, ip_sub):
        dt = t_sub[1] - t_sub[0]

        def body(psi, ip_k):
            psi = step(psi, dt, ip_k)
            return psi, psi

        _, out = jax.lax.scan(body, psi0, ip_sub[1:])
        return jnp.concatenate([psi0[None], out], axis=0)

    return jax.jit(rollout), step


def _build_zoh_scan(arrs):
    """Device exact-ZOH modal recurrence over the staged sub-step grid."""
    import jax
    import jax.numpy as jnp

    tau = jnp.asarray(arrs["zoh_tau"])
    dt = float(arrs["zoh_dt"])
    decay = jnp.exp(-dt / tau)
    coeff = tau / dt * (1.0 - decay)

    def rollout(psi_m):
        u = jnp.zeros_like(psi_m).at[1:].set(-(psi_m[1:] - psi_m[:-1]))

        def body(a, u_k):
            a = decay * a + coeff * u_k
            return a, a

        _, out = jax.lax.scan(body, jnp.zeros(psi_m.shape[1]), u)
        return out

    return jax.jit(rollout)


# ---------------------------------------------------------------------------
# stage 2 — the three device legs
# ---------------------------------------------------------------------------


def eval_sequential_march(
    arrs, n_first: int, n_warm: int, read: str = "hard", tau: float = 1e-3
) -> dict:
    """Sequential device march: reproduction, accelerator A/B, backend parity."""
    import jax
    import jax.numpy as jnp

    T = int(arrs["ip"].shape[0])
    G = jnp.asarray(arrs["G"])
    psi_coil = jnp.asarray(arrs["psi_coil"])
    ip = jnp.asarray(arrs["ip"])
    disc = jnp.asarray(arrs["disc_seed"])
    axis0 = jnp.asarray(arrs["axis_seed"])
    ref_axis = np.asarray(arrs["ref_axis"])
    ref_conv = np.asarray(arrs["ref_converged"]) & np.asarray(arrs["ref_confined"])

    out: dict = {"n_slices": T, "n_first": n_first, "n_warm": n_warm}
    walls: dict[str, float] = {}
    results: dict[str, dict] = {}
    for acc in ("anderson", "picard"):
        march = _build_march(arrs, "fp64", n_first, n_warm, acc, read, tau)
        t0 = time.perf_counter()
        res = march(G, psi_coil, ip, disc, axis0, axis0)
        jax.block_until_ready(res["axis"])
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = march(G, psi_coil, ip, disc, axis0, axis0)
        jax.block_until_ready(res["axis"])
        walls[acc] = time.perf_counter() - t0
        results[acc] = {k: np.asarray(v) for k, v in res.items()}
        out[f"{acc}_wall_s"] = round(walls[acc], 4)
        out[f"{acc}_compile_s"] = round(compile_s, 2)
        out[f"{acc}_slices_per_s"] = round(T / walls[acc], 2)
        out[f"{acc}_residual_median"] = float(np.median(results[acc]["residual"]))

    # production arm = the measured better solver (final map residual); the
    # loser is retained as the A/B control, never silently discarded
    prod = min(("anderson", "picard"), key=lambda a: out[f"{a}_residual_median"])
    out["production_arm"] = prod
    res = results[prod]
    axis_dev = res["axis"]
    n_fabricated = int(np.sum(~np.isfinite(axis_dev).all(axis=1)))

    def _score(anchor_axis, anchor_mask):
        dcm = np.array(
            [
                _axis_cm(axis_dev[j], anchor_axis[j])
                for j in range(T)
                if anchor_mask[j] and np.isfinite(axis_dev[j]).all()
            ]
        )
        return {
            "n_scored": int(dcm.size),
            "n_ref_scored_confined": int(anchor_mask.sum()),
            "n_fabricated_readouts": n_fabricated,
            "axis_median_cm": float(np.median(dcm)) if dcm.size else float("nan"),
            "axis_p90_cm": float(np.percentile(dcm, 90)) if dcm.size else float("nan"),
            "per_slice_cm": dcm.tolist(),
            "pass": bool(dcm.size and float(np.median(dcm)) <= AXIS_TOL_CM),
        }

    # the GATE anchor is the same-read host march (device and host share the
    # connectivity read definition, so this isolates the device rollout from
    # the hard-vs-smooth read difference the differentiable-solve gate already
    # quantified); the hard-read production-label anchor is reported alongside
    out["reproduction_hard_read_anchor"] = _score(ref_axis, ref_conv)
    if "ref_c_axis" in arrs:
        ref_c_conv = np.asarray(arrs["ref_c_converged"]) & np.asarray(
            arrs["ref_c_confined"]
        )
        out["reproduction"] = _score(np.asarray(arrs["ref_c_axis"]), ref_c_conv)
    else:
        out["reproduction"] = out["reproduction_hard_read_anchor"]

    # sweeps-to-tolerance from the warm traces (accelerator A/B)
    tol = 3e-4
    sweeps = {}
    for acc in ("anderson", "picard"):
        tr = results[acc]["warm_traces"]  # (T-1, n_warm)
        first_hit = np.array(
            [
                int(np.argmax(row <= tol)) + 1 if np.any(row <= tol) else n_warm
                for row in tr
            ]
        )
        sweeps[acc] = first_hit
    out["_median_trace_anderson"] = np.median(
        results["anderson"]["warm_traces"], axis=0
    )
    out["_median_trace_picard"] = np.median(results["picard"]["warm_traces"], axis=0)
    out["accelerator"] = {
        "tolerance": tol,
        "anderson_sweeps_median": float(np.median(sweeps["anderson"])),
        "picard_sweeps_median": float(np.median(sweeps["picard"])),
        "sweep_speedup_median": float(
            np.median(sweeps["picard"]) / max(np.median(sweeps["anderson"]), 1.0)
        ),
        "anderson_converged_frac": float(np.mean(sweeps["anderson"] < n_warm)),
        "picard_converged_frac": float(np.mean(sweeps["picard"] < n_warm)),
    }

    # GPU/CPU backend parity on a short prefix of the march
    on_gpu = any(d.platform == "gpu" for d in jax.devices())
    small = min(8, T)
    with jax.default_device(jax.devices("cpu")[0]):
        march_cpu = _build_march(arrs, "fp64", n_first, n_warm, prod, read, tau)
        rc = march_cpu(
            jnp.asarray(np.asarray(arrs["G"])),
            jnp.asarray(np.asarray(arrs["psi_coil"][:small])),
            jnp.asarray(np.asarray(arrs["ip"][:small])),
            jnp.asarray(np.asarray(arrs["disc_seed"][:small])),
            jnp.asarray(np.asarray(arrs["axis_seed"][:small])),
            jnp.asarray(np.asarray(arrs["axis_seed"][:small])),
        )
        jax.block_until_ready(rc["axis"])
    axis_cpu = np.asarray(rc["axis"])
    fin = np.isfinite(axis_cpu).all(axis=1) & np.isfinite(axis_dev[:small]).all(axis=1)
    parity = (
        float(np.max(np.abs(axis_cpu[fin] - axis_dev[:small][fin])))
        if fin.any()
        else float("nan")
    )
    out["parity"] = {
        "small_prefix": small,
        "max_abs_axis_diff_m": parity,
        "pass": bool(np.isfinite(parity) and parity < 1e-6) if on_gpu else None,
    }

    # tiered-precision march (tf32 GEMM, fp64 state) vs the fp64 march
    march_tf = _build_march(arrs, "tf32", n_first, n_warm, prod, read, tau)
    rt = march_tf(G, psi_coil, ip, disc, axis0, axis0)
    jax.block_until_ready(rt["axis"])
    axis_tf = np.asarray(rt["axis"])
    deltas = np.array(
        [
            _axis_cm(axis_tf[j], axis_dev[j])
            for j in range(T)
            if np.isfinite(axis_tf[j]).all() and np.isfinite(axis_dev[j]).all()
        ]
    )
    out["precision_tf32"] = {
        "axis_delta_vs_fp64_median_cm": float(np.median(deltas))
        if deltas.size
        else float("nan"),
        "axis_delta_vs_fp64_p90_cm": float(np.percentile(deltas, 90))
        if deltas.size
        else float("nan"),
    }
    out["march_axis"] = axis_dev.tolist()
    out["_march_wall_s"] = walls[prod]
    return out


def eval_parallel_in_time(
    arrs,
    march: dict,
    window: int,
    outers: int,
    n_pint: int,
    n_pre: int,
    pre_first: int,
    read: str = "hard",
    tau: float = 1e-3,
) -> dict:
    """Coarse pre-march + batched continuation outers vs the sequential march.

    The pinned map is still multi-stable: a cold decoupled pass converges each
    slice to a fixed point that need not be the warm chain's (measured ~44 cm
    off on this shot).  The plan's basin insurance is therefore a CHEAP COARSE
    PRE-MARCH — the same sequential scan at a few sweeps per slice — whose
    chain-consistent states seed the batched (vmap over slices) refinement
    outers.  Wall clock counts the pre-march plus every outer.
    """
    import jax
    import jax.numpy as jnp

    T = int(arrs["ip"].shape[0])
    G = jnp.asarray(arrs["G"])
    psi_coil = np.asarray(arrs["psi_coil"])
    ip = np.asarray(arrs["ip"])
    axis0 = np.asarray(arrs["axis_seed"])
    march_axis = np.asarray(march["march_axis"])
    W = T if window <= 0 else int(window)
    prod = march["production_arm"]

    solver = _build_window_solver(arrs, "fp64", n_pint, prod, read, tau)
    pre_march = _build_march(arrs, "fp64", pre_first, n_pre, prod, read, tau)

    def run(record: bool):
        t0 = time.perf_counter()
        pre = pre_march(
            G,
            jnp.asarray(arrs["psi_coil"]),
            jnp.asarray(arrs["ip"]),
            jnp.asarray(arrs["disc_seed"]),
            jnp.asarray(axis0),
            jnp.asarray(axis0),
        )
        jax.block_until_ready(pre["axis"])
        pre_wall = time.perf_counter() - t0
        jphi_iter = np.array(pre["jphi"])
        axis_iter = np.array(pre["axis"])
        agree: list[float] = []
        agree_wp: list[float] = []
        wp_mask = np.asarray(arrs["ref_converged"]) & np.asarray(arrs["ref_confined"])
        pint_resid = np.zeros(T)
        t0 = time.perf_counter()
        for _ in range(outers):
            for s0 in range(0, T, W):
                s1 = min(s0 + W, T)
                res = solver(
                    G,
                    jnp.asarray(psi_coil[s0:s1]),
                    jnp.asarray(ip[s0:s1]),
                    jnp.asarray(jphi_iter[s0:s1]),
                    jnp.asarray(axis_iter[s0:s1]),
                    jnp.asarray(axis0[s0:s1]),
                )
                jax.block_until_ready(res["axis"])
                jphi_iter[s0:s1] = np.asarray(res["jphi"])
                axis_iter[s0:s1] = np.asarray(res["axis"])
                pint_resid[s0:s1] = np.asarray(res["residual"])
            if record:
                d, d_wp = [], []
                for i in range(T):
                    if not (
                        np.isfinite(axis_iter[i]).all()
                        and np.isfinite(march_axis[i]).all()
                    ):
                        continue
                    v = _axis_cm(axis_iter[i], march_axis[i])
                    d.append(v)
                    if wp_mask[i]:
                        d_wp.append(v)
                agree.append(float(np.median(d)) if d else float("nan"))
                agree_wp.append(float(np.median(d_wp)) if d_wp else float("nan"))
        outer_wall = time.perf_counter() - t0
        return axis_iter, pint_resid, (agree, agree_wp), pre_wall, outer_wall

    # instrumented pass (also warms compilation), then the timed pass
    pint_axis, pint_resid, (agree_per_outer, agree_per_outer_wp), _, _ = run(True)
    axis_t, resid_t, _, pre_wall, outer_wall = run(False)
    pint_axis, pint_resid = axis_t, resid_t
    wall = pre_wall + outer_wall

    d = [
        _axis_cm(pint_axis[i], march_axis[i])
        for i in range(T)
        if np.isfinite(pint_axis[i]).all() and np.isfinite(march_axis[i]).all()
    ]
    med = float(np.median(d)) if d else float("nan")
    # the well-posed subset: slices the host anchor itself scores/confines —
    # outside it the trajectory is under-determined (both engines drift in Z)
    scored_mask = np.asarray(arrs["ref_converged"]) & np.asarray(arrs["ref_confined"])
    d_sc = [
        _axis_cm(pint_axis[i], march_axis[i])
        for i in range(T)
        if scored_mask[i]
        and np.isfinite(pint_axis[i]).all()
        and np.isfinite(march_axis[i]).all()
    ]
    med_sc = float(np.median(d_sc)) if d_sc else float("nan")
    n_fab = int(np.sum(~np.isfinite(pint_axis).all(axis=1)))
    speedup = march["_march_wall_s"] / wall if wall > 0 else float("nan")
    outers_needed = next(
        (
            k + 1
            for k, v in enumerate(agree_per_outer_wp)
            if np.isfinite(v) and v <= PINT_TOL_CM
        ),
        None,
    )
    return {
        "window": W,
        "outers": outers,
        "n_pint_sweeps": n_pint,
        "n_pre_sweeps": n_pre,
        "pre_first_sweeps": pre_first,
        "wall_s": round(wall, 4),
        "pre_march_wall_s": round(pre_wall, 4),
        "outer_wall_s": round(outer_wall, 4),
        "march_wall_s": round(float(march["_march_wall_s"]), 4),
        "speedup_vs_march": round(float(speedup), 2),
        "axis_vs_march_median_cm": med,
        "axis_vs_march_p90_cm": float(np.percentile(d, 90)) if d else float("nan"),
        "axis_vs_march_median_wellposed_cm": med_sc,
        "n_wellposed": len(d_sc),
        "agreement_per_outer_cm": agree_per_outer,
        "agreement_per_outer_wellposed_cm": agree_per_outer_wp,
        "outers_to_tolerance": outers_needed,
        "n_fabricated_readouts": n_fab,
        "residual_median": float(np.median(pint_resid)),
        "pint_axis": pint_axis.tolist(),
        "pass_convergence": bool(np.isfinite(med_sc) and med_sc <= PINT_TOL_CM),
        "pass_speedup": bool(np.isfinite(speedup) and speedup >= 3.0),
    }


def eval_temporal_carry(arrs) -> dict:
    """fp64 temporal-carry kernels vs the host references + a threaded scan."""
    import jax
    import jax.numpy as jnp

    if "diff_t_sub" not in arrs:
        return {"skipped": "no diffusion geometry staged"}

    out: dict = {}
    # diffusion parity (device Thomas scan vs host dense-solve diffuse_psi)
    rollout, _step = _build_diffusion_scan(arrs)
    psi0 = jnp.asarray(arrs["diff_psi_face0"])
    t_sub = jnp.asarray(arrs["diff_t_sub"])
    ip_sub = jnp.asarray(arrs["diff_ip_sub"])
    psi_dev = rollout(psi0, t_sub, ip_sub)
    jax.block_until_ready(psi_dev)
    t0 = time.perf_counter()
    psi_dev = rollout(psi0, t_sub, ip_sub)
    jax.block_until_ready(psi_dev)
    wall = time.perf_counter() - t0
    psi_dev = np.asarray(psi_dev)
    psi_ref = np.asarray(arrs["diff_psi_face_ref"])
    scale = max(float(np.max(np.abs(psi_ref))), 1e-12)
    diff_max = float(np.max(np.abs(psi_dev - psi_ref)))
    out["diffusion"] = {
        "n_substeps": int(psi_ref.shape[0]),
        "n_faces": int(psi_ref.shape[1]),
        "wall_s": round(wall, 4),
        "max_abs_diff_wb": diff_max,
        "max_rel_diff": diff_max / scale,
        "is_f64": bool(psi_dev.dtype == np.float64),
        "pass": bool(diff_max / scale < 1e-9),
        "geo_slice": int(arrs["diff_geo_slice"]),
        "per_step_max_diff": np.max(np.abs(psi_dev - psi_ref), axis=1).tolist(),
    }

    # exact-ZOH modal recurrence parity
    zoh = _build_zoh_scan(arrs)
    a_dev = zoh(jnp.asarray(arrs["zoh_psi_m"]))
    jax.block_until_ready(a_dev)
    a_dev = np.asarray(a_dev)
    a_ref = np.asarray(arrs["zoh_a_ref"])
    zscale = max(float(np.max(np.abs(a_ref))), 1e-30)
    zdiff = float(np.max(np.abs(a_dev - a_ref)))
    out["zoh"] = {
        "n_modes": int(a_ref.shape[1]),
        "max_abs_diff": zdiff,
        "max_rel_diff": zdiff / zscale,
        "is_f64": bool(a_dev.dtype == np.float64),
        "pass": bool(zdiff / zscale < 1e-12),
    }

    # threaded carry: diffusion + ZOH state through ONE scan over the substeps
    _, dstep = _build_diffusion_scan(arrs)
    tau = jnp.asarray(arrs["zoh_tau"])
    dt = float(arrs["zoh_dt"])
    decay = jnp.exp(-dt / tau)
    coeff = tau / dt * (1.0 - decay)
    psi_m = jnp.asarray(arrs["zoh_psi_m"])
    u = jnp.zeros_like(psi_m).at[1:].set(-(psi_m[1:] - psi_m[:-1]))

    def body(carry, xs):
        psi_face, a = carry
        ip_k, u_k = xs
        psi_face = dstep(psi_face, dt, ip_k)
        a = decay * a + coeff * u_k
        return (psi_face, a), None

    (psi_end, a_end), _ = jax.lax.scan(
        body,
        (psi0, jnp.zeros(int(arrs["zoh_tau"].size))),
        (ip_sub[1:], u[1:]),
    )
    jax.block_until_ready(psi_end)
    psi_end = np.asarray(psi_end)
    a_end = np.asarray(a_end)
    out["threaded_scan"] = {
        "psi_face_end_matches_diffusion": float(np.max(np.abs(psi_end - psi_ref[-1]))),
        "a_end_matches_zoh": float(np.max(np.abs(a_end - a_ref[-1]))),
        "all_finite_f64": bool(
            np.isfinite(psi_end).all()
            and np.isfinite(a_end).all()
            and psi_end.dtype == np.float64
            and a_end.dtype == np.float64
        ),
    }
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _figures(arrs, b1: dict, b2: dict, b3: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    time_s = np.asarray(arrs["time_s"])
    ref_axis = np.asarray(arrs["ref_axis"])
    ref_conv = np.asarray(arrs["ref_converged"]) & np.asarray(arrs["ref_confined"])
    march_axis = np.asarray(b1["march_axis"])
    pint_axis = np.asarray(b2["pint_axis"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    a0 = axes[0, 0]
    a0.plot(
        time_s[ref_conv],
        ref_axis[ref_conv, 0],
        "k.-",
        lw=1,
        ms=4,
        label="host reference march",
    )
    a0.plot(time_s, march_axis[:, 0], color="#268", lw=1.2, label="device march")
    a0.plot(time_s, pint_axis[:, 0], color="#c73", lw=1, ls="--", label="PinT")
    a0.set_xlabel("time [s]")
    a0.set_ylabel("axis R [m]")
    a0.set_title(f"shot {int(arrs['shot'])} — axis trajectory")
    a0.legend(fontsize=8)

    a1 = axes[0, 1]
    rep = b1["reproduction"]
    dcm = np.asarray(rep["per_slice_cm"])
    a1.semilogy(np.arange(dcm.size), dcm, "o-", color="#268", ms=4)
    a1.axhline(
        AXIS_TOL_CM, color="k", ls="--", lw=0.8, label=f"tolerance {AXIS_TOL_CM} cm"
    )
    a1.axhline(
        rep["axis_median_cm"],
        color="#268",
        ls=":",
        lw=1,
        label=f"median {rep['axis_median_cm']:.3f} cm",
    )
    a1.set_xlabel("scored slice")
    a1.set_ylabel("device − host axis [cm]")
    a1.set_title(
        f"B1 reproduction — {'PASS' if rep['pass'] else 'FAIL'} (n={rep['n_scored']})"
    )
    a1.legend(fontsize=8)

    a2 = axes[1, 0]
    ag = b2["agreement_per_outer_cm"]
    ag_wp = b2.get("agreement_per_outer_wellposed_cm", [])
    a2.semilogy(np.arange(1, len(ag) + 1), ag, "o-", color="#c73", label="all slices")
    if ag_wp:
        a2.semilogy(
            np.arange(1, len(ag_wp) + 1),
            ag_wp,
            "s-",
            color="#268",
            label="well-posed slices",
        )
    a2.axhline(
        PINT_TOL_CM, color="k", ls="--", lw=0.8, label=f"tolerance {PINT_TOL_CM} cm"
    )
    a2.set_xlabel("PinT outer iteration")
    a2.set_ylabel("median axis vs march [cm]")
    a2.set_title(
        f"B2 PinT convergence (well-posed) — "
        f"{'PASS' if b2['pass_convergence'] else 'FAIL'} "
        f"({b2['axis_vs_march_median_wellposed_cm']:.2f} cm, "
        f"outers: {b2['outers_to_tolerance']})"
    )
    a2.set_xticks(np.arange(1, len(ag) + 1))
    a2.legend(fontsize=8)

    a3 = axes[1, 1]
    bars = ["march", "PinT"]
    vals = [b2["march_wall_s"], b2["wall_s"]]
    a3.bar(bars, vals, color=["#268", "#c73"])
    a3.set_ylabel("wall clock [s]")
    a3.set_title(
        f"B2 single-shot wall — speedup {b2['speedup_vs_march']:.2f}× "
        f"({'PASS' if b2['pass_speedup'] else 'MISS'} vs 3×)"
    )
    for i, v in enumerate(vals):
        a3.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Single-shot on-device rollout — sequential march vs windowed PinT")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-s3-march-pint.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    a0 = axes[0]
    if "diffusion" in b3:
        d = b3["diffusion"]
        a0.semilogy(d["per_step_max_diff"], color="#268", lw=1)
        a0.set_title(
            f"ψ-diffusion device vs host — max rel {d['max_rel_diff']:.1e} "
            f"({'PASS' if d['pass'] else 'FAIL'})"
        )
        a0.set_xlabel("sub-step")
        a0.set_ylabel("max |Δψ_face| [Wb]")
    else:
        a0.text(0.5, 0.5, "diffusion leg skipped", ha="center")

    a1 = axes[1]
    tr_a = np.asarray(b1["_median_trace_anderson"])
    tr_p = np.asarray(b1["_median_trace_picard"])
    a1.semilogy(tr_p, color="#a33", lw=1.2, label="Picard")
    a1.semilogy(tr_a, color="#2a7", lw=1.2, label="Anderson")
    a1.axhline(3e-4, color="k", ls="--", lw=0.8, label="tolerance")
    a1.set_xlabel("warm-slice sweep")
    a1.set_ylabel("median map residual")
    a1.set_title(
        f"inner accelerator (median warm slice) — "
        f"{b1['accelerator']['sweep_speedup_median']:.2f}× fewer sweeps"
    )
    a1.legend(fontsize=8)

    a2 = axes[2]
    if "zoh" in b3:
        z = b3["zoh"]
        a2.bar(
            ["ZOH modes", "diffusion"],
            [z["max_rel_diff"], b3["diffusion"]["max_rel_diff"]],
            color=["#a3a", "#268"],
        )
        a2.set_yscale("log")
        a2.set_ylabel("max relative diff vs host fp64")
        a2.set_title("fp64 carry-kernel parity")
    fig.suptitle("fp64 temporal scan carry + inner accelerator")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig-s3-carry-parity.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def stage_gpu(args) -> int:
    if not STAGE_NPZ.exists():
        raise SystemExit(f"missing staged inputs {STAGE_NPZ}; run --stage prep first")
    arrs = dict(np.load(STAGE_NPZ, allow_pickle=True))
    import jax

    jax.config.update("jax_enable_x64", True)
    devs = jax.devices()
    on_gpu = any(d.platform == "gpu" for d in devs)
    print(f"jax {jax.__version__} | backend={jax.default_backend()} | devices={devs}")
    T = int(arrs["ip"].shape[0])
    print(f"staged shot {int(arrs['shot'])}: {T} slices, G {arrs['G'].shape}")

    print(
        f"\n[B1] sequential warm-started march (disc seed + warm chain, "
        f"{args.device_read} read)"
    )
    b1 = eval_sequential_march(
        arrs, args.n_first, args.n_warm, args.device_read, args.device_tau
    )
    rep = b1["reproduction"]
    print(
        f"  reproduction: median {rep['axis_median_cm']:.3f} cm "
        f"({'PASS' if rep['pass'] else 'FAIL'}); parity {b1['parity']}; "
        f"accel {b1['accelerator']['sweep_speedup_median']:.2f}x"
    )

    print("\n[B2] windowed PinT (Jacobi waveform relaxation)")
    b2 = eval_parallel_in_time(
        arrs,
        b1,
        args.window,
        args.outers,
        args.n_pint,
        args.n_pre,
        args.pre_first,
        args.device_read,
        args.device_tau,
    )
    print(
        f"  vs march: median {b2['axis_vs_march_median_cm']:.3f} cm in "
        f"{b2['outers_to_tolerance']} outers; speedup {b2['speedup_vs_march']:.2f}x "
        f"({b2['wall_s']:.2f}s vs {b2['march_wall_s']:.2f}s)"
    )

    print("\n[B3] fp64 temporal scan carry (diffusion Thomas + exact ZOH)")
    b3 = eval_temporal_carry(arrs)
    if "diffusion" in b3:
        print(
            f"  diffusion max rel {b3['diffusion']['max_rel_diff']:.2e}, "
            f"zoh max rel {b3['zoh']['max_rel_diff']:.2e}, "
            f"threaded {b3['threaded_scan']['all_finite_f64']}"
        )

    if not args.no_figures:
        _figures(arrs, b1, b2, b3)

    gate = {
        "march_reproduction_pass": rep["pass"],
        "march_fabricated_readouts": rep["n_fabricated_readouts"],
        "pint_convergence_pass": b2["pass_convergence"],
        "pint_fabricated_readouts": b2["n_fabricated_readouts"],
        "pint_speedup": b2["speedup_vs_march"],
        "pint_speedup_pass_3x": b2["pass_speedup"],
        "carry_diffusion_pass": b3.get("diffusion", {}).get("pass"),
        "carry_zoh_pass": b3.get("zoh", {}).get("pass"),
    }
    g_b = bool(
        gate["march_reproduction_pass"]
        and gate["pint_convergence_pass"]
        and gate["pint_speedup_pass_3x"]
        and gate["march_fabricated_readouts"] == 0
        and gate["pint_fabricated_readouts"] == 0
    )
    b1.pop("_march_jphi", None)
    b1["_median_trace_anderson"] = list(map(float, b1["_median_trace_anderson"]))
    b1["_median_trace_picard"] = list(map(float, b1["_median_trace_picard"]))
    stamp = {
        "kind": "device-rollout-single-shot",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "on_gpu": on_gpu,
        "devices": [f"{d.platform}:{d.id}" for d in devs],
        "shot": int(arrs["shot"]),
        "n_slices": T,
        "n_slices_without_disc_read": int(arrs["n_no_disc"]),
        "host_march_wall_s": float(arrs["host_march_wall_s"]),
        "config": {
            "n_first": args.n_first,
            "n_warm": args.n_warm,
            "n_pint": args.n_pint,
            "window": args.window,
            "outers": args.outers,
            "anderson_m": ANDERSON_M,
            "read": [READ_N_LEVELS, READ_N_BISECT, READ_N_RAY],
            "device_read": args.device_read,
            "device_tau": args.device_tau,
        },
        "sequential_march": b1,
        "parallel_in_time": b2,
        "temporal_carry": b3,
        "gate_g_b": {**gate, "pass": g_b},
        "git_commit": _run(["git", "rev-parse", "HEAD"]) or "unknown",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(stamp, indent=2))
    print(f"\nwrote {ARTIFACT}")
    print(
        f"\nDEVICE={'GPU' if on_gpu else 'CPU'}  G-B={'PASS' if g_b else 'FAIL'}  "
        f"march {rep['axis_median_cm']:.3f} cm | PinT "
        f"{b2['axis_vs_march_median_cm']:.3f} cm @ {b2['speedup_vs_march']:.2f}x"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prep", "gpu", "all"], default="all")
    ap.add_argument(
        "--shot", type=int, default=0, help="explicit shot (else first held-out)"
    )
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--max-slices", type=int, default=48)
    ap.add_argument("--min-ip-ka", type=float, default=200.0)
    ap.add_argument(
        "--n-first",
        type=int,
        default=60,
        help="sweep budget for the disc-seeded first slice",
    )
    ap.add_argument(
        "--n-warm",
        type=int,
        default=20,
        help="sweep budget per warm-started march slice",
    )
    ap.add_argument(
        "--n-pint", type=int, default=20, help="sweep budget per PinT outer iteration"
    )
    ap.add_argument(
        "--n-pre",
        type=int,
        default=6,
        help="sweep budget per slice of the coarse PinT pre-march",
    )
    ap.add_argument(
        "--pre-first",
        type=int,
        default=24,
        help="first-slice sweep budget of the coarse PinT pre-march",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=0,
        help="PinT window size in slices (0 = whole shot)",
    )
    ap.add_argument(
        "--outers",
        type=int,
        default=3,
        help="PinT outer (waveform-relaxation) iterations",
    )
    ap.add_argument(
        "--host-reads",
        choices=["hard", "both"],
        default="hard",
        help="host anchor arms to stage: the production hard read, or both "
        "hard + connectivity (adds the same-read anchor at ~2.6x prep cost)",
    )
    ap.add_argument(
        "--device-read",
        choices=["hard", "smooth"],
        default="hard",
        help="in-loop device topology read: the shipped hard kernel "
        "(byte-identical default) or the temperature-smoothed kernel",
    )
    ap.add_argument(
        "--device-tau",
        type=float,
        default=1e-3,
        help="smoothing temperature for --device-read smooth "
        "(the read's gate-calibrated accuracy point is 1e-3)",
    )
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.stage in ("prep", "all"):
        stage_prep(
            args.nr,
            args.nz,
            args.max_slices,
            args.min_ip_ka,
            args.shot,
            host_reads=args.host_reads,
        )
    if args.stage in ("gpu", "all"):
        return stage_gpu(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
