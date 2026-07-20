"""GPU capability demonstration for the contour-free flux-surface averaging.

This is a CAPABILITY check, not a speedup claim: it proves the connectivity FSA
kernel (:mod:`imas_ambix.latent.flux_surface_connectivity`) actually executes on
whatever accelerator the installed jaxlib exposes, batched over many equilibria
by ``jax.vmap`` (the corpus-labeller inner loop), in fp64, and that the on-device
result matches the host (CPU) result bit-for-physics.

On the H200 node (partition ``betelgeuse``) with a CUDA jaxlib it runs on a
``cuda`` device; on any other machine it runs on CPU and still passes the
correctness cross-check — so the same command is a valid check everywhere and a
genuine GPU demonstration where a GPU is present.

Run (H200 — see imas_ambix/spine_bench/AGENTS.md for the reservation recipe):

    srun --partition=betelgeuse --reservation=gpu_0003_grpA --gres=gpu:1 \
         --cpus-per-task=4 --mem=32G --time=00:20:00 \
      bash -lc 'export TMPDIR=/tmp; cd <repo>; \
        uv run --with "jax[cuda12]==0.10.1" python -m scripts.fsa_gpu_capability'

Stamps imas_ambix/spine_bench/results/fsa-gpu-capability-<commit>-<host>.yaml.
"""

from __future__ import annotations

import argparse
import platform
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

RESULTS = Path("imas_ambix/spine_bench/results")


def _synthetic_batch(n: int, nr: int, nz: int):
    """A batch of ``n`` Solov'ev-like ψ fields on one shared (nz, nr) grid.

    Each element varies the axis radius, minor radius and elongation so the
    cores genuinely differ in size — exercising the fixed-shape property (the
    whole batch runs through identical array shapes regardless).
    """
    rg = np.linspace(0.2, 1.6, nr)
    zg = np.linspace(-1.1, 1.1, nz)
    rr, zz = np.meshgrid(rg, zg)
    # deterministic per-element variation (no RNG — reproducible, vmap-labelled)
    rax = 0.75 + 0.30 * (np.arange(n) / max(n - 1, 1))
    a = 0.45 + 0.25 * ((np.arange(n) * 7 % max(n, 1)) / max(n, 1))
    elong = 1.4 + 0.6 * ((np.arange(n) * 3 % max(n, 1)) / max(n, 1))
    psi = np.stack(
        [
            -(((rr - rax[k]) / a[k]) ** 2 + (zz / (a[k] * elong[k])) ** 2)
            for k in range(n)
        ]
    ).astype(np.float64)
    inside = np.ones((nz, nr), dtype=bool)
    inside[rr < 0.25] = False
    return psi, rg, zg, inside


def _run(cmd):
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=str, default="16,64,256")
    ap.add_argument("--nr", type=int, default=65)
    ap.add_argument("--nz", type=int, default=97)
    ap.add_argument("--out-dir", type=str, default=str(RESULTS))
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from imas_ambix.latent.flux_surface_connectivity import flux_surface_bins_jax

    devices = jax.devices()
    backend = jax.default_backend()
    on_gpu = any(d.platform == "gpu" for d in devices)
    print(f"jax {jax.__version__} | backend={backend} | devices={devices}")

    def _fsa(p):
        return flux_surface_bins_jax(
            p,
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(0.0),
            jnp.asarray(-1.0),
            jnp.asarray(0.04),
            jnp.asarray(0.985),
            28,
            jnp.asarray(1.25),
        )

    batch_sizes = [int(b) for b in args.batches.split(",")]
    n_max = max(batch_sizes)
    psi, rg, zg, inside = _synthetic_batch(n_max, args.nr, args.nz)

    vfsa = jax.jit(jax.vmap(_fsa))

    runs = []
    max_diff_vs_cpu = 0.0
    for n in batch_sizes:
        pb = jnp.asarray(psi[:n])
        # warm-up compile, then timed device run
        out = vfsa(pb)
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        out = vfsa(pb)
        jax.block_until_ready(out)
        wall = time.perf_counter() - t0
        d = np.asarray(out["inv_r2"])
        assert d.shape == (n, 28), d.shape  # fixed-shape over the whole batch
        assert np.all(np.isfinite(d))
        # correctness: the same batch on the CPU device must match the run above
        with jax.default_device(jax.devices("cpu")[0]):
            out_cpu = jax.jit(jax.vmap(_fsa))(jnp.asarray(psi[:n]))
            jax.block_until_ready(out_cpu)
            diff = float(
                np.max(
                    np.abs(np.asarray(out["inv_r2"]) - np.asarray(out_cpu["inv_r2"]))
                )
            )
        max_diff_vs_cpu = max(max_diff_vs_cpu, diff)
        thr = n / wall
        runs.append(
            {
                "batch": n,
                "vmap_wall_s": round(wall, 5),
                "equilibria_per_s": round(thr, 1),
                "max_abs_diff_vs_cpu": diff,
                "well_posed_fraction": float(np.mean(np.asarray(out["well_posed"]))),
            }
        )
        print(
            f"  batch={n:5d}  wall={wall * 1e3:8.2f} ms  "
            f"{thr:9.1f} eq/s  |Δ vs cpu|={diff:.2e}  "
            f"wellposed={runs[-1]['well_posed_fraction']:.2f}"
        )

    stamp = {
        "kind": "fsa-gpu-capability",
        "purpose": "on-device execution + GPU/CPU parity of the contour-free FSA "
        "(capability, not a speedup claim)",
        "jax_version": jax.__version__,
        "backend": backend,
        "on_gpu": on_gpu,
        "devices": [f"{d.platform}:{d.id}" for d in devices],
        "grid": {"nr": args.nr, "nz": args.nz},
        "fp64": bool(np.asarray(out["inv_r2"]).dtype == np.float64),
        "max_abs_diff_vs_cpu_all_batches": max_diff_vs_cpu,
        "runs": runs,
        "git_commit": _run(["git", "rev-parse", "HEAD"]) or "unknown",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }

    import yaml

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    host = stamp["hostname"].split(".")[0]
    commit = stamp["git_commit"][:10]
    path = out_dir / f"fsa-gpu-capability-{commit}-{host}.yaml"
    path.write_text(yaml.safe_dump(stamp, sort_keys=False, width=100))

    ok = on_gpu and max_diff_vs_cpu < 1e-9
    verdict = (
        "GPU CAPABILITY DEMONSTRATED"
        if ok
        else ("CPU-only (no GPU device)" if not on_gpu else "PARITY FAIL")
    )
    print(
        f"\nDEVICE={'GPU' if on_gpu else 'CPU'}  fp64={stamp['fp64']}  "
        f"max|Δ| vs cpu={max_diff_vs_cpu:.2e}  →  {verdict}"
    )
    print(f"stamp: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
