"""Profile the per-slice boundary-read chain: contour-push LCFS vs ray-cast.

Times each operation of the connectivity boundary read on a representative
65x97 scoring grid (single-thread CPU) and contrasts the NEW chain (one
flux-offset contour push + emergent X-points) against the OLD chain
(``find_critical_points`` + innermost-X bounding flux + outward ray-cast).  The
grid, the masked near-pole plateau, and the diverted geometry mimic the masked
TOTAL psi the harmonic gate reads.

Run: ``uv run python scripts/profile_boundary_read.py``.
"""

from __future__ import annotations

import time

import contourpy
import numpy as np

from imas_ambix.latent import topology as topo
from imas_ambix.latent.boundary_harmonic import mask_invalid_interior


def _diverted_masked_field(nr: int = 65, nz: int = 97):
    """A representative masked TOTAL psi: plasma well + divertor bump, near-pole
    interior masked to a deep plateau (the field the harmonic gate reads)."""
    r_1d = np.linspace(0.2, 2.0, nr)
    z_1d = np.linspace(-1.5, 1.5, nz)
    rr, zz = np.meshgrid(r_1d, z_1d)
    r0, z0 = 0.9, 0.0
    # plasma well (confined min) + a positive divertor-coil bump below → an
    # X-point between them (a lower-single-null-like separatrix)
    psi = (rr - r0) ** 2 + (zz - z0) ** 2
    psi = psi + 1.6 * np.exp(-(((rr - r0) / 0.28) ** 2 + ((zz + 0.85) / 0.22) ** 2))
    pole_r = r0 * (1.0 - 0.41)  # the gate's inboard pole
    field = mask_invalid_interior(
        psi, r_1d, z_1d, pole_r, 0.0, 0.5 * (r0 - pole_r), axis_rz=(r0, z0)
    )
    # a limiter that hugs the confined region (legs exit it → diverted)
    t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    lim_r = r0 + 0.55 * np.cos(t)
    lim_z = z0 + 0.55 * np.sin(t)
    return field, r_1d, z_1d, (r0, z0), lim_r, lim_z


def _bench(fn, n: int) -> float:
    """Median ms per call over ``n`` reps (a short warm-up first)."""
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def main() -> int:
    field, r_1d, z_1d, axis, lim_r, lim_z = _diverted_masked_field()
    psi_axis = topo._bilerp(field, r_1d, z_1d, *axis)
    edge = np.concatenate([field[0, :], field[-1, :], field[:, 0], field[:, -1]])
    psi_out = float(edge[int(np.argmax(np.abs(edge - psi_axis)))])
    mid = 0.5 * (psi_axis + psi_out)

    def one_level():
        gen = contourpy.contour_generator(
            r_1d, z_1d, field, line_type="SeparateCode", quad_as_tri=True
        )
        gen.lines(mid)

    def full_push():
        topo.lcfs_contour(field, r_1d, z_1d, axis, limiter_r=lim_r, limiter_z=lim_z)

    def crit():
        topo.find_critical_points(field, r_1d, z_1d)

    def raycast():
        topo.lcfs_radii(field, r_1d, z_1d, axis, mid)

    lcfs = topo.lcfs_contour(field, r_1d, z_1d, axis, limiter_r=lim_r, limiter_z=lim_z)
    cp = topo.find_critical_points(field, r_1d, z_1d)

    def emergent():
        topo.emergent_xpoints(cp.x_points, lcfs.ring, tol=0.05)

    n = 200
    rows = [
        ("contourpy — one level (gen + lines)", _bench(one_level, n)),
        ("full LCFS contour push (NEW primary)", _bench(full_push, n)),
        ("find_critical_points", _bench(crit, n)),
        ("lcfs_radii outward ray-cast (OLD)", _bench(raycast, n)),
        ("emergent_xpoints (proximity read)", _bench(emergent, n)),
    ]
    print(
        f"grid {field.shape[1]}x{field.shape[0]}  diverted={lcfs.found}  "
        f"psi_bnd={lcfs.psi_bnd:.4g}  x_saddles={cp.x_points.shape[0]}"
    )
    print(f"{'operation':40s}  {'ms/slice':>10s}")
    print("-" * 54)
    for name, ms in rows:
        print(f"{name:40s}  {ms:10.3f}")

    push = dict(rows)["full LCFS contour push (NEW primary)"]
    emg = dict(rows)["emergent_xpoints (proximity read)"]
    old = (
        dict(rows)["find_critical_points"]
        + dict(rows)["lcfs_radii outward ray-cast (OLD)"]
    )
    new_lcfs_only = push
    new_with_emergent = push + dict(rows)["find_critical_points"] + emg
    print("-" * 54)
    print(f"{'OLD chain (crit + ray-cast)':40s}  {old:10.3f}")
    print(f"{'NEW LCFS-only (push)':40s}  {new_lcfs_only:10.3f}")
    print(f"{'NEW + emergent X (push + crit + prox)':40s}  {new_with_emergent:10.3f}")
    print(f"speedup (LCFS-only vs OLD): {old / new_lcfs_only:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
