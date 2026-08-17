"""Prototype: 3D polyline Biot-Savart -> sensor coupling for RMP-like coils.

Validates the physics premises of the nonaxisymmetric-subtraction plan:
  P1. A finite straight-segment Biot-Savart (field B and vector potential A)
      is enough to model picture-frame in-vessel coils (nova RDP output =
      lines+arcs; arcs are finely polylined here).
  P2. A FULL toroidal flux loop rejects n!=0 energised fields (flux linkage
      from a symmetric RMP coil set cancels), while a point B-probe sees them
      fully -> loops and probes have OPPOSITE sensitivity to the energised
      3D field.
  P3. Magnitude: probe contamination per kA-turn of RMP current vs typical
      axisymmetric signal scales -> is subtraction material?

Checks:
  C1. B at centre of a square loop matches the analytic 2*sqrt(2)*mu0*I/(pi*a).
  C2. Flux linkage via A-line-integral matches mutual inductance symmetry
      (loop<->coil reciprocity) on a coaxial-circles case vs Maxwell formula.
"""

import numpy as np

MU0 = 4e-7 * np.pi


# ---------------------------------------------------------------- primitives
def seg_B(p, a, b, I):
    """B field at points p (N,3) of a finite segment a->b carrying I (analytic)."""
    p = np.atleast_2d(p)
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    ap = p - a
    # component along the wire and perpendicular distance vector
    s = ap @ u
    perp = ap - np.outer(s, u)
    rho = np.linalg.norm(perp, axis=1)
    # angles to the two ends
    s2 = s - L
    r1 = np.linalg.norm(ap, axis=1)
    r2 = np.linalg.norm(p - b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Bmag = MU0 * I / (4 * np.pi * rho) * (s / r1 - s2 / r2)
        e_phi = np.cross(np.broadcast_to(u, perp.shape), perp)
        n = np.linalg.norm(e_phi, axis=1, keepdims=True)
        e_phi = np.where(n > 0, e_phi / n, 0.0)
        B = Bmag[:, None] * e_phi
    return np.nan_to_num(B)


def seg_A(p, a, b, I):
    """Vector potential at points p of finite segment a->b (Coulomb gauge)."""
    p = np.atleast_2d(p)
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    s = (p - a) @ u
    r1 = np.linalg.norm(p - a, axis=1)
    r2 = np.linalg.norm(p - b, axis=1)
    # A = mu0 I/4pi * u * ln( (r1 + s) / (r2 + s - L) )
    with np.errstate(divide="ignore", invalid="ignore"):
        val = MU0 * I / (4 * np.pi) * np.log((r1 + s) / (r2 + (s - L)))
    return np.nan_to_num(np.outer(val, u))


def poly_B(p, path, I):
    return sum(seg_B(p, path[i], path[i + 1], I) for i in range(len(path) - 1))


def poly_A(p, path, I):
    return sum(seg_A(p, path[i], path[i + 1], I) for i in range(len(path) - 1))


def loop_flux(path_src, I, loop_pts):
    """Flux linked by a closed loop (points, closed) from a source polyline:
    Phi = oint A . dl  (trapezoid over loop segments)."""
    mid = 0.5 * (loop_pts[:-1] + loop_pts[1:])
    dl = loop_pts[1:] - loop_pts[:-1]
    A = poly_A(mid, path_src, I)
    return float(np.sum(A * dl))


# ------------------------------------------------------------------- checks
# C1: square loop, side a, field at centre = 2*sqrt(2)*mu0*I/(pi*a)
a_side, I = 0.6, 1000.0
sq = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], float) * (
    a_side / 2
)
Bc = poly_B(np.array([[0, 0, 0.0]]), sq, I)[0]
B_analytic = 2 * np.sqrt(2) * MU0 * I / (np.pi * a_side)
print(
    f"C1 square-centre: Bz={Bc[2]:.6e}  analytic={B_analytic:.6e}  "
    f"rel err={abs(Bc[2] - B_analytic) / B_analytic:.2e}"
)

# C2: mutual inductance of coaxial circles (R1=1.0, R2=0.5, dz=0.3) vs Maxwell
from scipy.special import ellipe, ellipk


def maxwell_M(R1, R2, dz):
    k2 = 4 * R1 * R2 / ((R1 + R2) ** 2 + dz**2)
    k = np.sqrt(k2)
    return MU0 * np.sqrt(R1 * R2) * ((2 / k - k) * ellipk(k2) - (2 / k) * ellipe(k2))


def circle(R, z, n=720):
    t = np.linspace(0, 2 * np.pi, n + 1)
    return np.column_stack([R * np.cos(t), R * np.sin(t), np.full(t.size, z)])


M_num = loop_flux(circle(1.0, 0.0), 1.0, circle(0.5, 0.3))
M_ana = maxwell_M(1.0, 0.5, 0.3)
print(
    f"C2 coaxial-circles mutual: num={M_num:.6e}  Maxwell={M_ana:.6e}  "
    f"rel err={abs(M_num - M_ana) / abs(M_ana):.2e}"
)


# ------------------------------------------------- MAST-like RMP coil set
def frame_coil(phi0, dphi, R, z_lo, z_hi, n_arc=40):
    """Picture-frame coil on a cylinder: two toroidal arcs + two vertical legs."""
    p1 = np.linspace(phi0 - dphi / 2, phi0 + dphi / 2, n_arc)
    lower = np.column_stack([R * np.cos(p1), R * np.sin(p1), np.full(n_arc, z_lo)])
    upper = np.column_stack(
        [R * np.cos(p1[::-1]), R * np.sin(p1[::-1]), np.full(n_arc, z_hi)]
    )
    return np.vstack([lower, upper, lower[:1]])


# 6 lower coils (M7 config, n=3 even parity), R=1.45 m, z in [-1.0,-0.6], 30deg wide
def rmp_set(n_coils=6, polarity_n3=True):
    coils = []
    for k in range(n_coils):
        phi0 = 2 * np.pi * k / n_coils
        sgn = (-1) ** k if polarity_n3 else 1  # n=3 alternating polarity
        coils.append((frame_coil(phi0, np.deg2rad(30), 1.45, -1.0, -0.6), sgn))
    return coils


I_RMP = 5600.0  # A-turns (4 turns x 1.4 kA)
coils = rmp_set()

# sensors: catalogued MAST positions
loop_p5l = circle(1.163, -1.089)  # fl_p5l_1 (bay loop, near coils)
loop_cc = circle(0.28, 0.0)  # centre-column loop (far)
probe_pos = np.array([[1.85, 0.0, -0.15]])  # obv-like outboard probe
probe_n = np.array([0.0, 0.0, 1.0])  # vertical pickup

flux_p5l = sum(sgn * loop_flux(path, I_RMP, loop_p5l) for path, sgn in coils)
flux_cc = sum(sgn * loop_flux(path, I_RMP, loop_cc) for path, sgn in coils)
flux_one = loop_flux(coils[0][0], I_RMP, loop_p5l)  # single coil, no cancellation
B_probe = sum(sgn * poly_B(probe_pos, path, I_RMP) for path, sgn in coils)[0]
B_probe_one = poly_B(probe_pos, coils[0][0], I_RMP)[0]

print(f"\nRMP n=3 set @ {I_RMP:.0f} A-turns:")
print(f"  bay flux loop (fl_p5l_1)  full-set linkage: {flux_p5l:+.3e} Wb")
print(f"  bay flux loop             SINGLE-coil link : {flux_one:+.3e} Wb")
print(f"  centre-column loop        full-set linkage: {flux_cc:+.3e} Wb")
print(
    f"  outboard probe B.n        full-set: {B_probe @ probe_n:+.3e} T "
    f"(|B|={np.linalg.norm(B_probe):.3e})"
)
print(f"  outboard probe B.n        single-coil: {B_probe_one @ probe_n:+.3e} T")

# scale context: typical MAST equilibrium probe fields ~0.05-0.3 T; loop flux ~0.1-1 Wb
print(
    f"\nContamination scale: probe {1e4 * abs(B_probe @ probe_n):.2f} G vs "
    f"equilibrium-scale ~500-3000 G -> {100 * abs(B_probe @ probe_n) / 0.1:.2f}% of a 0.1 T signal"
)
