"""Rectangular-circuit collapse: one thick cylinder per filled coil pack.

A lattice of co-current filaments filling a rectangle has the same field as a
single finite-cross-section cylinder carrying the summed current -- the collapse
must reproduce that field (area-conservation gate), leave sparse/non-rectangular
circuits alone, and preserve the summed weight/turns.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.gs.cylinder import hybrid_greens
from imas_ambix.gs.geometry import PFFilament, collapse_rectangular_circuits


def _pack(circuit, r0, z0, w, h, nr, nz, xmult_each):
    """A filled nr x nz lattice of equal filaments tiling the (w, h) box."""
    dr, dz = w / nr, h / nz
    out = []
    for i in range(nr):
        for j in range(nz):
            out.append(
                PFFilament(
                    r=r0 - w / 2 + (i + 0.5) * dr,
                    z=z0 - h / 2 + (j + 0.5) * dz,
                    turns=1.0,
                    width=dr,
                    height=dz,
                    circuit=circuit,
                    xmult=xmult_each,
                )
            )
    return out


def test_filled_pack_collapses_to_one_cylinder():
    fils = _pack(circuit=7, r0=0.5, z0=0.1, w=0.12, h=0.8, nr=3, nz=20, xmult_each=0.5)
    out = collapse_rectangular_circuits(fils)
    assert len(out) == 1
    c = out[0]
    assert c.circuit == 7
    assert abs(c.xmult - 0.5 * 60) < 1e-9  # Σ xmult
    assert abs(c.turns - 60.0) < 1e-9  # Σ turns
    assert abs(c.width - 0.12) < 1e-6 and abs(c.height - 0.8) < 1e-6
    assert abs(c.r - 0.5) < 1e-6 and abs(c.z - 0.1) < 1e-6


def test_collapsed_field_matches_filament_sum():
    fils = _pack(circuit=7, r0=0.5, z0=0.1, w=0.12, h=0.8, nr=3, nz=20, xmult_each=0.5)
    out = collapse_rectangular_circuits(fils)
    # sensors a realistic distance away (a few pack-widths outboard + off-axis)
    sr = np.array([0.9, 1.2, 0.75, 1.0])
    sz = np.array([0.0, 0.3, -0.4, 0.6])

    def field(fs):
        acc = np.zeros(sr.size)
        for f in fs:
            psi, _, _ = hybrid_greens(
                sr, sz, f.r, f.z, max(abs(f.width), 0.005), max(abs(f.height), 0.005)
            )
            acc += f.xmult * psi
        return acc

    full = field(fils)
    coll = field(out)
    rel = np.abs(coll - full) / np.abs(full).max()
    assert rel.max() < 0.02, f"collapsed field deviates {rel.max():.3%}"


def test_sparse_ring_not_collapsed():
    # four filaments at the corners of a big box — area sum << box area
    fils = [
        PFFilament(r, z, 1.0, 0.02, 0.02, 3, 1.0)
        for r, z in [(0.4, -0.4), (0.6, -0.4), (0.4, 0.4), (0.6, 0.4)]
    ]
    out = collapse_rectangular_circuits(fils)
    assert len(out) == 4  # left untouched


def test_mixed_sign_pack_not_collapsed():
    fils = _pack(circuit=9, r0=0.5, z0=0.0, w=0.1, h=0.4, nr=2, nz=10, xmult_each=0.5)
    fils[0] = PFFilament(
        fils[0].r, fils[0].z, 1.0, fils[0].width, fils[0].height, 9, -0.5
    )
    out = collapse_rectangular_circuits(fils)
    assert len(out) == len(fils)  # mixed sign — keep the lattice


def test_single_filament_passes_through():
    fils = [PFFilament(0.5, 0.0, 1.0, 0.05, 0.05, 4, 1.0)]
    out = collapse_rectangular_circuits(fils)
    assert len(out) == 1 and out[0] is fils[0]
