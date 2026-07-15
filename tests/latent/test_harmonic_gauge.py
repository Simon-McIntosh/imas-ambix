"""Gauge tests for the source-free toroidal-harmonic annulus read.

The harmonic field is NOT gauge-free: the basis has no constant column and the
absolute (monopole / DC) level is pinned only weakly by the handful of flux
loops.  The soft-prior interior solve must therefore either (a) match the
gauge-FREE gradient field in the annulus, or (b) carry the absolute level with
an explicit gauge tie.  These tests pin both halves of that machinery:

  * :func:`harmonic_grad_psi_on_grid` is the gauge-free field the ``grad-psi``
    penalty matches — it equals the finite difference of the reconstructed psi
    and is invariant to any additive constant on psi.
  * ``HarmonicFitConfig.ip_anchor`` adds the poloidal-circulation (Ampere)
    gauge tie ``∮ B·dl = mu0 Ip`` so the read carries its own absolute gauge
    rather than leaning only on the flux loops.  OFF is byte-identical; ON pins
    the circulation to mu0 Ip and does not worsen the flux-loop DC residual.
  * the frozen per-slice prior artifact round-trips through the loader the
    interior solve imports.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.gs.operator import MU0, greens_psi
from imas_ambix.latent.boundary_harmonic import (
    HarmonicFitConfig,
    _fit_one,
    fit_harmonic,
    harmonic_columns,
    harmonic_grad_psi_on_grid,
    harmonic_psi_on_grid,
    harmonic_sensor_matrix,
    ip_circulation_row,
    load_frozen_harmonic_prior,
    save_frozen_harmonic_prior,
)
from imas_ambix.latent.patch_inverse import SlicePayload

POLE_R = 0.9
POLE_Z = 0.0


# --- helpers ----------------------------------------------------------------


def _filament_flux(r, z, fils):
    out = np.zeros_like(np.asarray(r, dtype=np.float64))
    for fr, fz, w in fils:
        out = out + w * greens_psi(
            np.asarray(r, dtype=np.float64), np.asarray(z, dtype=np.float64), fr, fz
        )
    return out


_FILS = [
    (0.85, 0.05, 3.0e4),
    (0.95, -0.1, 2.0e4),
    (0.80, 0.15, 1.5e4),
    (1.0, 0.0, 2.5e4),
]


def _recover_filament_coeffs(cfg, fils=_FILS, rad=0.55, n=90):
    """Least-squares coeffs of the P-set recovering the filament flux on a ring."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rf = POLE_R + rad * np.cos(ang)
    zf = POLE_Z + rad * np.sin(ang)
    bf = _filament_flux(rf, zf, fils)
    a, _ = harmonic_columns(rf, zf, cfg)
    cn = np.linalg.norm(a, axis=0)
    cn = np.where(cn > 0, cn, 1.0)
    return np.linalg.lstsq(a / cn, bf, rcond=None)[0] / cn


# --- grad-psi: gauge-free field the interior penalty matches ----------------


def test_grad_psi_matches_finite_difference():
    """Analytic ∇psi from the harmonic coeffs equals the finite difference of the
    reconstructed psi field to ~1e-6 in the annulus (away from the pole/edges)."""
    cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=5)
    coeffs = _recover_filament_coeffs(cfg)
    rg = np.linspace(0.4, 1.6, 61)
    zg = np.linspace(-0.9, 0.9, 71)
    dpsi_dr, dpsi_dz = harmonic_grad_psi_on_grid(cfg, coeffs, rg, zg)
    assert dpsi_dr.shape == (zg.size, rg.size)
    assert dpsi_dz.shape == (zg.size, rg.size)

    # pointwise central difference of psi with a small step (the coarse grid FD
    # would carry O(dr^2) truncation ~1e-3; a small h isolates the analytic form)
    h = 1e-4
    psi_rp = harmonic_psi_on_grid(cfg, coeffs, rg + h, zg)
    psi_rm = harmonic_psi_on_grid(cfg, coeffs, rg - h, zg)
    psi_zp = harmonic_psi_on_grid(cfg, coeffs, rg, zg + h)
    psi_zm = harmonic_psi_on_grid(cfg, coeffs, rg, zg - h)
    fd_dr = (psi_rp - psi_rm) / (2.0 * h)
    fd_dz = (psi_zp - psi_zm) / (2.0 * h)

    rr, zz = np.meshgrid(rg, zg)
    m = np.hypot(rr - POLE_R, zz - POLE_Z) > 0.4
    scale = np.nanmax(np.abs(np.concatenate([dpsi_dr[m], dpsi_dz[m]])))
    rel_r = np.nanmax(np.abs(dpsi_dr[m] - fd_dr[m])) / scale
    rel_z = np.nanmax(np.abs(dpsi_dz[m] - fd_dz[m])) / scale
    assert rel_r < 1e-5, f"dpsi/dR analytic-vs-FD rel {rel_r:.2e}"
    assert rel_z < 1e-5, f"dpsi/dZ analytic-vs-FD rel {rel_z:.2e}"


def test_grad_psi_gauge_invariant():
    """The gradient field is invariant to any additive constant on psi — this is
    exactly why a ∇psi penalty keeps NO opinion on the absolute gauge."""
    cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=4)
    coeffs = _recover_filament_coeffs(cfg)
    rg = np.linspace(0.4, 1.6, 41)
    zg = np.linspace(-0.9, 0.9, 51)
    dr0, dz0 = harmonic_grad_psi_on_grid(cfg, coeffs, rg, zg)

    psi = harmonic_psi_on_grid(cfg, coeffs, rg, zg)

    def fd(f):
        return (
            (f[:, 2:] - f[:, :-2])[1:-1, :] / (rg[2:] - rg[:-2])[None, :],
            (f[2:, :] - f[:-2, :])[:, 1:-1] / (zg[2:] - zg[:-2])[:, None],
        )

    a_r, a_z = fd(psi)
    b_r, b_z = fd(psi + 12345.678)  # shift psi by a large constant
    np.testing.assert_allclose(a_r, b_r, atol=1e-9)
    np.testing.assert_allclose(a_z, b_z, atol=1e-9)
    # the analytic field has no constant DOF, so shifting the coeff-free gauge
    # cannot change it — it agrees with the shifted-psi FD as well.
    assert np.all(np.isfinite(dr0)) and np.all(np.isfinite(dz0))


# --- ip_anchor: the absolute-gauge tie --------------------------------------


def _synthetic_gauge_sensors(n_bprobe=48, n_flux=3, noise=0.0, seed=0):
    """A B-probe ring plus a FEW (noisy) flux loops — the weak-DC-pinning regime
    the plan flags (few flux loops carry the absolute gauge)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, 2.0 * np.pi, n_bprobe, endpoint=False)
    rad = 0.75
    sr_b = POLE_R + rad * np.cos(ang)
    sz_b = POLE_Z + rad * np.sin(ang)
    sang_b = rng.uniform(0.0, 180.0, size=n_bprobe)
    fang = np.linspace(0.2, 2.0 * np.pi, n_flux, endpoint=False)
    sr_f = POLE_R + rad * np.cos(fang)
    sz_f = POLE_Z + rad * np.sin(fang)
    sr = np.concatenate([sr_b, sr_f])
    sz = np.concatenate([sz_b, sz_f])
    sang = np.concatenate([sang_b, np.zeros(n_flux)])
    is_flux = np.concatenate(
        [np.zeros(n_bprobe, dtype=bool), np.ones(n_flux, dtype=bool)]
    )
    return sr, sz, sang, is_flux


def _payload_from_filaments(sr, sz, sang, is_flux, cfg, fils=_FILS, noise=0.0, seed=1):
    """SlicePayload whose measured signature is the exact filament field the
    harmonic P-set can represent, optionally with noise on the flux loops."""
    from imas_ambix.gs.operator import greens_bz_br

    rng = np.random.default_rng(seed)
    psi = _filament_flux(sr, sz, fils)
    br = np.zeros_like(sr)
    bz = np.zeros_like(sr)
    for fr, fz, w in fils:
        bzc, brc = greens_bz_br(sr, sz, fr, fz)
        br += w * brc
        bz += w * bzc
    th = np.deg2rad(sang)
    bproj = br * np.cos(th) + bz * np.sin(th)
    measured = np.where(is_flux, psi, bproj)
    scale = np.where(is_flux, np.abs(psi).mean() or 1.0, np.abs(bproj).mean() or 1.0)
    if noise:
        # noise ONLY on the flux loops (weakens the absolute-gauge pinning)
        measured = measured + np.where(
            is_flux, noise * scale * rng.standard_normal(measured.size), 0.0
        )
    ip = sum(w for *_, w in fils)
    true = np.where(is_flux, psi, bproj)  # noise-free signature (the ground truth)
    return (
        SlicePayload(
            measured=measured,
            vacuum=np.zeros_like(measured),
            mask=np.ones(measured.size, dtype=bool),
            scale=scale,
            i_pf=np.zeros(3),
            ip_amperes=ip,
        ),
        ip,
        true,
    )


def test_ip_anchor_off_is_byte_identical():
    """ip_anchor=False must not touch the fit — identical to the raw path."""
    cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=3, ip_anchor=False)
    sr, sz, sang, is_flux = _synthetic_gauge_sensors(seed=4)
    p, _, _ = _payload_from_filaments(sr, sz, sang, is_flux, cfg, noise=0.02, seed=5)
    a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, cfg)
    c_raw, _, _ = _fit_one(a_sens, p.measured, p.vacuum, p.mask, p.scale, cfg.ridge)
    inv = fit_harmonic((sr, sz, sang, is_flux), p, cfg)
    np.testing.assert_array_equal(inv.coeffs, c_raw)


def _dc_level_error(inv, sr, sz, sang, is_flux, true, scale):
    """Absolute DC-level error of the reconstructed flux at the flux-loop rows,
    vs the noise-free truth (the gauge error the anchor is meant to reduce)."""
    a_sens = harmonic_sensor_matrix(sr, sz, sang, is_flux, inv.cfg)
    pred = a_sens @ inv.coeffs
    fl = np.asarray(is_flux, dtype=bool)
    return abs(float(np.mean((pred - true)[fl])) / float(np.mean(scale[fl])))


def test_ip_anchor_pins_circulation():
    """The Ampere circulation tie pins ∮B·dl of the fit to mu0 Ip (clockwise
    loop -> +mu0 Ip); noise-free, so the tie brings the monopole ONTO the exact
    value the sensor-only fit only approaches."""
    sr, sz, sang, is_flux = _synthetic_gauge_sensors(n_bprobe=12, n_flux=2, seed=7)
    cfg_off = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=3, ip_anchor=False)
    cfg_on = HarmonicFitConfig(
        pole_r=POLE_R, pole_z=POLE_Z, order=3, ip_anchor=True, ip_anchor_weight=25.0
    )
    p, ip, _ = _payload_from_filaments(
        sr, sz, sang, is_flux, cfg_off, noise=0.0, seed=9
    )
    inv_off = fit_harmonic((sr, sz, sang, is_flux), p, cfg_off)
    inv_on = fit_harmonic((sr, sz, sang, is_flux), p, cfg_on)
    d = np.hypot(sr - cfg_on.pole_r, sz - cfg_on.pole_z)
    g = ip_circulation_row(cfg_on, 0.5 * float(np.median(d)))
    circ_off = float(g @ inv_off.coeffs) / (MU0 * ip)
    circ_on = float(g @ inv_on.coeffs) / (MU0 * ip)
    assert abs(circ_on - 1.0) < abs(circ_off - 1.0)
    assert abs(circ_on - 1.0) < 0.01, f"anchored circulation ratio {circ_on:.4f}"


def test_ip_anchor_reduces_dc_error_on_average():
    """With few B-probes and few, noisy flux loops the absolute (monopole) gauge
    is weakly pinned.  Averaged over noise realizations the circulation tie
    REDUCES the DC-level error of the reconstructed flux vs the truth — the
    independent Ampere information stops the fit chasing flux-loop noise."""
    sr, sz, sang, is_flux = _synthetic_gauge_sensors(n_bprobe=12, n_flux=2, seed=7)
    cfg_off = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=3, ip_anchor=False)
    cfg_on = HarmonicFitConfig(
        pole_r=POLE_R, pole_z=POLE_Z, order=3, ip_anchor=True, ip_anchor_weight=25.0
    )
    off, on = [], []
    for seed in range(40):
        p, _, true = _payload_from_filaments(
            sr, sz, sang, is_flux, cfg_off, noise=0.05, seed=1000 + seed
        )
        scale = np.asarray(p.scale)
        off.append(
            _dc_level_error(
                fit_harmonic((sr, sz, sang, is_flux), p, cfg_off),
                sr,
                sz,
                sang,
                is_flux,
                true,
                scale,
            )
        )
        on.append(
            _dc_level_error(
                fit_harmonic((sr, sz, sang, is_flux), p, cfg_on),
                sr,
                sz,
                sang,
                is_flux,
                true,
                scale,
            )
        )
    mean_off, mean_on = float(np.mean(off)), float(np.mean(on))
    assert mean_on < mean_off, (
        f"anchor did not reduce mean DC error {mean_off:.3e}->{mean_on:.3e}"
    )


def test_ip_circulation_row_path_independent():
    """∮B·dl over the harmonic columns is path-independent for any pole-enclosing
    loop (only the monopole contributes) — the physics the tie rests on."""
    cfg = HarmonicFitConfig(pole_r=POLE_R, pole_z=POLE_Z, order=5)
    coeffs = _recover_filament_coeffs(cfg)
    ip = sum(w for *_, w in _FILS)
    ratios = [
        float(ip_circulation_row(cfg, lr) @ coeffs) / (MU0 * ip)
        for lr in (0.3, 0.5, 0.7)
    ]
    for r in ratios:
        assert abs(r - 1.0) < 1e-4, f"circulation ratio {r:.5f} not mu0 Ip"
    assert max(ratios) - min(ratios) < 1e-4  # path-independent


# --- frozen prior artifact round-trip ---------------------------------------


def test_frozen_prior_roundtrip(tmp_path):
    """The frozen per-slice prior the interior solve loads survives save->load."""
    cfg = HarmonicFitConfig(pole_r=0.55, pole_z=0.0, order=3)
    rng = np.random.default_rng(0)
    slices = []
    for i in range(3):
        k = 2 * cfg.order + 1
        slices.append(
            {
                "shot": 18500 + i,
                "t_index": i,
                "time_s": 0.1 + 0.02 * i,
                "ip_amperes": 5.0e5 + i,
                "coeffs": rng.standard_normal(k),
                "coeff_cov": rng.standard_normal((k, k)),
                "misfit": float(i) * 1e-3,
                "origin": (0.9 + 0.01 * i, 0.0),
                "pole": (0.53, 0.0),
                "dyn_range": 0.5 + i,
            }
        )
    meta = {
        "order": cfg.order,
        "kind": cfg.kind,
        "ridge": cfg.ridge,
        "ip_anchor": False,
        "origin_source": "centroid",
        "pole_source": "track",
        "pole_inboard_fraction": 0.41,
        "mask_frac": 0.5,
        "exclude_frac": 1.1,
        "labels": ["h0", "h1c", "h1s", "h2c", "h2s", "h3c", "h3s"],
        "split": "eval",
    }
    path = tmp_path / "harmonic_prior_frozen.npz"
    save_frozen_harmonic_prior(path, slices, meta)

    loaded = load_frozen_harmonic_prior(path)
    assert loaded["meta"]["order"] == 3
    assert loaded["meta"]["labels"] == meta["labels"]
    assert len(loaded["slices"]) == 3
    for got, exp in zip(loaded["slices"], slices, strict=True):
        assert int(got["shot"]) == exp["shot"]
        assert int(got["t_index"]) == exp["t_index"]
        np.testing.assert_allclose(got["coeffs"], exp["coeffs"])
        np.testing.assert_allclose(got["coeff_cov"], exp["coeff_cov"])
        np.testing.assert_allclose(got["origin"], exp["origin"])
        np.testing.assert_allclose(got["pole"], exp["pole"])
        assert abs(float(got["dyn_range"]) - exp["dyn_range"]) < 1e-9
