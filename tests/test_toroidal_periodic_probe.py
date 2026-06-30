"""Tests for periodic-φ geometry, the toroidal saddle array, and STFT phase.

Three contracts the toroidal-array build must satisfy:

1. PERIODIC adjacency — the encoded geometry distance between φ=359° and φ=1° is
   SMALL (they are physically adjacent on the torus) while φ=0° vs φ=180° is
   LARGE.  A linear angle-in-degrees encoding does the opposite; this is the
   sharp correctness point the lead asked for (identify 2π−ε and 0+ε as
   adjacent).
2. The L2 saddle toroidal array resolves to 12 channels at distinct toroidal
   angles with finite φ/R/Z.
3. SKILL — a periodic-PE + STFT relational encoder over the 12-φ saddle geometry
   RECOVERS the toroidal mode number n of a synthetic rotating mode
   B(φ,t)=cos(nφ−ωt), and that recovery is INVARIANT to a global φ-offset that
   crosses the 360°→0° seam.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from imas_ambix.worldmodel.diagnostics_equilibrium_probe import (
    N_GEOM_FEATURES,
    DiagnosticsEquilibriumProbe,
    DiagnosticsProbeConfig,
    StreamSpec,
    encode_periodic_geometry,
)


def _geom_row_with_phi_deg(phi_deg: float) -> torch.Tensor:
    """A geometry row (N_GEOM_FEATURES,) carrying φ (radians) in the phi column."""
    row = torch.full((N_GEOM_FEATURES,), float("nan"))
    row[0] = 2.0  # r
    row[1] = 0.0  # z
    row[2] = float(np.deg2rad(phi_deg))  # phi (radians)
    return row


def test_periodic_phi_adjacency_across_seam():
    """φ=359° and φ=1° encode CLOSE; φ=0° and φ=180° encode FAR.

    The whole point of the periodic encoding: angles on a circle must be
    continuous across the 0/2π seam.
    """
    rows = torch.stack(
        [
            _geom_row_with_phi_deg(359.0),
            _geom_row_with_phi_deg(1.0),
            _geom_row_with_phi_deg(0.0),
            _geom_row_with_phi_deg(180.0),
        ]
    )
    enc = encode_periodic_geometry(rows)  # NaN -> 0 happens at the encoder; here
    # only the phi/angle columns matter — compare on the finite (sin,cos) cols.
    enc = torch.where(torch.isfinite(enc), enc, torch.zeros_like(enc))

    d_seam = torch.linalg.norm(enc[0] - enc[1]).item()  # 359° vs 1° (adjacent)
    d_far = torch.linalg.norm(enc[2] - enc[3]).item()  # 0° vs 180° (opposite)

    assert d_seam < 0.1, f"359° and 1° should be adjacent, got distance {d_seam}"
    assert d_far > 1.5, f"0° and 180° should be far, got distance {d_far}"
    assert d_seam < d_far


def test_periodic_phi_monotone_distance_to_half_turn():
    """Encoded distance grows from 0 at Δφ=0 to a max at Δφ=180°, then shrinks."""
    base = encode_periodic_geometry(_geom_row_with_phi_deg(0.0).unsqueeze(0))
    base = torch.where(torch.isfinite(base), base, torch.zeros_like(base))
    dists = []
    for dphi in (0, 30, 90, 180, 270, 330, 360):
        e = encode_periodic_geometry(_geom_row_with_phi_deg(float(dphi)).unsqueeze(0))
        e = torch.where(torch.isfinite(e), e, torch.zeros_like(e))
        dists.append(torch.linalg.norm(e - base).item())
    # Δφ=0 and Δφ=360 are identical (seam); Δφ=180 is the maximum.
    assert dists[0] == pytest.approx(0.0, abs=1e-5)
    assert dists[-1] == pytest.approx(0.0, abs=1e-5)
    assert dists[3] == max(dists)  # 180° is farthest
    # symmetric: 30° and 330° equidistant from 0°.
    assert dists[1] == pytest.approx(dists[5], abs=1e-4)


# ---------------------------------------------------------------------------
# Toroidal saddle-array geometry (needs the L2 store on /work)
# ---------------------------------------------------------------------------


def _have_l2(shot_id: int) -> bool:
    from imas_ambix.data.paths import LEVEL2_DIR

    return (LEVEL2_DIR / f"{shot_id}.zarr").exists()


@pytest.mark.skipif(not _have_l2(11766), reason="L2 magnetics store not reachable")
def test_saddle_toroidal_geometry_resolves():
    """The L2 saddle array -> 12 channels at distinct φ with finite R/Z."""
    from imas_ambix.tokenizer.geometry_reader import saddle_toroidal_geometry

    src = saddle_toroidal_geometry(11766)
    assert src is not None
    names, feats, kinds = src
    assert len(names) == 12
    assert all(k == "toroidal_saddle" for k in kinds)
    phi = feats[:, 2]
    assert np.isfinite(phi).all()
    assert np.isfinite(feats[:, 0]).all() and np.isfinite(feats[:, 1]).all()
    # 12 distinct toroidal angles spanning the torus (15°..345° in 30° steps).
    phi_deg = np.sort(np.rad2deg(phi) % 360)
    assert len(np.unique(np.round(phi_deg))) == 12
    assert phi_deg.max() - phi_deg.min() > 300  # spread around the torus


# ---------------------------------------------------------------------------
# SKILL: recover the toroidal mode number from a synthetic rotating mode
# ---------------------------------------------------------------------------


def _saddle_phi_deg() -> np.ndarray:
    """The 12 saddle toroidal angles (deg): 15, 45, ..., 345."""
    return np.arange(15.0, 360.0, 30.0)


def _rotating_mode_batch(
    n_values, phi_deg, n_steps, *, phi_offset_deg=0.0, seed=0, samples_per_n=12
):
    """Build B(φ,t)=cos(nφ−ωt) samples for a list of mode numbers n.

    Returns (values (B, n_steps, C) float32, labels (B,) int64) where the label
    is the index into ``n_values``.  φ is the saddle geometry rotated by
    ``phi_offset_deg`` (to test seam invariance).  ω sign is randomised per
    sample so the readout must use the spatial mode, not the temporal frequency.
    """
    rng = np.random.default_rng(seed)
    phi = np.deg2rad((phi_deg + phi_offset_deg) % 360.0)  # (C,)
    t = np.linspace(0.0, 1.0, n_steps)  # (S,)
    vals, labels = [], []
    for li, n in enumerate(n_values):
        for _ in range(samples_per_n):
            omega = rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 6.0) * np.pi
            ph0 = rng.uniform(0, 2 * np.pi)
            # (S, C) = cos(n φ − ω t + ph0)
            field = np.cos(n * phi[None, :] - omega * t[:, None] + ph0).astype(
                np.float32
            )
            field += 0.02 * rng.standard_normal(field.shape).astype(np.float32)
            vals.append(field)
            labels.append(li)
    v = np.stack(vals).astype(np.float32)  # (B, S, C)
    y = np.asarray(labels, dtype=np.int64)
    return v, y


def _mode_recovery_accuracy(phi_offset_deg, *, seed=0, scramble_phi=False):
    """Train the periodic-PE + STFT encoder + readout to classify the mode n.

    Returns held-out accuracy.  Trains the relational encoder END-TO-END on the
    rotating-mode samples; if the periodic-φ geometry + STFT phase path captures
    the toroidal mode, held-out accuracy rises well above the 1/3 chance.  With
    ``scramble_phi`` the 12 sensors' φ are permuted (the geometry no longer
    matches the data) — a control that must score WORSE if the model is using
    the geometry rather than a channel-index shortcut.
    """
    n_values = [1, 2, 3]
    n_steps, c = 12, 12
    samples_per_n = 40
    phi_deg = _saddle_phi_deg()

    specs = [StreamSpec(name="saddle", vocab=257, channels=c)]
    cfg = DiagnosticsProbeConfig(
        streams=specs,
        n_steps=n_steps,
        target_dim=12,
        d_model=48,
        n_heads=4,
        n_layers=2,
        continuous_value=True,
        stft_phase=True,
        use_machine_tokens=False,
        dropout=0.0,
    )
    torch.manual_seed(seed)
    model = DiagnosticsEquilibriumProbe(cfg)
    readout = torch.nn.Linear(cfg.d_model, len(n_values))

    # geometry: 12 saddle channels at their φ (rotated by the offset).  The DATA
    # always uses the true (rotated) φ; only the geometry the model SEES is
    # optionally scrambled, to test that the geometry is what carries the mode.
    used_phi = (phi_deg + phi_offset_deg) % 360.0
    seen_phi = (
        np.random.default_rng(0).permutation(used_phi) if scramble_phi else used_phi
    )
    geom = torch.full((c, N_GEOM_FEATURES), float("nan"))
    geom[:, 0] = 2.0
    geom[:, 1] = 0.0
    geom[:, 2] = torch.from_numpy(np.deg2rad(seen_phi).astype(np.float32))
    kinds = torch.full((c,), 9, dtype=torch.int64)  # "toroidal_saddle" index

    v_tr, y_tr = _rotating_mode_batch(
        n_values,
        phi_deg,
        n_steps,
        phi_offset_deg=phi_offset_deg,
        seed=seed,
        samples_per_n=samples_per_n,
    )
    v_te, y_te = _rotating_mode_batch(
        n_values,
        phi_deg,
        n_steps,
        phi_offset_deg=phi_offset_deg,
        seed=seed + 100,
        samples_per_n=samples_per_n,
    )

    def _embed(values):
        b, s, cc = values.shape
        signals = {"saddle": torch.zeros(b, s, cc, dtype=torch.int64)}
        vdict = {"saddle": torch.from_numpy(values)}
        gdict = {"saddle": geom.unsqueeze(0).expand(b, -1, -1).contiguous()}
        kdict = {"saddle": kinds.unsqueeze(0).expand(b, -1).contiguous()}
        return model.pooled_embedding(signals, gdict, kdict, values=vdict, machine=None)

    yt = torch.from_numpy(y_tr)
    params = list(model.parameters()) + list(readout.parameters())
    opt = torch.optim.Adam(params, lr=5e-3, weight_decay=1e-4)
    model.train()
    for _ in range(250):
        opt.zero_grad()
        logits = readout(_embed(v_tr))
        loss = torch.nn.functional.cross_entropy(logits, yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = readout(_embed(v_te)).argmax(1).numpy()
    return float((pred == y_te).mean())


def test_recovers_toroidal_mode_number():
    """The periodic-PE + STFT encoder recovers n, and the φ geometry carries it.

    Recovery is well above the 1/3 chance, AND beats a φ-scrambled control — the
    model uses the toroidal GEOMETRY (not a channel-index shortcut) to read the
    mode number off the rotating-mode samples.
    """
    acc = _mode_recovery_accuracy(phi_offset_deg=0.0, seed=0)
    acc_scrambled = _mode_recovery_accuracy(
        phi_offset_deg=0.0, seed=0, scramble_phi=True
    )
    assert acc > 0.6, f"toroidal mode-number recovery accuracy {acc} too low"
    assert acc > acc_scrambled + 0.1, (
        f"correct-φ ({acc}) must beat scrambled-φ ({acc_scrambled}) — proves the "
        "periodic geometry carries the mode"
    )


def test_mode_recovery_invariant_to_seam_crossing_offset():
    """Rotating all φ across the 360°→0° seam leaves n-recovery intact.

    A global toroidal offset of 350° wraps every sensor past the seam; a correct
    periodic encoding keeps the relative phase ramp (the mode) intact, so the
    readout still classifies n.  A linear-degrees encoding would scramble it.
    """
    acc_base = _mode_recovery_accuracy(phi_offset_deg=0.0, seed=1)
    acc_seam = _mode_recovery_accuracy(phi_offset_deg=350.0, seed=1)
    assert acc_seam > 0.6, f"seam-crossing recovery {acc_seam} too low"
    # recovery is essentially unchanged by the seam-crossing offset.
    assert abs(acc_base - acc_seam) < 0.25
