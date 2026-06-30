"""Tests for the space-time diagnostics->equilibrium probe + geometry mapping.

Covers two contracts:

1. the probe forward runs on synthetic ``(B, n_steps, channels)`` ids + a
   ``(channels, 10)`` geometry block for streams of different channel counts,
   returns ``(B, 2*target_dim)`` (mean + log_sigma), preserves the spatial AND
   temporal axes through attention (no channel-mean / no step-mean pooling in
   the module — every ``(sensor, step)`` token reaches the encoder), and the
   continuous-value ablation path runs;
2. the staged-magnetics name -> geometry mapping resolves a sample of column
   names (``b_field_pol_probe_{kind}_field[i]``, ``flux_loop_flux[i]``, ``ip``)
   to finite R/Z, with the orientation normal correct per probe family.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from imas_ambix.worldmodel.diagnostics_equilibrium_probe import (
    N_GEOM_FEATURES,
    DiagnosticsEquilibriumProbe,
    DiagnosticsProbeConfig,
    StreamSpec,
    sensor_kind_index,
    set_xpoint_loss,
)


def test_set_xpoint_loss_is_permutation_invariant():
    """Swapping the two target nulls yields IDENTICAL loss (the core property)."""
    torch.manual_seed(0)
    b, dim = 5, 14
    mean = torch.randn(b, dim)
    log_sigma = torch.zeros(b, dim)
    presence = torch.randn(b, 2)
    target = torch.randn(b, dim)
    mask = torch.ones(b, dim)  # both null slots present (DN) for every sample

    loss_ab = set_xpoint_loss(
        mean, log_sigma, presence, target, mask, xpoint_start=2, n_slots=2
    )
    # swap target slot 0 <-> slot 1 (components 2,3 <-> 4,5) AND the mask.
    swapped = target.clone()
    swapped[:, 2:4], swapped[:, 4:6] = target[:, 4:6], target[:, 2:4]
    mask_sw = mask.clone()
    mask_sw[:, 2:4], mask_sw[:, 4:6] = mask[:, 4:6], mask[:, 2:4]
    loss_ba = set_xpoint_loss(
        mean, log_sigma, presence, swapped, mask_sw, xpoint_start=2, n_slots=2
    )
    assert torch.allclose(loss_ab, loss_ba, atol=1e-5), (
        f"set loss not permutation-invariant: {loss_ab.item()} vs {loss_ba.item()}"
    )


def test_set_xpoint_loss_handles_counts():
    """Loss is finite + differentiable for 0/1/2-null presence patterns."""
    torch.manual_seed(1)
    b, dim = 6, 14
    mean = torch.randn(b, dim, requires_grad=True)
    log_sigma = torch.zeros(b, dim)
    presence = torch.randn(b, 2, requires_grad=True)
    target = torch.randn(b, dim)
    mask = torch.ones(b, dim)
    # sample 0: 0 nulls; sample 1: 1 null (slot0 only); rest: 2 nulls.
    mask[0, 2:6] = 0.0
    mask[1, 4:6] = 0.0
    loss = set_xpoint_loss(
        mean, log_sigma, presence, target, mask, xpoint_start=2, n_slots=2
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert presence.grad is not None


def _synthetic_inputs(specs, n_steps, target_dim, *, batch=3, continuous=False):
    """Build synthetic per-stream ids + geometry + kinds + values + machine."""
    rng = np.random.default_rng(0)
    signals, geometry, kinds, values = {}, {}, {}, {}
    for s in specs:
        ids = rng.integers(0, s.vocab, size=(batch, n_steps, s.channels))
        signals[s.name] = torch.from_numpy(ids.astype(np.int64))
        # half the channels carry geometry, half are NaN (geometry-free).
        geo = rng.normal(size=(batch, s.channels, N_GEOM_FEATURES)).astype(np.float32)
        geo[:, ::2, :] = np.nan
        geometry[s.name] = torch.from_numpy(geo)
        k = rng.integers(0, 8, size=(batch, s.channels)).astype(np.int64)
        kinds[s.name] = torch.from_numpy(k)
        v = rng.normal(size=(batch, n_steps, s.channels)).astype(np.float32)
        values[s.name] = torch.from_numpy(v)
    machine = torch.from_numpy(rng.normal(size=(batch, 6, 3)).astype(np.float32))
    return signals, geometry, kinds, values, machine


def test_probe_forward_shape_and_no_pooling():
    """Forward runs on 2 streams of different widths -> (B, 2*target_dim)."""
    n_steps, target_dim, batch = 12, 12, 3
    specs = [
        StreamSpec(name="magnetics", vocab=257, channels=94),
        StreamSpec(name="xma", vocab=8, channels=20),
    ]
    cfg = DiagnosticsProbeConfig(
        streams=specs,
        n_steps=n_steps,
        target_dim=target_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    model = DiagnosticsEquilibriumProbe(cfg)
    signals, geometry, kinds, _values, machine = _synthetic_inputs(
        specs, n_steps, target_dim, batch=batch
    )
    mean, log_sigma = model(signals, geometry, kinds, machine=machine)
    assert mean.shape == (batch, target_dim)
    assert log_sigma.shape == (batch, target_dim)
    out = torch.cat([mean, log_sigma], dim=-1)
    assert out.shape == (batch, 2 * target_dim)
    assert torch.isfinite(out).all()


def test_probe_has_no_channel_or_step_mean_pool():
    """The module source must not collapse the channel or time axis by a mean.

    The whole point is to preserve per-sensor + temporal structure: the encoder
    attends over the flattened (sensor x time) set and aggregates with a learned
    query token, never a mean over channels or steps.
    """
    src = inspect.getsource(DiagnosticsEquilibriumProbe)
    # no mean over the channel axis (dim=2) nor the time/step axis (dim=1).
    assert "mean(dim=2)" not in src
    assert "mean(dim=1)" not in src
    # aggregation is the query token's encoded state, not a pooled mean.
    assert "x[:, 0]" in src


def test_probe_reaches_full_sensor_time_sequence():
    """The encoder input length == query + machine + sum(channels*steps)."""
    n_steps, batch = 12, 2
    specs = [
        StreamSpec(name="a", vocab=16, channels=5),
        StreamSpec(name="b", vocab=16, channels=7),
    ]
    cfg = DiagnosticsProbeConfig(
        streams=specs,
        n_steps=n_steps,
        target_dim=12,
        d_model=32,
        n_heads=4,
        n_layers=1,
        use_machine_tokens=True,
    )
    model = DiagnosticsEquilibriumProbe(cfg)
    signals, geometry, kinds, _v, machine = _synthetic_inputs(
        specs, n_steps, 12, batch=batch
    )
    captured = {}
    orig = model.encoder.forward

    def _spy(x, *a, **k):
        captured["len"] = x.shape[1]
        return orig(x, *a, **k)

    model.encoder.forward = _spy
    model(signals, geometry, kinds, machine=machine)
    n_sensor_time = sum(s.channels for s in specs) * n_steps
    # query (1) + machine (6) + sensor*time tokens — all axes preserved.
    assert captured["len"] == 1 + machine.shape[1] + n_sensor_time


def test_probe_continuous_value_ablation_runs():
    """The continuous-value (non-quantised) ablation path runs + backprops."""
    n_steps, target_dim, batch = 8, 12, 3
    specs = [StreamSpec(name="magnetics", vocab=257, channels=30)]
    cfg = DiagnosticsProbeConfig(
        streams=specs,
        n_steps=n_steps,
        target_dim=target_dim,
        d_model=48,
        n_layers=2,
        continuous_value=True,
    )
    model = DiagnosticsEquilibriumProbe(cfg)
    signals, geometry, kinds, values, machine = _synthetic_inputs(
        specs, n_steps, target_dim, batch=batch, continuous=True
    )
    mean, log_sigma = model(signals, geometry, kinds, values=values, machine=machine)
    assert mean.shape == (batch, target_dim)
    loss = (mean**2 + log_sigma**2).mean()
    loss.backward()
    # the continuous value projection received gradient (the value lane is live).
    vp = model.tokenisers["magnetics"].value_proj
    assert vp.weight.grad is not None
    assert torch.isfinite(vp.weight.grad).all()


def test_sensor_kind_index():
    assert sensor_kind_index("bpol_probe") == 1
    assert sensor_kind_index("flux_loop") == 2
    assert sensor_kind_index("coil") == 6
    assert sensor_kind_index("scalar") == 7
    assert sensor_kind_index("global_scalar") == 8
    assert sensor_kind_index("something_unknown") == 0


# ---------------------------------------------------------------------------
# Name -> geometry mapping (needs the L2 magnetics store on /work)
# ---------------------------------------------------------------------------


def _have_l2_magnetics(shot_id: int) -> bool:
    from imas_ambix.data.paths import LEVEL2_DIR

    return (LEVEL2_DIR / f"{shot_id}.zarr").exists()


@pytest.mark.skipif(
    not _have_l2_magnetics(11766), reason="L2 magnetics store not reachable"
)
def test_staged_magnetics_names_resolve_to_finite_coords():
    """Staged magnetics column names resolve to finite R/Z via the L2 IDS.

    Cross-checks ccbv01 / obr01 against the gs.geometry BProbe positions and the
    orientation normal per probe family.
    """
    from imas_ambix.tokenizer.geometry_reader import magnetics_geometry_for_channels
    from imas_ambix.worldmodel.spacetime_dataset_v2 import _read_staged_raw

    sid = 11766
    _raw, names, _time = _read_staged_raw("magnetics", sid, profile_r_stride=1)
    ag = magnetics_geometry_for_channels(names, sid)

    assert ag.n_channels == len(names)
    finite_r = np.isfinite(ag.features[:, 0])
    # every named B-probe / flux-loop channel resolves; only the geometry-free
    # scalar(s) (ip) carry NaN coords.
    n_no_geom = sum(1 for k in ag.sensor_kinds if k in ("scalar", "global_scalar"))
    assert int(finite_r.sum()) == len(names) - n_no_geom
    assert int(finite_r.sum()) >= len(names) - 2  # at most a couple of scalars

    idx = {n: i for i, n in enumerate(names)}

    # ccbv[0] is a VERTICAL probe -> normal (0, 1); angle 90.
    i = idx["b_field_pol_probe_ccbv_field[0]"]
    assert ag.sensor_kinds[i] == "bpol_probe"
    assert ag.features[i, 4] == pytest.approx(0.0)  # normal_r
    assert ag.features[i, 5] == pytest.approx(1.0)  # normal_z

    # obr[0] is a RADIAL probe -> normal (1, 0); angle 0.
    i = idx["b_field_pol_probe_obr_field[0]"]
    assert ag.sensor_kinds[i] == "bpol_probe"
    assert ag.features[i, 4] == pytest.approx(1.0)
    assert ag.features[i, 5] == pytest.approx(0.0)

    # flux_loop[0] resolves to a finite point sensor (no orientation).
    i = idx["flux_loop_flux[0]"]
    assert ag.sensor_kinds[i] == "flux_loop"
    assert np.isfinite(ag.features[i, 0]) and np.isfinite(ag.features[i, 1])

    # ip is the device-global plasma current: a SIGNED scalar with no geometry,
    # tagged with the dedicated global-scalar kind so the head gives it a slot.
    i = idx["ip"]
    assert ag.sensor_kinds[i] == "global_scalar"
    assert not np.isfinite(ag.features[i, 0])

    # cross-check ccbv01 against the gs.geometry BProbe position (within mm).
    from imas_ambix.gs.geometry import build_table_for_shot

    table = build_table_for_shot(sid)
    by_name = {m.amb_channel.lower(): m for m in table.sensor_map}
    if "ccbv01" in by_name:
        gm = by_name["ccbv01"]
        i = idx["b_field_pol_probe_ccbv_field[0]"]
        assert ag.features[i, 0] == pytest.approx(gm.r, abs=0.01)
        assert ag.features[i, 1] == pytest.approx(gm.z, abs=0.01)


@pytest.mark.skipif(
    not _have_l2_magnetics(11766), reason="L2 pf_active store not reachable"
)
def test_pf_active_coil_names_resolve_to_coil_geometry():
    """PF-active coil-current channel names resolve to coil-centroid R/Z.

    Each ``AMC_PXX FEED CURRENT`` maps to its circuit's filament centroid; the
    P-coils sit at increasing major radius (P2 inner-most, P5 outer-most), and
    every resolved channel is kinded ``coil`` with finite R/Z.
    """
    import zarr

    from imas_ambix.tokenizer.geometry_reader import pf_active_geometry_for_channels

    sid = 11766
    grp = zarr.open_group(
        f"/work/projects/imas_gpu/mast/level2/shots/{sid}.zarr", mode="r"
    )["pf_active"]
    cc = [str(x) for x in np.asarray(grp["current_channel"]).reshape(-1)]
    ag = pf_active_geometry_for_channels(cc, sid)

    assert ag.n_channels == len(cc)
    by_name = {n: i for i, n in enumerate(cc)}
    # every named coil current resolves to a finite coil position, kind=coil.
    for n, i in by_name.items():
        assert ag.sensor_kinds[i] == "coil", n
        assert np.isfinite(ag.features[i, 0]) and np.isfinite(ag.features[i, 1]), n
    # P2 inner sits at a smaller major radius than P5.
    p2 = next(i for n, i in by_name.items() if "P2IL" in n)
    p5 = next(i for n, i in by_name.items() if "P5" in n)
    assert ag.features[p2, 0] < ag.features[p5, 0]
