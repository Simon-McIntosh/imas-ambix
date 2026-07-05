"""Tests for the corpus cache + per-batch geometry binding in the amortised
patch-current encoder trainer (``scripts/train_patch_encoder.py``).

Two levers, both exercised offline (synthetic geometry, no IMAS/MAST access):

* corpus cache — a signature's example arrays + a FULLY self-contained
  ``PatchBasis`` (every constructor argument is a plain geometry-derived
  numpy array, so no IMAS re-read is needed on load) round-trip through an
  npz byte-identically; the config hash busts on any assembly-parameter
  change; a shard-assembled corpus merges deterministically with contiguous
  example ids;
* per-batch geometry binding — one shared encoder trains against TWO
  campaign signatures with different sensor counts by rebinding its geometry
  buffers before each batch, never dropping the mismatched one, with the
  per-example discrepancy-λ buffer indexed correctly across both.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import torch

import scripts.train_patch_encoder as tpe
from imas_ambix.latent.patch_basis import PatchBasis
from imas_ambix.latent.patch_encoder import (
    DiscrepancyLambda,
    PatchCurrentEncoder,
    PatchEncoderConfig,
    amortised_losses,
)
from scripts.train_patch_encoder import (
    SignatureCorpus,
    _bind_signature,
    _load_corpus_dir,
    _load_signature_npz,
    _make_batch,
    _merge_corpus_dirs,
    _save_corpus_dir,
    _save_signature_npz,
    channel_stats_for_signature,
    sensor_geometry_array,
    token_channel_stats_by_name,
)

NR, NZ, T_STEPS = 25, 33, 4


def _synthetic_table(n_probe: int, digest: str):
    """Rectangular-limiter synthetic machine with ``n_probe`` B-probes, no coils."""
    from imas_ambix.gs import geometry as gsg

    probes = [
        gsg.BProbe(
            index=i,
            r=1.35,
            z=-0.6 + (1.2 / max(n_probe - 1, 1)) * i,
            angle_deg=90.0,
            length=0.02,
        )
        for i in range(n_probe)
    ]
    sensor_map = [
        gsg.SensorMapping(f"obv{i:02d}", "b_probe", i, p.r, p.z, p.angle_deg, 0.001, "")
        for i, p in enumerate(probes)
    ]
    return gsg.GeometryTable(
        signature=gsg.SetupSignature(
            n_bprobe=n_probe, n_fluxloop=0, n_pf_filament=0, n_limiter=5, digest=digest
        ),
        shots=[1],
        b_probes=probes,
        flux_loops=[],
        pf_filaments=[],
        limiter_r=[0.35, 1.45, 1.45, 0.35, 0.35],
        limiter_z=[-0.85, -0.85, 0.85, 0.85, -0.85],
        sensor_map=sensor_map,
        passive_structures=[],
        amc_current_channels=[],
        unmatched_amb=[],
    )


def _make_signature_corpus(
    n_probe: int, digest: str, n_examples: int, *, seed: int = 0
) -> SignatureCorpus:
    table = _synthetic_table(n_probe, digest)
    basis = PatchBasis.from_table(table, nr=NR, nz=NZ, cache_dir=None)
    channels = list(basis.sensor_channels)
    sensor_geometry = sensor_geometry_array(table, channels)
    n_cells = int(basis.candidate_mask.shape[0])
    candidate_mask = np.asarray(
        basis.candidate_mask.detach().cpu().numpy(), dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    s = len(channels)
    ip = rng.uniform(1e5, 5e5, n_examples)
    return SignatureCorpus(
        key=table.signature.key,
        basis=basis,
        sensor_channels=channels,
        sensor_geometry=sensor_geometry,
        coil_centroids=np.zeros((0, 2)),
        n_cells=n_cells,
        candidate_mask=candidate_mask,
        values=rng.standard_normal((n_examples, T_STEPS, s)),
        finite=np.ones((n_examples, T_STEPS, s), dtype=bool),
        measured=rng.standard_normal((n_examples, s)),
        vacuum=np.zeros((n_examples, s)),
        mask=np.ones((n_examples, s), dtype=bool),
        scale=np.ones((n_examples, s)),
        i_pf=np.zeros((n_examples, 0)),
        ip=ip,
        ids=np.arange(n_examples, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
#  Corpus cache                                                                #
# --------------------------------------------------------------------------- #
def test_config_hash_busts_on_param_change():
    base = dict(
        shots=[18501, 18502, 18503],
        t_steps=12,
        stride_s=0.025,
        min_ip_ka=300.0,
        nr=65,
        nz=97,
    )
    h0 = tpe._config_hash(**base)
    assert h0 == tpe._config_hash(**base)  # deterministic

    for changed in (
        {"t_steps": 8},
        {"stride_s": 0.05},
        {"min_ip_ka": 200.0},
        {"nr": 33},
        {"nz": 65},
        {"shots": [18501, 18502]},
    ):
        variant = {**base, **changed}
        assert tpe._config_hash(**variant) != h0, changed

    # shot ORDER must not matter — the hash sorts the shot list
    reordered = {**base, "shots": list(reversed(base["shots"]))}
    assert tpe._config_hash(**reordered) == h0


def test_config_hash_busts_on_any_version_constant(monkeypatch):
    """A behavioural fix to this module's own assembly semantics (not just an
    upstream coil-model/geometry-table change) must ALSO bust the cache key —
    the concrete failure this pins: geometry_shots landed without bumping
    anything, so a stale shard cache was served unchanged under the new
    hash-identical config."""
    base = dict(
        shots=[1, 2, 3], t_steps=12, stride_s=0.025, min_ip_ka=300.0, nr=65, nz=97
    )
    h0 = tpe._config_hash(**base)
    for const_name in (
        "COIL_MODEL_VERSION",
        "GEOMETRY_TABLE_VERSION",
        "CORPUS_ASSEMBLY_VERSION",
    ):
        monkeypatch.setattr(tpe, const_name, "changed-for-test")
        assert tpe._config_hash(**base) != h0, const_name
        monkeypatch.undo()


def test_signature_npz_roundtrip_byte_identical(tmp_path):
    corp = _make_signature_corpus(6, "feed1001", n_examples=5, seed=3)
    path = tmp_path / f"{corp.key}.npz"
    _save_signature_npz(
        path,
        corp,
        shots=[101, 102],
        t_steps=T_STEPS,
        stride_s=0.025,
        min_ip_ka=300.0,
        nr=NR,
        nz=NZ,
        config_hash="deadbeef",
    )
    loaded = _load_signature_npz(path)

    assert loaded.key == corp.key
    assert loaded.sensor_channels == corp.sensor_channels
    assert loaded.n_cells == corp.n_cells
    for field in (
        "sensor_geometry",
        "coil_centroids",
        "candidate_mask",
        "values",
        "finite",
        "measured",
        "vacuum",
        "mask",
        "scale",
        "i_pf",
        "ip",
        "ids",
    ):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(corp, field))

    # the reconstructed PatchBasis is numerically identical (fp32<->fp64<->fp32
    # round-trip through the npz is exact — every original value IS
    # fp32-representable, so the up/down-cast recovers the same bits)
    for attr in ("m_sens", "m_coil", "g_cc", "r_cells", "z_cells", "candidate_mask"):
        a = getattr(corp.basis, attr).detach().cpu().numpy()
        b = getattr(loaded.basis, attr).detach().cpu().numpy()
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(corp.basis._g_pg_np, loaded.basis._g_pg_np)


def test_save_load_corpus_dir_round_trip_multi_signature(tmp_path):
    a = _make_signature_corpus(5, "feed2001", n_examples=3, seed=1)
    b = _make_signature_corpus(7, "feed2002", n_examples=4, seed=2)
    corpora = {a.key: a, b.key: b}
    dir_path = tmp_path / "full"
    assert not tpe._corpus_dir_complete(dir_path)
    _save_corpus_dir(
        dir_path,
        corpora,
        shots=[1, 2, 3],
        t_steps=T_STEPS,
        stride_s=0.025,
        min_ip_ka=300.0,
        nr=NR,
        nz=NZ,
        config_hash="cafef00d",
    )
    assert tpe._corpus_dir_complete(dir_path)
    loaded = _load_corpus_dir(dir_path)
    assert set(loaded) == {a.key, b.key}
    assert loaded[a.key].values.shape[0] == 3
    assert loaded[b.key].values.shape[0] == 4


def test_merge_corpus_dirs_renumbers_ids_contiguously(tmp_path):
    """Two shards of the SAME signature merge with a deterministic concatenation
    order and freshly contiguous ids, regardless of what ids each shard stored."""
    shard_a = _make_signature_corpus(5, "feed3001", n_examples=2, seed=10)
    shard_a.ids = np.array([7, 8], dtype=np.int64)  # deliberately non-contiguous
    shard_b = _make_signature_corpus(5, "feed3001", n_examples=3, seed=11)
    shard_b.ids = np.array([0, 0, 0], dtype=np.int64)  # deliberately wrong

    dir_a = tmp_path / "shard_000"
    dir_b = tmp_path / "shard_001"
    _save_corpus_dir(
        dir_a,
        {shard_a.key: shard_a},
        shots=[1, 2],
        t_steps=T_STEPS,
        stride_s=0.025,
        min_ip_ka=300.0,
        nr=NR,
        nz=NZ,
        config_hash="x",
    )
    _save_corpus_dir(
        dir_b,
        {shard_b.key: shard_b},
        shots=[3, 4, 5],
        t_steps=T_STEPS,
        stride_s=0.025,
        min_ip_ka=300.0,
        nr=NR,
        nz=NZ,
        config_hash="x",
    )

    merged = _merge_corpus_dirs([dir_a, dir_b])
    assert set(merged) == {shard_a.key}
    m = merged[shard_a.key]
    assert m.values.shape[0] == 5
    np.testing.assert_array_equal(m.ids, np.arange(5))
    # shard order preserved: rows 0-1 from shard_a, rows 2-4 from shard_b
    np.testing.assert_array_equal(m.values[:2], shard_a.values)
    np.testing.assert_array_equal(m.values[2:], shard_b.values)


def test_merge_corpus_dirs_reconstructs_basis_once_per_signature(tmp_path):
    """MEMORY REGRESSION: a full PatchBasis (dominated by the O(grid x cells)
    g_pg matrix) must be reconstructed ONCE per signature, not once per shard
    file — reconstructing it per shard is what OOM'd the merge of a real
    32-shard corpus.  Pin this via a call-count on the full loader; every
    shard beyond the first for a signature must go through the light loader
    only (no basis_* keys touched)."""
    n_shards = 6
    shard_dirs = []
    for i in range(n_shards):
        corp = _make_signature_corpus(5, "feed7001", n_examples=2, seed=100 + i)
        d = tmp_path / f"shard_{i:03d}"
        _save_corpus_dir(
            d,
            {corp.key: corp},
            shots=[i],
            t_steps=T_STEPS,
            stride_s=0.025,
            min_ip_ka=300.0,
            nr=NR,
            nz=NZ,
            config_hash="x",
        )
        shard_dirs.append(d)

    with mock.patch(
        "scripts.train_patch_encoder._load_signature_npz",
        wraps=tpe._load_signature_npz,
    ) as spy_full:
        merged = tpe._merge_corpus_dirs(shard_dirs)
    assert spy_full.call_count == 1  # one signature -> exactly one full load
    assert merged[next(iter(merged))].values.shape[0] == 2 * n_shards


def _fake_assemble_corpus(base_corp, calls):
    def _fn(
        shots,
        *,
        nr,
        nz,
        t_steps,
        stride_s,
        min_ip_ka,
        max_populated_shots=None,
        operator_out=None,
        geometry_shots=None,
    ):
        calls.append(list(shots))
        n = len(shots)
        s = len(base_corp.sensor_channels)
        rng = np.random.default_rng(sum(int(x) for x in shots) + 1)
        corp = SignatureCorpus(
            key=base_corp.key,
            basis=base_corp.basis,
            sensor_channels=base_corp.sensor_channels,
            sensor_geometry=base_corp.sensor_geometry,
            coil_centroids=base_corp.coil_centroids,
            n_cells=base_corp.n_cells,
            candidate_mask=base_corp.candidate_mask,
            values=rng.standard_normal((n, T_STEPS, s)),
            finite=np.ones((n, T_STEPS, s), dtype=bool),
            measured=rng.standard_normal((n, s)),
            vacuum=np.zeros((n, s)),
            mask=np.ones((n, s), dtype=bool),
            scale=np.ones((n, s)),
            i_pf=np.zeros((n, 0)),
            ip=rng.uniform(1e5, 5e5, n),
            ids=np.arange(n, dtype=np.int64),
        )
        return {corp.key: corp}

    return _fn


def test_assemble_corpus_cached_hits_after_shard_merge(tmp_path):
    base = _make_signature_corpus(5, "feed4001", n_examples=1)
    calls: list[list[int]] = []
    shots = [101, 102, 103, 104]
    kwargs = dict(nr=NR, nz=NZ, t_steps=T_STEPS, stride_s=0.025, min_ip_ka=300.0)

    with mock.patch(
        "scripts.train_patch_encoder.assemble_corpus",
        side_effect=_fake_assemble_corpus(base, calls),
    ):
        tpe.assemble_corpus_cached(shots, cache_root=tmp_path, shard=(0, 2), **kwargs)
        tpe.assemble_corpus_cached(shots, cache_root=tmp_path, shard=(1, 2), **kwargs)
        assert len(calls) == 2

        # re-requesting a shard that is already complete does NOT reassemble
        tpe.assemble_corpus_cached(shots, cache_root=tmp_path, shard=(0, 2), **kwargs)
        assert len(calls) == 2

        merged = tpe.assemble_corpus_cached(shots, cache_root=tmp_path, **kwargs)
        assert len(calls) == 2  # the merge path never calls assemble_corpus
        assert merged[base.key].values.shape[0] == 4
        np.testing.assert_array_equal(merged[base.key].ids, np.arange(4))

        # the full cache is itself now cached — a second full request hits it
        merged2 = tpe.assemble_corpus_cached(shots, cache_root=tmp_path, **kwargs)
        assert len(calls) == 2
        np.testing.assert_array_equal(merged2[base.key].values, merged[base.key].values)


def test_assemble_corpus_cached_shard_branch_passes_full_shot_list_as_geometry_shots(
    tmp_path,
):
    """A rare channel present on only SOME shots of a signature can land in
    one shard's own slice and not another's — resolving canonical geometry
    from a shard's local slice alone can therefore still disagree ACROSS
    shards.  assemble_corpus_cached's shard branch must pass the FULL,
    un-sliced shot list through as geometry_shots so every shard resolves the
    identical canonical schema regardless of which shots landed in it."""
    shots = [101, 102, 103, 104]
    seen_geometry_shots = []

    def _fake(shot_list, **kwargs):
        seen_geometry_shots.append(kwargs.get("geometry_shots"))
        return {}

    with mock.patch("scripts.train_patch_encoder.assemble_corpus", side_effect=_fake):
        tpe.assemble_corpus_cached(
            shots,
            nr=NR,
            nz=NZ,
            t_steps=T_STEPS,
            stride_s=0.025,
            min_ip_ka=300.0,
            cache_root=tmp_path,
            shard=(0, 2),
        )
    assert seen_geometry_shots == [shots]  # the FULL list, not shots[0::2]


def test_assemble_corpus_resolves_geometry_over_geometry_shots_not_shots():
    """assemble_corpus itself must hand geometry_shots (when given) to
    discovery/canonical-schema resolution — shots stays the narrower slice
    actually assembled into examples."""
    narrow = [1, 2]
    wide = [1, 2, 3, 4, 5]
    seen: dict[str, list[int]] = {}

    def fake_discover(shot_ids):
        seen["discover"] = list(shot_ids)
        return {}

    with mock.patch(
        "scripts.train_patch_encoder.discover_signatures", side_effect=fake_discover
    ):
        corpora = tpe.assemble_corpus(
            narrow,
            nr=NR,
            nz=NZ,
            t_steps=T_STEPS,
            stride_s=0.025,
            min_ip_ka=300.0,
            geometry_shots=wide,
        )
    assert seen["discover"] == wide
    assert corpora == {}  # no signatures discovered -> nothing to assemble


# --------------------------------------------------------------------------- #
#  Per-batch geometry binding                                                  #
# --------------------------------------------------------------------------- #
def test_token_channel_stats_by_name_and_kind_fallback():
    a = _make_signature_corpus(5, "feed5001", n_examples=20, seed=1)
    b = _make_signature_corpus(7, "feed5002", n_examples=20, seed=2)
    # rename b's channels so exactly one name overlaps with a and one is unique
    shared = a.sensor_channels[0]
    b.sensor_channels = [shared] + [
        f"unique_{i}" for i in range(len(b.sensor_channels) - 1)
    ]
    corpora = {a.key: a, b.key: b}

    stats = token_channel_stats_by_name(corpora)
    # every channel name from both signatures is present
    assert set(stats) == set(a.sensor_channels) | set(b.sensor_channels)

    # the shared channel pools observations from BOTH signatures — not equal to
    # either signature's own column-position stat alone
    idx_a = a.sensor_channels.index(shared)
    col_a = a.values[:, :, idx_a][a.finite[:, :, idx_a]]
    idx_b = 0
    col_b = b.values[:, :, idx_b][b.finite[:, :, idx_b]]
    pooled_mean = np.concatenate([col_a, col_b]).mean()
    assert abs(stats[shared][0] - pooled_mean) < 1e-8

    # per-signature lookups align BY NAME, not by position
    mean_a, std_a = channel_stats_for_signature(a.sensor_channels, stats)
    mean_b, std_b = channel_stats_for_signature(b.sensor_channels, stats)
    assert mean_a[idx_a] == stats[shared][0]
    assert mean_b[idx_b] == stats[shared][0]


def test_bind_signature_shares_trunk_across_two_signatures():
    """One shared encoder trains against two DIFFERENT sensor-count signatures:
    both contribute gradient to the shared trunk, and the per-example λ buffer
    (sized over the combined corpus) indexes correctly for both."""
    torch.manual_seed(0)
    corp_a = _make_signature_corpus(5, "feed6001", n_examples=6, seed=1)
    corp_b = _make_signature_corpus(9, "feed6002", n_examples=4, seed=2)
    assert corp_a.n_cells == corp_b.n_cells  # same limiter+grid -> same substrate
    assert len(corp_a.sensor_channels) != len(corp_b.sensor_channels)  # the point

    # global contiguous ids across both signatures (mirrors main()'s renumbering)
    corp_a.ids = np.arange(0, 6, dtype=np.int64)
    corp_b.ids = np.arange(6, 10, dtype=np.int64)
    corpora = {corp_a.key: corp_a, corp_b.key: corp_b}

    stats = token_channel_stats_by_name(corpora)
    stats_a = channel_stats_for_signature(corp_a.sensor_channels, stats)
    stats_b = channel_stats_for_signature(corp_b.sensor_channels, stats)

    ipf_mean = np.zeros(0)
    ipf_std = np.ones(0)

    cfg = PatchEncoderConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        n_time=T_STEPS,
    )
    encoder = PatchCurrentEncoder(
        cfg,
        sensor_geometry=corp_a.sensor_geometry,
        coil_centroids=corp_a.coil_centroids,
        candidate_mask=corp_a.candidate_mask,
    )
    opt = torch.optim.SGD(encoder.parameters(), lr=1e-3)
    disc = DiscrepancyLambda(10, warmup_epochs=0, lam0=1.0)

    device = "cpu"
    grad_norms = {}
    for key, corp, (ch_mean, ch_std) in (
        (corp_a.key, corp_a, stats_a),
        (corp_b.key, corp_b, stats_b),
    ):
        _bind_signature(encoder, corp, device)
        assert encoder.n_sensor == len(corp.sensor_channels)
        rows = np.arange(corp.values.shape[0])
        enc_in, payload = _make_batch(
            corp, rows, ch_mean, ch_std, ipf_mean, ipf_std, device
        )
        ids = torch.as_tensor(corp.ids[rows])
        lam = disc.get(ids)
        opt.zero_grad()
        i_cell = encoder(
            enc_in["values"], enc_in["finite"], enc_in["i_pf_std"], enc_in["ip"]
        )
        assert i_cell.shape == (len(rows), corp.n_cells)
        losses = amortised_losses(corp.basis, i_cell, lam=lam, **payload)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        assert encoder.value_proj.weight.grad is not None
        grad_norms[key] = float(encoder.value_proj.weight.grad.norm())
        opt.step()

    # both signatures actually produced a nonzero gradient on the shared trunk
    assert all(g > 0 for g in grad_norms.values())

    # the λ buffer, sized over the FULL 10-example corpus, indexes each
    # signature's own id range with no overlap and no out-of-range access
    lam_a = disc.get(torch.as_tensor(corp_a.ids))
    lam_b = disc.get(torch.as_tensor(corp_b.ids))
    assert lam_a.shape == (6,)
    assert lam_b.shape == (4,)


# --------------------------------------------------------------------------- #
#  Balanced sampling (signature-balanced / regime-balanced)                    #
# --------------------------------------------------------------------------- #
def test_ip_regime_buckets_splits_into_roughly_equal_terciles():
    rng = np.random.default_rng(0)
    ip = rng.uniform(1e5, 1e6, 3000)
    buckets = tpe._ip_regime_buckets(ip, n_buckets=3)
    assert set(np.unique(buckets)) == {0, 1, 2}
    counts = np.bincount(buckets, minlength=3)
    # terciles of a uniform sample land close to 1000 each
    assert all(900 < c < 1100 for c in counts)
    # bucket index tracks the Ip ORDERING (0 = lowest Ip tercile)
    assert ip[buckets == 0].mean() < ip[buckets == 1].mean() < ip[buckets == 2].mean()


def test_ip_regime_buckets_degenerate_constant_ip_falls_back_gracefully():
    """All-identical Ip (a degenerate/tiny corpus) must not crash — every row
    collapses into one bucket rather than raising."""
    ip = np.full(20, 5e5)
    buckets = tpe._ip_regime_buckets(ip, n_buckets=3)
    assert buckets.shape == (20,)
    assert np.all(np.isfinite(buckets))


def test_epoch_batches_balanced_gives_equal_step_budget_per_signature():
    """signature-balanced: every signature gets steps_per_epoch // n_sig
    batches per epoch, REGARDLESS of its own example count — the fix for
    natural sampling's batch-count-proportional-to-corpus-size behaviour."""
    rng = np.random.default_rng(0)
    corp_small = _make_signature_corpus(5, "feed8001", n_examples=8, seed=1)
    corp_big = _make_signature_corpus(5, "feed8002", n_examples=400, seed=2)
    corpora = {corp_small.key: corp_small, corp_big.key: corp_big}

    batches = tpe._epoch_batches_balanced(
        corpora, batch_size=16, rng=rng, steps_per_epoch=40
    )
    from collections import Counter

    counts = Counter(key for key, _rows in batches)
    assert counts[corp_small.key] == counts[corp_big.key] == 20  # 40 // 2 sigs
    # the small signature's batches draw WITH replacement (its 8 examples
    # can't cover 20 batches of 16 rows without-replacement)
    for key, rows in batches:
        if key == corp_small.key:
            assert rows.max() < corp_small.values.shape[0]
            assert len(rows) == 16


def test_epoch_batches_balanced_regime_stratifies_within_signature():
    """regime-balanced: each batch draws from EVERY Ip tercile of its own
    signature, not just whichever tercile dominates that signature's corpus."""
    seed_rng = np.random.default_rng(9)
    rng = np.random.default_rng(0)
    corp = _make_signature_corpus(5, "feed8003", n_examples=300, seed=3)
    # skewed but CONTINUOUS Ip distribution: 90% low-Ip, 10% high-Ip (mimics a
    # signature dominated by one regime, as fc938 dominates the full corpus) —
    # real Ip values are never bit-identical, so use narrow Gaussian clusters
    # rather than exactly-repeated constants (which degenerately collapse the
    # percentile edges and defeat stratification for a reason unrelated to
    # the code under test).
    corp.ip = np.concatenate(
        [
            seed_rng.normal(3.1e5, 5e3, 270),
            seed_rng.normal(1.2e6, 5e3, 30),
        ]
    )
    corpora = {corp.key: corp}

    batches = tpe._epoch_batches_balanced(
        corpora, batch_size=30, rng=rng, steps_per_epoch=10, regime_balanced=True
    )
    buckets = tpe._ip_regime_buckets(corp.ip)
    for _key, rows in batches:
        drawn_buckets = set(buckets[rows])
        # a batch drawing only from the dominant regime would be a single
        # bucket; stratification must pull in more than one
        assert len(drawn_buckets) > 1


def test_epoch_batches_balanced_deterministic_given_seed():
    """Same rng seed -> identical batch plan (no hidden global state)."""
    corp = _make_signature_corpus(5, "feed8004", n_examples=50, seed=4)
    corpora = {corp.key: corp}
    b1 = tpe._epoch_batches_balanced(
        corpora, batch_size=8, rng=np.random.default_rng(7), steps_per_epoch=6
    )
    b2 = tpe._epoch_batches_balanced(
        corpora, batch_size=8, rng=np.random.default_rng(7), steps_per_epoch=6
    )
    for (k1, r1), (k2, r2) in zip(b1, b2, strict=True):
        assert k1 == k2
        np.testing.assert_array_equal(r1, r2)
