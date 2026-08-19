"""Tests for the camdyn training loop and reconstruction evaluation.

The heavy end-to-end test runs a 3-step CPU smoke against the synthetic
corpus fixture (a tiny model, a tiny split) so it catches shape/wiring
bugs without any GPU.  Lighter tests cover config (de)serialisation and
the batch-assembly path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.camdyn.dataset import discover_token_shots
from imas_ambix.camdyn.masking import ClipMaskConfig
from imas_ambix.camdyn.model import N_COND_CHANNELS, CamdynConfig
from imas_ambix.camdyn.splits import CamdynSplit
from imas_ambix.camdyn.train import (
    TrainConfig,
    Trainer,
    _agg,
    _assemble_batch,
    _bootstrap_sanity,
)

# ---------------------------------------------------------------------------
# TrainConfig (de)serialisation
# ---------------------------------------------------------------------------


def test_train_config_roundtrip():
    cfg = TrainConfig(
        model=CamdynConfig(temporal_attention=True, dim=64),
        batch_size=8,
        max_steps=10,
        conditioning_variant="shuffled",
    )
    d = cfg.to_dict()
    cfg2 = TrainConfig.from_dict(d)
    assert cfg2.model.temporal_attention is True
    assert cfg2.model.dim == 64
    assert cfg2.batch_size == 8
    assert cfg2.betas == (0.9, 0.95)
    assert cfg2.conditioning_variant == "shuffled"


def test_train_config_loads_shipped_yaml():
    yaml = pytest.importorskip("yaml")  # noqa: F841
    cfgdir = Path(__file__).resolve().parents[2] / "imas_ambix" / "camdyn" / "configs"
    base = TrainConfig.load(cfgdir / "baseline_w1_v0.yaml")
    assert base.model.temporal_attention is False
    assert base.run_name == "baseline_w1_v0"
    smoke = TrainConfig.load(cfgdir / "smoke_cpu.yaml")
    assert smoke.device == "cpu"
    assert smoke.max_steps == 3


def test_baseline_config_is_matched_to_a_temporal_arm():
    """The per-frame config differs from a temporal arm only in the toggle."""
    yaml = pytest.importorskip("yaml")  # noqa: F841
    cfgdir = Path(__file__).resolve().parents[2] / "imas_ambix" / "camdyn" / "configs"
    per_frame = TrainConfig.load(cfgdir / "baseline_w1_v0.yaml").model
    temporal = CamdynConfig.from_dict(
        {**per_frame.to_dict(), "temporal_attention": True}
    )
    a, b = per_frame.to_dict(), temporal.to_dict()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"temporal_attention"}


def test_headline_arms_matched_except_toggle_and_run_name():
    """The headline arms differ only in the model toggle and run name.

    The parsed full configuration isolates the value of temporal attention.
    Watchdog and evaluation controls must remain identical in both arms.
    """
    yaml = pytest.importorskip("yaml")  # noqa: F841
    cfgdir = Path(__file__).resolve().parents[2] / "imas_ambix" / "camdyn" / "configs"
    base = TrainConfig.load(cfgdir / "cap_v1_baseline.yaml").to_dict()
    dyn = TrainConfig.load(cfgdir / "cap_v1_dynamics.yaml").to_dict()

    # top-level run knobs (everything except the nested model block + run_name)
    top_diff = {
        k for k in base if k != "model" and k != "run_name" and base[k] != dyn[k]
    }
    assert top_diff == set(), f"unexpected top-level config diff: {top_diff}"
    assert base["run_name"] == "cap_v1_baseline"
    assert dyn["run_name"] == "cap_v1_dynamics"

    # nested model block differs ONLY in temporal_attention
    model_diff = {k for k in base["model"] if base["model"][k] != dyn["model"][k]}
    assert model_diff == {"temporal_attention"}, model_diff
    assert base["model"]["temporal_attention"] is False
    assert dyn["model"]["temporal_attention"] is True


def test_structure_loss_arms_matched_except_toggle_and_run_name():
    """The structure-loss arms differ only in toggle and run name.

    Matched-arm contract: both arms get the SAME structure loss + λ, so the
    comparison isolates exactly the value of temporal attention.  The new
    spectral_aux configs add structure_loss_weight (shared, > 0) but it MUST
    be identical in both arms.
    """
    yaml = pytest.importorskip("yaml")  # noqa: F841
    cfgdir = Path(__file__).resolve().parents[2] / "imas_ambix" / "camdyn" / "configs"
    base = TrainConfig.load(cfgdir / "spectral_aux_baseline.yaml").to_dict()
    dyn = TrainConfig.load(cfgdir / "spectral_aux_dynamics.yaml").to_dict()

    # top-level run knobs (everything except the nested model block + run_name)
    top_diff = {
        k for k in base if k != "model" and k != "run_name" and base[k] != dyn[k]
    }
    assert top_diff == set(), f"unexpected top-level config diff: {top_diff}"
    assert base["run_name"] == "spectral_aux_baseline"
    assert dyn["run_name"] == "spectral_aux_dynamics"

    # nested model block differs ONLY in temporal_attention
    model_diff = {k for k in base["model"] if base["model"][k] != dyn["model"][k]}
    assert model_diff == {"temporal_attention"}, model_diff
    assert base["model"]["temporal_attention"] is False
    assert dyn["model"]["temporal_attention"] is True
    # both arms carry the SAME positive structure-loss weight (the new objective)
    assert base["model"]["structure_loss_weight"] > 0.0
    assert (
        base["model"]["structure_loss_weight"] == dyn["model"]["structure_loss_weight"]
    )


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def test_assemble_batch_shapes_and_mask_complement(synthetic_corpus):
    sc = synthetic_corpus
    specs = discover_token_shots(
        token_root=sc["token_root"],
        level1_dir=sc["level1_dir"],
        shot_ids=sc["shot_ids"],
        read_n_frames=True,
    )
    from imas_ambix.camdyn.dataset import FrameTokenDataset, FrameWindowConfig
    from imas_ambix.camdyn.train import _level1_for

    ds = FrameTokenDataset(specs, FrameWindowConfig(n_frames=6, stride=4), as_dict=True)
    windows = []
    for i in range(min(3, len(ds))):
        w = ds[i]
        w["level1_path"] = _level1_for(specs, int(w["shot_id"]))
        windows.append(w)
    rng = np.random.default_rng(0)
    arr = _assemble_batch(windows, ClipMaskConfig(), rng, progress=None)
    b = len(windows)
    assert arr["tokens"].shape == (b, 6, 16, 16)
    assert arr["cond_values"].shape == (b, 6, N_COND_CHANNELS)
    assert arr["cond_missing"].shape == (b, 6, N_COND_CHANNELS)
    # loss_mask must be the exact complement of visible
    np.testing.assert_array_equal(arr["loss_mask"], ~arr["visible"])


# ---------------------------------------------------------------------------
# Watchdog: paused during eval/val/ckpt, still fires on a wedged TRAIN step
# ---------------------------------------------------------------------------


def _watchdog_trainer():
    """A Trainer with a tiny watchdog grace for fast tests (no GPU/data)."""
    cfg = TrainConfig(
        model=CamdynConfig(temporal_attention=False, dim=16, n_layers=1, n_heads=2),
        watchdog_grace_s=0.2,
        val_windows=4,
        eval_windows=4,
    )
    return Trainer(cfg)


def test_watchdog_does_not_fire_during_paused_phase():
    """A slow phase wrapped in _pause_watchdog must NOT trip the watchdog.

    The pause context makes the watchdog ignore validation, final evaluation,
    and checkpoint wall-clock while preserving the training-step deadline.
    """
    import time

    from imas_ambix.camdyn.train import STOP

    STOP.clear()
    tr = _watchdog_trainer()
    median_step = [0.01]  # warm median so deadline = max(grace, 8*med) = grace
    tr._last_step_t[0] = time.time()
    tr._arm_watchdog(median_step, poll_s=0.05)
    try:
        # Simulate a slow non-training phase (much longer than the grace).
        with tr._pause_watchdog("val"):
            time.sleep(0.6)  # 3x the 0.2s grace
        # The watchdog must NOT have fired during the paused phase.
        assert not STOP.is_set(), "watchdog fired during a paused (eval) phase"
    finally:
        tr._watchdog_stop.set()  # kill THIS trainer's thread (no zombie leak)
        time.sleep(0.1)
        STOP.clear()


def test_watchdog_fires_on_wedged_training_step():
    """A genuinely stalled TRAIN step (no pause, clock not refreshed) must
    still trip the watchdog; paused phases must not disarm this guard."""
    import time

    from imas_ambix.camdyn.train import STOP

    STOP.clear()
    tr = _watchdog_trainer()
    median_step = [0.01]
    tr._last_step_t[0] = time.time()
    tr._arm_watchdog(median_step, poll_s=0.05)
    try:
        # Do NOT refresh last_step_t and do NOT pause — simulate a hung step.
        deadline = time.time() + 1.5
        while time.time() < deadline and not STOP.is_set():
            time.sleep(0.05)
        assert STOP.is_set(), "watchdog failed to fire on a wedged training step"
    finally:
        tr._watchdog_stop.set()  # kill the watchdog thread (it fired by design)
        STOP.clear()
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Aggregation / bootstrap helpers
# ---------------------------------------------------------------------------


def test_agg_handles_empty_and_nonempty():
    assert _agg(np.array([]))["n"] == 0
    a = _agg(np.array([1.0, 2.0, 3.0]))
    assert a["mean"] == pytest.approx(2.0)
    assert a["n"] == 3


def test_bootstrap_sanity_within_arm_is_not_significant():
    """Two halves of the SAME arm must not favour either side."""
    rng = np.random.default_rng(0)
    nll = rng.standard_normal(2000) + 5.0
    ci = _bootstrap_sanity(nll)
    assert ci["favours_dynamics"] is False
    assert ci["lo"] < 0.0 < ci["hi"]  # straddles zero


# ---------------------------------------------------------------------------
# End-to-end CPU smoke — the real wiring test
# ---------------------------------------------------------------------------


def _write_split(tmp_path: Path, sc) -> Path:
    """A tiny split over the synthetic shots: train/val/held_out."""
    ids = sc["shot_ids"]
    split = CamdynSplit(
        train=[ids[0]],
        val=[ids[1]],
        held_out=[ids[2]] if len(ids) > 2 else [ids[1]],
        n_token_shots=len(ids),
    )
    path = tmp_path / "smoke_split.json"
    path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    return path


def test_cpu_smoke_train_and_reconstruction_artifact(
    synthetic_corpus, tmp_path, monkeypatch
):
    """CPU training and held-out evaluation against the synthetic corpus.

    Patches discovery + the default split so the trainer runs entirely on
    the fixture's tiny synthetic stores (no real corpus, no GPU).
    """
    pytest.importorskip("torch")
    sc = synthetic_corpus
    split_path = _write_split(tmp_path, sc)

    # Route discover_token_shots at the synthetic roots.
    import imas_ambix.camdyn.train as trainmod

    real_discover = discover_token_shots

    def _patched_discover(*, shot_ids=None, read_n_frames=False, **_kw):
        return real_discover(
            token_root=sc["token_root"],
            level1_dir=sc["level1_dir"],
            shot_ids=shot_ids,
            read_n_frames=read_n_frames,
        )

    monkeypatch.setattr(trainmod, "discover_token_shots", _patched_discover)

    cfg = TrainConfig(
        model=CamdynConfig(
            temporal_attention=False,
            dim=32,
            n_layers=2,
            n_heads=4,
            mlp_ratio=2.0,
            n_frames=6,
            cond_channels=N_COND_CHANNELS,
        ),
        n_frames=6,
        stride=4,
        batch_size=2,
        max_steps=4,
        warmup_frac=0.0,
        curriculum=False,
        num_workers=0,
        val_windows=4,
        eval_windows=4,
        log_every=1,
        # Exercise both periodic validation and final evaluation.
        val_every=2,
        ckpt_every=2,
        device="cpu",
        split_path=str(split_path),
        ckpt_root=str(tmp_path / "ckpt"),
        run_name="smoke",
        artifact_out=str(tmp_path / "reconstruction.json"),
    )

    trainer = Trainer(cfg)
    from imas_ambix.camdyn.train import STOP

    STOP.clear()
    evaluation = trainer.train()

    # the watchdog must NOT have fired during the run (VAL + final eval ran
    # inside _pause_watchdog; training steps were fast)
    assert not STOP.is_set(), "watchdog fired during the CPU smoke (false positive)"

    # --- artifact written + structurally complete ---
    art = json.loads((tmp_path / "reconstruction.json").read_text())
    assert art["arm"] == "per-frame"
    assert art["temporal_attention"] is False
    assert "held_out" in art
    assert "masked_nll" in art["held_out"]
    assert "masked_top1" in art["held_out"]
    assert "motion_weighted" in art["held_out"]
    # every frozen named geometry was scored
    from imas_ambix.camdyn.masking import NAMED_GEOMETRIES

    assert set(art["named_geometry"]) == set(NAMED_GEOMETRIES)
    # bootstrap CI helper was exercised
    assert "bootstrap_ci_sanity" in art
    # a checkpoint exists on disk
    assert Path(art["checkpoint"]).exists()
    # finite scores
    assert np.isfinite(evaluation["held_out"]["masked_nll"]["mean"])
    assert 0.0 <= evaluation["held_out"]["masked_top1"]["mean"] <= 1.0


def test_cpu_smoke_train_with_structure_loss(synthetic_corpus, tmp_path, monkeypatch):
    """Full CPU train + VAL + final-eval + artifact + clean-exit with λ>0.

    A periodic validation fires, final evaluation writes its artifact, and the
    watchdog must not fire. Confirms the loss history records the BCE and
    structure components and the artifact is structurally complete.
    """
    pytest.importorskip("torch")
    sc = synthetic_corpus
    split_path = _write_split(tmp_path, sc)

    import imas_ambix.camdyn.train as trainmod

    real_discover = discover_token_shots

    def _patched_discover(*, shot_ids=None, read_n_frames=False, **_kw):
        return real_discover(
            token_root=sc["token_root"],
            level1_dir=sc["level1_dir"],
            shot_ids=shot_ids,
            read_n_frames=read_n_frames,
        )

    monkeypatch.setattr(trainmod, "discover_token_shots", _patched_discover)

    cfg = TrainConfig(
        model=CamdynConfig(
            temporal_attention=False,
            dim=32,
            n_layers=2,
            n_heads=4,
            mlp_ratio=2.0,
            n_frames=6,
            cond_channels=N_COND_CHANNELS,
            structure_loss_weight=0.05,  # λ>0 — the new objective active
        ),
        n_frames=6,
        stride=4,
        batch_size=2,
        max_steps=4,
        warmup_frac=0.0,
        curriculum=False,
        num_workers=0,
        val_windows=4,
        eval_windows=4,
        log_every=1,
        val_every=2,
        ckpt_every=2,
        device="cpu",
        split_path=str(split_path),
        ckpt_root=str(tmp_path / "ckpt"),
        run_name="smoke_structure",
        artifact_out=str(tmp_path / "w1_struct.json"),
    )

    trainer = Trainer(cfg)
    from imas_ambix.camdyn.train import STOP

    STOP.clear()
    evaluation = trainer.train()

    assert not STOP.is_set(), "watchdog fired during the structure-loss smoke"

    art = json.loads((tmp_path / "w1_struct.json").read_text())
    assert art["arm"] == "per-frame"
    assert art["model_config"]["structure_loss_weight"] == pytest.approx(0.05)
    assert "held_out" in art and "masked_nll" in art["held_out"]
    # the loss history logged both the BCE and structure components
    train_rows = [h for h in art["loss_history"] if "train_loss" in h]
    assert train_rows, "no training-loss rows recorded"
    row = train_rows[0]
    assert "train_bce" in row and "train_struct" in row
    assert np.isfinite(row["train_struct"]) and row["train_struct"] >= 0.0
    assert Path(art["checkpoint"]).exists()
    assert np.isfinite(evaluation["held_out"]["masked_nll"]["mean"])


def test_arm_compare_paired_verdict(synthetic_corpus, tmp_path, monkeypatch):
    """Cross-arm paired comparison runs end-to-end on tiny ckpts (CPU).

    Builds matched baseline (temporal OFF) + dynamics (temporal ON) ckpts in
    the on-disk format ``arm_compare._load_arm`` expects, scores BOTH (and the
    carry-forward reference) on the SAME synthetic held-out windows, and asserts
    the paired verdict structure: dynamics-vs-baseline and dynamics-vs-ZOH CIs
    with aligned pair counts, a boolean ``dynamics_wins``, and every frozen
    named geometry compared.
    """
    torch = pytest.importorskip("torch")
    from imas_ambix.camdyn import arm_compare as ac
    from imas_ambix.camdyn.masking import NAMED_GEOMETRIES
    from imas_ambix.camdyn.model import CamdynModel

    sc = synthetic_corpus
    split_path = _write_split(tmp_path, sc)

    import imas_ambix.camdyn.train as trainmod

    real_discover = discover_token_shots

    def _patched_discover(*, shot_ids=None, read_n_frames=False, **_kw):
        return real_discover(
            token_root=sc["token_root"],
            level1_dir=sc["level1_dir"],
            shot_ids=shot_ids,
            read_n_frames=read_n_frames,
        )

    monkeypatch.setattr(trainmod, "discover_token_shots", _patched_discover)

    def _cfg(temporal):
        return TrainConfig(
            model=CamdynConfig(
                temporal_attention=temporal,
                dim=32,
                n_layers=2,
                n_heads=4,
                mlp_ratio=2.0,
                n_frames=6,
                cond_channels=N_COND_CHANNELS,
            ),
            n_frames=6,
            stride=4,
            batch_size=2,
            num_workers=0,
            eval_windows=4,
            max_heldout_shots=None,
            seed=0,
            split_path=str(split_path),
            device="cpu",
        )

    cond_stats = [[0.0] * N_COND_CHANNELS, [1.0] * N_COND_CHANNELS]
    ckpts = {}
    for arm, temporal in (("baseline", False), ("dynamics", True)):
        cfg = _cfg(temporal)
        model = CamdynModel.from_config(cfg.model)
        p = tmp_path / f"{arm}.pt"
        torch.save(
            {
                "config": cfg.to_dict(),
                "model_state": model.module.state_dict(),
                "cond_stats": cond_stats,
            },
            p,
        )
        ckpts[arm] = p

    verdict = ac.compare_arms(
        ckpts["baseline"], ckpts["dynamics"], split_path=str(split_path), device="cpu"
    )

    # --- paired verdict structure (dynamics vs baseline AND vs carry-forward) ---
    v = verdict["verdict"]
    assert isinstance(v["dynamics_wins"], bool)
    assert isinstance(v["dynamics_beats_baseline_nll"], bool)
    assert isinstance(v["dynamics_beats_carry_forward_top1"], bool)
    ho = verdict["held_out"]
    vb = ho["dynamics_vs_baseline_nll"]
    vz = ho["dynamics_vs_carry_forward_top1"]
    assert vb["n_pairs"] > 0
    # paired arrays are aligned element-wise across baseline / dynamics / ZOH
    assert vz["n_pairs"] == vb["n_pairs"]
    assert ho["dynamics_vs_baseline_top1"]["n_pairs"] == vb["n_pairs"]
    assert "favours_dynamics" in vb and "favours_dynamics" in vz
    assert "baseline" in ho and "dynamics" in ho and "carry_forward" in ho
    # every frozen named geometry was compared (incl. the carry-forward ref)
    assert set(verdict["named_geometry"]) == set(NAMED_GEOMETRIES)
    for g in verdict["named_geometry"].values():
        assert "dynamics_vs_carry_forward_top1" in g
    assert verdict["baseline_params"] == verdict["dynamics_params"]  # matched-arm
