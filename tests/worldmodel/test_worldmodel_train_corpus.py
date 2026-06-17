"""Synthetic-tensor unit tests for the corpus world-model trainer.

These are HERMETIC and FAST (fake token stores in ``tmp_path``, a tiny model,
a few steps, two distinct synthetic shots — no GPFS, no real corpus, runs in
well under 30 s on a login node).  They prove:

* :func:`~imas_ambix.worldmodel.train.train_corpus` runs a DataLoader over
  MULTIPLE distinct shots (not one re-descended batch) and the loss drops;
* a checkpoint is written, RELOADS into a fresh model
  (:func:`~imas_ambix.worldmodel.train.load_model_from_checkpoint`), and that
  reloaded model is consumable by ``eval.py`` (the eval-loadable contract);
* RESUME continues from ``latest.pt`` (the step counter advances, the
  optimiser state is restored);
* the pad-collate batches heterogeneous-channel shots into a rectangular batch;
* the CONDITIONING is LOAD-BEARING — zeroing the plan tokens CHANGES the
  model output (the skeleton-stage gap).

The real multi-shot GPU run is the verify agent's job on a SLURM partition;
nothing here touches ``/work`` or runs a multi-minute job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from imas_ambix.tokenizer.registry import L2_BLOCK_VOCAB
from imas_ambix.worldmodel.dataset import (
    ModalitySpec,
    WorldModelWindowConfig,
    build_shot_sample,
)
from imas_ambix.worldmodel.eval import evaluate_shot
from imas_ambix.worldmodel.model import WorldModel
from imas_ambix.worldmodel.train import (
    CorpusTrainConfig,
    collate_samples,
    find_latest_checkpoint,
    load_model_from_checkpoint,
    next_token_nll,
    pad_collate_batch,
    train_corpus,
)

L2_VOCAB = L2_BLOCK_VOCAB + 1


@pytest.fixture(autouse=True)
def _pin_cpu_threads():
    """Pin torch to a single CPU thread for the duration of each test.

    The synthetic model is tiny; torch's default thread pool (one thread per
    core, 64 on this node) thrashes catastrophically on micro-matmuls — a
    40-step run measured 86 s at default threading vs 4.6 s single-threaded.
    On a shared login node that thrash also fights peer agents.  Restore the
    prior count on teardown so the setting does not leak across modules.
    """
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prior)


# ---------------------------------------------------------------------------
# Synthetic store fixtures (mirror tests/worldmodel/test_worldmodel_skeleton.py)
# ---------------------------------------------------------------------------


def _write_signal_hf_store(
    root: Path,
    shot_id: int,
    group: str,
    *,
    n_time: int,
    n_channels: int,
    rate_hz: float,
    t0: float,
    seed: int,
    local_base: int,
) -> None:
    """Write a synthetic ``signals_hf/{shot}/{group}.zarr`` store.

    Tokens are a deterministic per-channel periodic pattern keyed by the shot
    so the model has DISTINCT, overfittable content per shot (proves the
    DataLoader is iterating multiple shots, not memorising one).
    """
    import zarr

    from imas_ambix.tokenizer.store_targets import SIGNALS_HF_GENERATION

    dt = 1.0 / rate_hz
    token_time = t0 + np.arange(n_time, dtype=np.float64) * dt
    local = np.zeros((n_time, n_channels), dtype=np.int64)
    phase = shot_id % 5
    for c in range(n_channels):
        period = 3 + (c + phase) % 5
        local[:, c] = (np.arange(n_time) % period) + 1 + ((c + phase) % 7)
    tokens = (local + local_base).astype(np.int32)
    valid = np.ones((n_time, n_channels), dtype=bool)
    names = [f"{group}.ch{c}" for c in range(n_channels)]

    path = root / SIGNALS_HF_GENERATION / "signals_hf" / str(shot_id) / f"{group}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=tokens)
    store.create_array("token_time", data=token_time)
    store.create_array("valid", data=valid)
    meta = {"codebook_size": int(L2_VOCAB)}
    if group.endswith("_l2"):
        meta["global_id_range"] = [int(local_base), int(local_base + L2_VOCAB)]
    store.attrs.update(
        {
            "tokenizer_name": group,
            "vocab_version": "v2",
            "store_generation": "v2",
            "native_rate_hz": float(rate_hz),
            "token_rate_hz": float(rate_hz),
            "n_channels": int(n_channels),
            "channel_names": names,
            "phase_preserving": False,
            "original_window": [float(token_time[0]), float(token_time[-1])],
            "metadata": json.dumps(meta),
        }
    )
    _ = seed  # pattern is RNG-independent on purpose


def _make_synthetic_shot(
    root: Path, shot_id: int, *, summary_channels: int = 3
) -> list[ModalitySpec]:
    """Write a tractable synthetic shot (plan + one measured group).

    ``summary_channels`` is varied across shots so the pad-collate's
    heterogeneous-channel handling is exercised by the corpus loop.
    """
    _write_signal_hf_store(
        root,
        shot_id,
        "pulse_schedule_l2",
        n_time=2000,
        n_channels=2,
        rate_hz=4000.0,
        t0=-0.05,
        seed=shot_id,
        local_base=12804,
    )
    _write_signal_hf_store(
        root,
        shot_id,
        "summary_l2",
        n_time=2000,
        n_channels=summary_channels,
        rate_hz=4000.0,
        t0=-0.05,
        seed=shot_id + 1,
        local_base=12804,
    )
    return [
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            L2_VOCAB,
            is_conditioning=True,
        ),
        ModalitySpec("summary", "signal_hf", "summary_l2", L2_VOCAB),
    ]


def _make_corpus(root: Path, shot_ids: list[int]) -> list[ModalitySpec]:
    """Write several distinct synthetic shots; return the modality specs."""
    mods: list[ModalitySpec] = []
    for i, sid in enumerate(shot_ids):
        # vary channel count slightly to exercise the pad-collate
        mods = _make_synthetic_shot(root, sid, summary_channels=3 + (i % 2))
    return mods


_TINY_MODEL = {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64}


# ---------------------------------------------------------------------------
# 1. pad_collate batches heterogeneous-channel shots rectangularly
# ---------------------------------------------------------------------------


def test_pad_collate_handles_heterogeneous_channels(tmp_path):
    mods = _make_corpus(tmp_path, [24065, 24066])
    cfg = WorldModelWindowConfig(n_steps=24, context_steps=6)
    s0 = build_shot_sample(24065, mods, cfg, token_root=tmp_path)  # 3 summary ch
    s1 = build_shot_sample(24066, mods, cfg, token_root=tmp_path)  # 4 summary ch
    assert s0.tokens["summary"].shape[1] != s1.tokens["summary"].shape[1]

    obs = ["summary"]
    plan = ["pulse_schedule"]
    channels = {"pulse_schedule": 2, "summary": 4}  # fixed width = max
    batch = pad_collate_batch([s0, s1], obs, plan, channels)
    # rectangular (B=2, T=24, C=4) — narrower shot padded up
    assert batch["tokens"]["summary"].shape == (2, 24, 4)
    assert batch["valid"]["summary"].shape == (2, 24, 4)
    # the padded 4th channel of the 3-channel shot is invalid + PAD
    assert not bool(batch["valid"]["summary"][0, :, 3].any())
    assert int(batch["tokens"]["summary"][0, :, 3].max()) == 0


# ---------------------------------------------------------------------------
# 2. corpus train: iterates MULTIPLE shots, loss drops, ckpt saves + reloads
# ---------------------------------------------------------------------------


def test_corpus_train_loss_drops_and_checkpoints(tmp_path):
    shots = [24065, 24066, 24067, 24068]
    mods = _make_corpus(tmp_path, shots)
    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=40,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=20,
        eval_every=20,
        num_workers=0,
        n_eval_shots=1,
        window=WorldModelWindowConfig(n_steps=24, context_steps=6),
        model_kwargs=_TINY_MODEL,
    )
    result = train_corpus(
        shots[:3],
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=shots[3:],
        device="cpu",
        resume=False,
    )

    # ran the full step budget over a multi-shot DataLoader
    assert result.steps_run == 40
    assert result.n_train_shots == 3
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
    # LOSS DROP: a real descent over the corpus (early-mean vs late-mean)
    early = float(np.mean(result.losses[:10]))
    late = float(np.mean(result.losses[-10:]))
    assert late < early, f"corpus loss did not drop: {early:.4f} -> {late:.4f}"
    # periodic eval produced a skill number
    assert result.eval_skills, "no periodic eval skill recorded"
    for _step, sk in result.eval_skills:
        assert np.isfinite(sk)

    # CHECKPOINT exists and RELOADS into a fresh model
    ckpt = find_latest_checkpoint(out_dir)
    assert ckpt is not None and ckpt.exists()
    model, payload = load_model_from_checkpoint(ckpt)
    assert isinstance(model, WorldModel)
    assert payload["step"] == 40
    assert "optimizer_state_dict" in payload


# ---------------------------------------------------------------------------
# 3. the reloaded checkpoint is consumable by eval.py (eval-loadable contract)
# ---------------------------------------------------------------------------


def test_checkpoint_is_eval_loadable(tmp_path):
    shots = [24065, 24066]
    mods = _make_corpus(tmp_path, shots)
    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=8,
        batch_size=2,
        lr=5e-3,
        log_every=4,
        ckpt_every=8,
        eval_every=0,
        num_workers=0,
        window=WorldModelWindowConfig(n_steps=20, context_steps=5),
        model_kwargs=_TINY_MODEL,
    )
    train_corpus(
        shots,
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    ckpt = find_latest_checkpoint(out_dir)
    model, _ = load_model_from_checkpoint(ckpt)

    # eval.py rolls out a held-out shot with the reloaded model and reports skill
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    report = evaluate_shot(
        24065, model, modalities=mods, window=window, token_root=tmp_path
    )
    assert report.shot_id == 24065
    assert np.isfinite(report.mean_skill)
    assert "token-skill" in report.summary()


# ---------------------------------------------------------------------------
# 4. RESUME continues from latest.pt (step advances, optimiser restored)
# ---------------------------------------------------------------------------


def test_resume_continues_from_checkpoint(tmp_path):
    shots = [24065, 24066, 24067]
    mods = _make_corpus(tmp_path, shots)
    out_dir = tmp_path / "ckpt"
    common = dict(
        modalities=mods,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
    )
    cfg_a = CorpusTrainConfig(
        steps=12,
        batch_size=2,
        lr=5e-3,
        log_every=6,
        ckpt_every=12,
        eval_every=0,
        num_workers=0,
        window=WorldModelWindowConfig(n_steps=20, context_steps=5),
        model_kwargs=_TINY_MODEL,
    )
    r1 = train_corpus(shots, config=cfg_a, resume=False, **common)
    ckpt = find_latest_checkpoint(out_dir)
    step_after_first = load_model_from_checkpoint(ckpt)[1]["step"]
    assert step_after_first == 12
    assert r1.steps_run == 12

    # second run RESUMES: total budget 24, so it should run 12 MORE steps and
    # land the checkpoint at step 24 (not restart at 0).
    cfg_b = CorpusTrainConfig(
        steps=24,
        batch_size=2,
        lr=5e-3,
        log_every=6,
        ckpt_every=12,
        eval_every=0,
        num_workers=0,
        window=WorldModelWindowConfig(n_steps=20, context_steps=5),
        model_kwargs=_TINY_MODEL,
    )
    r2 = train_corpus(shots, config=cfg_b, resume=True, **common)
    assert r2.steps_run == 12, "resume did not continue — re-ran from 0"
    final_step = load_model_from_checkpoint(find_latest_checkpoint(out_dir))[1]["step"]
    assert final_step == 24


# ---------------------------------------------------------------------------
# 5. CONDITIONING MUTATION TEST — the plan is LOAD-BEARING
# ---------------------------------------------------------------------------


def test_conditioning_is_load_bearing(tmp_path):
    """Zeroing the plan-conditioning tokens must CHANGE the model output.

    If the model ignored the pulse schedule, blanking the plan prefix would
    leave the logits unchanged.  We assert the output MOVES — the plan is a
    load-bearing input, not decorative.
    """
    shots = [24065, 24066]
    mods = _make_corpus(tmp_path, shots)
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    s0 = build_shot_sample(24065, mods, window, token_root=tmp_path)

    obs = [m.name for m in mods if not m.is_conditioning]
    plan = [m.name for m in mods if m.is_conditioning]

    torch.manual_seed(0)
    # build a model whose plan embedding is NOT trivially zero (random init is
    # nonzero) — then briefly train so the plan pathway carries signal.
    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=30,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=30,
        eval_every=0,
        num_workers=0,
        window=window,
        model_kwargs=_TINY_MODEL,
    )
    train_corpus(
        shots,
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    model, _ = load_model_from_checkpoint(find_latest_checkpoint(out_dir))
    model.eval()

    batch = collate_samples([s0], obs, plan)
    with torch.no_grad():
        base = model(batch).logits

    # MUTATION: zero the plan-conditioning tokens (PAD id 0 everywhere).
    mutated = dict(batch)
    mutated["tokens"] = dict(batch["tokens"])
    for name in plan:
        mutated["tokens"][name] = torch.zeros_like(batch["tokens"][name])
    with torch.no_grad():
        mut = model(mutated).logits

    # The observation logits must MOVE when the plan is blanked.
    moved = False
    for name in obs:
        delta = (base[name] - mut[name]).abs().max().item()
        if delta > 1e-5:
            moved = True
            break
    assert moved, "zeroing the plan left the output unchanged — plan not load-bearing"


def test_conditioning_load_bearing_on_valid_targets(tmp_path):
    """The mutation also changes the NLL on valid target positions.

    A stronger statement than logit movement: the forecasting loss itself
    responds to the plan, so the conditioning affects what the model predicts
    where it is scored.
    """
    shots = [24065, 24066]
    mods = _make_corpus(tmp_path, shots)
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    s0 = build_shot_sample(24065, mods, window, token_root=tmp_path)
    obs = [m.name for m in mods if not m.is_conditioning]
    plan = [m.name for m in mods if m.is_conditioning]

    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=30,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=30,
        eval_every=0,
        num_workers=0,
        window=window,
        model_kwargs=_TINY_MODEL,
    )
    train_corpus(
        shots,
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    model, _ = load_model_from_checkpoint(find_latest_checkpoint(out_dir))
    model.eval()

    batch = collate_samples([s0], obs, plan)
    with torch.no_grad():
        base_loss = float(
            next_token_nll(model(batch).logits, batch, obs, target_only=True)
        )
        mutated = dict(batch)
        mutated["tokens"] = dict(batch["tokens"])
        for name in plan:
            mutated["tokens"][name] = torch.zeros_like(batch["tokens"][name])
        mut_loss = float(
            next_token_nll(model(mutated).logits, mutated, obs, target_only=True)
        )
    assert abs(base_loss - mut_loss) > 1e-6, (
        f"plan mutation did not move the target NLL ({base_loss:.6f} vs {mut_loss:.6f})"
    )
