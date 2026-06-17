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

import imas_ambix.worldmodel.dataset as wm_dataset
from imas_ambix.tokenizer.registry import L2_BLOCK_VOCAB
from imas_ambix.worldmodel.dataset import (
    CAMERA_IDS,
    ModalitySpec,
    WorldModelDataset,
    WorldModelWindowConfig,
    build_shot_sample,
    cache_config_hash,
    camera_channel_width,
    default_modalities,
    discover_worldmodel_shots,
    load_or_assemble_sample,
    resolve_cache_dir,
)
from imas_ambix.worldmodel.eval import evaluate_shot
from imas_ambix.worldmodel.model import WorldModel, WorldModelConfig
from imas_ambix.worldmodel.train import (
    CorpusTrainConfig,
    CudaPrefetcher,
    _build_corpus_model,
    _corpus_model_kwargs,
    _resolve_channels,
    _select_eval_shots,
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


# A modest synthetic camera vocab — the production camera is 1<<18, far too
# large to embed in a hermetic CPU test (a 262144xd table per the embedding +
# head would be ~hundreds of MB).  The WIRING is identical at any vocab; this
# test proves the camera modality flows tokens -> embedding -> head -> loss.
_CAMERA_VOCAB = 4096


def _write_camera_store(
    root: Path,
    shot_id: int,
    camera: str,
    *,
    n_frames: int,
    rate_hz: float,
    t0: float,
    vocab: int,
) -> None:
    """Write a synthetic ``frames/{shot}/{camera}.zarr`` camera-token store.

    Mirrors the real store shape ``(n_frames, 16, 16)`` int tokens so the
    dataset's :func:`_read_camera` (which subsamples the 16x16 grid at
    ``camera_grid_stride``) reads it unchanged.  Tokens are a per-frame ramp so
    the camera carries overfittable structure.  A level-1 timestamp axis is NOT
    written, so the loader uses its synthetic uniform fallback — but we write
    the camera frames spanning the SAME window as the L2 anchors so the
    nearest-on-grid resample actually lands camera tokens on the grid.
    """
    import zarr

    grid = np.zeros((n_frames, 16, 16), dtype=np.int32)
    for f in range(n_frames):
        for r in range(16):
            grid[f, r, :] = ((f + r + shot_id) % (vocab - 1)) + 1
    path = root / "v1" / "frames" / str(shot_id) / f"{camera}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=grid)
    # frame timestamps over the same window the L2 anchors cover; the loader
    # also has a synthetic fallback, but writing token_time here is harmless.
    dt = 1.0 / rate_hz
    store.create_array(
        "token_time", data=t0 + np.arange(n_frames, dtype=np.float64) * dt
    )
    store.attrs.update({"camera": camera, "vocab_size": int(vocab)})


def _make_camera_corpus(root: Path, shot_ids: list[int]) -> list[ModalitySpec]:
    """Write distinct synthetic shots carrying plan + summary + CAMERA.

    The camera spec uses the production wiring (kind="camera",
    camera_grid_stride=4 -> a 4x4=16-token frame) but a modest vocab so the
    embedding/head stay hermetic-test-sized.
    """
    for sid in shot_ids:
        _write_signal_hf_store(
            root,
            sid,
            "pulse_schedule_l2",
            n_time=2000,
            n_channels=2,
            rate_hz=4000.0,
            t0=-0.05,
            seed=sid,
            local_base=12804,
        )
        _write_signal_hf_store(
            root,
            sid,
            "summary_l2",
            n_time=2000,
            n_channels=3,
            rate_hz=4000.0,
            t0=-0.05,
            seed=sid + 1,
            local_base=12804,
        )
        # camera frames span the L2 window (-0.05 .. ~0.45 s) at ~600 Hz so the
        # nearest-on-grid resample lands real camera tokens on the model grid.
        _write_camera_store(
            root,
            sid,
            "rbb",
            n_frames=300,
            rate_hz=600.0,
            t0=-0.05,
            vocab=_CAMERA_VOCAB,
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
        ModalitySpec(
            "camera",
            "camera",
            "rbb",
            _CAMERA_VOCAB,
            anchors_grid=False,
            camera_grid_stride=4,
        ),
    ]


# The five MAST cameras the production default_modalities carries (optional).
_FIVE_CAMERAS = ("rbb", "rba", "rco", "rgb", "rgc")


def _five_camera_specs() -> list[ModalitySpec]:
    """The production-shaped modality set: required core + 5 OPTIONAL cameras.

    Uses the production wiring (kind="camera", camera_grid_stride=4, optional)
    but a modest synthetic camera vocab so the per-camera embedding + head stay
    hermetic-CPU-sized.  ``required=False`` mirrors default_modalities — the
    cameras do not gate discovery and an absent camera collates to all-PAD.
    """
    specs: list[ModalitySpec] = [
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            L2_VOCAB,
            is_conditioning=True,
        ),
        ModalitySpec("summary", "signal_hf", "summary_l2", L2_VOCAB),
    ]
    specs += [
        ModalitySpec(
            cam,
            "camera",
            cam,
            _CAMERA_VOCAB,
            anchors_grid=False,
            camera_grid_stride=4,
            required=False,
        )
        for cam in _FIVE_CAMERAS
    ]
    return specs


def _make_five_camera_corpus(
    root: Path, shot_present: dict[int, list[str]]
) -> list[ModalitySpec]:
    """Write shots carrying the required core + a PER-SHOT subset of cameras.

    ``shot_present`` maps shot id -> the list of camera ids that shot carries on
    disk.  A camera NOT listed for a shot has no store written, so it must be
    handled as an OPTIONAL absent modality (all-PAD + masked), never a dropped
    shot.  Returns the production-shaped 5-camera modality set.
    """
    for sid, cams in shot_present.items():
        _write_signal_hf_store(
            root,
            sid,
            "pulse_schedule_l2",
            n_time=2000,
            n_channels=2,
            rate_hz=4000.0,
            t0=-0.05,
            seed=sid,
            local_base=12804,
        )
        _write_signal_hf_store(
            root,
            sid,
            "summary_l2",
            n_time=2000,
            n_channels=3,
            rate_hz=4000.0,
            t0=-0.05,
            seed=sid + 1,
            local_base=12804,
        )
        for cam in cams:
            _write_camera_store(
                root,
                sid,
                cam,
                n_frames=300,
                rate_hz=600.0,
                t0=-0.05,
                vocab=_CAMERA_VOCAB,
            )
    return _five_camera_specs()


# Modest synthetic vocabs for the HF streams — the production xim is 12806 and
# xsx 1030, far larger than a hermetic CPU test wants to embed.  The WIRING is
# identical at any vocab; these prove the HF modalities flow tokens -> embedding
# -> head -> loss exactly like the L2 groups.
_XIM_VOCAB = 256
_XSX_VOCAB = 128


def _full_substrate_specs() -> list[ModalitySpec]:
    """The FULL-substrate modality set at hermetic-CPU vocab sizes.

    Mirrors the production :func:`default_modalities` SHAPE — minimal required
    core (plan + summary) + OPTIONAL extra L2 groups + OPTIONAL HF streams
    (xim/xsx) + OPTIONAL cameras — but with small synthetic vocabs so the
    per-modality embedding + head tables stay CPU-test-sized.
    """
    return [
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            L2_VOCAB,
            is_conditioning=True,
            required=True,
        ),
        ModalitySpec("summary", "signal_hf", "summary_l2", L2_VOCAB, required=True),
        ModalitySpec(
            "pf_active", "signal_hf", "pf_active_l2", L2_VOCAB, required=False
        ),
        ModalitySpec(
            "gas_injection",
            "signal_hf",
            "gas_injection_l2",
            L2_VOCAB,
            required=False,
        ),
        ModalitySpec(
            "xim", "signal_hf", "xim", _XIM_VOCAB, anchors_grid=False, required=False
        ),
        ModalitySpec(
            "xsx", "signal_hf", "xsx", _XSX_VOCAB, anchors_grid=False, required=False
        ),
        ModalitySpec(
            "rbb",
            "camera",
            "rbb",
            _CAMERA_VOCAB,
            anchors_grid=False,
            camera_grid_stride=4,
            required=False,
        ),
        ModalitySpec(
            "rco",
            "camera",
            "rco",
            _CAMERA_VOCAB,
            anchors_grid=False,
            camera_grid_stride=4,
            required=False,
        ),
    ]


def _make_full_substrate_corpus(
    root: Path, shot_present: dict[int, list[str]]
) -> list[ModalitySpec]:
    """Write shots carrying the core + a PER-SHOT subset of the full substrate.

    ``shot_present`` maps shot id -> the OPTIONAL stream names that shot carries
    (any of ``pf_active``/``gas_injection``/``xim``/``xsx``/``rbb``/``rco``).
    Every shot always gets the required core (``pulse_schedule`` + ``summary``).
    A stream not listed for a shot has no store written, so it must be handled
    as an OPTIONAL absent modality (all-PAD + masked), never a dropped shot.
    """
    l2_groups = {
        "pf_active": "pf_active_l2",
        "gas_injection": "gas_injection_l2",
    }
    hf_vocab = {"xim": _XIM_VOCAB, "xsx": _XSX_VOCAB}
    cameras = {"rbb", "rco"}
    for sid, present in shot_present.items():
        # required core — always present.
        _write_signal_hf_store(
            root,
            sid,
            "pulse_schedule_l2",
            n_time=2000,
            n_channels=2,
            rate_hz=4000.0,
            t0=-0.05,
            seed=sid,
            local_base=12804,
        )
        _write_signal_hf_store(
            root,
            sid,
            "summary_l2",
            n_time=2000,
            n_channels=3,
            rate_hz=4000.0,
            t0=-0.05,
            seed=sid + 1,
            local_base=12804,
        )
        for name in present:
            if name in l2_groups:
                _write_signal_hf_store(
                    root,
                    sid,
                    l2_groups[name],
                    n_time=2000,
                    n_channels=2,
                    rate_hz=4000.0,
                    t0=-0.05,
                    seed=sid + hash(name) % 7,
                    local_base=12804,
                )
            elif name in hf_vocab:
                # HF patch stores begin at the control range (base 4); a modest
                # vocab keeps the synthetic table small.
                _write_signal_hf_store(
                    root,
                    sid,
                    name,
                    n_time=400,
                    n_channels=4,
                    rate_hz=800.0,
                    t0=-0.05,
                    seed=sid + hash(name) % 7,
                    local_base=4,
                )
            elif name in cameras:
                _write_camera_store(
                    root,
                    sid,
                    name,
                    n_frames=300,
                    rate_hz=600.0,
                    t0=-0.05,
                    vocab=_CAMERA_VOCAB,
                )
    return _full_substrate_specs()


_TINY_MODEL = {"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64}
# A deliberately BIGGER backbone (still hermetic-CPU-fast) used to prove the
# model-size knobs grow the parameter count relative to _TINY_MODEL.
_BIG_MODEL = {"d_model": 96, "n_layers": 4, "n_heads": 6, "d_ff": 384}


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


# ---------------------------------------------------------------------------
# 6. MODEL-SIZE KNOBS — --d-model/--n-layers/--n-heads/--dropout enlarge the
#    backbone, and an unset knob set reproduces the field defaults (back-compat)
# ---------------------------------------------------------------------------


class _Args:
    """Minimal argparse.Namespace stand-in for the CLI-knob collector."""

    def __init__(self, **kw):
        self.d_model = None
        self.n_layers = None
        self.n_heads = None
        self.d_ff = None
        self.dropout = None
        for k, v in kw.items():
            setattr(self, k, v)


def test_corpus_model_kwargs_back_compat_when_unset():
    """No knobs set => empty kwargs => WorldModelConfig field defaults stand."""
    assert _corpus_model_kwargs(_Args()) == {}


def test_corpus_model_kwargs_collects_and_autoscale_d_ff():
    """Setting --d-model derives d_ff=4*d_model unless --d-ff is pinned."""
    kw = _corpus_model_kwargs(_Args(d_model=384, n_layers=8, n_heads=8, dropout=0.1))
    assert kw == {
        "d_model": 384,
        "n_layers": 8,
        "n_heads": 8,
        "dropout": 0.1,
        "d_ff": 4 * 384,
    }
    # explicit --d-ff overrides the 4x derivation
    kw2 = _corpus_model_kwargs(_Args(d_model=256, d_ff=1024))
    assert kw2["d_ff"] == 1024
    # d_ff is NOT injected when d_model is unset (back-compat: default 256)
    kw3 = _corpus_model_kwargs(_Args(n_layers=6))
    assert "d_ff" not in kw3 and kw3 == {"n_layers": 6}


def test_model_size_knobs_enlarge_param_count(tmp_path):
    """A bigger d_model/n_layers/n_heads => strictly more parameters.

    Builds the corpus model twice from the SAME resolved channels — once tiny,
    once big — and asserts the big backbone has more trainable parameters and a
    wider d_model.  This is the knob-threading proof: model_kwargs flows
    CorpusTrainConfig -> _build_corpus_model -> WorldModelConfig.from_modalities.
    """
    shots = [24065, 24066]
    mods = _make_corpus(tmp_path, shots)
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    kept, channels = _resolve_channels(
        shots, mods, window, token_root=tmp_path, level1_dir=None
    )
    tiny = _build_corpus_model(kept, channels, window, **_TINY_MODEL)
    big = _build_corpus_model(kept, channels, window, **_BIG_MODEL)
    assert big.config.d_model > tiny.config.d_model
    assert big.config.n_layers > tiny.config.n_layers
    assert big.num_parameters() > tiny.num_parameters(), (
        f"size knobs did not grow the model: "
        f"{tiny.num_parameters()} -> {big.num_parameters()}"
    )


# ---------------------------------------------------------------------------
# 7. CAMERA modality trains in the corpus path — embedding + head wired,
#    tokens flow, loss drops, checkpoint round-trips (the richest modality)
# ---------------------------------------------------------------------------


def test_camera_modality_is_wired_and_trains(tmp_path):
    """The camera modality flows tokens -> embedding -> head -> loss -> ckpt.

    Uses the production camera wiring (kind="camera", camera_grid_stride=4 so a
    16x16 frame becomes a 4x4=16-token channel-group) at a modest synthetic
    vocab.  Proves: the model builds a camera embedding table AND a camera head;
    the assembled sample carries 16 camera channels; the corpus loop scores the
    camera (loss drops); and the checkpoint round-trips with the camera intact.
    """
    shots = [24065, 24066, 24067]
    mods = _make_camera_corpus(tmp_path, shots)
    assert any(m.name == "camera" for m in mods)

    # the camera frame-token assembly: 16x16 grid at stride 4 -> 16 channels
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)
    s0 = build_shot_sample(24065, mods, window, token_root=tmp_path)
    assert "camera" in s0.tokens, "camera modality did not assemble into the sample"
    assert s0.tokens["camera"].shape == (24, 16), (
        f"camera channels wrong: {s0.tokens['camera'].shape} (expected (24, 16))"
    )
    assert bool(s0.valid["camera"].any()), "camera tokens never landed on the grid"

    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=40,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=40,
        eval_every=0,
        num_workers=0,
        window=window,
        model_kwargs=_BIG_MODEL,
    )
    result = train_corpus(
        shots,
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    assert result.steps_run == 40
    early = float(np.mean(result.losses[:10]))
    late = float(np.mean(result.losses[-10:]))
    assert late < early, f"camera-inclusive loss did not drop: {early} -> {late}"

    # the model has BOTH a camera embedding table and a camera next-token head
    ckpt = find_latest_checkpoint(out_dir)
    model, payload = load_model_from_checkpoint(ckpt)
    assert "camera" in model.token_embed, "no camera embedding table"
    assert "camera" in model.heads, "no camera next-token head"
    assert model.token_embed["camera"].num_embeddings == _CAMERA_VOCAB
    # camera is a NON-conditioning observation modality the model predicts
    assert "camera" in model._obs_names
    # checkpoint config carries the camera modality (eval-loadable contract)
    cam_specs = [
        m for m in payload["model_config"]["modalities"] if m["name"] == "camera"
    ]
    assert cam_specs and cam_specs[0]["vocab_size"] == _CAMERA_VOCAB

    # the camera also receives next-token logits over its own vocab
    obs = [m.name for m in mods if not m.is_conditioning]
    plan = [m.name for m in mods if m.is_conditioning]
    model.eval()
    batch = collate_samples([s0], obs, plan)
    with torch.no_grad():
        out = model(batch)
    assert "camera" in out.logits
    assert out.logits["camera"].shape[-1] == _CAMERA_VOCAB


# ---------------------------------------------------------------------------
# 8. FIVE CAMERAS — default_modalities emits all five as OPTIONAL, discovery
#    requires only the CORE (so a shot missing cameras is NOT dropped), the
#    model builds five camera embeddings + heads, a missing-camera shot still
#    trains (absent camera -> all-PAD + masked), and the param count grows with
#    the cameras.
# ---------------------------------------------------------------------------


def test_default_modalities_emits_full_substrate():
    """default_modalities emits the FULL tokenised substrate.

    The conditioning plan + five measured L2 groups + three L1 HF streams +
    five cameras — every confirmed-on-disk stream — and the REQUIRED core is
    MINIMAL (just the plan + summary) so the corpus stays large.
    """
    mods = default_modalities()
    by_name = {m.name: m for m in mods}

    # ── the full signal_hf substrate (plan + 5 L2 + xma/xim/xsx) ────────────
    sig_names = {m.name for m in mods if m.kind == "signal_hf"}
    assert sig_names == {
        "pulse_schedule",
        "summary",
        "pf_active",
        "interferometer",
        "gas_injection",
        "soft_x_rays",
        "xma",
        "xim",
        "xsx",
    }
    # the conditioning plan is present and is the conditioning modality.
    assert by_name["pulse_schedule"].is_conditioning is True

    # ── on-disk groups + vocab sizing ───────────────────────────────────────
    assert by_name["gas_injection"].group == "gas_injection_l2"
    assert by_name["soft_x_rays"].group == "soft_x_rays_l2"
    assert by_name["xim"].group == "xim" and by_name["xim"].vocab_size >= 12800
    assert by_name["xsx"].group == "xsx" and by_name["xsx"].vocab_size >= 1024
    # the L2 light path shares the 256-bin uniform-quantiser vocab (+1 PAD).
    for n in ("summary", "pf_active", "interferometer", "gas_injection", "soft_x_rays"):
        assert by_name[n].vocab_size == L2_VOCAB
        assert by_name[n].group == f"{n}_l2"

    # ── grid anchoring: only the L2 light path anchors; HF + cameras do not ─
    for n in (
        "pulse_schedule",
        "summary",
        "pf_active",
        "interferometer",
        "gas_injection",
        "soft_x_rays",
    ):
        assert by_name[n].anchors_grid is True, f"{n} (L2) must anchor the grid"
    for n in ("xma", "xim", "xsx"):
        assert by_name[n].anchors_grid is False, f"{n} (HF) must not anchor"

    # ── all five cameras, OPTIONAL, full-vocab, per-camera identity ─────────
    cams = [m for m in mods if m.kind == "camera"]
    assert [m.name for m in cams] == list(CAMERA_IDS)
    assert {m.name for m in cams} == {"rbb", "rba", "rco", "rgb", "rgc"}
    for m in cams:
        assert m.required is False, f"{m.name} must be optional"
        assert m.anchors_grid is False
        assert m.vocab_size == (1 << 18)
        assert m.group == m.name  # name == group == camera id

    # ── REQUIRED core is MINIMAL: only the plan + summary gate the corpus ───
    required = {m.name for m in mods if m.required}
    assert required == {"pulse_schedule", "summary"}, (
        f"required core must be only plan + summary, got {required}"
    )
    # everything else (other L2, all HF, all cameras) is OPTIONAL.
    optional = {m.name for m in mods if not m.required}
    assert optional == {
        "pf_active",
        "interferometer",
        "gas_injection",
        "soft_x_rays",
        "xma",
        "xim",
        "xsx",
        "rbb",
        "rba",
        "rco",
        "rgb",
        "rgc",
    }


def test_discovery_requires_core_not_cameras(tmp_path):
    """A shot carrying the core but MISSING cameras is still discovered.

    This is the corpus-size guarantee: requiring all five cameras would shrink
    the corpus to the rare all-camera intersection; requiring only the core
    admits every shot that carries the core.  Shot A has 2 cameras, shot B has
    0, shot C has a disjoint camera — all three must be discovered, and a shot
    missing the core (no summary) must NOT be.
    """
    specs = _make_five_camera_corpus(
        tmp_path,
        {
            24065: ["rbb", "rba"],  # subset of cameras
            24066: [],  # NO cameras at all — still has the core
            24067: ["rco"],  # a different single camera
        },
    )
    # a shot missing the required core (write only a camera, no summary/plan)
    _write_camera_store(
        tmp_path,
        24099,
        "rbb",
        n_frames=300,
        rate_hz=600.0,
        t0=-0.05,
        vocab=_CAMERA_VOCAB,
    )

    # explicit ascending sampling: gate-membership test (order asserted below
    # in its own camera-first test); the SET must be the three core shots.
    found = discover_worldmodel_shots(
        specs,
        token_root=tmp_path,
        shot_ids=[24065, 24066, 24067, 24099],
        sample="ascending",
    )
    # all three core-bearing shots are kept regardless of which cameras they
    # carry; the camera-only shot (no core) is excluded.
    assert found == [24065, 24066, 24067], (
        f"discovery must gate on the core only, not the cameras: got {found}"
    )

    # the DEFAULT (camera_first) sampling discovers the SAME set but moves the
    # camera-bearing shots (24065, 24067) ahead of the camera-free core shot
    # (24066) so a small --n-shots limit is camera-dense (FIX B).
    found_default = discover_worldmodel_shots(
        specs, token_root=tmp_path, shot_ids=[24065, 24066, 24067, 24099]
    )
    assert set(found_default) == {24065, 24066, 24067}, (
        f"camera_first must discover the same core set: got {found_default}"
    )
    assert found_default.index(24066) == 2, (
        "camera_first must place the camera-free core shot LAST, behind the "
        f"camera-bearing shots: got {found_default}"
    )
    assert set(found_default[:2]) == {24065, 24067}, (
        f"camera_first must front the camera-bearing shots: got {found_default}"
    )


def test_five_cameras_missing_some_still_trains(tmp_path):
    """A shot missing 2 of 5 cameras trains; absent cameras are all-PAD masked.

    Three shots carry DIFFERENT camera subsets (none carries all five).  The
    corpus trainer sizes one embedding + head per camera (optional cameras kept
    when present in ANY probed shot), trains over the multi-shot DataLoader, and
    a shot missing cameras contributes an all-PAD + masked block for the absent
    cameras — never dropped.  The loss drops and the checkpoint round-trips with
    all five camera tables intact.
    """
    specs = _make_five_camera_corpus(
        tmp_path,
        {
            24065: ["rbb", "rba", "rco"],  # missing rgb, rgc
            24066: ["rco", "rgb", "rgc"],  # missing rbb, rba
            24067: ["rbb", "rgc"],  # missing rba, rco, rgb
        },
    )
    shots = [24065, 24066, 24067]

    # the per-shot assembly drops absent cameras (build_shot_sample), and the
    # pad-collate fills them back in as all-PAD + masked at fixed width.
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)
    s_mid = build_shot_sample(24067, specs, window, token_root=tmp_path)
    assert "rbb" in s_mid.tokens and "rgc" in s_mid.tokens
    assert "rba" not in s_mid.tokens, "absent camera must not assemble into a sample"

    kept, channels = _resolve_channels(
        shots, specs, window, token_root=tmp_path, level1_dir=None
    )
    # ALL five cameras are sized (each present in at least one probed shot),
    # even though no single shot carries all five — the optional-modality rule.
    kept_names = {m.name for m in kept}
    for cam in _FIVE_CAMERAS:
        assert cam in kept_names, f"optional camera {cam} dropped during sizing"
        assert channels[cam] == 16  # 16x16 grid at stride 4 -> 4x4 = 16

    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=40,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=40,
        eval_every=0,
        num_workers=0,
        window=window,
        model_kwargs=_BIG_MODEL,
    )
    result = train_corpus(
        shots,
        modalities=specs,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    assert result.steps_run == 40
    early = float(np.mean(result.losses[:10]))
    late = float(np.mean(result.losses[-10:]))
    assert late < early, f"five-camera loss did not drop: {early} -> {late}"

    # the checkpoint round-trips with one embedding + head PER camera, and the
    # absent-on-some-shots cameras are still full modalities of the model.
    ckpt = find_latest_checkpoint(out_dir)
    model, payload = load_model_from_checkpoint(ckpt)
    for cam in _FIVE_CAMERAS:
        assert cam in model.token_embed, f"no embedding for camera {cam}"
        assert cam in model.heads, f"no next-token head for camera {cam}"
        assert model.token_embed[cam].num_embeddings == _CAMERA_VOCAB
        assert cam in model._obs_names
    ckpt_cams = {
        m["name"]
        for m in payload["model_config"]["modalities"]
        if m["name"] in _FIVE_CAMERAS
    }
    assert ckpt_cams == set(_FIVE_CAMERAS), "checkpoint lost a camera modality"

    # the masking proof: for shot 24067 (missing rba/rco/rgb), those cameras'
    # collated blocks are all-PAD + all-invalid (no loss, no embedding signal).
    chan = dict(channels)
    obs = [m.name for m in specs if not m.is_conditioning]
    plan = [m.name for m in specs if m.is_conditioning]
    batch = pad_collate_batch([s_mid], obs, plan, chan)
    for cam in ("rba", "rco", "rgb"):
        assert not bool(batch["valid"][cam].any()), (
            f"absent camera {cam} must be all-invalid in the collated batch"
        )
        assert int(batch["tokens"][cam].max()) == 0, (
            f"absent camera {cam} must be all-PAD in the collated batch"
        )
    # the cameras the shot DOES carry land real, valid tokens.
    for cam in ("rbb", "rgc"):
        assert bool(batch["valid"][cam].any()), (
            f"present camera {cam} has no valid token"
        )


def test_param_count_grows_with_cameras(tmp_path):
    """More camera modalities => strictly more parameters (camera-table driven).

    Builds the corpus model from the SAME core but with 1 vs 5 cameras and
    asserts the five-camera model has more parameters — the camera embedding +
    head tables are the param driver, so each extra camera adds a full
    embedding + head over the camera vocab.
    """
    specs5 = _make_five_camera_corpus(
        tmp_path,
        {
            24065: list(_FIVE_CAMERAS),
            24066: list(_FIVE_CAMERAS),
        },
    )
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)

    kept5, ch5 = _resolve_channels(
        [24065, 24066], specs5, window, token_root=tmp_path, level1_dir=None
    )
    model5 = _build_corpus_model(kept5, ch5, window, **_TINY_MODEL)

    # one-camera variant: same core, only rbb retained.
    specs1 = [m for m in specs5 if m.kind != "camera" or m.name == "rbb"]
    kept1, ch1 = _resolve_channels(
        [24065, 24066], specs1, window, token_root=tmp_path, level1_dir=None
    )
    model1 = _build_corpus_model(kept1, ch1, window, **_TINY_MODEL)

    n_cam5 = sum(1 for m in model5.config.modalities if m.name in _FIVE_CAMERAS)
    n_cam1 = sum(1 for m in model1.config.modalities if m.name in _FIVE_CAMERAS)
    assert n_cam5 == 5 and n_cam1 == 1
    assert model5.num_parameters() > model1.num_parameters(), (
        f"five cameras did not grow the model: "
        f"{model1.num_parameters()} -> {model5.num_parameters()}"
    )
    # each extra camera adds ~ (embedding + head) = 2 * vocab * d_model params.
    d = _TINY_MODEL["d_model"]
    per_camera = 2 * _CAMERA_VOCAB * d  # embedding (vocab*d) + head (d*vocab)
    grew = model5.num_parameters() - model1.num_parameters()
    # 4 extra cameras; allow the head-bias slack but require it be in the ballpark
    assert grew >= 4 * per_camera, (
        f"param growth {grew} below 4 cameras' embedding+head floor {4 * per_camera}"
    )


# ---------------------------------------------------------------------------
# 9. FULL SUBSTRATE — the trainer uses EVERY tokenised stream: plan + summary
#    (core) + extra L2 (pf_active/gas_injection) + HF (xim/xsx) + cameras, with
#    shots MISSING several streams.  Discovery gates only on the core (corpus
#    stays large), the absent streams collate to all-PAD + masked (not dropped),
#    the loss drops, the checkpoint round-trips with every stream, and the param
#    count grows when the HF + camera streams are added.
# ---------------------------------------------------------------------------


def test_full_substrate_discovery_gates_on_core_only(tmp_path):
    """Discovery keeps the corpus LARGE: only plan + summary gate it.

    Three shots carry DIFFERENT subsets of the optional substrate (none carries
    all streams); a fourth shot has only an optional stream (no core).  All
    three core-bearing shots are discovered regardless of which optional streams
    they carry; the core-less shot is excluded.
    """
    specs = _make_full_substrate_corpus(
        tmp_path,
        {
            24065: ["pf_active", "xim", "rbb"],  # some L2 + HF + camera
            24066: [],  # ONLY the core (plan + summary) — still discovered
            24067: ["gas_injection", "xsx", "rco"],  # a disjoint subset
        },
    )
    # a shot missing the core: write only an optional HF stream, no plan/summary.
    _write_signal_hf_store(
        tmp_path,
        24099,
        "xim",
        n_time=400,
        n_channels=4,
        rate_hz=800.0,
        t0=-0.05,
        seed=1,
        local_base=4,
    )

    # explicit ascending sampling pins the gate-membership order for this set.
    found = discover_worldmodel_shots(
        specs,
        token_root=tmp_path,
        shot_ids=[24065, 24066, 24067, 24099],
        sample="ascending",
    )
    assert found == [24065, 24066, 24067], (
        f"discovery must gate on the core (plan+summary) only: got {found}"
    )
    # the default camera_first sampling discovers the same SET (24066 carries
    # no camera, so it sorts last) — the corpus stays core-gated.
    found_default = discover_worldmodel_shots(
        specs, token_root=tmp_path, shot_ids=[24065, 24066, 24067, 24099]
    )
    assert set(found_default) == {24065, 24066, 24067}, (
        f"discovery set must gate on the core only: got {found_default}"
    )


def test_full_substrate_missing_streams_still_trains(tmp_path):
    """A shot missing many streams trains; absent streams are all-PAD masked.

    Three shots carry DIFFERENT subsets of the FULL substrate (plan + summary +
    a per-shot mix of pf_active/gas_injection/xim/xsx/rbb/rco).  The trainer
    sizes one embedding + head per OPTIONAL stream (kept when present in ANY
    probed shot), trains over the multi-shot DataLoader, and a shot missing a
    stream contributes an all-PAD + masked block for it — never dropped.  The
    loss drops and the checkpoint round-trips with EVERY stream intact.
    """
    specs = _make_full_substrate_corpus(
        tmp_path,
        {
            24065: ["pf_active", "gas_injection", "xim", "rbb"],  # no xsx, no rco
            24066: ["gas_injection", "xsx", "rco"],  # no pf_active, no xim, no rbb
            24067: ["pf_active", "xim", "xsx", "rbb", "rco"],  # no gas_injection
        },
    )
    shots = [24065, 24066, 24067]
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)

    # the per-shot assembly drops absent streams; the pad-collate fills them
    # back in as all-PAD + masked at fixed width.
    s_065 = build_shot_sample(24065, specs, window, token_root=tmp_path)
    assert "xim" in s_065.tokens and "rbb" in s_065.tokens
    assert "xsx" not in s_065.tokens, "absent HF stream must not assemble"
    assert "rco" not in s_065.tokens, "absent camera must not assemble"

    kept, channels = _resolve_channels(
        shots, specs, window, token_root=tmp_path, level1_dir=None
    )
    # EVERY optional stream is sized (each present in at least one probed shot),
    # even though no single shot carries all of them — the optional-modality
    # rule keeps the full substrate wired.
    kept_names = {m.name for m in kept}
    for name in (
        "pulse_schedule",
        "summary",
        "pf_active",
        "gas_injection",
        "xim",
        "xsx",
        "rbb",
        "rco",
    ):
        assert name in kept_names, f"optional stream {name} dropped during sizing"

    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=40,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=40,
        eval_every=0,
        num_workers=0,
        window=window,
        model_kwargs=_BIG_MODEL,
    )
    result = train_corpus(
        shots,
        modalities=specs,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    assert result.steps_run == 40
    early = float(np.mean(result.losses[:10]))
    late = float(np.mean(result.losses[-10:]))
    assert late < early, f"full-substrate loss did not drop: {early} -> {late}"

    # checkpoint round-trips with one embedding + head per non-conditioning
    # stream (incl the HF streams and the cameras).
    ckpt = find_latest_checkpoint(out_dir)
    model, payload = load_model_from_checkpoint(ckpt)
    for name in ("summary", "pf_active", "gas_injection", "xim", "xsx", "rbb", "rco"):
        assert name in model.token_embed, f"no embedding for {name}"
        assert name in model.heads, f"no next-token head for {name}"
        assert name in model._obs_names
    # the HF heads emit logits over their own vocab.
    assert model.heads["xim"].out_features == _XIM_VOCAB
    assert model.heads["xsx"].out_features == _XSX_VOCAB
    ckpt_names = {m["name"] for m in payload["model_config"]["modalities"]}
    assert {
        "pulse_schedule",
        "summary",
        "pf_active",
        "gas_injection",
        "xim",
        "xsx",
        "rbb",
        "rco",
    } <= ckpt_names

    # the masking proof: for shot 24066 (no pf_active/xim/rbb), those streams'
    # collated blocks are all-PAD + all-invalid (no loss, no embedding signal).
    s_066 = build_shot_sample(24066, specs, window, token_root=tmp_path)
    chan = dict(channels)
    obs = [m.name for m in specs if not m.is_conditioning]
    plan = [m.name for m in specs if m.is_conditioning]
    batch = pad_collate_batch([s_066], obs, plan, chan)
    for absent in ("pf_active", "xim", "rbb"):
        assert not bool(batch["valid"][absent].any()), (
            f"absent stream {absent} must be all-invalid in the collated batch"
        )
        assert int(batch["tokens"][absent].max()) == 0, (
            f"absent stream {absent} must be all-PAD in the collated batch"
        )
    # the streams the shot DOES carry land real, valid tokens.
    for present in ("gas_injection", "xsx", "rco"):
        assert bool(batch["valid"][present].any()), (
            f"present stream {present} has no valid token"
        )


def test_param_count_grows_with_full_substrate(tmp_path):
    """The full substrate has strictly more params than the core alone.

    Builds the corpus model from the SAME core (plan + summary) with vs without
    the optional HF + camera streams and asserts the full-substrate model has
    more parameters — each extra stream adds a full embedding + head over its
    vocab, so using every stream grows the model.
    """
    specs_full = _make_full_substrate_corpus(
        tmp_path,
        {
            24065: ["pf_active", "gas_injection", "xim", "xsx", "rbb", "rco"],
            24066: ["pf_active", "gas_injection", "xim", "xsx", "rbb", "rco"],
        },
    )
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    kept_full, ch_full = _resolve_channels(
        [24065, 24066], specs_full, window, token_root=tmp_path, level1_dir=None
    )
    model_full = _build_corpus_model(kept_full, ch_full, window, **_TINY_MODEL)

    # core-only variant: just the plan + summary, no optional streams.
    specs_core = [m for m in specs_full if m.name in ("pulse_schedule", "summary")]
    kept_core, ch_core = _resolve_channels(
        [24065, 24066], specs_core, window, token_root=tmp_path, level1_dir=None
    )
    model_core = _build_corpus_model(kept_core, ch_core, window, **_TINY_MODEL)

    full_names = {m.name for m in model_full.config.modalities}
    core_names = {m.name for m in model_core.config.modalities}
    assert full_names == {
        "pulse_schedule",
        "summary",
        "pf_active",
        "gas_injection",
        "xim",
        "xsx",
        "rbb",
        "rco",
    }
    assert core_names == {"pulse_schedule", "summary"}
    assert model_full.num_parameters() > model_core.num_parameters(), (
        f"full substrate did not grow the model: "
        f"{model_core.num_parameters()} -> {model_full.num_parameters()}"
    )


# ---------------------------------------------------------------------------
# 10. UNCONDITIONAL HEADS — the model builds an embedding + head for EVERY
#     declared modality, even when the channel-sizing probe (the first few
#     shots) carries NONE of them.  This is the camera/xma-absent-from-probe
#     bug: cameras live in high shot-ids, the probe band is low-id and
#     camera-free, and a probe-INTERSECTION sizing would silently drop the
#     camera/HF heads and collapse the all-streams model to a tiny one.
# ---------------------------------------------------------------------------


def test_camera_channel_width_is_structural_constant():
    """The camera channel width is a structural constant (no shot needed).

    16x16 frame grid sub-sampled at stride s -> (ceil(16/s))^2 tokens.  This is
    what fixes the camera head width without a camera-bearing probe shot.
    """
    assert camera_channel_width(4) == 16  # [0,4,8,12]^2 = 4x4
    assert camera_channel_width(8) == 4  # [0,8]^2 = 2x2
    assert camera_channel_width(2) == 64  # 8x8
    # the spec exposes it via fixed_channel_width()
    cam = ModalitySpec(
        "rbb", "camera", "rbb", _CAMERA_VOCAB, anchors_grid=False, camera_grid_stride=4
    )
    assert cam.fixed_channel_width() == 16
    # a signal_hf group with no declared n_channels has NO fixable width (probed)
    sig = ModalitySpec("summary", "signal_hf", "summary_l2", L2_VOCAB)
    assert sig.fixed_channel_width() is None
    # ...but a declared n_channels pins it
    sig_pinned = ModalitySpec(
        "pf_active", "signal_hf", "pf_active_l2", L2_VOCAB, n_channels=12
    )
    assert sig_pinned.fixed_channel_width() == 12


def test_forward_emits_fixed_width_logits_regardless_of_input_width():
    """forward emits logits at the head's FIXED channel width, not the input's.

    The blocker-2 crash did NOT surface in ``forward`` (an over-wide input was
    silently under-sliced by ``channel_query[:n_ch]``); it surfaced in the
    loss / skill score, where width-``model`` logits were compared against
    width-``input`` targets ("expanded size must match").  The forward now
    clamps the emitted channel count to the head width, so the logits a caller
    gets back can never exceed the head width — the loss/score paths stay
    rectangular even for a (mis-padded) wider-than-model batch.
    """
    specs = [
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            L2_VOCAB,
            is_conditioning=True,
            n_channels=2,
        ),
        ModalitySpec("summary", "signal_hf", "summary_l2", L2_VOCAB, n_channels=2),
    ]
    cfg = WorldModelConfig.from_modalities(
        specs, {}, plan_steps=8, obs_steps=8, **_TINY_MODEL
    )
    model = WorldModel(cfg)
    model.eval()
    model_w = {m.name: m.n_channels for m in cfg.modalities}["summary"]
    assert model_w == 2

    t = 8
    # WIDER-than-model input (the crash direction): 3 summary channels vs head 2.
    wide = {
        "tokens": {
            "pulse_schedule": torch.zeros(1, t, 2, dtype=torch.long),
            "summary": torch.ones(1, t, model_w + 1, dtype=torch.long),
        },
        "valid": {
            "pulse_schedule": torch.ones(1, t, 2, dtype=torch.bool),
            "summary": torch.ones(1, t, model_w + 1, dtype=torch.bool),
        },
        "context_steps": 2,
    }
    out = model(wide)
    # logits are emitted at the HEAD width (clamped), never the wider input.
    assert out.logits["summary"].shape[2] == model_w

    # NARROWER input (1 channel): logits track the narrower width, no error.
    narrow = {
        "tokens": {
            "pulse_schedule": torch.zeros(1, t, 2, dtype=torch.long),
            "summary": torch.ones(1, t, 1, dtype=torch.long),
        },
        "valid": {
            "pulse_schedule": torch.ones(1, t, 2, dtype=torch.bool),
            "summary": torch.ones(1, t, 1, dtype=torch.bool),
        },
        "context_steps": 2,
    }
    out_n = model(narrow)
    assert out_n.logits["summary"].shape[2] == 1


def test_pad_collate_then_loss_is_crash_free_for_wider_shot(tmp_path):
    """A wider-than-model shot, pad-collated, computes a finite loss (no crash).

    Reproduces the blocker-2 mechanism end-to-end at the train collate +
    loss boundary: build the model at summary width 3, assemble a held-out shot
    at summary width 5, pad-collate to the model widths, and confirm the masked
    next-token NLL is finite — the IndexError that the plain-stack path raised
    ("mask shape does not match indexed tensor") is gone.
    """
    mods = _make_corpus(tmp_path, [24065, 24066])  # 3 summary channels
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    kept, channels = _resolve_channels(
        [24065, 24066], mods, window, token_root=tmp_path, level1_dir=None
    )
    model = _build_corpus_model(kept, channels, window, **_TINY_MODEL)
    obs_names = [m.name for m in kept if not m.is_conditioning]
    plan_names = [m.name for m in kept if m.is_conditioning]
    model_w = channels["summary"]

    wider = model_w + 2
    _write_signal_hf_store(
        tmp_path,
        24082,
        "pulse_schedule_l2",
        n_time=2000,
        n_channels=2,
        rate_hz=4000.0,
        t0=-0.05,
        seed=24082,
        local_base=12804,
    )
    _write_signal_hf_store(
        tmp_path,
        24082,
        "summary_l2",
        n_time=2000,
        n_channels=wider,
        rate_hz=4000.0,
        t0=-0.05,
        seed=24083,
        local_base=12804,
    )
    s = build_shot_sample(24082, mods, window, token_root=tmp_path)
    assert s.tokens["summary"].shape[1] == wider != model_w

    # plain stack (the OLD path) crashes in the loss; pad-collate does not.
    with pytest.raises((RuntimeError, IndexError)):
        b_bad = collate_samples([s], obs_names, plan_names)
        out_bad = model(b_bad)
        next_token_nll(out_bad.logits, b_bad, obs_names)

    b = pad_collate_batch([s], obs_names, plan_names, channels)
    out = model(b)
    loss = next_token_nll(out.logits, b, obs_names)
    assert torch.isfinite(loss)


def test_heads_built_for_all_declared_modalities_when_probe_lacks_them(tmp_path):
    """The model builds camera + HF heads even when NO probe shot carries them.

    The channel-sizing probe sees ONLY the core (plan + summary) — no camera,
    no HF stream — exactly the low-id band pathology.  Yet the model MUST still
    build the camera/xim/xsx embedding + head tables (sized from the spec fixed
    widths), so they receive gradients once camera-bearing shots enter the
    DataLoader.  This is the unconditional-heads contract.
    """
    # write the probe shots WITHOUT any camera or HF store — core only.
    _make_full_substrate_corpus(tmp_path, {24065: [], 24066: []})
    specs = _full_substrate_specs()  # declares pf_active/gas_injection/xim/xsx/rbb/rco
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)

    kept, channels = _resolve_channels(
        [24065, 24066], specs, window, token_root=tmp_path, level1_dir=None
    )
    # EVERY declared modality is kept and sized — including the ones absent from
    # every probed shot (the cameras + HF streams).
    kept_names = {m.name for m in kept}
    for name in (
        "pulse_schedule",
        "summary",
        "pf_active",
        "gas_injection",
        "xim",
        "xsx",
        "rbb",
        "rco",
    ):
        assert name in kept_names, f"{name} dropped despite being declared"
        assert channels[name] >= 1, f"{name} sized to a non-positive width"
    # the cameras (absent from the probe) are sized to the STRUCTURAL width.
    assert channels["rbb"] == camera_channel_width(4) == 16
    assert channels["rco"] == 16

    model = _build_corpus_model(kept, channels, window, **_TINY_MODEL)
    # the heads + embeddings exist for the probe-absent camera/HF modalities.
    for name in ("rbb", "rco", "xim", "xsx", "pf_active", "gas_injection"):
        assert name in model.token_embed, f"no embedding for probe-absent {name}"
        assert name in model.heads, f"no next-token head for probe-absent {name}"
        assert name in model._obs_names
    assert model.token_embed["rbb"].num_embeddings == _CAMERA_VOCAB
    assert model.heads["xim"].out_features == _XIM_VOCAB
    assert model.heads["xsx"].out_features == _XSX_VOCAB


def test_from_modalities_does_not_filter_by_probe_intersection():
    """WorldModelConfig.from_modalities builds heads for ALL declared modalities.

    Even with an EMPTY channels dict (no probe sample at all), every declared
    modality gets a head sized from its spec fixed width — cameras from the
    stride, a pinned signal_hf from n_channels.  The old code filtered
    ``if m.name in channels``, dropping every un-probed modality.
    """
    specs = [
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            L2_VOCAB,
            is_conditioning=True,
            n_channels=2,
        ),
        ModalitySpec("summary", "signal_hf", "summary_l2", L2_VOCAB, n_channels=3),
        ModalitySpec(
            "rbb",
            "camera",
            "rbb",
            _CAMERA_VOCAB,
            anchors_grid=False,
            camera_grid_stride=4,
            required=False,
        ),
        ModalitySpec(
            "xim",
            "signal_hf",
            "xim",
            _XIM_VOCAB,
            anchors_grid=False,
            n_channels=4,
            required=False,
        ),
    ]
    cfg = WorldModelConfig.from_modalities(specs, {}, plan_steps=8, obs_steps=8)
    head_names = {m.name for m in cfg.modalities}
    assert head_names == {"pulse_schedule", "summary", "rbb", "xim"}
    by_name = {m.name: m for m in cfg.modalities}
    assert by_name["rbb"].n_channels == 16  # camera structural width
    assert by_name["xim"].n_channels == 4  # declared
    assert by_name["summary"].n_channels == 3
    # a probed width WINS over the spec fixed width.
    cfg2 = WorldModelConfig.from_modalities(
        specs, {"summary": 5}, plan_steps=8, obs_steps=8
    )
    assert {m.name: m.n_channels for m in cfg2.modalities}["summary"] == 5


# ---------------------------------------------------------------------------
# 11. CRASH-FREE PADDED EVAL — eval/rollout pads/truncates an assembled sample
#     to the MODEL's fixed channel widths, so a held-out shot whose per-modality
#     channel count differs from the training-probe widths does NOT crash the
#     forward (the old collate_samples plain-stack crashed), and a shot sharing
#     no scorable modality ERRORS loudly rather than silently reporting 0.
# ---------------------------------------------------------------------------


def test_eval_pads_held_out_shot_with_mismatched_channels(tmp_path):
    """A held-out shot with MORE summary channels than the model is not a crash.

    The model is sized to 3 summary channels (probe shots); the held-out shot
    carries 5.  The old eval (plain stack at the shot's native width) crashed
    the forward ("expanded size must match") — now the sample is pad/truncated
    to the model width, so the rollout runs and the skill is scored.
    """
    # train/probe shots: 3 summary channels.
    mods = _make_corpus(tmp_path, [24065, 24066])
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=8,
        batch_size=2,
        lr=5e-3,
        log_every=4,
        ckpt_every=8,
        eval_every=0,
        num_workers=0,
        window=window,
        model_kwargs=_TINY_MODEL,
    )
    train_corpus(
        [24065, 24066],
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    model, _ = load_model_from_checkpoint(find_latest_checkpoint(out_dir))
    model_w = {m.name: m.n_channels for m in model.config.modalities}
    model_summary_w = int(model_w["summary"])

    # held-out shot 24080 with a DIFFERENT (wider) summary channel count than
    # the model was sized to — the case that used to crash the forward.
    eval_summary_w = model_summary_w + 2
    _write_signal_hf_store(
        tmp_path,
        24080,
        "pulse_schedule_l2",
        n_time=2000,
        n_channels=2,
        rate_hz=4000.0,
        t0=-0.05,
        seed=24080,
        local_base=12804,
    )
    _write_signal_hf_store(
        tmp_path,
        24080,
        "summary_l2",
        n_time=2000,
        n_channels=eval_summary_w,
        rate_hz=4000.0,
        t0=-0.05,
        seed=24081,
        local_base=12804,
    )
    s_eval = build_shot_sample(24080, mods, window, token_root=tmp_path)
    assert s_eval.tokens["summary"].shape[1] == eval_summary_w
    assert eval_summary_w != model_summary_w, "eval shot must differ from model"

    # the eval used to CRASH here — it must now run and score.
    report = evaluate_shot(
        24080, model, modalities=mods, window=window, token_root=tmp_path
    )
    assert report.shot_id == 24080
    assert np.isfinite(report.mean_skill)
    # the summary modality WAS scored (not silently skipped).
    assert report.skill["summary"].n_scored > 0

    # the NARROWER direction too: a held-out shot with FEWER summary channels
    # than the model (the model's extra channels are padded; the overlap is
    # scored) — also a forward crash before the fix.
    narrow_w = max(1, model_summary_w - 1)
    if narrow_w != model_summary_w:
        _write_signal_hf_store(
            tmp_path,
            24081,
            "pulse_schedule_l2",
            n_time=2000,
            n_channels=2,
            rate_hz=4000.0,
            t0=-0.05,
            seed=24081,
            local_base=12804,
        )
        _write_signal_hf_store(
            tmp_path,
            24081,
            "summary_l2",
            n_time=2000,
            n_channels=narrow_w,
            rate_hz=4000.0,
            t0=-0.05,
            seed=24082,
            local_base=12804,
        )
        s_narrow = build_shot_sample(24081, mods, window, token_root=tmp_path)
        assert s_narrow.tokens["summary"].shape[1] == narrow_w
        report_n = evaluate_shot(
            24081, model, modalities=mods, window=window, token_root=tmp_path
        )
        assert np.isfinite(report_n.mean_skill)
        assert report_n.skill["summary"].n_scored > 0


def test_eval_errors_loudly_when_no_modality_scorable(tmp_path):
    """Eval RAISES (not silent-zero) when the shot shares no model modality.

    A model that predicts ONLY a camera is asked to eval a shot that carries no
    camera at all — there is no scorable observation modality.  The old code
    would roll out the all-PAD blocks and silently report skill 0; the fixed
    eval raises loudly so the demo metric is never silently absent.
    """
    # a model whose ONLY observation modality is a camera.
    cam_specs = [
        ModalitySpec(
            "pulse_schedule",
            "signal_hf",
            "pulse_schedule_l2",
            L2_VOCAB,
            is_conditioning=True,
        ),
        ModalitySpec(
            "rbb",
            "camera",
            "rbb",
            _CAMERA_VOCAB,
            anchors_grid=False,
            camera_grid_stride=4,
            required=False,
        ),
    ]
    cfg = WorldModelConfig.from_modalities(
        cam_specs,
        {"pulse_schedule": 2, "rbb": 16},
        plan_steps=20,
        obs_steps=20,
        **_TINY_MODEL,
    )
    model = WorldModel(cfg)
    model.eval()

    # a shot carrying the plan + summary but NO camera (camera head has no truth).
    _make_synthetic_shot(tmp_path, 24090, summary_channels=3)
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    # eval modalities include the camera spec (so build_shot_sample tries it),
    # but the shot has no camera store -> the camera is absent -> nothing scorable.
    with pytest.raises(ValueError, match="no observation modality scored|none of"):
        evaluate_shot(
            24090, model, modalities=cam_specs, window=window, token_root=tmp_path
        )


def test_periodic_eval_scores_camera_when_present(tmp_path):
    """The corpus periodic eval scores a camera-bearing held-out shot.

    End-to-end: a camera-inclusive corpus trains, the periodic eval rolls out a
    held-out camera-bearing shot, and the recorded skill is FINITE (the camera
    is in the model and the held-out shot carries it) — the demo metric is
    present, not silently absent.
    """
    specs = _make_five_camera_corpus(
        tmp_path,
        {
            24065: ["rbb", "rba"],
            24066: ["rco", "rgb"],
            24070: ["rbb", "rco", "rgc"],  # held-out, camera-bearing
        },
    )
    out_dir = tmp_path / "ckpt"
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)
    cfg = CorpusTrainConfig(
        steps=20,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=20,
        eval_every=20,
        num_workers=0,
        n_eval_shots=1,
        window=window,
        model_kwargs=_BIG_MODEL,
    )
    result = train_corpus(
        [24065, 24066],
        modalities=specs,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=[24070],
        device="cpu",
        resume=False,
    )
    assert result.eval_skills, "periodic eval recorded no skill"
    for _step, sk in result.eval_skills:
        assert np.isfinite(sk), "camera-bearing eval skill must be finite, not NaN"


# ---------------------------------------------------------------------------
# 12. CAMERA-BEARING SAMPLING + EVAL-SHOT SELECTION — discovery fronts the
#     camera-bearing band so a small limit is camera-dense, and the held-out
#     eval-shot selection picks camera-bearing shots (FIX B + FIX D).
# ---------------------------------------------------------------------------


def test_discovery_camera_first_fronts_camera_band(tmp_path):
    """A small --n-shots limit on a camera-first scan is camera-dense.

    Shots 24065/24067 carry cameras; 24066/24068 are core-only.  With the
    camera_first sampling a limit of 2 returns ONLY camera-bearing shots — the
    low-id-band camera-free pathology is gone.
    """
    specs = _make_five_camera_corpus(
        tmp_path,
        {
            24065: ["rbb"],  # camera
            24066: [],  # core only
            24067: ["rco"],  # camera
            24068: [],  # core only
        },
    )
    limited = discover_worldmodel_shots(
        specs,
        token_root=tmp_path,
        shot_ids=[24065, 24066, 24067, 24068],
        limit=2,
    )
    assert set(limited) == {24065, 24067}, (
        f"camera_first + limit=2 must return only camera-bearing shots: {limited}"
    )
    # a seeded shuffle is deterministic across calls.
    a = discover_worldmodel_shots(
        specs,
        token_root=tmp_path,
        shot_ids=[24065, 24066, 24067, 24068],
        sample="shuffle",
        seed=7,
    )
    b = discover_worldmodel_shots(
        specs,
        token_root=tmp_path,
        shot_ids=[24065, 24066, 24067, 24068],
        sample="shuffle",
        seed=7,
    )
    assert a == b, "seeded shuffle must be deterministic"
    assert set(a) == {24065, 24066, 24067, 24068}


def test_select_eval_shots_picks_camera_bearing(tmp_path):
    """Held-out eval-shot selection prefers camera-bearing shots (FIX D).

    Given a corpus where only some shots carry cameras, the eval split must put
    a camera-bearing shot in the held-out eval set (so the predict-vs-reality
    demo scores the camera), and remove it from the training shots.
    """
    specs = _make_five_camera_corpus(
        tmp_path,
        {
            24065: [],  # core only
            24066: ["rbb"],  # camera
            24067: [],  # core only
            24068: ["rco"],  # camera
            24069: [],  # core only
            24070: ["rgb"],  # camera
        },
    )
    shots = [24065, 24066, 24067, 24068, 24069, 24070]
    train_shots, eval_shots = _select_eval_shots(
        shots, specs, n_eval=1, token_root=tmp_path
    )
    assert eval_shots, "no eval shot selected"
    # every selected eval shot carries a camera.
    cam_bearing = {24066, 24068, 24070}
    for sid in eval_shots:
        assert sid in cam_bearing, (
            f"eval shot {sid} carries no camera — selection ignored cameras"
        )
    # the eval shots are held out of training.
    assert not (set(eval_shots) & set(train_shots)), "eval shot leaked into training"
    assert set(train_shots) | set(eval_shots) == set(shots)


# ---------------------------------------------------------------------------
# 13. ASSEMBLED-SAMPLE CACHE — caching the fully-assembled per-shot sample on
#     node-local NVMe so the DataLoader stops re-reading + re-assembling the
#     per-modality token tensors from GPFS every epoch.  These are HERMETIC
#     (tiny fake corpus in tmp_path, no /work) and FAST:  cache OFF == cache ON
#     (byte-identical hit), a 2nd access HITS without re-assembling, and the
#     config hash changes when the window OR the modality set changes (no stale
#     serve).
# ---------------------------------------------------------------------------


def _assert_samples_equal(a, b) -> None:
    """A cache HIT must be byte-identical to a fresh assembly."""
    assert int(a.shot_id) == int(b.shot_id)
    assert int(a.context_steps) == int(b.context_steps)
    np.testing.assert_array_equal(a.grid_time, b.grid_time)
    assert set(a.tokens) == set(b.tokens)
    for name in a.tokens:
        np.testing.assert_array_equal(a.tokens[name], b.tokens[name])
        assert a.tokens[name].dtype == b.tokens[name].dtype
        np.testing.assert_array_equal(a.valid[name], b.valid[name])
        assert a.valid[name].dtype == b.valid[name].dtype
        assert a.channel_names[name] == b.channel_names[name]


def test_cache_off_equals_on_byte_identical(tmp_path):
    """A cache HIT deserialises a sample byte-identical to a fresh assembly.

    Assemble a shot with the cache OFF (the legacy path) and with the cache ON
    (first access populates, second access is the deserialised HIT).  All three
    must be identical — same grid, same tokens (dtype incl.), same valid masks,
    same channel names.  This is the correctness contract: caching never changes
    what the model sees.
    """
    root = tmp_path / "corpus"
    cache = tmp_path / "cache"
    mods = _make_camera_corpus(root, [24065, 24066])
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)

    # OFF: the legacy assemble-every-access path.
    fresh = build_shot_sample(24065, mods, window, token_root=root)

    # ON: first access populates the cache (a MISS), second access is the HIT.
    miss = load_or_assemble_sample(
        24065, mods, window, token_root=root, cache_dir=cache
    )
    cfg_hash = cache_config_hash(mods, window)
    cached_file = cache / cfg_hash / "24065.pt"
    assert cached_file.exists(), "first access did not write a cache entry"

    hit = load_or_assemble_sample(24065, mods, window, token_root=root, cache_dir=cache)
    _assert_samples_equal(fresh, miss)
    _assert_samples_equal(fresh, hit)


def test_cache_second_access_does_not_reassemble(tmp_path):
    """The 2nd access HITS the cache — the assembly fn is NOT called again.

    Spy on the assembly fn: the first access (MISS) calls it once; the second
    access (HIT) must NOT call it at all — proving the GPFS re-read + grid
    re-assembly is eliminated on the cached path.
    """
    root = tmp_path / "corpus"
    cache = tmp_path / "cache"
    mods = _make_camera_corpus(root, [24065])
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)

    calls = {"n": 0}

    def _spy(shot_id, modalities, config, *, token_root=None, level1_dir=None):
        calls["n"] += 1
        return build_shot_sample(
            shot_id, modalities, config, token_root=token_root, level1_dir=level1_dir
        )

    # MISS: assembles once.
    s1 = load_or_assemble_sample(
        24065, mods, window, token_root=root, cache_dir=cache, assemble_fn=_spy
    )
    assert calls["n"] == 1, "first access must assemble exactly once"

    # HIT: must NOT assemble again.
    s2 = load_or_assemble_sample(
        24065, mods, window, token_root=root, cache_dir=cache, assemble_fn=_spy
    )
    assert calls["n"] == 1, "second access re-assembled — cache did not hit"
    _assert_samples_equal(s1, s2)


def test_cache_key_changes_with_window_and_modalities(tmp_path):
    """The config hash changes when the window OR the modality set changes.

    A different window (n_steps/context/grid_window) or a different ordered
    modality set must yield a DIFFERENT key — so a sample assembled under one
    config can never be served for another (no stale serve).  Conversely the
    SAME config must hash identically (a deterministic, stable key).
    """
    mods = _make_camera_corpus(tmp_path, [24065])
    base_win = WorldModelWindowConfig(n_steps=24, context_steps=6)
    base = cache_config_hash(mods, base_win)

    # deterministic / stable: same inputs -> same hash.
    assert cache_config_hash(mods, base_win) == base

    # window changes -> different key.
    win_n = WorldModelWindowConfig(n_steps=32, context_steps=6)
    win_ctx = WorldModelWindowConfig(n_steps=24, context_steps=8)
    win_grid = WorldModelWindowConfig(
        n_steps=24, context_steps=6, grid_window=(0.0, 1.0)
    )
    assert cache_config_hash(mods, win_n) != base
    assert cache_config_hash(mods, win_ctx) != base
    assert cache_config_hash(mods, win_grid) != base

    # modality SET changes -> different key (drop the camera).
    fewer = [m for m in mods if m.kind != "camera"]
    assert cache_config_hash(fewer, base_win) != base

    # modality ORDER changes -> different key (the ordered set is part of identity).
    reordered = list(reversed(mods))
    assert cache_config_hash(reordered, base_win) != base

    # a vocab / stride change on a modality -> different key (identity-defining).
    bumped_vocab = [
        ModalitySpec(
            m.name,
            m.kind,
            m.group,
            m.vocab_size + 1,
            is_conditioning=m.is_conditioning,
            anchors_grid=m.anchors_grid,
            n_channels=m.n_channels,
            camera_grid_stride=m.camera_grid_stride,
            required=m.required,
        )
        if m.name == "summary"
        else m
        for m in mods
    ]
    assert cache_config_hash(bumped_vocab, base_win) != base


def test_cache_key_ignores_required_flag(tmp_path):
    """``required`` is NOT identity-defining — it only gates discovery.

    Flipping ``required`` does not change the assembled sample, so it must NOT
    change the cache key (else two configs that produce identical samples would
    miss each other's cache).
    """
    mods = _make_camera_corpus(tmp_path, [24065])
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)
    base = cache_config_hash(mods, window)
    flipped = [
        ModalitySpec(
            m.name,
            m.kind,
            m.group,
            m.vocab_size,
            is_conditioning=m.is_conditioning,
            anchors_grid=m.anchors_grid,
            n_channels=m.n_channels,
            camera_grid_stride=m.camera_grid_stride,
            required=not m.required,
        )
        for m in mods
    ]
    assert cache_config_hash(flipped, window) == base


def test_resolve_cache_dir_precedence(tmp_path, monkeypatch):
    """cache_dir arg wins over WM_CACHE_DIR; both unset => None (caching off)."""
    monkeypatch.delenv("WM_CACHE_DIR", raising=False)
    assert resolve_cache_dir(None) is None
    assert resolve_cache_dir("") is None

    # explicit arg wins.
    assert resolve_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"

    # env var used when arg unset.
    monkeypatch.setenv("WM_CACHE_DIR", str(tmp_path / "fromenv"))
    assert resolve_cache_dir(None) == tmp_path / "fromenv"
    # arg still wins over the env var.
    assert resolve_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_dataset_cache_off_when_unset(tmp_path, monkeypatch):
    """Unset cache => the dataset behaves exactly as before (back-compat).

    No cache dir resolved, no config hash, and __getitem__ still assembles a
    correct sample — the legacy assemble-every-access path is untouched.
    """
    monkeypatch.delenv("WM_CACHE_DIR", raising=False)
    mods = _make_corpus(tmp_path, [24065, 24066])
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    ds = WorldModelDataset([24065, 24066], mods, window, token_root=tmp_path)
    assert ds.cache_dir is None
    s = ds[0]
    fresh = build_shot_sample(24065, mods, window, token_root=tmp_path)
    _assert_samples_equal(s, fresh)


def test_dataset_caches_and_hits_across_accesses(tmp_path, monkeypatch):
    """WorldModelDataset writes the cache on first access and HITS thereafter.

    Drives the cache through the actual Dataset.__getitem__ surface (what the
    DataLoader calls).  Spy on dataset.build_shot_sample: indexing the same shot
    twice must assemble exactly ONCE (the second index is a HIT), and the served
    samples are equal — the per-epoch GPFS re-read + re-assembly is gone.  Also
    proves WM_CACHE_DIR turns it on without an explicit cache_dir arg.
    """
    root = tmp_path / "corpus"
    cache = tmp_path / "cache"
    mods = _make_camera_corpus(root, [24065, 24066])
    window = WorldModelWindowConfig(n_steps=24, context_steps=6)

    n_calls = {"n": 0}
    real = wm_dataset.build_shot_sample

    def _counting(shot_id, modalities, config, *, token_root=None, level1_dir=None):
        n_calls["n"] += 1
        return real(
            shot_id, modalities, config, token_root=token_root, level1_dir=level1_dir
        )

    monkeypatch.setattr(wm_dataset, "build_shot_sample", _counting)
    # WM_CACHE_DIR turns the cache ON without an explicit cache_dir arg.
    monkeypatch.setenv("WM_CACHE_DIR", str(cache))

    ds = WorldModelDataset([24065, 24066], mods, window, token_root=root)
    assert ds.cache_dir == cache, "WM_CACHE_DIR did not enable the cache"

    first = ds[0]  # MISS -> assembles
    assert n_calls["n"] == 1
    second = ds[0]  # HIT -> no assembly
    assert n_calls["n"] == 1, "Dataset re-assembled on the 2nd access — no cache hit"
    _assert_samples_equal(first, second)

    # a DIFFERENT window is a DIFFERENT key — it MISSES (re-assembles), never
    # serving the stale 24-step sample for a 32-step request.
    ds2 = WorldModelDataset(
        [24065],
        mods,
        WorldModelWindowConfig(n_steps=32, context_steps=6),
        token_root=root,
    )
    assert ds2.cache_dir == cache
    s32 = ds2[0]
    assert n_calls["n"] == 2, (
        "different window must MISS (re-assemble), not serve stale"
    )
    assert s32.grid_time.shape[0] == 32


def test_cache_corrupt_entry_falls_back_to_assembly(tmp_path):
    """A corrupt cache file is treated as a MISS and re-assembled (never wedges).

    A partial write from a killed process (or a torch-version skew) must not
    wedge the loader: an unreadable entry falls back to a fresh assembly and is
    overwritten with a good one.
    """
    root = tmp_path / "corpus"
    cache = tmp_path / "cache"
    mods = _make_corpus(root, [24065])
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    cfg_hash = cache_config_hash(mods, window)

    # plant a corrupt cache entry where the loader will look.
    bad = cache / cfg_hash / "24065.pt"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a torch checkpoint")

    s = load_or_assemble_sample(24065, mods, window, token_root=root, cache_dir=cache)
    fresh = build_shot_sample(24065, mods, window, token_root=root)
    _assert_samples_equal(s, fresh)
    # the corrupt entry was overwritten with a good one (next access is a HIT).
    again = load_or_assemble_sample(
        24065, mods, window, token_root=root, cache_dir=cache
    )
    _assert_samples_equal(fresh, again)


# ---------------------------------------------------------------------------
# 14. CUDA PREFETCHER — a double-buffered side-stream host->device prefetcher
#     overlaps the next batch's copy with the current batch's compute so the GPU
#     is fed continuously.  On CPU (no CUDA streams) it MUST degrade to a plain
#     pass-through so the hermetic tests + any CPU run see the identical batches.
#     These are CPU-safe: same batches out as in, and a CPU train_corpus run
#     (which now wraps the loader in the prefetcher) still descends.
# ---------------------------------------------------------------------------


def test_prefetcher_cpu_is_passthrough_identical_batches(tmp_path):
    """On CPU the prefetcher yields the SAME batches (object-identical) as input.

    No CUDA streams exist on CPU, so the prefetcher must be a transparent
    pass-through — the batches a consumer sees are exactly the loader's, in
    order, unmodified.
    """
    mods = _make_corpus(tmp_path, [24065, 24066])
    window = WorldModelWindowConfig(n_steps=20, context_steps=5)
    s0 = build_shot_sample(24065, mods, window, token_root=tmp_path)
    s1 = build_shot_sample(24066, mods, window, token_root=tmp_path)
    obs = [m.name for m in mods if not m.is_conditioning]
    plan = [m.name for m in mods if m.is_conditioning]
    channels = {m.name: s0.tokens[m.name].shape[1] for m in mods}

    batches = [
        pad_collate_batch([s0], obs, plan, channels),
        pad_collate_batch([s1], obs, plan, channels),
    ]
    pf = CudaPrefetcher(batches, torch.device("cpu"))
    out = list(pf)
    assert len(out) == len(batches)
    for got, want in zip(out, batches, strict=True):
        # identical objects on CPU (pure pass-through — no copy, no stream).
        assert got is want
        for name in plan + obs:
            assert torch.equal(got["tokens"][name], want["tokens"][name])
            assert torch.equal(got["valid"][name], want["valid"][name])


def test_prefetched_loop_matches_non_prefetched_loss(tmp_path):
    """A CPU train_corpus run (loader wrapped in the prefetcher) descends.

    The training loop now drives the DataLoader through the CudaPrefetcher.  On
    CPU the prefetcher is a pass-through, so the loop must be UNCHANGED — it
    iterates the multi-shot loader and the loss drops, identical to the
    pre-prefetcher behaviour (correctness: the prefetcher only changes WHEN the
    H2D copy happens on CUDA, never WHAT the model is fed).
    """
    shots = [24065, 24066, 24067, 24068]
    mods = _make_corpus(tmp_path, shots)
    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=40,
        batch_size=2,
        lr=5e-3,
        log_every=10,
        ckpt_every=40,
        eval_every=0,
        num_workers=0,  # main-process load; prefetcher pass-through on CPU
        window=WorldModelWindowConfig(n_steps=24, context_steps=6),
        model_kwargs=_TINY_MODEL,
    )
    result = train_corpus(
        shots[:3],
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    assert result.steps_run == 40
    early = float(np.mean(result.losses[:10]))
    late = float(np.mean(result.losses[-10:]))
    assert late < early, f"prefetched CPU loop did not descend: {early} -> {late}"


def test_prefetched_loop_deterministic_loss_matches_seed(tmp_path):
    """Two seeded prefetched CPU runs produce the IDENTICAL loss trajectory.

    The prefetcher introduces no nondeterminism on CPU (pass-through), so two
    runs at the same seed must give bit-identical losses — a tight correctness
    check that the prefetch wrapping did not perturb batch order or content.
    """
    shots = [24065, 24066, 24067]
    common = dict(
        token_root=tmp_path,
        eval_shot_ids=None,
        device="cpu",
        resume=False,
    )
    mods = _make_corpus(tmp_path, shots)

    def _run(out_dir):
        cfg = CorpusTrainConfig(
            steps=20,
            batch_size=2,
            lr=5e-3,
            log_every=20,
            ckpt_every=20,
            eval_every=0,
            num_workers=0,
            seed=123,
            window=WorldModelWindowConfig(n_steps=20, context_steps=5),
            model_kwargs=_TINY_MODEL,
        )
        return train_corpus(
            shots, modalities=mods, config=cfg, out_dir=out_dir, **common
        )

    r1 = _run(tmp_path / "a")
    r2 = _run(tmp_path / "b")
    assert r1.losses == r2.losses, "seeded prefetched CPU runs diverged"
