"""End-to-end skeleton tests for the plan-conditioned world model.

Covers the four §6 pieces:

* the boundary guard — the input loader REFUSES a ``TARGET_ROOT`` path
  (synthetic + real-reader), so an eval-only target can never be ingested;
* the dataset assembly — synthetic stores assemble onto a common grid with the
  context/target split + per-step coverage mask;
* the model — forward pass shapes, param-count / context-length contract;
* the train overfit — a tiny model overfits a synthetic shot, loss drops
  substantially (the end-to-end wiring proof);
* the eval — autoregressive rollout reports a skill-vs-persistence number.

The CPU-tiny tests build SYNTHETIC token stores so the suite is hermetic and
fast (no GPFS dependency); a single opt-in real-shot smoke runs against the
on-disk corpus when it is reachable, exercising the real readers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imas_ambix.tokenizer.registry import CONTROL_RANGE, L2_BLOCK_VOCAB
from imas_ambix.tokenizer.store_targets import (
    SIGNALS_HF_GENERATION,
    assert_not_target_path,
)
from imas_ambix.worldmodel.dataset import (
    ModalitySpec,
    WorldModelDataset,
    WorldModelWindowConfig,
    build_shot_sample,
    discover_worldmodel_shots,
)
from imas_ambix.worldmodel.eval import evaluate_shot, load_target_reference
from imas_ambix.worldmodel.train import (
    TrainConfig,
    build_model_for_samples,
    next_token_nll,
    overfit,
)

L2_VOCAB = L2_BLOCK_VOCAB + 1
CTRL = CONTROL_RANGE[1]  # 4 — the HF/L2 block base


# ---------------------------------------------------------------------------
# Synthetic store fixtures
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

    Tokens are deterministic per (channel) so the model can overfit them.  The
    on-disk global ids are ``local + local_base`` so the loader's rebasing is
    exercised; the L2 ``global_id_range`` metadata is recorded for ``*_l2``.
    """
    import zarr

    rng = np.random.default_rng(seed)
    dt = 1.0 / rate_hz
    token_time = t0 + np.arange(n_time, dtype=np.float64) * dt
    # a simple periodic local-id pattern per channel (overfittable)
    local = np.zeros((n_time, n_channels), dtype=np.int64)
    for c in range(n_channels):
        period = 3 + c % 5
        local[:, c] = (np.arange(n_time) % period) + 1 + (c % 7)
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
    _ = rng  # determinism noted; pattern is RNG-independent on purpose


def _make_synthetic_corpus(
    root: Path, shot_id: int, *, seed: int = 0
) -> list[ModalitySpec]:
    """Write a tractable synthetic shot (plan + two measured groups + camera).

    Returns the matching modality specs.
    """
    import zarr

    # L2 light-path groups on a common 4 kHz grid spanning [-0.05, 0.45]
    # (2000 samples) so the camera window sits inside it.
    _write_signal_hf_store(
        root,
        shot_id,
        "pulse_schedule_l2",
        n_time=2000,
        n_channels=2,
        rate_hz=4000.0,
        t0=-0.05,
        seed=seed,
        local_base=12804,
    )
    _write_signal_hf_store(
        root,
        shot_id,
        "summary_l2",
        n_time=2000,
        n_channels=3,
        rate_hz=4000.0,
        t0=-0.05,
        seed=seed + 1,
        local_base=12804,
    )
    # camera: a tiny synthetic frame token store on [0, ~0.13] (inside the
    # L2 window).  No level-1 store is written, so the loader uses its uniform
    # fallback Δt = 1/600 s — we choose n_frames so the span overlaps.
    cam_path = root / "v1" / "frames" / str(shot_id) / "rbb.zarr"
    cam_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = 80
    grid = np.zeros((n_frames, 16, 16), dtype=np.int32)
    for c in range(16):
        grid[:, c, :] = (np.arange(n_frames) % (2 + c % 4))[:, None] + 5
    cstore = zarr.open_group(str(cam_path), mode="w")
    cstore.create_array("tokens", data=grid)
    cstore.attrs.update(
        {
            "shot_id": int(shot_id),
            "camera": "rbb",
            "vocab_version": "v1",
            "tokenizer_name": "openmagvit2",
            "shape": list(grid.shape),
            "metadata": json.dumps({"temporal_compression": 1}),
        }
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
        ModalitySpec("camera", "camera", "rbb", 1 << 18, camera_grid_stride=4),
    ]


# ---------------------------------------------------------------------------
# 1. Boundary guard — the input loader REFUSES a TARGET_ROOT path
# ---------------------------------------------------------------------------


def test_input_loader_refuses_target_root_path(tmp_path):
    """A token_root resolving under TARGET_ROOT is hard-refused at load time."""
    from imas_ambix.data import paths as paths_mod

    # point TARGET_ROOT at a temp dir and ask the loader to use it as token_root
    target_root = tmp_path / "mast-targets"
    (target_root / "24065").mkdir(parents=True)
    import imas_ambix.tokenizer.store_targets as st

    orig = st.TARGET_ROOT
    try:
        st.TARGET_ROOT = target_root
        paths_mod.TARGET_ROOT = target_root
        # the guard itself refuses a path under the target root
        with pytest.raises(ValueError, match="TARGET_ROOT"):
            assert_not_target_path(target_root / "24065" / "equilibrium.zarr")
        # and the dataset's discovery, handed the target root as token_root,
        # refuses it (the input enumerator can never reach a target)
        mods = [ModalitySpec("summary", "signal_hf", "summary_l2", L2_VOCAB)]
        with pytest.raises(ValueError, match="TARGET_ROOT"):
            build_shot_sample(
                24065,
                mods,
                WorldModelWindowConfig(n_steps=8, context_steps=2),
                token_root=target_root,
            )
    finally:
        st.TARGET_ROOT = orig
        paths_mod.TARGET_ROOT = orig


def test_camera_loader_refuses_target_root(tmp_path):
    """The camera path builder also routes through the target guard."""
    import imas_ambix.tokenizer.store_targets as st
    from imas_ambix.camdyn.dataset import frames_token_path

    target_root = tmp_path / "mast-targets"
    target_root.mkdir()
    orig = st.TARGET_ROOT
    try:
        st.TARGET_ROOT = target_root
        with pytest.raises(ValueError, match="TARGET_ROOT"):
            frames_token_path(24065, "rbb", token_root=target_root)
    finally:
        st.TARGET_ROOT = orig


# ---------------------------------------------------------------------------
# 2. Dataset assembly — common grid + context/target split + mask
# ---------------------------------------------------------------------------


def test_dataset_assembles_common_grid(tmp_path):
    mods = _make_synthetic_corpus(tmp_path, 24065)
    cfg = WorldModelWindowConfig(n_steps=32, context_steps=8)
    sample = build_shot_sample(24065, mods, cfg, token_root=tmp_path)

    assert sample.n_steps == 32
    assert sample.context_steps == 8
    assert set(sample.tokens) == {"pulse_schedule", "summary", "camera"}
    # every modality is on the SAME grid
    for name in sample.tokens:
        assert sample.tokens[name].shape[0] == 32
        assert sample.valid[name].shape == sample.tokens[name].shape
    # local rebasing: L2 tokens are small local ids, not the 12804+ globals
    assert sample.tokens["summary"].max() < L2_VOCAB
    # camera stride 4 over 16x16 -> 4x4 = 16 channels
    assert sample.tokens["camera"].shape[1] == 16
    # grid times are uniform and monotone
    dt = np.diff(sample.grid_time)
    assert np.all(dt > 0)
    assert np.allclose(dt, dt[0])


def test_dataset_discovery_and_map_style(tmp_path):
    mods = _make_synthetic_corpus(tmp_path, 24065)
    _make_synthetic_corpus(tmp_path, 24066, seed=10)
    found = discover_worldmodel_shots(
        mods, token_root=tmp_path, shot_ids=[24065, 24066]
    )
    assert found == [24065, 24066]
    ds = WorldModelDataset(
        found,
        mods,
        WorldModelWindowConfig(n_steps=16, context_steps=4),
        token_root=tmp_path,
    )
    assert len(ds) == 2
    s0 = ds[0]
    assert s0.shot_id == 24065
    assert s0.n_steps == 16


# ---------------------------------------------------------------------------
# 3. Model — forward shapes + documented contract
# ---------------------------------------------------------------------------


def test_model_forward_shapes_and_contract(tmp_path):
    mods = _make_synthetic_corpus(tmp_path, 24065)
    cfg = WorldModelWindowConfig(n_steps=24, context_steps=6)
    sample = build_shot_sample(24065, mods, cfg, token_root=tmp_path)
    model = build_model_for_samples(
        [sample], mods, cfg, d_model=32, n_layers=2, n_heads=2
    )

    # documented contract: live param count + context length
    assert model.num_parameters() > 0
    assert model.context_length() == cfg.n_steps * 2  # plan_steps + obs_steps

    from imas_ambix.worldmodel.train import collate_samples

    plan = [m.name for m in mods if m.is_conditioning]
    obs = [m.name for m in mods if not m.is_conditioning]
    batch = collate_samples([sample], obs, plan)
    out = model(batch)
    # logits only for observation modalities, shaped (B, T, C, V)
    assert set(out.logits) == set(obs)
    for name in obs:
        lg = out.logits[name]
        assert lg.shape[0] == 1
        assert lg.shape[1] == cfg.n_steps
        assert lg.shape[2] == sample.tokens[name].shape[1]
    # the loss is finite and backprops
    loss = next_token_nll(out.logits, batch, obs)
    assert np.isfinite(float(loss.detach()))
    loss.backward()


# ---------------------------------------------------------------------------
# 4. Train overfit — the end-to-end wiring proof (loss drops substantially)
# ---------------------------------------------------------------------------


def test_overfit_loss_drops_substantially(tmp_path):
    mods = _make_synthetic_corpus(tmp_path, 24065)
    cfg = TrainConfig(
        steps=250,
        lr=5e-3,
        window=WorldModelWindowConfig(n_steps=32, context_steps=8),
        model_kwargs={"d_model": 48, "n_layers": 2, "n_heads": 2, "d_ff": 96},
    )
    result = overfit([24065], modalities=mods, config=cfg, token_root=tmp_path)

    assert len(result.losses) == 250
    assert np.isfinite(result.initial_loss)
    assert np.isfinite(result.final_loss)
    # END-TO-END PROOF: the loss must drop substantially (model overfits).
    assert result.final_loss < 0.5 * result.initial_loss, (
        f"loss did not drop enough: {result.initial_loss:.4f} "
        f"-> {result.final_loss:.4f}"
    )
    # param-count / context-length contract is surfaced
    assert result.n_parameters > 0
    assert result.context_length == 64


# ---------------------------------------------------------------------------
# 5. Eval — autoregressive rollout reports a skill-vs-persistence number
# ---------------------------------------------------------------------------


def test_eval_rollout_reports_skill(tmp_path):
    mods = _make_synthetic_corpus(tmp_path, 24065)
    cfg = WorldModelWindowConfig(n_steps=32, context_steps=8)
    sample = build_shot_sample(24065, mods, cfg, token_root=tmp_path)
    model = build_model_for_samples(
        [sample], mods, cfg, d_model=32, n_layers=2, n_heads=2
    )

    report = evaluate_shot(
        24065, model, modalities=mods, window=cfg, token_root=tmp_path
    )
    assert report.shot_id == 24065
    assert report.n_steps == 32
    assert report.context_steps == 8
    # skill is reported for every observation modality
    obs = [m.name for m in mods if not m.is_conditioning]
    assert set(report.skill) == set(obs)
    for s in report.skill.values():
        assert s.n_scored >= 0
        assert np.isfinite(s.skill)
    # the summary renders (the predict-vs-reality report)
    text = report.summary()
    assert "token-skill" in text
    assert isinstance(report.mean_skill, float)


def test_eval_target_reference_is_eval_only(tmp_path):
    """The target reference loads via the eval-only reader, never the input path."""
    # no target store on disk for this synthetic shot -> empty reference, but
    # the eval loop still runs
    ref = load_target_reference(99999, target_root=tmp_path / "mast-targets")
    assert ref.shot_id == 99999
    assert ref.available is False


# ---------------------------------------------------------------------------
# Optional real-shot smoke (skipped when the corpus is unreachable)
# ---------------------------------------------------------------------------


def _real_corpus_shots(n: int = 2) -> list[int]:
    from imas_ambix.worldmodel.dataset import default_modalities

    try:
        return discover_worldmodel_shots(default_modalities(), limit=n)
    except Exception:  # noqa: BLE001 — corpus unreachable
        return []


@pytest.mark.skipif(
    not _real_corpus_shots(1), reason="on-disk token corpus not reachable"
)
def test_real_shot_overfit_smoke():
    """Overfit a real on-disk shot a few steps — loss drops (real readers)."""
    from imas_ambix.worldmodel.dataset import default_modalities

    shots = _real_corpus_shots(1)
    mods = default_modalities()
    cfg = TrainConfig(
        steps=60,
        lr=5e-3,
        window=WorldModelWindowConfig(n_steps=32, context_steps=8),
        model_kwargs={"d_model": 32, "n_layers": 2, "n_heads": 2, "d_ff": 64},
    )
    result = overfit(shots[:1], modalities=mods, config=cfg)
    assert result.final_loss < result.initial_loss
