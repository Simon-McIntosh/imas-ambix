"""Unit tests for the held-out RSSM controllability eval (CPU, decoder-free).

These exercise the eval's pure / token-space paths so the harness is green BEFORE
a trained RSSM checkpoint exists (the orchestrator runs the real GPU eval after
the overnight train):

* ``load_rssm_from_checkpoint`` reconstructs the config + loads the weights from a
  saved ``{model_config, model_state_dict, extra.stream_names}`` payload;
* ``_rssm_full_rollout`` returns a ``(T, S)`` window whose context frames are the
  GT context and whose forecast frames come from the RSSM prior — and a DIFFERENT
  plan moves the forecast (controllable by construction);
* ``multi_shot_delta_nm_rssm`` (token-space, ``decode=False``, monkeypatched
  ``_assemble_heldout``) returns the IDENTICAL robust summary schema as the token
  gate (mean/median normalised ratio, pass-fraction, bootstrap CI, variance
  decomposition, per-shot ratios) with finite values;
* ``diagnostic_match_rssm`` returns finite per-stream accuracy + CE and an honest
  ``diagnostics_generated`` flag.

The decoded-pixel + GIF paths need the GPU VQ stack and are run on a compute node
by the orchestrator after the train.
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.worldmodel.rssm import RSSMConfig, RSSMWorldModel, SignalStreamSpec
from imas_ambix.worldmodel.rssm_eval import (
    _rssm_full_rollout,
    diagnostic_match_rssm,
    load_rssm_from_checkpoint,
    multi_shot_delta_nm_rssm,
    multi_shot_diagnostic_match_rssm,
)

# ---------------------------------------------------------------------------
# tiny model + synthetic held-out sample
# ---------------------------------------------------------------------------


def _tiny_cfg(**kw) -> RSSMConfig:
    base = dict(
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        d_model=32,
        h_dim=48,
        s_dim=8,
        a_dim=16,
        cmd_hidden=32,
        latent_hidden=32,
        decoder_hidden=48,
        beta=1.0,
        free_bits=1.0,
        diagnostic_weight=0.5,
        signal_streams=(
            SignalStreamSpec("gas_injection", vocab=8, channels=2),
            SignalStreamSpec("xma", vocab=8, channels=3),
        ),
        actuator_channels=6,
        masked_command_indices=(0, 1),
    )
    base.update(kw)
    return RSSMConfig(**base)


def _mk_sample(cfg, shot, *, t=6, ctx=3, n_act=4, n_sig=4, seed=0):
    """A synthetic ControllableSpacetimeSample (mirrors test_controllable_eval)."""
    from imas_ambix.worldmodel.actuator_plan import (
        ACTUATOR_CHANNEL_KEYS,
        ActuatorPlan,
        normalise_actuator_values,
    )
    from imas_ambix.worldmodel.controllable_dataset import ControllableSpacetimeSample
    from imas_ambix.worldmodel.spacetime_dataset import SpacetimeSample
    from imas_ambix.worldmodel.spacetime_dataset_v2 import SignalSpacetimeSample

    g = torch.Generator().manual_seed(seed)
    frames = torch.randint(0, cfg.vocab_size, (t, cfg.n_spatial), generator=g).numpy()
    plan = torch.randint(0, 16, (4, 2), generator=g).numpy()
    base = SpacetimeSample(
        shot_id=shot,
        camera="rbb",
        start_frame=0,
        frames=frames,
        plan=plan,
        frame_time=np.linspace(0.0, 0.1, t),
        context_frames=ctx,
    )
    sigs = {}
    for st in cfg.signal_streams:
        # ids in [1, vocab) -> no PAD targets, so every stream scores.
        sigs[st.name] = torch.randint(
            1, st.vocab, (n_sig, st.channels), generator=g
        ).numpy()
    signal = SignalSpacetimeSample(base=base, signals=sigs)
    c = cfg.actuator_channels
    # a VARYING per-channel ramp so plan_variation > 0 (the shot is transient) and
    # the bounded counterfactual has a non-zero range to perturb.
    raw = (
        np.linspace(1e3, 1e5, n_act, dtype=np.float32)[:, None]
        * (1.0 + 0.1 * np.arange(c, dtype=np.float32))[None, :]
    )
    keys = list(ACTUATOR_CHANNEL_KEYS[:c]) + [
        f"c{i}" for i in range(max(0, c - len(ACTUATOR_CHANNEL_KEYS[:c])))
    ]
    plan_act = ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=np.zeros_like(raw),
        channel_keys=keys,
        raw_values=raw,
    )
    return ControllableSpacetimeSample(signal=signal, actuator=plan_act)


# ---------------------------------------------------------------------------
# load_rssm_from_checkpoint
# ---------------------------------------------------------------------------


def _model_config_dict(cfg: RSSMConfig) -> dict:
    """The full RSSMConfig scalar/stream record the trainer saves as model_config.

    Emits EVERY non-default RSSMConfig field so the eval reconstructs a byte-equal
    config (a partial record would default the hidden widths and mismatch the
    state dict — exactly the trainer's contract).
    """
    out = {
        k: getattr(cfg, k)
        for k in RSSMConfig.__dataclass_fields__
        if k not in ("signal_streams", "masked_command_indices")
    }
    out["masked_command_indices"] = list(cfg.masked_command_indices)
    out["signal_streams"] = [
        {"name": s.name, "vocab": s.vocab, "channels": s.channels}
        for s in cfg.signal_streams
    ]
    return out


def test_load_rssm_from_checkpoint_roundtrips(tmp_path):
    cfg = _tiny_cfg()
    model = RSSMWorldModel(cfg)
    payload = {
        "model_config": _model_config_dict(cfg),
        "model_state_dict": model.state_dict(),
        "extra": {"stream_names": [s.name for s in cfg.signal_streams]},
    }
    ckpt = tmp_path / "rssm_best.pt"
    torch.save(payload, ckpt)

    loaded, pl = load_rssm_from_checkpoint(str(ckpt), map_location="cpu")
    assert loaded.config.vocab_size == cfg.vocab_size
    assert loaded.config.actuator_channels == cfg.actuator_channels
    assert loaded.config.masked_command_indices == cfg.masked_command_indices
    assert [s.name for s in loaded.config.signal_streams] == [
        s.name for s in cfg.signal_streams
    ]
    # the weights round-tripped.
    assert torch.allclose(
        loaded.token_embed.weight.detach(), model.token_embed.weight.detach()
    )
    # the head is re-tied to the loaded token embed.
    assert loaded.head.weight.data_ptr() == loaded.token_embed.weight.data_ptr()
    assert pl["extra"]["stream_names"] == [s.name for s in cfg.signal_streams]


def test_load_rssm_strips_ddp_prefix(tmp_path):
    cfg = _tiny_cfg(signal_streams=())
    model = RSSMWorldModel(cfg)
    ddp_state = {f"module.{k}": v for k, v in model.state_dict().items()}
    payload = {"model_config": _model_config_dict(cfg), "model": ddp_state}
    ckpt = tmp_path / "rssm_ddp.pt"
    torch.save(payload, ckpt)
    loaded, _ = load_rssm_from_checkpoint(str(ckpt), map_location="cpu")
    assert torch.allclose(
        loaded.token_embed.weight.detach(), model.token_embed.weight.detach()
    )


# ---------------------------------------------------------------------------
# _rssm_full_rollout: GT context + prior forecast; a plan moves the forecast
# ---------------------------------------------------------------------------


def test_full_rollout_layout_and_controllability():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = RSSMWorldModel(cfg).eval()
    t, ctx = 6, 3
    sample = _mk_sample(cfg, 18502, t=t, ctx=ctx, seed=1)

    true_roll = _rssm_full_rollout(model, sample, sample.actuator, torch.device("cpu"))
    assert true_roll.shape == (t, cfg.n_spatial)
    # the context frames are the GT frames verbatim.
    gt = np.asarray(sample.frames, dtype=np.int64)
    assert np.array_equal(true_roll[:ctx], gt[:ctx])

    # a DIFFERENT plan moves the prior forecast (controllable by construction).
    from imas_ambix.worldmodel.controllable_train import _random_actuator_like

    rng = np.random.default_rng(0)
    rplan = _random_actuator_like(sample.actuator, rng=rng, perturb_scale=0.5)
    rand_roll = _rssm_full_rollout(model, sample, rplan, torch.device("cpu"))
    assert rand_roll.shape == (t, cfg.n_spatial)
    assert np.array_equal(rand_roll[:ctx], gt[:ctx])
    # the forecast (frames >= ctx) responds to the plan — at least one token moves
    # on this untrained model (argmax can tie, so allow the latent test as backup).
    forecast_moved = not np.array_equal(true_roll[ctx:], rand_roll[ctx:])
    if not forecast_moved:
        # fall back to the continuous latent surface (argmax tie at init).
        act_t = {
            "values": torch.as_tensor(
                sample.actuator.values[None], dtype=torch.float32
            ),
            "missing": torch.as_tensor(
                sample.actuator.missing[None], dtype=torch.float32
            ),
        }
        act_r = {
            "values": torch.as_tensor(rplan.values[None], dtype=torch.float32),
            "missing": torch.as_tensor(rplan.missing[None], dtype=torch.float32),
        }
        ctx_t = torch.as_tensor(gt[:ctx][None], dtype=torch.long)
        with torch.no_grad():
            rt = model.rollout_prior(ctx_t, act_t, t - ctx, sample=False)
            rr = model.rollout_prior(ctx_t, act_r, t - ctx, sample=False)
        assert not torch.allclose(rt.h, rr.h)


# ---------------------------------------------------------------------------
# multi_shot_delta_nm_rssm: token-space, monkeypatched assembly, full schema
# ---------------------------------------------------------------------------


def test_multi_shot_delta_nm_rssm_token_schema(tmp_path, monkeypatch):
    """The token-space RSSM gate returns the IDENTICAL summary schema as the token
    backbone gate, with finite values, on a tiny synthetic cohort."""
    import imas_ambix.worldmodel.rssm_eval as rssm_eval
    from imas_ambix.worldmodel.controllable_eval import EvalConfig

    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = RSSMWorldModel(cfg).eval()

    # a small synthetic cohort: four transient shots assembled on demand.
    cohort = (18502, 18503, 18504, 18505)
    samples = {sid: _mk_sample(cfg, sid, t=6, ctx=3, seed=sid % 97) for sid in cohort}

    def fake_assemble(shot_id, _config, *, camera, token_root):
        return samples[int(shot_id)]

    monkeypatch.setattr(rssm_eval, "_assemble_heldout", fake_assemble, raising=False)
    # also patch the name imported INTO controllable_eval's namespace is unused; the
    # eval imports _assemble_heldout locally from controllable_eval, so patch there.
    import imas_ambix.worldmodel.controllable_eval as ce

    monkeypatch.setattr(ce, "_assemble_heldout", fake_assemble, raising=False)

    eval_cfg = EvalConfig(
        held_out=cohort,
        n_random=4,
        robust_gate=True,
        ratio_threshold=1.5,
        n_bootstrap=200,
        n_within_bootstrap=100,
        chunk=64,
    )
    out_json = tmp_path / "heldout_delta_nm.json"
    summary = multi_shot_delta_nm_rssm(
        model,
        config=eval_cfg,
        camera="rbb",
        token_root=None,
        device="cpu",
        out_json=out_json,
        decode=False,
    )

    # the schema MUST match the token-backbone gate (directly comparable JSON).
    required = {
        "metric",
        "robust_gate",
        "n_samples",
        "n_transient",
        "n_pass",
        "pass_fraction",
        "mean_true_vs_random",
        "mean_random_vs_random_noise_floor",
        "mean_margin",
        "mean_normalised_ratio",
        "median_normalised_ratio",
        "ratio_ci_lo",
        "ratio_ci_hi",
        "per_shot_ratios_sorted",
        "variance_decomposition",
        "n_random",
        "gate_pass",
        "verdict",
    }
    assert required <= set(summary), f"missing keys: {required - set(summary)}"
    assert summary["metric"] == "delta_nm_token_lowerbound_robust"
    assert summary["rollout"] == "rssm_prior"
    assert summary["n_samples"] == len(cohort)
    assert 0.0 <= summary["pass_fraction"] <= 1.0
    assert np.isfinite(summary["mean_normalised_ratio"])
    assert np.isfinite(summary["ratio_ci_lo"]) and np.isfinite(summary["ratio_ci_hi"])
    vd = summary["variance_decomposition"]
    assert {"across_over_within", "interpretation"} <= set(vd)
    assert summary["per_shot_ratios_sorted"] == sorted(
        summary["per_shot_ratios_sorted"]
    )
    assert summary["verdict"] in ("PASS", "FAIL")

    # the JSON written to disk has the per-shot + summary blocks.
    import json

    written = json.loads(out_json.read_text())
    assert "per_shot" in written and "summary" in written
    assert len(written["per_shot"]) == len(cohort)
    for v in written["per_shot"]:
        assert {"shot_id", "ratio", "true_vs_random", "random_vs_random", "passed"} <= (
            set(v)
        )


# ---------------------------------------------------------------------------
# diagnostic_match_rssm
# ---------------------------------------------------------------------------


def test_diagnostic_match_rssm_scores_per_stream():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = RSSMWorldModel(cfg).eval()
    assert model.config.has_diagnostics
    sample = _mk_sample(cfg, 18502, t=6, ctx=3, seed=2)
    res = diagnostic_match_rssm(model, sample, device="cpu", chunk=64)
    assert res["diagnostics_generated"] is True
    assert set(res["per_stream"]) == {s.name for s in cfg.signal_streams}
    for d in res["per_stream"].values():
        assert 0.0 <= d["accuracy"] <= 1.0 and np.isfinite(d["accuracy"])
        assert np.isfinite(d["ce"]) and d["ce"] >= 0.0
        assert d["n"] > 0
    assert 0.0 <= res["mean_accuracy"] <= 1.0 and np.isfinite(res["mean_accuracy"])
    assert np.isfinite(res["mean_ce"])


def test_diagnostic_match_rssm_camera_only_flag():
    cfg = _tiny_cfg(signal_streams=())
    torch.manual_seed(0)
    model = RSSMWorldModel(cfg).eval()
    assert not model.config.has_diagnostics
    sample = _mk_sample(cfg, 18503, t=6, ctx=3, seed=3)
    res = diagnostic_match_rssm(model, sample, device="cpu", chunk=64)
    assert res["diagnostics_generated"] is False
    assert res["per_stream"] == {}
    assert res["mean_accuracy"] == 0.0 and res["mean_ce"] == 0.0


def test_multi_shot_diagnostic_match_rssm_summary(tmp_path, monkeypatch):
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = RSSMWorldModel(cfg).eval()
    cohort = (18502, 18503)
    samples = {sid: _mk_sample(cfg, sid, t=6, ctx=3, seed=sid % 13) for sid in cohort}

    def fake_assemble(shot_id, _config, *, camera, token_root):
        return samples[int(shot_id)]

    import imas_ambix.worldmodel.controllable_eval as ce
    from imas_ambix.worldmodel.controllable_eval import EvalConfig

    monkeypatch.setattr(ce, "_assemble_heldout", fake_assemble, raising=False)

    summary = multi_shot_diagnostic_match_rssm(
        model,
        config=EvalConfig(held_out=cohort, chunk=64),
        camera="rbb",
        token_root=None,
        device="cpu",
        out_json=tmp_path / "heldout_diagnostic_match.json",
    )
    assert summary["metric"] == "diagnostic_match_next_step"
    assert summary["diagnostics_generated"] is True
    assert summary["n_samples"] == len(cohort)
    assert np.isfinite(summary["mean_accuracy"]) and np.isfinite(summary["mean_ce"])
    assert set(summary["per_stream"]) <= {s.name for s in cfg.signal_streams}
