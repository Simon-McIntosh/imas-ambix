"""Chunked next-token cross-entropy + full-resolution camera rollout.

The full-resolution camera head (256 channels × 2^18 vocab) cannot
materialise its all-channel logits ``(B, T, 256, 2^18)`` — that is the ~50 GB
memory wall.  :func:`imas_ambix.worldmodel.train.chunked_next_token_nll` and
the chunked argmax in :func:`imas_ambix.worldmodel.eval.rollout` apply the head
one channel-chunk at a time so peak memory is ``chunk × vocab``.

These tests are the correctness contract:

* the chunked loss is NUMERICALLY EQUAL to the naive full-logits loss on a
  small synthetic case (``allclose``), for both the whole-sequence and the
  target-only objective, and across several chunk sizes;
* a 256-channel (full-resolution) rbb rollout runs and the chunked argmax
  returns the full ``(T, 256)`` predicted-token grid.

These run on CPU and are hermetic (tiny synthetic model + batch).
"""

from __future__ import annotations

import numpy as np
import torch

from imas_ambix.worldmodel.dataset import (
    PAD_LOCAL_ID,
    WorldModelSample,
)
from imas_ambix.worldmodel.model import (
    ModalityHeadSpec,
    WorldModel,
    WorldModelConfig,
)
from imas_ambix.worldmodel.train import (
    chunked_next_token_nll,
    collate_samples,
    next_token_nll,
)


def _tiny_model(*, plan_ch=2, summary_ch=5, cam_ch=16, cam_vocab=64, d_model=24):
    """A tiny multi-modal model: a plan, a small obs group, and a camera.

    The camera vocab is kept small (64) so the NAIVE full-logits loss is
    cheap to compute for the allclose reference; the chunked path is exercised
    over this same camera.
    """
    cfg = WorldModelConfig(
        modalities=[
            ModalityHeadSpec(
                "plan", vocab_size=17, n_channels=plan_ch, is_conditioning=True
            ),
            ModalityHeadSpec("summary", vocab_size=17, n_channels=summary_ch),
            ModalityHeadSpec("rbb", vocab_size=cam_vocab, n_channels=cam_ch),
        ],
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        d_ff=48,
        dropout=0.0,
        plan_steps=12,
        obs_steps=12,
    )
    torch.manual_seed(0)
    return WorldModel(cfg)


def _synthetic_sample(model, *, n_steps=12, context_steps=4, seed=1):
    """One synthetic sample matching the model's channel widths.

    Some target positions are marked invalid so the masked-mean path (the load-
    bearing numeric identity) is genuinely exercised, not a trivial all-valid.
    """
    rng = np.random.default_rng(seed)
    tokens: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    names: dict[str, tuple] = {}
    for m in model.config.modalities:
        c = int(m.n_channels)
        tok = rng.integers(0, m.vocab_size, size=(n_steps, c)).astype(np.int64)
        val = rng.random((n_steps, c)) > 0.25  # ~75% valid → exercises the mask
        # ensure at least one valid target so the modality is scored
        val[context_steps + 1, 0] = True
        tokens[m.name] = tok
        valid[m.name] = val
        names[m.name] = tuple(f"{m.name}.{i}" for i in range(c))
    return WorldModelSample(
        shot_id=1,
        grid_time=np.linspace(0.0, 1.0, n_steps),
        tokens=tokens,
        valid=valid,
        channel_names=names,
        context_steps=context_steps,
    )


def _names(model):
    plan = [m.name for m in model.config.modalities if m.is_conditioning]
    obs = [m.name for m in model.config.modalities if not m.is_conditioning]
    return obs, plan


def test_chunked_ce_equals_naive_full_logits():
    """Chunked NLL == naive full-logits NLL (whole-sequence objective)."""
    model = _tiny_model()
    model.eval()
    obs, plan = _names(model)
    sample = _synthetic_sample(model)
    batch = collate_samples([sample], obs, plan)

    # naive reference: build ALL per-channel logits, then the existing loss
    with torch.no_grad():
        out = model(batch)
        naive = next_token_nll(out.logits, batch, obs, target_only=False)
        # chunked: encode once, accumulate channel-chunk at a time
        obs_hidden = model.encode(batch)
        for chunk in (1, 4, 16, 1000):
            chunked = chunked_next_token_nll(
                model,
                obs_hidden,
                batch,
                obs,
                target_only=False,
                chunk_channels=chunk,
            )
            assert torch.allclose(naive, chunked, atol=1e-5, rtol=1e-5), (
                f"chunk={chunk}: naive {float(naive):.6f} != "
                f"chunked {float(chunked):.6f}"
            )


def test_chunked_ce_equals_naive_target_only():
    """Chunked NLL == naive full-logits NLL (target-only forecasting loss)."""
    model = _tiny_model()
    model.eval()
    obs, plan = _names(model)
    sample = _synthetic_sample(model, seed=7)
    batch = collate_samples([sample], obs, plan)

    with torch.no_grad():
        out = model(batch)
        naive = next_token_nll(out.logits, batch, obs, target_only=True)
        obs_hidden = model.encode(batch)
        for chunk in (1, 3, 16):
            chunked = chunked_next_token_nll(
                model,
                obs_hidden,
                batch,
                obs,
                target_only=True,
                chunk_channels=chunk,
            )
            assert torch.allclose(naive, chunked, atol=1e-5, rtol=1e-5), (
                f"chunk={chunk}: {float(naive):.6f} != {float(chunked):.6f}"
            )


def test_chunked_ce_gradients_match():
    """Chunked NLL backprops to the SAME grads as the naive full-logits loss.

    The training loop only ever sees the chunked path, so the gradient — not
    just the scalar — must match the naive reference.
    """
    obs_name = "rbb"

    def _grad(use_chunk):
        m = _tiny_model()
        m.train()
        obs, plan = _names(m)
        sample = _synthetic_sample(m, seed=3)
        batch = collate_samples([sample], obs, plan)
        m.zero_grad(set_to_none=True)
        if use_chunk:
            oh = m.encode(batch)
            loss = chunked_next_token_nll(m, oh, batch, obs, chunk_channels=4)
        else:
            out = m(batch)
            loss = next_token_nll(out.logits, batch, obs)
        loss.backward()
        return m.heads[obs_name].weight.grad.detach().clone()

    g_naive = _grad(False)
    g_chunk = _grad(True)
    assert torch.allclose(g_naive, g_chunk, atol=1e-5, rtol=1e-5)


def test_full_res_camera_rollout_returns_full_grid():
    """A 256-channel (full-res) rbb rollout runs; chunked argmax → (T, 256)."""
    from imas_ambix.worldmodel.eval import rollout

    # full-resolution camera: 256 channels.  Keep vocab small so the test is
    # fast — the CHUNKING is what is under test, not the 2^18 vocab itself.
    cfg = WorldModelConfig(
        modalities=[
            ModalityHeadSpec("plan", vocab_size=17, n_channels=2, is_conditioning=True),
            ModalityHeadSpec("summary", vocab_size=17, n_channels=4),
            ModalityHeadSpec("rbb", vocab_size=128, n_channels=256),
        ],
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        plan_steps=10,
        obs_steps=10,
    )
    torch.manual_seed(0)
    model = WorldModel(cfg)
    model.eval()
    obs, plan = _names(model)
    sample = _synthetic_sample(model, n_steps=10, context_steps=3, seed=5)

    for chunk in (16, 64, 256):
        pred = rollout(model, sample, obs, plan, chunk_channels=chunk)
        assert "rbb" in pred
        assert pred["rbb"].shape == (10, 256), pred["rbb"].shape
        # the context steps are copied from truth; target steps are generated
        assert pred["rbb"].dtype == np.int64
        # generated ids are valid token ids (within the camera vocab)
        assert pred["rbb"].min() >= 0
        assert pred["rbb"].max() < 128


def test_chunked_argmax_matches_full_argmax():
    """Per-step chunked argmax == full-logits argmax over all channels."""
    from imas_ambix.worldmodel.eval import _chunked_argmax_step

    model = _tiny_model(cam_ch=256, cam_vocab=96)
    model.eval()
    obs, plan = _names(model)
    sample = _synthetic_sample(model, seed=9)
    batch = collate_samples([sample], obs, plan)
    with torch.no_grad():
        obs_hidden = model.encode(batch)
        step = 5
        # full reference
        full_lg = model.channel_logits(obs_hidden, "rbb")  # (1, T, 256, V)
        full_arg = full_lg[:, step].argmax(dim=-1)  # (1, 256)
        chunk_arg = _chunked_argmax_step(
            model, obs_hidden, "rbb", step, chunk_channels=32
        )
        assert torch.equal(full_arg, chunk_arg)


def test_forward_return_logits_false_skips_logits():
    """forward(return_logits=False) returns obs_hidden and NO logits."""
    model = _tiny_model()
    model.eval()
    obs, plan = _names(model)
    sample = _synthetic_sample(model)
    batch = collate_samples([sample], obs, plan)
    with torch.no_grad():
        out = model(batch, return_logits=False)
    assert out.logits == {}
    assert out.obs_hidden is not None
    assert out.obs_hidden.shape[-1] == model.config.d_model
    # equals encode() exactly
    with torch.no_grad():
        oh = model.encode(batch)
    assert torch.equal(out.obs_hidden, oh)
    _ = PAD_LOCAL_ID  # imported for parity with the dataset's pad contract
