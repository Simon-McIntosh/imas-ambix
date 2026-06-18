"""DDP path is import-safe and the single-process path is unchanged.

Data-parallel 2-GPU training is opt-in via the launcher (torchrun sets
RANK/LOCAL_RANK/WORLD_SIZE).  These CPU/1-proc unit tests are the back-compat
contract — the 2-card run is the integration check on the GPU shell:

* ``DistEnv.from_environment`` reports single-proc when WORLD_SIZE is unset or
  1, and reads the launcher env when WORLD_SIZE > 1;
* ``init_distributed`` / ``shutdown_distributed`` / ``_barrier`` are no-ops in
  single-proc (no process group created);
* ``_unwrap`` returns the bare model when there is no DDP wrapper;
* the ``loss_spec`` forward path (the DDP loss, computed INSIDE forward so the
  reducer sees the full graph) equals the standalone chunked NLL;
* ``train_corpus`` with WORLD_SIZE unset runs the legacy single-device path and
  produces a checkpoint (the existing corpus tests already cover the training
  numerics; here we only assert the DDP wiring did not perturb the default).
"""

from __future__ import annotations

import torch

from imas_ambix.worldmodel.model import (
    ModalityHeadSpec,
    WorldModel,
    WorldModelConfig,
)
from imas_ambix.worldmodel.train import (
    DistEnv,
    _barrier,
    _unwrap,
    chunked_next_token_nll,
    collate_samples,
    init_distributed,
    shutdown_distributed,
)

# reuse the tiny synthetic helpers from the chunked-loss test module
from tests.worldmodel.test_chunked_loss import (  # noqa: E402
    _names,
    _synthetic_sample,
    _tiny_model,
)


def test_distenv_defaults_single_proc(monkeypatch):
    """Unset WORLD_SIZE (or ==1) -> single-proc; >1 reads the launcher env."""
    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    env = DistEnv.from_environment()
    assert env.world_size == 1
    assert env.rank == 0
    assert env.local_rank == 0
    assert env.enabled is False
    assert env.is_main is True

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert DistEnv.from_environment().enabled is False

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    env2 = DistEnv.from_environment()
    assert env2.world_size == 2
    assert env2.rank == 1
    assert env2.local_rank == 1
    assert env2.enabled is True
    assert env2.is_main is False


def test_dist_helpers_are_noops_single_proc():
    """init / shutdown / barrier do nothing (no process group) single-proc."""
    import torch.distributed as dist

    env = DistEnv(rank=0, local_rank=0, world_size=1)
    init_distributed(env)  # must NOT create a process group
    assert not dist.is_initialized()
    _barrier(env)  # no-op
    shutdown_distributed(env)  # no-op
    assert not dist.is_initialized()


def test_unwrap_returns_bare_model():
    """_unwrap returns the WorldModel itself when there is no DDP wrapper."""
    model = _tiny_model()
    assert _unwrap(model) is model

    # a stand-in wrapper exposing .module (mirrors DistributedDataParallel)
    class _Wrap:
        def __init__(self, m):
            self.module = m

    assert _unwrap(_Wrap(model)) is model


def test_loss_spec_forward_equals_standalone_chunked():
    """forward(loss_spec=...) (the DDP loss) == standalone chunked NLL."""
    model = _tiny_model()
    model.eval()
    obs, plan = _names(model)
    sample = _synthetic_sample(model, seed=11)
    batch = collate_samples([sample], obs, plan)
    with torch.no_grad():
        # standalone (what tests + the overfit smoke call)
        oh = model.encode(batch)
        standalone = chunked_next_token_nll(
            model, oh, batch, obs, target_only=True, chunk_channels=4
        )
        # the DDP path: loss computed INSIDE forward
        via_forward = model(
            batch,
            loss_spec={
                "obs_names": obs,
                "target_only": True,
                "chunk_channels": 4,
            },
        )
    assert isinstance(via_forward, torch.Tensor)
    assert via_forward.ndim == 0
    assert torch.allclose(standalone, via_forward, atol=1e-6, rtol=1e-6)


def test_loss_spec_backprops_to_full_graph():
    """forward(loss_spec=...) grads flow to BOTH backbone and head params.

    This is the DDP correctness invariant: computing the loss inside forward
    means a DDP reducer sees the whole graph, so every parameter gets a grad.
    """
    cfg = WorldModelConfig(
        modalities=[
            ModalityHeadSpec("plan", vocab_size=17, n_channels=2, is_conditioning=True),
            ModalityHeadSpec("summary", vocab_size=17, n_channels=4),
            ModalityHeadSpec("rbb", vocab_size=64, n_channels=256),
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
    model.train()
    obs, plan = _names(model)
    sample = _synthetic_sample(model, n_steps=10, context_steps=3, seed=4)
    batch = collate_samples([sample], obs, plan)
    model.zero_grad(set_to_none=True)
    loss = model(
        batch,
        loss_spec={"obs_names": obs, "target_only": False, "chunk_channels": 64},
    )
    loss.backward()
    # a backbone param AND the full-res head/channel_query got a grad
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert model.heads["rbb"].weight.grad is not None
    assert model.channel_query["rbb"].grad is not None
    assert torch.isfinite(model.heads["rbb"].weight.grad).all()


def test_train_corpus_single_proc_unchanged(tmp_path, monkeypatch):
    """train_corpus with WORLD_SIZE unset runs the legacy single-device path.

    Builds a tiny synthetic corpus, trains a few steps on CPU, and asserts a
    checkpoint is written and the loss is finite — the DDP wiring must not
    perturb the default path.
    """
    import sys

    from imas_ambix.worldmodel.dataset import WorldModelWindowConfig
    from imas_ambix.worldmodel.train import CorpusTrainConfig, train_corpus

    for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)

    # reuse the corpus-test synthetic store builder
    sys.path.insert(0, "tests/worldmodel")
    from test_worldmodel_skeleton import _make_synthetic_corpus  # noqa: PLC0415

    mods = _make_synthetic_corpus(tmp_path, 24065)
    _make_synthetic_corpus(tmp_path, 24066, seed=3)
    out_dir = tmp_path / "ckpt"
    cfg = CorpusTrainConfig(
        steps=5,
        batch_size=1,
        num_workers=0,
        ckpt_every=5,
        eval_every=0,
        window=WorldModelWindowConfig(n_steps=16, context_steps=4),
        model_kwargs={"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32},
    )
    result = train_corpus(
        [24065, 24066],
        modalities=mods,
        config=cfg,
        out_dir=out_dir,
        token_root=tmp_path,
        device="cpu",
        resume=False,
    )
    assert result.steps_run == 5
    assert result.checkpoint_path is not None
    assert (out_dir / "latest.pt").exists()
    assert torch.isfinite(torch.tensor(result.final_loss))
