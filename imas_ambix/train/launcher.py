"""Accelerate / FSDP launch wrapper for the WHAM training loop.

Builds an ``accelerate.Accelerator`` pre-configured for FSDP full-shard,
sharded state-dict checkpoints, activation checkpointing, and bf16.

When ``fsdp=False`` (the CPU smoke-test path) a vanilla Accelerator is
returned instead so tests can run without a GPU or NCCL.

All ``accelerate`` imports are deferred to :func:`build_accelerator` so
that the module can be imported even when the ``train`` extra is not
installed.
"""

from __future__ import annotations


class AccelerateUnavailableError(ImportError):
    """Raised when ``accelerate`` is not installed.

    Install the training extra::

        uv pip install "imas-ambix[train]"
    """


def build_accelerator(
    *,
    precision: str = "bf16",
    fsdp: bool = True,
    activation_checkpoint: bool = True,
) -> object:
    """Build and return an ``accelerate.Accelerator``.

    Parameters
    ----------
    precision:
        Mixed-precision mode passed to ``Accelerator``.  Typical values:
        ``"bf16"``, ``"fp16"``, ``"no"`` (fp32 / CPU).
    fsdp:
        When ``True`` (the default), attach an :class:`FullyShardedDataParallelPlugin`
        configured with:

        * ``fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD`` (ZeRO-3)
        * ``fsdp_state_dict_type = StateDictType.SHARDED_STATE_DICT``
        * ``fsdp_activation_checkpointing = activation_checkpoint``
        * ``fsdp_offload_params = False``

        When ``False``, a vanilla ``Accelerator`` is returned (used for CPU
        smoke tests where FSDP / NCCL is unavailable).
    activation_checkpoint:
        Enable activation checkpointing inside FSDP.  Ignored when
        ``fsdp=False``.

    Returns
    -------
    accelerate.Accelerator

    Raises
    ------
    AccelerateUnavailableError
        If ``accelerate`` is not installed.
    """
    try:
        import accelerate  # noqa: F401 — existence check
        from accelerate import Accelerator
    except ImportError as exc:
        raise AccelerateUnavailableError(
            "accelerate is not installed. "
            'Install the training extra: uv pip install "imas-ambix[train]"'
        ) from exc

    if not fsdp:
        # CPU / single-device path — no FSDP plugin
        return Accelerator(mixed_precision=precision if precision != "bf16" else "no")

    try:
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import ShardingStrategy, StateDictType
    except ImportError as exc:
        raise AccelerateUnavailableError(
            "torch.distributed.fsdp is not available. "
            "Install PyTorch >= 2.6 with distributed support."
        ) from exc

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        activation_checkpointing=activation_checkpoint,
        cpu_offload=False,
    )

    return Accelerator(
        mixed_precision=precision,
        fsdp_plugin=fsdp_plugin,
    )
