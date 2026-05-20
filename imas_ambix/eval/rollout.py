"""Forward-rollout stub for the WHAM-style Fusion World Model.

The rollout API stitches together the trained AR transformer with the
Open-MAGVIT2 decoder and the control-token splicing logic described in
``plans/world-model-v0.md`` §7. The implementation lands once the WHAM
model (``imas_ambix/model/``) and its sampling infrastructure are in place.
This module holds the public-surface dataclass and function signature so
downstream code (demo CLI, eval harness, notebooks) can import and type-check
against them today.

Related plans:
- ``plans/world-model-v0.md`` §7 (generation / rollout algorithm)
- ``plans/demo.md`` §5 (CLI surface and output layout)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RolloutConfig:
    """Configuration for a single forward rollout.

    Attributes
    ----------
    prefix_tokens:
        Number of context tokens fed to the model before free-running
        generation begins. At 130 tokens/step and 100 Hz this corresponds
        to ``prefix_tokens / 130 / 100`` seconds of physical pulse time
        (~0.15 s for the default 2048 tokens).
    rollout_steps:
        Number of autoregressive steps to generate. At 100 Hz model time,
        100 steps = 1 second of physical pulse time.
    top_k:
        Top-k truncation applied to the frame-token logit distribution.
        Signal and action tokens are forced to ground-truth values
        (``force_signal_action_tokens=True``) and are never sampled.
    temperature:
        Softmax temperature for frame-token sampling. Values < 1.0 sharpen
        the distribution; 0.8 is empirically stable for Open-MAGVIT2
        codebooks.
    force_signal_action_tokens:
        If True (default), the rollout splices in the ground-truth /
        scheduler-provided signal and action tokens at each step rather
        than letting the model sample them. This matches the WHAM recipe
        where the control schedule is "given" to the model.
    """

    prefix_tokens: int = 2048
    rollout_steps: int = 100  # at 100 Hz model time = 1 s
    top_k: int = 64
    temperature: float = 0.8
    force_signal_action_tokens: bool = True


def rollout(
    model: Any,
    initial_tokens: Any,
    control_tokens: Any,
    config: RolloutConfig,
) -> dict[str, Any]:
    """Forward-rollout the world model from an initial context.

    Given the first ``config.prefix_tokens`` tokens from a held-out shot
    (produced by the multi-modal tokenizer) and the ground-truth control
    schedule, this function auto-regressively generates ``config.rollout_steps``
    timesteps of predicted frame tokens, splicing in the control-token block
    at each step.

    The algorithm (once implemented) is:

    1. Feed ``initial_tokens[:config.prefix_tokens]`` to the model; compute
       the KV cache.
    2. For each rollout step i in ``range(config.rollout_steps)``:
       a. Sample the 64-token frame block using top-k (``config.top_k``)
          sampling at ``config.temperature``, conditioned on the current
          KV cache.
       b. If ``config.force_signal_action_tokens``, splice
          ``control_tokens[i]`` (the known signal + action tokens for
          timestep i) into the sequence at the correct positions.
       c. Extend the KV cache with the new tokens.
    3. Decode the predicted frame tokens through the Open-MAGVIT2 decoder
       into ``(T, H, W, 3)`` uint8 frames.
    4. Return a dict with keys ``frame_tokens``, ``frames``,
       ``signal_tokens``, ``config``.

    Parameters
    ----------
    model:
        Trained WHAM transformer (``imas_ambix.model.WhamModel`` or
        HuggingFace-compatible causal LM).
    initial_tokens:
        1-D int32 array of length >= ``config.prefix_tokens``.
    control_tokens:
        Array of shape ``(rollout_steps, n_control_tokens)`` containing
        the ground-truth signal+action tokens for the rollout window.
    config:
        :class:`RolloutConfig` controlling sampling parameters.

    Returns
    -------
    dict
        - ``frame_tokens`` — predicted frame token ids ``(T,)``
        - ``frames`` — decoded uint8 frames ``(T, H, W, 3)``
        - ``signal_tokens`` — spliced signal tokens ``(T, n_signal)``
        - ``config`` — the :class:`RolloutConfig` used

    Raises
    ------
    NotImplementedError
        Always — the rollout implementation lands in a later commit once
        ``imas_ambix/model/`` is in place.
    """
    raise NotImplementedError(
        "rollout() is not yet implemented. "
        "The implementation lands once imas_ambix/model/ is merged. "
        "See plans/world-model-v0.md §7 for the algorithm."
    )
