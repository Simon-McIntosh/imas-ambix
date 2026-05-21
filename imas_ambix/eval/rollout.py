"""Forward-rollout implementation for the WHAM-style Fusion World Model.

Implements autoregressive rollout with forced signal+action positions as
described in ``plans/world-model-v0.md`` §7.  Frame tokens are sampled via
top-k + temperature; signal/action tokens are spliced in from the provided
``control_tokens`` schedule rather than sampled.

Block-size constants
--------------------
The per-timestep token layout (``plans/world-model-v0.md`` §2) is::

    <step_start>  <frame_tokens×K_FRAME>  <signal+action tokens×K_CTRL>  <step_end>

For v0 placeholder tokenizers (PlaceholderFrameTokenizer, 8× spatial + 4×
temporal compression, 256² input → 32²/4 tokens → 256 tokens/step):

- ``K_FRAME = 256``  — frame tokens per model timestep
- ``K_CTRL  = 50``   — signal + action tokens per model timestep
                       (30 diagnostic + 12 action + 4 structural ≈ 50)
- ``K_STEP  = 2``    — structural tokens per step (step_start, step_end)

These are the *hardcoded v0 defaults* used when the model cannot be
queried for its own block sizes.  Pass ``k_frame``/``k_ctrl``/``k_step``
kwargs to override for non-default tokenizer configurations.

Related plans:
- ``plans/world-model-v0.md`` §7 (generation / rollout algorithm)
- ``plans/demo.md`` §5 (CLI surface and output layout)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch  # noqa: TC002

# ---------------------------------------------------------------------------
# v0 hardcoded block-size constants
# ---------------------------------------------------------------------------

#: Frame tokens per model timestep for the v0 PlaceholderFrameTokenizer.
#: Open-MAGVIT2 at 16× spatial → (256/16)² = 256 tokens per frame per step
#: (temporal_compression=1 for MAGVIT2; 4 for placeholder giving 64 tokens).
#: We use 256 as the default: MAGVIT2-compatible and documented in the plan.
K_FRAME_DEFAULT: int = 256

#: Signal + action tokens per model timestep.
#: ~30 diagnostic channels × 1 Chronos token + ~12 action + ~8 structural = 50.
K_CTRL_DEFAULT: int = 50

#: Structural tokens per step (``<step_start>`` + ``<step_end>``).
K_STEP_DEFAULT: int = 2


# ---------------------------------------------------------------------------
# RolloutConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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


# ---------------------------------------------------------------------------
# rollout()
# ---------------------------------------------------------------------------


def rollout(
    model: Any,
    initial_tokens: torch.LongTensor,
    control_tokens: torch.LongTensor,
    config: RolloutConfig,
    *,
    block_kind: torch.LongTensor | None = None,
    block_kind_per_step: list[int] | None = None,
    k_frame: int = K_FRAME_DEFAULT,
    k_ctrl: int = K_CTRL_DEFAULT,
    k_step: int = K_STEP_DEFAULT,
) -> dict[str, Any]:
    """Autoregressive WHAM rollout with forced signal+action positions.

    Performs ``config.rollout_steps`` autoregressive steps, alternating between:

    1. Sampling ``k_frame`` frame tokens via top-k + temperature from the model.
    2. Forcing ``k_ctrl`` signal/action tokens from ``control_tokens`` (no
       sampling) when ``config.force_signal_action_tokens`` is True.

    Parameters
    ----------
    model:
        Trained WHAM transformer (``imas_ambix.model.WhamModel`` or any object
        with a ``forward(input_ids, ...) -> {"logits": Tensor}`` method).
    initial_tokens:
        ``(B, L)`` prefix token ids (long tensor). The first
        ``config.prefix_tokens`` positions form the initial context window.
    control_tokens:
        ``(B, rollout_steps * k_ctrl)`` ground-truth signal+action tokens for
        the entire rollout horizon.  These are spliced in step-by-step rather
        than sampled.
    config:
        :class:`RolloutConfig` controlling sampling parameters.
    block_kind:
        Optional ``(B, L)`` block-kind labels for the prefix (unused in v0 but
        threaded through for future KV-cache filtering).
    block_kind_per_step:
        Optional list of block-kind codes to annotate newly generated tokens.
        Length must equal ``rollout_steps * (k_frame + k_ctrl)`` if provided.
    k_frame:
        Frame tokens per rollout step (default: :data:`K_FRAME_DEFAULT`).
    k_ctrl:
        Control tokens per rollout step (default: :data:`K_CTRL_DEFAULT`).
    k_step:
        Structural tokens per step such as ``<step_start>/<step_end>``
        (default: :data:`K_STEP_DEFAULT`).  Currently unused in sampling but
        counted in ``rollout_len``.

    Returns
    -------
    dict with keys:
        ``"tokens"``
            ``LongTensor`` of shape ``(B, prefix_len + rollout_len)`` — the
            full sequence (prefix + predicted tokens).
        ``"predicted_tokens"``
            ``LongTensor`` of shape ``(B, rollout_len)`` — only the generated
            suffix appended after the prefix.
        ``"log_probs"``
            ``FloatTensor`` of shape ``(B, frame_tokens_generated)`` — log
            probabilities for the sampled frame positions only.  Forced
            control positions are omitted because they are deterministic.

    Notes
    -----
    **Mock / untrained model:** if ``checkpoint_path="mock"`` was used in
    :func:`~imas_ambix.demo.runner.run_demo`, the caller passes a
    :class:`~imas_ambix.demo.runner._MockWhamModel` whose ``forward``
    returns uniform logits.  The rollout still executes the full sampling
    loop — the generated tokens are essentially random, which suffices to
    exercise the pipeline end-to-end.

    **Block sizes:** the v0 defaults (``K_FRAME=256``, ``K_CTRL=50``) are
    documented in the module docstring.  When a real trained checkpoint is
    used, pass ``k_frame`` / ``k_ctrl`` to match the tokenizer configuration
    used during training.
    """
    import torch

    if model is None or initial_tokens is None or control_tokens is None:
        raise ValueError(
            "rollout() requires non-None model, initial_tokens, and control_tokens. "
            "Pass a WhamModel (or mock) and LongTensor inputs."
        )

    # Ensure 2-D (B, L) tensors
    if initial_tokens.dim() == 1:
        initial_tokens = initial_tokens.unsqueeze(0)
    if control_tokens.dim() == 1:
        control_tokens = control_tokens.unsqueeze(0)

    b = initial_tokens.shape[0]
    device = initial_tokens.device

    # Validate control_tokens shape
    expected_ctrl_len = config.rollout_steps * k_ctrl
    if control_tokens.shape[1] < expected_ctrl_len:
        raise ValueError(
            f"control_tokens has {control_tokens.shape[1]} tokens but "
            f"rollout_steps={config.rollout_steps} × k_ctrl={k_ctrl} = "
            f"{expected_ctrl_len} are needed."
        )

    # Working sequence: start from prefix
    tokens = initial_tokens.clone()

    # Accumulate log-probs for sampled (frame) positions only
    log_probs_list: list[torch.Tensor] = []

    # Generated suffix (before force-appending control)
    generated_list: list[torch.Tensor] = []

    for step in range(config.rollout_steps):
        # ---- 1. Sample frame tokens ----------------------------------------
        for _ in range(k_frame):
            with torch.no_grad():
                out = model.forward(tokens)
                logits = out["logits"][:, -1, :]  # (B, V)

            # Top-k truncation
            top_k = min(config.top_k, logits.shape[-1])
            topk_logits, topk_idx = logits.topk(top_k, dim=-1)  # (B, k)

            # Temperature scaling + softmax
            scaled = topk_logits / max(config.temperature, 1e-7)
            probs = torch.softmax(scaled, dim=-1)  # (B, k)

            # Multinomial sampling → (B, 1) index into top-k
            sampled_local = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Map back to vocab ids
            sampled_ids = topk_idx.gather(1, sampled_local)  # (B, 1)

            # Log-prob of selected token
            log_p = torch.log(probs.gather(1, sampled_local) + 1e-12)  # (B, 1)
            log_probs_list.append(log_p)

            tokens = torch.cat([tokens, sampled_ids], dim=1)
            generated_list.append(sampled_ids)

        # ---- 2. Force control tokens (signal + action) ---------------------
        if config.force_signal_action_tokens:
            ctrl_slice = control_tokens[
                :, step * k_ctrl : (step + 1) * k_ctrl
            ]  # (B, k_ctrl)
            tokens = torch.cat([tokens, ctrl_slice], dim=1)
            generated_list.append(ctrl_slice)
        else:
            # Sample control tokens too (not recommended in WHAM recipe)
            for _ in range(k_ctrl):
                with torch.no_grad():
                    out = model.forward(tokens)
                    logits = out["logits"][:, -1, :]
                top_k = min(config.top_k, logits.shape[-1])
                topk_logits, topk_idx = logits.topk(top_k, dim=-1)
                scaled = topk_logits / max(config.temperature, 1e-7)
                probs = torch.softmax(scaled, dim=-1)
                sampled_local = torch.multinomial(probs, num_samples=1)
                sampled_ids = topk_idx.gather(1, sampled_local)
                tokens = torch.cat([tokens, sampled_ids], dim=1)
                generated_list.append(sampled_ids)

    # Concatenate generated suffix
    predicted_tokens = torch.cat(generated_list, dim=1)  # (B, rollout_len)

    # Stack log-probs for frame positions (B, total_frame_tokens)
    if log_probs_list:
        log_probs = torch.cat(log_probs_list, dim=1)  # (B, steps * k_frame)
    else:
        log_probs = torch.zeros(b, 0, device=device)

    return {
        "tokens": tokens,
        "predicted_tokens": predicted_tokens,
        "log_probs": log_probs,
    }
