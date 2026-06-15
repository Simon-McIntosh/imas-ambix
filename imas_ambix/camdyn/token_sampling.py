"""Truth-free sampled decoders for the LFQ 18-bit head — recover hedged structure.

The camera-dynamics head emits 18 INDEPENDENT bit-logits per cell (a bitwise
factorised LFQ likelihood).  Both scoring and rendering currently decode each
cell by the per-bit MAP::

    pred_id = Σ_b (z_b > 0) << b

Under genuine aleatoric uncertainty about the exact position of a bright
edge/SOL filament, this mode collapses to a smeared "mean" token grid — the
striations the ground-truth frames show blur out.  The hypothesis (probed in
:mod:`structure_fidelity`) is that the head's per-bit distribution still
CONTAINS the filament structure; taking the mode destroys it, and SAMPLING
recovers it.

This module provides the two production sampled decoders, both **pure numpy on
the head's own bit-logits** — they take NO access to the ground-truth tokens,
so they are shippable inference paths (unlike the oracle "joint" sampler in
:mod:`structure_fidelity`, whose candidate set is the true id XOR offsets and is
therefore an upper-bound probe only):

* :func:`bernoulli_sample` — independent per-bit Bernoulli sample at
  temperature ``T``: ``bit_b ~ σ(z_b / T)``.  Cheap, but the 18 bits are drawn
  independently, so it cannot represent a coherent joint over codebook ids.

* :func:`bit_beam_sample` — a truth-free BIT-BEAM joint sampler.  Per cell it
  enumerates a candidate id set by expanding the head's OWN most-uncertain bits
  (smallest ``|z_b|``) around the per-bit MAP id, scores every candidate with
  the EXACT bit-factorised log-likelihood
  (:func:`imas_ambix.camdyn.model.bit_logits_to_token_logits`), and draws one id
  by temperature + nucleus / top-k sampling.  The candidates are real codebook
  ids ranked by the head's factorised likelihood — a coherent joint sample that
  never sees the truth.

All decoders return ``(F, H, W)`` int64 global-id grids (the model's native
space, which the OMAG2 decode subprocess offsets), matching
:func:`reconstruction_demo._bit_map_tokens` so they drop straight into the
renderers behind the ``--decode`` flag.
"""

from __future__ import annotations

import numpy as np

from imas_ambix.camdyn.model import LFQ_BITS, bit_logits_to_token_logits

__all__ = [
    "DECODE_MODES",
    "bernoulli_sample",
    "bit_beam_sample",
    "decode_tokens",
    "map_decode",
]

#: The decode modes the renderers expose via ``--decode``.  ``map`` is the
#: scoring default (the per-bit mode); the two stochastic modes are the
#: sampled-decode fix.
DECODE_MODES = ("map", "bernoulli", "beam")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def map_decode(bit_logits: np.ndarray) -> np.ndarray:
    """Per-bit MAP token id — ``id = Σ_b (z_b > 0) << b`` (the current decode).

    The deterministic mode of the head's factorised distribution; the scoring
    default.  Identical to :func:`reconstruction_demo._bit_map_tokens`, repeated
    here so this module is the single decode entry point for the renderers.
    """
    bl = np.asarray(bit_logits)
    shifts = np.arange(bl.shape[-1], dtype=np.int64)
    return ((bl > 0.0).astype(np.int64) << shifts).sum(axis=-1)


def bernoulli_sample(
    bit_logits: np.ndarray,
    *,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Independent per-bit Bernoulli sample: ``bit_b`` is 1 with prob ``σ(z_b/T)``.

    Each of the bits is sampled INDEPENDENTLY (the head's factorisation) and the
    bits are packed to the token id.  ``T < 1`` sharpens toward the MAP, ``T=1``
    samples the head's native per-bit marginal, ``T > 1`` flattens it.  Truth-free.
    """
    rng = np.random.default_rng() if rng is None else rng
    z = np.asarray(bit_logits, dtype=np.float64) / max(temperature, 1e-6)
    p = _sigmoid(z)  # (F,H,W,bits)
    bits = (rng.random(p.shape) < p).astype(np.int64)
    shifts = np.arange(bits.shape[-1], dtype=np.int64)
    return (bits << shifts).sum(axis=-1)


def _beam_candidate_offsets(n_expand_bits: int, nbits: int) -> np.ndarray:
    """XOR offsets enumerating every subset of the ``n_expand_bits`` chosen bits.

    The bit-beam expands the head's most-uncertain bits: given the per-cell MAP
    id, flipping any subset of the ``n_expand_bits`` lowest-confidence bits gives
    ``2**n_expand_bits`` candidate ids.  This returns the XOR pattern for each
    subset as an integer offset over a FIXED bit-slot table (the actual
    per-cell bit slots are supplied at call time).  Returns ``(2**n_expand_bits,
    n_expand_bits)`` boolean inclusion table.
    """
    n_expand_bits = int(min(n_expand_bits, nbits))
    k = 1 << n_expand_bits
    subset_idx = np.arange(k, dtype=np.int64)
    incl = ((subset_idx[:, None] >> np.arange(n_expand_bits)) & 1).astype(bool)
    return incl  # (K, n_expand_bits)


def bit_beam_sample(
    bit_logits: np.ndarray,
    *,
    temperature: float = 1.0,
    n_expand_bits: int = 8,
    top_k: int | None = None,
    top_p: float | None = 0.95,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Truth-free BIT-BEAM joint sampler over real codebook ids.

    Per cell:

    1. Take the per-bit MAP id ``m = Σ_b (z_b > 0) << b``.
    2. Identify the ``n_expand_bits`` MOST UNCERTAIN bits — smallest ``|z_b|``
       (the head's own least-confident bits — exactly the ones a filament's
       position uncertainty would hedge over).  Enumerate the
       ``2**n_expand_bits`` candidate ids by flipping every subset of those bits
       in ``m``.  These are real codebook ids near the MAP, chosen by the head's
       OWN uncertainty — no truth access.
    3. Score each candidate with the exact bit-factorised log-likelihood
       :func:`model.bit_logits_to_token_logits` (``Σ_b log σ(s_b·z_b)``).
    4. Apply temperature, then nucleus (``top_p``) and/or ``top_k`` truncation,
       renormalise, and draw one id (inverse-CDF categorical).

    This is a COHERENT joint sample (candidates are whole codebook ids ranked by
    the head's factorised likelihood) without seeing the ground truth — the
    shippable counterpart to the oracle "joint" probe in
    :mod:`structure_fidelity`.

    Parameters
    ----------
    bit_logits:
        ``(F,H,W,bits)`` per-bit logits ``z_b``.
    temperature:
        Softmax temperature over the candidate scores (``T<1`` sharpens toward
        the MAP id; ``T=1`` samples the head's restricted joint; ``T>1``
        flattens).
    n_expand_bits:
        Number of least-confident bits to expand per cell (candidate count is
        ``2**n_expand_bits``).  8 → 256 candidates / cell.
    top_k:
        Keep only the ``top_k`` highest-scoring candidates before sampling
        (``None`` = no top-k truncation).
    top_p:
        Nucleus threshold: keep the smallest candidate set whose cumulative
        probability ≥ ``top_p`` (``None`` = no nucleus truncation).
    rng:
        Numpy Generator (default fresh).
    """
    rng = np.random.default_rng() if rng is None else rng
    z = np.asarray(bit_logits, dtype=np.float64)  # (F,H,W,bits)
    nbits = z.shape[-1]
    fhw = z.shape[:-1]
    n_expand = int(min(n_expand_bits, nbits))

    # per-cell MAP id and the n_expand least-confident bit SLOTS (smallest |z|)
    map_id = map_decode(z)  # (F,H,W) int64
    conf = np.abs(z)  # (F,H,W,bits)
    # argsort ascending → first n_expand columns are the most-uncertain bit slots
    order = np.argsort(conf, axis=-1)  # (F,H,W,bits)
    uncertain_slots = order[..., :n_expand]  # (F,H,W,n_expand) bit indices

    # candidate ids: for each subset of the uncertain bits, XOR those bit
    # positions into the MAP id.  incl: (K, n_expand) subset-inclusion table.
    incl = _beam_candidate_offsets(n_expand, nbits)  # (K, n_expand)
    k = incl.shape[0]
    # per-cell XOR mask for candidate j = OR over included uncertain bit slots
    # uncertain_slots: (F,H,W,n_expand) → bit values 1<<slot
    slot_bits = np.int64(1) << uncertain_slots.astype(np.int64)  # (F,H,W,n_expand)
    # (F,H,W,K) XOR mask: sum of slot_bits where incl[j] is True
    # incl (K,n_expand) → broadcast against slot_bits (F,H,W,1,n_expand)
    xor_mask = (slot_bits[..., None, :] * incl[None, None, None, :, :]).sum(axis=-1)
    cand = (map_id[..., None] ^ xor_mask).astype(np.int64)  # (F,H,W,K)
    cand &= (1 << nbits) - 1  # clamp to vocab

    # exact bit-factorised log-likelihood per candidate.  The adapter wants a
    # flat (K,) candidate vector with a SHARED logit; here candidates are
    # per-cell, so score directly: log σ(s_b·z_b) summed over bits.
    cand_bits = ((cand[..., None] >> np.arange(nbits)) & 1).astype(np.float64)
    signs = 2.0 * cand_bits - 1.0  # (F,H,W,K,bits)
    signed = signs * z[..., None, :]
    log_sig = -np.logaddexp(0.0, -signed)  # log σ(s·z)
    scores = log_sig.sum(axis=-1)  # (F,H,W,K)

    # de-duplicate candidates within a cell: when fewer than n_expand distinct
    # uncertain slots collapse the subset lattice (never here — slots distinct),
    # identical ids would double-count.  Slots are distinct by argsort, so the
    # K ids are unique per cell; no dedup needed.

    # At (near-)zero temperature the draw degenerates to the argmax candidate
    # (the most-likely id under the head); under bit-independence that is the
    # per-bit MAP id, so beam(T→0) ≡ map_decode.  Snapping here avoids the
    # inverse-CDF float-rounding picking an equally-/less-likely neighbour when
    # one candidate's probability is ~1.
    if temperature <= 1e-3:
        argmax = np.argmax(scores, axis=-1)  # (F,H,W)
        out = np.take_along_axis(cand, argmax[..., None], axis=-1)[..., 0]
        return out.astype(np.int64)

    scores = scores / max(temperature, 1e-6)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)  # (F,H,W,K)

    if top_k is not None and 0 < top_k < k:
        # zero out all but the top_k highest-prob candidates per cell
        kth = np.sort(p, axis=-1)[..., -top_k][..., None]  # (F,H,W,1)
        p = np.where(p >= kth, p, 0.0)
        p = p / p.sum(axis=-1, keepdims=True)

    if top_p is not None and 0.0 < top_p < 1.0:
        # nucleus: sort descending, keep the smallest prefix with cumsum ≥ top_p
        sort_idx = np.argsort(-p, axis=-1)  # (F,H,W,K)
        p_sorted = np.take_along_axis(p, sort_idx, axis=-1)
        csum = np.cumsum(p_sorted, axis=-1)
        # keep a candidate if the cumulative prob BEFORE it is < top_p (so the
        # first candidate that crosses the threshold is included)
        keep_sorted = (csum - p_sorted) < top_p
        keep_sorted[..., 0] = True  # always keep the top candidate
        keep = np.zeros_like(keep_sorted)
        np.put_along_axis(keep, sort_idx, keep_sorted, axis=-1)
        p = np.where(keep, p, 0.0)
        p = p / p.sum(axis=-1, keepdims=True)

    # inverse-CDF categorical draw over K candidates per cell
    cdf = np.cumsum(p, axis=-1)
    u = rng.random(fhw + (1,))
    choice = (u > cdf).sum(axis=-1)  # index in [0, K)
    choice = np.clip(choice, 0, k - 1)
    out = np.take_along_axis(cand, choice[..., None], axis=-1)[..., 0]
    return out.astype(np.int64)


def decode_tokens(
    bit_logits: np.ndarray,
    mode: str = "map",
    *,
    temperature: float = 1.0,
    n_expand_bits: int = 8,
    top_p: float | None = 0.95,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Dispatch ``mode`` ∈ :data:`DECODE_MODES` → a ``(F,H,W)`` global-id grid.

    ``map`` ignores ``temperature`` (it is the deterministic mode); ``bernoulli``
    and ``beam`` are the truth-free sampled decoders.  This is the single entry
    point the renderers call behind their ``--decode`` / ``--temp`` flags.
    """
    if mode == "map":
        return map_decode(bit_logits)
    if mode == "bernoulli":
        return bernoulli_sample(bit_logits, temperature=temperature, rng=rng)
    if mode == "beam":
        return bit_beam_sample(
            bit_logits,
            temperature=temperature,
            n_expand_bits=n_expand_bits,
            top_p=top_p,
            top_k=top_k,
            rng=rng,
        )
    raise ValueError(f"unknown decode mode {mode!r}; expected one of {DECODE_MODES}")


# Keep the model adapter import live (used to document the scoring identity and
# available for callers that want the flat-candidate path); the per-cell beam
# above inlines the same Σ_b log σ(s_b·z_b) score for the per-cell candidate set.
_ = (bit_logits_to_token_logits, LFQ_BITS)
