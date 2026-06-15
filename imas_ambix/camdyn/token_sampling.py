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

All single-shot decoders return ``(F, H, W)`` int64 global-id grids (the
model's native space, which the OMAG2 decode subprocess offsets), matching
:func:`reconstruction_demo._bit_map_tokens` so they drop straight into the
renderers behind the ``--decode`` flag.

The :func:`maskgit_decode` decoder is the one exception that needs more than a
static logit tensor: it is the **cross-cell-coherence** lever the per-bit and
bit-beam samplers structurally lack.  The bernoulli and beam decoders draw each
cell INDEPENDENTLY from one frozen forward pass, so a sampled bright filament
lands in a statistically-plausible-but-displaced place.  MaskGIT decode instead
runs the model's forward MANY times: it commits the most-confident still-masked
cells, writes their chosen ids back into the token tensor + flips them visible
(the model's own ``[MASK]``-embedding conditioning path), and re-forwards so the
remaining cells re-predict conditioned on their already-decided neighbours.  The
joint sample is therefore COHERENT, not a product of per-cell marginals.

Because it re-runs the model, :func:`maskgit_decode` takes a ``forward_fn``
closure ``(tokens, visible) -> bit_logits`` rather than a static logit array —
the caller (structure_fidelity / the renderers) builds that closure around the
torch model + frozen conditioning, so this module never imports model.py beyond
the pure-numpy bit utilities and never touches the model definition itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.camdyn.model import LFQ_BITS, bit_logits_to_token_logits

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DECODE_MODES",
    "bernoulli_sample",
    "bit_beam_sample",
    "cosine_mask_schedule",
    "decode_tokens",
    "map_decode",
    "maskgit_decode",
]

#: The decode modes the renderers expose via ``--decode``.  ``map`` is the
#: scoring default (the per-bit mode); ``bernoulli`` / ``beam`` are the
#: single-pass stochastic decoders; ``maskgit`` is the iterative coherent
#: decoder (it needs a model-forward closure, wired by the renderers).
DECODE_MODES = ("map", "bernoulli", "beam", "maskgit")


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


def cosine_mask_schedule(n_masked: int, n_rounds: int) -> np.ndarray:
    """Cells to commit per round under a cosine masking-ratio schedule.

    MaskGIT decodes over ``n_rounds`` parallel rounds; the fraction of cells
    that REMAIN masked after round ``t`` (0-based) follows the cosine schedule
    ``γ(r) = cos(π/2 · r)`` with ``r = (t+1)/n_rounds`` — slow at first (commit
    few high-confidence cells), accelerating later.  Returns an integer array of
    length ``n_rounds`` whose entries sum to exactly ``n_masked`` (the last
    round mops up any rounding remainder), each ≥ 0.
    """
    n_masked = int(n_masked)
    n_rounds = max(1, int(n_rounds))
    if n_masked <= 0:
        return np.zeros(n_rounds, dtype=np.int64)
    # number still masked AFTER round t (t = 0..n_rounds-1)
    r = (np.arange(1, n_rounds + 1)) / n_rounds
    remaining = np.round(np.cos(np.pi / 2.0 * r) * n_masked).astype(np.int64)
    remaining = np.clip(remaining, 0, n_masked)
    remaining[-1] = 0  # everything committed by the final round
    # commits in round t = (masked before t) - (masked after t)
    before = np.concatenate([[n_masked], remaining[:-1]])
    commits = before - remaining
    commits = np.clip(commits, 0, n_masked)
    # fix any rounding drift so the schedule commits exactly n_masked cells
    drift = n_masked - int(commits.sum())
    commits[-1] += drift
    commits = np.clip(commits, 0, None)
    return commits.astype(np.int64)


def _chosen_id_and_confidence(
    bit_logits: np.ndarray,
    *,
    temperature: float,
    top_k: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell chosen id + its joint log-confidence under the head.

    The chosen id is a per-bit Bernoulli SAMPLE at ``temperature`` (so the
    round is a coherent sample, not greedy) optionally restricted to the
    ``top_k`` most-confident bits flipped (``top_k=None`` samples every bit).
    The confidence returned is the EXACT bit-factorised joint log-likelihood of
    the chosen id, ``Σ_b log σ(s_b·z_b)`` — the same score the bit-beam ranks
    by — so the MaskGIT commit step keeps the cells the head is most certain
    about, exactly as MaskGIT keeps the highest-softmax tokens.

    ``temperature <= 0`` returns the deterministic per-bit MAP id (greedy).
    Returns ``(chosen_id (F,H,W) int64, log_conf (F,H,W) float64)``.
    """
    z = np.asarray(bit_logits, dtype=np.float64)
    nbits = z.shape[-1]
    if temperature <= 1e-6:
        bits = (z > 0.0).astype(np.int64)
    else:
        p = _sigmoid(z / temperature)
        if top_k is not None and 0 < top_k < nbits:
            # only RESAMPLE the head's least-confident bits; keep the MAP bit on
            # the (nbits - top_k) most-confident slots (analogue of top-k: the
            # confident bits are locked, the uncertain ones are sampled).
            conf = np.abs(z)
            order = np.argsort(conf, axis=-1)  # ascending → uncertain first
            resample = np.zeros(z.shape, dtype=bool)
            np.put_along_axis(resample, order[..., :top_k], True, axis=-1)
            sampled = (rng.random(p.shape) < p).astype(np.int64)
            mapbit = (z > 0.0).astype(np.int64)
            bits = np.where(resample, sampled, mapbit)
        else:
            bits = (rng.random(p.shape) < p).astype(np.int64)
    shifts = np.arange(nbits, dtype=np.int64)
    chosen = (bits << shifts).sum(axis=-1)
    # joint log-confidence of the chosen id = Σ_b log σ(s_b · z_b)
    signs = 2.0 * bits.astype(np.float64) - 1.0
    log_conf = (-np.logaddexp(0.0, -(signs * z))).sum(axis=-1)
    return chosen.astype(np.int64), log_conf


def maskgit_decode(
    forward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tokens: np.ndarray,
    visible: np.ndarray,
    *,
    n_rounds: int = 8,
    temperature: float = 1.0,
    top_k: int | None = 6,
    confidence_noise: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Iterative confidence-based COHERENT parallel decode (MaskGIT-style).

    Decodes the ORIGINALLY-MASKED cells (``visible == False``) of a token grid
    over ``n_rounds`` parallel rounds while NEVER altering the visible context
    (observed / pre-frontier cells).  Each round:

    1. ``bit_logits = forward_fn(cur_tokens, cur_visible)`` — re-run the model
       with the cells committed so far folded into the visible context (their
       ids written into ``cur_tokens``, their flags flipped True), so the
       still-masked cells re-predict conditioned on their decided neighbours.
       This conditioning is the model's own ``[MASK]``-embedding path — exactly
       the MaskGIT mechanism — and is what makes the joint sample COHERENT
       rather than a product of independent per-cell marginals.
    2. For every still-masked cell pick a chosen id by per-bit sampling at
       ``temperature`` (optionally locking the ``nbits - top_k`` most-confident
       bits to their MAP value; ``top_k`` then bounds the per-cell entropy) and
       score its joint log-confidence under the head.
    3. Commit the highest-confidence still-masked cells per the cosine schedule
       (:func:`cosine_mask_schedule`): write their chosen ids into
       ``cur_tokens`` and flip them visible.  MaskGIT's annealed-noise trick
       adds Gumbel noise scaled by ``confidence_noise · (1 - round_frac)`` to
       the confidence before selecting, so early rounds explore and late rounds
       are near-greedy (a SAMPLE over commit ORDER, not just over ids).

    After the final round every originally-masked cell carries a sampled id and
    the grid is returned (``(F,H,W)`` int64 global ids).  Visible cells keep
    their input ids untouched.

    Parameters
    ----------
    forward_fn:
        ``(tokens (F,H,W) int, visible (F,H,W) bool) -> bit_logits
        (F,H,W,bits)``.  Pure function of the current token/visibility state;
        the caller closes over the torch model + frozen conditioning.
    tokens:
        ``(F,H,W)`` int global ids.  Visible cells hold the true observed id;
        masked cells' input ids are irrelevant (the model replaces them with the
        ``[MASK]`` embedding) — they are overwritten as they commit.
    visible:
        ``(F,H,W)`` bool — True = observed context (never re-decoded).
    n_rounds:
        Number of parallel commit rounds (~8 is the MaskGIT default).
    temperature:
        Per-bit sampling temperature for the chosen id (``<=0`` = greedy MAP).
    top_k:
        Lock all but the ``top_k`` least-confident bits per cell to their MAP
        value (``None`` = sample every bit).  Bounds per-cell deviation from the
        MAP id so a committed cell stays a plausible neighbour.
    confidence_noise:
        Scale of the annealed Gumbel noise added to the commit confidence
        (``0`` = deterministic confidence ordering).
    rng:
        Numpy Generator (default fresh) — seed it for reproducibility.
    """
    rng = np.random.default_rng() if rng is None else rng
    cur_tokens = np.asarray(tokens, dtype=np.int64).copy()
    cur_visible = np.asarray(visible, dtype=bool).copy()
    masked = ~cur_visible  # the cells this decode owns (never touch visible)
    n_masked = int(masked.sum())
    if n_masked == 0:
        return cur_tokens

    schedule = cosine_mask_schedule(n_masked, n_rounds)
    masked_idx = np.argwhere(masked)  # (n_masked, 3) coordinates
    committed = np.zeros(n_masked, dtype=bool)  # over masked_idx rows

    for t, n_commit in enumerate(schedule):
        still = ~committed
        if not still.any():
            break
        bit_logits = forward_fn(cur_tokens, cur_visible)
        chosen, log_conf = _chosen_id_and_confidence(
            bit_logits, temperature=temperature, top_k=top_k, rng=rng
        )
        rows = masked_idx[still]  # coordinates of still-masked cells
        fi, hi, wi = rows[:, 0], rows[:, 1], rows[:, 2]
        conf = log_conf[fi, hi, wi]
        ids = chosen[fi, hi, wi]
        if confidence_noise > 0.0:
            # annealed Gumbel noise on the SELECTION score (MaskGIT): explore
            # the commit order early, near-greedy late.
            anneal = confidence_noise * (1.0 - (t + 1) / max(1, len(schedule)))
            u = np.clip(rng.random(conf.shape), 1e-12, 1.0 - 1e-12)
            gumbel = -np.log(-np.log(u))
            select_score = conf + anneal * gumbel
        else:
            select_score = conf
        n_commit = int(min(n_commit, rows.shape[0]))
        if t == len(schedule) - 1:
            n_commit = rows.shape[0]  # final round commits everything left
        if n_commit <= 0:
            continue
        # indices (into the still-masked subset) of the highest-score cells
        take = np.argpartition(-select_score, n_commit - 1)[:n_commit]
        gfi, ghi, gwi = fi[take], hi[take], wi[take]
        cur_tokens[gfi, ghi, gwi] = ids[take]
        cur_visible[gfi, ghi, gwi] = True
        # mark those rows committed (map subset index → masked_idx row index)
        still_rows = np.flatnonzero(still)
        committed[still_rows[take]] = True

    return cur_tokens


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
    and ``beam`` are the truth-free single-pass sampled decoders.  This is the
    single entry point the renderers call behind their ``--decode`` / ``--temp``
    flags for the SINGLE-PASS modes.  ``maskgit`` is iterative and needs a
    model-forward closure, so it is NOT dispatched here — the renderers call
    :func:`maskgit_decode` directly with the forward closure they hold.
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
    if mode == "maskgit":
        raise ValueError(
            "maskgit decode is iterative and requires a model-forward closure; "
            "call token_sampling.maskgit_decode(forward_fn, tokens, visible, ...) "
            "directly — it cannot be produced from a single static bit_logits array"
        )
    raise ValueError(f"unknown decode mode {mode!r}; expected one of {DECODE_MODES}")


# Keep the model adapter import live (used to document the scoring identity and
# available for callers that want the flat-candidate path); the per-cell beam
# above inlines the same Σ_b log σ(s_b·z_b) score for the per-cell candidate set.
_ = (bit_logits_to_token_logits, LFQ_BITS)
