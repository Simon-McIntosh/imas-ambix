"""Per-frame timescale (Δt) + camera-view conditioning for the camera transformer.

Why this exists (one model for ALL of MAST)
--------------------------------------------
The camera world model is trained on ONE camera (``rbb``) at ONE slow cadence
(~6 ms/frame full-shot windows).  To make a single model ingest ALL the MAST
imaging data we must tell it two things the token sequence alone does not carry:

* **how fast time is moving** — the SAME 256-token frame sequence sampled at
  6 ms/frame (the slow regime where the coil currents move the plasma position)
  versus 50 µs/frame (the fast regime where MHD / fast instabilities live) is a
  DIFFERENT physical process, and the model must interpret the temporal axis
  accordingly.  Without a cadence signal the temporal attention would apply the
  same learned dynamics to both, which is wrong by ~250× in rate.
* **which camera took it** — the views differ in field-of-view, optics, and
  colour (``rco`` is RGB), so a token id means a different thing per camera; a
  single shared next-frame law over all views is under-identified.

The encoding
------------
**Δt — log-Δt scalar → small MLP → added to the per-frame temporal embedding.**
The per-frame inter-frame interval Δt (seconds) is mapped to ``log10(Δt)`` (a
finite, well-conditioned scalar that turns the ~250× dynamic range into a small
additive offset), normalised to a reference decade, and pushed through a tiny
2-layer MLP producing a ``d_model`` vector ADDED to that frame's temporal
position embedding.  log over a learned bucket because the cadence is genuinely
continuous (per-shot windows span a continuum, not two discrete classes — see
the per-shot horizon windowing) and a continuous code lets the model INTERPOLATE
between regimes it has seen; a bucket would quantise and could not.  The MLP's
final layer is ZERO-INIT so a fresh (or OFF) timescale head adds exactly nothing
— the model starts byte-identical to the cadence-blind forecaster and the Δt
signal EARNS influence through training (the AdaLN-Zero discipline used by the
actuator path).

**camera — a learned per-camera embedding** added to the camera-frame token
embeddings (exactly like the existing ``cam_marker``).  ZERO-INIT so OFF /
fresh is identity; an unknown / missing camera falls back to the reference
camera's row (``rbb`` = index 0).

Both pieces are OPTIONAL config knobs (``timescale_conditioning`` /
``camera_conditioning``); when off the model is byte-identical to the prior
model and a prior checkpoint loads with the new tables left at their fresh
(zero) init.  Conditioning prefix frames (signals / plan) are NOT on the camera
cadence, so they receive a neutral (zero-Δt) temporal offset — the Δt head only
modulates the real camera frames.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Camera vocabulary (the MAST visible-light cameras the corpus spans)
# ---------------------------------------------------------------------------
#
# Order is STABLE — the index is what the checkpoint records, so appending a new
# camera is safe (it takes the next index) but reordering would silently remap a
# trained embedding.  Index 0 is the reference camera (``rbb``), the fallback for
# an unknown / missing camera so an out-of-vocab view degrades to "the camera the
# model was first trained on" rather than erroring.

CAMERA_IDS: tuple[str, ...] = ("rbb", "rco", "rgb", "rgc", "rba", "rbc")
REFERENCE_CAMERA_INDEX = 0


def camera_index(name: str | None) -> int:
    """Map a camera name to its stable embedding index (unknown → reference).

    ``None`` / an unrecognised name maps to :data:`REFERENCE_CAMERA_INDEX`
    (``rbb``) so a shot whose camera id is missing or outside the known set is
    interpreted as the reference view rather than crashing the embedding lookup.
    """
    if not name:
        return REFERENCE_CAMERA_INDEX
    try:
        return CAMERA_IDS.index(str(name))
    except ValueError:
        return REFERENCE_CAMERA_INDEX


def camera_indices(names: Sequence[str | None]) -> list[int]:
    """Vectorised :func:`camera_index` over a batch of camera names."""
    return [camera_index(n) for n in names]


# ---------------------------------------------------------------------------
# Per-frame Δt from frame timestamps
# ---------------------------------------------------------------------------
#
# A window's per-frame Δt is the local inter-frame interval.  We define it as the
# gap to the PREVIOUS frame (the cadence that produced this frame), with frame 0
# taking the same Δt as frame 1 (it has no predecessor — the leading edge shares
# the window's cadence).  This is robust to a non-uniform window (a jittered
# cadence gives a per-frame Δt rather than one window scalar) and matches the
# temporal-attention semantics: frame t's hidden predicts frame t+1, so the Δt
# that matters at position t is the step taken INTO t.

#: Reference cadence (seconds) the log-Δt scalar is centred on.  Chosen as the
#: rbb full-shot cadence (~6 ms) so the slow regime the model was first trained
#: on maps to ~0 and faster cadences map negative — a small, centred input.
REFERENCE_DT_SECONDS = 6.0e-3


def frame_dt_seconds(frame_time: np.ndarray) -> np.ndarray:
    """``(T,)`` frame timestamps (s) → ``(T,)`` per-frame Δt (s), forward-filled.

    Δt[t] is ``frame_time[t] - frame_time[t-1]`` (the interval INTO frame t);
    Δt[0] copies Δt[1] (the leading frame shares the window cadence).  A window
    with fewer than two frames, or non-finite / non-increasing times, returns the
    reference cadence everywhere (a safe neutral that maps to log-offset 0) so the
    Δt head never sees a NaN / non-positive interval.
    """
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)
    n = ft.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    out = np.full((n,), float(REFERENCE_DT_SECONDS), dtype=np.float64)
    if n >= 2 and np.all(np.isfinite(ft)):
        d = np.diff(ft)  # (T-1,) interval into frames 1..T-1
        # only trust strictly-positive intervals; fall back to reference for any
        # non-positive / degenerate gap (keeps the log finite).
        good = d > 0
        if good.any():
            filled = np.where(good, d, float(REFERENCE_DT_SECONDS))
            out[1:] = filled
            out[0] = out[1]
    return out


def log_dt_offset(dt_seconds: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Map Δt (s) → ``log10(Δt / reference)`` — a centred, finite cadence scalar.

    The reference cadence maps to 0; a 10× faster cadence to −1, 10× slower to
    +1, so the ~250× corpus dynamic range becomes a small ``[-2.6, +0.4]``-ish
    offset.  Accepts numpy or torch and returns the same type.  Non-positive /
    non-finite Δt is clamped to the reference (offset 0) defensively.
    """
    ref = float(REFERENCE_DT_SECONDS)
    if isinstance(dt_seconds, torch.Tensor):
        dt = dt_seconds.to(torch.float64)
        dt = torch.where(torch.isfinite(dt) & (dt > 0), dt, torch.full_like(dt, ref))
        return torch.log10(dt / ref)
    dt = np.asarray(dt_seconds, dtype=np.float64)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, ref)
    return np.log10(dt / ref)


# ---------------------------------------------------------------------------
# The Δt → temporal-offset encoder
# ---------------------------------------------------------------------------


class TimescaleEncoder(nn.Module):
    """Map a per-frame log-Δt scalar to a ``d_model`` temporal-offset vector.

    A 2-layer MLP ``1 → hidden → d_model`` over the (already centred) log-Δt
    scalar.  The OUTPUT layer is ZERO-INIT so the encoder starts as the constant
    zero map: a fresh / OFF timescale head adds nothing and the model is
    byte-identical to the cadence-blind backbone, with the Δt signal earning
    influence through training (AdaLN-Zero discipline).

    ``forward`` takes ``(B, T)`` log-Δt offsets and returns ``(B, T, d)`` to ADD
    to the per-frame temporal position embedding of the CAMERA frames.
    """

    def __init__(self, d_model: int, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        # zero-init the output projection → the head starts as the zero map.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, log_dt: torch.Tensor) -> torch.Tensor:
        """``(B, T) log-Δt → (B, T, d)`` temporal offset (zero at init)."""
        h = self.fc1(log_dt.unsqueeze(-1).to(self.fc1.weight.dtype))
        return self.fc2(self.act(h))

    def zero_touch(self) -> torch.Tensor:
        """Zero-magnitude sum over the encoder params (DDP-uniform graph).

        A batch that supplies no Δt (the conditioning is off for that step) would
        leave these params grad-less and desync a DDP rank; touching them with a
        ``*0.0`` contribution keeps them in the autograd graph with no effect.
        """
        return (
            self.fc1.weight.sum()
            + self.fc1.bias.sum()
            + self.fc2.weight.sum()
            + self.fc2.bias.sum()
        ) * 0.0


def reference_log_dt() -> float:
    """The log-Δt offset of the reference cadence (0.0) — the neutral fill value.

    Conditioning prefix frames (signals / plan) and any frame with unknown Δt are
    fed this offset so the timescale head contributes the reference (slow-regime)
    interpretation rather than an arbitrary one.
    """
    return float(math.log10(1.0))  # == 0.0; named for call-site clarity


__all__ = [
    "CAMERA_IDS",
    "REFERENCE_CAMERA_INDEX",
    "REFERENCE_DT_SECONDS",
    "TimescaleEncoder",
    "camera_index",
    "camera_indices",
    "frame_dt_seconds",
    "log_dt_offset",
    "reference_log_dt",
]
