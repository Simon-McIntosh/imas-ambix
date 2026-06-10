"""Camera-Dynamics World Model — D0 substrate.

The CPU-only foundation that gates the GPU dynamics arms of the
``camera-dynamics-wm-v0`` plan.  Everything here is token-space and
decoder-free; no model exists yet.  The four pillars are:

``dataset``
    A FRAME-GRID-PRESERVING token-stream dataset over the 9,527-shot rbb
    corpus.  Unlike :class:`imas_ambix.data.loaders.ShotTokenDataset`
    (which flattens frames into a 1-D world-model stream), this dataset
    yields per-sample windows of shape ``(n_frames, 16, 16)`` int32 with
    the per-frame 16×16 token grid intact, plus per-frame timestamps and
    Δt joined from the level-1 ``rbb/time`` axis.

``masking``
    The clip-mask sampler covering all §4a modes (random position+size,
    moving/panning, temporal frontier, full mask) PLUS a frozen,
    deterministic named-geometry eval suite so every arm and ablation is
    scored on identical reconstruction tasks.

``conditioning``
    Actuator/scalar loaders (coil/feed/sol/tf currents, NBI powers,
    gas-puff flows, plasma current, line-integrated density, optional
    Dα) held to the rbb frame times in PHYSICAL units, with per-channel
    missingness flags.  EFIT (efm/esm) and pulse-schedule (xdc) signals
    are BANNED as inputs/conditioning everywhere — they embed the
    reconstruction (leakage).

``metrics``
    The PRE-REGISTERED scoring interfaces that lock W1/W2/W3 before any
    model number exists: masked-token NLL + top-1 accuracy (with a
    bootstrap-CI helper), horizon-h reconstruction, a motion-weighted
    token subset, and the frozen-probe protocol.  rFID is banned as a
    primary metric (S5 lesson).

``splits``
    Glue that builds the shot-level split manifest over shots that
    actually carry rbb tokens, forcing the 112 MSE held-out shots into
    the held-out split for comparability with the S9/S12 oracles.

Time-grid recommendation (D0's call, plan §7)
---------------------------------------------
The dataset preserves each frame's native ``rbb/time`` timestamp and the
per-frame Δt rather than resampling to a common grid.  This keeps BOTH
downstream options open — native-Δt conditioning and a later
resample-to-common-grid pass — without baking either in.  **D0
recommends native frames + Δt conditioning** for D1/D2: the rbb cadence
is per-shot stable (~600 Hz on the reference shot, Δt ≈ 1.67 ms) but
varies shot-to-shot, and resampling the token stream would require
re-tokenising interpolated pixel frames (the decoder is frozen and
re-encode is the expensive corpus pass we are avoiding).  Conditioning
on the measured Δt lets a single model absorb the cadence heterogeneity
with no resampling artefact and no information loss.  The ``resample-common``
option remains reachable: callers can resample the *conditioning* traces
onto any grid via :func:`conditioning.resample_to_frames` — only the
camera token stream is held native.
"""

from __future__ import annotations

from imas_ambix.camdyn.conditioning import (
    BANNED_CONDITIONING_SOURCES,
    CONDITIONING_CHANNELS,
    ConditioningChannel,
    ConditioningSample,
    assert_no_leakage_sources,
    load_conditioning,
)
from imas_ambix.camdyn.dataset import (
    FRAME_GRID,
    VOCAB_SIZE,
    FrameTokenDataset,
    FrameTokenShotSpec,
    FrameWindow,
    FrameWindowConfig,
    discover_token_shots,
)
from imas_ambix.camdyn.masking import (
    NAMED_GEOMETRIES,
    ClipMaskConfig,
    MaskMode,
    NamedGeometry,
    named_geometry_mask,
    sample_clip_mask,
)
from imas_ambix.camdyn.metrics import (
    ProbeProtocol,
    bootstrap_ci,
    crps_gaussian,
    horizon_frame_offsets,
    horizon_reconstruction_accuracy,
    masked_token_nll,
    masked_top1_accuracy,
    motion_weighted_subset,
    probe_rmse,
)
from imas_ambix.camdyn.splits import (
    CamdynSplit,
    build_camdyn_split,
    load_mse_heldout_shots,
)

__all__ = [
    # dataset
    "FRAME_GRID",
    "VOCAB_SIZE",
    "FrameWindow",
    "FrameWindowConfig",
    "FrameTokenDataset",
    "FrameTokenShotSpec",
    "discover_token_shots",
    # masking
    "NAMED_GEOMETRIES",
    "ClipMaskConfig",
    "MaskMode",
    "NamedGeometry",
    "named_geometry_mask",
    "sample_clip_mask",
    # conditioning
    "BANNED_CONDITIONING_SOURCES",
    "CONDITIONING_CHANNELS",
    "ConditioningChannel",
    "ConditioningSample",
    "assert_no_leakage_sources",
    "load_conditioning",
    # metrics
    "ProbeProtocol",
    "bootstrap_ci",
    "crps_gaussian",
    "horizon_frame_offsets",
    "horizon_reconstruction_accuracy",
    "masked_token_nll",
    "masked_top1_accuracy",
    "motion_weighted_subset",
    "probe_rmse",
    # splits
    "CamdynSplit",
    "build_camdyn_split",
    "load_mse_heldout_shots",
]
