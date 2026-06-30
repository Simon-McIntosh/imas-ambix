"""Multi-rate, phase-preserving token store (schema generation 2).

The generation-1 store (:mod:`imas_ambix.data.persist`) resamples every
modality onto a single 100 Hz model grid (see
:data:`imas_ambix.tokenizer.alignment.MODEL_HZ_DEFAULT`).  A 100 Hz grid
has a 50 Hz Nyquist limit — it **destroys** the kHz/MHz fluctuation
content that the high-frequency MAST diagnostics (xma fast magnetics,
xim Dα/CII, xsx soft X-ray) carry.  The interior-information study showed
that phase / mode structure lives precisely in that high-frequency band.

This store keeps every modality at its **native token cadence**.  Each
file records, per token position, an explicit time coordinate; per channel
a validity mask (never silent zero-fill); and a ``phase_preserving`` flag
so a consumer can tell a phase-faithful representation (complex STFT,
patch-transformer codes) from a magnitude-only one.

Layout (rooted at ``TOKEN_ROOT/v2``)::

    mast-tokens/
      v2/
        signals_hf/
          {shot_id}/
            {group}.zarr   # native-cadence tokens + per-token time + masks
        registry.json      # the v2 token-id allocation manifest

Each ``{group}.zarr`` group contains:

============== ============================== =================================
array          shape / dtype                  meaning
============== ============================== =================================
``tokens``     ``(n_tokens, n_channels)`` i32 global token ids (registry-shifted)
``token_time`` ``(n_tokens,)`` float64        time coordinate of each token (s)
``valid``      ``(n_tokens, n_channels)`` bool per-token-per-channel validity
============== ============================== =================================

and these required ``.attrs`` (a :class:`StoreV2Attrs`)::

    tokenizer_name, vocab_version, native_rate_hz, token_rate_hz,
    n_channels, channel_names, phase_preserving, original_window

``native_rate_hz`` is the sampling rate of the *raw* signal the tokens
were produced from; ``token_rate_hz`` is the cadence of the emitted
tokens (raw_rate / patch_stride).  ``original_window`` is the
``[t_start, t_end]`` of the raw signal window so a decoder can place the
reconstruction back on the absolute time axis.

This module is the **frozen schema dependency root**: the writer
(:func:`save_signal_hf_tokens`), the reader
(:func:`load_signal_hf_tokens`), and the on-disk attribute contract live
here and must not change shape under a running encode.  Bump
:data:`imas_ambix.tokenizer.registry.VOCAB_VERSION` and re-encode if the
contract has to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from imas_ambix.data.paths import TOKEN_ROOT

if TYPE_CHECKING:
    from pathlib import Path

# Store schema generation.  Distinct from the vocab version (which tracks
# the token-id allocation): the store generation tracks the on-disk array /
# attribute *layout*.  v2 == native-cadence, phase-preserving.
STORE_GENERATION = "v2"

# The required attribute keys.  A reader validates against this set so a
# half-written or stale-schema store is rejected loudly rather than read
# with missing fields silently defaulted.
REQUIRED_ATTRS: tuple[str, ...] = (
    "tokenizer_name",
    "vocab_version",
    "native_rate_hz",
    "token_rate_hz",
    "n_channels",
    "channel_names",
    "phase_preserving",
    "original_window",
)


@dataclass(frozen=True)
class StoreV2Attrs:
    """The required attribute block for one ``signals_hf`` token group.

    Mirrors the on-disk ``.attrs`` contract.  Constructing this object
    validates field types so an inconsistent attribute set fails at write
    time, not read time.
    """

    tokenizer_name: str
    vocab_version: str
    native_rate_hz: float
    token_rate_hz: float
    n_channels: int
    channel_names: tuple[str, ...]
    phase_preserving: bool
    original_window: tuple[float, float]
    # Free-form per-tokenizer metadata (codebook params, patch size, …).
    # Stored JSON-encoded; never part of the required contract.
    metadata: dict[str, object] = field(default_factory=dict)
    # Optional companion descriptors for the per-channel geometry array (see
    # SignalHFTokens.geometry).  Empty for a geometry-less store.  NOT part of
    # REQUIRED_ATTRS — an old store missing these loads with both empty, so
    # backward compatibility is unconditional.  ``geometry_feature_names`` names
    # the ``(n_channels, n_geom_features)`` columns; ``geometry_sensor_kinds``
    # is one categorical kind per channel (parallel to ``channel_names``).
    geometry_feature_names: tuple[str, ...] = ()
    geometry_sensor_kinds: tuple[str, ...] = ()
    # Normalisation regime the tokens were written under: "absolute" (corpus-
    # calibrated — a physical value maps to the same token everywhere) or
    # "per_shot" (per-window z-scored — magnitude not preserved).  Optional /
    # backward-compatible: a legacy store omits it and loads as "per_shot".
    calibration_mode: str = "per_shot"

    def __post_init__(self) -> None:
        if self.n_channels != len(self.channel_names):
            raise ValueError(
                f"n_channels={self.n_channels} disagrees with "
                f"{len(self.channel_names)} channel_names"
            )
        if self.native_rate_hz <= 0 or self.token_rate_hz <= 0:
            raise ValueError(
                f"rates must be positive: native={self.native_rate_hz}, "
                f"token={self.token_rate_hz}"
            )
        if len(self.original_window) != 2:
            raise ValueError("original_window must be (t_start, t_end)")

    def to_attrs(self) -> dict[str, object]:
        """Return the JSON-safe ``.attrs`` dict written to Zarr.

        Emits the geometry companion descriptors ONLY when they are present, so
        a geometry-less store's on-disk attrs are byte-identical to the legacy
        layout and a reader that never saw geometry is unaffected.
        """
        out: dict[str, object] = {
            "tokenizer_name": self.tokenizer_name,
            "vocab_version": self.vocab_version,
            "store_generation": STORE_GENERATION,
            "native_rate_hz": float(self.native_rate_hz),
            "token_rate_hz": float(self.token_rate_hz),
            "n_channels": int(self.n_channels),
            "channel_names": list(self.channel_names),
            "phase_preserving": bool(self.phase_preserving),
            "original_window": [
                float(self.original_window[0]),
                float(self.original_window[1]),
            ],
            "metadata": json.dumps(_json_safe(self.metadata)),
            "calibration_mode": str(self.calibration_mode),
        }
        if self.geometry_feature_names:
            out["geometry_feature_names"] = list(self.geometry_feature_names)
        if self.geometry_sensor_kinds:
            out["geometry_sensor_kinds"] = list(self.geometry_sensor_kinds)
        return out

    @classmethod
    def from_attrs(cls, attrs: dict) -> StoreV2Attrs:
        """Reconstruct from an on-disk ``.attrs`` dict (validates contract)."""
        missing = [k for k in REQUIRED_ATTRS if k not in attrs]
        if missing:
            raise ValueError(f"store v2 attrs missing required keys: {missing}")
        meta_raw = attrs.get("metadata", "{}")
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
        win = attrs["original_window"]
        # Geometry companion descriptors are OPTIONAL — a legacy store omits
        # them and they default to empty (geometry=None on load), so an old
        # store loads with the identical contract it was written under.
        geom_features = attrs.get("geometry_feature_names", ())
        geom_kinds = attrs.get("geometry_sensor_kinds", ())
        return cls(
            tokenizer_name=str(attrs["tokenizer_name"]),
            vocab_version=str(attrs["vocab_version"]),
            native_rate_hz=float(attrs["native_rate_hz"]),
            token_rate_hz=float(attrs["token_rate_hz"]),
            n_channels=int(attrs["n_channels"]),
            channel_names=tuple(str(c) for c in attrs["channel_names"]),
            phase_preserving=bool(attrs["phase_preserving"]),
            original_window=(float(win[0]), float(win[1])),
            metadata=meta,
            geometry_feature_names=tuple(str(f) for f in geom_features),
            geometry_sensor_kinds=tuple(str(k) for k in geom_kinds),
            calibration_mode=str(attrs.get("calibration_mode", "per_shot")),
        )


@dataclass(frozen=True)
class SignalHFTokens:
    """In-memory representation of one native-cadence token group.

    Returned by :func:`load_signal_hf_tokens` and accepted (as its
    components) by :func:`save_signal_hf_tokens`.
    """

    tokens: np.ndarray  # (n_tokens, n_channels) int32 — global ids
    token_time: np.ndarray  # (n_tokens,) float64 — per-token time (s)
    valid: np.ndarray  # (n_tokens, n_channels) bool — per-channel validity
    attrs: StoreV2Attrs
    # Optional per-token-per-channel continuous embedding.  When the chosen
    # tokenizer is a continuous-embedding+mask bottleneck (no quantisation),
    # the discrete ``tokens`` are vestigial and the phase-preserving payload
    # lives here, shape ``(n_tokens, n_channels, embed_dim)`` float32.  ``None``
    # for a quantised tokenizer whose ``tokens`` carry the full information.
    embedding: np.ndarray | None = None
    # Optional per-CHANNEL sensor-geometry positional encoding, shape
    # ``(n_channels, n_geom_features)`` float32, aligned 1:1 with
    # ``attrs.channel_names``.  Carries each channel's apparatus geometry
    # (sensor R/Z/phi, orientation, line-of-sight chord endpoints) — the
    # positional encoding a machine-agnostic model attends with.  Companion
    # ``attrs.geometry_feature_names`` names the columns and
    # ``attrs.geometry_sensor_kinds`` the per-channel categorical kind.  ``None``
    # for a store written without geometry (every legacy v2 store) — geometry is
    # OPTIONAL and a geometry-less store loads exactly as before.
    geometry: np.ndarray | None = None

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.tokens.shape[1]) if self.tokens.ndim == 2 else 0


def _json_safe(obj: object) -> object:
    """Recursively coerce to a JSON-serialisable form (numpy-aware)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def signal_hf_token_path(
    shot_id: int, group: str, store_generation: str = STORE_GENERATION
) -> Path:
    """Canonical Zarr path for one shot's native-cadence signal-HF tokens.

    ``TOKEN_ROOT/{store_generation}/signals_hf/{shot_id}/{group}.zarr``.
    The path may not yet exist on disk.
    """
    return TOKEN_ROOT / store_generation / "signals_hf" / str(shot_id) / f"{group}.zarr"


def registry_v2_path(store_generation: str = STORE_GENERATION) -> Path:
    """Path to the v2 token-id allocation manifest."""
    return TOKEN_ROOT / store_generation / "registry.json"


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


def save_signal_hf_tokens(
    shot_id: int,
    group: str,
    tokens: np.ndarray,
    token_time: np.ndarray,
    valid: np.ndarray,
    attrs: StoreV2Attrs,
    *,
    embedding: np.ndarray | None = None,
    geometry: np.ndarray | None = None,
    store_generation: str = STORE_GENERATION,
) -> Path:
    """Write one native-cadence, phase-preserving token group to Zarr.

    Validates that the token / time / mask shapes agree with each other and
    with ``attrs`` **before** any data is written, so a malformed group is
    never persisted.

    Parameters
    ----------
    tokens:
        ``(n_tokens, n_channels)`` int32 global token ids.
    token_time:
        ``(n_tokens,)`` per-token time coordinate (s).  Never resampled to a
        model grid — these are the native token cadence times.
    valid:
        ``(n_tokens, n_channels)`` bool validity mask.  ``False`` marks a
        token position whose channel had no usable data; consumers must
        honour the mask rather than treat zeros as real readings.
    attrs:
        The required attribute block.
    embedding:
        Optional ``(n_tokens, n_channels, embed_dim)`` float32 continuous
        latent.  Written when the chosen tokenizer is a continuous-embedding
        bottleneck whose discrete ``tokens`` are vestigial — the
        phase-preserving payload then lives here.
    geometry:
        Optional ``(n_channels, n_geom_features)`` float32 per-channel
        sensor-geometry positional encoding, aligned 1:1 with
        ``attrs.channel_names``.  When given, ``attrs.geometry_feature_names``
        must name its columns (a downstream consumer reads the array cheaply
        without JSON-parsing).  ``None`` writes no geometry array and the
        on-disk store is byte-identical to a legacy geometry-less store.

    Returns
    -------
    Path
        The written ``.zarr`` path.
    """
    import zarr

    tokens = np.asarray(tokens, dtype=np.int32)
    token_time = np.asarray(token_time, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool)

    if tokens.ndim != 2:
        raise ValueError(
            f"tokens must be 2-D (n_tokens, n_channels); got {tokens.shape}"
        )
    n_tok, n_ch = tokens.shape
    if token_time.shape[0] != n_tok:
        raise ValueError(f"token_time length {token_time.shape[0]} != n_tokens {n_tok}")
    if valid.shape != tokens.shape:
        raise ValueError(f"valid shape {valid.shape} != tokens shape {tokens.shape}")
    if attrs.n_channels != n_ch:
        raise ValueError(
            f"attrs.n_channels {attrs.n_channels} != tokens n_channels {n_ch}"
        )
    if embedding is not None:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim != 3 or embedding.shape[:2] != (n_tok, n_ch):
            raise ValueError(
                f"embedding shape {embedding.shape} must be "
                f"(n_tokens={n_tok}, n_channels={n_ch}, embed_dim)"
            )
    if geometry is not None:
        geometry = np.asarray(geometry, dtype=np.float32)
        if geometry.ndim != 2 or geometry.shape[0] != n_ch:
            raise ValueError(
                f"geometry shape {geometry.shape} must be "
                f"(n_channels={n_ch}, n_geom_features)"
            )
        n_feat = geometry.shape[1]
        if attrs.geometry_feature_names and len(attrs.geometry_feature_names) != n_feat:
            raise ValueError(
                f"geometry has {n_feat} feature columns but "
                f"attrs.geometry_feature_names names "
                f"{len(attrs.geometry_feature_names)}"
            )
        if attrs.geometry_sensor_kinds and len(attrs.geometry_sensor_kinds) != n_ch:
            raise ValueError(
                f"geometry has {n_ch} channels but attrs.geometry_sensor_kinds "
                f"names {len(attrs.geometry_sensor_kinds)}"
            )

    path = signal_hf_token_path(shot_id, group, store_generation)
    path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=tokens)
    store.create_array("token_time", data=token_time)
    store.create_array("valid", data=valid)
    if embedding is not None:
        store.create_array("embedding", data=embedding)
    if geometry is not None:
        store.create_array("geometry", data=geometry)
    out_attrs = attrs.to_attrs()
    out_attrs.update(
        {
            "shot_id": int(shot_id),
            "group": str(group),
            "has_embedding": embedding is not None,
            "has_geometry": geometry is not None,
        }
    )
    store.attrs.update(out_attrs)
    return path


def load_signal_hf_tokens(
    shot_id: int, group: str, *, store_generation: str = STORE_GENERATION
) -> SignalHFTokens:
    """Read one native-cadence token group back from Zarr.

    Validates the required-attribute contract on read; a store missing any
    required key raises rather than silently defaulting.
    """
    import zarr

    path = signal_hf_token_path(shot_id, group, store_generation)
    store = zarr.open_group(str(path), mode="r")
    attrs = StoreV2Attrs.from_attrs(dict(store.attrs))
    arrays = set(store.array_keys())
    embedding = (
        np.asarray(store["embedding"], dtype=np.float32)
        if "embedding" in arrays
        else None
    )
    # Geometry is optional — absent for every legacy store, where it loads None.
    geometry = (
        np.asarray(store["geometry"], dtype=np.float32)
        if "geometry" in arrays
        else None
    )
    return SignalHFTokens(
        tokens=np.asarray(store["tokens"], dtype=np.int32),
        token_time=np.asarray(store["token_time"], dtype=np.float64),
        valid=np.asarray(store["valid"], dtype=bool),
        attrs=attrs,
        embedding=embedding,
        geometry=geometry,
    )
