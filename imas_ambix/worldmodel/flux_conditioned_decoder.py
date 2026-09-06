"""Flux-conditioned token prediction and Nova camera-decoder adapter.

The token model preserves the camera model's factorised attention and tied
18-bit LFQ head.  Its only conditioning is the solved flux geometry: a spatial
adapter aligns the six-channel 64 by 64 rendering with the 16 by 16 token
positions, while a projected 12-value geometry vector occupies one auxiliary
spatial lane.  Runtime decoding is stateful because Nova presents one steering
frame at a time while the model consumes four camera-token history frames.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F  # noqa: N812

from imas_ambix.camdyn.dataset import frames_token_path, level1_shot_path
from imas_ambix.data.paths import TOKEN_ROOT
from imas_ambix.data.stream_encode import REGISTRY_OFFSET
from imas_ambix.worldmodel.flux_conditioning import (
    geometry_vector,
    render_flux_conditioning,
)
from imas_ambix.worldmodel.spacetime_model import (
    SpacetimeConfig,
    _nucleus_mask_logits,
    _SpaceTimeBlock,
)

CAMERA_VOCAB_SIZE = 1 << 18
TOKEN_GRID_SHAPE = (16, 16)
TOKEN_POSITION_COUNT = 16 * 16
FLUX_CHANNEL_COUNT = 6
FLUX_GRID_SHAPE = (64, 64)
GEOMETRY_VALUE_COUNT = 12
DEFAULT_HISTORY_FRAMES = 4
DEFAULT_CONDITION_DROPOUT = 0.1
DEFAULT_SESSION_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)

IntArray = NDArray[np.int64]
ImageArray = NDArray[np.uint8]


class DecodedFrame(NamedTuple):
    """One image returned through Nova's camera protocol."""

    image: object
    decode_wall: float
    decoder_identity: str


class FluxDecoderOutput(NamedTuple):
    """Next-frame hidden states and the sampled training dropout mask."""

    hidden: torch.Tensor
    condition_dropped: torch.Tensor


@dataclass(frozen=True, slots=True)
class FluxDecoderModelConfig:
    """Architecture settings for the flux-conditioned token predictor."""

    history_frames: int = DEFAULT_HISTORY_FRAMES
    vocab_size: int = CAMERA_VOCAB_SIZE
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.0
    condition_dropout: float = DEFAULT_CONDITION_DROPOUT

    def __post_init__(self) -> None:
        if self.history_frames != DEFAULT_HISTORY_FRAMES:
            raise ValueError("the decoder contract requires four history frames")
        if self.vocab_size != CAMERA_VOCAB_SIZE:
            raise ValueError("the decoder contract requires the 18-bit LFQ vocabulary")
        if self.d_model <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError("n_layers and d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.condition_dropout != DEFAULT_CONDITION_DROPOUT:
            raise ValueError("classifier-free condition dropout is fixed at 0.1")


class SpatialConditioningAdapter(nn.Module):
    """Align a six-channel 64 by 64 flux rendering to token positions."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(8, d_model // 2)
        self.layers = nn.Sequential(
            nn.Conv2d(FLUX_CHANNEL_COUNT, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, flux: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, 256, d_model)`` position-aligned embeddings."""
        embedded = self.layers(flux)
        if embedded.shape[-2:] != TOKEN_GRID_SHAPE:
            raise RuntimeError(
                f"conditioning adapter produced {embedded.shape[-2:]}, "
                f"expected {TOKEN_GRID_SHAPE}"
            )
        return embedded.flatten(2).transpose(1, 2)


class FluxConditionedTokenModel(nn.Module):
    """Predict one LFQ token frame from history and solved flux geometry."""

    def __init__(self, config: FluxDecoderModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        block_config = SpacetimeConfig(
            vocab_size=config.vocab_size,
            grid_h=TOKEN_GRID_SHAPE[0],
            grid_w=TOKEN_GRID_SHAPE[1],
            max_frames=config.history_frames,
            plan_channels=0,
            d_model=d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        self.token_embed = nn.Embedding(config.vocab_size, d_model)
        self.row_embed = nn.Embedding(TOKEN_GRID_SHAPE[0], d_model)
        self.col_embed = nn.Embedding(TOKEN_GRID_SHAPE[1], d_model)
        self.frame_embed = nn.Embedding(config.history_frames, d_model)
        self.flux_adapter = SpatialConditioningAdapter(d_model)
        self.geometry_projection = nn.Linear(GEOMETRY_VALUE_COUNT, d_model)
        self.geometry_marker = nn.Parameter(torch.zeros(d_model))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [_SpaceTimeBlock(block_config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _validate_inputs(
        self,
        history: torch.Tensor,
        flux: torch.Tensor,
        geometry: torch.Tensor,
    ) -> None:
        batch = history.shape[0] if history.ndim else -1
        expected_history = (
            batch,
            self.config.history_frames,
            *TOKEN_GRID_SHAPE,
        )
        if tuple(history.shape) != expected_history:
            actual_history = tuple(history.shape)
            raise ValueError(
                f"history must have shape {expected_history}, got {actual_history}"
            )
        expected_flux = (batch, FLUX_CHANNEL_COUNT, *FLUX_GRID_SHAPE)
        if tuple(flux.shape) != expected_flux:
            raise ValueError(
                f"flux must have shape {expected_flux}, got {tuple(flux.shape)}"
            )
        expected_geometry = (batch, GEOMETRY_VALUE_COUNT)
        if tuple(geometry.shape) != expected_geometry:
            raise ValueError(
                "geometry must have shape "
                f"{expected_geometry}, got {tuple(geometry.shape)}"
            )
        if history.dtype != torch.long:
            raise TypeError("history tokens must use torch.long")
        if bool((history < 0).any()) or bool((history >= self.config.vocab_size).any()):
            raise ValueError("history contains a token outside the LFQ vocabulary")

    def _drop_mask(
        self,
        batch: int,
        device: torch.device,
        *,
        force_condition: bool | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if force_condition is not None:
            return torch.full(
                (batch,), not force_condition, dtype=torch.bool, device=device
            )
        if not self.training:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        return (
            torch.rand(batch, device=device, generator=generator)
            < self.config.condition_dropout
        )

    def forward(
        self,
        history: torch.Tensor,
        flux: torch.Tensor,
        geometry: torch.Tensor,
        *,
        force_condition: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> FluxDecoderOutput:
        """Return hidden states predicting the next 16 by 16 token frame."""
        self._validate_inputs(history, flux, geometry)
        batch = history.shape[0]
        drop_mask = self._drop_mask(
            batch,
            history.device,
            force_condition=force_condition,
            generator=generator,
        )
        keep = (~drop_mask).to(flux.dtype).view(batch, 1, 1)
        spatial_condition = self.flux_adapter(flux) * keep
        geometry_condition = self.geometry_projection(geometry) * keep.squeeze(-1)

        frames = history.flatten(2)
        token_embeddings = self.token_embed(frames)
        rows = torch.arange(TOKEN_GRID_SHAPE[0], device=history.device)
        rows = rows.repeat_interleave(TOKEN_GRID_SHAPE[1])
        columns = torch.arange(TOKEN_GRID_SHAPE[1], device=history.device)
        columns = columns.repeat(TOKEN_GRID_SHAPE[0])
        spatial_positions = self.row_embed(rows) + self.col_embed(columns)
        token_embeddings = (
            token_embeddings
            + spatial_positions.view(1, 1, TOKEN_POSITION_COUNT, -1)
            + spatial_condition[:, None]
        )

        geometry_token = geometry_condition + self.geometry_marker
        geometry_token = geometry_token[:, None, None, :].expand(
            -1, self.config.history_frames, 1, -1
        )
        sequence = torch.cat([token_embeddings, geometry_token], dim=2)
        frame_positions = self.frame_embed(
            torch.arange(self.config.history_frames, device=history.device)
        )
        sequence = sequence + frame_positions.view(
            1, self.config.history_frames, 1, self.config.d_model
        )
        sequence = self.drop(sequence)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = self.final_norm(sequence)
        hidden = sequence[:, -1, :TOKEN_POSITION_COUNT]
        return FluxDecoderOutput(hidden=hidden, condition_dropped=drop_mask)

    def chunked_nll(
        self,
        history: torch.Tensor,
        flux: torch.Tensor,
        geometry: torch.Tensor,
        target: torch.Tensor,
        *,
        chunk: int = 32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return next-frame cross entropy without materialising all logits."""
        output = self(history, flux, geometry, generator=generator)
        expected_target = (history.shape[0], *TOKEN_GRID_SHAPE)
        if tuple(target.shape) != expected_target or target.dtype != torch.long:
            raise ValueError(f"target must be torch.long with shape {expected_target}")
        flat_hidden = output.hidden.reshape(-1, self.config.d_model)
        flat_target = target.reshape(-1)
        total = output.hidden.new_zeros(())
        for start in range(0, flat_hidden.shape[0], chunk):
            stop = min(start + chunk, flat_hidden.shape[0])
            total = total + F.cross_entropy(
                self.head(flat_hidden[start:stop]),
                flat_target[start:stop],
                reduction="sum",
            )
        return total / flat_hidden.shape[0]

    @torch.no_grad()
    def sample_next_frame(
        self,
        history: torch.Tensor,
        flux: torch.Tensor,
        geometry: torch.Tensor,
        *,
        guidance_weight: float,
        temperature: float = 1.0,
        top_p: float = 0.95,
        chunk: int = 32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample local LFQ ids using classifier-free flux guidance."""
        if not np.isfinite(guidance_weight) or guidance_weight < 0.0:
            raise ValueError("guidance_weight must be finite and non-negative")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive; decode samples")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

        conditioned = None
        unconditioned = None
        if guidance_weight != 0.0:
            conditioned = self(
                history, flux, geometry, force_condition=True
            ).hidden.reshape(-1, self.config.d_model)
        if guidance_weight != 1.0:
            unconditioned = self(
                history, flux, geometry, force_condition=False
            ).hidden.reshape(-1, self.config.d_model)
        reference = conditioned if conditioned is not None else unconditioned
        if reference is None:
            raise RuntimeError("guided decoding produced no hidden state")
        sampled = torch.empty(
            reference.shape[0], dtype=torch.long, device=reference.device
        )
        generator_device = generator.device if generator is not None else None
        for start in range(0, reference.shape[0], chunk):
            stop = min(start + chunk, reference.shape[0])
            conditional_logits = (
                self.head(conditioned[start:stop]).float()
                if conditioned is not None
                else None
            )
            unconditional_logits = (
                self.head(unconditioned[start:stop]).float()
                if unconditioned is not None
                else None
            )
            if unconditional_logits is None:
                if conditional_logits is None:
                    raise RuntimeError("guided decoding produced no logits")
                logits = conditional_logits
            elif conditional_logits is None:
                logits = unconditional_logits
            else:
                logits = unconditional_logits + float(guidance_weight) * (
                    conditional_logits - unconditional_logits
                )
            logits = logits / float(temperature)
            if top_p < 1.0:
                logits = _nucleus_mask_logits(logits, top_p)
            probabilities = torch.softmax(logits, dim=-1)
            if (
                generator_device is not None
                and generator_device != probabilities.device
            ):
                choice = torch.multinomial(
                    probabilities.to(generator_device), 1, generator=generator
                ).to(sampled.device)
            else:
                choice = torch.multinomial(probabilities, 1, generator=generator)
            sampled[start:stop] = choice.squeeze(-1)
        return sampled.reshape(history.shape[0], *TOKEN_GRID_SHAPE)


class _StubVQDecoder:
    """Deterministic unit-test VQ stage selected explicitly in runtime JSON."""

    def decode(self, tokens: IntArray) -> ImageArray:
        values = np.bitwise_and(tokens, 255).astype(np.uint8)
        image = np.repeat(np.repeat(values, 16, axis=0), 16, axis=1)
        return np.repeat(image[..., None], 3, axis=2)


class _OpenMagvitVQDecoder:
    """Frozen OpenMAGVIT2 decoder loaded once through the benchmark route."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        from imas_ambix.bench.stream_worker import load_model  # noqa: PLC0415

        self.device = device
        expected = checkpoint.parent.parent / "weights" / "imagenet_256_L.ckpt"
        if checkpoint.resolve() != expected.resolve():
            raise ValueError(
                "OpenMAGVIT2 decoding requires the frozen imagenet_256_L "
                f"checkpoint at {expected}"
            )
        self.model = load_model(checkpoint.parent.parent, device)

    def decode(self, tokens: IntArray) -> ImageArray:
        from imas_ambix.bench.stream_worker import decode_batch  # noqa: PLC0415

        return decode_batch(
            self.model,
            tokens[None],
            self.device,
            model_forward_batch=1,
            target_hw=(256, 256),
        )[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError("decoder checkpoint must contain a mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError("decoder checkpoint contains no model state dictionary")


def _model_config(
    payload: Mapping[str, object], runtime: Mapping[str, object]
) -> FluxDecoderModelConfig:
    candidate = payload.get("model_config", runtime.get("model_config", {}))
    if not isinstance(candidate, Mapping):
        raise ValueError("model_config must be a JSON object")
    return FluxDecoderModelConfig(**dict(candidate))


def _runtime_configuration() -> tuple[Path, dict[str, object]]:
    variable = "IMAS_AMBIX_FLUX_DECODER"
    configured = os.environ.get(variable)
    if not configured:
        raise RuntimeError(f"{variable} must name the decoder runtime JSON")
    path = Path(configured).expanduser()
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("decoder runtime JSON must contain an object")
    return path, value


def _required_path(configuration: Mapping[str, object], name: str) -> Path:
    value = configuration.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"decoder runtime configuration requires {name!r}")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class FluxConditionedDecoder:
    """Stateful implementation of Nova's one-frame ``FrameDecoder`` seam."""

    decoder_identity: str

    def __init__(self) -> None:
        _configuration_path, configuration = _runtime_configuration()
        checkpoint = _required_path(configuration, "checkpoint")
        vq_checkpoint = _required_path(configuration, "vq_decoder_path")
        raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(raw_payload, Mapping):
            raise ValueError("decoder checkpoint must contain a mapping")
        model_config = _model_config(raw_payload, configuration)
        requested_device = str(configuration.get("device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        self.model = FluxConditionedTokenModel(model_config)
        self.model.load_state_dict(_state_dict(raw_payload), strict=True)
        self.model.to(self.device).eval()

        corpus_digest = configuration.get(
            "corpus_digest", raw_payload.get("corpus_digest")
        )
        if not isinstance(corpus_digest, str) or not corpus_digest:
            raise ValueError("corpus_digest is required in runtime JSON or checkpoint")
        vq_decoder_id = str(configuration.get("vq_decoder_id", vq_checkpoint.stem))
        self.decoder_identity = f"{_sha256(checkpoint)}:{vq_decoder_id}:{corpus_digest}"
        self.guidance_weight = float(configuration["guidance_weight"])
        if not np.isfinite(self.guidance_weight) or self.guidance_weight < 0.0:
            raise ValueError("guidance_weight must be finite and non-negative")
        self.temperature = float(configuration.get("temperature", 1.0))
        self.top_p = float(configuration.get("top_p", 0.95))
        self.sample_chunk = int(configuration.get("sample_chunk", 32))
        if self.sample_chunk <= 0:
            raise ValueError("sample_chunk must be positive")
        self.seed_shot = int(configuration["seed_shot"])
        self.seed_slice = int(configuration["seed_slice"])
        self.session_root = Path(
            str(configuration.get("session_root", DEFAULT_SESSION_ROOT))
        )
        self.token_root = Path(str(configuration.get("token_root", TOKEN_ROOT)))
        sample_seed = int(configuration.get("sample_seed", 0))
        self.generator = torch.Generator(device=self.device.type).manual_seed(
            sample_seed
        )

        vq_stage = str(configuration.get("vq_stage", "open-magvit2"))
        if vq_stage == "open-magvit2":
            self.vq_decoder: Any = _OpenMagvitVQDecoder(vq_checkpoint, str(self.device))
        elif vq_stage == "stub":
            self.vq_decoder = _StubVQDecoder()
        else:
            raise ValueError("vq_stage must be 'open-magvit2' or 'stub'")
        self._history: list[IntArray] = []
        self.reset(self._configured_seed_frames())

    def _configured_seed_frames(self) -> IntArray:
        import xarray as xr  # noqa: PLC0415
        import zarr  # noqa: PLC0415

        session_path = self.session_root / f"{self.seed_shot}.nc"
        with xr.open_dataset(session_path, group="steering") as session:
            if not 0 <= self.seed_slice < session.sizes["time"]:
                raise IndexError(
                    f"seed_slice {self.seed_slice} outside {session.sizes['time']} rows"
                )
            seed_time = float(session["time"].isel(time=self.seed_slice).item())

        source = zarr.open_group(str(level1_shot_path(self.seed_shot)), mode="r")
        times = np.asarray(source["rbb"]["time"], dtype=np.float64)
        token_store = zarr.open_group(
            str(frames_token_path(self.seed_shot, "rbb", token_root=self.token_root)),
            mode="r",
        )
        token_count = int(token_store["tokens"].shape[0])
        if times.shape[0] != token_count:
            raise ValueError(
                "seed camera times and token frames have different lengths"
            )
        frame_index = int(np.argmin(np.abs(times - seed_time)))
        first = max(0, frame_index - self.model.config.history_frames + 1)
        stored = np.asarray(
            token_store["tokens"][first : frame_index + 1], dtype=np.int64
        )
        if stored.shape[0] < self.model.config.history_frames:
            padding = np.repeat(
                stored[:1], self.model.config.history_frames - stored.shape[0], axis=0
            )
            stored = np.concatenate([padding, stored], axis=0)
        local = np.clip(stored - REGISTRY_OFFSET, 0, CAMERA_VOCAB_SIZE - 1)
        return local.astype(np.int64, copy=False)

    def reset(self, seed_frames: Sequence[object] | np.ndarray | None = None) -> None:
        """Clear history, or replace it with four real-token seed frames.

        Construction supplies the configured base-shot seed.  A caller may pass
        another ``(4, 16, 16)`` local-token array; passing ``None`` or an empty
        sequence clears the history.
        """
        if seed_frames is None:
            self._history = []
            return
        frames = np.asarray(seed_frames, dtype=np.int64)
        if frames.size == 0:
            self._history = []
            return
        expected = (self.model.config.history_frames, *TOKEN_GRID_SHAPE)
        if frames.shape == (self.model.config.history_frames, TOKEN_POSITION_COUNT):
            frames = frames.reshape(expected)
        if frames.shape != expected:
            raise ValueError(
                f"seed_frames must have shape {expected}, got {frames.shape}"
            )
        if np.any(frames < 0) or np.any(frames >= CAMERA_VOCAB_SIZE):
            raise ValueError("seed_frames contains a token outside the LFQ vocabulary")
        self._history = [frame.copy() for frame in frames]

    def decode(self, frame: object) -> DecodedFrame:
        """Render flux geometry, sample one token frame, and VQ-decode it."""
        started = perf_counter()
        if len(self._history) != self.model.config.history_frames:
            raise RuntimeError("decoder history is empty; call reset with seed frames")
        history = torch.as_tensor(
            np.stack(self._history), dtype=torch.long, device=self.device
        )[None]
        flux = torch.as_tensor(
            render_flux_conditioning(frame), dtype=torch.float32, device=self.device
        )[None]
        geometry = torch.as_tensor(
            geometry_vector(frame), dtype=torch.float32, device=self.device
        )[None]
        with torch.inference_mode():
            predicted = self.model.sample_next_frame(
                history,
                flux,
                geometry,
                guidance_weight=self.guidance_weight,
                temperature=self.temperature,
                top_p=self.top_p,
                chunk=self.sample_chunk,
                generator=self.generator,
            )[0]
        tokens = predicted.cpu().numpy().astype(np.int64, copy=False)
        self._history = [*self._history[1:], tokens.copy()]
        rgb = np.asarray(self.vq_decoder.decode(tokens), dtype=np.uint8)
        if rgb.shape != (256, 256, 3):
            raise ValueError(f"VQ decoder returned {rgb.shape}, expected (256, 256, 3)")
        monochrome = np.rint(rgb.astype(np.float32).mean(axis=2)).astype(np.uint8)
        image = np.repeat(monochrome[..., None], 3, axis=2)
        elapsed = max(perf_counter() - started, np.finfo(np.float64).eps)
        return DecodedFrame(image, elapsed, self.decoder_identity)


def checkpoint_payload(
    model: FluxConditionedTokenModel, *, corpus_digest: str
) -> dict[str, object]:
    """Return the stable checkpoint mapping consumed by the protocol class."""
    if not corpus_digest:
        raise ValueError("corpus_digest must be non-empty")
    return {
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "corpus_digest": corpus_digest,
    }


__all__ = [
    "DecodedFrame",
    "FluxConditionedDecoder",
    "FluxConditionedTokenModel",
    "FluxDecoderModelConfig",
    "FluxDecoderOutput",
    "SpatialConditioningAdapter",
    "checkpoint_payload",
]
