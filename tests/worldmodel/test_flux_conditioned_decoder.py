from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from apps.playable.camera import DecodedFrame as NovaDecodedFrame

from imas_ambix.worldmodel.flux_conditioned_decoder import (
    CAMERA_VOCAB_SIZE,
    DecodedFrame,
    FluxConditionedDecoder,
    FluxConditionedTokenModel,
    FluxDecoderModelConfig,
    _decoded_frame_class,
    checkpoint_payload,
)
from imas_ambix.worldmodel.spacetime_model import _SpaceTimeBlock

SESSION_PATH = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/21858.nc"
)
OPEN_MAGVIT_CHECKPOINT = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/weights/imagenet_256_L.ckpt"
)


def _model_config() -> FluxDecoderModelConfig:
    return FluxDecoderModelConfig(
        d_model=4,
        n_layers=1,
        n_heads=1,
        d_ff=8,
        dropout=0.0,
    )


def _random_inputs(
    config: FluxDecoderModelConfig, *, batch: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    history = torch.randint(
        0,
        config.vocab_size,
        (batch, config.history_frames, 16, 16),
        generator=generator,
    )
    flux = torch.rand((batch, 6, 64, 64), generator=generator)
    geometry = torch.rand((batch, 12), generator=generator)
    return history, flux, geometry


def test_random_batch_forward_preserves_spatial_prediction_shape() -> None:
    config = _model_config()
    model = FluxConditionedTokenModel(config).train()
    history, flux, geometry = _random_inputs(config)

    output = model(history, flux, geometry, force_condition=True)

    assert output.hidden.shape == (2, 16 * 16, config.d_model)
    assert output.condition_dropped.shape == (2,)
    assert not output.condition_dropped.any()
    assert model.flux_adapter(flux).shape == (2, 16 * 16, config.d_model)
    assert isinstance(model.blocks[0], _SpaceTimeBlock)
    assert model.head.out_features == CAMERA_VOCAB_SIZE
    assert model.head.weight.data_ptr() == model.token_embed.weight.data_ptr()
    assert config.history_frames == 4
    assert config.condition_dropout == 0.1


def test_classifier_free_path_drops_both_spatial_and_geometry_condition() -> None:
    config = _model_config()
    model = FluxConditionedTokenModel(config).eval()
    history, flux, geometry = _random_inputs(config, batch=1)

    without_a = model(history, flux, geometry, force_condition=False).hidden.detach()
    without_b = model(
        history,
        flux + 3.0,
        geometry - 7.0,
        force_condition=False,
    ).hidden.detach()
    with_a = model(history, flux, geometry, force_condition=True).hidden.detach()
    with_b = model(
        history,
        flux + 3.0,
        geometry - 7.0,
        force_condition=True,
    ).hidden.detach()

    assert torch.equal(without_a, without_b)
    assert not torch.allclose(with_a, with_b)


def test_decode_refuses_argmax_temperature() -> None:
    config = _model_config()
    model = FluxConditionedTokenModel(config).eval()
    history, flux, geometry = _random_inputs(config, batch=1)

    with pytest.raises(ValueError, match="decode samples"):
        model.sample_next_frame(
            history,
            flux,
            geometry,
            guidance_weight=1.5,
            temperature=0.0,
        )


def _configured_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FluxConditionedDecoder, Path]:
    model = FluxConditionedTokenModel(_model_config()).eval()
    checkpoint = tmp_path / "flux-decoder.pt"
    torch.save(
        checkpoint_payload(model, corpus_digest="real-session-smoke"), checkpoint
    )
    vq_checkpoint = OPEN_MAGVIT_CHECKPOINT
    if not vq_checkpoint.is_file():
        vq_checkpoint = tmp_path / "imagenet_256_L.ckpt"
        vq_checkpoint.write_bytes(b"unit-test path placeholder")
    runtime = tmp_path / "decoder.json"
    runtime.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "vq_decoder_path": str(vq_checkpoint),
                "vq_decoder_id": "imagenet_256_L",
                "vq_stage": "stub",
                "seed_shot": 21858,
                "seed_slice": 50,
                "guidance_weight": 1.0,
                "temperature": 1.0,
                "top_p": 1.0,
                "sample_chunk": 32,
                "device": "cpu",
                "sample_seed": 7,
                "corpus_digest": "real-session-smoke",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("IMAS_AMBIX_FLUX_DECODER", str(runtime))
    return FluxConditionedDecoder(), checkpoint


def test_decoder_result_type_falls_back_only_when_nova_camera_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "apps.playable.camera", None)
    assert _decoded_frame_class() is DecodedFrame

    incompatible_camera = ModuleType("apps.playable.camera")
    monkeypatch.setitem(sys.modules, "apps.playable.camera", incompatible_camera)
    with pytest.raises(AttributeError):
        _decoded_frame_class()

    def missing_camera_dependency(_name: str) -> ModuleType:
        raise ModuleNotFoundError(name="apps.playable.camera_dependency")

    monkeypatch.setattr(
        "imas_ambix.worldmodel.flux_conditioned_decoder.importlib.import_module",
        missing_camera_dependency,
    )
    with pytest.raises(ModuleNotFoundError):
        _decoded_frame_class()


def test_protocol_decode_on_real_steering_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the live record with the explicitly configured unit-test VQ.

    Loading and running the 921 MB frozen VQ checkpoint is a compute-node
    integration gate.  This CPU unit test still pins that path and identity,
    while selecting the deterministic stub stage in its runtime JSON.
    """
    if not SESSION_PATH.exists():
        pytest.skip("nova labeller session is unavailable on this host")
    xarray = pytest.importorskip("xarray")

    decoder, checkpoint = _configured_decoder(tmp_path, monkeypatch)
    seed = np.stack(decoder._history)
    with xarray.open_dataset(SESSION_PATH, group="steering") as session:
        frame = session.isel(time=50).load()

    decoded = decoder.decode(frame)

    assert isinstance(decoded, NovaDecodedFrame)
    assert decoded._fields == ("image", "decode_wall", "decoder_identity")
    assert decoded.image.shape == (256, 256, 3)
    assert decoded.image.dtype == np.uint8
    assert decoded.decode_wall > 0.0
    assert np.array_equal(decoded.image[..., 0], decoded.image[..., 1])
    assert np.array_equal(decoded.image[..., 1], decoded.image[..., 2])
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert decoder.decoder_identity == (
        f"{checkpoint_digest}:imagenet_256_L:real-session-smoke"
    )
    assert callable(decoder.decode)
    assert callable(decoder.reset)

    decoder.reset(np.empty((0, 16, 16), dtype=np.int64))
    with pytest.raises(RuntimeError, match="history is empty"):
        decoder.decode(frame)
    decoder.reset(seed)
    assert len(decoder._history) == 4


def test_zero_argument_reset_restores_history_and_sampling_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not SESSION_PATH.exists():
        pytest.skip("nova labeller session is unavailable on this host")
    xarray = pytest.importorskip("xarray")
    decoder, _checkpoint = _configured_decoder(tmp_path, monkeypatch)
    with xarray.open_dataset(SESSION_PATH, group="steering") as session:
        frame = session.isel(time=50).load()

    first = decoder.decode(frame)
    decoder.decode(frame)
    decoder.reset()
    repeated = decoder.decode(frame)

    assert isinstance(repeated, NovaDecodedFrame)
    assert np.array_equal(repeated.image, first.image)
