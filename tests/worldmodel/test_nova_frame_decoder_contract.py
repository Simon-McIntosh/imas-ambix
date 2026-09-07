from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from apps.playable.camera import (
    FLUX_CONDITIONED_DECODER_PATH,
    DecodedFrame,
    FrameDecoder,
    load_decoder,
)
from nova.equilibrium.steering_frames import SteeringFrame, frames_from_session

from imas_ambix.worldmodel.flux_conditioned_decoder import (
    FluxConditionedTokenModel,
    FluxDecoderModelConfig,
    checkpoint_payload,
)

SESSION_PATH = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/21858.nc"
)
OPEN_MAGVIT_CHECKPOINT = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/weights/imagenet_256_L.ckpt"
)
DECODER_DOTTED_PATH = (
    "imas_ambix.worldmodel.flux_conditioned_decoder:FluxConditionedDecoder"
)
FRAME_INDICES = (50, 51, 52)
SAMPLE_SEED = 7


def _runtime_file(tmp_path: Path) -> Path:
    torch.manual_seed(13)
    model = FluxConditionedTokenModel(
        FluxDecoderModelConfig(
            d_model=4,
            n_layers=1,
            n_heads=1,
            d_ff=8,
            dropout=0.0,
        )
    ).eval()
    checkpoint = tmp_path / "flux-decoder.pt"
    torch.save(
        checkpoint_payload(model, corpus_digest="nova-protocol-conformance"),
        checkpoint,
    )
    runtime = tmp_path / "decoder.json"
    runtime.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "vq_decoder_path": str(OPEN_MAGVIT_CHECKPOINT),
                "vq_decoder_id": "imagenet_256_L",
                "vq_stage": "stub",
                "seed_shot": 21858,
                "seed_slice": FRAME_INDICES[0],
                "guidance_weight": 1.0,
                "temperature": 1.0,
                "top_p": 1.0,
                "sample_chunk": 32,
                "device": "cpu",
                "sample_seed": SAMPLE_SEED,
                "corpus_digest": "nova-protocol-conformance",
            }
        ),
        encoding="utf-8",
    )
    return runtime


def _session_frames() -> list[SteeringFrame]:
    with xr.open_dataset(
        SESSION_PATH,
        group="steering",
        engine="h5netcdf",
    ) as session:
        selected = session.isel(time=list(FRAME_INDICES)).load()
    return frames_from_session(selected)


def test_flux_decoder_conforms_to_nova_runtime_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not SESSION_PATH.is_file():
        pytest.skip("nova labeller session is unavailable on this host")

    assert OPEN_MAGVIT_CHECKPOINT.is_file()
    assert FLUX_CONDITIONED_DECODER_PATH == DECODER_DOTTED_PATH
    monkeypatch.setenv("IMAS_AMBIX_FLUX_DECODER", str(_runtime_file(tmp_path)))

    frames = _session_frames()
    assert len(frames) == len(FRAME_INDICES)
    assert all(isinstance(frame, SteeringFrame) for frame in frames)

    decoder = load_decoder(FLUX_CONDITIONED_DECODER_PATH)
    protocol_holds = isinstance(decoder, FrameDecoder)
    identity = decoder.decoder_identity
    decoded = [decoder.decode(frame) for frame in frames]
    walls = np.asarray([result.decode_wall for result in decoded], dtype=np.float64)

    reset_error: Exception | None = None
    repeated_first: object | None = None
    try:
        decoder.reset()
        repeated_first = decoder.decode(frames[0])
    except Exception as error:  # the assertion below reports the protocol defect
        reset_error = error

    observed_image = np.asarray(decoded[0].image)
    repeat_matches = repeated_first is not None and np.array_equal(
        observed_image, np.asarray(repeated_first.image)
    )
    print(
        json.dumps(
            {
                "decode_wall_max_s": float(np.max(walls)),
                "decode_wall_median_s": float(np.median(walls)),
                "decoder_identity": identity,
                "image_dtype": str(observed_image.dtype),
                "image_shape": list(observed_image.shape),
                "protocol_isinstance": protocol_holds,
                "repeat_after_reset_matches": repeat_matches,
                "reset_error": None
                if reset_error is None
                else f"{type(reset_error).__name__}: {reset_error}",
            },
            sort_keys=True,
        )
    )

    failures: list[str] = []
    if not protocol_holds:
        failures.append("decoder is not an instance of nova FrameDecoder")
    if not identity:
        failures.append("decoder_identity is empty")
    for index, result in enumerate(decoded):
        image = np.asarray(result.image)
        if not isinstance(result, DecodedFrame):
            failures.append(f"decode {index} did not return nova DecodedFrame")
        if result.decoder_identity != identity:
            failures.append(f"decode {index} changed decoder_identity")
        if image.shape != (256, 256, 3):
            failures.append(f"decode {index} image shape is {image.shape}")
        if image.dtype != np.uint8:
            failures.append(f"decode {index} image dtype is {image.dtype}")
        if image.ndim == 3 and not (
            np.array_equal(image[..., 0], image[..., 1])
            and np.array_equal(image[..., 1], image[..., 2])
        ):
            failures.append(f"decode {index} image channels differ")
        if not np.isfinite(result.decode_wall) or result.decode_wall <= 0.0:
            failures.append(f"decode {index} wall is not positive and finite")
    if reset_error is not None:
        failures.append(f"zero-argument reset left decoder unusable: {reset_error}")
    elif not isinstance(repeated_first, DecodedFrame):
        failures.append("decode after reset did not return nova DecodedFrame")
    if not repeat_matches:
        failures.append("fixed-seed first decode changed after reset")

    assert not failures, "; ".join(failures)
