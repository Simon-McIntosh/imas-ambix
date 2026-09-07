from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import imas_ambix.worldmodel.flux_decoder_train as training
from imas_ambix.worldmodel.flux_conditioned_decoder import FluxDecoderModelConfig
from imas_ambix.worldmodel.flux_decoder_train import (
    FluxTrainingConfig,
    corpus_identity,
    evaluate_model,
    train_flux_decoder,
)

POLICY_DIGEST = "policy-digest"
CARRIER_IDENTITY = "carrier-identity"


@dataclass(frozen=True)
class _Reference:
    shot_id: int
    split: str
    conditioned: bool


class _SyntheticDataset:
    def __init__(self, split: str, shots: tuple[int, ...]) -> None:
        self.references = tuple(
            _Reference(shot, split, index % 2 == 0)
            for index, shot in enumerate(shots)
            for _ in range(index + 1)
        )
        self.receipt = {
            "pins": {
                "policy_digest": POLICY_DIGEST,
                "carrier_identity": CARRIER_IDENTITY,
            }
        }

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, object]:
        reference = self.references[index]
        history = np.zeros((4, 16, 16), dtype=np.int64)
        target = np.ones((16, 16), dtype=np.int64)
        return {
            "history_tokens": history,
            "conditioning": np.full((6, 64, 64), 0.25, dtype=np.float32),
            "geometry": np.linspace(0.0, 1.0, 12, dtype=np.float32),
            "target_tokens": target,
            "conditioned": reference.conditioned,
            "shot_id": reference.shot_id,
        }


class _TinyTokenModel(nn.Module):
    def __init__(self, config: FluxDecoderModelConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.tensor(0.0))

    def chunked_nll(
        self,
        history: torch.Tensor,
        flux: torch.Tensor,
        geometry: torch.Tensor,
        target: torch.Tensor,
        *,
        chunk: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del history, flux, geometry, chunk, generator
        return (self.weight - target.float().mean()).square()

    def sample_next_frame(
        self,
        history: torch.Tensor,
        flux: torch.Tensor,
        geometry: torch.Tensor,
        *,
        guidance_weight: float,
        top_p: float,
        chunk: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del flux, geometry, guidance_weight, top_p, chunk, generator
        return torch.ones_like(history[:, -1])


def _config(tmp_path: Path, **overrides: object) -> FluxTrainingConfig:
    values: dict[str, object] = {
        "run_dir": tmp_path / "run",
        "epochs": 2,
        "max_steps": 2,
        "batch_size": 1,
        "checkpoint_interval_s": 1800.0,
        "validation_batches": 1,
        "pixel_validation_samples": 0,
        "device": "cpu",
        "vq_checkpoint": None,
        "model_config": FluxDecoderModelConfig(
            d_model=4,
            n_layers=1,
            n_heads=1,
            d_ff=8,
        ),
    }
    values.update(overrides)
    return FluxTrainingConfig(**values)


def test_corpus_identity_is_sorted_and_counts_conditioned_slices() -> None:
    train = _SyntheticDataset("train", (20002, 20001))
    validation = _SyntheticDataset("validation", (20020,))

    identity = corpus_identity(train, validation)

    assert [row["shot_id"] for row in identity["shots"]] == [20001, 20002, 20020]
    assert [row["slice_count"] for row in identity["shots"]] == [2, 1, 1]
    assert [row["conditioned_slice_count"] for row in identity["shots"]] == [0, 1, 1]
    assert len(identity["sha256"]) == 64


def test_training_rescans_each_epoch_and_writes_checkpoint_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "FluxConditionedTokenModel", _TinyTokenModel)
    calls = {"train": 0, "validation": 0}

    def factory(split: str) -> _SyntheticDataset:
        calls[split] += 1
        if split == "train":
            shots = (20001,) if calls[split] == 1 else (20001, 20002)
        else:
            shots = (20020,)
        return _SyntheticDataset(split, shots)

    ticks = iter((0.0, 1801.0, 1801.0, 3602.0, 3602.0))
    result = train_flux_decoder(
        _config(tmp_path), dataset_factory=factory, clock=lambda: next(ticks)
    )

    assert calls["train"] == 2
    assert calls["validation"] == 3
    assert result.steps == 2
    assert result.epochs_completed == 2
    assert result.admitted_shots == 3
    assert result.admitted_slices == 4
    assert result.checkpoint.is_file()
    assert result.receipt.is_file()
    assert sorted(result.checkpoint.parent.glob("checkpoint-*.pt")) == [
        result.checkpoint.parent / "checkpoint-000000001.pt",
        result.checkpoint.parent / "checkpoint-000000002.pt",
    ]

    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "complete"
    assert receipt["step"] == 2
    assert receipt["policy_digest"] == POLICY_DIGEST
    assert receipt["carrier_identity"] == CARRIER_IDENTITY
    assert receipt["history_spacing_s"] == pytest.approx(0.005)
    assert receipt["checkpoint_interval_seconds"] == pytest.approx(1800.0)
    assert [row["shot_id"] for row in receipt["corpus_digest"]["shots"]] == [
        20001,
        20002,
        20020,
    ]
    assert receipt["decoder_identity"].endswith(
        f":imagenet_256_L:{receipt['corpus_digest']['sha256']}"
    )
    assert receipt["validation"]["persistence_definition"] == (
        "repeat the last history token frame"
    )
    assert receipt["validation"]["persistence_token_error_rate"] == 1.0

    payload = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    assert payload["corpus_digest"] == receipt["corpus_digest"]["sha256"]
    assert payload["policy_digest"] == POLICY_DIGEST
    assert payload["carrier_identity"] == CARRIER_IDENTITY
    assert payload["step"] == 2


def test_validation_reports_decoded_pixel_error_against_persistence(
    tmp_path: Path,
) -> None:
    dataset = _SyntheticDataset("validation", (20020,))
    model = _TinyTokenModel(
        FluxDecoderModelConfig(d_model=4, n_layers=1, n_heads=1, d_ff=8)
    )

    def decode(tokens: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(tokens, 2, axis=1), 2, axis=2).astype(np.uint8)

    metrics = evaluate_model(
        model,
        dataset,
        config=_config(tmp_path, pixel_validation_samples=1),
        device=torch.device("cpu"),
        pixel_decoder=decode,
    )

    assert metrics["sample_count"] == 1
    assert metrics["token_nll"] == pytest.approx(1.0)
    assert metrics["persistence_token_error_rate"] == 1.0
    pixel = metrics["decoded_pixel_error"]
    assert pixel == {
        "available": True,
        "sample_count": 1,
        "model_mae_u8": 0.0,
        "persistence_mae_u8": 1.0,
        "model_minus_persistence_mae_u8": -1.0,
    }


def test_pixel_validation_uses_persistent_subprocess_and_records_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_ambix.worldmodel import flux_conditioned_decoder, flux_decoder_video

    vq_checkpoint = tmp_path / "frozen-vq.ckpt"
    vq_checkpoint.write_bytes(b"frozen decoder")

    def reject_in_process_decoder(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the in-process VQ decoder must not be constructed")

    monkeypatch.setattr(
        flux_conditioned_decoder,
        "_OpenMagvitVQDecoder",
        reject_in_process_decoder,
    )
    calls: list[dict[str, object]] = []

    def decode_batch(
        tokens: np.ndarray,
        *,
        route: str,
        vq_checkpoint: Path,
        device: str,
        batch_size: int,
    ) -> tuple[np.ndarray, str, str]:
        calls.append(
            {
                "token_shape": tokens.shape,
                "route": route,
                "vq_checkpoint": vq_checkpoint,
                "device": device,
                "batch_size": batch_size,
            }
        )
        values = np.bitwise_and(tokens, 255).astype(np.uint8)
        images = np.repeat(np.repeat(values, 2, axis=1), 2, axis=2)
        return images[..., None], route, "dedicated decoder interpreter"

    monkeypatch.setattr(flux_decoder_video, "_decode_vq", decode_batch)
    config = _config(
        tmp_path,
        max_steps=1,
        pixel_validation_samples=1,
        vq_checkpoint=vq_checkpoint,
    )
    pixel_decoder = training._pixel_decoder(vq_checkpoint, torch.device("cpu"))

    result = train_flux_decoder(
        config,
        dataset_factory=lambda split: _SyntheticDataset(
            split, (20001,) if split == "train" else (20020,)
        ),
        pixel_decoder=pixel_decoder,
    )

    assert calls == [
        {
            "token_shape": (3, 16, 16),
            "route": "persistent-subprocess",
            "vq_checkpoint": vq_checkpoint,
            "device": "cpu",
            "batch_size": 8,
        }
    ]
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["vq_route"] == "persistent-subprocess"
    assert receipt["validation"]["decoded_pixel_error"]["available"] is True


def test_training_config_rejects_invalid_checkpoint_cadence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_s"):
        _config(tmp_path, checkpoint_interval_s=0.0)
