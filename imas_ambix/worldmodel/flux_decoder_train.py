"""Train the flux-conditioned camera-token decoder on growing sessions.

The corpus producer writes one atomic shot manifest at a time.  This trainer
therefore rebuilds both dataset splits at every epoch boundary, allowing newly
completed shots to join without restarting the allocation.  Checkpoints and the
receipt are atomically replaced so a reader never observes a partial artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import signal
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

from imas_ambix.data.paths import LEVEL1_DIR, TOKEN_ROOT
from imas_ambix.worldmodel.flux_conditioned_decoder import (
    FluxConditionedTokenModel,
    FluxDecoderModelConfig,
    checkpoint_payload,
)
from imas_ambix.worldmodel.flux_label_dataset import (
    DEFAULT_COHORT_REPORT,
    DEFAULT_HISTORY_SPACING_SECONDS,
    DEFAULT_SESSION_ROOT,
    FluxLabelDataset,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 30 * 60
DEFAULT_RUN_DIR = Path("/work/projects/imas_gpu/ambix/flux_decoder/overnight-20260906")
DEFAULT_VQ_CHECKPOINT = Path(
    "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/weights/imagenet_256_L.ckpt"
)
PERSISTENT_VQ_ROUTE = "persistent-subprocess"


class DatasetLike(Protocol):
    """Minimal dataset surface consumed by the training loop."""

    receipt: Mapping[str, object]
    references: Sequence[object]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, object]: ...


class PixelDecoder(Protocol):
    """Decode a batch of local token grids to image arrays."""

    def __call__(self, tokens: np.ndarray) -> np.ndarray: ...


DatasetFactory = Callable[[str], DatasetLike]


@dataclass(frozen=True, slots=True)
class FluxTrainingConfig:
    """Runtime settings for one decoder-training allocation."""

    session_root: Path = DEFAULT_SESSION_ROOT
    run_dir: Path = DEFAULT_RUN_DIR
    cohort_report: Path = DEFAULT_COHORT_REPORT
    token_root: Path = TOKEN_ROOT
    level1_root: Path = LEVEL1_DIR
    epochs: int = 1_000_000
    max_steps: int | None = None
    batch_size: int = 1
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    checkpoint_interval_s: float = DEFAULT_CHECKPOINT_INTERVAL_SECONDS
    validation_batches: int = 4
    pixel_validation_samples: int = 4
    device: str = "auto"
    seed: int = 42
    num_workers: int = 0
    loss_chunk: int = 32
    guidance_weight: float = 1.0
    vq_checkpoint: Path | None = DEFAULT_VQ_CHECKPOINT
    vq_decoder_id: str = "imagenet_256_L"
    model_config: FluxDecoderModelConfig = FluxDecoderModelConfig()

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when supplied")
        if self.batch_size < 1 or self.validation_batches < 1:
            raise ValueError("batch_size and validation_batches must be positive")
        if self.pixel_validation_samples < 0:
            raise ValueError("pixel_validation_samples must be non-negative")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer settings must be non-negative")
        if self.checkpoint_interval_s <= 0.0:
            raise ValueError("checkpoint_interval_s must be positive")
        if self.num_workers < 0 or self.loss_chunk < 1:
            raise ValueError("num_workers must be non-negative and loss_chunk positive")
        if DEFAULT_HISTORY_SPACING_SECONDS != 0.005:
            raise RuntimeError("the training contract requires 0.005 s history spacing")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and counters produced by one completed training invocation."""

    checkpoint: Path
    receipt: Path
    steps: int
    epochs_completed: int
    admitted_shots: int
    admitted_slices: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _dataset_factory(config: FluxTrainingConfig) -> DatasetFactory:
    def make(split: str) -> FluxLabelDataset:
        return FluxLabelDataset(
            config.session_root,
            split=split,
            history_spacing_s=DEFAULT_HISTORY_SPACING_SECONDS,
            token_root=config.token_root,
            level1_root=config.level1_root,
            cohort_report=config.cohort_report,
        )

    return make


def corpus_identity(*datasets: DatasetLike) -> dict[str, object]:
    """Return a stable digest and sorted admitted-shot slice census."""
    slices: Counter[int] = Counter()
    conditioned: Counter[int] = Counter()
    split_by_shot: dict[int, str] = {}
    for dataset in datasets:
        for reference in dataset.references:
            shot = int(reference.shot_id)
            split = str(reference.split)
            slices[shot] += 1
            conditioned[shot] += int(bool(reference.conditioned))
            previous = split_by_shot.setdefault(shot, split)
            if previous != split:
                raise ValueError(f"shot {shot} appears in both dataset splits")
    shots = [
        {
            "shot_id": shot,
            "split": split_by_shot[shot],
            "slice_count": slices[shot],
            "conditioned_slice_count": conditioned[shot],
        }
        for shot in sorted(slices)
    ]
    encoded = json.dumps(shots, separators=(",", ":"), sort_keys=True).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "shots": shots}


def _pins(dataset: DatasetLike) -> tuple[str, str]:
    pins = dataset.receipt.get("pins")
    if not isinstance(pins, Mapping):
        raise ValueError("dataset receipt has no corpus pins")
    policy_digest = str(pins.get("policy_digest", ""))
    carrier_identity = str(pins.get("carrier_identity", ""))
    if not policy_digest or not carrier_identity:
        raise ValueError("dataset receipt has incomplete corpus pins")
    return policy_digest, carrier_identity


def _device(requested: str) -> torch.device:
    selected = (
        "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    )
    if selected == "auto":
        selected = "cpu"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def _loader(
    dataset: DatasetLike,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[Mapping[str, object]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _model_inputs(
    batch: Mapping[str, object], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = torch.as_tensor(batch["history_tokens"], dtype=torch.long, device=device)
    flux = torch.as_tensor(batch["conditioning"], dtype=torch.float32, device=device)
    geometry = torch.as_tensor(batch["geometry"], dtype=torch.float32, device=device)
    target = torch.as_tensor(batch["target_tokens"], dtype=torch.long, device=device)
    return history, flux, geometry, target


@torch.no_grad()
def evaluate_model(
    model: FluxConditionedTokenModel,
    dataset: DatasetLike,
    *,
    config: FluxTrainingConfig,
    device: torch.device,
    pixel_decoder: PixelDecoder | None,
) -> dict[str, object]:
    """Measure held-out likelihood and persistence-relative decoded error."""
    if not len(dataset):
        raise ValueError("validation split contains no admitted slices")
    model.eval()
    total_nll = 0.0
    total_samples = 0
    token_mismatches = 0
    token_count = 0
    decoded_samples = 0
    pixel_token_groups: list[np.ndarray] = []
    loader = _loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    sample_generator = torch.Generator(device=device.type).manual_seed(config.seed)
    for batch_index, batch in enumerate(loader):
        if batch_index >= config.validation_batches:
            break
        history, flux, geometry, target = _model_inputs(batch, device)
        batch_size = int(history.shape[0])
        nll = model.chunked_nll(
            history,
            flux,
            geometry,
            target,
            chunk=config.loss_chunk,
        )
        total_nll += float(nll) * batch_size
        total_samples += batch_size
        persistence = history[:, -1]
        token_mismatches += int((persistence != target).sum().item())
        token_count += int(target.numel())
        if pixel_decoder is None or decoded_samples >= config.pixel_validation_samples:
            continue
        predicted = model.sample_next_frame(
            history,
            flux,
            geometry,
            guidance_weight=config.guidance_weight,
            top_p=1.0,
            chunk=config.loss_chunk,
            generator=sample_generator,
        )
        remaining = config.pixel_validation_samples - decoded_samples
        for sample in range(min(batch_size, remaining)):
            pixel_token_groups.append(
                np.stack(
                    (
                        target[sample].cpu().numpy(),
                        predicted[sample].cpu().numpy(),
                        persistence[sample].cpu().numpy(),
                    )
                )
            )
            decoded_samples += 1
    if not total_samples:
        raise RuntimeError("validation loader yielded no batches")
    pixel_metrics: dict[str, object] = {
        "available": decoded_samples > 0,
        "sample_count": decoded_samples,
        "model_mae_u8": None,
        "persistence_mae_u8": None,
        "model_minus_persistence_mae_u8": None,
    }
    if decoded_samples:
        token_batch = np.concatenate(pixel_token_groups, axis=0)
        decoded_batch = np.asarray(pixel_decoder(token_batch))
        expected_images = decoded_samples * 3
        if decoded_batch.ndim < 2 or decoded_batch.shape[0] != expected_images:
            raise ValueError(
                "pixel decoder returned "
                f"{decoded_batch.shape}; expected {expected_images} image arrays"
            )
        decoded_groups = decoded_batch.reshape(
            (decoded_samples, 3, *decoded_batch.shape[1:])
        ).astype(np.float32)
        target_images = decoded_groups[:, 0]
        model_images = decoded_groups[:, 1]
        persistence_images = decoded_groups[:, 2]
        model_mean = float(np.abs(model_images - target_images).mean(axis=None))
        persistence_mean = float(
            np.abs(persistence_images - target_images).mean(axis=None)
        )
        pixel_metrics.update(
            {
                "model_mae_u8": model_mean,
                "persistence_mae_u8": persistence_mean,
                "model_minus_persistence_mae_u8": model_mean - persistence_mean,
            }
        )
    return {
        "sample_count": total_samples,
        "token_count": token_count,
        "token_nll": total_nll / total_samples,
        "persistence_token_error_rate": token_mismatches / token_count,
        "persistence_definition": "repeat the last history token frame",
        "decoded_pixel_error": pixel_metrics,
    }


def _checkpoint_and_receipt(
    model: FluxConditionedTokenModel,
    optimizer: torch.optim.Optimizer,
    *,
    config: FluxTrainingConfig,
    corpus: Mapping[str, object],
    policy_digest: str,
    carrier_identity: str,
    validation: Mapping[str, object],
    git_revision: str,
    device: torch.device,
    step: int,
    epoch: int,
    started_at: str,
    status: str,
) -> tuple[Path, Path, str]:
    corpus_digest = str(corpus["sha256"])
    checkpoint = config.run_dir / f"checkpoint-{step:09d}.pt"
    payload = checkpoint_payload(model, corpus_digest=corpus_digest)
    payload.update(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "git_revision": git_revision,
            "policy_digest": policy_digest,
            "carrier_identity": carrier_identity,
            "admitted_shots": corpus["shots"],
            "validation": dict(validation),
        }
    )
    _atomic_torch_save(checkpoint, payload)
    checkpoint_digest = _file_sha256(checkpoint)
    decoder_identity = f"{checkpoint_digest}:{config.vq_decoder_id}:{corpus_digest}"
    receipt = config.run_dir / "receipt.json"
    _atomic_json(
        receipt,
        {
            "schema": "flux-conditioned-decoder-training",
            "status": status,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "run_directory": str(config.run_dir),
            "step": step,
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_interval_seconds": config.checkpoint_interval_s,
            "policy_digest": policy_digest,
            "carrier_identity": carrier_identity,
            "corpus_digest": dict(corpus),
            "git_revision": git_revision,
            "decoder_identity": decoder_identity,
            "vq_decoder_id": config.vq_decoder_id,
            "vq_route": (
                PERSISTENT_VQ_ROUTE if config.vq_checkpoint is not None else "disabled"
            ),
            "device": str(device),
            "history_spacing_s": DEFAULT_HISTORY_SPACING_SECONDS,
            "model_config": asdict(config.model_config),
            "validation": dict(validation),
        },
    )
    return checkpoint, receipt, decoder_identity


def train_flux_decoder(
    config: FluxTrainingConfig,
    *,
    dataset_factory: DatasetFactory | None = None,
    pixel_decoder: PixelDecoder | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TrainingResult:
    """Train with an epoch-boundary corpus rescan and periodic checkpoints."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    model = FluxConditionedTokenModel(config.model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    make_dataset = dataset_factory or _dataset_factory(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    git_revision = _git_revision()
    started_at = _utc_now()
    next_checkpoint = clock() + config.checkpoint_interval_s
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    step = 0
    completed_epochs = 0
    latest_checkpoint: Path | None = None
    latest_receipt: Path | None = None
    latest_corpus: Mapping[str, object] | None = None
    latest_validation: Mapping[str, object] | None = None
    policy_digest = ""
    carrier_identity = ""
    try:
        for epoch in range(1, config.epochs + 1):
            train_dataset = make_dataset("train")
            validation_dataset = make_dataset("validation")
            if not len(train_dataset):
                raise ValueError("training split contains no admitted slices")
            train_pins = _pins(train_dataset)
            validation_pins = _pins(validation_dataset)
            if train_pins != validation_pins:
                raise ValueError("training and validation corpus pins disagree")
            policy_digest, carrier_identity = train_pins
            latest_corpus = corpus_identity(train_dataset, validation_dataset)
            LOGGER.info(
                "epoch %d rescan: %d train slices, %d validation slices, %d shots",
                epoch,
                len(train_dataset),
                len(validation_dataset),
                len(latest_corpus["shots"]),
            )
            train_loader = _loader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                seed=config.seed + epoch,
            )
            for batch in train_loader:
                model.train()
                history, flux, geometry, target = _model_inputs(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.chunked_nll(
                    history,
                    flux,
                    geometry,
                    target,
                    chunk=config.loss_chunk,
                )
                loss.backward()
                optimizer.step()
                step += 1
                if step == 1 or step % 10 == 0:
                    LOGGER.info("step %d token_nll %.6f", step, float(loss.detach()))
                interval_reached = clock() >= next_checkpoint
                max_steps_reached = (
                    config.max_steps is not None and step >= config.max_steps
                )
                if interval_reached:
                    latest_validation = evaluate_model(
                        model,
                        validation_dataset,
                        config=config,
                        device=device,
                        pixel_decoder=pixel_decoder,
                    )
                    latest_checkpoint, latest_receipt, decoder_identity = (
                        _checkpoint_and_receipt(
                            model,
                            optimizer,
                            config=config,
                            corpus=latest_corpus,
                            policy_digest=policy_digest,
                            carrier_identity=carrier_identity,
                            validation=latest_validation,
                            git_revision=git_revision,
                            device=device,
                            step=step,
                            epoch=epoch,
                            started_at=started_at,
                            status="training",
                        )
                    )
                    LOGGER.info(
                        "checkpoint %s decoder_identity=%s",
                        latest_checkpoint,
                        decoder_identity,
                    )
                    next_checkpoint = clock() + config.checkpoint_interval_s
                if max_steps_reached or stop_requested:
                    break
            completed_epochs = epoch
            if (
                config.max_steps is not None and step >= config.max_steps
            ) or stop_requested:
                break
        if latest_corpus is None:
            raise RuntimeError("training ended before the first corpus rescan")
        validation_dataset = make_dataset("validation")
        latest_validation = evaluate_model(
            model,
            validation_dataset,
            config=config,
            device=device,
            pixel_decoder=pixel_decoder,
        )
        latest_checkpoint, latest_receipt, decoder_identity = _checkpoint_and_receipt(
            model,
            optimizer,
            config=config,
            corpus=latest_corpus,
            policy_digest=policy_digest,
            carrier_identity=carrier_identity,
            validation=latest_validation,
            git_revision=git_revision,
            device=device,
            step=step,
            epoch=completed_epochs,
            started_at=started_at,
            status="stopped" if stop_requested else "complete",
        )
        LOGGER.info(
            "final checkpoint %s decoder_identity=%s",
            latest_checkpoint,
            decoder_identity,
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if latest_checkpoint is None or latest_receipt is None or latest_corpus is None:
        raise RuntimeError("training produced no checkpoint receipt")
    shots = latest_corpus["shots"]
    return TrainingResult(
        checkpoint=latest_checkpoint,
        receipt=latest_receipt,
        steps=step,
        epochs_completed=completed_epochs,
        admitted_shots=len(shots),
        admitted_slices=sum(int(row["slice_count"]) for row in shots),
    )


def _pixel_decoder(path: Path | None, device: torch.device) -> PixelDecoder | None:
    if path is None:
        return None

    def decode(tokens: np.ndarray) -> np.ndarray:
        from imas_ambix.worldmodel.flux_decoder_video import (  # noqa: PLC0415
            _decode_vq,
        )

        images, actual_route, detail = _decode_vq(
            np.asarray(tokens, dtype=np.int64),
            route=PERSISTENT_VQ_ROUTE,
            vq_checkpoint=path,
            device=str(device),
            batch_size=8,
        )
        if actual_route != PERSISTENT_VQ_ROUTE:
            raise RuntimeError(
                f"pixel validation selected unexpected VQ route {actual_route!r}"
            )
        LOGGER.info("pixel validation VQ route %s: %s", actual_route, detail)
        return images

    return decode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    parser.add_argument("--level1-root", type=Path, default=LEVEL1_DIR)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    )
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--pixel-validation-samples", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss-chunk", type=int, default=32)
    parser.add_argument("--guidance-weight", type=float, default=1.0)
    parser.add_argument("--vq-checkpoint", type=Path, default=DEFAULT_VQ_CHECKPOINT)
    parser.add_argument("--no-pixel-validation", action="store_true")
    parser.add_argument("--model-width", type=int, default=512)
    parser.add_argument("--model-layers", type=int, default=8)
    parser.add_argument("--model-heads", type=int, default=8)
    parser.add_argument("--model-feedforward", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    model_config = FluxDecoderModelConfig(
        d_model=args.model_width,
        n_layers=args.model_layers,
        n_heads=args.model_heads,
        d_ff=args.model_feedforward,
    )
    config = FluxTrainingConfig(
        session_root=args.session_root,
        run_dir=args.run_dir,
        cohort_report=args.cohort_report,
        token_root=args.token_root,
        level1_root=args.level1_root,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_interval_s=args.checkpoint_interval_seconds,
        validation_batches=args.validation_batches,
        pixel_validation_samples=args.pixel_validation_samples,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        loss_chunk=args.loss_chunk,
        guidance_weight=args.guidance_weight,
        vq_checkpoint=None if args.no_pixel_validation else args.vq_checkpoint,
        model_config=model_config,
    )
    device = _device(config.device)
    decoder = _pixel_decoder(config.vq_checkpoint, device)
    result = train_flux_decoder(config, pixel_decoder=decoder)
    print(json.dumps(asdict(result), default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FluxTrainingConfig",
    "TrainingResult",
    "corpus_identity",
    "evaluate_model",
    "main",
    "train_flux_decoder",
]
