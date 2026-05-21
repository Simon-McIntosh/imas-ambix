"""Demo orchestration for the WHAM Fusion World Model.

Implements :func:`run_demo` which takes a held-out shot id, loads persisted
tokens, runs a forward rollout via :func:`~imas_ambix.eval.rollout.rollout`,
decodes predicted frames, computes evaluation metrics, and writes all output
artefacts described in ``plans/demo.md`` §5.

Mock mode
---------
Pass ``checkpoint_path="mock"`` to run without a real trained checkpoint.
The :class:`_MockWhamModel` returns the model-input ids repeated as logits,
producing a token distribution that is trivially non-uniform but still
exercises every code path in the pipeline.

Block-size assumptions (v0)
---------------------------
``K_FRAME = 256`` tokens per frame step (MAGVIT2 16× spatial at 256² input).
``K_CTRL  = 50``  signal + action tokens per step.
These match the constants in :mod:`imas_ambix.eval.rollout`.

Related plans:
- ``plans/demo.md`` §5 (CLI surface and output layout)
- ``plans/world-model-v0.md`` §7 (rollout algorithm)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imas_ambix.eval.metrics import compute_all_metrics
from imas_ambix.eval.rollout import (
    K_CTRL_DEFAULT,
    K_FRAME_DEFAULT,
    RolloutConfig,
    rollout,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model time-grid constants
# ---------------------------------------------------------------------------

#: Model sampling frequency in Hz (100 Hz = every 10 ms of physical time).
MODEL_HZ: int = 100


# ---------------------------------------------------------------------------
# Mock model for pipeline exercise without a real checkpoint
# ---------------------------------------------------------------------------


class _MockWhamModel:
    """Minimal stand-in for WhamModel when checkpoint_path='mock'.

    ``forward`` returns a dict with ``logits`` of shape ``(B, L, V)`` where
    each position emits a fixed uniform distribution across a small vocab.
    The model does not learn anything; it exists to exercise the rollout
    code path end-to-end.
    """

    MOCK_VOCAB: int = 512

    def forward(self, input_ids: object, **_kwargs: object) -> dict:
        import torch

        t = input_ids  # type: ignore[assignment]
        b, seq_len = t.shape
        # Uniform logits over mock vocab — deterministic under torch seed
        logits = torch.zeros(b, seq_len, self.MOCK_VOCAB, device=t.device)
        return {"logits": logits, "loss": None}


# ---------------------------------------------------------------------------
# Placeholder frame decoder
# ---------------------------------------------------------------------------


def _decode_tokens_to_frames(
    token_ids: np.ndarray,
    tokenizer_version: str,
    h: int = 32,
    w: int = 32,
    k_frame: int = K_FRAME_DEFAULT,
) -> np.ndarray:
    """Decode a 1-D token array to ``(T, H, W, 3)`` uint8 frames.

    For v0 the ``PlaceholderFrameTokenizer`` is used regardless of
    ``tokenizer_version`` because the real Open-MAGVIT2 venv may be absent.
    The decoder produces low-resolution colour frames that preserve relative
    token-id brightness.

    Parameters
    ----------
    token_ids:
        1-D int32 token array of length ``T * k_frame``.
    tokenizer_version:
        Ignored in v0 (always uses PlaceholderFrameTokenizer); threaded
        through for future tokenizer selection.
    h, w:
        Output frame dimensions (spatial, pre-upsampling).
    k_frame:
        Frame tokens per timestep; determines how many tokens map to one frame.

    Returns
    -------
    np.ndarray
        ``(T, h*sc, w*sc, 3)`` uint8 where ``sc`` is the spatial upscale
        factor from :class:`~imas_ambix.tokenizer.frames.PlaceholderFrameTokenizer`.
    """
    from imas_ambix.tokenizer.base import EncodedFrames
    from imas_ambix.tokenizer.frames import PlaceholderFrameTokenizer

    tok = PlaceholderFrameTokenizer()
    # tokens per frame for placeholder: (H//sc) × (W//sc)
    sc = tok.spatial_compression
    tc = tok.temporal_compression
    h_tokens = h // sc
    w_tokens = w // sc
    tokens_per_frame = h_tokens * w_tokens  # tokens per compressed frame

    # Flatten and truncate to a multiple of tokens_per_frame
    ids = np.asarray(token_ids, dtype=np.int32).ravel()
    n_frames = len(ids) // tokens_per_frame
    if n_frames == 0:
        # Return a single blank frame if too few tokens
        return np.zeros((1, h, w, 3), dtype=np.uint8)

    ids = ids[: n_frames * tokens_per_frame]
    ids_3d = ids.reshape(n_frames, h_tokens, w_tokens)

    encoded = EncodedFrames(
        token_ids=ids_3d,
        shape=ids_3d.shape,
        tokenizer_name=tok.name,
        metadata={"input_shape": [n_frames * tc, h, w]},
    )
    gray = tok.decode(encoded)  # (T*tc, h, w) uint8

    # Convert grayscale to RGB by stacking
    rgb = np.stack([gray, gray, gray], axis=-1)  # (T*tc, h, w, 3)
    return rgb


# ---------------------------------------------------------------------------
# DemoArtifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoArtifacts:
    """Paths to all artefacts written by :func:`run_demo`.

    Attributes
    ----------
    ground_truth_zarr:
        Zarr archive of decoded ground-truth frames (float32, T×H×W×3).
    prediction_zarr:
        Zarr archive of decoded predicted frames (float32, T×H×W×3).
    tokens_ground_truth_zarr:
        Zarr archive of the raw ground-truth token sequence used for rollout.
    tokens_prediction_zarr:
        Zarr archive of the raw predicted token sequence from rollout.
    metrics_json:
        JSON file with keys: ``psnr``, ``lpips``, ``rfid``, ``centroid_mse``,
        ``chord_nrmse``, ``edge_displacement_mad``.
    figure_png:
        4-panel matplotlib comparison figure (top: GT, bottom: prediction
        at t=0 / 0.2 / 0.5 / 1.0 s).
    video_mp4:
        Side-by-side MP4 (GT left, prediction right), or ``None`` if imageio
        / ffmpeg was unavailable.
    """

    ground_truth_zarr: Path
    prediction_zarr: Path
    tokens_ground_truth_zarr: Path
    tokens_prediction_zarr: Path
    metrics_json: Path
    figure_png: Path
    video_mp4: Path | None


# ---------------------------------------------------------------------------
# run_demo
# ---------------------------------------------------------------------------


def run_demo(
    shot_id: int,
    checkpoint_path: str | Path,
    *,
    prefix_ms: int = 150,
    rollout_ms: int = 1000,
    output_dir: Path,
    tokenizer_version: str = "v1",
    rollout_config: RolloutConfig | None = None,
    decode_frames: bool = True,
    no_video: bool = False,
    _k_frame: int = K_FRAME_DEFAULT,
    _k_ctrl: int = K_CTRL_DEFAULT,
) -> DemoArtifacts:
    """Run the full demo pipeline for one held-out shot.

    Steps
    -----
    1. Load the shot's persisted token stream via :func:`load_shot_stream`
       (or create a synthetic stream if the real data is absent — useful for
       CI).
    2. Compute prefix and rollout token counts from ``prefix_ms``/
       ``rollout_ms`` at :data:`MODEL_HZ` cadence.
    3. Load the model from ``checkpoint_path`` (real checkpoint via
       :meth:`~imas_ambix.model.WhamModel.from_pretrained`), or create a
       :class:`_MockWhamModel` when ``checkpoint_path == "mock"``.
    4. Call :func:`~imas_ambix.eval.rollout.rollout` with the prefix and the
       ground-truth control-token slice as the forcing schedule.
    5. Decode both ground-truth and predicted frame tokens to ``uint8`` RGB
       via :func:`_decode_tokens_to_frames`.
    6. Save Zarr artefacts (frames + tokens).
    7. Compute metrics via :func:`~imas_ambix.eval.metrics.compute_all_metrics`.
    8. Render the 4-panel comparison figure.
    9. Optionally render an MP4 via ``imageio``/``imageio-ffmpeg``.

    Parameters
    ----------
    shot_id:
        Numeric MAST shot identifier.
    checkpoint_path:
        Path to a saved WhamModel checkpoint (passed to
        :meth:`~imas_ambix.model.WhamModel.from_pretrained`), or the
        literal string ``"mock"`` to use :class:`_MockWhamModel` instead.
    prefix_ms:
        Duration of the initial context window in milliseconds.
        Default 150 ms.
    rollout_ms:
        Duration of the rollout window in milliseconds.  Default 1000 ms.
    output_dir:
        Directory to write all artefacts.  Created if absent.
    tokenizer_version:
        Token vocabulary version label (default ``"v1"``).
    rollout_config:
        Override the default :class:`~imas_ambix.eval.rollout.RolloutConfig`.
        When ``None`` the defaults are used (prefix_tokens and rollout_steps
        are overridden by ``prefix_ms``/``rollout_ms`` anyway).
    decode_frames:
        If False, skip decoding tokens to RGB frames (useful for token-only
        analysis or very fast smoke tests).
    no_video:
        If True, skip MP4 rendering even if imageio-ffmpeg is available.

    Returns
    -------
    DemoArtifacts
        Paths to all generated artefacts.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "run_demo: shot=%d prefix_ms=%d rollout_ms=%d",
        shot_id,
        prefix_ms,
        rollout_ms,
    )

    # ------------------------------------------------------------------
    # 1. Load or synthesise the shot's token stream
    # ------------------------------------------------------------------
    tokens_np, block_kind_np = _load_or_synthesise_stream(
        shot_id=shot_id,
        tokenizer_version=tokenizer_version,
        prefix_ms=prefix_ms,
        rollout_ms=rollout_ms,
    )

    tokens_per_step = _k_frame + _k_ctrl
    total_steps_needed = (prefix_ms + rollout_ms) * MODEL_HZ // 1000
    min_tokens = total_steps_needed * tokens_per_step + _k_ctrl

    if len(tokens_np) < min_tokens:
        # Tile the stream to be long enough
        repeats = (min_tokens // len(tokens_np)) + 2
        tokens_np = np.tile(tokens_np, repeats)[:min_tokens]
        block_kind_np = np.tile(block_kind_np, repeats)[:min_tokens]

    # ------------------------------------------------------------------
    # 2. Compute prefix / rollout token counts
    # ------------------------------------------------------------------
    prefix_steps = max(1, prefix_ms * MODEL_HZ // 1000)
    rollout_steps = max(1, rollout_ms * MODEL_HZ // 1000)

    prefix_token_count = prefix_steps * tokens_per_step
    rollout_ctrl_count = rollout_steps * _k_ctrl

    # Slice prefix (initial context)
    prefix_len = min(
        prefix_token_count,
        len(tokens_np) - rollout_ctrl_count - _k_ctrl,
    )
    prefix_len = max(prefix_len, _k_frame + _k_ctrl)

    prefix_tokens = torch.tensor(tokens_np[:prefix_len], dtype=torch.long).unsqueeze(
        0
    )  # (1, L)

    # Control tokens: the ground-truth signal+action tokens for the rollout
    ctrl_start = prefix_len
    ctrl_end = ctrl_start + rollout_steps * _k_ctrl
    ctrl_end = min(ctrl_end, len(tokens_np))
    raw_ctrl = tokens_np[ctrl_start:ctrl_end]

    # Pad if needed
    if len(raw_ctrl) < rollout_steps * _k_ctrl:
        pad_len = rollout_steps * _k_ctrl - len(raw_ctrl)
        raw_ctrl = np.concatenate([raw_ctrl, np.zeros(pad_len, dtype=np.int32)])

    control_tokens = torch.tensor(raw_ctrl, dtype=torch.long).unsqueeze(0)  # (1, N)

    # Ground-truth frame tokens for the rollout window (for decoding)
    gt_frame_start = prefix_len
    gt_frame_end = gt_frame_start + rollout_steps * _k_frame
    gt_frame_end = min(gt_frame_end, len(tokens_np))
    gt_frame_tokens = tokens_np[gt_frame_start:gt_frame_end]

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    if str(checkpoint_path).strip().lower() == "mock":
        model = _MockWhamModel()
        log.info("run_demo: using mock WhamModel (no real checkpoint)")
    else:
        from imas_ambix.model.wham import WhamModel

        model = WhamModel.from_pretrained(str(checkpoint_path))
        log.info("run_demo: loaded checkpoint from %s", checkpoint_path)

    # ------------------------------------------------------------------
    # 4. Run rollout
    # ------------------------------------------------------------------
    cfg = rollout_config or RolloutConfig(
        prefix_tokens=prefix_len,
        rollout_steps=rollout_steps,
        top_k=64,
        temperature=0.8,
    )
    # Override prefix_tokens / rollout_steps from the ms-based calculation
    cfg = RolloutConfig(
        prefix_tokens=prefix_len,
        rollout_steps=rollout_steps,
        top_k=cfg.top_k,
        temperature=cfg.temperature,
        force_signal_action_tokens=cfg.force_signal_action_tokens,
    )

    log.info(
        "run_demo: rollout prefix_tokens=%d rollout_steps=%d",
        cfg.prefix_tokens,
        cfg.rollout_steps,
    )

    rollout_result = rollout(
        model=model,
        initial_tokens=prefix_tokens,
        control_tokens=control_tokens,
        config=cfg,
        k_frame=_k_frame,
        k_ctrl=_k_ctrl,
    )

    predicted_np = rollout_result["predicted_tokens"][0].numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # 5. Decode tokens to frames
    # ------------------------------------------------------------------
    if decode_frames:
        gt_frames = _decode_tokens_to_frames(
            gt_frame_tokens, tokenizer_version=tokenizer_version
        )
        pred_frames = _decode_tokens_to_frames(
            predicted_np, tokenizer_version=tokenizer_version
        )
    else:
        # Minimal placeholder frames for artefact writing
        gt_frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        pred_frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 6. Save Zarr artefacts
    # ------------------------------------------------------------------
    gt_zarr_path = output_dir / "ground-truth.zarr"
    pred_zarr_path = output_dir / "prediction.zarr"
    tok_gt_zarr_path = output_dir / "tokens-ground-truth.zarr"
    tok_pred_zarr_path = output_dir / "tokens-prediction.zarr"

    _save_frames_zarr(gt_zarr_path, gt_frames)
    _save_frames_zarr(pred_zarr_path, pred_frames)
    _save_tokens_zarr(tok_gt_zarr_path, gt_frame_tokens)
    _save_tokens_zarr(tok_pred_zarr_path, predicted_np)

    # ------------------------------------------------------------------
    # 7. Compute metrics
    # ------------------------------------------------------------------
    # Align frame counts (rollout may produce more / fewer frames than GT)
    t_min = min(len(gt_frames), len(pred_frames))
    if t_min == 0:
        t_min = 1
        gt_frames_aligned = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        pred_frames_aligned = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    else:
        gt_frames_aligned = gt_frames[:t_min]
        pred_frames_aligned = pred_frames[:t_min]

    metrics = compute_all_metrics(gt_frames_aligned, pred_frames_aligned)

    metrics_path = output_dir / "metrics.json"
    # Convert any non-finite floats to strings for JSON compatibility
    metrics_serialisable = {
        k: (v if (v == v and v != float("inf") and v != float("-inf")) else str(v))
        for k, v in metrics.items()
    }
    metrics_path.write_text(json.dumps(metrics_serialisable, indent=2))
    log.info("run_demo: metrics written to %s", metrics_path)

    # ------------------------------------------------------------------
    # 8. Render 4-panel figure
    # ------------------------------------------------------------------
    figure_path = output_dir / f"demo-{shot_id}.png"
    _render_figure(
        shot_id=shot_id,
        gt_frames=gt_frames,
        pred_frames=pred_frames,
        rollout_ms=rollout_ms,
        output_path=figure_path,
    )

    # ------------------------------------------------------------------
    # 9. Optional MP4
    # ------------------------------------------------------------------
    video_path: Path | None = None
    if not no_video:
        video_path = _render_video(
            shot_id=shot_id,
            gt_frames=gt_frames,
            pred_frames=pred_frames,
            output_dir=output_dir,
        )

    return DemoArtifacts(
        ground_truth_zarr=gt_zarr_path,
        prediction_zarr=pred_zarr_path,
        tokens_ground_truth_zarr=tok_gt_zarr_path,
        tokens_prediction_zarr=tok_pred_zarr_path,
        metrics_json=metrics_path,
        figure_png=figure_path,
        video_mp4=video_path,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_or_synthesise_stream(
    shot_id: int,
    tokenizer_version: str,
    prefix_ms: int,
    rollout_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load persisted stream or return synthetic data if not present on disk."""
    from imas_ambix.data.persist import load_shot_stream

    try:
        tokens_np, block_kind_np = load_shot_stream(
            shot_id, vocab_version=tokenizer_version
        )
        log.info(
            "run_demo: loaded persisted stream for shot %d (%d tokens)",
            shot_id,
            len(tokens_np),
        )
        return tokens_np, block_kind_np
    except FileNotFoundError:
        log.warning(
            "run_demo: no persisted stream for shot %d; using synthetic data "
            "(pipeline smoke-test only, not real physics).",
            shot_id,
        )
        # Synthesise enough tokens for the requested duration
        total_steps = (prefix_ms + rollout_ms) * MODEL_HZ // 1000 + 10
        n_tokens = total_steps * (K_FRAME_DEFAULT + K_CTRL_DEFAULT)
        rng = np.random.default_rng(shot_id)
        tokens_np = rng.integers(0, 256, size=n_tokens, dtype=np.int32)
        block_kind_np = np.zeros(n_tokens, dtype=np.uint8)
        return tokens_np, block_kind_np


def _save_frames_zarr(path: Path, frames: np.ndarray) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames to a Zarr group."""
    import zarr

    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store.create_array("frames", data=np.asarray(frames, dtype=np.uint8))
    store.attrs.update({"shape": list(frames.shape), "dtype": "uint8"})


def _save_tokens_zarr(path: Path, tokens: np.ndarray) -> None:
    """Write a 1-D int32 token array to a Zarr group."""
    import zarr

    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=np.asarray(tokens, dtype=np.int32))
    store.attrs.update({"n_tokens": int(len(tokens))})


def _pick_frame_times(n_frames: int, rollout_ms: int) -> list[int]:
    """Return frame indices for t=0, 0.2, 0.5, 1.0 s within the rollout."""
    targets_s = [0.0, 0.2, 0.5, 1.0]
    fps = n_frames / (rollout_ms / 1000.0) if rollout_ms > 0 else 1.0
    indices = []
    for t in targets_s:
        idx = min(int(t * fps), n_frames - 1)
        indices.append(idx)
    return indices


def _render_figure(
    shot_id: int,
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    rollout_ms: int,
    output_path: Path,
) -> None:
    """Render and save a 4-panel ground-truth vs prediction comparison figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("run_demo: matplotlib not available, skipping figure render.")
        output_path.write_bytes(b"")  # create empty placeholder
        return

    n_gt = len(gt_frames)
    n_pred = len(pred_frames)
    gt_indices = _pick_frame_times(n_gt, rollout_ms)
    pred_indices = _pick_frame_times(n_pred, rollout_ms)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"MAST shot {shot_id} — Ground truth (top) vs Prediction (bottom)",
        fontsize=14,
    )
    time_labels = ["t = 0.0 s", "t = 0.2 s", "t = 0.5 s", "t = 1.0 s"]

    for col, (gt_idx, pred_idx, label) in enumerate(
        zip(gt_indices, pred_indices, time_labels, strict=False)
    ):
        ax_gt = axes[0, col]
        ax_pred = axes[1, col]

        gt_frame = gt_frames[gt_idx]
        pred_frame = pred_frames[pred_idx]

        ax_gt.imshow(gt_frame if gt_frame.ndim == 3 else gt_frame[..., 0], cmap="gray")
        ax_gt.set_title(f"GT  {label}")
        ax_gt.axis("off")

        ax_pred.imshow(
            pred_frame if pred_frame.ndim == 3 else pred_frame[..., 0], cmap="gray"
        )
        ax_pred.set_title(f"Pred  {label}")
        ax_pred.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("run_demo: figure saved to %s", output_path)


def _render_video(
    shot_id: int,
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    output_dir: Path,
) -> Path | None:
    """Render a side-by-side GT/prediction MP4 using imageio.

    Returns the path to the written video, or ``None`` if imageio or
    imageio-ffmpeg is unavailable.
    """
    try:
        import imageio  # noqa: F401
    except ImportError:
        log.info("run_demo: imageio not available — skipping MP4.")
        return None

    # Check ffmpeg plugin is available
    try:
        import imageio_ffmpeg as _  # noqa: F401
    except ImportError:
        log.info("run_demo: imageio-ffmpeg not available — skipping MP4.")
        return None

    try:
        n = min(len(gt_frames), len(pred_frames))
        if n == 0:
            return None

        h = max(gt_frames.shape[1], pred_frames.shape[1])

        video_path = output_dir / f"demo-{shot_id}.mp4"
        frames_out = []
        for i in range(n):
            gt_f = gt_frames[i]
            pred_f = pred_frames[i]
            # Pad height if needed
            if gt_f.shape[0] < h:
                gt_f = np.pad(gt_f, [(0, h - gt_f.shape[0]), (0, 0), (0, 0)])
            if pred_f.shape[0] < h:
                pred_f = np.pad(pred_f, [(0, h - pred_f.shape[0]), (0, 0), (0, 0)])
            side_by_side = np.concatenate([gt_f, pred_f], axis=1)
            frames_out.append(side_by_side)

        imageio.mimsave(str(video_path), frames_out, fps=10, format="ffmpeg")
        log.info("run_demo: video saved to %s", video_path)
        return video_path
    except Exception as exc:  # noqa: BLE001
        log.warning("run_demo: video render failed: %s", exc)
        return None
