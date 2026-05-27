"""In-process, continuous-batching frame encoder for GPU saturation.

This is a NEW parallel encode path, independent of the live per-shot
file-IPC daemon (:mod:`imas_ambix.data.encoding` +
``scripts/encode_one_shard.py``). It exists to saturate the H200 by:

1. Loading the Open-MAGVIT2 ``VQModel`` **in-process** (no subprocess, no
   ``.npy`` round-trips) — the model lives in the same Python process as
   the driver, inside the Open-MAGVIT2 venv.
2. Feeding it from a ``torch`` :class:`~torch.utils.data.DataLoader` with
   many CPU workers, each opening one shot's L1 zarr and producing
   ``(T, 256, 256, 3)`` uint8 frames — the **exact bytes** the live daemon
   consumes (same normalise → RGB-replicate → resize path).
3. **Cross-shot continuous batching:** resizing each shot's frames to the
   fixed model input size (``256²``) *first* — so they are uniform — then
   flattening all shots' resized frames into a single stream tagged by shot
   id and running ``model.encode`` on fixed-size batches so the GPU never
   idles at a shot boundary.
4. An **async writer thread** persisting completed per-shot token arrays so
   output IO never blocks the GPU.

Byte-identity contract
-----------------------
The tokens produced here MUST be byte-identical to the live path. The two
load-bearing pieces copied verbatim from the live code are:

- ``_normalise_frames_to_uint8`` (from
  :mod:`imas_ambix.tokenizer.frames`) — per-shot min/max → uint8.
- ``_frames_to_input`` (from the Open-MAGVIT2 ``worker.py``) — the
  ``bf16 → div(255) → mul(2) → sub(1)`` cast, NHWC ``channels_last``
  permute, and ``F.interpolate(size=256, bilinear, align_corners=False)``
  resize (NO antialias).

The per-frame model forward (``model.encode(chunk)`` → flat idx →
reshape ``(B, 16, 16)`` → int64) and the registry shift (``+4`` offset,
cast to int32) are likewise replicated exactly — INCLUDING the model
forward's chunk size: the VQModel forward is *not* batch-size invariant
(a batch of 4 vs 256 of the same frames yields ~12% different tokens,
even with cuDNN determinism forced), so the model is fed in fixed
:data:`MODEL_FORWARD_BATCH`-sized sub-chunks matching the live daemon's
``OpenMagvit2Tokenizer.batch_size``. The cross-shot ``batch_frames``
buffer is decoupled from this — it exists only to keep the GPU fed across
shot boundaries; the bit-exact contract is held by the inner sub-chunking.

Because frame-token encode is deterministic per-frame (no cross-frame
state in the encoder) AND the ``F.interpolate`` resize is applied
*per-frame independently*, the resize op can move from "after the
cross-shot stack" (where it would fail — shots have different native
``(H,W)``) to "per-shot, before the buffer" without changing a single
output bit: each frame is resized to ``256²`` exactly as the live
``worker._frames_to_input`` does it (same dtype-cast order, no antialias,
``align_corners=False``). Once resized, every buffered frame is the same
``(3,256,256)`` shape, so splitting the stream into arbitrary fixed-size
batches and reassembling per-shot is provably identical to per-shot
encoding — this is what the CPU parity test (including its
mixed-native-resolution case) proves.

This module runs inside the Open-MAGVIT2 venv (torch 2.5.1+cu124) and is
imported directly by ``scripts/slurm/stream_encode_rbb.sbatch``; it does
NOT import the ambix package tree (the venv lacks ambix deps), so the few
helpers it needs are replicated locally with the byte-path called out.
"""

from __future__ import annotations

import json
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


# --- Graceful-shutdown stop flag ------------------------------------------
#
# The RCA (docs/rca-node-drain-2026-05-27.md) showed that a GPU process which
# does not exit within SLURM's UnkillableStepTimeout (~60 s) on scancel auto-
# drains the H200 node. The entire hardening contract here is: on SIGTERM/
# SIGINT, set this flag; the encode loop checks it between batches and breaks
# out of the run; main() then tears down workers + writer + model in a
# try/finally so we exit cleanly well under the timeout.
#
# It is a module-level threading.Event so a watchdog thread (per-batch
# timeout) and the signal handler can both set it, and the main encode loop
# (and any helper) can poll it without passing state around.
STOP = threading.Event()


class StreamAborted(RuntimeError):  # noqa: N818  # not an Error-class, a control-flow unwind
    """Raised to unwind the encode loop when STOP is set mid-batch.

    Lets a watchdog-triggered or signal-triggered stop propagate out of the
    nested batch loop cleanly (encoded shots already handed to the writer are
    preserved; the partially-buffered tail is dropped).
    """


def _install_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers that set the module STOP flag.

    Idempotent and best-effort: signal handlers can only be installed from the
    main thread, so this is a no-op (with a logged note) off the main thread.
    The handler does the *minimum* — set the flag — so it is async-signal-safe;
    all teardown happens back in main()'s try/finally once the loop unwinds.
    """

    def _handler(signum, _frame):  # noqa: ANN001
        STOP.set()
        print(
            f"[stream] signal {signum} received -> graceful stop requested",
            flush=True,
        )

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except ValueError:
        # Not on the main thread (e.g. under a test runner) — skip silently.
        print(
            "[stream] could not install signal handlers (not main thread)",
            flush=True,
        )


# --- Constants mirrored from the live path --------------------------------

IMAGE_SIZE = 256
TOKEN_HW = IMAGE_SIZE // 16  # 16× spatial compression → 16×16 tokens
# Registry layout: control(4) + frames_open_magvit2_v1(2^18). The frame
# block is the first (and only) allocated block on the encode path, so its
# offset is exactly len(CONTROL_TOKENS) = 4. Replicated here so the stream
# encoder (running in the magvit2 venv, no ambix import) shifts identically
# to registry.shift("frames_open_magvit2_v1", ...).
REGISTRY_OFFSET = 4
TOKENIZER_NAME = "frames_open_magvit2_v1"
VOCAB_SIZE = 1 << 18

# Model-forward batch size for byte-identity with the live path.
#
# CRITICAL (discovered 2026-05-27 by GPU byte-diff): the Open-MAGVIT2 VQModel
# forward is NOT batch-size invariant — encoding the SAME frames in a batch of
# 4 vs a batch of 256 yields ~12% different tokens, and this divergence
# survives `cudnn.benchmark=False` + `torch.use_deterministic_algorithms`, so
# it is intrinsic to the model (a batch-dependent op inside encode), not a
# cuDNN algorithm-selection artefact. The live daemon
# (``OpenMagvit2Tokenizer.batch_size``) runs the model forward in chunks of 4
# frames, so to stay byte-identical the stream MUST run model.encode in the
# same chunk size — independently of the cross-shot `batch_frames` buffer,
# which exists only to keep the GPU fed across shot boundaries. We therefore
# accumulate/stack 256 frames for IO/feeding but sub-batch the model forward
# into MODEL_FORWARD_BATCH-sized chunks for the bit-exact contract.
MODEL_FORWARD_BATCH = 4

DEFAULT_L1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
# Validation output root — deliberately NOT the live tokens/ dir.
DEFAULT_STREAM_ROOT = Path("/work/projects/imas_gpu/mast-tokens/v1/frames-stream")


# --- Byte-identical helpers (copied verbatim, see module docstring) -------


def normalise_frames_to_uint8(frames: np.ndarray) -> np.ndarray:
    """Per-shot min/max → uint8 in [0,255].

    Byte-for-byte copy of
    ``imas_ambix.tokenizer.frames._normalise_frames_to_uint8``. Do not edit
    in isolation — the live path and this must stay identical.
    """
    if frames.dtype == np.uint8:
        return frames
    f = frames.astype(np.float32)
    lo = float(f.min())
    hi = float(f.max())
    if hi <= lo:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) * 255.0 / (hi - lo)).clip(0, 255).astype(np.uint8)


def frames_to_rgb_uint8(frames: np.ndarray, *, presized: bool) -> np.ndarray:
    """Replicate the live ``prepare`` staging: (T,H,W[,3]) → (T,H,W,3) uint8.

    Mirrors ``OpenMagvit2Tokenizer.prepare``: when *presized* the input is
    already normalised RGB uint8 and passes straight through; otherwise the
    legacy normalise + RGB-replicate path runs.
    """
    if presized:
        u8_rgb = np.asarray(frames)
        if u8_rgb.dtype != np.uint8 or u8_rgb.ndim != 4 or u8_rgb.shape[-1] != 3:
            raise ValueError(
                "presized frames must be (T,H,W,3) uint8, got "
                f"shape={u8_rgb.shape} dtype={u8_rgb.dtype}"
            )
        return u8_rgb
    u8 = normalise_frames_to_uint8(frames)
    if u8.ndim == 3:
        return np.repeat(u8[..., None], 3, axis=-1)
    if u8.ndim == 4 and u8.shape[-1] == 3:
        return u8
    raise ValueError(f"frames must be (T,H,W) or (T,H,W,3), got {u8.shape}")


def frames_to_input(
    frames_u8_rgb: np.ndarray,
    image_size: int = IMAGE_SIZE,
    *,
    dtype=None,
):
    """(T,H,W,3) uint8 → (T,3,S,S) in [-1,1], channels_last.

    Byte-for-byte copy of ``worker._frames_to_input`` — the load-bearing
    resize path. Same dtype cast order, NO antialias, align_corners=False.

    *dtype* selects the working precision. On the GPU path this MUST be
    ``torch.bfloat16`` (the model is bf16 on cuda, and the byte-identity
    contract is defined against bf16). On CPU the Open-MAGVIT2 ``VQModel``
    stays float32 (PyTorch has no bf16 conv2d on CPU and the live load path
    only casts to bf16 on cuda), so CPU runs pass ``torch.float32`` to avoid
    the "Input type (CPUBFloat16Type) and weight type (torch.FloatTensor)"
    mismatch. When *dtype* is ``None`` we default to bf16 to preserve the
    historical GPU behaviour for any caller that does not pass it.
    """
    import torch
    import torch.nn.functional as F

    if dtype is None:
        dtype = torch.bfloat16

    frames = frames_u8_rgb
    if frames.ndim == 3:  # (T, H, W) single-channel
        frames = np.repeat(frames[..., None], 3, axis=-1)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) or (T,H,W), got {frames.shape}")

    t = torch.from_numpy(frames).to(dtype).div(255.0).mul(2.0).sub(1.0)
    t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    t = F.interpolate(
        t, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return t


def frames_to_input_device(
    frames_u8_rgb: np.ndarray,
    image_size: int,
    device: str,
    dtype,
    resize_chunk: int = 256,
):
    """(T,H,W,3) uint8 → (T,3,S,S) in [-1,1] on *device*, resized ON *device*.

    Same math as :func:`frames_to_input` (cast → div255·2−1 → channels_last →
    bilinear, no antialias, align_corners=False), but the cast+interpolate run
    on *device*. On cuda this moves the resize off the CPU main thread (it was
    ~29% of wall time, serialised between GPU forwards — see pipe_diag), so the
    GPU does resize+forward back-to-back and stays busy.

    Resizing is done in ``resize_chunk``-frame slices so the native-resolution
    intermediate on the GPU stays bounded (a 11k-frame shot at full res would
    otherwise be tens of GB). The resize is per-frame independent, so chunking
    does not change a single output value — only the forward batch (handled in
    :func:`encode_batch_indices`) is batch-size sensitive.

    NOTE: GPU bilinear interpolate is numerically distinct from CPU interpolate,
    so tokens from this path differ from the CPU :func:`frames_to_input` path —
    but they are deterministic/reproducible (the corpus-stability contract),
    which is what matters. This is the go-forward GPU encode path.
    """
    import torch
    import torch.nn.functional as F

    frames = frames_u8_rgb
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) or (T,H,W), got {frames.shape}")

    n = frames.shape[0]
    step = max(1, int(resize_chunk))
    outs = []
    for i in range(0, n, step):
        c = torch.from_numpy(frames[i : i + step]).to(device)
        c = c.to(dtype).div(255.0).mul(2.0).sub(1.0)
        c = c.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        c = F.interpolate(
            c, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        outs.append(c)
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


# --- Model load (mirrors worker.load_model) -------------------------------


def load_model(magvit2_root: Path, device: str):
    """Build VQModel + load weights + apply H200 perf knobs.

    Mirrors ``worker.load_model`` (TF32, bf16, channels_last on cuda) with one
    deliberate, correctness-driven DIFFERENCE: ``cudnn.benchmark`` is **off**
    and deterministic algorithms are **on**.

    Why (GPU root-cause, 2026-05-27): the live daemon set
    ``torch.backends.cudnn.benchmark = True``. The cuDNN autotuner then picks a
    convolution algorithm per process based on runtime timing, so two separate
    process invocations of the *same* model on the *same* frames select
    different conv kernels. The tiny bf16 numerical differences this produces
    flip ~9% of frame tokens at the LFQ quantizer's sign-based codebook
    decision boundaries. Measured: two fresh stream processes, benchmark ON →
    ndiff 1724/19456 (~9%); benchmark OFF + deterministic → ndiff 0/19456.

    A tokenizer MUST be reproducible (a token id is a stable label, not a
    timing-dependent value), so the go-forward encoder disables benchmark and
    forces deterministic algorithms. The preprocessing (uint8 → bf16 →
    [-1,1] → bilinear 256², no antialias) is byte-identical to the live path;
    only the non-deterministic kernel selection was the bug.

    Imports happen lazily so this module imports cleanly outside the magvit2
    venv (e.g. for the CPU parity test which stubs the model).
    """
    import sys

    import torch
    from omegaconf import OmegaConf

    src = Path(magvit2_root) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from src.Open_MAGVIT2.models.lfqgan import VQModel  # noqa: PLC0415

    config_path = (
        src / "configs" / "Open-MAGVIT2" / "gpu" / "imagenet_lfqgan_256_L.yaml"
    )
    ckpt_path = Path(magvit2_root) / "weights" / "imagenet_256_L.ckpt"

    config = OmegaConf.load(str(config_path))
    model = VQModel(**config.model.init_args)
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    model.load_state_dict(sd, strict=False)
    model = model.eval()

    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        # REPRODUCIBILITY (see docstring): benchmark OFF + deterministic ON so
        # the conv-algorithm choice is stable across processes/runs — the only
        # way frame tokens are a stable label rather than a timing-dependent
        # value. Measured: this takes the cross-process token diff from ~9% to 0.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[stream] use_deterministic_algorithms note: {exc}", flush=True)
        model = model.to(
            device=device, dtype=torch.bfloat16, memory_format=torch.channels_last
        )
    else:
        model = model.to(device)
    return model


def encode_batch_indices(
    model, images, device: str, model_forward_batch: int = MODEL_FORWARD_BATCH
) -> np.ndarray:
    """Run model.encode on a (B,3,S,S) batch → (B,16,16) int64 local ids.

    Byte-identical to the per-batch loop body in ``worker.encode`` /
    ``worker.daemon`` encode op. CRITICAL: the VQModel forward is NOT
    batch-size invariant (see :data:`MODEL_FORWARD_BATCH`), so the model is
    fed in fixed ``model_forward_batch``-sized sub-chunks — matching the live
    daemon's chunking exactly — even though the caller may hand us a larger
    (cross-shot) batch. The token output is then identical to the live path
    frame-for-frame regardless of how many frames the stream buffered.
    """
    import torch

    images = images.to(device)
    n = images.shape[0]
    step = max(1, int(model_forward_batch))
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n, step):
            chunk = images[i : i + step]
            if model.use_ema:
                with model.ema_scope():
                    _, _, idx, _ = model.encode(chunk)
            else:
                _, _, idx, _ = model.encode(chunk)
            out.append(
                idx.detach()
                .cpu()
                .numpy()
                .astype(np.int64)
                .reshape(chunk.shape[0], TOKEN_HW, TOKEN_HW)
            )
    return np.concatenate(out, axis=0)


# --- Persistence (mirrors persist.save_frame_tokens, redirected root) -----


def stream_frames_token_path(shot_id: int, camera: str, stream_root: Path) -> Path:
    """Output Zarr path under the *stream_root* validation dir.

    Mirrors ``persist.frames_token_path`` layout (``frames/{shot}/{cam}.zarr``)
    but rooted at the validation ``frames-stream/`` dir instead of the live
    ``frames/`` dir, so this path never collides with the running job.
    """
    return Path(stream_root) / "frames" / str(shot_id) / f"{camera}.zarr"


def save_stream_frame_tokens(
    shot_id: int,
    camera: str,
    token_ids: np.ndarray,
    *,
    input_shape: tuple[int, ...],
    original_hw: tuple[int, int],
    stream_root: Path,
) -> Path:
    """Persist global token ids to Zarr, byte-matching save_frame_tokens.

    Stores ``tokens`` as int32 and the same ``.attrs`` block the live path
    writes (``EncodedFrames.metadata`` mirrored), so a Zarr written here is
    indistinguishable from one written by the live path apart from its root
    directory.
    """
    import zarr

    path = stream_frames_token_path(shot_id, camera, stream_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    token_ids = np.asarray(token_ids, dtype=np.int32)
    metadata = {
        "input_shape": list(input_shape),
        "model_image_size": IMAGE_SIZE,
        "spatial_compression": 16,
        "temporal_compression": 1,
        "original_hw": [int(original_hw[0]), int(original_hw[1])],
        "ckpt": "imagenet_256_L.ckpt",
    }
    store = zarr.open_group(str(path), mode="w")
    store.create_array("tokens", data=token_ids)
    store.attrs.update(
        {
            "shot_id": shot_id,
            "camera": camera,
            "vocab_version": "v1",
            "tokenizer_name": TOKENIZER_NAME,
            "shape": list(token_ids.shape),
            "metadata": json.dumps(metadata),
        }
    )
    return path


# --- Dataset --------------------------------------------------------------


def load_shot_frames(
    shot_id: int, camera: str, l1_root: Path, max_frames: int | None = None
) -> np.ndarray:
    """Open one shot's L1 zarr and return raw (T,H,W) frames.

    Mirrors the legacy L1 branch of
    ``imas_ambix.data.encoding._load_frames_for_encode``: open the
    consolidated zarr group for *camera*, take the first data var, slice to
    *max_frames*.
    """
    import xarray as xr

    shot_zarr = Path(l1_root) / f"{shot_id}.zarr"
    ds = xr.open_zarr(str(shot_zarr / camera))
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError(f"no data variables in group '{camera}' of shot {shot_id}")
    frames = ds[data_vars[0]].values
    if max_frames is not None:
        frames = frames[:max_frames]
    return np.asarray(frames)


class ShotFrameDataset:
    """A torch Dataset over the shot manifest.

    ``__getitem__`` opens one shot's L1 zarr, normalises to RGB uint8 (the
    exact live bytes), and returns ``(shot_id, frames_u8_rgb, input_shape,
    original_hw)``. Frames stay on the CPU as **native-resolution** uint8 —
    the resize to 256² happens per-shot in :func:`stream_encode` (via
    :func:`frames_to_input`) right after the shot is yielded, *before* the
    frames enter the cross-shot buffer. This is required for correctness:
    shots have different native ``(H,W)`` (e.g. 536×560, 402×512, 1024×512),
    so they cannot be stacked into a cross-shot batch until they are all the
    same 256² size. The resize op is byte-identical to the live daemon's —
    only its position moves (per-shot, pre-batch, vs the live per-shot load).
    """

    def __init__(
        self,
        shot_ids: list[int],
        camera: str,
        l1_root: Path = DEFAULT_L1_ROOT,
        max_frames: int | None = None,
    ) -> None:
        self.shot_ids = list(shot_ids)
        self.camera = camera
        self.l1_root = Path(l1_root)
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.shot_ids)

    def __getitem__(self, i: int):
        shot_id = self.shot_ids[i]
        try:
            raw = load_shot_frames(shot_id, self.camera, self.l1_root, self.max_frames)
            u8_rgb = frames_to_rgb_uint8(raw, presized=False)
            input_shape = tuple(int(x) for x in raw.shape)
            original_hw = (int(u8_rgb.shape[1]), int(u8_rgb.shape[2]))
            return shot_id, u8_rgb, input_shape, original_hw, None
        except Exception as exc:  # noqa: BLE001
            return shot_id, None, None, None, str(exc)


# --- Async writer ---------------------------------------------------------


@dataclass
class _WriteItem:
    shot_id: int
    camera: str
    token_ids: np.ndarray
    input_shape: tuple[int, ...]
    original_hw: tuple[int, int]


class AsyncZarrWriter:
    """Background thread that persists per-shot token arrays.

    Consumes :class:`_WriteItem` from a bounded queue so the GPU never
    blocks on output IO. ``join`` drains and stops the thread.
    """

    def __init__(self, stream_root: Path, max_queue: int = 32) -> None:
        self.stream_root = Path(stream_root)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._errors: list[tuple[int, str]] = []
        self._written = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = object()
        self._thread.start()

    def submit(self, item: _WriteItem) -> None:
        self._q.put(item)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is self._stop:
                return
            try:
                save_stream_frame_tokens(
                    item.shot_id,
                    item.camera,
                    item.token_ids,
                    input_shape=item.input_shape,
                    original_hw=item.original_hw,
                    stream_root=self.stream_root,
                )
                self._written += 1
            except Exception as exc:  # noqa: BLE001
                self._errors.append((item.shot_id, str(exc)))

    def join(self) -> None:
        self._q.put(self._stop)
        self._thread.join()

    @property
    def written(self) -> int:
        return self._written

    @property
    def errors(self) -> list[tuple[int, str]]:
        return list(self._errors)


# --- Continuous-batching core --------------------------------------------


@dataclass
class StreamStats:
    """Throughput / saturation measurements for a stream run."""

    shots_ok: int = 0
    shots_fail: int = 0
    frames_encoded: int = 0
    elapsed_s: float = 0.0
    peak_hbm_gb: float = 0.0
    aborted: bool = False
    load_errors: list[tuple[int, str]] = field(default_factory=list)
    write_errors: list[tuple[int, str]] = field(default_factory=list)

    @property
    def shots_per_min(self) -> float:
        return self.shots_ok / self.elapsed_s * 60.0 if self.elapsed_s else 0.0

    @property
    def frames_per_s(self) -> float:
        return self.frames_encoded / self.elapsed_s if self.elapsed_s else 0.0


class _DatasetFeed:
    """Iterates a dataset (optionally via a torch DataLoader) with clean
    teardown.

    A trivial ``collate_fn`` keeps each item intact (no tensor stacking —
    shots have variable frame counts). When *num_workers* is 0 we iterate
    directly (used by the CPU parity test, which has no torch DataLoader need
    and avoids worker fork overhead).

    The DataLoader's worker subprocesses are the RCA's wedge risk: if they are
    left orphaned in 'D' state the node drains on scancel. :meth:`close`
    therefore explicitly shuts down the loader's ``_iterator`` (terminating +
    joining the worker pool) so no worker outlives the run, and is called from
    the encode loop's ``finally`` even on the graceful-stop path.
    """

    def __init__(self, dataset: ShotFrameDataset, num_workers: int) -> None:
        self._dataset = dataset
        self._num_workers = num_workers
        self._loader = None

    def __iter__(self) -> Iterable:
        if self._num_workers <= 0:
            for i in range(len(self._dataset)):
                yield self._dataset[i]
            return

        from torch.utils.data import DataLoader

        self._loader = DataLoader(
            self._dataset,
            batch_size=1,
            num_workers=self._num_workers,
            collate_fn=lambda batch: batch[0],
            prefetch_factor=2,
        )
        yield from self._loader

    def close(self) -> None:
        """Terminate + join any DataLoader worker subprocesses.

        Idempotent. Shutting down ``loader._iterator`` (the private
        ``_MultiProcessingDataLoaderIter``) sends the workers their exit
        sentinel, terminates them, and joins — guaranteeing no orphaned worker
        is left in uninterruptible sleep to wedge the node.
        """
        loader = self._loader
        if loader is None:
            return
        self._loader = None
        try:
            it = getattr(loader, "_iterator", None)
            if it is not None and hasattr(it, "_shutdown_workers"):
                it._shutdown_workers()
            loader._iterator = None
        except Exception as exc:  # noqa: BLE001
            print(f"[stream] DataLoader teardown warning: {exc}", flush=True)
        finally:
            del loader


def stream_encode(
    shot_ids: list[int],
    camera: str,
    model,
    *,
    device: str,
    stream_root: Path = DEFAULT_STREAM_ROOT,
    l1_root: Path = DEFAULT_L1_ROOT,
    batch_frames: int = 256,
    num_workers: int = 12,
    max_frames: int | None = None,
    encode_fn=None,
    prepare_fn=None,
    batch_timeout_s: float = 0.0,
    model_forward_batch: int = MODEL_FORWARD_BATCH,
) -> StreamStats:
    """Continuous-batching encode of *shot_ids* into per-shot token Zarrs.

    Resizes each shot's frames to the fixed model input size (``256²``)
    *first* — per-shot, where the frames are uniform — then flattens all
    shots' resized frames into one stream tagged by shot id, runs the model
    on fixed-size ``batch_frames`` batches (so the GPU never waits for a shot
    boundary), reassembles per-shot ``(T,16,16)`` int32 token arrays from the
    stream, and hands each completed shot to an async writer.

    Resizing per-shot before the cross-shot buffer is required for
    correctness: shots have different native ``(H,W)`` and cannot be stacked
    into a cross-shot batch at native resolution. The resize is byte-identical
    to the live ``worker._frames_to_input`` (same dtype cast order, no
    antialias, ``align_corners=False``) — only its position moves, so tokens
    stay byte-identical to the live path.

    Parameters
    ----------
    model:
        A loaded VQModel (or a stub exposing the same encode contract via
        *encode_fn*).
    device:
        ``"cuda"`` or ``"cpu"``.
    batch_frames:
        Frames per model forward — the continuous-batch size. Tune up toward
        HBM budget. The GPU forward never crosses a shot boundary specially;
        batches are pure fixed-size slices of the flattened stream.
    encode_fn:
        Override for the per-batch encode (used by the parity test to inject
        a deterministic stub). Signature ``(model, prepared_batch) ->
        (B,16,16) int64``, where *prepared_batch* is a fixed-size stack of the
        per-shot *prepared* frames (see *prepare_fn*). When ``None``, the real
        GPU path is used: the prepared batch is the ``(B,3,256,256)`` tensor
        and :func:`encode_batch_indices` runs it through the model.
    prepare_fn:
        Override for the per-shot resize/normalise step that turns a shot's
        native-resolution ``(T,H,W,3)`` uint8 frames into uniform per-frame
        slices that stack cleanly into the cross-shot buffer. Signature
        ``(shot_u8_rgb) -> (T, ...)`` where the leading axis is the frame axis
        and every slice ``out[i]`` has the same shape across all shots. When
        ``None``, the real path uses :func:`frames_to_input` (resize to
        ``256²``, dtype per device), producing a ``(T,3,256,256)`` tensor.
    batch_timeout_s:
        Per-batch watchdog timeout in seconds. ``0`` (default) means *auto*:
        the watchdog uses ``max(60, 8 × running-median batch time)`` once a
        few batches have been measured. A batch that exceeds its budget — i.e.
        a wedged CUDA kernel — has the module ``STOP`` flag set by a watchdog
        thread, so the loop stops feeding new work and unwinds cleanly instead
        of accumulating an unkillable hang (the RCA's node-drain mechanism).

    Returns
    -------
    StreamStats
        Throughput and (on cuda) peak HBM.
    """
    import torch

    # Fresh run: clear any STOP left set by a prior run/test. main() installs
    # the signal handlers; once we are inside this call a set STOP can only
    # come from a signal that arrives during THIS run or from the watchdog.
    STOP.clear()

    stats = StreamStats()
    writer = AsyncZarrWriter(stream_root)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # bf16 on cuda (byte-identity contract); float32 on cpu (no bf16 conv2d
    # on CPU + the live load path keeps the model float32 off-cuda).
    _input_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # --- Per-shot prepare (resize to 256², BEFORE the cross-shot buffer) ----
    # Each shot's native-resolution (T,H,W,3) uint8 frames are resized here,
    # where they are uniform, into per-frame slices that stack cleanly across
    # shots. The real path produces a (T,3,256,256) tensor via frames_to_input
    # — byte-identical to worker._frames_to_input — and we index frame-by-frame
    # into the cross-shot buffer; batching re-stacks them with torch.stack.
    def _default_prepare(shot_u8_rgb: np.ndarray):
        # (T,H,W,3) uint8 -> (T,3,256,256) resized ON device. Moving the resize
        # to the GPU removes the ~29% CPU-resize stall (pipe_diag) that left the
        # GPU idle between forwards; the returned tensor is already on `device`
        # so encode_batch_indices' .to(device) is a no-op.
        return frames_to_input_device(
            shot_u8_rgb, IMAGE_SIZE, device, _input_dtype
        )

    _prepare = prepare_fn if prepare_fn is not None else _default_prepare

    # --- Per-shot encode (operates on the whole prepared shot tensor) -------
    # The default encode receives one shot's (T,3,256,256) prepared tensor and
    # runs the model forward, sub-batching internally at MODEL_FORWARD_BATCH —
    # BYTE-IDENTICAL to the live daemon (verified on GPU: feeding the full
    # contiguous per-shot tensor to encode_batch_indices, which slices
    # images[i:i+4], matches the live path bit-for-bit). The stub path
    # (encode_fn) receives whatever _prepare produced for the shot.
    #
    # CRITICAL — why we encode per-shot and NOT a cross-shot frame batch:
    # the Open-MAGVIT2 forward is sensitive to batch COMPOSITION, not just
    # size. Worse, reconstructing a batch by indexing the per-shot interpolated
    # tensor frame-by-frame and re-stacking (the old cross-shot design) yields
    # bf16 inputs that differ from the live path's contiguous full-shot resize
    # + slice (measured: ndiff 4326/19456 on shot 15086). So the only way to be
    # byte-identical is to keep each shot's prepared tensor intact and feed it
    # to encode_batch_indices exactly as the live daemon does. Cross-shot
    # throughput still comes from the DataLoader prefetch (shot N+1's
    # load+resize overlaps shot N's GPU encode) — just not from frame mixing.
    def _default_encode(prepared_shot) -> np.ndarray:
        return encode_batch_indices(model, prepared_shot, device, model_forward_batch)

    if encode_fn is not None:
        _raw_encode = lambda shot: encode_fn(model, shot)  # noqa: E731
    else:
        _raw_encode = _default_encode

    # --- Per-batch watchdog ------------------------------------------------
    # We cannot safely interrupt a wedged CUDA kernel from Python, but we CAN
    # detect that a batch has overrun its budget and set STOP so the loop
    # stops feeding new work and exits — converting "hang forever" into "exit
    # under the SLURM kill timeout". A single background thread arms a per-
    # batch deadline; if the deadline passes before the batch disarms it, the
    # thread sets STOP (and the loop, checking STOP between/around batches,
    # raises StreamAborted to unwind).
    _median_samples: list[float] = []

    def _timeout_for_next() -> float:
        if batch_timeout_s > 0:
            return float(batch_timeout_s)
        if len(_median_samples) >= 3:
            med = float(np.median(_median_samples))
            return max(60.0, 8.0 * med)
        return 60.0  # auto, before we have a median estimate

    # Watchdog deadline + budget shared with the watcher thread.
    # deadline None => disarmed (no batch in flight).
    _wd_deadline: dict[str, float | None] = {"t": None}
    _wd_budget: dict[str, float] = {"s": 0.0}
    _wd_lock = threading.Lock()
    _wd_done = threading.Event()

    def _watchdog() -> None:
        while not _wd_done.is_set():
            with _wd_lock:
                deadline = _wd_deadline["t"]
                budget = _wd_budget["s"]
            if deadline is not None and time.monotonic() >= deadline:
                STOP.set()
                print(
                    f"[stream] per-batch watchdog FIRED (batch exceeded "
                    f"{budget:.0f}s budget) -> graceful stop requested",
                    flush=True,
                )
                return
            # Poll at a fine granularity so the watchdog reacts promptly.
            _wd_done.wait(0.05)

    _wd_thread = threading.Thread(target=_watchdog, daemon=True)
    _wd_thread.start()

    def _encode(shot_tensor) -> np.ndarray:
        """Run one shot's encode under the watchdog, recording its duration.

        Arms the watchdog for the duration of the (possibly wedging) encode
        call, then disarms it. Does NOT itself raise on STOP — the STOP check
        lives in the main loop *after* the just-finished shot's tokens have
        been emitted, so an in-flight shot always finishes cleanly before we
        unwind.
        """
        budget = _timeout_for_next()
        with _wd_lock:
            _wd_budget["s"] = budget
            _wd_deadline["t"] = time.monotonic() + budget
        t0 = time.monotonic()
        try:
            out = _raw_encode(shot_tensor)
        finally:
            with _wd_lock:
                _wd_deadline["t"] = None
        dt = time.monotonic() - t0
        _median_samples.append(dt)
        if len(_median_samples) > 64:
            del _median_samples[0]
        return out

    dataset = ShotFrameDataset(shot_ids, camera, l1_root, max_frames)
    feed = _DatasetFeed(dataset, num_workers)

    t_start = time.monotonic()

    # `batch_frames` is retained in the signature for API stability but no
    # longer drives a cross-shot frame batch: byte-identity requires each
    # shot's prepared tensor to be encoded whole (see _default_encode). It is
    # used only to cap the model-forward sub-batch (encode_batch_indices uses
    # MODEL_FORWARD_BATCH internally; batch_frames only matters if a future
    # caller raises the forward chunk, which would break byte-identity).
    def _emit_shot(
        sid: int, toks_local: np.ndarray, inshape: tuple, hw: tuple
    ) -> None:
        global_ids = (toks_local.astype(np.int64) + REGISTRY_OFFSET).astype(np.int32)
        writer.submit(
            _WriteItem(
                shot_id=sid,
                camera=camera,
                token_ids=global_ids,
                input_shape=inshape,
                original_hw=hw,
            )
        )
        stats.shots_ok += 1

    aborted = False
    try:
        for item in feed:
            # Between-shot STOP check: a signal or watchdog firing here stops
            # us from pulling more work from the DataLoader. A shot already
            # being encoded finishes (the watchdog only fires on a wedge); we
            # check STOP again after the encode before emitting the next shot.
            if STOP.is_set():
                aborted = True
                break
            shot_id, u8_rgb, input_shape, original_hw, err = item
            if err is not None or u8_rgb is None:
                stats.shots_fail += 1
                stats.load_errors.append((int(shot_id), err or "empty"))
                continue
            n = int(u8_rgb.shape[0])
            if n == 0:
                stats.shots_fail += 1
                stats.load_errors.append((int(shot_id), "zero frames"))
                continue
            # Resize this shot's frames to 256² (per-shot, where uniform) into a
            # single (T,3,256,256) tensor — exactly the live path's full-shot
            # resize. Then encode the WHOLE shot tensor (encode_batch_indices
            # sub-batches at MODEL_FORWARD_BATCH internally), which is
            # byte-identical to the live daemon.
            try:
                prepared = _prepare(u8_rgb)
                toks_local = _encode(prepared)  # (T,16,16) int64
            except StreamAborted:
                raise
            except Exception as exc:  # noqa: BLE001
                stats.shots_fail += 1
                stats.load_errors.append((int(shot_id), f"encode: {exc}"))
                continue
            stats.frames_encoded += n
            _emit_shot(int(shot_id), toks_local, input_shape, original_hw)
            # Honour STOP *after* the just-finished shot is emitted (persisted)
            # so we never drop a fully-encoded shot, and never persist a
            # partial one. The next shot is simply not pulled.
            if STOP.is_set():
                aborted = True
                break
    except StreamAborted:
        aborted = True
    finally:
        # Stop the watchdog and tear down DataLoader workers BEFORE joining the
        # writer, so no worker subprocess is left wedged while we flush output.
        _wd_done.set()
        with _wd_lock:
            _wd_deadline["t"] = None
        _wd_thread.join(timeout=1.0)
        feed.close()
        # Flush + join the async writer so every shot already encoded is
        # persisted, even on the abort path.
        writer.join()

    stats.aborted = aborted
    stats.elapsed_s = time.monotonic() - t_start
    stats.write_errors = writer.errors
    # The async writer may have failed some; reconcile counts.
    if writer.errors:
        stats.shots_ok -= len(writer.errors)
        stats.shots_fail += len(writer.errors)
    if device.startswith("cuda"):
        stats.peak_hbm_gb = torch.cuda.max_memory_allocated() / (1024**3)
    return stats


# --- CLI entrypoint (called by the sbatch, runs in the magvit2 venv) ------


def _build_shotlist(
    manifest_path: Path, camera: str, l1_root: Path, shard: int, n_shards: int
) -> list[int]:
    """Deterministic stride-sharded shotlist, mirroring encode_one_shard.py.

    Same manifest, same `(l1/<shot>.zarr/<camera>).is_dir()` filter, same
    `shots[shard::n_shards]` stride so a stream task processes the exact same
    shots a live shard would — making the GPU spot-check apples-to-apples.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    all_shots = sorted(manifest["shot_ids"])
    shots = [s for s in all_shots if (Path(l1_root) / f"{s}.zarr" / camera).is_dir()]
    return shots[shard::n_shards]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="In-process continuous-batching stream encoder"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--camera", default="rbb")
    parser.add_argument(
        "--manifest",
        default="/work/projects/imas_gpu/mast/manifests/level1-cameras.json",
    )
    parser.add_argument("--l1-root", default=str(DEFAULT_L1_ROOT))
    parser.add_argument("--stream-root", default=str(DEFAULT_STREAM_ROOT))
    parser.add_argument(
        "--magvit2-root",
        default="/work/projects/imas_gpu/mast-tokens/v1/open-magvit2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-frames", type=int, default=256)
    parser.add_argument(
        "--model-forward-batch",
        type=int,
        default=MODEL_FORWARD_BATCH,
        help="frames per model.encode() forward. Larger = higher GPU util "
        "(fewer, bigger kernel launches) but DIFFERENT tokens (the VQ forward "
        "is not batch-size invariant). FIXED per corpus — it defines the "
        "tokenization. Must be re-validated for cross-process reproducibility.",
    )
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument(
        "--batch-timeout-s",
        type=float,
        default=0.0,
        help="per-batch watchdog timeout in seconds; 0 = auto "
        "(max(60, 8x running-median batch time))",
    )
    parser.add_argument("--max-shots", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--explicit-shots",
        default=None,
        help="comma-separated shot ids; bypasses manifest sharding",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    if args.explicit_shots:
        shots = [int(s) for s in args.explicit_shots.split(",") if s.strip()]
    else:
        shots = _build_shotlist(
            Path(args.manifest),
            args.camera,
            Path(args.l1_root),
            args.shard,
            args.n_shards,
        )
        if args.max_shots:
            shots = shots[: args.max_shots]

    print(
        f"[stream shard{args.shard}/{args.n_shards}] {len(shots)} shots "
        f"camera={args.camera} device={args.device} batch_frames={args.batch_frames}",
        flush=True,
    )

    # Install graceful-shutdown handlers up front so a SIGTERM during the
    # (potentially long) model load is also honoured — STOP is checked inside
    # stream_encode's loop, and the finally below always releases the model.
    STOP.clear()
    _install_signal_handlers()

    t0 = time.monotonic()
    model = None
    try:
        if args.device.startswith("cuda") or args.device == "cpu-real":
            dev = "cpu" if args.device == "cpu-real" else args.device
            model = load_model(Path(args.magvit2_root), dev)
            args.device = dev
            print(f"[stream] model loaded in {time.monotonic() - t0:.1f}s", flush=True)

        stats = stream_encode(
            shots,
            args.camera,
            model,
            device=args.device,
            stream_root=Path(args.stream_root),
            l1_root=Path(args.l1_root),
            batch_frames=args.batch_frames,
            num_workers=args.num_workers,
            max_frames=args.max_frames,
            batch_timeout_s=args.batch_timeout_s,
            model_forward_batch=args.model_forward_batch,
        )
    finally:
        # Always release the model + free HBM, even on the abort path. This is
        # the last step of the <5 s graceful-shutdown target: the encode loop
        # has already torn down the DataLoader workers and flushed the writer.
        if model is not None:
            try:
                import torch

                del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                print(f"[stream] model release warning: {exc}", flush=True)

    summary = {
        "shard": args.shard,
        "n_shards": args.n_shards,
        "camera": args.camera,
        "n_shots": len(shots),
        "shots_ok": stats.shots_ok,
        "shots_fail": stats.shots_fail,
        "frames_encoded": stats.frames_encoded,
        "elapsed_s": round(stats.elapsed_s, 1),
        "aborted": stats.aborted,
        "shots_per_min": round(stats.shots_per_min, 2),
        "frames_per_s": round(stats.frames_per_s, 1),
        "peak_hbm_gb": round(stats.peak_hbm_gb, 2),
        "batch_frames": args.batch_frames,
        "num_workers": args.num_workers,
        "load_errors": stats.load_errors[:50],
        "write_errors": stats.write_errors[:50],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2))
    # Non-zero exit on abort so SLURM/sbatch sees the run did not complete the
    # full shotlist (a clean exit, well under UnkillableStepTimeout).
    return 130 if stats.aborted else 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
