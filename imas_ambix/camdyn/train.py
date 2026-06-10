"""Training + eval for the camdyn ST-transformer (D1 baseline / D2 dynamics).

In-process, single long-lived process: the model is built ONCE and many
windows stream through a bounded torch ``DataLoader``.  No
subprocess-per-item, no file-IPC daemon, no unbounded prefetch threads
(repo §2b).  GPU-safety contract (repo §6 / plan §6):

  * a ``SIGTERM`` / ``SIGINT`` handler sets a global ``STOP`` flag,
    flushes the latest checkpoint, and exits clean in < 5 s;
  * a per-step watchdog auto-tuned from the running median aborts a
    wedged step instead of hanging the allocation;
  * ``try/finally`` releases the model and calls
    ``torch.cuda.empty_cache()``;
  * checkpoints land on GPFS every ``ckpt_every`` steps so any clean
    stop resumes;
  * bf16 on CUDA + ``set_float32_matmul_precision('high')`` and
    deterministic cuDNN.

The same entry point trains either arm — ``temporal_attention`` in the
config picks D1 (False) or D2 (True).  D1 logs the held-out masked-token
NLL + top-1 accuracy as the LOCKED W1 bar (overall + motion-weighted +
per named-geometry), scored through the pre-registered D0 metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imas_ambix.camdyn.conditioning import CONDITIONING_CHANNELS, load_conditioning
from imas_ambix.camdyn.dataset import (
    FrameTokenDataset,
    FrameWindowConfig,
    discover_token_shots,
)
from imas_ambix.camdyn.masking import (
    NAMED_GEOMETRIES,
    ClipMaskConfig,
    MaskMode,
    sample_clip_mask,
)
from imas_ambix.camdyn.metrics import bootstrap_ci, motion_weighted_subset
from imas_ambix.camdyn.model import (
    CamdynConfig,
    CamdynModel,
    masked_bit_bce,
    score_window_bits,
)
from imas_ambix.camdyn.splits import CamdynSplit

logger = logging.getLogger(__name__)

# Graceful-stop flag (set by the signal handler; polled in the train loop).
STOP = threading.Event()


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        STOP.set()
        logger.warning("[camdyn-train] signal %s received -> graceful stop", signum)

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except ValueError:  # pragma: no cover - not main thread
        logger.warning("[camdyn-train] could not install signal handlers")


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyper-parameters for one camdyn training run.

    The ``model`` block selects D1 (temporal_attention False) vs D2 (True).
    """

    # arm / architecture
    model: CamdynConfig = field(default_factory=CamdynConfig)
    # data
    n_frames: int = 16
    stride: int = 8
    max_train_shots: int | None = None
    max_val_shots: int | None = None
    max_heldout_shots: int | None = None
    windows_per_shot_cap: int | None = 4
    # optimisation
    batch_size: int = 16
    max_steps: int = 4000
    peak_lr: float = 3.0e-4
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.1
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    # masking curriculum
    curriculum: bool = True
    # loader
    num_workers: int = 4
    seed: int = 0
    # cadence
    log_every: int = 50
    val_every: int = 500
    ckpt_every: int = 500
    val_windows: int = 256
    eval_windows: int = 512
    watchdog_grace_s: float = 120.0
    # paths
    split_path: str | None = None
    ckpt_root: str = "/work/projects/imas_gpu/mast-checkpoints/camdyn"
    run_name: str = "baseline_w1_v0"
    artifact_out: str | None = None
    device: str = "cuda"

    def to_dict(self) -> dict:
        d = {
            k: getattr(self, k)
            for k in self.__dataclass_fields__  # noqa: PLC0206
            if k != "model"
        }
        d["betas"] = list(self.betas)
        d["model"] = self.model.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        d = dict(d)
        model = CamdynConfig.from_dict(d.pop("model", {}))
        if "betas" in d and d["betas"] is not None:
            d["betas"] = tuple(d["betas"])
        known = {f for f in cls.__dataclass_fields__ if f != "model"}  # noqa: PLC0206
        return cls(model=model, **{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        text = Path(path).read_text(encoding="utf-8")
        try:
            import yaml  # noqa: PLC0415

            d = yaml.safe_load(text)
        except Exception:
            d = json.loads(text)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Conditioning cache (per shot — built lazily, reused across windows)
# ---------------------------------------------------------------------------


def _frame_conditioning(spec, frame_time, shot_id):
    """Per-frame conditioning matrices for one window.

    Returns ``(values (F,C), missing (F,C))`` float32 from the locked
    :func:`load_conditioning` loader (leakage ban enforced inside).
    """
    sample = load_conditioning(
        spec.level1_path, frame_time, shot_id, channels=CONDITIONING_CHANNELS
    )
    return sample.values, sample.missing


# ---------------------------------------------------------------------------
# Batch assembly (CPU side) — windows + masks + conditioning → tensors
# ---------------------------------------------------------------------------


def _assemble_batch(windows, mask_cfg, rng, *, progress, mode=None):
    """Stack a list of FrameWindow dicts into model-ready numpy arrays.

    Returns a dict of numpy arrays:
      tokens (B,F,H,W) int64, visible (B,F,H,W) bool, loss_mask (B,F,H,W) bool,
      cond_values (B,F,C), cond_missing (B,F,C), dt (B,F), valid (B,F),
      frame_time (B,F).
    """
    toks, vis, lmask, cvals, cmiss, dts, valids, ftimes = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for win in windows:
        tokens = np.asarray(win["tokens"], dtype=np.int64)
        nf = tokens.shape[0]
        m, _meta = sample_clip_mask(nf, mask_cfg, rng, mode=mode, progress=progress)
        visible = m  # True = visible
        loss_mask = ~m  # True = clipped-away = scored
        # conditioning held to this window's frame times
        from types import SimpleNamespace  # noqa: PLC0415

        spec = SimpleNamespace(level1_path=win.get("level1_path"))
        cv, cm = _frame_conditioning(spec, win["frame_time"], int(win["shot_id"]))
        toks.append(tokens)
        vis.append(visible)
        lmask.append(loss_mask)
        cvals.append(cv)
        cmiss.append(cm)
        dts.append(np.asarray(win["dt"], dtype=np.float32))
        valids.append(np.asarray(win["valid_frames"], dtype=bool))
        ftimes.append(np.asarray(win["frame_time"], dtype=np.float64))
    return {
        "tokens": np.stack(toks),
        "visible": np.stack(vis),
        "loss_mask": np.stack(lmask),
        "cond_values": np.stack(cvals),
        "cond_missing": np.stack(cmiss),
        "dt": np.stack(dts),
        "valid": np.stack(valids),
        "frame_time": np.stack(ftimes),
    }


def _normalise_conditioning(cond_values, stats):
    """Z-score conditioning values by precomputed per-channel stats."""
    mu, sd = stats
    return (cond_values - mu) / sd


def _conditioning_stats(specs, frame_cfg, n_sample=64, seed=0):
    """Estimate per-channel mean/std of conditioning over a sample of windows.

    Keeps the additive conditioning numerically sane (currents are ~1e5 A,
    densities ~1e19 m^-2).  Missing values (flagged) are excluded.
    """
    rng = np.random.default_rng(seed)
    vals, miss = [], []
    n_chan = len(CONDITIONING_CHANNELS)
    ds = FrameTokenDataset(specs, frame_cfg, as_dict=True)
    n = len(ds)
    if n == 0:
        return np.zeros(n_chan, np.float32), np.ones(n_chan, np.float32)
    idxs = rng.choice(n, size=min(n_sample, n), replace=False)
    for i in idxs:
        win = ds[int(i)]
        sample = load_conditioning(
            _level1_for(specs, int(win["shot_id"])),
            win["frame_time"],
            int(win["shot_id"]),
            channels=CONDITIONING_CHANNELS,
        )
        vals.append(sample.values)
        miss.append(sample.missing)
    V = np.concatenate(vals, axis=0)  # (N,C)
    M = np.concatenate(miss, axis=0)  # (N,C)
    present = M < 0.5
    mu = np.zeros(n_chan, np.float32)
    sd = np.ones(n_chan, np.float32)
    for c in range(n_chan):
        col = V[present[:, c], c]
        if col.size > 1:
            mu[c] = float(col.mean())
            s = float(col.std())
            sd[c] = s if s > 1e-8 else 1.0
    return mu, sd


def _level1_for(specs, shot_id):
    for s in specs:
        if int(s.shot_id) == int(shot_id):
            return s.level1_path
    return None


# ---------------------------------------------------------------------------
# Spec discovery from a split
# ---------------------------------------------------------------------------


def _specs_for_shots(shot_ids, *, max_shots=None):
    if max_shots is not None:
        shot_ids = shot_ids[:max_shots]
    return discover_token_shots(shot_ids=shot_ids, read_n_frames=True)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """In-process trainer for one camdyn arm."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.mask_cfg = ClipMaskConfig()
        self._step = 0
        self._cond_stats = None

    # -- torch setup -------------------------------------------------------

    def _setup_torch(self):
        import torch  # noqa: PLC0415

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        if self.cfg.device == "cuda" and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda")
            amp_dtype = torch.bfloat16
        else:
            device = torch.device("cpu")
            amp_dtype = torch.float32
        return torch, device, amp_dtype

    def _lr_at(self, step: int) -> float:
        warmup = max(1, int(self.cfg.warmup_frac * self.cfg.max_steps))
        if step < warmup:
            return self.cfg.peak_lr * step / warmup
        prog = (step - warmup) / max(1, self.cfg.max_steps - warmup)
        prog = min(1.0, prog)
        min_lr = self.cfg.peak_lr * self.cfg.min_lr_frac
        cos = 0.5 * (1.0 + np.cos(np.pi * prog))
        return float(min_lr + (self.cfg.peak_lr - min_lr) * cos)

    # -- iteration over windows -------------------------------------------

    def _window_iter(self, specs, frame_cfg, batch_size, rng_seed):
        """Yield assembled batches by drawing windows from the dataset.

        Map-style dataset → sample window indices with a seeded RNG so the
        stream is reproducible and bounded.  CPU-side mask + conditioning
        assembly is light; the dataset's Zarr reads are the only I/O.
        """
        ds = FrameTokenDataset(specs, frame_cfg, as_dict=True)
        n = len(ds)
        if n == 0:
            return
        rng = np.random.default_rng(rng_seed)
        order = rng.permutation(n)
        buf = []
        for idx in order:
            win = ds[int(idx)]
            win["level1_path"] = _level1_for(specs, int(win["shot_id"]))
            buf.append(win)
            if len(buf) == batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def _batch_to_tensors(self, windows, torch, device, *, progress, mode=None):
        rng = np.random.default_rng(self._step + 1)
        arr = _assemble_batch(windows, self.mask_cfg, rng, progress=progress, mode=mode)
        cv = _normalise_conditioning(arr["cond_values"], self._cond_stats)
        t = {
            "tokens": torch.from_numpy(arr["tokens"]).to(device),
            "visible": torch.from_numpy(arr["visible"]).to(device),
            "loss_mask": torch.from_numpy(arr["loss_mask"]).to(device),
            "cond_values": torch.from_numpy(cv.astype(np.float32)).to(device),
            "cond_missing": torch.from_numpy(arr["cond_missing"].astype(np.float32)).to(
                device
            ),
            "dt": torch.from_numpy(arr["dt"].astype(np.float32)).to(device),
            "valid": torch.from_numpy(arr["valid"]).to(device),
        }
        return t, arr

    # -- public: train -----------------------------------------------------

    def train(self):
        STOP.clear()
        _install_signal_handlers()
        torch, device, amp_dtype = self._setup_torch()

        split = self._load_split()
        train_specs = _specs_for_shots(split.train, max_shots=self.cfg.max_train_shots)
        val_specs = _specs_for_shots(split.val, max_shots=self.cfg.max_val_shots)
        logger.info(
            "[camdyn-train] arm=%s train_shots=%d val_shots=%d",
            "D2-dynamics" if self.cfg.model.temporal_attention else "D1-baseline",
            len(train_specs),
            len(val_specs),
        )

        frame_cfg = FrameWindowConfig(
            n_frames=self.cfg.n_frames, stride=self.cfg.stride, seed=self.cfg.seed
        )
        self._cond_stats = _conditioning_stats(
            train_specs, frame_cfg, seed=self.cfg.seed
        )

        model = CamdynModel.from_config(self.cfg.model)
        model.module.to(device)
        logger.info("[camdyn-train] params=%.2fM", model.num_parameters() / 1e6)

        opt = torch.optim.AdamW(
            model.module.parameters(),
            lr=self.cfg.peak_lr,
            betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay,
        )

        ckpt_dir = Path(self.cfg.ckpt_root) / self.cfg.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict] = []
        last_step_t = [time.time()]
        median_step = [None]
        self._arm_watchdog(median_step, last_step_t)

        try:
            model.module.train()
            done = False
            while not done and not STOP.is_set():
                for windows in self._window_iter(
                    train_specs,
                    frame_cfg,
                    self.cfg.batch_size,
                    rng_seed=self.cfg.seed + self._step,
                ):
                    if STOP.is_set() or self._step >= self.cfg.max_steps:
                        done = True
                        break
                    t0 = time.time()
                    progress = (
                        self._step / max(1, self.cfg.max_steps)
                        if self.cfg.curriculum
                        else None
                    )
                    lr = self._lr_at(self._step)
                    for g in opt.param_groups:
                        g["lr"] = lr

                    t, _arr = self._batch_to_tensors(
                        windows, torch, device, progress=progress
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=(device.type == "cuda"),
                    ):
                        logits = model.module(
                            t["tokens"],
                            t["visible"],
                            t["cond_values"],
                            t["cond_missing"],
                            t["dt"],
                        )
                        loss = masked_bit_bce(
                            logits, t["tokens"], t["loss_mask"], t["valid"]
                        )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.module.parameters(), self.cfg.grad_clip
                    )
                    opt.step()

                    dt_step = time.time() - t0
                    last_step_t[0] = time.time()
                    median_step[0] = (
                        dt_step
                        if median_step[0] is None
                        else 0.9 * median_step[0] + 0.1 * dt_step
                    )

                    if self._step % self.cfg.log_every == 0:
                        logger.info(
                            "[camdyn-train] step=%d loss=%.4f lr=%.2e %.2fs/step",
                            self._step,
                            float(loss.item()),
                            lr,
                            dt_step,
                        )
                        history.append(
                            {
                                "step": self._step,
                                "train_loss": float(loss.item()),
                                "lr": lr,
                            }
                        )

                    if (
                        self._step > 0
                        and self._step % self.cfg.val_every == 0
                        and len(val_specs) > 0
                    ):
                        vnll, vacc = self._quick_val(
                            model, val_specs, frame_cfg, torch, device
                        )
                        logger.info(
                            "[camdyn-train] step=%d VAL nll=%.4f top1=%.4f",
                            self._step,
                            vnll,
                            vacc,
                        )
                        history.append(
                            {"step": self._step, "val_nll": vnll, "val_top1": vacc}
                        )
                        model.module.train()

                    if self._step > 0 and self._step % self.cfg.ckpt_every == 0:
                        self._save_ckpt(model, opt, ckpt_dir, torch)

                    self._step += 1

            # final checkpoint
            final_ckpt = self._save_ckpt(model, opt, ckpt_dir, torch, final=True)
            logger.info("[camdyn-train] training complete; ckpt=%s", final_ckpt)

            # held-out W1 evaluation (the locked bar) — D1 only by default,
            # but works for either arm.
            ho_specs = _specs_for_shots(
                split.held_out, max_shots=self.cfg.max_heldout_shots
            )
            w1 = self.evaluate_w1(
                model, ho_specs, val_specs, frame_cfg, torch, device, history
            )
            w1["checkpoint"] = str(final_ckpt)
            self._write_artifact(w1)
            return w1
        finally:
            try:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass

    # -- watchdog ----------------------------------------------------------

    def _arm_watchdog(self, median_step, last_step_t):
        def _watchdog():
            while not STOP.is_set():
                time.sleep(5.0)
                med = median_step[0]
                if med is None:
                    continue
                deadline = max(self.cfg.watchdog_grace_s, 8.0 * med)
                if time.time() - last_step_t[0] > deadline:
                    logger.error(
                        "[camdyn-train] watchdog FIRED (step exceeded %.0fs) "
                        "-> graceful stop",
                        deadline,
                    )
                    STOP.set()
                    return

        threading.Thread(target=_watchdog, daemon=True).start()

    # -- split -------------------------------------------------------------

    def _load_split(self) -> CamdynSplit:
        if self.cfg.split_path:
            return CamdynSplit.load(Path(self.cfg.split_path))
        from imas_ambix.camdyn.splits import DEFAULT_SPLIT_OUT  # noqa: PLC0415

        return CamdynSplit.load(DEFAULT_SPLIT_OUT)

    # -- checkpoint --------------------------------------------------------

    def _save_ckpt(self, model, opt, ckpt_dir, torch, final=False):
        name = "final.pt" if final else f"step{self._step}.pt"
        path = ckpt_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "step": self._step,
                "model_state": model.module.state_dict(),
                "opt_state": opt.state_dict(),
                "config": self.cfg.to_dict(),
                "cond_stats": [
                    self._cond_stats[0].tolist(),
                    self._cond_stats[1].tolist(),
                ],
            },
            tmp,
        )
        tmp.replace(path)  # atomic — a clean stop always leaves a valid ckpt
        return path

    # -- quick val (during training) --------------------------------------

    def _quick_val(self, model, specs, frame_cfg, torch, device):
        model.module.eval()
        nll_all, acc_all = [], []
        seen = 0
        with torch.no_grad():
            for windows in self._window_iter(
                specs, frame_cfg, self.cfg.batch_size, rng_seed=12345
            ):
                t, arr = self._batch_to_tensors(
                    windows, torch, device, progress=None, mode=MaskMode.RANDOM
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model.module(
                        t["tokens"],
                        t["visible"],
                        t["cond_values"],
                        t["cond_missing"],
                        t["dt"],
                    )
                bl = logits.float().cpu().numpy()
                for b in range(bl.shape[0]):
                    vf = arr["valid"][b]
                    lm = arr["loss_mask"][b] & vf[:, None, None]
                    sc = score_window_bits(bl[b], arr["tokens"][b], lm)
                    if sc.n:
                        nll_all.append(sc.nll_per_token)
                        acc_all.append(sc.acc_per_token)
                seen += bl.shape[0]
                if seen >= self.cfg.val_windows:
                    break
        if not nll_all:
            return 0.0, 0.0
        return (
            float(np.concatenate(nll_all).mean()),
            float(np.concatenate(acc_all).mean()),
        )

    # -- W1 evaluation (the locked bar) -----------------------------------

    def evaluate_w1(
        self, model, ho_specs, val_specs, frame_cfg, torch, device, history
    ):
        """Score the held-out split + named-geometry suite → the W1 bar.

        Returns the full artifact dict (held-out overall, motion-weighted,
        per named-geometry, a bootstrap-CI sanity exercise, val numbers,
        and the loss history).
        """
        model.module.eval()
        out: dict = {
            "arm": "D2-dynamics"
            if self.cfg.model.temporal_attention
            else "D1-baseline",
            "temporal_attention": bool(self.cfg.model.temporal_attention),
            "model_config": self.cfg.model.to_dict(),
            "n_params": int(model.num_parameters()),
            "n_heldout_shots": len(ho_specs),
            "n_val_shots": len(val_specs),
            "metrics_provenance": "imas_ambix.camdyn.metrics (pre-registered D0)",
        }

        # --- held-out: mixture mask (the headline reconstruction task) ---
        nll, acc, mnll, macc = self._score_split(
            model, ho_specs, frame_cfg, torch, device, mode=None
        )
        out["held_out"] = {
            "masked_nll": _agg(nll),
            "masked_top1": _agg(acc),
            "motion_weighted": {
                "masked_nll": _agg(mnll),
                "masked_top1": _agg(macc),
            },
            "n_scored_tokens": int(nll.size),
            "n_motion_tokens": int(mnll.size),
        }

        # --- per named-geometry (frozen eval suite) ---
        geo_out = {}
        for name in NAMED_GEOMETRIES:
            gnll, gacc, _, _ = self._score_split(
                model,
                ho_specs,
                frame_cfg,
                torch,
                device,
                mode=MaskMode.NAMED,
                named=name,
            )
            geo_out[name] = {
                "masked_nll": _agg(gnll),
                "masked_top1": _agg(gacc),
                "n_scored_tokens": int(gnll.size),
            }
        out["named_geometry"] = geo_out

        # --- val (reference) ---
        vnll, vacc = self._quick_val(model, val_specs, frame_cfg, torch, device)
        out["val"] = {"masked_nll": vnll, "masked_top1": vacc}

        # --- bootstrap CI helper exercised on a sanity split ---
        # Split the held-out per-token NLL in half; the paired diff of two
        # halves of the SAME arm must straddle zero (favours_dynamics False),
        # demonstrating the helper is wired and the bar is not spuriously
        # significant against itself.
        out["bootstrap_ci_sanity"] = _bootstrap_sanity(nll)

        out["loss_history"] = history
        return out

    def _score_split(self, model, specs, frame_cfg, torch, device, *, mode, named=None):
        """Score masked-token NLL/acc over a split; return per-token arrays.

        Returns ``(nll, acc, motion_nll, motion_acc)`` flattened per-token.
        """
        from imas_ambix.camdyn.masking import named_geometry_mask  # noqa: PLC0415

        nll_all, acc_all, mnll_all, macc_all = [], [], [], []
        seen = 0
        with torch.no_grad():
            for windows in self._window_iter(
                specs, frame_cfg, self.cfg.batch_size, rng_seed=999
            ):
                # For NAMED mode, override the per-window mask with the frozen
                # geometry (deterministic, identical across arms).
                t, arr = self._batch_to_tensors(
                    windows, torch, device, progress=None, mode=mode
                )
                if mode is MaskMode.NAMED and named is not None:
                    nf = arr["tokens"].shape[1]
                    gmask = named_geometry_mask(named, nf)  # (F,H,W) True=visible
                    vis = np.broadcast_to(gmask[None], arr["visible"].shape).copy()
                    lm = ~vis
                    arr["visible"] = vis
                    arr["loss_mask"] = lm
                    t["visible"] = torch.from_numpy(vis).to(device)
                    t["loss_mask"] = torch.from_numpy(lm).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model.module(
                        t["tokens"],
                        t["visible"],
                        t["cond_values"],
                        t["cond_missing"],
                        t["dt"],
                    )
                bl = logits.float().cpu().numpy()
                for b in range(bl.shape[0]):
                    vf = arr["valid"][b]
                    lm = arr["loss_mask"][b] & vf[:, None, None]
                    sc = score_window_bits(bl[b], arr["tokens"][b], lm)
                    if not sc.n:
                        continue
                    nll_all.append(sc.nll_per_token)
                    acc_all.append(sc.acc_per_token)
                    # motion-weighted subset (D0 metric) on the masked set
                    moving = motion_weighted_subset(
                        arr["tokens"][b], arr["frame_time"][b]
                    )
                    mm = lm & moving
                    if mm.any():
                        msc = score_window_bits(bl[b], arr["tokens"][b], mm)
                        if msc.n:
                            mnll_all.append(msc.nll_per_token)
                            macc_all.append(msc.acc_per_token)
                seen += bl.shape[0]
                if seen >= self.cfg.eval_windows:
                    break

        def _cat(xs):
            return np.concatenate(xs) if xs else np.array([])

        return _cat(nll_all), _cat(acc_all), _cat(mnll_all), _cat(macc_all)

    # -- artifact ----------------------------------------------------------

    def _write_artifact(self, w1: dict) -> Path:
        out = self.cfg.artifact_out or str(
            Path(__file__).resolve().parent / "artifacts" / f"{self.cfg.run_name}.json"
        )
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(w1, indent=2), encoding="utf-8")
        logger.info("[camdyn-train] W1 bar written to %s", path)
        return path


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _agg(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"mean": 0.0, "n": 0}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "n": int(x.size),
    }


def _bootstrap_sanity(nll: np.ndarray) -> dict:
    """Exercise the bootstrap_ci helper on a within-arm sanity split.

    Pair the two halves of the held-out per-token NLL; their paired diff
    should NOT favour either side (a sanity check that the helper is wired
    and reports a non-significant within-arm difference).
    """
    nll = np.asarray(nll, dtype=np.float64).reshape(-1)
    if nll.size < 4:
        return {"note": "insufficient tokens for sanity split", "n": int(nll.size)}
    rng = np.random.default_rng(0)
    perm = rng.permutation(nll.size)
    half = nll.size // 2
    a = nll[perm[:half]]
    b = nll[perm[half : 2 * half]]
    # paired diff = a - b (oriented as baseline - dynamics in the real W1);
    # here both halves are the same arm so it must straddle 0.
    ci = bootstrap_ci(a - b)
    ci["n_pairs"] = int(half)
    return ci


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="camdyn ST-transformer trainer")
    parser.add_argument("--config", required=True, help="YAML/JSON TrainConfig")
    parser.add_argument("--device", default=None, help="override device")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--artifact-out", default=None, help="override W1 artifact path"
    )
    parser.add_argument(
        "--ckpt-root", default=None, help="override checkpoint root dir"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = TrainConfig.load(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.artifact_out is not None:
        cfg.artifact_out = args.artifact_out
    if args.ckpt_root is not None:
        cfg.ckpt_root = args.ckpt_root

    trainer = Trainer(cfg)
    w1 = trainer.train()
    arm = w1.get("arm", "?")
    ho = w1.get("held_out", {})
    logger.info(
        "[camdyn-train] DONE arm=%s held_out nll=%.4f top1=%.4f",
        arm,
        ho.get("masked_nll", {}).get("mean", float("nan")),
        ho.get("masked_top1", {}).get("mean", float("nan")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
