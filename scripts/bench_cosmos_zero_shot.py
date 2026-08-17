"""Zero-shot bench of NVIDIA Cosmos-Tokenizer DI16x16 on the rbb 100-shot config.

Mirrors :func:`imas_ambix.bench.tokenizer.benchmark_frame_tokenizer_in_process`
but drives :mod:`imas_ambix.bench.cosmos_worker` instead of the Open-MAGVIT2
``stream_worker``.  Computes per-class stratified rFID identical to the
existing bench so the result lines up apples-to-apples with 1209100
(baseline imagenet) and 1209101 (C5+C6+C7 fine-tuned).

Output JSON has the same schema as the existing bench's per-shot result file,
written to ``imas_ambix/bench/results/v0-rbb-100shot-cosmos-<jobid>.json``.

Run via the same SLURM sbatch shape as ``bench_rbb.sbatch`` — the config is
selected via ``--cosmos`` flag here rather than a YAML.  Single-process,
single-GPU (cuda:0) — same constraint as the Open-MAGVIT2 bench since the
worker itself is one process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="imas_ambix/bench/configs/v0-rbb-100shot.yaml",
        help="Bench config — only camera + shot_ids + max_items_per_shot are read",
    )
    p.add_argument(
        "--cosmos-root",
        default="/work/projects/imas_gpu/mast-tokens/cosmos/v1/DI16x16",
        help="Path to Cosmos DI16x16 .jit files",
    )
    p.add_argument("--output", required=True, help="Result JSON path")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--l1-root",
        default=None,
        help="Override level-1 root; default from data.paths.LEVEL1_DIR",
    )
    args = p.parse_args()

    import yaml

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    shot_ids = list(cfg_raw["shot_ids"])
    camera = cfg_raw.get("camera", "rbb")
    max_items = cfg_raw.get("max_items_per_shot")

    if args.l1_root is None:
        from imas_ambix.data.paths import LEVEL1_DIR

        l1_root = str(LEVEL1_DIR)
    else:
        l1_root = args.l1_root

    repo_root = Path(__file__).resolve().parent.parent
    worker_path = repo_root / "imas_ambix" / "bench" / "cosmos_worker.py"
    venv_py = Path(
        "/work/projects/imas_gpu/mast-tokens/v1/open-magvit2/.venv/bin/python"
    )

    with tempfile.TemporaryDirectory(
        prefix="cosmos-bench-", dir=os.environ.get("TMPDIR", "/tmp")
    ) as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "outputs"
        output_dir.mkdir()
        manifest = {
            "shots": shot_ids,
            "camera": camera,
            "l1_root": l1_root,
            "cosmos_root": args.cosmos_root,
            "max_items_per_shot": max_items,
            "output_dir": str(output_dir),
        }
        manifest_path = tmpdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        report_path = tmpdir / "report.json"

        # 1. Run the Cosmos worker
        t0 = time.perf_counter()
        cmd = [
            str(venv_py),
            str(worker_path),
            "--manifest",
            str(manifest_path),
            "--device",
            args.device,
            "--report",
            str(report_path),
        ]
        print(f"[cosmos-bench] launching worker: {' '.join(cmd)}", flush=True)
        worker_shot_times: dict[int, dict] = {}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                print(line, flush=True)
                try:
                    obj = json.loads(line)
                    if "shot_id" in obj and "encode_seconds" in obj:
                        worker_shot_times[int(obj["shot_id"])] = obj
                except json.JSONDecodeError, KeyError:
                    pass
        finally:
            proc.wait()

        worker_exit = proc.returncode
        if worker_exit != 0:
            print(
                f"[cosmos-bench] worker exited with code {worker_exit}", file=sys.stderr
            )

        # 2. Collect per-shot artefacts + compute metrics
        import numpy as np

        from imas_ambix.eval.metrics import (
            centroid_mse,
            chord_nrmse,
        )
        from imas_ambix.eval.metrics import (
            lpips as _lpips,
        )
        from imas_ambix.eval.metrics import (
            psnr as _psnr,
        )
        from imas_ambix.eval.metrics import (
            rfid as _rfid,
        )
        from imas_ambix.eval.metrics import (
            rfid_stratified as _rfid_stratified,
        )

        rfid_src: list[np.ndarray] = []
        rfid_dec: list[np.ndarray] = []
        per_shot_results = []
        rng = np.random.default_rng(seed=0)
        RFID_PER_SHOT = 8
        RFID_RESIZE = 256

        def _resize_for_rfid(frames_u8: np.ndarray) -> np.ndarray:
            from PIL import Image

            out = np.empty(
                (frames_u8.shape[0], RFID_RESIZE, RFID_RESIZE, 3), dtype=np.uint8
            )
            for i in range(frames_u8.shape[0]):
                img = Image.fromarray(frames_u8[i]).resize(
                    (RFID_RESIZE, RFID_RESIZE), Image.BILINEAR
                )
                out[i] = np.asarray(img, dtype=np.uint8)
            return out

        for shot_id in shot_ids:
            tok_p = output_dir / f"{shot_id}-tokens.npy"
            dec_p = output_dir / f"{shot_id}-decoded.npy"
            src_p = output_dir / f"{shot_id}-src.npy"
            if not (tok_p.exists() and dec_p.exists() and src_p.exists()):
                per_shot_results.append(
                    {
                        "shot_id": shot_id,
                        "error": "missing artifacts",
                    }
                )
                continue
            src = np.load(src_p)
            dec = np.load(dec_p)
            n = min(src.shape[0], dec.shape[0])
            src = src[:n]
            dec = dec[:n]
            per_shot = {
                "shot_id": shot_id,
                "n_items": int(n),
                "psnr": float(_psnr(src, dec)),
                "lpips": float(_lpips(src, dec)),
                "centroid_mse": float(centroid_mse(src, dec)),
                "chord_nrmse": float(chord_nrmse(src, dec)),
                "encode_seconds": worker_shot_times.get(shot_id, {}).get(
                    "encode_seconds"
                ),
                "decode_seconds": worker_shot_times.get(shot_id, {}).get(
                    "decode_seconds"
                ),
            }
            per_shot_results.append(per_shot)

            # Sample frames for the corpus rFID pool (same logic as benchmark_frame_tokenizer_in_process)
            k = min(RFID_PER_SHOT, n)
            idx = rng.choice(n, size=k, replace=False)
            try:
                rfid_src.append(_resize_for_rfid(src[idx]))
                rfid_dec.append(_resize_for_rfid(dec[idx]))
            except Exception:
                pass

        # 3. Corpus rFID + stratified
        aggregate: dict[str, float | int] = {
            "n_shots_ok": sum(1 for r in per_shot_results if "error" not in r),
            "n_shots_err": sum(1 for r in per_shot_results if "error" in r),
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "worker_exit_code": worker_exit,
            "tokenizer": "frames_cosmos_di16x16_v1",
        }
        if rfid_src:
            src_all = np.concatenate(rfid_src, axis=0)
            dec_all = np.concatenate(rfid_dec, axis=0)
            try:
                aggregate["rfid_overall"] = float(_rfid(src_all, dec_all))
                aggregate["rfid_n_frames"] = int(src_all.shape[0])
                strat = _rfid_stratified(src_all, dec_all)
                aggregate.update(strat)
            except Exception as exc:  # noqa: BLE001
                aggregate["rfid_overall"] = float("nan")
                aggregate["rfid_error"] = str(exc)

            # Per-class aggregate averages from per-shot
            from statistics import mean

            ok_shots = [r for r in per_shot_results if "error" not in r]
            if ok_shots:
                for m in ("psnr", "lpips", "centroid_mse", "chord_nrmse"):
                    aggregate[f"mean_{m}"] = float(mean(r[m] for r in ok_shots))

        # 4. Write result JSON
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result = [
            {
                "config": {
                    "name": "v0-rbb-100shot-cosmos",
                    "tokenizer_kind": "frame",
                    "tokenizer": "cosmos_di16x16",
                    "device": args.device,
                    "max_items_per_shot": max_items,
                },
                "per_shot": per_shot_results,
                "aggregate": aggregate,
                "elapsed_s": aggregate["elapsed_s"],
            }
        ]
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"[cosmos-bench] result saved: {args.output}", flush=True)
        print(
            f"[cosmos-bench] aggregate:\n{json.dumps(aggregate, indent=2)}", flush=True
        )
        return worker_exit


if __name__ == "__main__":
    sys.exit(main())
