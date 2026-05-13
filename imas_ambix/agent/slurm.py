"""SLURM script generation helpers for ``ambix agent``."""

from __future__ import annotations

import shlex
import subprocess
from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from imas_ambix.agent.profile import ModelProfile, SiteConfig

_MODEL_DIR_TOKEN = "__AMBIX_MODEL_DIR__"
_PORT_TOKEN = "__AMBIX_PORT__"

# Python one-liner that calls posix_fadvise(FADV_DONTNEED) on all
# safetensor files in a directory, advising the kernel to drop their
# file-backed pages from the page cache.  Used as a post-load safety
# net to release any residual mmap'd pages.
_FADVISE_DROP_CODE = (
    "import os, sys\n"
    "d = sys.argv[1]\n"
    "for f in sorted(os.listdir(d)):\n"
    "    if f.endswith('.safetensors'):\n"
    "        p = os.path.join(d, f)\n"
    "        fd = os.open(p, os.O_RDONLY)\n"
    "        os.posix_fadvise(fd, 0, os.fstat(fd).st_size, os.POSIX_FADV_DONTNEED)\n"
    "        os.close(fd)\n"
)


def _sbatch_headers(
    *,
    job_name: str,
    partition: str,
    account: str,
    reservation: str | None,
    gpus: int,
    cpus: int,
    memory: str,
    time_limit: str,
    output_name: str,
) -> list[str]:
    headers = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
    ]
    if reservation:
        headers.append(f"#SBATCH --reservation={reservation}")
    headers.append(f"#SBATCH --account={account}")
    if gpus > 0:
        headers.append(f"#SBATCH --gres=gpu:{gpus}")
    headers.extend(
        [
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={memory}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={output_name}",
        ]
    )
    return headers


def _append_option(
    args: list[str],
    flag: str,
    value: str | int | float | Path | None,
) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])


def _append_flag(args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def _render_shell_command(args: list[str]) -> str:
    command = shlex.join(args)
    return (
        command.replace(_MODEL_DIR_TOKEN, '"$MODEL_DIR"').replace(
            _PORT_TOKEN, '"$PORT"'
        )
    )


def _build_sglang_args(profile: ModelProfile, site: SiteConfig) -> list[str]:
    """Build common SGLang launch_server arguments."""
    engine = profile.engine
    max_tokens = engine.max_total_tokens or profile.model.max_context
    python = str(site.python_path)
    args = [
        python,
        "-m",
        "sglang.launch_server",
        "--model",
        _MODEL_DIR_TOKEN,
        "--served-model-name",
        profile.model.served_name,
        "--tensor-parallel-size",
        str(engine.tensor_parallel),
        "--mem-fraction-static",
        str(engine.mem_fraction_static),
        "--chunked-prefill-size",
        str(engine.chunked_prefill_size),
        "--max-total-tokens",
        str(max_tokens),
        "--attention-backend",
        engine.attention_backend,
        "--host",
        "0.0.0.0",
        "--port",
        _PORT_TOKEN,
    ]

    _append_flag(args, "--trust-remote-code", engine.trust_remote_code)
    _append_flag(args, "--enable-mixed-chunk", engine.enable_mixed_chunk)
    _append_flag(args, "--enable-p2p-check", engine.enable_p2p_check)
    _append_flag(args, "--disable-cuda-graph", engine.disable_cuda_graph)
    _append_flag(
        args,
        "--disable-custom-all-reduce",
        engine.disable_custom_all_reduce,
    )
    _append_flag(
        args,
        "--enable-auto-tool-choice",
        engine.enable_auto_tool_choice,
    )
    _append_option(args, "--moe-runner-backend", engine.moe_runner_backend)
    _append_option(args, "--cuda-graph-max-bs", engine.cuda_graph_max_bs)
    _append_flag(
        args,
        "--weight-loader-disable-mmap",
        engine.weight_loader_disable_mmap,
    )

    if engine.parsers.tool_call:
        _append_option(args, "--tool-call-parser", engine.parsers.tool_call)
    if engine.parsers.reasoning:
        _append_option(args, "--reasoning-parser", engine.parsers.reasoning)

    return args


def _build_serve_command(profile: ModelProfile, site: SiteConfig) -> str:
    engine = profile.engine

    if engine.type == "sglang":
        args = _build_sglang_args(profile, site)
        return _render_shell_command(args)

    if engine.type == "vllm":
        python = str(site.python_path)
        max_tokens = engine.max_total_tokens or profile.model.max_context
        args = [
            python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            _MODEL_DIR_TOKEN,
            "--served-model-name",
            profile.model.served_name,
            "--tensor-parallel-size",
            str(engine.tensor_parallel),
            "--max-model-len",
            str(max_tokens),
            "--host",
            "0.0.0.0",
            "--port",
            _PORT_TOKEN,
        ]
        _append_flag(args, "--trust-remote-code", engine.trust_remote_code)
        _append_flag(
            args,
            "--enable-auto-tool-choice",
            engine.enable_auto_tool_choice,
        )
        if engine.parsers.tool_call:
            _append_option(
                args, "--tool-call-parser", engine.parsers.tool_call
            )
        if engine.parsers.reasoning:
            _append_option(
                args, "--reasoning-parser", engine.parsers.reasoning
            )
        return _render_shell_command(args)

    if engine.type != "ktransformers":
        msg = f"Unsupported engine type: {engine.type}"
        raise ValueError(msg)

    # KTransformers: SGLang + KT flags
    args = _build_sglang_args(profile, site)

    kt = engine.ktransformers
    if kt is None:
        msg = "KTransformers engine requires engine.ktransformers settings"
        raise ValueError(msg)

    args.extend(
        [
            "--kt-weight-path",
            _MODEL_DIR_TOKEN,
            "--kt-method",
            kt.method,
            "--kt-cpuinfer",
            str(kt.cpuinfer),
            "--kt-threadpool-count",
            str(kt.threadpool_count),
            "--kt-num-gpu-experts",
            str(kt.gpu_experts),
        ]
    )
    _append_flag(
        args,
        "--disable-shared-experts-fusion",
        kt.disable_shared_experts_fusion,
    )
    return _render_shell_command(args)


def generate_serve_script(
    profile: ModelProfile, site: SiteConfig, port: int = 8000
) -> str:
    """Generate a SLURM batch script for serving a model profile."""
    model_dir = site.model_dir(profile)
    headers = _sbatch_headers(
        job_name=f"ambix-serve-{profile.slug}",
        partition=site.partition,
        account=site.account,
        reservation=site.reservation,
        gpus=profile.slurm.gpus,
        cpus=profile.slurm.cpus,
        memory=profile.slurm.memory,
        time_limit=profile.slurm.time_serve,
        output_name="ambix-serve-%j.log",
    )
    launch_command = _build_serve_command(profile, site)
    venv_bin = str(site.python_path.parent)
    is_kt = profile.engine.type == "ktransformers"

    # KTransformers needs extra env vars and fadvise evictor
    kt_env_block = ""
    if is_kt:
        evictor_python = shlex.quote(str(site.python_path))
        fadvise_cmd = (
            f"{evictor_python} -c {shlex.quote(_FADVISE_DROP_CODE)}"
            ' "$MODEL_DIR"'
        )
        kt_env_block = dedent(
            f"""
            # Force glibc to return freed memory to the OS immediately,
            # reducing heap fragmentation under the tight cgroup limit.
            export MALLOC_TRIM_THRESHOLD_=0
            # Force large allocations (>32KB) through mmap instead of heap;
            # mmap pages are returned to the OS on free, avoiding fragmentation.
            export MALLOC_MMAP_THRESHOLD_=32768

            # sglang-kernel ships pre-compiled CUDA 13 binaries; expose the
            # nvidia-cu13 runtime libs so they can be loaded alongside the
            # CUDA 12.6 PyTorch stack (the driver supports both).
            _CU13_LIB={shlex.quote(str(site.venv_path / "lib/python3.12/site-packages/nvidia/cu13/lib"))}
            if [[ -d "$_CU13_LIB" ]]; then
                export LD_LIBRARY_PATH="${{_CU13_LIB}}:${{LD_LIBRARY_PATH:-}}"
            fi
            """
        ).strip()
    else:
        fadvise_cmd = None

    evictor_block = ""
    if is_kt and fadvise_cmd:
        evictor_block = dedent(
            f"""
            # Event-driven page cache cleanup — wait for the server to finish
            # ALL loading phases (shard loading + KT expert loading + warmup),
            # then drop mmap'd safetensor pages from the page cache.
            (
                LOG_FILE="ambix-serve-$SLURM_JOB_ID.log"
                for _i in $(seq 1 300); do
                    if grep -q "fired up" "$LOG_FILE" 2>/dev/null; then
                        sleep 5
                        {fadvise_cmd} 2>/dev/null && echo "[$(date)] Post-startup: dropped safetensor page cache" || true
                        echo "[$(date)] === Memory diagnostics ==="
                        CGROUP_PATH="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${{SLURM_JOB_ID}}"
                        echo "cgroup memory.current: $(cat ${{CGROUP_PATH}}/memory.current 2>/dev/null || echo N/A)"
                        echo "cgroup memory.max: $(cat ${{CGROUP_PATH}}/memory.max 2>/dev/null || echo N/A)"
                        for pid in $(pgrep -P $SERVER_PID 2>/dev/null | head -8); do
                            if [[ -f /proc/$pid/smaps_rollup ]]; then
                                echo "--- PID $pid ---"
                                grep -E "Rss:|Anonymous:|Shared|Private" /proc/$pid/smaps_rollup 2>/dev/null | head -10
                            fi
                        done
                        echo "[$(date)] === End diagnostics ==="
                        break
                    fi
                    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                        break
                    fi
                    sleep 10
                done
            ) &
            _EVICTOR_PID=$!
            trap 'kill $_EVICTOR_PID 2>/dev/null || true' EXIT
            """
        ).strip()

    script_body = dedent(
        f"""
        set -euo pipefail

        export TMPDIR=/scratch_local/$SLURM_JOB_ID
        mkdir -p "$TMPDIR"

        {kt_env_block}

        # Ensure venv tools (ninja, etc.) are on PATH for JIT compilation
        export PATH={shlex.quote(venv_bin)}:$PATH

        MODEL_DIR={shlex.quote(str(model_dir))}
        PORT=${{AMBIX_PORT:-{port}}}

        if [[ ! -f "$MODEL_DIR/config.json" ]]; then
            echo "Error: missing model config at $MODEL_DIR/config.json" >&2
            exit 1
        fi

        echo "[$(date)] Job details"
        echo "  profile: {profile.slug}"
        echo "  engine: {profile.engine.type}"
        echo "  job id: ${{SLURM_JOB_ID:-unknown}}"
        echo "  node: $(hostname)"
        echo "  allocated GPUs:"
        echo "    count=${{SLURM_GPUS_ON_NODE:-unknown}}"
        echo "    visible=${{CUDA_VISIBLE_DEVICES:-unknown}}"
        echo "  model dir: $MODEL_DIR"
        echo "  port: $PORT"

        echo "[$(date)] GPU inventory"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

        echo "[$(date)] Host memory"
        free -h

        echo "[$(date)] Starting {profile.model.name} server"
        {launch_command} &
        SERVER_PID=$!

        {evictor_block}

        sleep 10
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "============================================="
            echo "{profile.model.name} is starting on $(hostname):$PORT"
            echo ""
            echo "Connect from a login node with:"
            echo "  ssh -N -L $PORT:$(hostname):$PORT <login-node>"
            echo ""
            echo "Then verify with:"
            echo "  curl http://localhost:$PORT/v1/models"
            echo "============================================="
        fi

        wait "$SERVER_PID"
        """
    ).strip()
    return "\n".join([*headers, "", script_body, ""])


def generate_download_script(profile: ModelProfile, site: SiteConfig) -> str:
    """Generate a SLURM batch script for downloading model weights.

    Downloads run on a standard compute partition (not the GPU partition)
    because GPU nodes may lack outbound network access.
    """
    model_dir = site.model_dir(profile)
    cache_dir = site.cache_dir(profile)
    hf_bin = str(site.hf_path)
    headers = _sbatch_headers(
        job_name=f"ambix-download-{profile.slug}",
        partition=site.download_partition,
        account=site.account,
        reservation=None,
        gpus=0,
        cpus=4,
        memory="16G",
        time_limit=profile.slurm.time_download,
        output_name="ambix-download-%j.log",
    )
    download_command = shlex.join(
        [
            hf_bin,
            "download",
            profile.model.hf_repo,
            "--local-dir",
            _MODEL_DIR_TOKEN,
            "--max-workers",
            "4",
        ]
    ).replace(_MODEL_DIR_TOKEN, '"$MODEL_DIR"')
    script_body = dedent(
        f"""
        set -euo pipefail

        export TMPDIR=/tmp
        export HF_HOME={shlex.quote(str(cache_dir))}

        MODEL_DIR={shlex.quote(str(model_dir))}

        mkdir -p "$MODEL_DIR" "$HF_HOME"

        echo "[$(date)] Starting download for {profile.model.name}"
        echo "  repo: {profile.model.hf_repo}"
        echo "  model dir: $MODEL_DIR"
        echo "  HF_HOME: $HF_HOME"
        echo "  node: $(hostname)"

        {download_command}

        echo "[$(date)] Download finished, validating contents"
        if [[ ! -f "$MODEL_DIR/config.json" ]]; then
            echo "Error: missing model config at $MODEL_DIR/config.json" >&2
            exit 1
        fi

        du -sh "$MODEL_DIR"
        echo "[$(date)] Download complete"
        """
    ).strip()
    return "\n".join([*headers, "", script_body, ""])


def submit_script(script: str) -> str:
    """Submit a generated SLURM script to ``sbatch`` and return the job ID."""
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "sbatch failed"
        raise RuntimeError(message)
    return result.stdout.strip().split(";", maxsplit=1)[0]
