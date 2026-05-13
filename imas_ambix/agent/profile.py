"""Model profile schema and loader for the ``ambix agent`` CLI.

Profiles are TOML files shipped as package data under
``imas_ambix/agent/profiles/``.  Each file defines a single model's
identity, engine configuration, and default SLURM resource requests.

Site-specific settings (partition, account, storage path) are layered
separately via environment variables — see :class:`SiteConfig`.
"""

from __future__ import annotations

import os
import tomllib
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

# -- Model identity ----------------------------------------------------------


class ModelConfig(BaseModel):
    """HuggingFace model identity and sizing."""

    name: str
    hf_repo: str
    served_name: str
    size_gb: int
    max_context: int


# -- Engine configuration ----------------------------------------------------


class KTransformersConfig(BaseModel):
    """KTransformers-specific backend parameters."""

    method: str = "RAWINT4"
    gpu_experts: int = 30
    cpuinfer: int = 28
    threadpool_count: int = 2
    disable_shared_experts_fusion: bool = True


class ParsersConfig(BaseModel):
    """SGLang chat-template parser overrides."""

    tool_call: str | None = None
    reasoning: str | None = None


class EngineConfig(BaseModel):
    """Inference engine configuration.

    ``type`` selects the backend strategy:

    - ``"ktransformers"`` — SGLang with KTransformers CPU-offloading
      (for models exceeding total VRAM).
    - ``"sglang"`` — SGLang native serving (model fits entirely on GPU).
    - ``"vllm"`` — vLLM native serving.
    """

    type: Literal["ktransformers", "sglang", "vllm"]
    tensor_parallel: int = 4
    mem_fraction_static: float = 0.90
    attention_backend: str = "flashinfer"
    trust_remote_code: bool = True
    enable_mixed_chunk: bool = True
    enable_p2p_check: bool = True
    chunked_prefill_size: int = 32768
    cuda_graph_max_bs: int | None = None
    disable_cuda_graph: bool = False
    disable_custom_all_reduce: bool = False
    max_total_tokens: int | None = None
    moe_runner_backend: Literal["auto", "triton", "triton_kernel"] | None = None
    weight_loader_disable_mmap: bool = False
    enable_auto_tool_choice: bool = False
    kv_cache_dtype: str | None = None
    ktransformers: KTransformersConfig | None = None
    parsers: ParsersConfig = ParsersConfig()


# -- SLURM defaults ----------------------------------------------------------


class SlurmDefaults(BaseModel):
    """Default SLURM resource requests baked into the profile.

    These can be overridden per-site via :class:`SiteConfig` or
    per-invocation via CLI flags.
    """

    gpus: int = 4
    cpus: int = 30
    memory: str = "640G"
    time_serve: str = "7-00:00:00"
    time_download: str = "24:00:00"


# -- Top-level profile -------------------------------------------------------


class ModelProfile(BaseModel):
    """Complete deployment profile for one LLM model."""

    slug: str
    model: ModelConfig
    engine: EngineConfig
    slurm: SlurmDefaults = SlurmDefaults()


# -- Site / cluster configuration ---------------------------------------------


class SiteConfig(BaseModel):
    """Cluster-specific settings, layered separately from model profiles.

    Read from environment variables with ``AMBIX_AGENT_`` prefix.
    Defaults match the ITER SDCC betelgeuse GPU partition.

    Engine venvs live under ``{base_dir}/agents/{engine}/``, each
    managed by its own ``pyproject.toml`` + ``uv.lock``.
    """

    base_dir: str = "/work/projects/imas_gpu"
    partition: str = "betelgeuse"
    download_partition: str = "sirius"
    account: str = "grpa"
    reservation: str = "gpu_0003_grpA"
    default_port: int = 8000
    gpu_host: str = "98dci4-gpu-0003"

    @classmethod
    def from_env(cls) -> SiteConfig:
        """Build config from environment, falling back to defaults."""
        return cls(
            base_dir=os.environ.get("AMBIX_AGENT_BASE_DIR", "/work/projects/imas_gpu"),
            partition=os.environ.get("AMBIX_AGENT_PARTITION", "betelgeuse"),
            download_partition=os.environ.get(
                "AMBIX_AGENT_DOWNLOAD_PARTITION", "sirius"
            ),
            account=os.environ.get("AMBIX_AGENT_ACCOUNT", "grpa"),
            reservation=os.environ.get(
                "AMBIX_AGENT_RESERVATION", "gpu_0003_grpA"
            ),
            default_port=int(os.environ.get("AMBIX_AGENT_PORT", "8000")),
            gpu_host=os.environ.get("AMBIX_AGENT_GPU_HOST", "98dci4-gpu-0003"),
        )

    @property
    def default_url(self) -> str:
        """Default server URL constructed from gpu_host and default_port."""
        return f"http://{self.gpu_host}:{self.default_port}"

    def _engine_key(self, engine_type: str) -> str:
        """Map engine type to venv directory name.

        ``ktransformers`` shares the ``sglang`` venv since it runs
        as an SGLang plugin.
        """
        if engine_type == "ktransformers":
            return "sglang"
        return engine_type

    def env_dir(self, engine_type: str) -> Path:
        """Root of the uv-managed env for *engine_type*."""
        return Path(self.base_dir) / "agents" / self._engine_key(engine_type)

    def venv_path(self, engine_type: str) -> Path:
        """Path to the venv for *engine_type*."""
        return self.env_dir(engine_type) / ".venv"

    def python_path(self, engine_type: str) -> Path:
        """Path to the venv Python binary for *engine_type*."""
        return self.venv_path(engine_type) / "bin" / "python"

    def hf_path(self, engine_type: str) -> Path:
        """Path to the ``hf`` CLI binary for *engine_type*."""
        return self.venv_path(engine_type) / "bin" / "hf"

    def model_dir(self, profile: ModelProfile) -> Path:
        """Filesystem path for downloaded model weights."""
        return Path(self.base_dir) / "agents" / profile.slug / "model"

    def cache_dir(self, profile: ModelProfile) -> Path:
        """HuggingFace cache directory for a model."""
        return Path(self.base_dir) / "agents" / profile.slug / ".cache"


# -- Profile loader -----------------------------------------------------------

_PROFILES_PACKAGE = "imas_ambix.agent.profiles"


def list_profiles() -> list[str]:
    """Return sorted slugs of all available model profiles."""
    pkg = resources.files(_PROFILES_PACKAGE)
    return sorted(
        p.name.removesuffix(".toml")
        for p in pkg.iterdir()
        if p.name.endswith(".toml")
    )


def load_profile(slug: str) -> ModelProfile:
    """Load and validate a model profile by slug.

    Raises
    ------
    FileNotFoundError
        If no profile TOML exists for *slug*.
    """
    pkg = resources.files(_PROFILES_PACKAGE)
    toml_ref = pkg.joinpath(f"{slug}.toml")
    try:
        text = toml_ref.read_text(encoding="utf-8")
    except FileNotFoundError:
        available = list_profiles()
        msg = f"No profile '{slug}'. Available: {', '.join(available) or '(none)'}"
        raise FileNotFoundError(msg) from None
    data = tomllib.loads(text)
    return ModelProfile(slug=slug, **data)
