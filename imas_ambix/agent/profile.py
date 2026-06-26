"""Model profile schema and loader for the ``imas-ambix agent`` CLI.

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
    # Injected by the profile loader when this profile inherits from a base
    # via ``_base``.  Points to the root-of-chain slug whose ``agents/<slug>/``
    # directory holds the downloaded weights.  Never set manually in a TOML file.
    weights_slug: str | None = None


# -- Engine configuration ----------------------------------------------------


class KTransformersConfig(BaseModel):
    """KTransformers-specific backend parameters."""

    method: str = "RAWINT4"
    gpu_experts: int = 30
    cpuinfer: int = 28
    threadpool_count: int = 2
    disable_shared_experts_fusion: bool = True
    # FP8 path — required for models like GLM-5.1 and MiMo-V2.5-Pro
    gpu_prefill_token_threshold: int | None = None
    enable_dynamic_expert_update: bool = False
    expert_placement_strategy: str | None = None


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
    disable_piecewise_cuda_graph: bool = False
    disable_custom_all_reduce: bool = False
    max_total_tokens: int | None = None
    moe_runner_backend: Literal["auto", "triton", "triton_kernel"] | None = None
    # CLI flag is `--fp8-gemm-backend` but the ServerArgs attribute
    # SGLang uses internally is `fp8_gemm_runner_backend`; mirror the
    # internal name here. Allowed values match SGLang's argparse.
    fp8_gemm_runner_backend: (
        Literal[
            "auto", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass",
            "flashinfer_deepgemm", "cutlass", "triton", "aiter",
        ]
        | None
    ) = None
    weight_loader_disable_mmap: bool = False
    enable_auto_tool_choice: bool = False
    kv_cache_dtype: str | None = None
    # KV-cache block size. ``None`` keeps the vLLM default. MiniMax M3 requires
    # ``--block-size 128`` on every platform (its MSA sparse/index cache);
    # vLLM-only.
    block_size: int | None = None
    # vLLM scheduler caps. ``None`` keeps the vLLM default
    # (``max_num_seqs=256`` in recent releases, which becomes the hard
    # ceiling on in-flight requests and is the dominant throughput
    # bottleneck on a 4×H200 cluster — KV cache typically sits at ~12 %
    # usage under that cap). Set these explicitly to scale to the
    # available HBM. Only forwarded for the vLLM engine type.
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    # vLLM Multi-Token-Prediction (MTP) speculative decoding. When
    # ``speculative_method`` is set, the serve command emits
    # ``--speculative-config.method`` and
    # ``--speculative-config.num_speculative_tokens`` so the model drafts
    # several tokens per step (GLM-5.2 ships an MTP head tuned for 5 draft
    # tokens — the headline throughput win over GLM-5.1). vLLM-only.
    speculative_method: str | None = None
    speculative_num_tokens: int | None = None
    # Extra environment variables exported into the serve job before launch.
    # For engine/kernel quirks that are set via env, not CLI flags — e.g.
    # ``VLLM_USE_FLASHINFER_SAMPLER = "0"`` to route sampling around a broken
    # FlashInfer top-k kernel on this H200 + vLLM build. Values are stringified.
    env: dict[str, str] = {}
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
    default_port: int = 18800
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
            default_port=int(os.environ.get("AMBIX_AGENT_PORT", "18800")),
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

    def _weights_slug(self, profile: ModelProfile) -> str:
        """Slug whose ``agents/<slug>/`` directory holds the model weights.

        Variant profiles that inherit from a base via ``_base`` redirect to
        the base's directory so weights are not downloaded twice.
        """
        return profile.model.weights_slug or profile.slug

    def model_dir(self, profile: ModelProfile) -> Path:
        """Filesystem path for downloaded model weights."""
        return Path(self.base_dir) / "agents" / self._weights_slug(profile) / "model"

    def cache_dir(self, profile: ModelProfile) -> Path:
        """HuggingFace cache directory for a model."""
        return Path(self.base_dir) / "agents" / self._weights_slug(profile) / ".cache"

    @property
    def api_key_file(self) -> Path:
        """Shared API key file for model serving authentication."""
        return Path(self.base_dir) / "agents" / ".env"

    @property
    def clive_path(self) -> Path:
        """Deployed location of the ``clive`` agent-CLI launcher.

        A standalone, dependency-free shell script — generated by
        ``imas-ambix agent clive --deploy`` and placed on shared GPFS so
        everyone in the storage group can run it (it reads the shared
        :attr:`api_key_file` at runtime).  Distinct from the operator-only
        ``imas-ambix`` CLI, which lives in a per-user repo venv.
        """
        return Path(self.base_dir) / "agents" / "clive"

    @property
    def litellm_config_path(self) -> Path:
        """Deployed clive LiteLLM routing config (secret-free; keys via env)."""
        return Path(self.base_dir) / "agents" / "litellm_config.yaml"

    @property
    def litellm_service_path(self) -> Path:
        """Per-user systemd unit for the clive LiteLLM proxy (installed in $HOME)."""
        return Path.home() / ".config" / "systemd" / "user" / "imas-ambix-llm.service"


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


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* merged on top of *base*.

    Nested dicts are merged recursively — a partial ``[engine]`` table in the
    override only replaces the keys it specifies, leaving the rest of the
    base's ``[engine]`` intact.  All other values are replaced wholesale.
    """
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_raw(
    slug: str,
    *,
    _seen: frozenset[str] | None = None,
) -> tuple[dict, str]:
    """Load a profile TOML as a raw dict, resolving ``_base`` inheritance.

    Returns
    -------
    data : dict
        Fully merged profile data ready for Pydantic validation.
    canonical_slug : str
        Slug of the root-of-chain profile whose ``agents/<slug>/`` directory
        holds the actual downloaded model weights.  Equals *slug* for
        standalone profiles (no ``_base``).

    Raises
    ------
    FileNotFoundError
        If *slug* does not exist.
    ValueError
        If a circular inheritance chain is detected.
    """
    if _seen is None:
        _seen = frozenset()
    if slug in _seen:
        chain = " -> ".join([*sorted(_seen), slug])
        msg = f"Circular profile inheritance detected: {chain}"
        raise ValueError(msg)
    _seen = _seen | {slug}

    pkg = resources.files(_PROFILES_PACKAGE)
    toml_ref = pkg.joinpath(f"{slug}.toml")
    try:
        text = toml_ref.read_text(encoding="utf-8")
    except FileNotFoundError:
        available = list_profiles()
        msg = f"No profile '{slug}'. Available: {', '.join(available) or '(none)'}"
        raise FileNotFoundError(msg) from None

    data = tomllib.loads(text)

    if "_base" in data:
        base_slug = data.pop("_base")
        base_data, canonical_slug = _load_raw(base_slug, _seen=_seen)
        data = _deep_merge(base_data, data)
        return data, canonical_slug

    return data, slug


def load_profile(slug: str) -> ModelProfile:
    """Load and validate a model profile by slug.

    Variant profiles that declare ``_base = "<other-slug>"`` inherit all
    settings from the named profile and override only the keys they specify.
    The ``model.weights_slug`` field is automatically set to the root-of-chain
    slug so that ``SiteConfig.model_dir()`` resolves to the correct weights
    directory without re-downloading.

    Raises
    ------
    FileNotFoundError
        If no profile TOML exists for *slug*.
    ValueError
        If a circular ``_base`` chain is detected.
    """
    data, canonical_slug = _load_raw(slug)
    if canonical_slug != slug:
        data.setdefault("model", {})["weights_slug"] = canonical_slug
    return ModelProfile(slug=slug, **data)
