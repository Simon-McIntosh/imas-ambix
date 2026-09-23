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
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -- Model identity ----------------------------------------------------------


class ModelConfig(BaseModel):
    """HuggingFace model identity and sizing."""

    name: str
    hf_repo: str
    served_name: str
    size_gb: int
    max_context: int
    # Checkpoint precision is catalog metadata, distinct from KV-cache dtype.
    # It remains optional so existing profiles load, but vLLM catalog serving
    # requires an explicit value.
    checkpoint_precision: str | None = None
    # Directory key under ``agents/<slug>/`` holding the downloaded weights.
    # Injected by the loader for ``_base`` inheritance, so two profiles in one
    # chain share one download. Declare it explicitly to redirect the weights
    # elsewhere: in a ``gpu_variants`` entry whose card count needs a different
    # checkpoint, or on a profile whose release does not share a directory with
    # the one it superseded. An explicit value always wins over the injected
    # one, including for variants inheriting from it.
    weights_slug: str | None = None
    # Directory key under ``agents/<slug>/`` for tokenizer assets when they
    # come from a different released checkpoint than the model weights.
    # An empty variant override restores the default of loading tokenizer
    # assets from the weights directory.
    tokenizer_source_slug: str | None = None

    @field_validator("tokenizer_source_slug", mode="before")
    @classmethod
    def _empty_tokenizer_source_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


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


class ContainerBind(BaseModel):
    """A read-only repository file mounted at an image path for a serve."""

    source: str
    target: str

    @model_validator(mode="after")
    def _bind_paths_are_unambiguous(self) -> ContainerBind:
        """Require a repository-relative source and an absolute image target."""
        source = Path(self.source)
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("container bind source must be repository-relative")
        if not Path(self.target).is_absolute():
            raise ValueError("container bind target must be absolute")
        return self


class ContainerConfig(BaseModel):
    """Serve this model from a container image instead of an engine venv.

    Some architectures land in an engine's main branch well before they reach a
    released wheel, and the shared engine environments track releases. Declaring
    an image here routes the serve through ``apptainer exec --nv`` against
    :attr:`sif_path`, leaving every other profile on the venv path untouched.

    :attr:`image` is the upstream reference the SIF was built from. It is
    recorded so a deployment can say what it is running rather than only where
    the file sits; the serve reads :attr:`sif_path` and never pulls, because the
    GPU node has no egress.
    """

    image: str
    sif_path: str
    binds: list[ContainerBind] = []


class EngineConfig(BaseModel):
    """Inference engine configuration.

    ``type`` selects the backend strategy:

    - ``"ktransformers"`` — SGLang with KTransformers CPU-offloading
      (for models exceeding total VRAM).
    - ``"sglang"`` — SGLang native serving (model fits entirely on GPU).
    - ``"vllm"`` — vLLM native serving.
    """

    # A misspelled or unsupported key is a configuration error, not something
    # to drop. Pydantic's default is to ignore extras, which meant a profile
    # could declare a flag the engine never received and read as configured --
    # measured with DSpark, where the serve would have run without speculative
    # decoding while the profile said otherwise.
    model_config = ConfigDict(extra="forbid")

    type: Literal["ktransformers", "sglang", "vllm"]
    tensor_parallel: int = 4
    # Engine replicas sharing the cards, vLLM's ``--data-parallel-size``. The
    # card count is tensor_parallel * data_parallel, so 2 and 2 fill four cards
    # with two replicas rather than one four-wide engine.
    #
    # It exists because this serve is step-budget-bound rather than
    # pool-bound: one engine serialises every request's prefill and decode into
    # a single step budget, so a long prefill chunk starves decode for every
    # concurrent session. Replicas give that budget once each.
    #
    # On a MoE with ``enable_expert_parallel``, expert weights stay sharded
    # across all ranks rather than being duplicated per replica, so this does
    # NOT cost the KV pool the way running separate serves would. Replicas do
    # synchronise at the MoE all-to-all and idle ranks are padded with dummy
    # batches, so the independence is partial and the gain has to be measured
    # rather than assumed. 1 keeps a single engine.
    data_parallel: int = 1
    # Expert-parallel width for SGLang's ``--ep-size``. Distinct from
    # ``enable_expert_parallel`` below, which is vLLM's boolean switch. A MoE
    # layer with hundreds of routed experts is sharded by this rather than by
    # tensor parallelism; ``None`` keeps the SGLang default. SGLang-only.
    ep_size: int | None = None
    mem_fraction_static: float = 0.90
    attention_backend: str = "flashinfer"
    trust_remote_code: bool = True
    enable_mixed_chunk: bool = True
    enable_p2p_check: bool = True
    chunked_prefill_size: int = 32768
    # SGLang queues requests beyond this engine-side running bound rather than
    # rejecting them when its queue limit remains unset. ``None`` preserves the
    # engine default. SGLang-only.
    max_running_requests: int | None = None
    cuda_graph_max_bs: int | None = None
    # Separate decode-side CUDA-graph batch ceiling
    # (``--cuda-graph-max-bs-decode``). Architectures that split prefill and
    # decode into different graphs size them independently; ``None`` keeps the
    # SGLang default. SGLang-only.
    cuda_graph_max_bs_decode: int | None = None
    # Bounded replay on the decoder half of an encoder-decoder stack
    # (``--enable-decoder-swa-bounded-replay``). Measured by the vendor at 1.56x
    # prefill throughput on 8xH200 for DeepSeek-V4.1. SGLang-only.
    enable_decoder_swa_bounded_replay: bool = False
    # Publish SGLang's Prometheus endpoint (``--enable-metrics``). SGLang
    # defaults this off; vLLM exposes metrics by default and does not consume
    # this setting.
    enable_metrics: bool = False
    # SGLang speculative decoding. Distinct from the vLLM ``speculative_method``
    # family below, which emits --speculative-config; SGLang takes
    # --speculative-algorithm and its own per-algorithm options. Keeping both
    # names is deliberate: a profile that set the vLLM key on an SGLang engine
    # would silently emit nothing. SGLang-only.
    speculative_algorithm: str | None = None
    speculative_dspark_block_size: int | None = None
    disable_cuda_graph: bool = False
    disable_piecewise_cuda_graph: bool = False
    disable_custom_all_reduce: bool = False
    # Per-request context the ENGINE enforces (``--context-length``), distinct
    # from max_total_tokens, which is the shared pool across requests. Set this
    # below the model's native window when the native window is not reachable
    # in practice: the MXFP4 fused-MoE prefill workspace grows with prefill
    # length, so a request far inside the advertised context can still exhaust
    # the card. Capping at the engine turns that from an OOM that kills the
    # serve into a clean refusal of one request. ``None`` keeps the model's own
    # value. SGLang-only.
    context_length: int | None = None
    max_total_tokens: int | None = None
    # Keep evicted reusable prefix pages in pinned host RAM. SGLang's
    # ``hicache_ratio`` is the host-to-device KV-pool ratio; a configured
    # ratio is preferable to its engine default because it makes the host-RAM
    # reservation reviewable with the profile. DeepSeek-V4's hybrid cache does
    # not support a fixed ``hicache_size`` and rejects that option.
    enable_hierarchical_cache: bool = False
    hicache_ratio: float | None = None
    hicache_write_policy: Literal[
        "write_back", "write_through", "write_through_selective"
    ] = "write_through"
    hicache_mem_layout: Literal[
        "layer_first",
        "page_first",
        "page_first_direct",
        "page_first_kv_split",
        "page_head",
    ] = "page_first"
    # Let the engine size the KV pool from the memory actually left after
    # weights, instead of passing a figure. ``max_total_tokens`` otherwise
    # falls back to the model's full context, which is a per-request limit and
    # a poor pool size -- it makes one full-context request consume everything.
    # Sizing it by hand means extrapolating a per-token cost that is mostly a
    # fixed base, so the engine's own allocator is the better estimator.
    # SGLang-only; ignored when ``max_total_tokens`` is set.
    auto_size_kv_pool: bool = False
    # ``flashinfer_mxfp4`` is what keeps MXFP4 routed experts at their shipped
    # precision on SM90, where there are no FP4 tensor cores — without it the
    # experts need an FP8 conversion pass and a second checkpoint.
    moe_runner_backend: (
        Literal["auto", "triton", "triton_kernel", "flashinfer_mxfp4"] | None
    ) = None
    # Computation precision inside the FlashInfer MXFP4 MoE runner. ``default``
    # upcasts activations to BF16, which is the only thing SM90 can do with a
    # 4-bit weight through that path; ``fp8`` selects the Humming-style
    # MXFP4-weight x FP8-activation kernels, reaching the FP8 tensor cores
    # Hopper does have. It needs FlashInfer >= 0.6.18 -- the serving container
    # carries exactly 0.6.18, so the floor is met rather than exceeded, and a
    # container rebuild below that version silently reverts the path.
    #
    # This is a prefill lever, which is what makes it worth the field: measured
    # on real agent traffic, 99.4% of this deployment's token work is prefill
    # (270 input tokens per output token at a mean prompt of 83,814), so the
    # dequantisation cost sits on the dominant term. Only meaningful alongside
    # ``moe_runner_backend = "flashinfer_mxfp4"``.
    flashinfer_mxfp4_moe_precision: Literal["default", "bf16", "fp8"] | None = None
    # CLI flag is `--fp8-gemm-backend` but the ServerArgs attribute
    # SGLang uses internally is `fp8_gemm_runner_backend`; mirror the
    # internal name here. Allowed values match SGLang's argparse.
    fp8_gemm_runner_backend: (
        Literal[
            "auto",
            "deep_gemm",
            "flashinfer_trtllm",
            "flashinfer_cutlass",
            "flashinfer_deepgemm",
            "cutlass",
            "triton",
            "aiter",
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
    # Tokenizer mode override, e.g. DeepSeek-V4's dedicated fast tokenizer
    # (``--tokenizer-mode deepseek_v4``). ``None`` keeps the vLLM default.
    # vLLM-only.
    tokenizer_mode: str | None = None
    # Route MoE dispatch through vLLM's ``--moe-backend`` (distinct from
    # ``moe_runner_backend`` above, which is SGLang's ``--moe-runner-backend``
    # flag). ``None`` keeps the vLLM default. vLLM-only.
    moe_backend: str | None = None
    # Expert-parallel MoE dispatch (required alongside tensor parallelism for
    # DeepSeek-V4's routed-expert layout). vLLM-only.
    enable_expert_parallel: bool = False
    # vLLM scheduler caps. ``None`` keeps the vLLM default
    # (``max_num_seqs=256`` in recent releases, which becomes the hard
    # ceiling on in-flight requests and is the dominant throughput
    # bottleneck on a 4×H200 cluster — KV cache typically sits at ~12 %
    # usage under that cap). Set these explicitly to scale to the
    # available HBM. Only forwarded for the vLLM engine type.
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    # Host-RAM buffer for evicted prefix blocks, in GiB summed across tensor
    # ranks. Unset means vLLM recomputes an evicted prefix from scratch, which
    # is what makes a shared agent lane self-defeating: the recomputation is
    # itself what evicts the next session's prefix. Measured 2026-09-15 on the
    # four-card serve -- 22,557 tok/s of prefill against 190 of generation, of
    # which 18,863 tok/s was recomputation, and a 2,200,283-token pool turning
    # over completely every 98 s against turn gaps of about the same length.
    # Restoring a block over PCIe costs far less than a forward pass through
    # 284B parameters.
    kv_offloading_size: float | None = None
    # vLLM Multi-Token-Prediction (MTP) speculative decoding. When
    # ``speculative_method`` is set, the serve command emits
    # ``--speculative-config.method`` and
    # ``--speculative-config.num_speculative_tokens`` so the model drafts
    # several tokens per step (GLM-5.2 ships an MTP head tuned for 5 draft
    # tokens — the headline throughput win over GLM-5.1). vLLM-only.
    #
    # When ``speculative_model`` is also set, the speculative-config is
    # emitted as a compact JSON string:
    #   ``{"model": "<speculative_model>", "method": "mtp",
    #     "num_speculative_tokens": N}``
    # so vLLM loads a *separate* draft-model checkpoint (needed when AWQ
    # quantization damages the integrated MTP head — GLM-5.2 INT4 uses
    # the community ``CosmicRaisins/GLM-5.2-MTP-INT4`` draft).
    speculative_method: str | None = None
    speculative_num_tokens: int | None = None
    # HF repo or local path for a separate draft-model checkpoint.
    # Mutually complementary with ``speculative_method``: set both
    # together when the draft lives in a different weight directory.
    speculative_model: str | None = None
    # Draft-token sampling strategy for the speculative decoder (DSpark's
    # fused module accepts ``"greedy"`` or ``"probabilistic"``). Forces the
    # compact-JSON ``--speculative-config`` form below even when no separate
    # ``speculative_model`` is set, because DSpark's module is fused into the
    # checkpoint rather than loaded from a distinct draft-model path.
    # vLLM-only.
    speculative_draft_sample_method: str | None = None
    # Extra environment variables exported into the serve job before launch.
    # For engine/kernel quirks that are set via env, not CLI flags — e.g.
    # ``VLLM_USE_FLASHINFER_SAMPLER = "0"`` to route sampling around a broken
    # FlashInfer top-k kernel on this H200 + vLLM build. Values are stringified.
    env: dict[str, str] = {}
    ktransformers: KTransformersConfig | None = None
    container: ContainerConfig | None = None
    parsers: ParsersConfig = ParsersConfig()

    @model_validator(mode="after")
    def _parallelism_is_positive(self) -> EngineConfig:
        """Refuse a parallel width below one rather than silently clamping it.

        Both widths multiply into the card count, so a zero or negative value
        would produce a request for no cards or a nonsensical one, and pydantic
        would otherwise carry it to the engine untouched.
        """
        if self.tensor_parallel < 1:
            raise ValueError("tensor_parallel must be at least 1")
        if self.data_parallel < 1:
            raise ValueError("data_parallel must be at least 1")
        return self

    @model_validator(mode="after")
    def _absent_speculation_is_none(self) -> EngineConfig:
        """Treat an empty speculative setting as absent rather than as a value.

        A ``gpu_variants`` entry can only override a key, never delete one, so a
        variant that must run WITHOUT speculative decoding has to express "off"
        as an empty method and a zero draft count. Those would otherwise be
        forwarded verbatim and rejected by the engine, which accepts neither an
        empty method name nor a non-positive draft count. Normalising them to
        ``None`` here makes the serve command omit the flags entirely, which is
        what "off" means.
        """
        if not self.speculative_method:
            self.speculative_method = None
        if not self.speculative_num_tokens or self.speculative_num_tokens <= 0:
            self.speculative_num_tokens = None
        if not self.speculative_draft_sample_method:
            self.speculative_draft_sample_method = None
        if self.speculative_method is None:
            self.speculative_model = None
            self.speculative_draft_sample_method = None
        return self


# -- SLURM defaults ----------------------------------------------------------


class SlurmDefaults(BaseModel):
    """Default SLURM resource requests baked into the profile.

    These can be overridden per-site via :class:`SiteConfig` or
    per-invocation via CLI flags.
    """

    gpus: int = 4
    cpus: int = 30
    memory: str = "640G"
    port: int | None = Field(default=None, ge=1, le=65535)
    time_serve: str = "7-00:00:00"
    time_download: str = "24:00:00"


# -- Top-level profile -------------------------------------------------------


class ModelProfile(BaseModel):
    """Complete deployment profile for one LLM model.

    ``gpu_variants`` lets one profile carry more than one checkpoint for the
    same model release, keyed on the card count it is sized for. A release is
    then one slug whose deployment is chosen at launch with ``--gpus N``,
    rather than several slugs that each bake a card count or a precision into
    their name. Use it only where the card counts genuinely need DIFFERENT
    WEIGHTS; where they differ only in tensor-parallel width, plain ``--gpus``
    rescaling already covers it and no variant is needed.

    Each entry may override any ``[model]``, ``[engine]``, or ``[slurm]`` key,
    and is deep-merged over the base when that card count is requested.
    """

    slug: str
    model: ModelConfig
    engine: EngineConfig
    slurm: SlurmDefaults = SlurmDefaults()
    # Per-card-count checkpoint overrides, keyed on the GPU count they suit.
    gpu_variants: dict[int, dict] = {}

    def for_gpus(self, gpus: int) -> ModelProfile:
        """Return this profile resolved for a *gpus*-card deployment.

        Applies the matching :attr:`gpu_variants` entry when one exists and
        returns ``self`` unchanged otherwise, so a profile without variants
        behaves exactly as before.
        """
        variant = self.gpu_variants.get(gpus)
        if not variant:
            return self
        merged = _deep_merge(self.model_dump(exclude={"gpu_variants"}), dict(variant))
        merged.pop("slug", None)
        return ModelProfile(slug=self.slug, gpu_variants=self.gpu_variants, **merged)

    @property
    def weights_directory_slug(self) -> str:
        """Directory name this profile's checkpoint is stored under.

        A declared ``weights_slug`` wins, which is how a topology variant
        shares its base's download and how two releases of one model keep
        their shards apart; otherwise the profile's own slug is used.
        """
        return self.model.weights_slug or self.slug


# -- Site / cluster configuration ---------------------------------------------


def _default_engine_env_root() -> str:
    """Return the per-user data directory for serving environments."""
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return str(root / "ambix" / "engine-envs")


def _default_endpoint_document_path() -> str:
    """Return the public, home-backed endpoint publication path."""
    return str(Path.home() / "public" / "imas-ambix" / "endpoints.json")


class SiteConfig(BaseModel):
    """Cluster-specific settings, layered separately from model profiles.

    Read from environment variables with ``AMBIX_AGENT_`` prefix.
    Defaults match the ITER SDCC betelgeuse GPU partition.

    Model weights and shared launch artifacts live under the project base.
    Engine environments use a per-user home-backed root so networked setup
    nodes and GPU serving nodes resolve the same files.
    """

    base_dir: str = "/work/projects/imas_gpu"
    engine_env_root: str = Field(default_factory=_default_engine_env_root)
    engine_env_min_free_gb: int = Field(default=32, ge=1)
    partition: str = "betelgeuse"
    download_partition: str = "sirius"
    account: str = "grpa"
    reservation: str = "gpu_0003_grpA"
    # The interactive agent fleet is hosted on a CPU partition, so its
    # placement is site configuration in its own right rather than a reuse of
    # the GPU fields above. It charges ``iter`` rather than the GPU account
    # ``grpa``: ``grpa`` carries the GPU reservation's QOS, while ``iter``
    # holds the site-default ``normal`` QOS that a whole-node CPU allocation
    # belongs to.
    fleet_partition: str = "rigel"
    fleet_account: str = "iter"
    fleet_cpus: int = 28
    fleet_memory: str = "120G"
    default_port: int = 18800
    gpu_host: str = "98dci4-gpu-0003"
    global_origin: str = "http://98dci4-gpu-0003:18800"
    endpoint_document_path: str = Field(default_factory=_default_endpoint_document_path)
    preferred_release_id: str | None = None

    @field_validator("global_origin", mode="before")
    @classmethod
    def _normalize_global_origin(cls, value: object) -> str:
        """Validate and normalize the site-owned Clive catalog origin."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("global origin must be a non-empty HTTP(S) origin")
        candidate = value.strip()
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("global origin must be an absolute HTTP(S) origin")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("global origin must not contain user information")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError(
                "global origin must not contain a path, query, or fragment"
            )
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("global origin contains an invalid port") from exc
        return candidate.rstrip("/")

    @field_validator("preferred_release_id", mode="before")
    @classmethod
    def _normalize_preferred_release_id(cls, value: object) -> str | None:
        """Normalize the optional release selected by an unqualified launch."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("preferred release id must be text")
        candidate = value.strip()
        if not candidate:
            return None
        if any(ord(character) < 32 or ord(character) == 127 for character in candidate):
            raise ValueError("preferred release id contains control characters")
        return candidate

    @classmethod
    def from_env(cls) -> SiteConfig:
        """Build config from environment, falling back to defaults."""
        return cls(
            base_dir=os.environ.get("AMBIX_AGENT_BASE_DIR", "/work/projects/imas_gpu"),
            engine_env_root=os.environ.get(
                "AMBIX_AGENT_ENGINE_ENV_ROOT", _default_engine_env_root()
            ),
            engine_env_min_free_gb=int(
                os.environ.get("AMBIX_AGENT_ENGINE_ENV_MIN_FREE_GB", "32")
            ),
            partition=os.environ.get("AMBIX_AGENT_PARTITION", "betelgeuse"),
            download_partition=os.environ.get(
                "AMBIX_AGENT_DOWNLOAD_PARTITION", "sirius"
            ),
            account=os.environ.get("AMBIX_AGENT_ACCOUNT", "grpa"),
            reservation=os.environ.get("AMBIX_AGENT_RESERVATION", "gpu_0003_grpA"),
            fleet_partition=os.environ.get("AMBIX_AGENT_FLEET_PARTITION", "rigel"),
            fleet_account=os.environ.get("AMBIX_AGENT_FLEET_ACCOUNT", "iter"),
            fleet_cpus=int(os.environ.get("AMBIX_AGENT_FLEET_CPUS", "28")),
            fleet_memory=os.environ.get("AMBIX_AGENT_FLEET_MEMORY", "120G"),
            default_port=int(os.environ.get("AMBIX_AGENT_PORT", "18800")),
            gpu_host=os.environ.get("AMBIX_AGENT_GPU_HOST", "98dci4-gpu-0003"),
            global_origin=os.environ.get(
                # The ROUTER, not a serve port. This origin is baked into the
                # generated clive launcher as ANTHROPIC_BASE_URL, so it must
                # name the multi-engine front door rather than whichever serve
                # happened to hold 18800 when the default was written. It had
                # gone stale exactly that way: `clive --list` kept working
                # because it reads the endpoint document, while a clive session
                # dialled a dead port and failed with unrecognized_model.
                "AMBIX_AGENT_GLOBAL_URL",
                "http://98dci4-gpu-0003:18802",
            ),
            endpoint_document_path=os.environ.get(
                "AMBIX_AGENT_ENDPOINT_DOCUMENT", _default_endpoint_document_path()
            ),
            preferred_release_id=os.environ.get("AMBIX_AGENT_PREFERRED_RELEASE"),
        )

    @property
    def endpoint_document(self) -> Path:
        """Public document from which standalone launchers discover engines."""
        return Path(self.endpoint_document_path).expanduser()

    def _engine_key(self, engine_type: str) -> str:
        """Map engine type to venv directory name.

        Every engine type names its own environment. ``ktransformers`` runs as
        an SGLang plugin but is NOT served from the ``sglang`` environment:
        its ``kt-kernel`` dependency publishes cp312 wheels only, and sharing
        the environment would pin SGLang to that interpreter too.
        """
        return engine_type

    def env_dir(self, engine_type: str) -> Path:
        """Root of the uv-managed env for *engine_type*."""
        return Path(self.engine_env_root) / self._engine_key(engine_type)

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
        return profile.weights_directory_slug

    def _checkpoint_dir(self, slug: str) -> Path:
        """Return the model directory for a checkpoint storage slug."""
        return Path(self.base_dir) / "agents" / slug / "model"

    def model_dir(self, profile: ModelProfile) -> Path:
        """Filesystem path for downloaded model weights."""
        return self._checkpoint_dir(self._weights_slug(profile))

    def tokenizer_dir(self, profile: ModelProfile) -> Path | None:
        """Filesystem path for an explicitly separate tokenizer checkpoint."""
        slug = profile.model.tokenizer_source_slug
        return self._checkpoint_dir(slug) if slug is not None else None

    def cache_dir(self, profile: ModelProfile) -> Path:
        """HuggingFace cache directory for a model."""
        return Path(self.base_dir) / "agents" / self._weights_slug(profile) / ".cache"

    @property
    def receipts_dir(self) -> Path:
        """Where every serve's recorder appends, and where a reader looks.

        The writer is launched with this path and a reader discovers the
        record through it, so the two resolve the same directory by
        construction rather than by two constants agreeing. A reader holding
        its own spelling finds nothing and reports an empty record, which is
        indistinguishable from a lane that served nothing.
        """
        return Path(self.base_dir) / "agents" / "receipts"

    @property
    def api_key_file(self) -> Path:
        """Shared API key file for model serving authentication."""
        return Path(self.base_dir) / "agents" / ".env"

    @property
    def clive_path(self) -> Path:
        """Deployed location of the ``clive`` agent-CLI launcher.

        A standalone, dependency-free shell script — generated by
        ``imas-ambix agent clive --deploy`` and placed on shared GPFS so
        everyone in the storage group can run it anonymously. Distinct from
        the operator-only ``imas-ambix`` CLI, which lives in a per-user repo
        venv and manages authenticated backend serving.
        """
        return Path(self.base_dir) / "agents" / "clive"

    @property
    def litellm_config_path(self) -> Path:
        """Secret-free routing config for the opt-in per-user proxy."""
        return Path(self.base_dir) / "agents" / "litellm_config.yaml"

    @property
    def litellm_env_helper_path(self) -> Path:
        """Credential helper deployed beside the routing config."""
        return self.litellm_config_path.with_name("imas-ambix-llm-env.sh")

    @property
    def litellm_service_path(self) -> Path:
        """Per-user systemd unit for the opt-in proxy."""
        return Path.home() / ".config" / "systemd" / "user" / "imas-ambix-llm.service"


# -- Profile loader -----------------------------------------------------------

_PROFILES_PACKAGE = "imas_ambix.agent.profiles"


def list_profiles() -> list[str]:
    """Return sorted slugs of all available model profiles."""
    pkg = resources.files(_PROFILES_PACKAGE)
    return sorted(
        p.name.removesuffix(".toml") for p in pkg.iterdir() if p.name.endswith(".toml")
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
    values from the named profile and override only the keys they specify.
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
        # Point a variant at the root of its inheritance chain so one download
        # serves the whole chain -- but never over a weights_slug the chain
        # declared for itself. A profile that redirects its weights (because
        # the release it tracks ships a different shard count, say) means that
        # redirect for its variants too; overwriting it here would silently
        # load a different checkpoint under the variant's name.
        data.setdefault("model", {}).setdefault("weights_slug", canonical_slug)
    return ModelProfile(slug=slug, **data)
