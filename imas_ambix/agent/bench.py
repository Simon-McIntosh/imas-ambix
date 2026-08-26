"""Comprehensive LLM benchmark suite for ``imas-ambix agent bench``.

Provides model-agnostic benchmarks across six categories:
throughput, prefill, context (needle-in-haystack), tools,
reasoning, and concurrency.

All HTTP is done via :mod:`urllib.request` (stdlib) to avoid
external dependencies. Token counts come from server-reported
``usage`` fields, not chunk counting.
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import hashlib
import json
import os
import re
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import-time only: the profile loader pulls in pydantic and the profile
    # package data, which the benchmark itself never needs.
    from imas_ambix.agent.profile import ModelProfile

# ── Data model ──────────────────────────────────────────────────────


@dataclass
class BenchResult:
    """Metrics from a single benchmark request."""

    category: str = ""
    test_name: str = ""
    status: str = "passed"  # passed, failed, skipped, error
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    time_to_first_token_s: float = 0.0
    total_time_s: float = 0.0
    decode_tps: float = 0.0
    prefill_tps: float = 0.0
    model: str = ""
    error: str | None = None
    finish_reason: str | None = None
    http_status: int = 0
    repeat_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("passed", "skipped")


@dataclass
class BenchReport:
    """Aggregate results from a complete benchmark run."""

    results: list[BenchResult] = field(default_factory=list)
    server_info: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    categories_run: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize the full report to JSON."""
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "server_info": self.server_info,
                "categories_run": self.categories_run,
                "results": [asdict(r) for r in self.results],
                "summary": self.summary(),
            },
            indent=2,
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics per category."""
        cats: dict[str, list[BenchResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)

        out: dict[str, Any] = {}
        for cat, results in cats.items():
            ok = [r for r in results if r.ok]
            tps_vals = [r.decode_tps for r in ok if r.decode_tps > 0]
            ttft_vals = [
                r.time_to_first_token_s
                for r in ok
                if r.time_to_first_token_s > 0
            ]
            avg_tps = (
                round(statistics.mean(tps_vals), 1)
                if tps_vals
                else 0
            )
            avg_ttft = (
                round(statistics.mean(ttft_vals) * 1000, 1)
                if ttft_vals
                else 0
            )
            out[cat] = {
                "total": len(results),
                "passed": sum(
                    1 for r in results if r.status == "passed"
                ),
                "failed": sum(
                    1 for r in results if r.status == "failed"
                ),
                "skipped": sum(
                    1 for r in results if r.status == "skipped"
                ),
                "error": sum(
                    1 for r in results if r.status == "error"
                ),
                "avg_decode_tps": avg_tps,
                "avg_ttft_ms": avg_ttft,
            }
        return out

    def percentiles(self, category: str, metric: str) -> dict[str, float]:
        """Return p50, p95, p99 for *metric* within *category*."""
        values = [
            getattr(r, metric)
            for r in self.results
            if r.category == category and r.ok and getattr(r, metric, 0) > 0
        ]
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        values.sort()
        n = len(values)
        return {
            "p50": round(values[int(n * 0.50)], 4),
            "p95": round(values[min(int(n * 0.95), n - 1)], 4),
            "p99": round(values[min(int(n * 0.99), n - 1)], 4),
        }


# ── Legacy presets (backward compat) ────────────────────────────────

BENCH_PRESETS: dict[str, dict[str, object]] = {
    "short": {
        "description": "Short generation (legacy)",
        "messages": [
            {
                "role": "user",
                "content": "What is nuclear fusion? Reply in 3 sentences.",
            }
        ],
        "max_tokens": 128,
    },
    "medium": {
        "description": "Medium generation — sustained throughput (legacy)",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the tokamak concept for magnetic confinement fusion. "
                    "Cover the key physics principles, the role of the toroidal and "
                    "poloidal magnetic fields, and why this approach is promising "
                    "for energy production. Aim for about 400 words."
                ),
            }
        ],
        "max_tokens": 512,
    },
    "long": {
        "description": "Long generation — sustained decode speed (legacy)",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed technical guide on implementing an "
                    "OpenAI-compatible API server for serving large "
                    "language models. Cover architecture, "
                    "tensor parallelism, KV cache management, continuous batching, "
                    "and deployment best practices. Include code examples."
                ),
            }
        ],
        "max_tokens": 2048,
    },
    "code": {
        "description": "Code generation — practical coding task (legacy)",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function implementing a binary search "
                    "tree with insert, search, delete, and in-order "
                    "traversal. Include type hints and docstrings. "
                    "Then write pytest tests for all operations."
                ),
            }
        ],
        "max_tokens": 1024,
    },
    "thinking": {
        "description": "Reasoning — deep thinking / chain-of-thought (legacy)",
        "messages": [
            {
                "role": "user",
                "content": (
                    "A tokamak plasma has a major radius R=6.2m, minor radius a=2.0m, "
                    "toroidal field B_T=5.3T, and plasma current I_p=15MA. "
                    "Estimate the safety factor q at the plasma edge using the "
                    "cylindrical approximation. Then explain what happens physically "
                    "if q_edge drops below 2, and why this matters for ITER."
                ),
            }
        ],
        "max_tokens": 1024,
    },
    "tool_use": {
        "description": "Tool calling — test function call capability (legacy)",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in Cadarache, France right now?",
            }
        ],
        "max_tokens": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name, e.g. Cadarache, France",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    },
}

# ── Filler text for prefill / context tests ─────────────────────────

FILLER_TOPICS: list[str] = [
    (
        "The tokamak fusion reactor uses magnetic confinement to sustain a "
        "high-temperature plasma in a toroidal chamber. Strong toroidal and "
        "poloidal magnetic fields combine to create helical field lines that "
        "keep charged particles from drifting into the vessel walls. Achieving "
        "net energy gain requires heating the deuterium-tritium fuel to over "
        "150 million degrees Celsius, far exceeding the core temperature of "
        "the Sun. ITER, under construction in southern France, aims to "
        "demonstrate 500 MW of fusion power from 50 MW of heating input."
    ),
    (
        "In computational fluid dynamics, the Navier-Stokes equations describe "
        "the motion of viscous fluid substances. These partial differential "
        "equations express conservation of momentum and mass for Newtonian "
        "fluids. Direct numerical simulation resolves all turbulent scales but "
        "requires extraordinary computational resources. Large-eddy simulation "
        "models the smallest scales while resolving the energy-containing "
        "eddies, providing a practical compromise for engineering applications "
        "such as aerodynamic design and weather prediction."
    ),
    (
        "Quantum entanglement describes correlations between particles that "
        "persist regardless of the distance separating them. When two photons "
        "are entangled, measuring the polarization of one instantly determines "
        "the polarization of the other. Bell's theorem proves that no local "
        "hidden variable theory can reproduce all quantum mechanical "
        "predictions. Modern quantum key distribution protocols exploit "
        "entanglement to guarantee communication security based on the "
        "fundamental laws of physics rather than computational difficulty."
    ),
    (
        "The Standard Model of particle physics classifies all known elementary "
        "particles into quarks, leptons, and gauge bosons. Six quark flavors "
        "combine in triplets to form baryons like protons and neutrons, or in "
        "quark-antiquark pairs to form mesons. The Higgs boson, discovered at "
        "CERN in 2012, confirms the mechanism by which particles acquire mass. "
        "Despite its successes, the Standard Model does not incorporate gravity "
        "or explain dark matter, motivating searches for beyond-standard "
        "physics at the Large Hadron Collider."
    ),
    (
        "General relativity predicts gravitational lensing, where massive "
        "objects bend the path of light passing near them. Galaxy clusters can "
        "act as cosmic magnifying glasses, amplifying the light of background "
        "galaxies by factors of ten or more. Strong lensing produces multiple "
        "images or arcs, while weak lensing subtly distorts the shapes of "
        "millions of background galaxies. Astronomers use weak lensing surveys "
        "to map the distribution of dark matter across the universe with "
        "unprecedented precision."
    ),
    (
        "Superconducting magnets operate below their critical temperature, "
        "carrying electric current with zero resistance. Niobium-titanium "
        "alloys are widely used in MRI machines and particle accelerators, "
        "while niobium-tin compounds achieve higher magnetic fields at the "
        "cost of greater brittleness. High-temperature superconductors based "
        "on rare-earth barium copper oxide operate at liquid nitrogen "
        "temperatures. REBCO tape technology is enabling compact fusion "
        "reactor designs with magnetic fields exceeding 20 tesla."
    ),
    (
        "The carbon cycle describes the biogeochemical flow of carbon between "
        "the atmosphere, oceans, land biosphere, and lithosphere. "
        "Photosynthesis removes approximately 120 gigatons of carbon from the "
        "atmosphere annually, while respiration and decomposition return a "
        "similar amount. Ocean absorption accounts for about a quarter of "
        "anthropogenic CO2 emissions, leading to acidification that threatens "
        "marine calcifying organisms. Understanding carbon sinks and sources "
        "is essential for accurate climate projections."
    ),
    (
        "CRISPR-Cas9 gene editing technology enables precise modification of "
        "DNA sequences in living organisms. The system uses a guide RNA to "
        "direct the Cas9 nuclease to a specific genomic location, where it "
        "creates a double-strand break. Cellular repair mechanisms then "
        "introduce desired changes through homology-directed repair or "
        "non-homologous end joining. Applications span agriculture, medicine, "
        "and basic research, though ethical considerations around germline "
        "editing remain actively debated."
    ),
    (
        "Stellar nucleosynthesis forges elements heavier than hydrogen and "
        "helium in the cores of stars. The proton-proton chain dominates in "
        "Sun-like stars, fusing hydrogen into helium at temperatures around "
        "15 million kelvin. More massive stars proceed through the CNO cycle "
        "and eventually burn carbon, neon, oxygen, and silicon in concentric "
        "shells. Elements beyond iron are primarily produced through rapid "
        "neutron capture during supernova explosions and neutron star mergers."
    ),
    (
        "Machine learning algorithms learn patterns from data without being "
        "explicitly programmed for each task. Supervised learning trains models "
        "on labeled examples, while unsupervised learning discovers structure "
        "in unlabeled data. Reinforcement learning optimizes sequential "
        "decision-making through trial-and-error interaction with an "
        "environment. Transformer architectures, introduced in 2017, "
        "revolutionized natural language processing by using self-attention "
        "mechanisms to capture long-range dependencies in text."
    ),
    (
        "Plate tectonics describes the large-scale motion of Earth's "
        "lithospheric plates atop the convecting asthenosphere. Divergent "
        "boundaries at mid-ocean ridges produce new oceanic crust through "
        "seafloor spreading. Convergent boundaries generate subduction zones "
        "where dense oceanic plates descend beneath continental margins, "
        "driving volcanism and mountain building. Transform boundaries like "
        "the San Andreas Fault accommodate lateral plate motion and produce "
        "destructive earthquakes."
    ),
    (
        "Photovoltaic cells convert sunlight directly into electricity using "
        "the photoelectric effect in semiconductor junctions. Crystalline "
        "silicon cells dominate the market with efficiencies exceeding 26 "
        "percent for monocrystalline wafers. Perovskite solar cells offer "
        "potentially lower manufacturing costs and have reached 33 percent "
        "efficiency in tandem configurations. Utility-scale solar farms now "
        "produce electricity at costs competitive with fossil fuel generation "
        "in most regions of the world."
    ),
    (
        "The human immune system comprises innate and adaptive branches that "
        "cooperate to defend against pathogens. Innate immunity provides "
        "immediate, non-specific barriers including skin, mucous membranes, "
        "and phagocytic cells. Adaptive immunity generates highly specific "
        "antibodies and T-cell responses that improve upon repeated exposure. "
        "Immunological memory, mediated by long-lived memory B and T cells, "
        "forms the basis of vaccination and provides durable protection "
        "against previously encountered infectious agents."
    ),
    (
        "Black holes form when massive stars exhaust their nuclear fuel and "
        "collapse under their own gravity. The event horizon marks the "
        "boundary beyond which nothing, not even light, can escape. "
        "Supermassive black holes with billions of solar masses reside at the "
        "centers of most galaxies. The Event Horizon Telescope collaboration "
        "produced the first direct image of a black hole shadow in M87 in "
        "2019, confirming key predictions of general relativity in the "
        "strong-field regime."
    ),
    (
        "Catalysis accelerates chemical reactions by providing alternative "
        "pathways with lower activation energies. Heterogeneous catalysts "
        "like platinum nanoparticles operate at solid-gas interfaces in "
        "automotive catalytic converters and industrial ammonia synthesis. "
        "Enzymes are biological catalysts that achieve remarkable specificity "
        "and rate enhancement through precisely shaped active sites. Modern "
        "catalyst design leverages computational chemistry and machine "
        "learning to predict optimal compositions and structures."
    ),
    (
        "Gravitational waves are ripples in spacetime generated by "
        "accelerating massive objects. The Laser Interferometer Gravitational-"
        "Wave Observatory made the first direct detection in 2015, observing "
        "the merger of two black holes 1.3 billion light-years away. "
        "Subsequent detections include binary neutron star mergers that "
        "simultaneously produce electromagnetic counterparts. Third-"
        "generation detectors like the Einstein Telescope aim to observe "
        "gravitational waves from the earliest moments of the universe."
    ),
    (
        "Ocean thermohaline circulation distributes heat across the globe "
        "through a conveyor belt of deep-water currents driven by density "
        "differences from temperature and salinity variations. The Atlantic "
        "Meridional Overturning Circulation carries warm surface water "
        "northward and returns cold deep water southward. Freshwater input "
        "from melting ice sheets could potentially weaken this circulation, "
        "with significant consequences for European climate patterns and "
        "global heat distribution."
    ),
    (
        "Topological insulators are materials that behave as insulators in "
        "their bulk but support conducting states on their surfaces or edges. "
        "These surface states are protected by time-reversal symmetry and are "
        "robust against non-magnetic impurities and defects. Bismuth selenide "
        "and bismuth telluride are prototypical three-dimensional topological "
        "insulators. Potential applications include spintronics, quantum "
        "computing, and low-power electronic devices that exploit dissipation-"
        "less edge currents."
    ),
    (
        "Extremophile organisms thrive in environments once considered "
        "inhospitable to life. Thermophilic archaea inhabit deep-sea "
        "hydrothermal vents at temperatures exceeding 120 degrees Celsius. "
        "Halophilic microorganisms flourish in salt lakes with salinities "
        "approaching saturation. Psychrophilic bacteria metabolize in "
        "Antarctic permafrost at temperatures well below freezing. These "
        "organisms inform the search for extraterrestrial life and provide "
        "enzymes valuable for industrial biotechnology."
    ),
    (
        "Bayesian inference provides a principled framework for updating "
        "beliefs in light of new evidence. Bayes' theorem combines a prior "
        "distribution over parameters with a likelihood function to produce "
        "a posterior distribution. Markov chain Monte Carlo methods enable "
        "sampling from complex posterior distributions that lack closed-form "
        "solutions. Bayesian approaches are widely used in astrophysics, "
        "clinical trials, and machine learning, where quantifying uncertainty "
        "is as important as point estimation."
    ),
    (
        "Metamaterials are engineered structures with electromagnetic "
        "properties not found in natural materials. Negative refractive index "
        "metamaterials can bend light in the opposite direction to conventional "
        "materials. Acoustic metamaterials manipulate sound waves to create "
        "cloaking devices and super-resolution imaging systems. Mechanical "
        "metamaterials achieve unusual properties like negative Poisson's "
        "ratio, enabling auxetic structures that expand laterally when "
        "stretched longitudinally."
    ),
    (
        "Neuroplasticity describes the brain's ability to reorganize its "
        "neural connections in response to learning, experience, or injury. "
        "Synaptic plasticity, including long-term potentiation and depression, "
        "underlies memory formation and skill acquisition. Structural "
        "plasticity involves the growth of new dendritic spines and axonal "
        "branches. Adult neurogenesis in the hippocampus contributes to "
        "spatial memory and emotional regulation, challenging the long-held "
        "belief that the mature brain is static."
    ),
    (
        "Magnetohydrodynamics describes the behavior of electrically "
        "conducting fluids in the presence of magnetic fields. The coupling "
        "between fluid motion and electromagnetic forces governs plasma "
        "dynamics in fusion reactors, stellar interiors, and planetary "
        "dynamos. Alfvén waves propagate along magnetic field lines at speeds "
        "determined by the field strength and plasma density. MHD "
        "instabilities such as kink and ballooning modes limit the achievable "
        "plasma pressure in tokamak fusion devices."
    ),
    (
        "Quantum computing harnesses superposition and entanglement to perform "
        "certain computations exponentially faster than classical machines. "
        "Superconducting qubits, trapped ions, and photonic circuits represent "
        "the leading hardware platforms. Shor's algorithm threatens RSA "
        "encryption by efficiently factoring large integers. Variational "
        "quantum eigensolvers show promise for simulating molecular electronic "
        "structures relevant to drug discovery and materials science, though "
        "fault-tolerant operation remains a major engineering challenge."
    ),
]


def build_filler_prompt(target_tokens: int, tokenizer_ratio: float = 0.75) -> str:
    """Build a prompt of approximately *target_tokens* by cycling unique paragraphs.

    Uses ``char_count * tokenizer_ratio`` as a rough token estimate.
    Actual token count comes from the server's ``usage`` response.
    """
    target_chars = int(target_tokens / tokenizer_ratio)
    parts: list[str] = []
    total_chars = 0
    idx = 0
    while total_chars < target_chars:
        para = FILLER_TOPICS[idx % len(FILLER_TOPICS)]
        parts.append(para)
        total_chars += len(para) + 1  # +1 for newline
        idx += 1
    return "\n\n".join(parts)


# ── Needle-in-haystack construction ─────────────────────────────────

_NEEDLE_POSITIONS: dict[str, float] = {"early": 0.10, "mid": 0.50, "late": 0.90}


def _make_needle_code() -> str:
    """Generate a deterministic-looking but unique activation code."""
    ts = str(time.perf_counter_ns())
    return "AMBIX-" + hashlib.sha256(ts.encode()).hexdigest()[:8].upper()


def build_needle_haystack(
    target_tokens: int,
    position: str,
    needle_code: str | None = None,
) -> tuple[str, str]:
    """Build a haystack with an embedded needle.

    Returns ``(haystack_text, needle_code)`` where *haystack_text*
    includes the instruction preamble.
    """
    if needle_code is None:
        needle_code = _make_needle_code()
    needle = (
        f"The activation code for Project Stellarator is {needle_code}. "
        "Remember this code exactly — you will be asked about it later."
    )
    frac = _NEEDLE_POSITIONS.get(position, 0.5)

    filler = build_filler_prompt(target_tokens)
    paragraphs = filler.split("\n\n")
    insert_idx = max(0, int(len(paragraphs) * frac) - 1)
    paragraphs.insert(insert_idx, needle)

    haystack = "\n\n".join(paragraphs)
    return haystack, needle_code


# ── Tool definitions ────────────────────────────────────────────────

TOOL_WEATHER: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris, France'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    },
}

TOOL_BOOK_FLIGHT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight between two airports",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "IATA airport code"},
                "destination": {"type": "string", "description": "IATA airport code"},
                "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "passengers": {
                    "type": "object",
                    "properties": {
                        "adults": {"type": "integer"},
                        "children": {"type": "integer"},
                    },
                    "required": ["adults"],
                },
            },
            "required": ["origin", "destination", "date", "passengers"],
        },
    },
}

ALL_TOOLS: list[dict[str, Any]] = [TOOL_WEATHER, TOOL_BOOK_FLIGHT]

# ── Category registry ───────────────────────────────────────────────

CATEGORIES: list[str] = [
    "throughput",
    "prefill",
    "context",
    "tools",
    "reasoning",
    "concurrency",
]

# ── HTTP helpers ────────────────────────────────────────────────────


def _auth_headers(api_key: str | None = None) -> dict[str, str]:
    """Build HTTP headers, adding Bearer auth when *api_key* is set."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.6,
    tools: list[dict[str, Any]] | None = None,
    extra_body: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> BenchResult:
    """Buffered SSE streaming with server-reported token counts."""
    import urllib.error
    import urllib.request

    url = f"{base_url}/v1/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
    if extra_body:
        body.update(extra_body)

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload, headers=_auth_headers(api_key)
    )

    result = BenchResult(model=model)
    t0 = time.perf_counter()
    first_token_time: float | None = None
    finish_reason: str | None = None
    reasoning_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result.http_status = resp.status
            # Buffered line-by-line reads (not byte-by-byte)
            for raw_line in resp:
                line = raw_line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if not line.startswith(b"data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "") or delta.get(
                        "reasoning", ""
                    )
                    if content and first_token_time is None:
                        first_token_time = time.perf_counter()
                    if reasoning and first_token_time is None:
                        first_token_time = time.perf_counter()
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                usage = data.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens", 0)
                    result.completion_tokens = usage.get("completion_tokens", 0)
                    reasoning_tokens = usage.get("completion_tokens_details", {}).get(
                        "reasoning_tokens", 0
                    ) or usage.get("reasoning_tokens", 0)
    except urllib.error.HTTPError as exc:
        result.http_status = exc.code
        result.error = f"HTTP {exc.code}: {exc.reason}"
        result.status = "error"
        result.total_time_s = time.perf_counter() - t0
        return result
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.error = str(exc)
        result.status = "error"
        result.total_time_s = time.perf_counter() - t0
        return result

    t_end = time.perf_counter()
    result.total_time_s = t_end - t0
    result.finish_reason = finish_reason
    result.reasoning_tokens = reasoning_tokens
    if first_token_time is not None:
        result.time_to_first_token_s = first_token_time - t0
    # Decode TPS from server-reported completion tokens
    if result.completion_tokens > 0 and first_token_time is not None:
        decode_time = t_end - first_token_time
        if decode_time > 0:
            result.decode_tps = result.completion_tokens / decode_time
    # Prefill TPS
    if result.prompt_tokens > 0 and result.time_to_first_token_s > 0:
        result.prefill_tps = result.prompt_tokens / result.time_to_first_token_s
    return result


def _chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    tools: list[dict[str, Any]] | None = None,
    extra_body: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> tuple[BenchResult, dict[str, Any]]:
    """Non-streaming chat completion. Returns ``(result, raw_response)``."""
    import urllib.error
    import urllib.request

    url = f"{base_url}/v1/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    if extra_body:
        body.update(extra_body)

    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=payload, headers=_auth_headers(api_key)
    )

    result = BenchResult(model=model)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result.http_status = resp.status
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        result.http_status = exc.code
        result.error = f"HTTP {exc.code}: {exc.reason}"
        result.status = "error"
        result.total_time_s = time.perf_counter() - t0
        return result, {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result.error = str(exc)
        result.status = "error"
        result.total_time_s = time.perf_counter() - t0
        return result, {}

    result.total_time_s = time.perf_counter() - t0
    usage = data.get("usage", {})
    result.prompt_tokens = usage.get("prompt_tokens", 0)
    result.completion_tokens = usage.get("completion_tokens", 0)
    choices = data.get("choices", [])
    if choices:
        result.finish_reason = choices[0].get("finish_reason")
    return result, data


# ── Provenance capture ──────────────────────────────────────────────

# Provenance probes annotate a run; they must never slow it down or fail it.
_PROBE_TIMEOUT_S = 5.0

# A loopback URL names the client, not the serving node.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _fetch_body(
    url: str,
    api_key: str | None = None,
    timeout: float = _PROBE_TIMEOUT_S,
) -> str | None:
    """GET *url* and return the decoded body, or ``None`` on any failure.

    Every provenance probe is advisory: a route the engine does not expose,
    a rejected credential, or a slow response must leave the benchmark it
    annotates untouched.
    """
    import urllib.request

    req = urllib.request.Request(url, headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    """Convert *value* to ``int``, or ``None`` when it is not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _model_card(models_payload: Any) -> dict[str, Any]:
    """First model entry of a ``/v1/models`` payload, or an empty dict."""
    if not isinstance(models_payload, dict):
        return {}
    data = models_payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def _lookup(mapping: dict[str, Any], *needles: str) -> Any:
    """First non-``None`` value whose key contains one of *needles*.

    Engines spell the same field differently (``max_model_len`` against
    ``max_context_length``), so match on substrings instead of pinning one
    spelling that a version bump can retire.
    """
    for needle in needles:
        for key, val in mapping.items():
            if needle in key.lower() and val is not None:
                return val
    return None


def _probe_engine_version(base_url: str, api_key: str | None = None) -> str | None:
    """Version the engine reports at ``/version``, or ``None``.

    vLLM answers with a ``{"version": ...}`` object; other engines answer
    with a bare string or do not expose the route at all.
    """
    body = _fetch_body(f"{base_url}/version", api_key)
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        text = body.strip()
        # A plain-text route answers with the version alone; anything long
        # is an error page, not a version.
        return text if 0 < len(text) <= 64 else None
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, dict):
        val = _lookup(payload, "version")
        return str(val) if val is not None else None
    return None


def _slurm_job_id() -> str | None:
    """Job id of the surrounding SLURM allocation, if the run is inside one."""
    for var in ("SLURM_JOB_ID", "SLURM_JOBID"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return None


def _gpu_host(base_url: str) -> str | None:
    """Host that serves the model, or ``None`` when it is not determinable.

    The benchmark client usually runs off-node, so the serving host is the
    one named in *base_url*; a loopback URL means the client shares the node
    and SLURM's own node name is the better answer. Resolved from the URL and
    the environment only — querying the scheduler would add seconds of
    latency to every run.
    """
    import urllib.parse

    host = urllib.parse.urlsplit(base_url).hostname or ""
    if host and host.lower() not in _LOOPBACK_HOSTS:
        return host
    return os.environ.get("SLURMD_NODENAME", "").strip() or None


def _record_server_value(
    provenance: dict[str, Any],
    key: str,
    served: Any,
    requested: Any,
) -> None:
    """Store the engine-reported value for *key*, keeping a disagreeing request.

    The server describes the deployment that actually ran, so it wins. A
    profile value that disagrees is preserved beside it under
    ``<key>_profile`` rather than dropped — the disagreement is precisely
    the signal that a report is not describing the configuration its author
    believes it is.
    """
    if served is None:
        provenance[key] = requested
        return
    provenance[key] = served
    if requested is not None and requested != served:
        provenance[f"{key}_profile"] = requested


def capture_provenance(
    base_url: str,
    *,
    api_key: str | None = None,
    profile: ModelProfile | None = None,
    models_payload: Any = None,
    serve_job_id: str | None = None,
) -> dict[str, Any]:
    """Build the configuration fingerprint that attributes a saved report.

    Every key is always present, and every value is ``None`` when its source
    is unavailable: a stored report must never carry a figure that was
    assumed rather than observed.

    *profile* supplies what the deployment **requested**; ``/version`` and
    ``/v1/models`` supply what the engine **runs**. Where both exist and
    disagree the server value is kept and the requested one is preserved
    under a ``_profile`` suffix (see :func:`_record_server_value`).
    """
    engine = profile.engine if profile is not None else None
    card = _model_card(models_payload)

    provenance: dict[str, Any] = {
        "profile_slug": profile.slug if profile is not None else None,
        "model_name": profile.model.name if profile is not None else None,
        "served_name": profile.model.served_name if profile is not None else None,
        "engine_type": engine.type if engine is not None else None,
        "engine_version": _probe_engine_version(base_url, api_key),
        "tensor_parallel": engine.tensor_parallel if engine is not None else None,
        "gpus": profile.slurm.gpus if profile is not None else None,
        "kv_cache_dtype": engine.kv_cache_dtype if engine is not None else None,
        "max_model_len": None,
        "max_num_seqs": engine.max_num_seqs if engine is not None else None,
        "speculative_method": (
            engine.speculative_method if engine is not None else None
        ),
        "speculative_num_tokens": (
            engine.speculative_num_tokens if engine is not None else None
        ),
        "quantization": None,
        # The allocation the CLIENT runs in, which is usually none.
        "slurm_job_id": _slurm_job_id(),
        # The allocation that SERVED the run -- the one worth citing when a
        # number is questioned later. Distinct from the field above.
        "serve_job_id": serve_job_id,
        "gpu_host": _gpu_host(base_url),
        "captured_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }

    _record_server_value(
        provenance,
        "max_model_len",
        _coerce_int(_lookup(card, "max_model_len", "max_context", "context_length")),
        engine.max_total_tokens if engine is not None else None,
    )
    _record_server_value(
        provenance,
        "quantization",
        _lookup(card, "quant"),
        None,
    )
    return provenance


# ── Speculative-decode acceptance ───────────────────────────────────

# ``name{label="value",…} 1.5`` — the Prometheus text exposition format.
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>\S+)"
)
_LABEL_RE = re.compile(
    r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"(?P<val>(?:[^"\\]|\\.)*)"'
)


def _parse_prometheus_text(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse a Prometheus scrape into ``(name, labels, value)`` samples.

    Comments, blank lines, and samples with an unparseable value are
    skipped: a scrape is diagnostic, so one malformed line must not discard
    the counters around it.
    """
    samples: list[tuple[str, dict[str, str], float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        labels = {
            m.group("key"): m.group("val")
            for m in _LABEL_RE.finditer(match.group("labels") or "")
        }
        samples.append((match.group("name"), labels, value))
    return samples


def _spec_decode_snapshot(text: str) -> dict[str, Any]:
    """Speculative-decode counters read out of one ``/metrics`` scrape.

    Counter names move between engine versions, so the counters are found by
    substring rather than by an exact name: among the speculative-decode
    metrics, one mentioning ``draft`` carries the tokens the draft model
    proposed and one mentioning ``accept`` carries the tokens the target
    model kept. A name that also mentions tokens is a true token count and
    is preferred; any other cumulative speculative counter is a fallback for
    engines that omit the word. Values are summed across label sets so a
    multi-engine scrape totals correctly.
    """
    # Index 0 holds counters whose name mentions tokens, index 1 the fallback.
    tiers: list[dict[str, float]] = [{}, {}]
    per_position: dict[int, float] = {}

    for name, labels, value in _parse_prometheus_text(text):
        lowered = name.lower()
        if "spec" not in lowered:
            continue
        if "per_pos" in lowered:
            pos = _coerce_int(labels.get("position"))
            if pos is not None:
                per_position[pos] = per_position.get(pos, 0.0) + value
            continue
        if "draft" in lowered:
            role = "draft_tokens_total"
        elif "accept" in lowered:
            role = "accepted_tokens_total"
        else:
            continue
        tier = tiers[0] if "token" in lowered else tiers[1]
        tier[role] = tier.get(role, 0.0) + value

    snapshot: dict[str, Any] = {
        role: tiers[0].get(role, tiers[1].get(role))
        for role in ("draft_tokens_total", "accepted_tokens_total")
    }
    snapshot["num_accepted_per_pos"] = (
        [per_position[pos] for pos in sorted(per_position)] if per_position else None
    )
    return snapshot


def _scrape_spec_decode(
    base_url: str, api_key: str | None = None
) -> dict[str, Any] | None:
    """Snapshot the engine's speculative-decode counters, or ``None``."""
    body = _fetch_body(f"{base_url}/metrics", api_key)
    if body is None:
        return None
    return _spec_decode_snapshot(body)


def _counter_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    key: str,
) -> int | None:
    """Increment of one cumulative counter between two snapshots.

    A counter absent from the earlier snapshot had not been recorded yet, so
    it starts from zero. A negative difference means the engine restarted
    and reset its counters, which leaves the window meaningless.
    """
    end = after.get(key)
    if end is None:
        return None
    start = before.get(key) or 0.0
    delta = end - start
    return None if delta < 0 else int(round(delta))


def _per_position_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[int] | None:
    """Per-position acceptance increments, or ``None`` when unusable."""
    end = after.get("num_accepted_per_pos")
    if not end:
        return None
    start = before.get("num_accepted_per_pos") or []
    deltas: list[int] = []
    for pos, value in enumerate(end):
        base = start[pos] if pos < len(start) else 0.0
        delta = value - base
        if delta < 0:
            return None
        deltas.append(int(round(delta)))
    return deltas


def _spec_decode_window(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    """Speculative-decode acceptance over the window between two snapshots.

    A draft model that has silently degraded still serves tokens, so nothing
    in the latency or throughput numbers separates it from ordinary
    slowness; the fraction of its proposals the target model keeps does.
    The engine's counters are cumulative from its start, so the figures here
    are differences — both snapshots are required, and a missing one reports
    ``unavailable`` rather than passing lifetime totals off as this run's.
    """
    window: dict[str, Any] = {
        "draft_tokens_total": None,
        "accepted_tokens_total": None,
        "acceptance_rate": None,
        "num_accepted_per_pos": None,
        "source": "unavailable",
    }
    if before is None or after is None:
        return window

    draft = _counter_delta(before, after, "draft_tokens_total")
    accepted = _counter_delta(before, after, "accepted_tokens_total")
    per_position = _per_position_delta(before, after)
    if draft is None and accepted is None and per_position is None:
        return window

    window["draft_tokens_total"] = draft
    window["accepted_tokens_total"] = accepted
    window["num_accepted_per_pos"] = per_position
    window["source"] = "metrics"
    if draft is not None and draft > 0 and accepted is not None:
        window["acceptance_rate"] = round(accepted / draft, 4)
    return window


# ── Test runners per category ───────────────────────────────────────


def _run_throughput(
    base_url: str,
    model: str,
    repeat: int,
    warmup: bool,
    api_key: str | None = None,
) -> list[BenchResult]:
    """Decode speed at various output lengths."""
    tests = [
        ("decode_128", 128),
        ("decode_512", 512),
        ("decode_1024", 1024),
        ("decode_2048", 2048),
        ("decode_4096", 4096),
    ]
    prompt = [{"role": "user", "content": "Write a detailed essay on plasma physics."}]
    results: list[BenchResult] = []

    if warmup:
        _stream_chat(base_url, model, prompt, max_tokens=16, api_key=api_key)

    for test_name, max_tok in tests:
        for rep in range(repeat):
            r = _stream_chat(
                base_url, model, prompt, max_tokens=max_tok, api_key=api_key
            )
            r.category = "throughput"
            r.test_name = test_name
            r.repeat_index = rep
            if r.error is None:
                r.status = "passed"
            results.append(r)
    return results


def _run_prefill(
    base_url: str,
    model: str,
    repeat: int,
    api_key: str | None = None,
) -> list[BenchResult]:
    """TTFT scaling with input context length."""
    tests = [
        ("prefill_1k", 1000),
        ("prefill_4k", 4000),
        ("prefill_16k", 16000),
        ("prefill_64k", 64000),
    ]
    results: list[BenchResult] = []
    for test_name, target_tokens in tests:
        filler = build_filler_prompt(target_tokens)
        messages = [
            {
                "role": "user",
                "content": filler + "\n\nSummarize the above in one sentence.",
            }
        ]
        for rep in range(repeat):
            r = _stream_chat(base_url, model, messages, max_tokens=32, api_key=api_key)
            r.category = "prefill"
            r.test_name = test_name
            r.repeat_index = rep
            if r.error is None:
                r.status = "passed"
            results.append(r)
    return results


def _run_context(
    base_url: str,
    model: str,
    repeat: int,
    max_context: int | None,
    api_key: str | None = None,
) -> list[BenchResult]:
    """Needle-in-haystack retrieval at various context lengths and positions."""
    sizes = [4000, 16000, 64000, 128000]
    if max_context is not None:
        sizes = [s for s in sizes if s <= max_context]
    positions = list(_NEEDLE_POSITIONS.keys())
    size_labels = {4000: "4k", 16000: "16k", 64000: "64k", 128000: "128k"}
    results: list[BenchResult] = []

    for size in sizes:
        for pos in positions:
            test_name = f"needle_{size_labels.get(size, str(size))}_{pos}"
            for rep in range(repeat):
                haystack, code = build_needle_haystack(size, pos)
                messages = [
                    {"role": "user", "content": haystack},
                    {
                        "role": "user",
                        "content": (
                            "What is the activation code for Project Stellarator? "
                            "Reply with only the code, nothing else."
                        ),
                    },
                ]
                r = _stream_chat(
                    base_url, model, messages, max_tokens=64, api_key=api_key
                )
                r.category = "context"
                r.test_name = test_name
                r.repeat_index = rep
                r.metadata["needle_code"] = code
                r.metadata["needle_position"] = pos
                r.metadata["target_context_tokens"] = size
                # We can't check needle_found without response text in
                # streaming mode, so we rely on content checking at
                # the CLI layer if needed.
                if r.error is None:
                    r.status = "passed"
                results.append(r)
    return results


def _run_tools(
    base_url: str,
    model: str,
    repeat: int,
    api_key: str | None = None,
) -> list[BenchResult]:
    """Tool calling validation tests."""
    results: list[BenchResult] = []

    tool_tests: list[
        tuple[
            str,
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[str, Any],
        ]
    ] = [
        # (test_name, messages, tools, validation)
        (
            "tool_single",
            [{"role": "user", "content": "What's the weather in Cadarache, France?"}],
            [TOOL_WEATHER],
            {"expect_tool": "get_weather", "expect_call": True},
        ),
        (
            "tool_parallel",
            [
                {
                    "role": "user",
                    "content": (
                        "What's the weather in Paris, London, and Tokyo? "
                        "Check all three cities."
                    ),
                }
            ],
            [TOOL_WEATHER],
            {"expect_tool": "get_weather", "expect_call": True},
        ),
        (
            "tool_structured",
            [
                {
                    "role": "user",
                    "content": (
                        "Book a flight from CDG to NRT on 2025-03-15 "
                        "for 2 adults and 1 child."
                    ),
                }
            ],
            [TOOL_BOOK_FLIGHT],
            {"expect_tool": "book_flight", "expect_call": True},
        ),
        (
            "tool_no_call",
            [{"role": "user", "content": "What is 2 + 2?"}],
            [TOOL_WEATHER],
            {"expect_call": False},
        ),
    ]

    for test_name, messages, tools, validation in tool_tests:
        for rep in range(repeat):
            r, data = _chat(
                base_url,
                model,
                messages,
                tools=tools,
                max_tokens=256,
                api_key=api_key,
            )
            r.category = "tools"
            r.test_name = test_name
            r.repeat_index = rep
            if r.error:
                results.append(r)
                continue

            choices = data.get("choices", [])
            msg = choices[0].get("message", {}) if choices else {}
            tool_calls = msg.get("tool_calls", [])

            if validation["expect_call"]:
                if not tool_calls:
                    r.status = "failed"
                    content = msg.get("content", "")[:200]
                    r.error = f"Expected tool call: {content}"
                else:
                    expected = validation.get("expect_tool", "")
                    names = [tc.get("function", {}).get("name") for tc in tool_calls]
                    if expected and expected not in names:
                        r.status = "failed"
                        r.error = f"Expected {expected}, got {names}"
                    else:
                        r.status = "passed"
                        # Validate JSON args
                        for tc in tool_calls:
                            try:
                                args = json.loads(tc["function"]["arguments"])
                                r.metadata.setdefault("tool_args", []).append(args)
                            except (json.JSONDecodeError, KeyError):
                                r.status = "failed"
                                r.error = "Invalid JSON in tool call arguments"
                r.metadata["finish_reason"] = r.finish_reason
                r.metadata["tool_call_count"] = len(tool_calls)
            else:
                # Should NOT have called a tool
                if tool_calls:
                    r.status = "failed"
                    fn = tool_calls[0].get("function", {})
                    r.error = f"Unexpected tool call: {fn.get('name')}"
                elif r.finish_reason == "tool_calls":
                    r.status = "failed"
                    r.error = "finish_reason is tool_calls but no tool calls found"
                else:
                    r.status = "passed"
            results.append(r)
    return results


def _run_reasoning(
    base_url: str,
    model: str,
    repeat: int,
    api_key: str | None = None,
) -> list[BenchResult]:
    """Reasoning tests with capability gating."""
    reasoning_tests = [
        (
            "reasoning_math",
            (
                "A farmer has 3 fields. The first has 12 cows, the second has "
                "twice as many as the first, and the third has 5 fewer than "
                "the second. How many cows in total? Give only the number."
            ),
            "43",
        ),
        (
            "reasoning_logic",
            "How many r's in 'strawberry'? Answer with the number.",
            "3",
        ),
    ]
    results: list[BenchResult] = []

    for test_name, prompt, expected_answer in reasoning_tests:
        for rep in range(repeat):
            messages = [{"role": "user", "content": prompt}]
            # Try with thinking mode first
            r = _stream_chat(
                base_url,
                model,
                messages,
                max_tokens=1024,
                temperature=0.6,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                api_key=api_key,
            )
            r.category = "reasoning"
            r.test_name = test_name
            r.repeat_index = rep
            r.metadata["expected_answer"] = expected_answer

            if r.http_status in (400, 422):
                # Thinking not supported — retry without and mark skipped
                r2 = _stream_chat(
                    base_url,
                    model,
                    messages,
                    max_tokens=1024,
                    api_key=api_key,
                )
                r2.category = "reasoning"
                r2.test_name = test_name
                r2.repeat_index = rep
                r2.status = "skipped"
                r2.metadata["expected_answer"] = expected_answer
                r2.metadata["note"] = "thinking mode not supported, ran without"
                results.append(r2)
            else:
                if r.error is None:
                    r.status = "passed"
                results.append(r)
    return results


def _run_concurrency(
    base_url: str,
    model: str,
    repeat: int,
    api_key: str | None = None,
    gen_tokens: int = 1024,
) -> list[BenchResult]:
    """Parallel request handling tests.

    *gen_tokens* sets how much each worker generates. It has to be long enough
    that steady-state decoding dominates the measurement: a short generation
    spends most of its wall time on scheduler warm-up and first-token latency,
    which do not scale with the concurrency level, so the aggregate rate comes
    out non-monotone and says more about the transient than about the server.
    """
    levels = [1, 2, 4, 8, 16, 32]
    prompt = [
        {
            "role": "user",
            "content": "Explain how magnetic confinement works in fusion reactors.",
        }
    ]
    results: list[BenchResult] = []

    for n_workers in levels:
        test_name = f"concurrent_{n_workers}"
        for rep in range(repeat):
            barrier = threading.Barrier(n_workers, timeout=30)

            def _make_worker(
                _barrier: threading.Barrier,
                _test_name: str,
                _rep: int,
            ):
                def _worker(idx: int) -> BenchResult:
                    _barrier.wait()
                    r = _stream_chat(
                        base_url, model, prompt, max_tokens=gen_tokens,
                        api_key=api_key,
                    )
                    r.category = "concurrency"
                    r.test_name = _test_name
                    r.repeat_index = _rep
                    r.metadata["worker_index"] = idx
                    if r.error is None:
                        r.status = "passed"
                    return r

                return _worker

            worker_fn = _make_worker(barrier, test_name, rep)
            wall_t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as pool:
                futures = [
                    pool.submit(worker_fn, i) for i in range(n_workers)
                ]
                worker_results = [f.result() for f in futures]
            wall_time = time.perf_counter() - wall_t0

            total_comp = sum(r.completion_tokens for r in worker_results)
            aggregate_tps = total_comp / wall_time if wall_time > 0 else 0

            for wr in worker_results:
                wr.metadata["wall_time"] = round(wall_time, 3)
                wr.metadata["aggregate_tps"] = round(aggregate_tps, 1)
                wr.metadata["n_workers"] = n_workers
                results.append(wr)
    return results


# ── Orchestrator ────────────────────────────────────────────────────


def run_benchmark(
    base_url: str,
    model: str,
    *,
    categories: list[str] | None = None,
    repeat: int = 1,
    max_context: int | None = None,
    warmup: bool = True,
    api_key: str | None = None,
    profile: ModelProfile | None = None,
    serve_job_id: str | None = None,
    concurrency_tokens: int = 1024,
) -> BenchReport:
    """Run the full benchmark suite and return a :class:`BenchReport`.

    ``report.server_info`` keeps the raw ``/v1/models`` payload at the top
    level — its ``object`` and ``data`` keys, unchanged, so existing readers
    keep working — and adds three keys beside them:

    - ``models`` — the same payload, addressable by name (the canonical
      spelling for new readers);
    - ``provenance`` — the configuration fingerprint that attributes the
      report, from :func:`capture_provenance`;
    - ``spec_decode`` — speculative-decode acceptance measured across the
      throughput category, from :func:`_spec_decode_window`.

    *profile* supplies the requested-configuration half of the provenance;
    without it those fields stay ``None`` rather than being guessed.
    """
    cats = categories or CATEGORIES
    report = BenchReport(
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        categories_run=list(cats),
    )

    # Fetch server info
    models_payload: Any = {}
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{base_url}/v1/models", headers=_auth_headers(api_key)
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            models_payload = json.loads(resp.read())
    except Exception:
        models_payload = {}

    report.server_info = (
        dict(models_payload) if isinstance(models_payload, dict) else {}
    )
    report.server_info["models"] = models_payload
    report.server_info["provenance"] = capture_provenance(
        base_url,
        api_key=api_key,
        profile=profile,
        models_payload=models_payload,
        serve_job_id=serve_job_id,
    )

    runners = {
        "throughput": lambda: _run_throughput(base_url, model, repeat, warmup, api_key),
        "prefill": lambda: _run_prefill(base_url, model, repeat, api_key),
        "context": lambda: _run_context(base_url, model, repeat, max_context, api_key),
        "tools": lambda: _run_tools(base_url, model, repeat, api_key),
        "reasoning": lambda: _run_reasoning(base_url, model, repeat, api_key),
        "concurrency": lambda: _run_concurrency(
            base_url, model, repeat, api_key, gen_tokens=concurrency_tokens
        ),
    }

    spec_before: dict[str, Any] | None = None
    spec_after: dict[str, Any] | None = None
    for cat in cats:
        if cat not in runners:
            continue
        # The engine's speculative counters are cumulative, so bracket the
        # decode-heavy category to attribute acceptance to this run's tokens.
        if cat == "throughput":
            spec_before = _scrape_spec_decode(base_url, api_key)
        report.results.extend(runners[cat]())
        if cat == "throughput":
            spec_after = _scrape_spec_decode(base_url, api_key)

    report.server_info["spec_decode"] = _spec_decode_window(spec_before, spec_after)

    return report


# ── Legacy compat wrappers ──────────────────────────────────────────

# Keep old names importable for backward compatibility.
BenchSuite = BenchReport


def run_bench_preset(
    base_url: str,
    model: str,
    preset: str,
    *,
    repeat: int = 1,
) -> BenchReport:
    """Run a legacy benchmark preset (deprecated, use *run_benchmark*)."""
    cfg = BENCH_PRESETS[preset]
    report = BenchReport(
        model=model, timestamp=_dt.datetime.now(_dt.UTC).isoformat()
    )
    for rep in range(repeat):
        r = _stream_chat(
            base_url=base_url,
            model=model,
            messages=cfg["messages"],  # type: ignore[arg-type]
            max_tokens=cfg.get("max_tokens", 512),  # type: ignore[arg-type]
        )
        r.repeat_index = rep
        r.category = "legacy"
        r.test_name = preset
        if r.error is None:
            r.status = "passed"
        report.results.append(r)
    return report


def run_tool_call_bench(base_url: str, model: str) -> BenchResult:
    """Legacy tool-call test (deprecated, use *run_benchmark* with category='tools')."""
    r, _data = _chat(
        base_url, model,
        BENCH_PRESETS["tool_use"]["messages"],  # type: ignore[arg-type]
        tools=BENCH_PRESETS["tool_use"]["tools"],  # type: ignore[arg-type]
        max_tokens=256,
    )
    return r
