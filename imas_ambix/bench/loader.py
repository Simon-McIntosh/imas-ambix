"""YAML-based bench config loader.

Loads a bench spec YAML file into a :class:`~imas_ambix.bench.tokenizer.BenchConfig`
plus a ``run_kwargs`` dict that can be splatted into
``benchmark_frame_tokenizer(cfg, **run_kwargs)``.

Usage
-----
::

    from imas_ambix.bench.loader import load_bench_config, bundled_config

    cfg, run_kwargs = load_bench_config(bundled_config("v0-rir-25shot"))
    # result = benchmark_frame_tokenizer(cfg, **run_kwargs)  # needs GPU
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def bundled_config(name: str) -> Path:
    """Return the path to a bundled bench config YAML by stem name.

    Parameters
    ----------
    name:
        Config file stem, e.g. ``"v0-rir-25shot"`` (the ``.yaml`` suffix is
        added automatically).

    Returns
    -------
    Path
        Absolute path to ``imas_ambix/bench/configs/{name}.yaml``.
    """
    configs_dir = Path(__file__).parent / "configs"
    return configs_dir / f"{name}.yaml"


def load_bench_config(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a YAML bench spec into ``(BenchConfig, run_kwargs)``.

    The YAML schema is::

        name: <str>
        tokenizer_kind: frame | signal
        tokenizer:
          factory: "module.path:ClassName"   # resolved via importlib
          kwargs: {key: value, ...}           # passed to the class constructor
        max_items_per_shot: <int | null>
        metrics: [psnr, mae, ...]
        device: cpu | cuda
        # Any additional keys (e.g. camera, shot_ids) → run_kwargs

    Parameters
    ----------
    path:
        Path to a YAML file.  Raises :class:`FileNotFoundError` with a clear
        message when the file does not exist.

    Returns
    -------
    cfg : BenchConfig
        Populated benchmark configuration with a bound ``tokenizer_factory``
        callable that constructs the tokenizer when called with no arguments.
    run_kwargs : dict
        All remaining YAML keys (``camera``, ``shot_ids``, and any custom
        keys) for splatting into ``benchmark_frame_tokenizer(cfg, **run_kwargs)``.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for load_bench_config. "
            "Install it with:  uv pip install pyyaml"
        ) from exc

    from imas_ambix.bench.tokenizer import BenchConfig

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Bench config not found: {path}\n"
            f"Bundled configs live in imas_ambix/bench/configs/. "
            f"Use bundled_config('v0-rir-25shot') to get a valid path."
        )

    with path.open("r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    # --- Resolve tokenizer factory ---
    tok_section: dict[str, Any] = raw.pop("tokenizer", {})
    factory_str: str = tok_section.get("factory", "")
    tok_kwargs: dict[str, Any] = tok_section.get("kwargs", {})

    tokenizer_factory = _resolve_factory(factory_str, tok_kwargs)

    # --- Extract BenchConfig fields ---
    name: str = raw.pop("name")
    tokenizer_kind: str = raw.pop("tokenizer_kind")
    max_items_per_shot: int | None = raw.pop("max_items_per_shot", None)
    metrics_raw = raw.pop("metrics", ("psnr",))
    metrics: tuple[str, ...] = tuple(metrics_raw)
    device: str = raw.pop("device", "cpu")
    rfid_frames_per_shot: int = int(raw.pop("rfid_frames_per_shot", 32))

    cfg = BenchConfig(
        name=name,
        tokenizer_kind=tokenizer_kind,
        tokenizer_factory=tokenizer_factory,
        max_items_per_shot=max_items_per_shot,
        metrics=metrics,
        device=device,
        rfid_frames_per_shot=rfid_frames_per_shot,
    )

    # Everything left goes to run_kwargs (camera, shot_ids, etc.)
    run_kwargs: dict[str, Any] = dict(raw)

    return cfg, run_kwargs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_factory(factory_str: str, kwargs: dict[str, Any]):  # type: ignore[return]
    """Import the class named by ``"module:attr"`` and return a zero-arg lambda.

    Parameters
    ----------
    factory_str:
        A dotted import path of the form
        ``"imas_ambix.tokenizer.frames:OpenMagvit2Tokenizer"``.
    kwargs:
        Keyword arguments that will be forwarded to the class constructor.

    Returns
    -------
    callable
        A zero-argument callable ``() -> Tokenizer`` that instantiates the
        class with the provided kwargs when invoked.
    """
    if ":" not in factory_str:
        raise ValueError(
            f"tokenizer.factory must be 'module.path:ClassName', got: {factory_str!r}"
        )
    module_path, attr_name = factory_str.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, attr_name)
    bound_kwargs = dict(kwargs)
    return lambda: cls(**bound_kwargs)
