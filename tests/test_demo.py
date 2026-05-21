"""Tests for imas_ambix.demo (runner + CLI) and imas_ambix.eval.rollout.

Uses synthetic data and a mock WhamModel throughout — no real MAST data,
no trained checkpoints, no GPU required.

Coverage:
- RolloutConfig defaults / custom values
- rollout() shape contracts
- rollout() force_signal_action_tokens behaviour
- rollout() deterministic output under torch.manual_seed
- rollout() raises ValueError for None inputs
- run_demo() with checkpoint="mock" produces all 6 required artefacts
- run_demo() skips MP4 when no_video=True
- run_demo() metrics.json has expected keys
- CLI: ambix demo --help
- CLI: ambix demo wham-mast --help
- CLI: end-to-end wham-mast with mock checkpoint
- Metric table is printed in CLI output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from imas_ambix.cli import main
from imas_ambix.eval.rollout import RolloutConfig, rollout

# ---------------------------------------------------------------------------
# Mock WhamModel (mirrors runner._MockWhamModel but importable here)
# ---------------------------------------------------------------------------


class _MockModel:
    """Minimal WhamModel stub that returns uniform logits over a tiny vocab."""

    VOCAB: int = 128

    def forward(self, input_ids: torch.LongTensor, **_kwargs: Any) -> dict:
        b, seq_len = input_ids.shape
        logits = torch.zeros(b, seq_len, self.VOCAB, device=input_ids.device)
        return {"logits": logits, "loss": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prefix(b: int = 1, seq_len: int = 64, vocab: int = 128) -> torch.LongTensor:
    return torch.randint(0, vocab, (b, seq_len))


def _make_control(b: int = 1, steps: int = 4, k_ctrl: int = 10) -> torch.LongTensor:
    return torch.randint(0, 128, (b, steps * k_ctrl))


# ---------------------------------------------------------------------------
# RolloutConfig tests
# ---------------------------------------------------------------------------


def test_rollout_config_default_values() -> None:
    cfg = RolloutConfig()
    assert cfg.prefix_tokens == 2048
    assert cfg.rollout_steps == 100
    assert cfg.top_k == 64
    assert cfg.temperature == 0.8
    assert cfg.force_signal_action_tokens is True


def test_rollout_config_custom_values() -> None:
    cfg = RolloutConfig(prefix_tokens=512, rollout_steps=10, top_k=16, temperature=1.2)
    assert cfg.prefix_tokens == 512
    assert cfg.rollout_steps == 10
    assert cfg.top_k == 16
    assert cfg.temperature == 1.2


def test_rollout_config_is_frozen() -> None:
    cfg = RolloutConfig()
    with pytest.raises((TypeError, AttributeError)):
        cfg.top_k = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# rollout() shape tests
# ---------------------------------------------------------------------------


def test_rollout_returns_expected_keys() -> None:
    cfg = RolloutConfig(rollout_steps=2)
    k_frame, k_ctrl = 8, 5
    model = _MockModel()
    prefix = _make_prefix(seq_len=16)
    control = _make_control(steps=2, k_ctrl=k_ctrl)

    result = rollout(model, prefix, control, cfg, k_frame=k_frame, k_ctrl=k_ctrl)
    assert set(result.keys()) == {"tokens", "predicted_tokens", "log_probs"}


def test_rollout_tokens_shape() -> None:
    """tokens shape = (B, prefix_len + rollout_len)."""
    k_frame, k_ctrl = 8, 5
    steps = 3
    prefix_len = 16
    cfg = RolloutConfig(rollout_steps=steps)
    model = _MockModel()
    prefix = _make_prefix(b=1, seq_len=prefix_len)
    control = _make_control(b=1, steps=steps, k_ctrl=k_ctrl)

    result = rollout(model, prefix, control, cfg, k_frame=k_frame, k_ctrl=k_ctrl)

    expected_suffix = steps * (k_frame + k_ctrl)
    expected_total = prefix_len + expected_suffix
    assert result["tokens"].shape == (1, expected_total)


def test_rollout_predicted_tokens_shape() -> None:
    k_frame, k_ctrl = 6, 4
    steps = 2
    cfg = RolloutConfig(rollout_steps=steps)
    model = _MockModel()
    prefix = _make_prefix(b=2, seq_len=12)
    control = _make_control(b=2, steps=steps, k_ctrl=k_ctrl)

    result = rollout(model, prefix, control, cfg, k_frame=k_frame, k_ctrl=k_ctrl)

    expected_len = steps * (k_frame + k_ctrl)
    assert result["predicted_tokens"].shape == (2, expected_len)


def test_rollout_log_probs_shape() -> None:
    k_frame, k_ctrl = 4, 3
    steps = 2
    cfg = RolloutConfig(rollout_steps=steps)
    result = rollout(
        _MockModel(),
        _make_prefix(seq_len=8),
        _make_control(steps=steps, k_ctrl=k_ctrl),
        cfg,
        k_frame=k_frame,
        k_ctrl=k_ctrl,
    )
    # log_probs covers only sampled frame positions
    assert result["log_probs"].shape == (1, steps * k_frame)


# ---------------------------------------------------------------------------
# rollout() force_signal_action_tokens
# ---------------------------------------------------------------------------


def test_rollout_force_signal_action_tokens_exact_match() -> None:
    """Forced control positions in predicted_tokens must equal control_tokens."""
    k_frame, k_ctrl = 4, 5
    steps = 3
    cfg = RolloutConfig(rollout_steps=steps, force_signal_action_tokens=True)
    model = _MockModel()
    prefix = _make_prefix(seq_len=10)
    control = _make_control(steps=steps, k_ctrl=k_ctrl)

    result = rollout(model, prefix, control, cfg, k_frame=k_frame, k_ctrl=k_ctrl)
    predicted = result["predicted_tokens"][0]  # (steps*(k_frame+k_ctrl),)

    # Extract the control positions: after each k_frame block comes k_ctrl forced tokens
    for s in range(steps):
        offset = s * (k_frame + k_ctrl) + k_frame
        forced = predicted[offset : offset + k_ctrl]
        expected = control[0, s * k_ctrl : (s + 1) * k_ctrl]
        torch.testing.assert_close(forced, expected)


def test_rollout_frame_positions_not_forced() -> None:
    """Frame positions must not coincide with control token values (probabilistic)."""
    k_frame, k_ctrl = 4, 5
    steps = 2
    torch.manual_seed(42)

    # Use a model that produces highly non-uniform logits
    class _BiasedModel:
        def forward(self, input_ids: torch.LongTensor, **_kw: Any) -> dict:
            b, seq_len = input_ids.shape
            logits = torch.zeros(b, seq_len, 128)
            logits[:, :, 0] = 100.0  # strongly prefer token 0
            return {"logits": logits}

    model = _BiasedModel()
    cfg = RolloutConfig(rollout_steps=steps, force_signal_action_tokens=True, top_k=2)
    prefix = _make_prefix(seq_len=8)
    # Set control tokens to token id=127 (the biased model will never sample this)
    control = torch.full((1, steps * k_ctrl), 127, dtype=torch.long)

    result = rollout(model, prefix, control, cfg, k_frame=k_frame, k_ctrl=k_ctrl)
    predicted = result["predicted_tokens"][0]

    # Frame positions should be mostly 0 (model bias); control positions = 127
    for s in range(steps):
        frame_start = s * (k_frame + k_ctrl)
        frame_tokens = predicted[frame_start : frame_start + k_frame]
        ctrl_start = frame_start + k_frame
        ctrl_tokens = predicted[ctrl_start : ctrl_start + k_ctrl]

        # All control tokens must be exactly 127
        assert (ctrl_tokens == 127).all(), "Control tokens not forced correctly"
        # Frame tokens should not all be 127 (very unlikely under strong bias to 0)
        assert not (frame_tokens == 127).all(), "Frame tokens unexpectedly = 127"


# ---------------------------------------------------------------------------
# rollout() determinism under manual seed
# ---------------------------------------------------------------------------


def test_rollout_deterministic_under_seed() -> None:
    k_frame, k_ctrl = 4, 3
    steps = 2
    cfg = RolloutConfig(rollout_steps=steps, top_k=4, temperature=1.0)
    model = _MockModel()
    prefix = _make_prefix(seq_len=8)
    control = _make_control(steps=steps, k_ctrl=k_ctrl)

    torch.manual_seed(7)
    r1 = rollout(model, prefix, control, cfg, k_frame=k_frame, k_ctrl=k_ctrl)

    torch.manual_seed(7)
    r2 = rollout(model, prefix, control, cfg, k_frame=k_frame, k_ctrl=k_ctrl)

    torch.testing.assert_close(r1["predicted_tokens"], r2["predicted_tokens"])


# ---------------------------------------------------------------------------
# rollout() error handling
# ---------------------------------------------------------------------------


def test_rollout_raises_on_none_model() -> None:
    cfg = RolloutConfig(rollout_steps=1)
    with pytest.raises(ValueError, match="non-None"):
        rollout(None, _make_prefix(), _make_control(steps=1), cfg, k_frame=4, k_ctrl=3)


def test_rollout_raises_on_too_short_control() -> None:
    k_frame, k_ctrl = 4, 5
    steps = 3
    cfg = RolloutConfig(rollout_steps=steps)
    # Supply only 1 step worth of control instead of 3
    short_control = _make_control(steps=1, k_ctrl=k_ctrl)
    with pytest.raises(ValueError, match="control_tokens"):
        rollout(
            _MockModel(),
            _make_prefix(),
            short_control,
            cfg,
            k_frame=k_frame,
            k_ctrl=k_ctrl,
        )


# ---------------------------------------------------------------------------
# run_demo() artefact tests
# ---------------------------------------------------------------------------


def test_run_demo_mock_produces_all_artefacts(tmp_path: Path) -> None:
    """run_demo(..., checkpoint='mock') must create all 6 required artefacts."""
    from imas_ambix.demo.runner import run_demo

    artefacts = run_demo(
        shot_id=99999,
        checkpoint_path="mock",
        prefix_ms=50,
        rollout_ms=100,
        output_dir=tmp_path / "out",
        _k_frame=4,
        _k_ctrl=3,
    )

    assert artefacts.ground_truth_zarr.exists(), "ground-truth.zarr missing"
    assert artefacts.prediction_zarr.exists(), "prediction.zarr missing"
    assert artefacts.tokens_ground_truth_zarr.exists(), "tokens-gt.zarr missing"
    assert artefacts.tokens_prediction_zarr.exists(), "tokens-prediction.zarr missing"
    assert artefacts.metrics_json.exists(), "metrics.json missing"
    assert artefacts.figure_png.exists(), "figure.png missing"


def test_run_demo_metrics_json_has_expected_keys(tmp_path: Path) -> None:
    from imas_ambix.demo.runner import run_demo

    artefacts = run_demo(
        shot_id=99998,
        checkpoint_path="mock",
        prefix_ms=50,
        rollout_ms=100,
        output_dir=tmp_path / "out",
        _k_frame=4,
        _k_ctrl=3,
    )

    metrics = json.loads(artefacts.metrics_json.read_text())
    expected_keys = {
        "psnr",
        "lpips",
        "rfid",
        "centroid_mse",
        "chord_nrmse",
        "edge_displacement_mad",
    }
    assert set(metrics.keys()) == expected_keys


def test_run_demo_no_video_skips_mp4(tmp_path: Path) -> None:
    """no_video=True must leave video_mp4=None in artefacts."""
    from imas_ambix.demo.runner import run_demo

    artefacts = run_demo(
        shot_id=99997,
        checkpoint_path="mock",
        prefix_ms=50,
        rollout_ms=100,
        output_dir=tmp_path / "out",
        no_video=True,
        _k_frame=4,
        _k_ctrl=3,
    )

    assert artefacts.video_mp4 is None


def test_run_demo_with_persisted_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_demo uses load_shot_stream when data is present on disk."""
    import imas_ambix.demo.runner as runner_mod

    # Synthesise a token stream
    shot_id = 12345
    rng = np.random.default_rng(shot_id)
    n = 2000
    tokens_np = rng.integers(0, 128, size=n, dtype=np.int32)
    block_kind_np = np.zeros(n, dtype=np.uint8)

    # Patch the loader to return synthetic data
    monkeypatch.setattr(
        runner_mod,
        "_load_or_synthesise_stream",
        lambda shot_id, tokenizer_version, prefix_ms, rollout_ms: (
            tokens_np,
            block_kind_np,
        ),
    )

    from imas_ambix.demo.runner import run_demo

    artefacts = run_demo(
        shot_id=shot_id,
        checkpoint_path="mock",
        prefix_ms=50,
        rollout_ms=100,
        output_dir=tmp_path / "out",
        no_video=True,
        _k_frame=4,
        _k_ctrl=3,
    )
    assert artefacts.metrics_json.exists()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_ambix_demo_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--help"])
    assert result.exit_code == 0, result.output
    assert "demo" in result.output.lower()


def test_ambix_demo_wham_mast_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "wham-mast", "--help"])
    assert result.exit_code == 0, result.output
    assert "--shot" in result.output
    assert "--checkpoint" in result.output
    assert "--prefix-ms" in result.output
    assert "--rollout-ms" in result.output
    assert "--output" in result.output


def test_cli_demo_wham_mast_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI end-to-end: mock checkpoint + synthetic data produces artefacts."""
    import imas_ambix.demo.cli as cli_mod

    # Patch run_demo to use tiny block sizes for fast CI
    _orig_run_demo = cli_mod.run_demo if hasattr(cli_mod, "run_demo") else None

    import imas_ambix.demo.runner as runner_mod

    _real_run_demo = runner_mod.run_demo

    def _fast_run_demo(
        shot_id: int, checkpoint_path: object, **kwargs: object
    ) -> object:  # noqa: E501
        kwargs.setdefault("_k_frame", 4)
        kwargs.setdefault("_k_ctrl", 3)
        return _real_run_demo(shot_id, checkpoint_path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_mod, "run_demo", _fast_run_demo)
    # Also patch the symbol in cli module if it imported run_demo directly
    import imas_ambix.demo.cli as _cli

    if hasattr(_cli, "run_demo"):
        monkeypatch.setattr(_cli, "run_demo", _fast_run_demo)

    cli_runner = CliRunner()
    output_dir = str(tmp_path / "cli-out")

    result = cli_runner.invoke(
        main,
        [
            "demo",
            "wham-mast",
            "--shot",
            "99996",
            "--checkpoint",
            "mock",
            "--prefix-ms",
            "50",
            "--rollout-ms",
            "100",
            "--output",
            output_dir,
            "--no-video",
        ],
    )

    assert result.exit_code == 0, f"CLI failed:\n{result.output}"

    # All 5 mandatory artefacts must exist
    out = Path(output_dir)
    assert (out / "ground-truth.zarr").exists()
    assert (out / "prediction.zarr").exists()
    assert (out / "tokens-ground-truth.zarr").exists()
    assert (out / "tokens-prediction.zarr").exists()
    assert (out / "metrics.json").exists()
    assert (out / "demo-99996.png").exists()


def test_cli_demo_prints_metric_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI must print metric names in the output."""
    import imas_ambix.demo.runner as runner_mod

    _real_run_demo = runner_mod.run_demo

    def _fast_run_demo(
        shot_id: int, checkpoint_path: object, **kwargs: object
    ) -> object:  # noqa: E501
        kwargs.setdefault("_k_frame", 4)
        kwargs.setdefault("_k_ctrl", 3)
        return _real_run_demo(shot_id, checkpoint_path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_mod, "run_demo", _fast_run_demo)

    cli_runner = CliRunner()
    result = cli_runner.invoke(
        main,
        [
            "demo",
            "wham-mast",
            "--shot",
            "99995",
            "--checkpoint",
            "mock",
            "--prefix-ms",
            "50",
            "--rollout-ms",
            "100",
            "--output",
            str(tmp_path / "out"),
            "--no-video",
        ],
    )

    assert result.exit_code == 0, result.output
    # Metric names must appear in stdout
    assert "psnr" in result.output
    assert "centroid_mse" in result.output
    assert "chord_nrmse" in result.output
