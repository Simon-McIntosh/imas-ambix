"""Integration test: the manifest training path threads PER-WINDOW camera and
the corpus CLI can enable the Δt / camera / extended-signal conditioning.

The conditioning primitives (the Δt encoder, the camera embedding, the extended
signal streams, the ``collate_timescale_camera`` per-sample metadata) are pinned
by ``test_timescale_camera_conditioning``.  THIS file pins the LAST MILE that
makes them usable for a real multi-camera corpus run:

* **per-window camera is threaded** — :func:`manifest_train_windows` reads each
  window's ``camera_id`` (falling back to the reference camera when absent), and
  :class:`ManifestWindowDataset` assembles each window from THAT camera's tokens
  (not a single dataset-wide camera).  Consecutive samples from a mixed manifest
  carry DIFFERENT cameras + the right per-window horizon.
* **end-to-end forward/backward with conditioning ON** — a model built with
  ``timescale_conditioning`` + ``camera_conditioning`` + the extended signal
  streams runs one forward+backward on a multi-camera, multi-cadence batch
  assembled by the REAL collate; the loss is finite and gradients reach the Δt
  encoder + camera table (the conditioning is consumed, not dead).
* **real-data camera span** — the production unified manifest, read through
  :func:`manifest_train_windows`, spans all five cameras (a wrong ``camera_id``
  key would collapse the histogram to 100% reference camera).

The forward/backward and per-window-camera assertions are CPU-only (the smoke is
CPU); the real-data check is skipped when the production manifest is absent.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

import imas_ambix.worldmodel.controllable_train as ct
from imas_ambix.worldmodel.actuator_plan import (
    ACTUATOR_CHANNEL_KEYS,
    N_ACTUATOR_CHANNELS,
    ActuatorPlan,
    normalise_actuator_values,
)
from imas_ambix.worldmodel.controllable_dataset import (
    ControllableSpacetimeSample,
    extended_signal_modalities,
)
from imas_ambix.worldmodel.controllable_model import (
    ControllableSpacetimeConfig,
    ControllableSpacetimeTransformer,
)
from imas_ambix.worldmodel.controllable_train import (
    ManifestWindowDataset,
    manifest_train_windows,
)
from imas_ambix.worldmodel.spacetime_dataset import (
    REFERENCE_CAMERA,
    SpacetimeSample,
    SpacetimeWindowConfig,
)
from imas_ambix.worldmodel.spacetime_dataset_v2 import SignalSpacetimeSample
from imas_ambix.worldmodel.spacetime_model_v2 import SignalStreamSpec
from imas_ambix.worldmodel.timescale_conditioning import (
    CAMERA_IDS,
    REFERENCE_DT_SECONDS,
)

#: The production unified multi-camera manifest (skipped when not on the cluster).
UNIFIED_MANIFEST = Path(
    "/work/projects/imas_gpu/agents/excitation-corpus/curated_windows_unified.json"
)


# ---------------------------------------------------------------------------
# per-window camera is threaded through manifest_train_windows + the dataset
# ---------------------------------------------------------------------------


def _write_multicamera_manifest(path: Path) -> None:
    """A small synthetic unified manifest: >= 2 cameras, a fast + a slow window."""
    path.write_text(
        json.dumps(
            {
                "horizon_s": 0.25,
                "windows": [
                    # slow full-shot rco window (per-shot plasma_duration_s horizon)
                    {
                        "shot_id": 100,
                        "camera_id": "rco",
                        "start_frame": 4,
                        "fps": 500.0,
                        "plasma_duration_s": 0.30,
                        "timescale": "slow",
                    },
                    # fast rbb burst (global target horizon)
                    {
                        "shot_id": 200,
                        "camera_id": "rbb",
                        "start_frame": 9,
                        "fps": 2000.0,
                        "timescale": "fast",
                    },
                    # another camera again, so the per-window camera clearly varies
                    {
                        "shot_id": 300,
                        "camera_id": "rgb",
                        "start_frame": 2,
                        "fps": 1000.0,
                        "timescale": "slow",
                    },
                    # a window with NO camera_id -> back-compat: reference camera
                    {
                        "shot_id": 400,
                        "start_frame": 0,
                        "fps": 1500.0,
                        "timescale": "fast",
                    },
                ],
            }
        )
    )


def test_manifest_train_windows_reads_per_window_camera(tmp_path):
    manifest = tmp_path / "unified.json"
    _write_multicamera_manifest(manifest)
    ws = manifest_train_windows(
        manifest, held_out=set(), target_horizon_s=0.25, n_frames=24
    )
    by = {w.shot_id: w for w in ws}
    # each window carries ITS OWN camera (not a single dataset-wide camera).
    assert by[100].camera == "rco"
    assert by[200].camera == "rbb"
    assert by[300].camera == "rgb"
    # a window with no camera_id falls back to the reference camera (back-compat).
    assert by[400].camera == REFERENCE_CAMERA
    # the full-shot window keeps its own per-shot horizon; the rest the global one.
    assert by[100].horizon_s == pytest.approx(0.30)
    assert by[200].horizon_s == pytest.approx(0.25)


def test_single_camera_manifest_defaults_to_reference_camera(tmp_path):
    """A legacy single-camera manifest (no camera_id) yields all-reference windows."""
    manifest = tmp_path / "single.json"
    manifest.write_text(
        json.dumps(
            {
                "windows": [
                    {"shot_id": 1, "start_frame": 0, "fps": 2000.0},
                    {"shot_id": 2, "start_frame": 5, "fps": 2000.0},
                ]
            }
        )
    )
    ws = manifest_train_windows(
        manifest, held_out=set(), target_horizon_s=0.25, n_frames=24
    )
    assert {w.camera for w in ws} == {REFERENCE_CAMERA}


def test_manifest_window_dataset_assembles_with_per_window_camera(
    tmp_path, monkeypatch
):
    """ManifestWindowDataset passes EACH window's own camera to the assembler.

    Monkeypatches the heavy Zarr assembler (as the prior manifest tests do) and
    records the ``camera=`` kwarg per item, asserting consecutive samples carry
    DIFFERENT cameras + the right per-window horizon — i.e. the dataset really
    threads ``window.camera``, not a single ``self._camera``.
    """
    manifest = tmp_path / "unified.json"
    _write_multicamera_manifest(manifest)
    windows = manifest_train_windows(
        manifest, held_out=set(), target_horizon_s=0.25, n_frames=24
    )

    seen: list[tuple[int, str, float, int]] = []

    def _fake_assemble(shot_id, config, modalities, n_sig, n_act, **kw):
        seen.append(
            (shot_id, kw.get("camera"), config.target_horizon_s, kw.get("start_frame"))
        )
        return f"SAMPLE-{shot_id}-{kw.get('camera')}"

    monkeypatch.setattr(ct, "assemble_controllable_window", _fake_assemble)
    ds = ManifestWindowDataset(
        windows,
        SpacetimeWindowConfig(n_frames=24, target_horizon_s=0.25),
        [],
        4,
        8,
        # the dataset-wide default camera is rbb; the per-window camera must WIN.
        camera=REFERENCE_CAMERA,
    )
    assert len(ds) == len(windows)
    for i in range(len(ds)):
        _ = ds[i]
    by_shot = {sid: (cam, hor, sf) for sid, cam, hor, sf in seen}
    # each window assembled from ITS camera, not the dataset default.
    assert by_shot[100][0] == "rco"
    assert by_shot[200][0] == "rbb"
    assert by_shot[300][0] == "rgb"
    assert by_shot[400][0] == REFERENCE_CAMERA
    # consecutive items carry DIFFERENT cameras (the per-window camera varies).
    cams_in_order = [cam for _, cam, _, _ in seen]
    assert cams_in_order[0] != cams_in_order[1]
    assert len({c for c in cams_in_order}) >= 3
    # the full-shot window's own horizon (0.30) is passed through on its config
    # copy; the fixed-horizon windows keep the global 0.25.
    assert by_shot[100][1] == pytest.approx(0.30)
    assert by_shot[200][1] == pytest.approx(0.25)
    # B's start_frame is preserved.
    assert by_shot[100][2] == 4
    assert by_shot[200][2] == 9


# ---------------------------------------------------------------------------
# end-to-end forward/backward with conditioning ON via the real collate
# ---------------------------------------------------------------------------


def _tiny_conditioned_cfg(streams) -> ControllableSpacetimeConfig:
    """A tiny model with Δt + camera conditioning ON, sized to the given streams."""
    return ControllableSpacetimeConfig(
        vocab_size=64,
        grid_h=4,
        grid_w=4,
        max_frames=80,
        plan_vocab=16,
        plan_channels=2,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        n_signal_steps=3,
        signal_streams=tuple(streams),
        actuator_channels=N_ACTUATOR_CHANNELS,
        n_act_steps=4,
        timescale_conditioning=True,
        camera_conditioning=True,
    )


def _mk_sample(shot, camera, frame_time, streams, *, vocab=64, t=6):
    """A synthetic ControllableSpacetimeSample for the given camera + cadence."""
    s = 16  # grid_h * grid_w for the tiny cfg
    rng = np.random.default_rng(shot)
    frames = rng.integers(0, vocab, size=(t, s)).astype(np.int64)
    plan = np.zeros((4, 2), dtype=np.int64)
    base = SpacetimeSample(
        shot_id=shot,
        camera=camera,
        start_frame=0,
        frames=frames,
        plan=plan,
        frame_time=np.asarray(frame_time, dtype=np.float64),
        context_frames=2,
    )
    signals = {
        st.name: rng.integers(0, st.vocab, size=(3, st.channels)).astype(np.int64)
        for st in streams
    }
    sig = SignalSpacetimeSample(base=base, signals=signals)
    raw = (rng.standard_normal((4, N_ACTUATOR_CHANNELS)) * 0.1).astype(np.float32)
    act = ActuatorPlan(
        values=normalise_actuator_values(raw),
        missing=np.zeros_like(raw),
        channel_keys=list(ACTUATOR_CHANNEL_KEYS),
        raw_values=raw,
    )
    return ControllableSpacetimeSample(signal=sig, actuator=act)


def test_forward_backward_with_conditioning_on_via_real_collate():
    """A multi-camera, multi-cadence batch flows through the REAL collate + model.

    Builds a model with Δt + camera conditioning ON over the EXTENDED signal
    streams, collates a fast-rbb + slow-rco sample with the production
    ``collate_controllable_windows``, and runs one forward+backward — asserting
    the loss is finite and gradients reach the Δt encoder + the camera embedding
    (the conditioning is consumed end-to-end, not dead).
    """
    # the extended modalities define the HF streams (xsx/xim/ait); size the model's
    # signal streams to match a couple of them so the collate + model agree.
    streams = [
        SignalStreamSpec("xma", vocab=8, channels=3),
        SignalStreamSpec("xsx", vocab=1030, channels=4),
    ]
    stream_names = [st.name for st in streams]
    cfg = _tiny_conditioned_cfg(streams)
    torch.manual_seed(0)
    model = ControllableSpacetimeTransformer(cfg).train()

    t = 6
    fast = np.linspace(0.0, 50e-6 * (t - 1), t)  # ~2 kHz burst -> log-Δt << 0
    slow = np.linspace(0.0, REFERENCE_DT_SECONDS * (t - 1), t)  # reference cadence
    samples = [
        _mk_sample(200, "rbb", fast, streams, t=t),
        _mk_sample(100, "rco", slow, streams, t=t),
    ]
    batch = ct.collate_controllable_windows(samples, stream_names=stream_names)

    # the collate carries the per-window conditioning the model consumes.
    assert batch["camera_id"].tolist() == [0, 1]  # rbb=0, rco=1
    assert batch["frame_log_dt"].shape == (2, t)
    # the fast burst maps to a strongly negative log-Δt; the slow one to ~0.
    assert float(batch["frame_log_dt"][0].mean()) < -1.5
    assert abs(float(batch["frame_log_dt"][1].mean())) < 1e-4

    loss = model(
        batch,
        loss_spec={"chunk": 4096, "context_frames": 2},
    )
    assert torch.isfinite(loss), "loss is not finite with conditioning on"
    loss.backward()
    # the conditioning params received a gradient (they are in the graph + wired
    # into the prediction the loss depends on).
    assert model.timescale_encoder.fc1.weight.grad is not None
    assert model.timescale_encoder.fc2.weight.grad is not None
    assert model.camera_embed.weight.grad is not None
    assert float(model.camera_embed.weight.grad.abs().sum()) > 0.0


def test_build_controllable_model_passes_conditioning_flags():
    """build_controllable_model threads the conditioning flags into the config."""
    streams = [SignalStreamSpec("xma", vocab=8, channels=3)]
    model = ct.build_controllable_model(
        SpacetimeWindowConfig(n_frames=6, n_plan=2, context_frames=2),
        plan_channels=2,
        signal_streams=streams,
        n_signal_steps=3,
        actuator_channels=N_ACTUATOR_CHANNELS,
        n_act_steps=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        timescale_conditioning=True,
        camera_conditioning=True,
    )
    assert model.config.timescale_conditioning is True
    assert model.config.camera_conditioning is True
    assert model.config.has_timescale and model.config.has_camera
    assert hasattr(model, "timescale_encoder")
    assert hasattr(model, "camera_embed")


def test_extended_modalities_span_the_new_hf_streams():
    """The 'extended' modality set adds the xsx/xim/ait HF streams (the CLI flag)."""
    names = {m.name for m in extended_signal_modalities()}
    assert {"xsx", "xim", "ait"}.issubset(names)


# ---------------------------------------------------------------------------
# real-data check: the production unified manifest spans all 5 cameras
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not UNIFIED_MANIFEST.exists(),
    reason="production unified manifest not present (off-cluster)",
)
def test_real_unified_manifest_spans_all_five_cameras():
    """manifest_train_windows on the REAL unified manifest spans all 5 cameras.

    A wrong camera_id key would collapse this histogram to ~100% reference camera;
    the spread across rbb/rco/rgb/rgc/rba is the canary that the per-window camera
    is read from the right field.
    """
    held_out = {18502, 18503, 18504, 18505}
    ws = manifest_train_windows(
        UNIFIED_MANIFEST, held_out=held_out, target_horizon_s=0.25, n_frames=24
    )
    hist = Counter(w.camera for w in ws)
    print(f"real unified-manifest camera histogram: {dict(sorted(hist.items()))}")
    # not a single-camera collapse: the reference camera is well under 100%.
    assert hist[REFERENCE_CAMERA] < len(ws)
    # spans the production camera set (>= 5 distinct cameras).
    assert len(hist) >= 5
    assert set(hist).issubset(set(CAMERA_IDS))
