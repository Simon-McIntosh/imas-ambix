from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from imas_ambix.worldmodel.decode_guidance_sweep import (
    guided_token_probabilities,
    mean_collapse_mask,
    record_visual_assessment,
)


def test_guidance_arithmetic_changes_sampled_distribution() -> None:
    conditional = torch.tensor([[0.0, 1.5]])
    unconditional = torch.tensor([[0.0, 0.0]])
    weak = guided_token_probabilities(
        conditional,
        unconditional,
        guidance_weight=1.0,
        temperature=1.0,
    )
    strong = guided_token_probabilities(
        conditional,
        unconditional,
        guidance_weight=3.5,
        temperature=1.0,
    )
    generator = torch.Generator().manual_seed(91)
    weak_samples = torch.multinomial(weak.expand(20_000, -1), 1, generator=generator)
    generator.manual_seed(91)
    strong_samples = torch.multinomial(
        strong.expand(20_000, -1), 1, generator=generator
    )

    weak_frequency = float((weak_samples == 1).float().mean())
    strong_frequency = float((strong_samples == 1).float().mean())
    assert strong_frequency > weak_frequency + 0.15
    assert strong[0, 1] > weak[0, 1]


def test_mean_collapse_indicator_separates_uniform_and_structured_frames() -> None:
    axis = np.linspace(0.0, 255.0, 16, dtype=np.float64)
    structured = np.tile(axis, (16, 1))
    real = np.stack([structured, structured.T])[:, :, :, None]
    decoded = np.stack([np.full((16, 16), 127.0), structured])[:, :, :, None]

    collapsed = mean_collapse_mask(decoded, real)

    assert collapsed.tolist() == [True, False]


def test_visual_assessment_records_each_cell(tmp_path: Path) -> None:
    verdict = {
        "cells": [
            {
                "guidance_weight": 1.0,
                "temperature": 0.7,
                "recognisable_plasma_column": None,
            },
            {
                "guidance_weight": 2.0,
                "temperature": 1.0,
                "recognisable_plasma_column": None,
            },
        ],
        "visual_verdict": "pending",
    }
    (tmp_path / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")

    result = record_visual_assessment(
        tmp_path,
        recognisable_cells=[(2.0, 1.0)],
        note="Reviewed both contact sheets.",
    )

    assert [row["recognisable_plasma_column"] for row in result["cells"]] == [
        False,
        True,
    ]
    assert "guidance_weight=2" in str(result["visual_verdict"])
