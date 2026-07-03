#!/usr/bin/env python
r"""Figure for the amortised patch-current encoder (P4) record.

One figure, driven from ``imas_ambix/latent/artifacts/patch_gate/``:

``fig-encoder-gate.png`` — the single-forward-pass encoder's data-scaling,
grounding, and per-quantity story: (a) axis median error vs training-corpus
size, split in-signature / cross-signature, against the P3 variational
inverse / train-mean / corrected-Picard reference lines; (b) grounding
ratio (encoder misfit vs shuffled-encoder misfit) per model x signature
split, against the banked 5.53x bar; (c) per-quantity skill for the best
2k-corpus arm (in-signature), with the x-point-set skill annotated positive.

The lowrank head trained on the 588-example corpus only has an OLD
partial-coverage eval (31/80 shots) and is not comparable to the
full-coverage numbers plotted here — it is called out in a footnote only.
The 2k-corpus lowrank eval (``encoder_gate_lowrank2k.json``) is picked up
automatically if present; the script runs (and says so) without it.

Usage::

    uv run python scripts/plot_patch_encoder_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "imas_ambix" / "latent" / "artifacts" / "patch_gate"
FIG_DIR = REPO / "docs" / "figures" / "patch-current-force-balance"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIRECT_COLOR = "#2166ac"
LOWRANK_COLOR = "#7b3294"
BASELINE_COLOR = "#636363"
PICARD_COLOR = "#8a3324"
P3_INVERSE_COLOR = "#1b7837"

GROUNDING_BAR = 5.53

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_json(name: str) -> dict | None:
    path = ARTIFACT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def train_mean_baseline_cm() -> float:
    """Median axis error of the train-mean baseline over the gate-2 160 slices.

    Same ``baseline`` field the arm npz files carry (constant per slice,
    equal to the corpus-mean axis position); any gate-2 arm's npz works.
    """
    npz = np.load(ARTIFACT_DIR / "gate_arrays_fixed-30_gate2.npz")
    baseline = np.asarray(npz["baseline"], dtype=np.float64)
    ref = np.asarray(npz["ref"], dtype=np.float64)
    err_cm = np.hypot(baseline[:, 0] - ref[:, 0], baseline[:, 1] - ref[:, 1]) * 100.0
    return float(np.median(err_cm))


def p3_inverse_reference_cm() -> tuple[float, str]:
    """Median axis error of the P3 gate-2 winner arm (highest axis skill)."""
    eval_json = load_json("patch_gate_eval_gate2.json")
    per_policy = eval_json["per_policy"]
    winner = max(per_policy, key=lambda k: per_policy[k]["axis_skill"])
    return per_policy[winner]["axis_error_median_m"] * 100.0, winner


def picard_reference_cm() -> tuple[float, int, int]:
    picard = load_json("patch_gate_eval_gate2.json")["picard_reference"]
    return (
        picard["axis_error_median_m"] * 100.0,
        picard["n_scored"],
        picard["n_candidate"],
    )


# --------------------------------------------------------------------------
# encoder records
# --------------------------------------------------------------------------


class EncoderRecord:
    def __init__(self, label: str, head: str, n_examples: int, data: dict) -> None:
        self.label = label
        self.head = head
        self.n_examples = n_examples
        self.data = data
        self.full_coverage = data["scored_fraction"] >= 0.99

    def median_cm(self, split: str | None = None) -> float | None:
        block = self.data if split is None else self.data.get(split)
        if block is None:
            return None
        return block["axis_error_median_m"] * 100.0

    def grounding_ratio(self, split: str | None = None) -> float | None:
        block = self.data if split is None else self.data.get(split)
        if block is None:
            return None
        return block["grounding"]["ratio"]

    def skill(self, key: str, split: str | None = None) -> float | None:
        block = self.data if split is None else self.data.get(split)
        if block is None:
            return None
        return block["per_quantity_skill"].get(key)

    def aggregate_skill(self, key: str, split: str | None = None) -> float | None:
        block = self.data if split is None else self.data.get(split)
        if block is None:
            return None
        return block.get(key)


def load_records() -> tuple[list[EncoderRecord], EncoderRecord | None]:
    records = []
    v0_direct = load_json("encoder_gate_direct_v2.json")
    if v0_direct is not None:
        records.append(EncoderRecord("v0 direct", "direct", 588, v0_direct))

    v2k_direct = load_json("encoder_gate_direct2k.json")
    if v2k_direct is not None:
        records.append(EncoderRecord("2k direct", "direct", 14_525, v2k_direct))

    v2k_lowrank = load_json("encoder_gate_lowrank2k.json")
    if v2k_lowrank is not None:
        records.append(EncoderRecord("2k lowrank", "lowrank", 14_525, v2k_lowrank))
    else:
        print(
            "[fig-encoder-gate] encoder_gate_lowrank2k.json not found yet — "
            "figure omits the 2k-lowrank arm; re-run once it lands."
        )

    v0_lowrank = load_json("encoder_gate_lowrank.json")
    v0_lowrank_record = None
    if v0_lowrank is not None:
        v0_lowrank_record = EncoderRecord("v0 lowrank", "lowrank", 588, v0_lowrank)

    return records, v0_lowrank_record


# --------------------------------------------------------------------------
# panel (a) — data scaling
# --------------------------------------------------------------------------


def panel_data_scaling(ax, records: list[EncoderRecord]) -> None:
    color = {"direct": DIRECT_COLOR, "lowrank": LOWRANK_COLOR}
    for head in ("direct", "lowrank"):
        head_records = sorted(
            (r for r in records if r.head == head and r.full_coverage),
            key=lambda r: r.n_examples,
        )
        if not head_records:
            continue
        xs = [r.n_examples for r in head_records]
        in_ys = [r.median_cm("in_signature") for r in head_records]
        cross_ys = [r.median_cm("cross_signature") for r in head_records]
        style = "-" if head == "direct" else "--"
        marker_kw = {"lw": 1.8, "color": color[head], "ls": style}
        if len(xs) >= 2:
            ax.plot(
                xs, in_ys, marker="o", ms=7, label=f"{head}, in-signature", **marker_kw
            )
            ax.plot(
                xs,
                cross_ys,
                marker="^",
                ms=7,
                label=f"{head}, cross-signature",
                **marker_kw,
            )
        else:
            ax.plot(xs, in_ys, marker="o", ms=8, ls="none", color=color[head])
            ax.plot(xs, cross_ys, marker="^", ms=8, ls="none", color=color[head])
            ax.annotate(
                f"{head}, in-sig",
                xy=(xs[0], in_ys[0]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=7,
                color=color[head],
                va="center",
            )
            ax.annotate(
                f"{head}, cross-sig",
                xy=(xs[0], cross_ys[0]),
                xytext=(6, -8),
                textcoords="offset points",
                fontsize=7,
                color=color[head],
                va="center",
            )

    p3_cm, p3_arm = p3_inverse_reference_cm()
    train_mean_cm = train_mean_baseline_cm()
    picard_cm, picard_n, picard_n_cand = picard_reference_cm()

    ax.axhline(p3_cm, color=P3_INVERSE_COLOR, lw=1.3, ls=":")
    ax.axhline(train_mean_cm, color=BASELINE_COLOR, lw=1.3, ls=":")
    ax.axhline(picard_cm, color=PICARD_COLOR, lw=1.3, ls=":")
    ax.text(
        0.98,
        0.04,
        f"P3 variational inverse ({p3_arm}): {p3_cm:.1f} cm\n"
        f"train-mean baseline: {train_mean_cm:.1f} cm\n"
        f"corrected Picard: {picard_cm:.1f} cm ({picard_n}/{picard_n_cand})",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": "#999999",
            "alpha": 0.9,
        },
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training examples")
    ax.set_ylabel("axis median error  [cm]  (log scale)")
    ax.set_title(
        "data scaling: 25x data → 3x geometry\n(single forward pass, label-free)"
    )
    ax.legend(fontsize=7, loc="upper right")


# --------------------------------------------------------------------------
# panel (b) — grounding
# --------------------------------------------------------------------------


def panel_grounding(ax, records: list[EncoderRecord]) -> None:
    splits = [
        (None, "overall", None),
        ("in_signature", "in-sig", "//"),
        ("cross_signature", "cross-sig", "xx"),
    ]
    color = {"direct": DIRECT_COLOR, "lowrank": LOWRANK_COLOR}

    xticks = []
    xlabels = []
    x = 0
    for r in sorted(records, key=lambda r: (r.n_examples, r.head)):
        for split_key, split_label, hatch in splits:
            ratio = r.grounding_ratio(split_key)
            if ratio is None:
                continue
            ax.bar(
                x,
                ratio,
                width=0.7,
                color=color[r.head],
                hatch=hatch,
                edgecolor="white",
                alpha=0.9,
            )
            ax.annotate(
                f"{ratio:.1f}x",
                xy=(x, ratio),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
            xticks.append(x)
            xlabels.append(f"{r.label}\n{split_label}")
            x += 1
        x += 0.6

    ax.axhline(GROUNDING_BAR, color=BASELINE_COLOR, lw=1.4, ls="--")
    ax.text(
        -0.5,
        GROUNDING_BAR,
        f"bar {GROUNDING_BAR:.2f}x ",
        color=BASELINE_COLOR,
        fontsize=7.5,
        va="bottom",
        ha="left",
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("grounding ratio  (encoder / shuffled misfit, log scale)")
    ax.set_title(
        "grounding: encoder clears the banked bar\n"
        "in-signature; crosses it cross-signature"
    )


# --------------------------------------------------------------------------
# panel (c) — per-quantity skills, best 2k arm
# --------------------------------------------------------------------------


def pick_best_2k_arm(records: list[EncoderRecord]) -> EncoderRecord | None:
    candidates = [r for r in records if r.n_examples == 14_525]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.aggregate_skill("axis_skill") or -np.inf)


def panel_per_quantity(ax, records: list[EncoderRecord]) -> None:
    best = pick_best_2k_arm(records)
    if best is None:
        ax.text(0.5, 0.5, "no 2k-corpus arm available", ha="center", va="center")
        return

    rows = [
        ("axis_R", best.skill("axis_R", "in_signature")),
        ("axis_Z", best.skill("axis_Z", "in_signature")),
        ("x-point set", best.aggregate_skill("xpoint_set_skill", "in_signature")),
        ("LCFS", best.aggregate_skill("lcfs_skill", "in_signature")),
    ]
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [DIRECT_COLOR if best.head == "direct" else LOWRANK_COLOR] * len(values)

    y = np.arange(len(values))
    ax.barh(y, values, color=colors, alpha=0.9)
    ax.axvline(0.0, color=BASELINE_COLOR, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("skill  (0 = train-mean baseline, log-scaled residual)")
    ax.set_title(f"per-quantity skill, best 2k arm ({best.label}, in-signature)")

    finite = [v for v in values if v is not None]
    lo, hi = min(finite), max(finite)
    span = hi - lo
    ax.set_xlim(lo - 0.28 * span, hi + 0.55 * span)

    for yi, v in zip(y, values, strict=True):
        if v is None:
            continue
        label = f"{v:+.3f}" + ("  (only positive quantity)" if v > 0 else "")
        ax.annotate(
            label,
            xy=(v, yi),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=7.5,
            color="#1b7837" if v > 0 else "black",
        )


# --------------------------------------------------------------------------
# figure assembly
# --------------------------------------------------------------------------


def fig_encoder_gate(out: Path) -> None:
    records, v0_lowrank = load_records()

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(13.5, 4.6), constrained_layout=True
    )

    panel_data_scaling(axa, records)
    panel_grounding(axb, records)
    panel_per_quantity(axc, records)

    if v0_lowrank is not None:
        footnote = (
            f"lowrank v0 (588 examples): OLD partial-coverage eval "
            f"({v0_lowrank.data['n_scored']}/{v0_lowrank.data['n_candidate']} shots) — "
            f"median {v0_lowrank.median_cm():.1f} cm, "
            f"grounding {v0_lowrank.grounding_ratio():.2f}x; "
            f"not directly comparable to the full-coverage numbers above."
        )
        fig.text(
            0.01, -0.02, footnote, fontsize=6.5, color="#555555", ha="left", va="top"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig-encoder-gate] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=FIG_DIR / "fig-encoder-gate.png",
        help="output PNG path",
    )
    args = ap.parse_args()
    fig_encoder_gate(args.out)


if __name__ == "__main__":
    main()
