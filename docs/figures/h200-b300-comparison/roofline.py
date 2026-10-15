"""Roofline of the H200 NVL in service here against an HGX B300, in FP64 and BF16.

Attainable throughput is min(bandwidth x intensity, peak). Peaks are dense
datasheet figures: H200 NVL FP64 tensor 60 TFLOP/s, BF16 835 TFLOP/s at
4.8 TB/s; B300 FP64 1.2 TFLOP/s (vector and tensor alike), BF16 2,250
TFLOP/s at 8 TB/s. Writes roofline.svg beside this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
H200 = "#3d6fa5"
B300 = "#c06a35"
GREY = "#7a7a7a"

CARDS = {
    "H200": {"bw": 4.8, "fp64": 60.0, "bf16": 835.0, "color": H200},
    "B300": {"bw": 8.0, "fp64": 1.2, "bf16": 2250.0, "color": B300},
}


def attainable(bw_tb_s, peak_tflops, intensity):
    return np.minimum(bw_tb_s * intensity, peak_tflops)


def main():
    plt.style.use("data-ink")
    ai = np.logspace(-2, 3, 400)
    fig, ax = plt.subplots(figsize=(14, 6.2))
    for card in CARDS.values():
        ax.plot(
            ai, attainable(card["bw"], card["fp64"], ai), color=card["color"], lw=3.0
        )
        ax.plot(
            ai,
            attainable(card["bw"], card["bf16"], ai),
            color=card["color"],
            lw=2.6,
            ls="--",
        )
    ax.text(900, 60 * 1.15, "H200 FP64", color=H200, ha="right", va="bottom")
    ax.text(900, 1.2 * 1.15, "B300 FP64", color=B300, ha="right", va="bottom")
    ax.text(900, 835 / 1.25, "H200 BF16", color=H200, ha="right", va="top")
    ax.text(900, 2250 * 1.15, "B300 BF16", color=B300, ha="right", va="bottom")
    ridge = 1.2 / 8.0
    ax.axvline(ridge, color=GREY, lw=1.2, ls=":")
    ax.text(
        ridge * 1.08, 0.04, "B300 FP64 ridge\n0.15 flop/byte", color=GREY, va="bottom"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(0.03, 6000)
    ax.set_xlabel("arithmetic intensity [flop/byte]")
    ax.set_ylabel("attainable throughput [TFLOP/s]")
    fig.tight_layout()
    fig.savefig(HERE / "roofline.svg")


if __name__ == "__main__":
    main()
