"""Draw the watch cost figure: the tiers, and the period priced three ways.

Run from anywhere with the repository's own interpreter; it writes
``cost-by-rendering.png`` beside itself. The two panels are the mechanism and
its consequence: left, the input tier gap that makes the split matter at all;
right, the same recorded period priced by the three candidate renderings, two
of which the watch command must never print.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).with_name("cost-by-rendering.png")

# Per-token rates, in dollars per million tokens.
PROMPT = 10.0
CACHE_OWN = 1.0
COMPLETION = 20.0

# One recorded period: 278,174 prompt tokens and 40,000 generated.
TOKENS_IN = 278_174.0
TOKENS_OUT = 40_000.0

#: Our own prefix cache's measured share of that input, from the serve's own
#: cached-token counters. This is the share the old figure billed at the cheap
#: tier, and it is higher than any hosted endpoint would reach.
H_OWN = 0.9745

#: The installed third-party estimate, on the small-turn stratum.
H_ESTIMATE = 0.858


def usd(tokens_millions: float, rate: float) -> float:
    return tokens_millions * rate


def main() -> None:
    def bill(hit_rate: float) -> float:
        cached = hit_rate * TOKENS_IN
        uncached = (1.0 - hit_rate) * TOKENS_IN
        return (
            usd(cached / 1e6, CACHE_OWN)
            + usd(uncached / 1e6, PROMPT)
            + usd(TOKENS_OUT / 1e6, COMPLETION)
        )

    own = bill(H_OWN)
    estimate = bill(H_ESTIMATE)
    no_cache = bill(0.0)

    fig, (ax_rate, ax_cost) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    tiers = ["prompt\n(tier 1)", "cache read\n(tier 2)", "completion"]
    rates = [PROMPT, CACHE_OWN, COMPLETION]
    bars = ax_rate.bar(tiers, rates, color=["#c44e52", "#4c72b0", "#55a868"])
    ax_rate.set_title("Price tiers, per million tokens")
    ax_rate.set_ylabel("USD / M tokens")
    for rect, value in zip(bars, rates, strict=True):
        ax_rate.text(
            rect.get_x() + rect.get_width() / 2,
            value + 0.4,
            f"${value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_rate.annotate(
        "input costs 10x more\nat the prompt tier",
        xy=(1, CACHE_OWN),
        xytext=(0.55, 8.0),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=9,
        ha="left",
    )
    ax_rate.set_ylim(0, 23)

    renderings = [
        "our own\ncached split (h=0.97)",
        "third-party\nestimate (h=0.858)",
        "no cache\n(h=0)",
    ]
    costs = [own, estimate, no_cache]
    colours = ["#999999", "#4c72b0", "#999999"]
    edged = ["#999999", "#111111", "#999999"]
    bars = ax_cost.bar(renderings, costs, color=colours, edgecolor=edged, linewidth=2.0)
    ax_cost.set_title("The same period, priced three ways")
    ax_cost.set_ylabel("USD for the period")
    for rect, value in zip(bars, costs, strict=True):
        ax_cost.text(
            rect.get_x() + rect.get_width() / 2,
            value + 0.15,
            f"${value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_cost.set_ylim(0, max(costs) * 1.18)
    ax_cost.text(
        0.5,
        0.94,
        "print this one",
        transform=ax_cost.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#4c72b0",
        fontweight="bold",
    )

    fig.suptitle(
        "Watch prints the third-party estimate: neither our own cache "
        "nor a cold start",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()