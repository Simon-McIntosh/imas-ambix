"""Draw the watch cost figure: the tiers, and the period priced three ways.

Run with the repository's own interpreter; it writes ``cost-by-rendering.png``
beside itself. The two panels are the mechanism and its consequence: left, the
input tier gap that makes the split matter at all; right, the same recorded
period priced by the three candidate renderings, two of which the watch command
must never print. The third-party rate is read from the installed estimate
record through the very reader the watch command uses, so the figure moves with
the record rather than with a constant copied into it.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from imas_ambix.agent import watch  # noqa: E402

OUT = Path(__file__).with_name("cost-by-rendering.png")

#: The installed estimate record lives beside the price table in this directory;
#: :func:`watch.load_estimate` joins the table's directory with the record
#: filename, so reading a table path here reads the record installed with it.
INSTALLED_TABLE = Path("/work/projects/imas_gpu/agents/openrouter-prices.json")

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


def installed_rate() -> tuple[float, Path]:
    """The third-party hit rate the watch command would price at, and its record."""
    record = watch.estimate_path(INSTALLED_TABLE)
    estimate = watch.load_estimate(INSTALLED_TABLE)
    if estimate is None:
        raise SystemExit(
            f"no estimate record at {record}; install one before drawing this "
            "figure, because the third-party rate is read from it"
        )
    return estimate.rate, record


def usd(tokens_millions: float, rate: float) -> float:
    return tokens_millions * rate


def main() -> None:
    h_estimate, record = installed_rate()

    def bill(hit_rate: float) -> float:
        return (
            usd(hit_rate * TOKENS_IN / 1e6, CACHE_OWN)
            + usd((1.0 - hit_rate) * TOKENS_IN / 1e6, PROMPT)
            + usd(TOKENS_OUT / 1e6, COMPLETION)
        )

    own = bill(H_OWN)
    estimate = bill(h_estimate)
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
        f"our own\ncached split (h={H_OWN:.3f})",
        f"third-party\nestimate (h={h_estimate:.3f})",
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
    print(f"wrote {OUT}; h={h_estimate:.6f} read from {record}")


if __name__ == "__main__":
    main()
