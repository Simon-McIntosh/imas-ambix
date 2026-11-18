"""Draw the three figures for the lane-utilisation plan from the measured and simulated records."""
import json, sys, datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("data-ink")
OUT = sys.argv[1]
GREY = "#929292"; DARK = "#3a3a3a"; BLUE = "#2f6fb0"; GREEN = "#2e8b57"; PURPLE = "#7a4fa3"
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg"); fig.savefig(f"{name}-check.png", dpi=100); plt.close(fig)

# 1. rate against load, from receipts
cal = json.load(open("calib-1118.json")); k = sorted(int(x) for x in cal["rate"]); r = [cal["rate"][str(x)] for x in k]
fig, ax = plt.subplots(figsize=(14, 7.2))
ax.plot(k, [a * b for a, b in zip(k, r)], color=BLUE, lw=3.0, marker="o", zorder=3)
ax.plot(k, r, color=GREY, lw=2.6, marker="o", zorder=2)
ax.text(12.2, k[-1] * r[-1], "all streams together", color=BLUE, va="center", fontsize=14)
ax.text(12.2, r[-1], "one stream", color=GREY, va="center", fontsize=14)
ax.set_xlabel("requests in service"); ax.set_ylabel("generation rate [tok/s]"); ax.set_xticks(k); ax.set_xlim(0.5, 14.6); ax.set_ylim(0, 520)
save(fig, "rate-against-requests-in-service")

# 2. hourly delivered rate
rows = json.load(open("hourly.json")); t = [dt.datetime.fromisoformat(x["hour"].replace("Z", "+00:00")) for x in rows]; v = [x["tok_s"] for x in rows]
plateau = 444
fig, ax = plt.subplots(figsize=(14, 7.2))
ax.bar(t, v, width=1 / 24 * 0.9, color=[GREY if y >= 0.5 * plateau else "#c0392b" for y in v], align="edge", zorder=2)
ax.axhline(plateau, color=GREY, lw=1.2, ls=":"); ax.text(t[-1] + dt.timedelta(hours=2), plateau, "plateau", color=GREY, va="bottom", fontsize=14)
ax.set_ylabel("delivered rate [tok/s]"); ax.set_xlabel("hour [UTC]"); ax.set_xlim(t[0], t[-1] + dt.timedelta(hours=7))
import matplotlib.dates as md
ax.xaxis.set_major_locator(md.HourLocator(byhour=[0, 12])); ax.xaxis.set_major_formatter(md.DateFormatter("%d %b %H:%M"))
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=GREY, label="at least half the plateau"), Patch(color="#c0392b", label="below half the plateau")], loc="upper center", ncol=2, frameon=False, fontsize=14)
save(fig, "delivered-rate-by-hour")

# 3. simulation: fixed worker counts and the rule's operating points on two workloads
base = json.load(open("sim-w12.json"))["rows"]; fixed = [x for x in base if x["case"].startswith("fixed")]
n = [int(x["case"].split()[1]) for x in fixed]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7.2))
a1.plot(n, [x["tok_s"] for x in fixed], color=GREY, lw=2.6, marker="o", zorder=2)
a2.plot(n, [x["wait_p90"] for x in fixed], color=GREY, lw=2.6, marker="o", zorder=2)
for name, col, lab in (("sim-offered-w12.json", BLUE, "whole-night workload"), ("late-offered.json", GREEN, "late-window workload")):
    rr = json.load(open(name))["rows"]
    a1.plot([x["workers"] for x in rr], [x["tok_s"] for x in rr], color=col, lw=3.0, marker="s", zorder=3, label="rule, "+lab)
    a2.plot([x["workers"] for x in rr], [x["wait_p90"] for x in rr], color=col, lw=3.0, marker="s", zorder=3)
a1.plot([], [], color=GREY, lw=2.6, marker="o", label="fixed worker count, whole-night workload")
a1.set_xlabel("workers on the lane"); a1.set_ylabel("delivered rate [tok/s]"); a2.set_xlabel("workers on the lane"); a2.set_ylabel("gate wait, 90th percentile [s]")
a1.set_ylim(200, 470); a2.set_ylim(0, 30)
fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=14)
fig.subplots_adjust(top=0.88, wspace=0.25)
save(fig, "simulated-rate-and-wait-against-workers")
