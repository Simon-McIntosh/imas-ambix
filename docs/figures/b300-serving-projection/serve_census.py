"""One pass over a SGLang serve log and the router receipts -> census JSON."""

import datetime as dt
import json
import re
import sys
from collections import defaultdict

import numpy as np

LOG, RECEIPTS, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
GATE_CHANGE = dt.datetime(2026, 10, 8, 4, 38, tzinfo=dt.UTC).timestamp()
CEST = dt.timezone(dt.timedelta(hours=2))

ts_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
dec_re = re.compile(
    r"#running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+)"
    r".*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
pre_re = re.compile(
    r"#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)"
    r".*?#running-req: (\d+), #queue-req: (\d+)"
)


def t_of(line):
    m = ts_re.match(line)
    return (
        dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=CEST)
        .timestamp()
        if m
        else None
    )


dec, pre = [], []
with open(LOG, errors="replace") as fh:
    for line in fh:
        if "Decode batch" in line:
            m = dec_re.search(line)
            t = t_of(line)
            if m and t:
                dec.append(
                    (
                        t,
                        int(m[1]),
                        int(m[2]),
                        float(m[3]),
                        float(m[4]),
                        float(m[5]),
                        int(m[6]),
                    )
                )
        elif "Prefill batch" in line:
            m = pre_re.search(line)
            t = t_of(line)
            if m and t:
                pre.append((t, int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5])))
D = np.array(dec)
P = np.array(pre)
print("decode", D.shape, "prefill", P.shape, file=sys.stderr)


def width_table(mask):
    d = D[mask]
    t = d[:, 0]
    dtn = np.minimum(np.diff(t, append=t[-1]), 60.0)
    out = {}
    total = dtn.sum()
    for w in range(1, int(d[:, 1].max()) + 1):
        sel = d[:, 1] == w
        if sel.sum() < 30:
            continue
        g = d[sel, 5]
        out[w] = dict(
            n=int(sel.sum()),
            time_share=float(dtn[sel].sum() / total),
            agg_median=float(np.median(g)),
            agg_p25=float(np.percentile(g, 25)),
            agg_p75=float(np.percentile(g, 75)),
            per_req_median=float(np.median(g / w)),
            accept_len=float(np.median(d[sel, 4])),
            kv_median=float(np.median(d[sel, 3])),
            tokens_median=float(np.median(d[sel, 2])),
        )
    return out


windows = {
    "job": np.ones(len(D), bool),
    "pre_gate12": D[:, 0] < GATE_CHANGE,
    "post_gate12": D[:, 0] >= GATE_CHANGE,
}
res = {
    "source": {
        "log": LOG,
        "receipts": RECEIPTS,
        "decode_lines": len(D),
        "prefill_lines": len(P),
        "span_utc": [
            dt.datetime.fromtimestamp(D[0, 0], dt.UTC).isoformat(),
            dt.datetime.fromtimestamp(D[-1, 0], dt.UTC).isoformat(),
        ],
    }
}
res["width"] = {k: width_table(m) for k, m in windows.items()}

# hourly series from decode lines (time-weighted mean running, mean gen throughput)
hours = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
t = D[:, 0]
dtn = np.minimum(np.diff(t, append=t[-1]), 60.0)
for (ti, w, _, _, _, g, q), wgt in zip(dec, dtn, strict=True):
    h = int(ti // 3600) * 3600
    a = hours[h]
    a[0] += wgt
    a[1] += w * wgt
    a[2] += g * wgt
    a[3] += q * wgt
res["hourly"] = [
    dict(
        t=h,
        observed_s=a[0],
        running=a[1] / a[0] if a[0] else None,
        gen_tok_s=a[2] / 3600.0 if a[0] else None,
        gen_when_busy=a[2] / a[0] if a[0] else None,
        queued=a[3] / a[0] if a[0] else None,
    )
    for h, a in sorted(hours.items())
]

# prefill: share cached, uncached tok/s by hour
ph = defaultdict(lambda: [0, 0])
for ti, _, nt, ct, _, _ in pre:
    a = ph[int(ti // 3600) * 3600]
    a[0] += nt
    a[1] += ct
res["prefill_hourly"] = [
    dict(t=h, new=a[0], cached=a[1]) for h, a in sorted(ph.items())
]
tot_new, tot_cached = int(P[:, 2].sum()), int(P[:, 3].sum())
res["prefill_total"] = dict(
    new=tot_new, cached=tot_cached, cached_share=tot_cached / (tot_new + tot_cached)
)

# receipts
rows = []
with open(RECEIPTS) as fh:
    for line in fh:
        if MODEL not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("model") != MODEL or r.get("status") != "completed":
            continue
        ts = dt.datetime.fromisoformat(
            r["timestamp"].replace("Z", "+00:00")
        ).timestamp()
        rows.append(
            (
                ts,
                r.get("prompt_tokens") or 0,
                r.get("completion_tokens") or 0,
                r.get("time_to_first_token_s"),
                r.get("duration_s") or 0.0,
                r.get("gate_wait_s") or 0.0,
                r.get("run_id") or r.get("coordinator_session"),
            )
        )
rows.sort()
R = rows
print("receipts", len(R), file=sys.stderr)
start = D[0, 0]
inj = [r for r in R if r[0] >= start]


def rec_stats(sel):
    rate, ttft, prompt, comp, dur, wait = [], [], [], [], [], []
    for _, p, c, tf, d, gw, _ in sel:
        prompt.append(p)
        comp.append(c)
        dur.append(d)
        wait.append(gw)
        if tf is not None:
            ttft.append(tf)
        if tf is not None and c >= 64 and d - tf > 0.5:
            rate.append(c / (d - tf))

    def q(a, p):
        return float(np.percentile(a, p)) if a else None

    return dict(
        n=len(sel),
        prompt_median=q(prompt, 50),
        prompt_p90=q(prompt, 90),
        completion_median=q(comp, 50),
        completion_mean=float(np.mean(comp)) if comp else None,
        duration_median=q(dur, 50),
        duration_p90=q(dur, 90),
        ttft_median=q(ttft, 50),
        ttft_p90=q(ttft, 90),
        gate_wait_median=q(wait, 50),
        gate_wait_p90=q(wait, 90),
        gate_wait_p99=q(wait, 99),
        decode_rate_median=q(rate, 50),
        decode_rate_p10=q(rate, 10),
        decode_rate_p90=q(rate, 90),
        prompt_sum=int(sum(prompt)),
        completion_sum=int(sum(comp)),
    )


res["receipts"] = {
    "job": rec_stats(inj),
    "pre_gate12": rec_stats([r for r in inj if r[0] < GATE_CHANGE]),
    "post_gate12": rec_stats([r for r in inj if r[0] >= GATE_CHANGE]),
}
# daily token totals over the whole receipt stream
day = defaultdict(lambda: [0, 0, 0])
for ts, p, c, *_ in R:
    a = day[dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")]
    a[0] += p
    a[1] += c
    a[2] += 1
res["receipts_daily"] = {
    k: dict(prompt=v[0], completion=v[1], requests=v[2]) for k, v in sorted(day.items())
}
# in-flight from receipts (sweep line) -> time share by concurrent requests
ev = []
for ts, _, _, _, d, gw, _ in inj:
    s = ts + gw
    ev.append((s, 1))
    ev.append((s + max(d - gw, 0.0), -1))
ev.sort()
cur = 0
last = ev[0][0]
share = defaultdict(float)
for x, k in ev:
    share[cur] += x - last
    cur += k
    last = x
tot = sum(share.values())
res["inflight_share"] = {int(k): v / tot for k, v in sorted(share.items())}
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1, default=float)
print("ok", file=sys.stderr)
