"""Per-stream and aggregate generation rate against the number of requests in service.

usage: rate_by_load.py <receipts.jsonl> <start-iso> <end-iso>
"""
import json, sys, datetime as dt, bisect, collections
path, a, b = sys.argv[1:4]
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
t0, t1 = P(a), P(b); rows = []
for line in open(path):
    try: r = json.loads(line)
    except ValueError: continue
    if r.get("model") != "deepseek-v4.1-flash" or r.get("duration_s") is None or r.get("status") != "completed": continue
    s = P(r["timestamp"])
    if s < t0 or s > t1: continue
    rows.append((s, float(r["duration_s"]), float(r.get("time_to_first_token_s") or 0), int(r.get("completion_tokens") or 0), int(r.get("prompt_tokens") or 0)))
ev = sorted([(s, 1) for s, d, *_ in rows] + [(s + d, -1) for s, d, *_ in rows])
ts = []; lv = []; cum = []; cur = 0; area = 0.0; last = ev[0][0]
for t, dl in ev:
    area += cur * (t - last); last = t; cur += dl; ts.append(t); lv.append(cur); cum.append(area)
def mean_level(s, e):
    i = bisect.bisect_right(ts, s) - 1; j = bisect.bisect_right(ts, e) - 1
    ai = cum[i] + lv[i] * (s - ts[i]) if i >= 0 else 0.0
    aj = cum[j] + lv[j] * (e - ts[j]) if j >= 0 else 0.0
    return (aj - ai) / max(1e-9, e - s)
bins = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for s, d, ttft, ct, pt in rows:
    gen = d - ttft
    if ct < 50 or gen < 1: continue
    k = min(12, round(mean_level(s, s + d)))
    x = bins[k]; x[0] += 1; x[1] += ct; x[2] += gen; x[3] += ttft; x[4] += pt
print("in service   requests   per-stream tok/s   implied aggregate tok/s   mean ttft s   mean prompt k")
for k in sorted(bins):
    n, ct, gen, ttft, pt = bins[k]
    if n < 20: continue
    print(f"{k:10d} {n:10d} {ct/gen:18.1f} {k*ct/gen:25.0f} {ttft/n:13.2f} {pt/n/1000:15.0f}")
tot = sum(r[3] for r in rows); print(f"completion tokens {tot}  over {t1-t0:.0f}s = {tot/(t1-t0):.0f} tok/s delivered")
