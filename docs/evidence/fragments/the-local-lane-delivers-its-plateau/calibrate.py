"""Extract the worker behaviour a lane simulation needs from router receipts.

usage: calibrate.py <receipts.jsonl> <start-iso> <end-iso> <out.json>
Writes think times between a run's requests, per-request work, requests per run,
and the per-stream rate against requests in service.
"""
import json, sys, datetime as dt, bisect, collections, statistics as st
path, a, b, out = sys.argv[1:5]
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
t0, t1 = P(a), P(b); rows = []
for line in open(path):
    try: r = json.loads(line)
    except ValueError: continue
    if r.get("model") != "deepseek-v4.1-flash" or r.get("duration_s") is None: continue
    s = P(r["timestamp"])
    if s < t0 or s > t1 or not r.get("run_id"): continue
    rows.append(dict(run=r["run_id"], s=s, d=float(r["duration_s"]), w=float(r.get("gate_wait_s") or 0),
                     ttft=float(r.get("time_to_first_token_s") or 0), ct=int(r.get("completion_tokens") or 0)))
by = collections.defaultdict(list)
for r in rows: by[r["run"]].append(r)
think = []; per_run = []; life = []
for run, rs in by.items():
    rs.sort(key=lambda r: r["s"])
    # a run seen at either edge of the window is truncated; keep its gaps, not its length
    if rs[0]["s"] - t0 > 300 and t1 - (rs[-1]["s"] + rs[-1]["d"]) > 300:
        per_run.append(len(rs)); life.append(rs[-1]["s"] + rs[-1]["d"] - rs[0]["s"])
    for p, q in zip(rs, rs[1:]):
        g = (q["s"] - q["w"]) - (p["s"] + p["d"])
        if g >= 0: think.append(round(g, 2))
ev = sorted([(r["s"], 1) for r in rows] + [(r["s"] + r["d"], -1) for r in rows])
ts = []; lv = []; cur = 0
for t, dl in ev: cur += dl; ts.append(t); lv.append(cur)
rate = collections.defaultdict(lambda: [0.0, 0.0])
work = []
for r in rows:
    gen = r["d"] - r["ttft"]
    work.append([r["ct"], round(r["ttft"], 2)])
    if r["ct"] >= 50 and gen >= 1:
        k = lv[max(0, bisect.bisect_right(ts, r["s"] + r["d"] / 2) - 1)]
        rate[min(k, 12)][0] += r["ct"]; rate[min(k, 12)][1] += gen
curve = {k: round(v[0] / v[1], 1) for k, v in sorted(rate.items()) if v[1] > 300}
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
print(f"runs {len(by)} complete-in-window {len(per_run)} requests {len(rows)}")
print(f"think s: median {q(think,.5):.1f} mean {st.mean(think):.1f} p90 {q(think,.9):.1f} p99 {q(think,.99):.1f}")
print(f"requests per run: median {q(per_run,.5)} mean {st.mean(per_run):.0f} p90 {q(per_run,.9)}; run life min: median {q(life,.5)/60:.0f} p90 {q(life,.9)/60:.0f}")
print(f"completion tokens per request: median {q([w[0] for w in work],.5)} mean {st.mean(w[0] for w in work):.0f}")
print("per-stream tok/s by requests in service:", curve)
json.dump(dict(think=think, work=work, per_run=per_run, rate=curve, window=[a, b]), open(out, "w"))
