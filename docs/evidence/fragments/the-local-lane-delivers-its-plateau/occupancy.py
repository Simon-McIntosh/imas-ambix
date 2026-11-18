"""Reconstruct the lane's in-flight request count from router receipts.

usage: occupancy.py <receipts.jsonl> <start-iso> <end-iso> [width]
Each receipt is admitted at `timestamp` after waiting `gate_wait_s` at the gate, and is relayed for `duration_s`.
"""
import json, sys, datetime as dt, collections
path, a, b = sys.argv[1:4]; width = int(sys.argv[4]) if len(sys.argv) > 4 else 12
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
t0, t1 = P(a), P(b)
ev = []; qev = []; waits = []; runs = collections.defaultdict(lambda: [1e18, 0.0, 0.0])
n = 0
for line in open(path):
    try: r = json.loads(line)
    except ValueError: continue
    if r.get("model") != "deepseek-v4.1-flash" or r.get("duration_s") is None: continue
    start = P(r["timestamp"]); dur = float(r["duration_s"]); w = float(r.get("gate_wait_s") or 0)
    qstart = start - w; end = start + dur
    if end < t0 or start > t1: continue
    n += 1
    s, e = max(start, t0), min(end, t1)
    ev.append((s, 1)); ev.append((e, -1)); waits.append(w)
    if w > 0: qev.append((max(qstart, t0), 1)); qev.append((max(min(start, t1), t0), -1))
    k = r.get("run_id") or "unkeyed"
    runs[k][0] = min(runs[k][0], s); runs[k][1] = max(runs[k][1], e); runs[k][2] += e - s
ev.sort()
hist = collections.Counter(); cur = 0; last = t0
for t, d in ev:
    hist[cur] += t - last; last = t; cur += d
hist[cur] += t1 - last
span = t1 - t0
mean = sum(k * v for k, v in hist.items()) / span
print(f"window {a} .. {b}  requests {n}  mean in flight {mean:.2f} of {width}  utilisation {min(mean,width)/width:.1%}")
print("time share by in-flight count:")
for k in sorted(hist): print(f"  {k:3d}  {hist[k]/span:6.1%}")
print(f"share of time below width: {sum(v for k,v in hist.items() if k<width)/span:.1%}; at or above: {sum(v for k,v in hist.items() if k>=width)/span:.1%}")
waits.sort()
q = lambda p: waits[int(p * (len(waits) - 1))] if waits else 0
print(f"gate wait s: median {q(.5):.2f}  p90 {q(.9):.2f}  p99 {q(.99):.2f}  max {q(1):.1f}  share waiting >1s {sum(w>1 for w in waits)/max(1,len(waits)):.1%}")
duty = [(v[2] / max(1e-9, v[1] - v[0]), v[1] - v[0]) for k, v in runs.items() if k != "unkeyed" and v[1] - v[0] > 120]
duty.sort()
if duty:
    tot_busy = sum(v[2] for k, v in runs.items() if k != "unkeyed"); tot_live = sum(v[1] - v[0] for k, v in runs.items() if k != "unkeyed")
    print(f"runs {len(runs)}  per-run duty while live: pooled {tot_busy/tot_live:.3f}  median {duty[len(duty)//2][0]:.3f}  p10 {duty[len(duty)//10][0]:.3f}  p90 {duty[9*len(duty)//10][0]:.3f}")
    print(f"mean live runs (sum of live spans / window): {tot_live/span:.1f}")

qev.sort(); qh = collections.Counter(); cur = 0; last = t0
for t, d in qev:
    qh[cur] += t - last; last = t; cur += d
qh[cur] += t1 - last
print(f"mean waiting at the gate {sum(k*v for k,v in qh.items())/span:.2f}; share of time with nobody waiting {qh[0]/span:.1%}")
