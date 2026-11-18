"""Hourly delivered completion tokens, mean requests in service and gate wait from router receipts.

usage: hourly.py <out.json> <receipts.jsonl>...
"""
import json, sys, datetime as dt, collections
out = sys.argv[1]
P = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
tok = collections.Counter(); busy = collections.Counter(); waitsum = collections.Counter(); n = collections.Counter(); waited = collections.Counter()
for path in sys.argv[2:]:
    for line in open(path):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get("model") != "deepseek-v4.1-flash" or r.get("duration_s") is None: continue
        s = P(r["timestamp"]); d = float(r["duration_s"]); ct = int(r.get("completion_tokens") or 0); w = float(r.get("gate_wait_s") or 0)
        h = int(s // 3600); n[h] += 1; waitsum[h] += w; waited[h] += w > 1
        t = s
        while t < s + d:  # spread service time and tokens over the hours the request spans
            hh = int(t // 3600); e = min(s + d, (hh + 1) * 3600.0)
            busy[hh] += e - t; tok[hh] += ct * (e - t) / max(d, 1e-9); t = e
rows = [dict(hour=dt.datetime.fromtimestamp(h * 3600, dt.timezone.utc).strftime("%Y-%m-%dT%H:00Z"), tok_s=tok[h] / 3600, in_service=busy[h] / 3600,
             requests=n[h], mean_wait=waitsum[h] / max(1, n[h]), waited_share=waited[h] / max(1, n[h])) for h in sorted(busy)]
json.dump(rows, open(out, "w"), indent=1)
for r in rows: print(f"{r['hour']}  {r['tok_s']:6.0f} tok/s  {r['in_service']:5.2f} in service  wait {r['mean_wait']:5.1f}s  waited>1s {r['waited_share']:4.0%}")
