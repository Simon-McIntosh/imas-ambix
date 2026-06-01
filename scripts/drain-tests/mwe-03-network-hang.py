"""
MWE-03: Network Black-Hole D-State Hang (urllib download, revised)
===================================================================
PURPOSE
  Reproduces the EXACT mechanism of Event #3 (2026-05-27b, job ~1208596):
  a model-weight download via urllib.request.urlopen() from a betelgeuse
  GPU node (no outbound network, firewall DROPs packets) caused the Python
  process to enter D-state (TASK_UNINTERRUPTIBLE). SLURM scancel could not
  reap it within UnkillableStepTimeout → "Kill task failed" → node drained.

  The first MWE-03 run used raw socket.connect() to a bare IP, which Python
  handles via interruptible S-state futex — the wrong code path. The original
  failure used urllib.request.urlopen() with a *hostname* URL, which:
    1. Calls getaddrinfo() for DNS resolution — blocking kernel wait, D-state
       if the DNS server is unreachable or the network is firewalled DROP.
    2. Falls through to socket.connect() — another potential D-state if DNS
       resolves but TCP SYN is dropped.
  Either step can cause D-state; the raw-IP test misses both.

RISK LEVEL
  ⚠️  LIKELY DRAIN — urllib + hostname URL on a DROP-firewalled node.
  Expected recovery: scontrol resume only (no GPU reset needed — no CUDA
  context involved before the download attempt).
  Keep admin contact ready before running.

WHAT IT DOES
  1. Single-process (no DDP — network D-state is independent of NCCL/GPU).
  2. Prints PID and hostname for /proc/<pid>/status monitoring.
  3. Tests phase 1: raw socket.connect() to bare IP (first run finding).
  4. Tests phase 2: urllib.request.urlopen() to a hostname URL — the actual
     original drain path (DNS + TCP + HTTP read).
  5. Never self-terminates. Must be killed via scancel.

HOW TO TEST
  1. Submit: sbatch mwe-03-network-hang.sbatch
  2. Wait for "ATTEMPTING URLLIB DOWNLOAD" in the log.
  3. Optionally SSH to the node and: cat /proc/<pid>/status | grep State
       D = disk sleep (uninterruptible) → drain risk confirmed
       S = sleeping (interruptible) → safe to kill
  4. Issue: scancel <jobid>
  5. After ~70 s: sinfo -N -n 98dci4-gpu-0003 --noheader -o "%T"
       "drain" → D-state confirmed, node drained
       "idle/reserved" → S-state, SIGKILL worked, no drain

ENVIRONMENT VARIABLES
  DOWNLOAD_URL        URL to fetch (default: original vgg model URL from Event #3)
  WAIT_BEFORE         Seconds before attempting download (default: 10)
  DRAIN_TEST_LOG_DIR  Log directory (default: /tmp)
"""

import os
import socket
import sys
import time
import urllib.request

# Default: the exact URL from Event #3 (vgg_lpips model from heibox.uni-heidelberg.de)
DOWNLOAD_URL = os.environ.get(
    "DOWNLOAD_URL",
    "https://heibox.uni-heidelberg.de/f/607503859d364012b37e/?dl=1",
)
WAIT_BEFORE = float(os.environ.get("WAIT_BEFORE", "10"))
LOG_DIR = os.environ.get("DRAIN_TEST_LOG_DIR", "/tmp")
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

hostname = socket.gethostname()
pid = os.getpid()

print(f"[MWE-03] hostname={hostname}  pid={pid}  job={JOB_ID}", flush=True)
print(f"[MWE-03] URL: {DOWNLOAD_URL}", flush=True)
print(f"[MWE-03] To monitor process state from compute node:", flush=True)
print(f"[MWE-03]   ssh {hostname}", flush=True)
print(f"[MWE-03]   cat /proc/{pid}/status | grep -E '^State|^Name'", flush=True)
print(f"[MWE-03]   # D = disk sleep (uninterruptible) → drain risk", flush=True)
print(f"[MWE-03]   # S = sleeping (interruptible) → safe to kill", flush=True)
print(flush=True)

# Phase 1: raw socket.connect() to bare IP (confirmed S-state in first run)
print("[MWE-03] Phase 1: raw socket.connect() to bare IP 103.21.244.0:443", flush=True)
print("[MWE-03] (First run showed this is S-state — included for comparison)", flush=True)
t0 = time.monotonic()
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(("103.21.244.0", 443))
    s.close()
    print(f"[MWE-03] Phase 1: connected (unexpected) after {time.monotonic()-t0:.2f}s", flush=True)
except OSError as e:
    print(f"[MWE-03] Phase 1: {type(e).__name__} after {time.monotonic()-t0:.3f}s: {e}", flush=True)

print(f"[MWE-03] Waiting {WAIT_BEFORE:.0f} s before urllib download...", flush=True)
time.sleep(WAIT_BEFORE)

# Phase 2: urllib with hostname URL — the actual original drain mechanism
print(f"[MWE-03] ATTEMPTING URLLIB DOWNLOAD from {DOWNLOAD_URL}", flush=True)
print(f"[MWE-03] This requires DNS lookup + TCP connect + HTTP read.", flush=True)
print(f"[MWE-03] On betelgeuse (no outbound network, DROP firewall): WILL HANG.", flush=True)
print(f"[MWE-03] Issue 'scancel {JOB_ID}' now, then check node state in 70 s.", flush=True)

t_start = time.monotonic()
try:
    # No timeout — mirrors how LPIPS loader called urllib in the original drain
    resp = urllib.request.urlopen(DOWNLOAD_URL)
    data = resp.read(1024)
    elapsed = time.monotonic() - t_start
    print(f"[MWE-03] UNEXPECTED: urlopen returned after {elapsed:.2f}s ({len(data)} bytes)", flush=True)
    print(f"[MWE-03] Host is reachable — this node has outbound network access.", flush=True)
    sys.exit(0)
except OSError as exc:
    elapsed = time.monotonic() - t_start
    print(f"[MWE-03] urlopen raised after {elapsed:.2f}s: {type(exc).__name__}: {exc}", flush=True)
    print(f"[MWE-03] Fast failure (REJECT/unreachable) — process stayed killable.", flush=True)
    sys.exit(0)
except KeyboardInterrupt:
    elapsed = time.monotonic() - t_start
    print(f"[MWE-03] KeyboardInterrupt after {elapsed:.2f}s", flush=True)
    sys.exit(0)
