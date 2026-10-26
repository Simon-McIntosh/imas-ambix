#!/usr/bin/env python3
"""Audit the open-model checkpoint store under /work/projects/imas_gpu/agents.

Three phases, split so that the network work happens on a login node and every
size walk and hash runs on a compute node:

    fetch   resolve each checkpoint's Hugging Face repository and commit, then
            download the repository file listing at that commit
    audit   walk the store: sizes, shard presence against the index, first-shard
            SHA-256 against the repository LFS object, profile references and
            serving receipts
    control negative control for the completeness check: a scratch copy of one
            index naming an absent shard must be reported incomplete, and the
            same copy with every shard reachable must be reported complete

Nothing is written to the store and nothing is deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

STORE = Path("/work/projects/imas_gpu/agents")
HT_TREE = "https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"

# Dirs whose download metadata survives but whose huggingface hub cache does not.
# The reading was taken from the checkout's own README title; fetch verifies it
# against the repository's file listing before the row is written.
REPO_FALLBACK = {"deepseek-v4-1-flash-1017": "deepseek-ai/DeepSeek-V4.1-Flash-1017"}


def model_dirs(store: Path = STORE) -> list[Path]:
    return sorted(p for p in store.iterdir() if (p / "model").is_dir())


def _hf_json(url: str) -> object:
    req = urllib.request.Request(
        url, headers={"User-Agent": "imas-ambix-model-store-audit"}
    )
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)


def resolve_provenance(d: Path) -> dict:
    """Repository and commit: hub cache ref first, download metadata second."""
    out = {"repo": None, "commit": None, "source": None}
    hub = d / ".cache" / "hub"
    if hub.is_dir():
        hubs = [p for p in hub.iterdir() if p.name.startswith("models--")]
        if hubs:
            parts = hubs[0].name.split("--", 2)
            if len(parts) == 3:
                out["repo"] = f"{parts[1]}/{parts[2]}"
            ref = hubs[0] / "refs" / "main"
            if ref.is_file():
                out["commit"] = ref.read_text().strip()
            out["source"] = "hub cache refs/main"
    dl = d / "model" / ".cache" / "huggingface" / "download"
    if dl.is_dir():
        metas = sorted(dl.rglob("*.metadata"))
        if metas:
            lines = metas[0].read_text().splitlines()
            if lines and re.fullmatch(r"[0-9a-f]{40}", lines[0].strip()):
                out.setdefault("meta_commit", lines[0].strip())
                if not out["commit"]:
                    out["commit"] = lines[0].strip()
                    out["source"] = "download metadata commit"
    if not out["repo"] and d.name in REPO_FALLBACK:
        out["repo"] = REPO_FALLBACK[d.name]
        out["source"] = (out["source"] or "") + (
            ", repo from checkout README title"
            if out["source"]
            else "checkout README title"
        )
    return out


def cmd_fetch(args: argparse.Namespace) -> int:
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    index = {}
    for d in model_dirs():
        prov = resolve_provenance(d)
        row = dict(prov, tree_saved=False, tree_files=0, tree_error=None)
        if prov["repo"] and prov["commit"]:
            saved = outdir / f"{d.name}.tree.json"
            if not saved.exists():
                url = HT_TREE.format(repo=prov["repo"], rev=prov["commit"])
                try:
                    data = _hf_json(url)
                    saved.write_text(json.dumps(data))
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    row["tree_error"] = f"{type(exc).__name__}: {exc}"
            if saved.exists():
                tree = json.loads(saved.read_text())
                row["tree_saved"] = True
                row["tree_files"] = sum(1 for f in tree if f.get("type") == "file")
        else:
            row["tree_error"] = "no repository or commit resolved"
        index[d.name] = row
        print(
            f"{d.name:26s} repo={row['repo']} commit={(row['commit'] or '')[:10]} "
            f"files={row['tree_files']} err={row['tree_error']}",
            flush=True,
        )
    (outdir / "provenance.json").write_text(json.dumps(index, indent=2))
    return 0


def tree_map(outdir: Path, name: str) -> dict[str, dict]:
    p = outdir / f"{name}.tree.json"
    if not p.exists():
        return {}
    return {f["path"]: f for f in json.loads(p.read_text()) if f.get("type") == "file"}


def index_shards(d: Path) -> tuple[Path | None, list[str], int | None]:
    idx = sorted((d / "model").glob("*.safetensors.index.json"))
    if not idx:
        return None, [], None
    data = json.loads(idx[0].read_text())
    shards = sorted(set(data.get("weight_map", {}).values()))
    total = (data.get("metadata") or {}).get("total_size")
    return idx[0], shards, total


def completeness(model_dir: Path, shards: list[str], remote: dict[str, dict]) -> dict:
    """Every shard the index names is present, at the size the repository gives it."""
    missing, size_mismatch, size_unknown, present = [], [], [], 0
    for s in shards:
        f = model_dir / s
        if not f.is_file():
            missing.append(s)
            continue
        present += 1
        want = remote.get(s, {}).get("size")
        if want is None:
            size_unknown.append(s)
        elif f.stat().st_size != want:
            size_mismatch.append(
                {"shard": s, "local": f.stat().st_size, "repository": want}
            )
    return {
        "index_named": len(shards),
        "present": present,
        "missing": missing,
        "size_mismatch": size_mismatch,
        "size_unknown": size_unknown,
        "size_verified": bool(remote) and not size_unknown and not size_mismatch,
        "complete": not missing and not size_mismatch,
    }


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def du(path: Path, apparent: bool = True) -> int:
    flag = "-sb" if apparent else "-s"
    out = subprocess.run(
        ["du", flag, str(path)], capture_output=True, text=True, check=True
    )
    kb_or_b = int(out.stdout.split()[0])
    return kb_or_b if apparent else kb_or_b * 1024


def profile_refs() -> dict[str, list[str]]:
    """weights directory slug -> profiles and variants resolving there."""
    sys.path.insert(0, os.environ.get("AMBIX_REPO", "."))
    from imas_ambix.agent.profile import list_profiles, load_profile  # noqa: PLC0415

    refs: dict[str, list[str]] = {}
    for slug in list_profiles():
        try:
            prof = load_profile(slug)
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            refs.setdefault("__errors__", []).append(f"{slug}: {exc}")
            continue
        base = prof.weights_directory_slug
        refs.setdefault(base, []).append(slug)
        for cards, variant in (getattr(prof, "gpu_variants", None) or {}).items():
            wslug = (variant.get("model") or {}).get("weights_slug")
            if wslug and wslug != base:
                refs.setdefault(wslug, []).append(f"{slug}@{cards}gpu")
    return refs


def receipt_index(receipts_dir: Path) -> list[dict]:
    """Every receipt's header line, with the weights directory it actually served."""
    out = []
    if not receipts_dir.is_dir():
        return out
    for f in sorted(receipts_dir.iterdir()):
        try:
            with f.open() as fh:
                first = fh.readline()
            if not first.strip():
                continue
            head = json.loads(first)
        except Exception:  # noqa: BLE001 - a half-written receipt is not a finding
            continue
        out.append(
            {
                "file": f.name,
                "timestamp": head.get("timestamp"),
                "profile_slug": head.get("profile_slug"),
                "model_path": ((head.get("serve_config") or {}).get("model_path")),
                "bytes": f.stat().st_size,
            }
        )
    return out


def latest_of(rows: list[dict]) -> dict | None:
    rows = [r for r in rows if r.get("timestamp")]
    if not rows:
        return None
    rows.sort(key=lambda r: r["timestamp"])
    return rows[-1]


def cmd_audit(args: argparse.Namespace) -> int:
    trees = Path(args.trees)
    prov_all = json.loads((trees / "provenance.json").read_text())
    refs = profile_refs()
    receipts_dir = STORE / "receipts"
    receipts = receipt_index(receipts_dir)
    rows = []
    for d in model_dirs():
        name = d.name
        prov = prov_all.get(name, {})
        idx_path, shards, total = index_shards(d)
        remote = tree_map(trees, name)
        comp = (
            completeness(d / "model", shards, remote)
            if shards
            else {
                "index_named": 0,
                "present": 0,
                "missing": [],
                "size_mismatch": [],
                "size_unknown": [],
                "complete": False,
            }
        )
        first = shards[0] if shards else None
        first_sha = None
        first_lfs = None
        if first and (d / "model" / first).is_file():
            first_sha = sha256_file(d / "model" / first)
            r = remote.get(first) or {}
            first_lfs = (r.get("lfs") or {}).get("oid")
        local_bytes = sum(
            (d / "model" / s).stat().st_size
            for s in shards
            if (d / "model" / s).is_file()
        )
        row = {
            "dir": name,
            "path": str(d),
            "size_dir_apparent_bytes": du(d, True),
            "size_model_apparent_bytes": du(d / "model", True),
            "index": idx_path.name if idx_path else None,
            "index_total_size": total,
            "index_named_shards": len(shards),
            "safetensors_files_in_dir": len(list((d / "model").glob("*.safetensors"))),
            "safetensors_not_named_by_index": sorted(
                p.name
                for p in (d / "model").glob("*.safetensors")
                if p.name not in set(shards)
            ),
            "local_shard_bytes_sum": local_bytes,
            "repo": prov.get("repo"),
            "commit": prov.get("commit"),
            "commit_source": prov.get("source"),
            "completeness": comp,
            "first_shard": first,
            "first_shard_sha256": first_sha,
            "first_shard_repo_lfs": first_lfs,
            "first_shard_match": (
                None if not (first_sha and first_lfs) else first_sha == first_lfs
            ),
            "repository_listing_available": bool(remote),
            "remote_file_count": len(remote),
            "profiles": refs.get(name, []),
            "receipt_direct": latest_of(
                [
                    r
                    for r in receipts
                    if (r["model_path"] or "")
                    .rstrip("/")
                    .endswith(f"/agents/{name}/model")
                ]
            ),
            "receipt_via_profile": latest_of(
                [
                    r
                    for r in receipts
                    if r["profile_slug"]
                    in [p.split("@")[0] for p in refs.get(name, [])]
                ]
            ),
            "receipt_files_for_profile": sorted(
                {
                    r["file"]
                    for r in receipts
                    if r["profile_slug"]
                    in [p.split("@")[0] for p in refs.get(name, [])]
                }
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "dir",
                        "repo",
                        "commit",
                        "index_named_shards",
                        "first_shard_match",
                        "profiles",
                    )
                }
            ),
            flush=True,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"store": str(STORE), "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


def cmd_control(args: argparse.Namespace) -> int:
    """A scratch index naming an absent shard must be reported incomplete."""
    src = STORE / args.dir / "model"
    idx_path, shards, _ = index_shards(STORE / args.dir)
    if not idx_path:
        print(f"{args.dir} has no index")
        return 2
    remote = tree_map(Path(args.trees), args.dir)
    scratch = Path(args.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / "full" / "model").mkdir(parents=True)
    (scratch / "short" / "model").mkdir(parents=True)
    for arm in ("full", "short"):
        shutil.copy2(idx_path, scratch / arm / "model" / idx_path.name)
    # both arms reach the real shards by symlink, except the one the short arm omits
    for s in shards[1:]:
        os.symlink(src / s, scratch / "full" / "model" / s)
    for s in shards[1:]:
        target = scratch / "short" / "model" / s
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(src / s, target)
    omitted = shards[0]
    os.symlink(src / omitted, scratch / "full" / "model" / omitted)
    # the short arm simply has no entry for the first shard, which is the state a
    # download stopped one shard early leaves behind
    full = completeness(scratch / "full" / "model", shards, remote)
    short = completeness(scratch / "short" / "model", shards, remote)
    result = {
        "dir": args.dir,
        "omitted_shard": omitted,
        "full_arm": full,
        "short_arm": short,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"full arm complete={full['complete']} missing={len(full['missing'])} "
        f"size_mismatch={len(full['size_mismatch'])}"
    )
    print(
        f"short arm complete={short['complete']} missing={len(short['missing'])} "
        f"size_mismatch={len(short['size_mismatch'])}"
    )
    ok = full["complete"] is True and short["complete"] is False
    print("CONTROL PASS" if ok else "CONTROL FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("audit")
    p.add_argument("--trees", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("control")
    p.add_argument("--trees", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_control)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
