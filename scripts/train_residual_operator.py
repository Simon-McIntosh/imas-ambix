#!/usr/bin/env python
"""Train the static residual operator on spine-manufactured labels.

Loads the label shards the factory wrote, splits them train/val BY SHOT
(both inside the corpus train split — the gate-eval cohort and the standing
held-out shots never appear here), and trains the sensor-token encoder to
emit profile-DOF corrections decoded through the exact Green's layer.

Loss (all EFIT-free):
* whitened masked sensor reconstruction — self-supervised against the raw
  magnetics, the same misfit convention the classical solve minimises;
* a correction leash ``leash * ||dc||²`` — consistency with the spine's
  manufactured labels (``dc = 0`` IS the spine solution, so the leash is the
  label-consistency term in DOF space);
* a clamp-activity penalty discouraging corrections that drive cell currents
  negative (the classical solve's unidirectional-current fact).

The leash is swept and selected on the VAL shots by the honest criterion:
best val sensor misfit subject to a bounded median boundary shift against the
spine's own boundary (no referee involved — leakage-free model selection).

Checkpoint + report: ``imas_ambix/latent/artifacts/residual_operator/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from imas_ambix.latent.profile_greens_decoder import ProfileGreensDecoder
from imas_ambix.latent.residual_operator import (
    ResidualOperator,
    load_label_shards,
    save_checkpoint,
    slice_globals,
    slice_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_residual_operator")

ARTIFACTS = Path("imas_ambix/latent/artifacts/residual_operator")


def build_examples(shards) -> list[dict]:
    """Flatten shards into per-slice training examples (numpy, lazily small).

    The spine's own sensor prediction (Green's layer on the label currents +
    vacuum + fitted passive signature) anchors the residual input channel and
    the reference misfit.
    """
    examples: list[dict] = []
    for sh in shards:
        a = sh.arrays
        n = sh.n_slices
        sr, sz = a["sensor_r"], a["sensor_z"]
        sang, is_flux = a["sensor_angle_deg"], a["is_flux"]
        scale = a["scale"]
        for k in range(n):
            examples.append(
                {
                    "shot": sh.shot,
                    "row": k,
                    "campaign": sh.meta.get("campaign", "?"),
                    "measured": a["measured"][k],
                    "vacuum": a["vacuum"][k],
                    "mask": a["mask"][k],
                    "sens_passive": a["sens_passive"][k],
                    "scale": scale,
                    "i_pf": a["i_pf"][k],
                    "ip": float(a["ip_amperes"][k]),
                    "n_e": float(a["n_e"][k]),
                    "i_cell": a["i_cell"][k].astype(np.float64),
                    "psi_n_cells": a["psi_n_cells"][k].astype(np.float64),
                    "sr": sr,
                    "sz": sz,
                    "sang": sang,
                    "is_flux": is_flux,
                }
            )
    return examples


def prepare_batch(examples, decoders, device, dtype=torch.float64):
    """Group per-campaign tensors for a list of examples (same campaign)."""
    camp = examples[0]["campaign"]
    dec = decoders[camp]
    basis = dec.basis
    m_sens = basis.m_sens.to(dtype)
    spine_pred = []
    tokens, masks, gl = [], [], []
    for e in examples:
        pred = m_sens.numpy() @ e["i_cell"] + e["vacuum"] + e["sens_passive"]
        spine_pred.append(pred)
        t, m = slice_tokens(
            e["measured"],
            e["vacuum"],
            pred,
            e["scale"],
            e["mask"],
            e["sr"],
            e["sz"],
            e["sang"],
            e["is_flux"],
        )
        tokens.append(t)
        masks.append(m)
        gl.append(slice_globals(e["ip"], e["n_e"]))
    batch = {
        "tokens": torch.tensor(np.stack(tokens), dtype=torch.float32, device=device),
        "token_mask": torch.tensor(np.stack(masks), device=device),
        "globals": torch.tensor(np.stack(gl), dtype=torch.float32, device=device),
        "measured": torch.tensor(
            np.nan_to_num(np.stack([e["measured"] for e in examples])),
            dtype=dtype,
            device=device,
        ),
        "vac_pass": torch.tensor(
            np.stack([e["vacuum"] + e["sens_passive"] for e in examples]),
            dtype=dtype,
            device=device,
        ),
        "scale": torch.tensor(
            np.stack([e["scale"] for e in examples]), dtype=dtype, device=device
        ),
        "mask": torch.tensor(
            np.stack([e["mask"] & np.isfinite(e["measured"]) for e in examples]),
            device=device,
        ),
        "i_cell0": torch.tensor(
            np.stack([e["i_cell"] for e in examples]), dtype=dtype, device=device
        ),
        "psi_n": torch.tensor(
            np.stack([e["psi_n_cells"] for e in examples]), dtype=dtype, device=device
        ),
        "ip": torch.tensor([e["ip"] for e in examples], dtype=dtype, device=device),
        "campaign": camp,
    }
    return batch


def batch_losses(model, decoders, batch):
    """(sensor_misfit, spine_misfit, leash_term, clamp_term) for one batch.

    ``sensor_misfit`` is the mean whitened squared residual per kept channel —
    identical convention to the solve's ``cost``.
    """
    dec = decoders[batch["campaign"]]
    dc = model(batch["tokens"], batch["token_mask"], batch["globals"]).to(
        batch["i_cell0"].dtype
    )
    columns = dec.profile_columns(batch["psi_n"], batch["ip"])
    raw = batch["i_cell0"] + torch.einsum("bnk,bk->bn", columns, dc)
    i_cell = dec.cell_currents(batch["i_cell0"], dc, columns, batch["ip"])
    pred = dec.sensors(i_cell) + batch["vac_pass"]
    w = batch["mask"].to(pred.dtype)
    r = (pred - batch["measured"]) / batch["scale"]
    n_keep = w.sum(dim=-1).clamp(min=1.0)
    misfit = ((r**2) * w).sum(dim=-1) / n_keep

    pred0 = dec.sensors(batch["i_cell0"]) + batch["vac_pass"]
    r0 = (pred0 - batch["measured"]) / batch["scale"]
    misfit0 = ((r0**2) * w).sum(dim=-1) / n_keep

    leash = (dc**2).sum(dim=-1)
    clamp = (torch.relu(-raw).sum(dim=-1) / batch["ip"]) ** 2
    return misfit, misfit0, leash, clamp, dc


def boundary_shift_cm(dec, batch, model, n_max: int = 32) -> float:
    """Median push-out LCFS shift [cm] of the corrected field vs the spine,
    on up to ``n_max`` slices of ``batch`` — the leakage-free drift monitor.

    Uses the plasma-only flux change on the grid added to the spine's ψ, so
    the coil/passive background cancels exactly.
    """
    from imas_ambix.latent.boundary_disc import ring_shift_rms
    from imas_ambix.latent.topology import lcfs_contour

    basis = dec.basis
    with torch.no_grad():
        dc = model(batch["tokens"], batch["token_mask"], batch["globals"]).to(
            batch["i_cell0"].dtype
        )
        columns = dec.profile_columns(batch["psi_n"], batch["ip"])
        i_cell = dec.cell_currents(batch["i_cell0"], dc, columns, batch["ip"])
    shifts = []
    rg = basis.grid_r.numpy()
    zg = basis.grid_z.numpy()
    lim_r, lim_z = batch["limiter"]
    for k in range(min(n_max, batch["i_cell0"].shape[0])):
        dpsi = basis._g_pg_np @ (i_cell[k].numpy() - batch["i_cell0"][k].numpy())
        psi_spine = batch["psi_grid"][k]
        axis = batch["axis_rz"][k]
        kw = dict(limiter_r=lim_r, limiter_z=lim_z, clip_legs=True)
        lc0 = lcfs_contour(psi_spine, rg, zg, axis, **kw)
        lc1 = lcfs_contour(
            psi_spine + dpsi.reshape(psi_spine.shape), rg, zg, axis, **kw
        )
        if lc0.found and lc1.found:
            shifts.append(100.0 * ring_shift_rms(lc0.ring, lc1.ring, axis))
    return float(np.median(shifts)) if shifts else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", type=str, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--leash-sweep", type=str, default="0.03,0.1,0.3,1.0")
    ap.add_argument("--clamp-weight", type=float, default=100.0)
    ap.add_argument("--boundary-shift-max-cm", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    shard_paths = sorted(Path(args.labels_dir).glob("shot_*.npz"))
    shards = load_label_shards(shard_paths)
    logger.info("%d shards, %d slices", len(shards), sum(s.n_slices for s in shards))

    # rebuild one decoder per campaign present in the corpus
    from imas_ambix.gs.geometry import build_table_for_shot
    from imas_ambix.latent.patch_basis import PatchBasis

    decoders: dict[str, ProfileGreensDecoder] = {}
    spine_cfg = shards[0].meta["spine_config"]["interior_solve"]
    nr, nz = int(shards[0].meta["nr"]), int(shards[0].meta["nz"])
    for sh in shards:
        camp = sh.meta["campaign"]
        if camp in decoders:
            continue
        basis = PatchBasis.from_table(
            build_table_for_shot(sh.shot), nr=nr, nz=nz, dtype=torch.float64
        )
        decoders[camp] = ProfileGreensDecoder(
            basis,
            n_p=int(spine_cfg["n_p"]),
            n_f=int(spine_cfg["n_f"]),
            kind=str(spine_cfg["profile_kind"]),
        )
    logger.info("campaign decoders: %s", list(decoders))

    # shot-level split inside the train corpus
    shots = sorted({s.shot for s in shards})
    n_val = max(1, int(round(args.val_fraction * len(shots))))
    val_shots = set(rng.choice(shots, size=n_val, replace=False).tolist())
    examples = build_examples(shards)
    train_ex = [e for e in examples if e["shot"] not in val_shots]
    val_ex = [e for e in examples if e["shot"] in val_shots]
    logger.info(
        "train %d slices / val %d slices (val shots %s)",
        len(train_ex),
        len(val_ex),
        sorted(val_shots),
    )

    by_shot = {s.shot: s for s in shards}

    def make_batches(ex_list, batch_size, shuffle):
        idx = np.arange(len(ex_list))
        if shuffle:
            rng.shuffle(idx)
        groups: dict[str, list[int]] = {}
        for i in idx:
            groups.setdefault(ex_list[i]["campaign"], []).append(i)
        for _camp, ids in groups.items():
            for j in range(0, len(ids), batch_size):
                yield [ex_list[i] for i in ids[j : j + batch_size]]

    def attach_val_geometry(batch, ex_chunk):
        psi, axes, lim = [], [], None
        for e in ex_chunk:
            a = by_shot[e["shot"]].arrays
            k = int(e["row"])
            psi.append(a["psi"][k].astype(np.float64))
            t = a["target"][k]
            axes.append((float(t[0]), float(t[1])))
            lim = (a["limiter_r"], a["limiter_z"])
        batch["psi_grid"] = psi
        batch["axis_rz"] = axes
        batch["limiter"] = lim

    leash_values = [float(v) for v in args.leash_sweep.split(",")]
    results = []
    states: dict[float, dict] = {}
    best = None
    for leash in leash_values:
        model = ResidualOperator(sum(int(spine_cfg[k]) for k in ("n_p", "n_f"))).to(
            args.device
        )
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        t0 = time.perf_counter()
        val_curve = []
        best_state, best_val, patience = None, float("inf"), 0
        for _epoch in range(args.epochs):
            model.train()
            for chunk in make_batches(train_ex, args.batch, shuffle=True):
                b = prepare_batch(chunk, decoders, args.device)
                misfit, _m0, l2, clamp, _dc = batch_losses(model, decoders, b)
                loss = (
                    misfit.mean() + leash * l2.mean() + args.clamp_weight * clamp.mean()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            vm, vm0 = [], []
            with torch.no_grad():
                for chunk in make_batches(val_ex, args.batch, shuffle=False):
                    b = prepare_batch(chunk, decoders, args.device)
                    misfit, m0, _l2, _cl, _dc = batch_losses(model, decoders, b)
                    vm.append(misfit.numpy())
                    vm0.append(m0.numpy())
            v = float(np.mean(np.concatenate(vm)))
            v0 = float(np.mean(np.concatenate(vm0)))
            val_curve.append(v)
            if v < best_val - 1e-6:
                best_val, patience = v, 0
                best_state = {
                    k: t.detach().clone() for k, t in model.state_dict().items()
                }
            else:
                patience += 1
            if patience >= 8:
                break
        model.load_state_dict(best_state)
        model.eval()
        # boundary-drift monitor on val slices (vs the spine's own boundary)
        shift_cm = []
        for chunk in make_batches(val_ex, args.batch, shuffle=False):
            b = prepare_batch(chunk, decoders, args.device)
            attach_val_geometry(b, chunk)
            shift_cm.append(boundary_shift_cm(decoders[b["campaign"]], b, model))
        shift = float(np.nanmedian(shift_cm))
        rec = {
            "leash": leash,
            "val_sensor_misfit": best_val,
            "val_spine_misfit": v0,
            "val_boundary_shift_median_cm": shift,
            "epochs_run": len(val_curve),
            "train_s": time.perf_counter() - t0,
        }
        results.append(rec)
        states[leash] = best_state
        logger.info("leash %.3g: %s", leash, rec)
        ok = shift <= args.boundary_shift_max_cm or not np.isfinite(shift)
        if ok and (best is None or best_val < best["val_sensor_misfit"]):
            best = rec | {"state": best_state}

    if best is None:  # every leash drifted the boundary — keep the tightest
        tight = max(results, key=lambda r: r["leash"])
        logger.warning("all leashes exceed boundary-shift bound; keeping tightest")
        best = tight | {"state": states[tight["leash"]]}

    model = ResidualOperator(sum(int(spine_cfg[k]) for k in ("n_p", "n_f")))
    model.load_state_dict(best["state"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report = {
        "labels_dir": str(args.labels_dir),
        "n_shards": len(shards),
        "n_slices": len(examples),
        "n_train_slices": len(train_ex),
        "n_val_slices": len(val_ex),
        "val_shots": sorted(int(s) for s in val_shots),
        "spine_config_sha256": shards[0].meta.get("spine_config_sha256"),
        "leash_sweep": results,
        "selected_leash": best["leash"],
        "selected_val_sensor_misfit": best["val_sensor_misfit"],
        "val_spine_misfit": best["val_spine_misfit"],
        "seed": args.seed,
    }
    save_checkpoint(
        ARTIFACTS / "residual_operator.pt",
        model,
        {"report": report, "spine_interior": spine_cfg},
    )
    (ARTIFACTS / "training_report.json").write_text(json.dumps(report, indent=2))
    logger.info("saved %s", ARTIFACTS / "residual_operator.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
