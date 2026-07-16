#!/usr/bin/env python
"""Train the temporal operator on spine-manufactured label sequences.

Each label shard is one shot's time-ordered, warm-chained slice sequence — the
natural training example for the causal trunk.  The operator sees the whole
sensor history up to each step (tokens, globals, Δt, and the physically
integrated L/R eddy state + drive) and emits per-step profile-DOF corrections
``dc`` plus eddy mode amplitudes ``da``; both decode to sensors through the
exact Green's layers (profile columns and passive eigenmode columns).

Loss (all EFIT-free): whitened masked sensor reconstruction + a profile
correction leash ``leash·||dc||²`` (dc = 0 IS the spine) + an eddy ridge
``ridge·||da/σ||²`` (da = 0 IS the spine's static passive fit) + the
clamp-activity penalty.  Selection on VAL shots by the honest criterion of the
static rung: best val sensor misfit subject to a bounded median boundary shift
against the spine's own boundary — the shift now includes the eddy flux term.

Optionally warm-starts from a synthetic-pretrained checkpoint
(``--init-checkpoint``) so the eddy pathway arrives with known-eddy structure.

Checkpoint + report: ``imas_ambix/latent/artifacts/temporal_operator/``.
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
    load_label_shards,
    slice_globals,
    slice_tokens,
)
from imas_ambix.latent.temporal_operator import (
    TemporalOperator,
    build_passive_eigenbasis,
    load_eigenbasis,
    physical_eddy_history,
    save_checkpoint,
    save_eigenbasis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_temporal_operator")

ARTIFACTS = Path("imas_ambix/latent/artifacts/temporal_operator")


def campaign_eigenbasis(
    campaign: str, shot: int, *, nr, nz, scale, k, cache_dir, n_channels=None
):
    """Load (or build + cache) the campaign's passive L/R eigenbasis.

    A cached basis whose sensor rows disagree with the canonical channel
    count (built from a lean-sensor shot before the canonical was known) is
    rebuilt from the canonical shot rather than trusted.
    """
    cache = Path(cache_dir) / f"eigenbasis-{campaign}-k{k}.npz"
    if cache.exists():
        basis = load_eigenbasis(cache)
        if n_channels is None or basis.a_sens.shape[0] == int(n_channels):
            return basis
        logger.warning(
            "eigenbasis %s cache has %d sensor rows, canonical %d — rebuilding",
            campaign,
            basis.a_sens.shape[0],
            n_channels,
        )
    from imas_ambix.gs.geometry import build_table_for_shot
    from imas_ambix.latent.gs_solve import EquilibriumGrid

    table = build_table_for_shot(int(shot))
    grid = EquilibriumGrid.from_table(table, nr=nr, nz=nz)
    t0 = time.perf_counter()
    basis = build_passive_eigenbasis(table, grid, sensor_scale=scale, k=k)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_eigenbasis(cache, basis)
    logger.info(
        "eigenbasis %s built in %.0f s (tau %.1f–%.2f ms) -> %s",
        campaign,
        time.perf_counter() - t0,
        basis.tau.max() * 1e3,
        basis.tau.min() * 1e3,
        cache,
    )
    return basis


def align_shard_channels(arrays: dict, shard_channels, canon_channels) -> dict:
    """Re-map a shard's sensor-space arrays onto the campaign's canonical
    channel list BY NAME (the alignment the harness applies everywhere).

    Channels absent on the shard are masked out (values 0); shard channels
    outside the canonical set are dropped.  Sensor geometry rows come from the
    canonical shard, so tokens stay geometry-consistent with the decoder.
    """
    if list(shard_channels) == list(canon_channels):
        return arrays
    row_of = {ch: i for i, ch in enumerate(shard_channels)}
    idx = np.array([row_of.get(ch, -1) for ch in canon_channels])
    present = idx >= 0
    out = dict(arrays)
    take = np.clip(idx, 0, None)
    for key in ("measured", "vacuum", "sens_passive"):
        out[key] = np.where(present, arrays[key][:, take], 0.0)
    out["mask"] = arrays["mask"][:, take] & present
    out["scale"] = np.where(present, arrays["scale"][take], 1.0)
    return out


def build_sequences(shards, decoders, eigenbases, canon) -> list[dict]:
    """One training sequence per shard (numpy; tensors made at batch time)."""
    sequences = []
    for sh in shards:
        camp = sh.meta["campaign"]
        dec = decoders[camp]
        canon_channels, sr, sz, sang, is_flux = canon[camp]
        a = align_shard_channels(sh.arrays, sh.meta.get("channels", []), canon_channels)
        m_sens = np.asarray(dec.basis.m_sens.cpu().numpy(), dtype=np.float64)
        n = sh.n_slices
        scale = a["scale"]
        i_cell = a["i_cell"].astype(np.float64)
        spine_pred = i_cell @ m_sens.T + a["vacuum"] + a["sens_passive"]  # (T, S)
        tokens, masks, gl = [], [], []
        for t in range(n):
            tk, m = slice_tokens(
                a["measured"][t],
                a["vacuum"][t],
                spine_pred[t],
                scale,
                a["mask"][t],
                sr,
                sz,
                sang,
                is_flux,
            )
            tokens.append(tk)
            masks.append(m)
            gl.append(slice_globals(float(a["ip_amperes"][t]), float(a["n_e"][t])))
        times = a["time_s"].astype(np.float64)
        a_phys, u_drive = physical_eddy_history(
            eigenbases[camp], times, a["i_pf"], i_cell
        )
        dt = np.diff(times, prepend=times[0])
        dt[0] = float(np.median(dt[1:])) if n > 1 else 1e-2
        sequences.append(
            {
                "shot": sh.shot,
                "campaign": camp,
                "n": n,
                "tokens": np.stack(tokens),  # (T, S, F)
                "token_mask": np.stack(masks),  # (T, S)
                "globals": np.stack(gl),  # (T, 2)
                "dt": dt.astype(np.float64),
                "a_phys": a_phys,
                "u_drive": u_drive,
                "measured": np.nan_to_num(a["measured"]),
                "vac_pass": a["vacuum"] + a["sens_passive"],
                "scale": scale,
                "mask": a["mask"] & np.isfinite(a["measured"]),
                "i_cell0": i_cell,
                "psi_n": a["psi_n_cells"].astype(np.float64),
                "ip": a["ip_amperes"].astype(np.float64),
                "psi": a["psi"].astype(np.float64),
                "target": a["target"],
                "limiter": (a["limiter_r"], a["limiter_z"]),
            }
        )
    return sequences


def robust_std(x: np.ndarray) -> np.ndarray:
    """1.4826·MAD per mode over all (shot, step) samples, floored."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    return np.clip(1.4826 * mad, 1e-12 * max(np.abs(x).max(), 1e-30) + 1e-30, None)


def pad_batch(seqs: list[dict], device, dtype=torch.float64) -> dict:
    """Stack same-campaign sequences with trailing padding."""
    t_max = max(s["n"] for s in seqs)
    b = len(seqs)
    s_dim, f_dim = seqs[0]["tokens"].shape[1:]
    n_cells = seqs[0]["i_cell0"].shape[1]
    n_sens = seqs[0]["measured"].shape[1]
    k = seqs[0]["a_phys"].shape[1]

    def zeros(*shape, dt=np.float64):
        return np.zeros(shape, dtype=dt)

    out = {
        "tokens": zeros(b, t_max, s_dim, f_dim, dt=np.float32),
        "token_mask": zeros(b, t_max, s_dim, dt=bool),
        "globals": zeros(b, t_max, 2, dt=np.float32),
        "dt": np.full((b, t_max), 1e-2),
        "a_phys": zeros(b, t_max, k, dt=np.float32),
        "u_drive": zeros(b, t_max, k, dt=np.float32),
        "measured": zeros(b, t_max, n_sens),
        "vac_pass": zeros(b, t_max, n_sens),
        "scale": zeros(b, t_max, n_sens),
        "mask": zeros(b, t_max, n_sens, dt=bool),
        "i_cell0": zeros(b, t_max, n_cells),
        "psi_n": zeros(b, t_max, n_cells),
        "ip": np.ones((b, t_max)),
        "pad": np.ones((b, t_max), dtype=bool),
    }
    for i, s in enumerate(seqs):
        n = s["n"]
        out["tokens"][i, :n] = s["tokens"]
        out["token_mask"][i, :n] = s["token_mask"]
        out["globals"][i, :n] = s["globals"]
        out["dt"][i, :n] = s["dt"]
        out["a_phys"][i, :n] = s["a_phys"]
        out["u_drive"][i, :n] = s["u_drive"]
        out["measured"][i, :n] = s["measured"]
        out["vac_pass"][i, :n] = s["vac_pass"]
        out["scale"][i, :n] = np.broadcast_to(s["scale"], (n, n_sens))
        out["mask"][i, :n] = s["mask"]
        out["i_cell0"][i, :n] = s["i_cell0"]
        out["psi_n"][i, :n] = s["psi_n"]
        out["ip"][i, :n] = s["ip"]
        out["pad"][i, :n] = False
    tensors = {}
    for key, val in out.items():
        if key in ("tokens", "globals", "a_phys", "u_drive"):
            tensors[key] = torch.tensor(val, dtype=torch.float32, device=device)
        elif key in ("token_mask", "mask", "pad"):
            tensors[key] = torch.tensor(val, device=device)
        elif key == "dt":
            tensors[key] = torch.tensor(val, dtype=torch.float32, device=device)
        else:
            tensors[key] = torch.tensor(val, dtype=dtype, device=device)
    tensors["campaign"] = seqs[0]["campaign"]
    return tensors


def batch_losses(model, decoders, eddy_sens, batch):
    """(sensor misfit, spine misfit, dc leash, eddy ridge, clamp) per kept step."""
    dec = decoders[batch["campaign"]]
    dc, da = model(
        batch["tokens"],
        batch["token_mask"],
        batch["globals"],
        batch["dt"],
        batch["a_phys"],
        batch["u_drive"],
        pad_mask=batch["pad"],
    )
    dc = dc.to(batch["i_cell0"].dtype)
    da = da.to(batch["i_cell0"].dtype)
    b, t, n_cells = batch["i_cell0"].shape
    flat = lambda x: x.reshape(b * t, *x.shape[2:])  # noqa: E731

    columns = dec.profile_columns(flat(batch["psi_n"]), flat(batch["ip"]))
    raw = flat(batch["i_cell0"]) + torch.einsum("bnk,bk->bn", columns, flat(dc))
    i_cell = dec.cell_currents(
        flat(batch["i_cell0"]), flat(dc), columns, flat(batch["ip"])
    )
    a_eddy = eddy_sens[batch["campaign"]]  # (S, k) fp64 tensor
    pred = dec.sensors(i_cell) + flat(batch["vac_pass"]) + flat(da) @ a_eddy.T
    keep_step = (~batch["pad"]).reshape(b * t).to(pred.dtype)
    w = flat(batch["mask"]).to(pred.dtype) * keep_step.unsqueeze(-1)
    r = (pred - flat(batch["measured"])) / flat(batch["scale"]).clamp(min=1e-12)
    n_keep = w.sum(dim=-1).clamp(min=1.0)
    misfit = ((r**2) * w).sum(dim=-1) / n_keep

    pred0 = dec.sensors(flat(batch["i_cell0"])) + flat(batch["vac_pass"])
    r0 = (pred0 - flat(batch["measured"])) / flat(batch["scale"]).clamp(min=1e-12)
    misfit0 = ((r0**2) * w).sum(dim=-1) / n_keep

    leash = (flat(dc) ** 2).sum(dim=-1) * keep_step
    da_std = flat(da) / model.eddy_std.to(pred.dtype)
    ridge = (da_std**2).sum(dim=-1) * keep_step
    clamp = ((torch.relu(-raw).sum(dim=-1) / flat(batch["ip"])) ** 2) * keep_step
    misfit = misfit * keep_step
    misfit0 = misfit0 * keep_step
    n_valid = keep_step.sum().clamp(min=1.0)
    return (
        misfit.sum() / n_valid,
        misfit0.sum() / n_valid,
        leash.sum() / n_valid,
        ridge.sum() / n_valid,
        clamp.sum() / n_valid,
    )


def boundary_shift_cm(model, decoders, eigenbases, eddy_sens, seqs, device) -> float:
    """Median push-out LCFS shift [cm] vs the spine's own boundary (val monitor).

    The corrected plasma-only flux change PLUS the eddy mode flux is added to
    the spine's ψ, so the coil/passive-static background cancels exactly.
    """
    from imas_ambix.latent.boundary_disc import ring_shift_rms
    from imas_ambix.latent.topology import lcfs_contour

    shifts = []
    for s in seqs:
        dec = decoders[s["campaign"]]
        basis = dec.basis
        eb = eigenbases[s["campaign"]]
        batch = pad_batch([s], device)
        with torch.no_grad():
            dc, da = model(
                batch["tokens"],
                batch["token_mask"],
                batch["globals"],
                batch["dt"],
                batch["a_phys"],
                batch["u_drive"],
                pad_mask=batch["pad"],
            )
            dc = dc[0].to(torch.float64)
            da = da[0].to(torch.float64)
            columns = dec.profile_columns(batch["psi_n"][0], batch["ip"][0])
            i_cell = dec.cell_currents(batch["i_cell0"][0], dc, columns, batch["ip"][0])
        di = (i_cell - batch["i_cell0"][0]).cpu().numpy()
        da_np = da.cpu().numpy()
        rg = basis.grid_r.cpu().numpy()
        zg = basis.grid_z.cpu().numpy()
        lim_r, lim_z = s["limiter"]
        kw = dict(limiter_r=lim_r, limiter_z=lim_z, clip_legs=True)
        # drift is bounded where the spine is already good (quasi-flat-top);
        # early-shot boundary movement is the operator's whole job, not drift
        ip_seq = np.asarray(s["ip"][: s["n"]], dtype=np.float64)
        flattopish = np.flatnonzero(ip_seq >= 0.7 * ip_seq.max())[:8]
        for t in flattopish:
            dpsi = basis._g_pg_np @ di[t] + eb.g_grid @ da_np[t]
            psi_spine = s["psi"][t]
            axis = (float(s["target"][t][0]), float(s["target"][t][1]))
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
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-shots", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--leash-sweep", type=str, default="3.0,10.0")
    ap.add_argument("--eddy-ridge-sweep", type=str, default="0.3,3.0")
    ap.add_argument("--clamp-weight", type=float, default=100.0)
    ap.add_argument("--boundary-shift-max-cm", type=float, default=0.75)
    ap.add_argument("--k-modes", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--init-checkpoint", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out-suffix", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    shard_paths = sorted(Path(args.labels_dir).glob("shot_*.npz"))
    shards = load_label_shards(shard_paths)
    complete = [s for s in shards if "limiter_r" in s.arrays]
    if len(complete) < len(shards):
        logger.warning(
            "%d shard(s) incomplete (mid-write / truncated) — skipped",
            len(shards) - len(complete),
        )
    shards = complete
    logger.info("%d shards, %d slices", len(shards), sum(s.n_slices for s in shards))

    from imas_ambix.gs.geometry import build_table_for_shot
    from imas_ambix.latent.patch_basis import PatchBasis

    decoders: dict[str, ProfileGreensDecoder] = {}
    eigenbases: dict = {}
    canon: dict[str, tuple] = {}  # campaign → canonical channels + geometry
    spine_cfg = shards[0].meta["spine_config"]["interior_solve"]
    nr, nz = int(shards[0].meta["nr"]), int(shards[0].meta["nz"])
    # canonical shard per campaign = the richest channel set (channels absent
    # on other shards are masked; a lean canonical would DROP live channels)
    canon_shard: dict[str, object] = {}
    for sh in shards:
        camp = sh.meta["campaign"]
        best = canon_shard.get(camp)
        if best is None or len(sh.meta.get("channels", [])) > len(
            best.meta.get("channels", [])
        ):
            canon_shard[camp] = sh
    for sh in canon_shard.values():
        camp = sh.meta["campaign"]
        basis = PatchBasis.from_table(
            build_table_for_shot(sh.shot), nr=nr, nz=nz, dtype=torch.float64
        )
        decoders[camp] = ProfileGreensDecoder(
            basis,
            n_p=int(spine_cfg["n_p"]),
            n_f=int(spine_cfg["n_f"]),
            kind=str(spine_cfg["profile_kind"]),
        )
        if basis.m_sens.shape[0] != len(sh.meta.get("channels", [])):
            raise SystemExit(
                f"campaign {camp}: decoder sensor rows "
                f"{basis.m_sens.shape[0]} != canonical shard channels "
                f"{len(sh.meta.get('channels', []))} (shot {sh.shot})"
            )
        canon[camp] = (
            list(sh.meta["channels"]),
            sh.arrays["sensor_r"],
            sh.arrays["sensor_z"],
            sh.arrays["sensor_angle_deg"],
            sh.arrays["is_flux"],
        )
        eigenbases[camp] = campaign_eigenbasis(
            camp,
            sh.shot,
            nr=nr,
            nz=nz,
            scale=sh.arrays["scale"],
            k=args.k_modes,
            cache_dir=ARTIFACTS,
            n_channels=len(sh.meta["channels"]),
        )
    logger.info("campaigns: %s", list(decoders))

    sequences = build_sequences(shards, decoders, eigenbases, canon)
    for camp in decoders:  # Green's buffers to the training device (fp64 matmuls)
        decoders[camp] = decoders[camp].to(args.device)
    shots = sorted({s["shot"] for s in sequences})
    n_val = max(1, int(round(args.val_fraction * len(shots))))
    val_shots = set(rng.choice(shots, size=n_val, replace=False).tolist())
    train_seq = [s for s in sequences if s["shot"] not in val_shots]
    val_seq = [s for s in sequences if s["shot"] in val_shots]
    if not train_seq:
        logger.warning("too few shards for a shot split — train == val (smoke)")
        train_seq = val_seq
    logger.info(
        "train %d shots (%d slices) / val %d shots (%d slices)",
        len(train_seq),
        sum(s["n"] for s in train_seq),
        len(val_seq),
        sum(s["n"] for s in val_seq),
    )

    # per-mode standardisation from the TRAIN sequences only
    a_all = np.concatenate([s["a_phys"] for s in train_seq])
    u_all = np.concatenate([s["u_drive"] for s in train_seq])
    eddy_std, drive_std = robust_std(a_all), robust_std(u_all)
    tau_init = next(iter(eigenbases.values())).tau
    logger.info(
        "eddy_std %s  drive_std %s", np.round(eddy_std, 3), np.round(drive_std, 4)
    )

    eddy_sens = {
        camp: torch.tensor(eb.a_sens, dtype=torch.float64, device=args.device)
        for camp, eb in eigenbases.items()
    }

    def make_batches(seq_list, shuffle):
        idx = np.arange(len(seq_list))
        if shuffle:
            rng.shuffle(idx)
        groups: dict[str, list[int]] = {}
        for i in idx:
            groups.setdefault(seq_list[i]["campaign"], []).append(i)
        for _c, ids in groups.items():
            for j in range(0, len(ids), args.batch_shots):
                yield [seq_list[i] for i in ids[j : j + args.batch_shots]]

    def new_model() -> TemporalOperator:
        model = TemporalOperator(
            sum(int(spine_cfg[k]) for k in ("n_p", "n_f")),
            tau_init,
            eddy_std,
            drive_std,
            d_model=args.d_model,
            n_layers=args.n_layers,
        ).to(args.device)
        if args.init_checkpoint:
            ckpt = torch.load(
                args.init_checkpoint, map_location="cpu", weights_only=False
            )
            model.load_state_dict(ckpt["state_dict"])
            # keep the REAL-corpus feature standardisation — the checkpoint's
            # buffers carry the synthetic corpus scales
            with torch.no_grad():
                model.eddy_std.copy_(torch.as_tensor(eddy_std, dtype=torch.float32))
                model.drive_std.copy_(torch.as_tensor(drive_std, dtype=torch.float32))
            model = model.to(args.device)
            logger.info("warm-started from %s", args.init_checkpoint)
        return model

    leash_values = [float(v) for v in args.leash_sweep.split(",")]
    ridge_values = [float(v) for v in args.eddy_ridge_sweep.split(",")]
    results, states, best = [], {}, None
    for leash in leash_values:
        for ridge in ridge_values:
            model = new_model()
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            t0 = time.perf_counter()
            best_state, best_val, patience = None, float("inf"), 0
            epochs_run = 0
            for _epoch in range(args.epochs):
                epochs_run += 1
                model.train()
                for chunk in make_batches(train_seq, shuffle=True):
                    b = pad_batch(chunk, args.device)
                    misfit, _m0, l2, rd, clamp = batch_losses(
                        model, decoders, eddy_sens, b
                    )
                    loss = misfit + leash * l2 + ridge * rd + args.clamp_weight * clamp
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                vm, vm0 = [], []
                with torch.no_grad():
                    for chunk in make_batches(val_seq, shuffle=False):
                        b = pad_batch(chunk, args.device)
                        misfit, m0, _l2, _rd, _cl = batch_losses(
                            model, decoders, eddy_sens, b
                        )
                        vm.append(float(misfit))
                        vm0.append(float(m0))
                v, v0 = float(np.mean(vm)), float(np.mean(vm0))
                if v < best_val - 1e-6:
                    best_val, patience = v, 0
                    best_state = {
                        k: t.detach().clone() for k, t in model.state_dict().items()
                    }
                else:
                    patience += 1
                if patience >= args.patience:
                    break
            model.load_state_dict(best_state)
            model.eval()
            shift = boundary_shift_cm(
                model, decoders, eigenbases, eddy_sens, val_seq, args.device
            )
            rec = {
                "leash": leash,
                "eddy_ridge": ridge,
                "val_sensor_misfit": best_val,
                "val_spine_misfit": v0,
                "val_boundary_shift_median_cm": shift,
                "epochs_run": epochs_run,
                "train_s": time.perf_counter() - t0,
                "tau_learned_ms": (
                    1e3 * torch.exp(model.log_tau).detach().cpu().numpy()
                ).tolist(),
            }
            results.append(rec)
            states[(leash, ridge)] = best_state
            logger.info("leash %.3g ridge %.3g: %s", leash, ridge, rec)
            ok = shift <= args.boundary_shift_max_cm or not np.isfinite(shift)
            if ok and (best is None or best_val < best["val_sensor_misfit"]):
                best = rec | {"state": best_state}

    if best is None:  # every combo drifted the boundary — keep the tightest
        tight = max(results, key=lambda r: (r["leash"], r["eddy_ridge"]))
        logger.warning("all sweep points exceed boundary-shift bound; keep tightest")
        best = tight | {"state": states[(tight["leash"], tight["eddy_ridge"])]}

    model = TemporalOperator(
        sum(int(spine_cfg[k]) for k in ("n_p", "n_f")),
        tau_init,
        eddy_std,
        drive_std,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    model.load_state_dict(best["state"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report = {
        "labels_dir": str(args.labels_dir),
        "n_shards": len(shards),
        "n_slices": int(sum(s.n_slices for s in shards)),
        "n_train_shots": len(train_seq),
        "n_val_shots": len(val_seq),
        "val_shots": sorted(int(s) for s in val_shots),
        "spine_config_sha256": shards[0].meta.get("spine_config_sha256"),
        "k_modes": args.k_modes,
        "tau_init_ms": (1e3 * tau_init).tolist(),
        "sweep": results,
        "selected": {k: v for k, v in best.items() if k != "state"},
        "init_checkpoint": args.init_checkpoint,
        "seed": args.seed,
    }
    tag = f"-{args.out_suffix}" if args.out_suffix else ""
    save_checkpoint(
        ARTIFACTS / f"temporal_operator{tag}.pt",
        model,
        {
            "report": report,
            "spine_interior": spine_cfg,
            "eddy_std": eddy_std,
            "drive_std": drive_std,
            "tau_init": tau_init,
        },
    )
    (ARTIFACTS / f"training_report{tag}.json").write_text(json.dumps(report, indent=2))
    logger.info("saved %s", ARTIFACTS / f"temporal_operator{tag}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
