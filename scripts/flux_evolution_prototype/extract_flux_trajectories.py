#!/usr/bin/env python3
"""Extract flux-function trajectories and geometry receipts from labeled maps.

The challenge maps store poloidal flux per radian.  They are converted to
Nova's total-flux convention before applying the Grad-Shafranov identities.
For each normalized-flux annulus, Delta-star is separated by its R-squared
dependence to recover pressure-gradient and diamagnetic-gradient terms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from nova.equilibrium.flux_surface_connectivity import traced_flux_surface_bins
from scipy.constants import mu_0
from scipy.ndimage import distance_transform_edt, label, map_coordinates

jax.config.update("jax_enable_x64", True)

TWO_PI = 2.0 * np.pi
SURFACE_COUNT = 24
SURFACE_MIN = 0.05
SURFACE_MAX = 0.95
RHO_SAMPLES = np.linspace(0.08, 0.96, 24)

LABEL_COLUMNS = (
    "efit_times",
    "efit_psirz",
    "efit_grid_R",
    "efit_grid_Z",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
)


def _sample_map(field: np.ndarray, radius: np.ndarray, height: np.ndarray, r, z):
    sample_radius = np.asarray(r)
    sample_height = np.asarray(z)
    result_shape = sample_radius.shape
    r_index = (sample_radius.reshape(-1) - radius[0]) / (radius[1] - radius[0])
    z_index = (sample_height.reshape(-1) - height[0]) / (height[1] - height[0])
    values = map_coordinates(
        field, np.vstack((z_index, r_index)), order=1, mode="nearest"
    )
    if result_shape == ():
        return float(values[0])
    return values.reshape(result_shape)


def _axis_connected_core(psi_normalised: np.ndarray, axis_index: tuple[int, int]):
    confined = psi_normalised < 1.0
    components, _ = label(confined)
    component = components[axis_index]
    if component == 0:
        return np.zeros_like(confined, dtype=bool)
    return components == component


def _delta_star(total_flux: np.ndarray, radius: np.ndarray, height: np.ndarray):
    derivative_z, derivative_r = np.gradient(total_flux, height, radius, edge_order=2)
    second_z = np.gradient(derivative_z, height, axis=0, edge_order=2)
    second_r = np.gradient(derivative_r, radius, axis=1, edge_order=2)
    return second_r - derivative_r / radius[np.newaxis, :] + second_z


def _surface_profiles(
    delta_star: np.ndarray,
    psi_normalised: np.ndarray,
    core: np.ndarray,
    radius: np.ndarray,
):
    levels = np.linspace(SURFACE_MIN, SURFACE_MAX, SURFACE_COUNT + 1)
    centres = 0.5 * (levels[:-1] + levels[1:])
    mesh_radius = np.broadcast_to(radius[np.newaxis, :], delta_star.shape)
    interior = np.zeros_like(core)
    interior[2:-2, 2:-2] = True
    p_prime = np.full(SURFACE_COUNT, np.nan)
    ff_prime = np.full(SURFACE_COUNT, np.nan)
    current_mean = np.full(SURFACE_COUNT, np.nan)
    fit_fraction = np.full(SURFACE_COUNT, np.nan)
    for index, (low, high) in enumerate(zip(levels[:-1], levels[1:], strict=True)):
        mask = core & interior & (psi_normalised >= low) & (psi_normalised < high)
        if np.count_nonzero(mask) < 12:
            continue
        radius_squared = mesh_radius[mask] ** 2
        source = delta_star[mask] / (4.0 * np.pi**2)
        design = np.column_stack((radius_squared, np.ones_like(radius_squared)))
        coefficients, _, _, _ = np.linalg.lstsq(design, source, rcond=None)
        predicted = design @ coefficients
        residual = source - predicted
        variance = np.sum((source - np.mean(source)) ** 2)
        p_prime[index] = coefficients[0] / mu_0
        ff_prime[index] = coefficients[1]
        fit_fraction[index] = 1.0 - np.sum(residual**2) / max(variance, 1e-30)
        current = -delta_star[mask] / (TWO_PI * mu_0 * mesh_radius[mask])
        current_mean[index] = np.mean(current)
    return centres, p_prime, ff_prime, current_mean, fit_fraction


def _batched_geometry(flux, axis_flux, boundary_flux, radius, height):
    inside = jnp.ones((height.size, radius.size), dtype=bool)

    def one_frame(frame_flux, frame_axis_flux, frame_boundary_flux):
        return traced_flux_surface_bins(
            frame_flux,
            radius,
            height,
            inside,
            frame_axis_flux,
            frame_boundary_flux,
            jnp.asarray(SURFACE_MIN),
            jnp.asarray(SURFACE_MAX),
            SURFACE_COUNT,
        )

    return jax.vmap(one_frame, in_axes=(0, 0, 0))(flux, axis_flux, boundary_flux)


_BATCHED_GEOMETRY = jax.jit(_batched_geometry)


def _geometry_records(maps, axis_flux, boundary_flux, radius, height):
    batch_size = 64
    chunks: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(maps), batch_size):
        stop = min(start + batch_size, len(maps))
        count = stop - start
        pad = batch_size - count
        batch_maps = np.pad(maps[start:stop], ((0, pad), (0, 0), (0, 0)), mode="edge")
        batch_axis = np.pad(axis_flux[start:stop], (0, pad), mode="edge")
        batch_boundary = np.pad(boundary_flux[start:stop], (0, pad), mode="edge")
        result = _BATCHED_GEOMETRY(
            jnp.asarray(batch_maps),
            jnp.asarray(batch_axis),
            jnp.asarray(batch_boundary),
            jnp.asarray(radius),
            jnp.asarray(height),
        )
        for key, value in result.items():
            chunks.setdefault(key, []).append(np.asarray(value)[:count])
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _vacuum_receipt(delta_star, core):
    interior = np.zeros_like(core)
    interior[3:-3, 3:-3] = True
    vacuum = (~core) & interior & (distance_transform_edt(~core) >= 3.0)
    plasma = core & interior
    if np.count_nonzero(vacuum) < 30 or np.count_nonzero(plasma) < 30:
        return np.nan, False, int(np.count_nonzero(vacuum))
    vacuum_scale = float(np.nanquantile(np.abs(delta_star[vacuum]), 0.90))
    plasma_scale = float(np.nanquantile(np.abs(delta_star[plasma]), 0.90))
    ratio = vacuum_scale / max(plasma_scale, 1e-30)
    return ratio, bool(ratio <= 0.25), int(np.count_nonzero(vacuum))


def _rho_from_geometry(inverse_radius_squared, volume_derivative):
    density = np.clip(inverse_radius_squared * volume_derivative, 0.0, None)
    increments = 0.5 * (density[:, 1:] + density[:, :-1])
    cumulative = np.concatenate(
        (np.zeros((density.shape[0], 1)), np.cumsum(increments, axis=1)), axis=1
    )
    total = cumulative[:, -1:]
    normalised = np.divide(
        cumulative, total, out=np.zeros_like(cumulative), where=total > 0
    )
    return np.sqrt(np.clip(normalised, 0.0, 1.0))


def extract_shot(path: Path, output: Path) -> dict[str, object]:
    row = pq.read_table(path, columns=list(LABEL_COLUMNS)).to_pylist()[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    maps = TWO_PI * np.asarray(row["efit_psirz"], dtype=np.float64)
    radius = np.asarray(row["efit_grid_R"], dtype=np.float64)
    height = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    axis_radius = np.asarray(row["efit_r_axis"], dtype=np.float64)
    axis_height = np.asarray(row["efit_z_axis"], dtype=np.float64)
    boundary_count = np.asarray(row["efit_lcfs_n"], dtype=np.int64)
    boundary_radius = np.asarray(row["efit_lcfs_r"], dtype=np.float64)
    boundary_height = np.asarray(row["efit_lcfs_z"], dtype=np.float64)

    frame_count = len(times)
    axis_flux = np.empty(frame_count)
    boundary_flux = np.empty(frame_count)
    normalised_maps = np.empty_like(maps)
    cores = np.empty_like(maps, dtype=bool)
    profile_psi = np.empty((frame_count, SURFACE_COUNT))
    p_prime = np.empty_like(profile_psi)
    ff_prime = np.empty_like(profile_psi)
    current_mean = np.empty_like(profile_psi)
    fit_fraction = np.empty_like(profile_psi)
    vacuum_ratio = np.empty(frame_count)
    vacuum_pass = np.empty(frame_count, dtype=bool)
    vacuum_cells = np.empty(frame_count, dtype=np.int64)

    for frame in range(frame_count):
        axis_flux[frame] = float(
            _sample_map(
                maps[frame], radius, height, axis_radius[frame], axis_height[frame]
            )
        )
        count = int(boundary_count[frame])
        boundary_values = _sample_map(
            maps[frame],
            radius,
            height,
            boundary_radius[frame, :count],
            boundary_height[frame, :count],
        )
        boundary_flux[frame] = float(np.nanmedian(boundary_values))
        span = boundary_flux[frame] - axis_flux[frame]
        normalised_maps[frame] = (maps[frame] - axis_flux[frame]) / span
        r_index = int(np.argmin(np.abs(radius - axis_radius[frame])))
        z_index = int(np.argmin(np.abs(height - axis_height[frame])))
        cores[frame] = _axis_connected_core(normalised_maps[frame], (z_index, r_index))
        delta = _delta_star(maps[frame], radius, height)
        (
            profile_psi[frame],
            p_prime[frame],
            ff_prime[frame],
            current_mean[frame],
            fit_fraction[frame],
        ) = _surface_profiles(delta, normalised_maps[frame], cores[frame], radius)
        vacuum_ratio[frame], vacuum_pass[frame], vacuum_cells[frame] = _vacuum_receipt(
            delta, cores[frame]
        )

    geometry = _geometry_records(maps, axis_flux, boundary_flux, radius, height)
    rho_hat = _rho_from_geometry(geometry["inv_r2"], geometry["dv_dpn"])
    psi_on_rho = np.full((frame_count, len(RHO_SAMPLES)), np.nan)
    p_prime_on_rho = np.full_like(psi_on_rho, np.nan)
    ff_prime_on_rho = np.full_like(psi_on_rho, np.nan)
    for frame in range(frame_count):
        good = np.isfinite(rho_hat[frame]) & np.isfinite(p_prime[frame])
        if np.count_nonzero(good) < 4:
            continue
        order = np.argsort(rho_hat[frame, good])
        rho = rho_hat[frame, good][order]
        pn = profile_psi[frame, good][order]
        psi_on_rho[frame] = np.interp(
            RHO_SAMPLES,
            rho,
            axis_flux[frame] + pn * (boundary_flux[frame] - axis_flux[frame]),
            left=np.nan,
            right=np.nan,
        )
        p_prime_on_rho[frame] = np.interp(
            RHO_SAMPLES, rho, p_prime[frame, good][order], left=np.nan, right=np.nan
        )
        ff_prime_on_rho[frame] = np.interp(
            RHO_SAMPLES, rho, ff_prime[frame, good][order], left=np.nan, right=np.nan
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        times_ms=times,
        radius_m=radius,
        height_m=height,
        axis_flux_wb=axis_flux,
        boundary_flux_wb=boundary_flux,
        surface_psi_n=profile_psi,
        surface_rho_hat=rho_hat,
        rho_hat_samples=RHO_SAMPLES,
        psi_on_rho_wb=psi_on_rho,
        p_prime=p_prime,
        ff_prime=ff_prime,
        p_prime_on_rho=p_prime_on_rho,
        ff_prime_on_rho=ff_prime_on_rho,
        j_phi_surface_mean=current_mean,
        separation_fit_fraction=fit_fraction,
        vacuum_ratio=vacuum_ratio,
        vacuum_pass=vacuum_pass,
        vacuum_cell_count=vacuum_cells,
        fsa_psi_n=geometry["pn_s"],
        fsa_dv_dpsi_n=geometry["dv_dpn"],
        fsa_inverse_r2=geometry["inv_r2"],
        fsa_inverse_r=geometry["inv_r"],
        fsa_gradient2_over_r2=geometry["grad2_r2"],
        fsa_cumulative_volume=geometry["v_cum"],
        fsa_volume=geometry["v_total"],
        fsa_core_fraction=geometry["core_fraction"],
        fsa_core_cells=geometry["n_core_cells"],
        fsa_well_posed=geometry["well_posed"],
    )
    return {
        "shot": path.stem,
        "frames": frame_count,
        "vacuum_pass_frames": int(np.count_nonzero(vacuum_pass)),
        "vacuum_pass_fraction": float(np.mean(vacuum_pass)),
        "vacuum_ratio_median": float(np.nanmedian(vacuum_ratio)),
        "separation_fit_fraction_median": float(np.nanmedian(fit_fraction)),
        "fsa_well_posed_fraction": float(np.mean(geometry["well_posed"])),
        "bank": str(output),
    }


def _relative_temporal_variation(array: np.ndarray) -> float:
    scale = np.nanmedian(np.abs(array), axis=0)
    scale = np.where(scale > 0, scale, np.nan)
    return float(np.nanmedian(np.abs(np.diff(array, axis=0)) / scale))


def _basis_fraction(arrays: list[np.ndarray]) -> list[float]:
    rows = np.concatenate(arrays, axis=0)
    fill = np.nanmedian(rows, axis=0)
    fill = np.nan_to_num(fill)
    rows = np.where(np.isfinite(rows), rows, fill)
    rows -= np.mean(rows, axis=0)
    singular = np.linalg.svd(rows, full_matrices=False, compute_uv=False)
    energy = singular**2
    return np.cumsum(energy / max(np.sum(energy), 1e-30)).tolist()


def write_summary(receipts: list[dict], bank_paths: list[Path], output_dir: Path):
    banks = [np.load(path) for path in bank_paths]
    fields = {
        "p_prime": [bank["p_prime_on_rho"] for bank in banks],
        "ff_prime": [bank["ff_prime_on_rho"] for bank in banks],
        "psi_on_rho": [bank["psi_on_rho_wb"] for bank in banks],
    }
    total_frames = sum(int(receipt["frames"]) for receipt in receipts)
    passed_frames = sum(int(receipt["vacuum_pass_frames"]) for receipt in receipts)
    summary = {
        "shots_banked": len(receipts),
        "frames_banked": total_frames,
        "vacuum_pass_frames": passed_frames,
        "vacuum_pass_fraction": passed_frames / max(total_frames, 1),
        "shots_with_majority_vacuum_pass": sum(
            float(receipt["vacuum_pass_fraction"]) > 0.5 for receipt in receipts
        ),
        "median_surface_separation_fit_fraction": float(
            np.nanmedian([r["separation_fit_fraction_median"] for r in receipts])
        ),
        "median_fsa_well_posed_fraction": float(
            np.nanmedian([r["fsa_well_posed_fraction"] for r in receipts])
        ),
        "relative_temporal_variation": {
            key: float(np.nanmedian([_relative_temporal_variation(a) for a in arrays]))
            for key, arrays in fields.items()
        },
        "cumulative_basis_fraction": {
            key: _basis_fraction(arrays) for key, arrays in fields.items()
        },
        "per_shot": receipts,
    }
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    ratios = np.concatenate([bank["vacuum_ratio"] for bank in banks])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].hist(ratios[np.isfinite(ratios)], bins=40, color="steelblue")
    axes[0].axvline(0.25, color="#a33a2b", linestyle="--", label="receipt limit")
    axes[0].set(xlabel="vacuum/plasma Delta-star p90 ratio", ylabel="frames")
    axes[0].legend(frameon=False)
    for key, values in summary["cumulative_basis_fraction"].items():
        axes[1].plot(np.arange(1, len(values) + 1), values, label=key)
    axes[1].set(
        xlabel="basis rank",
        ylabel="cumulative trajectory variance",
        xlim=(1, 12),
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "receipt_and_basis_summary.png", dpi=160)
    plt.close(fig)

    representative = max(receipts, key=lambda item: int(item["frames"]))
    bank = np.load(representative["bank"])
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
    for axis, key, label_text in zip(
        axes,
        ("p_prime_on_rho", "ff_prime_on_rho", "psi_on_rho_wb"),
        ("p-prime", "FF-prime", "total psi [Wb]"),
        strict=True,
    ):
        image = axis.imshow(
            bank[key].T,
            aspect="auto",
            origin="lower",
            extent=(
                bank["times_ms"][0],
                bank["times_ms"][-1],
                RHO_SAMPLES[0],
                RHO_SAMPLES[-1],
            ),
            cmap="viridis",
        )
        axis.set(ylabel="rho-hat")
        axis.text(
            1.01, 0.5, label_text, transform=axis.transAxes, rotation=90, va="center"
        )
        fig.colorbar(image, ax=axis, pad=0.08)
    axes[-1].set(xlabel="time [ms]")
    fig.suptitle(f"Representative trajectory: {representative['shot']}")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_trajectories.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("summary_dir", type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text())["selection"]
    receipts = []
    bank_paths = []
    for index, item in enumerate(selection, start=1):
        source = Path(item["path"])
        bank = args.bank_dir / f"{source.stem}_flux_trajectory.npz"
        receipt = extract_shot(source, bank)
        receipts.append(receipt)
        bank_paths.append(bank)
        pass_fraction = receipt["vacuum_pass_fraction"]
        print(
            f"[{index}/{len(selection)}] {source.stem}: {pass_fraction:.3f}",
            flush=True,
        )
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    write_summary(receipts, bank_paths, args.summary_dir)


if __name__ == "__main__":
    main()
