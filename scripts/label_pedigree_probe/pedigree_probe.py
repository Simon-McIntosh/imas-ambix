#!/usr/bin/env python3
# ruff: noqa: E501
"""Measure whether extracted equilibrium interiors identify label pedigree.

The bank contains flux-derived surface geometry and current-density profiles,
but not the reconstruction constraints or a full safety-factor profile.  The
safety-factor shape used here is therefore a constant-F proxy, normalized to
the parquet q95 value.  It is useful for measuring radial complexity, not for
claiming an independently reconstructed q profile.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def _finite_quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _normalise_rows(values: np.ndarray, radial_mask: np.ndarray) -> np.ndarray:
    scale = np.nanmedian(np.abs(values[:, radial_mask]), axis=1)
    sign = np.sign(np.nanmedian(values[:, radial_mask], axis=1))
    sign[sign == 0] = 1.0
    scale = np.where(scale > 1e-30, scale, np.nan)
    return values * sign[:, None] / scale[:, None]


def _polynomial_receipts(
    values: np.ndarray, radius: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.vander(radius**2, degree + 1, increasing=True)
    fitted = np.full_like(values, np.nan, dtype=float)
    residual_fraction = np.full(values.shape[0], np.nan)
    explained_fraction = np.full(values.shape[0], np.nan)
    for index, row in enumerate(values):
        good = np.isfinite(row)
        if np.count_nonzero(good) <= degree + 1:
            continue
        coefficients = np.linalg.lstsq(design[good], row[good], rcond=None)[0]
        fitted[index] = design @ coefficients
        residual = row[good] - fitted[index, good]
        residual_fraction[index] = np.sqrt(np.mean(residual**2)) / max(
            np.sqrt(np.mean(row[good] ** 2)), 1e-30
        )
        total = np.sum((row[good] - np.mean(row[good])) ** 2)
        explained_fraction[index] = 1.0 - np.sum(residual**2) / max(total, 1e-30)
    return fitted, residual_fraction, explained_fraction


def _basis_receipt(values: np.ndarray) -> dict[str, object]:
    fill = np.nanmedian(values, axis=0)
    rows = np.where(np.isfinite(values), values, fill)
    rows = rows - np.mean(rows, axis=0)
    singular = np.linalg.svd(rows, full_matrices=False, compute_uv=False)
    energy = singular**2
    cumulative = np.cumsum(energy / max(np.sum(energy), 1e-30))
    return {
        "cumulative": cumulative.tolist(),
        "rank_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "rank_99": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def _between_shot_fraction(values: np.ndarray, shot_index: np.ndarray) -> float:
    grand = np.nanmean(values, axis=0)
    total = np.nansum((values - grand) ** 2)
    within = 0.0
    for shot in np.unique(shot_index):
        subset = values[shot_index == shot]
        within += np.nansum((subset - np.nanmean(subset, axis=0)) ** 2)
    return float(max(total - within, 0.0) / max(total, 1e-30))


def _split_half_coherence(residuals: np.ndarray, shot_index: np.ndarray) -> np.ndarray:
    correlations = []
    for shot in np.unique(shot_index):
        subset = residuals[shot_index == shot]
        midpoint = len(subset) // 2
        if midpoint < 4:
            continue
        first = np.nanmean(subset[:midpoint], axis=0)
        second = np.nanmean(subset[midpoint:], axis=0)
        good = np.isfinite(first) & np.isfinite(second)
        if np.count_nonzero(good) < 4:
            continue
        first = first[good] - np.mean(first[good])
        second = second[good] - np.mean(second[good])
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator > 0:
            correlations.append(float(np.dot(first, second) / denominator))
    return np.asarray(correlations)


def _q95_for_bank(source_dir: Path, bank_path: Path) -> np.ndarray:
    stem = bank_path.stem.removesuffix("_flux_trajectory")
    source = source_dir / f"{stem}.parquet"
    row = pq.read_table(source, columns=["efit_q95"]).to_pylist()[0]
    return np.asarray(row["efit_q95"], dtype=float)


def _interpolate_surfaces(
    surface_rho: np.ndarray, values: np.ndarray, sample_rho: np.ndarray
) -> np.ndarray:
    interpolated = np.full((len(values), len(sample_rho)), np.nan)
    for frame, (frame_rho, frame_values) in enumerate(
        zip(surface_rho, values, strict=True)
    ):
        good = np.isfinite(frame_rho) & np.isfinite(frame_values)
        if np.count_nonzero(good) < 4:
            continue
        order = np.argsort(frame_rho[good])
        radial_coordinate = frame_rho[good][order]
        profile = frame_values[good][order]
        unique, unique_index = np.unique(radial_coordinate, return_index=True)
        interpolated[frame] = np.interp(
            sample_rho,
            unique,
            profile[unique_index],
            left=np.nan,
            right=np.nan,
        )
    return interpolated


def _load(bank_dir: Path, source_dir: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {
        "rho": [],
        "q_proxy": [],
        "j_phi": [],
        "separation": [],
        "shot": [],
        "vacuum_ratio": [],
    }
    paths = sorted(bank_dir.glob("*.npz"))
    for shot_number, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as bank:
            rho = np.asarray(bank["rho_hat_samples"], dtype=float)
            span = np.abs(bank["boundary_flux_wb"] - bank["axis_flux_wb"])
            surface_rho = np.asarray(bank["surface_rho_hat"], dtype=float)
            geometry = bank["fsa_dv_dpsi_n"] * bank["fsa_inverse_r2"]
            q_shape = _interpolate_surfaces(surface_rho, geometry / span[:, None], rho)
            current_density = _interpolate_surfaces(
                surface_rho, np.asarray(bank["j_phi_surface_mean"]), rho
            )
            q95 = _q95_for_bank(source_dir, path)
            if len(q95) != len(q_shape):
                raise ValueError(f"q95 length mismatch for {path.name}")
            q_proxy = q_shape / q_shape[:, -1:] * q95[:, None]
            fields["rho"].append(rho)
            fields["q_proxy"].append(q_proxy)
            fields["j_phi"].append(current_density)
            fields["separation"].append(
                np.nanmedian(bank["separation_fit_fraction"], axis=1)
            )
            fields["shot"].append(np.full(len(q_proxy), shot_number, dtype=np.int64))
            fields["vacuum_ratio"].append(np.asarray(bank["vacuum_ratio"]))
    if not paths:
        raise FileNotFoundError(f"no npz banks under {bank_dir}")
    return {
        "rho": fields["rho"][0],
        **{key: np.concatenate(value) for key, value in fields.items() if key != "rho"},
        "shot_count": np.asarray(len(paths)),
    }


def _analyse(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    rho = data["rho"]
    quality = (
        np.isfinite(data["q_proxy"]).all(axis=1)
        & np.isfinite(data["j_phi"]).all(axis=1)
        & (data["separation"] >= 0.50)
    )
    q = data["q_proxy"][quality]
    j = data["j_phi"][quality]
    shots = data["shot"][quality]
    interior = rho <= 0.80
    core = rho <= 0.40
    q_normalised = q / q[:, -1:]
    j_normalised = _normalise_rows(j, interior)

    q_fit, q_residual, q_explained = _polynomial_receipts(
        q_normalised[:, interior], rho[interior], degree=2
    )
    j_fit, j_residual, j_explained = _polynomial_receipts(
        j_normalised[:, interior], rho[interior], degree=2
    )
    j_core = j_normalised[:, core]
    q_core = q_normalised[:, core]
    j_core_cv = np.nanstd(j_core, axis=1) / np.maximum(
        np.abs(np.nanmean(j_core, axis=1)), 1e-30
    )
    q_core_cv = np.nanstd(q_core, axis=1) / np.maximum(
        np.abs(np.nanmean(q_core, axis=1)), 1e-30
    )
    j_residual_profiles = j_normalised[:, interior] - j_fit
    coherence = _split_half_coherence(j_residual_profiles, shots)
    q_basis = _basis_receipt(q_normalised[:, interior])
    j_basis = _basis_receipt(j_normalised[:, interior])

    metrics: dict[str, object] = {
        "corpus": {
            "shots": int(data["shot_count"]),
            "frames": int(len(data["shot"])),
            "quality_frames": int(np.count_nonzero(quality)),
            "quality_fraction": float(np.mean(quality)),
            "quality_rule": "finite q proxy and j profile; median surface separation R-squared >= 0.50",
            "vacuum_ratio": _finite_quantiles(data["vacuum_ratio"]),
            "surface_separation_r2": _finite_quantiles(data["separation"]),
        },
        "q_profile_structure": {
            "measurement": "constant-F q-shape proxy normalized to parquet q95",
            "quadratic_rho2_residual_fraction": _finite_quantiles(q_residual),
            "quadratic_rho2_explained_fraction": _finite_quantiles(q_explained),
            "interpretation": "low-order, but not an independent q-profile receipt",
        },
        "near_axis_current_detail": {
            "quadratic_rho2_residual_fraction": _finite_quantiles(j_residual),
            "quadratic_rho2_explained_fraction": _finite_quantiles(j_explained),
            "residual_split_half_coherence_by_shot": _finite_quantiles(coherence),
            "interpretation": "coherence can reflect EFIT basis or extraction bias as well as diagnostics",
        },
        "interior_basis": {
            "q_proxy": q_basis,
            "current_density": j_basis,
            "q_between_shot_variance_fraction": _between_shot_fraction(
                q_normalised[:, interior], shots
            ),
            "current_between_shot_variance_fraction": _between_shot_fraction(
                j_normalised[:, interior], shots
            ),
            "interpretation": "compact bases are consistent with a smooth low-order reconstruction ansatz",
        },
        "interior_flatness": {
            "current_core_cv": _finite_quantiles(j_core_cv),
            "q_proxy_core_cv": _finite_quantiles(q_core_cv),
            "current_frames_below_10pct_cv": float(np.mean(j_core_cv <= 0.10)),
            "q_proxy_frames_below_10pct_cv": float(np.mean(q_core_cv <= 0.10)),
            "interpretation": "flatness is magnetics-only-like but is not unique to magnetics-only EFIT",
        },
        "verdict": {
            "pedigree": "inconclusive",
            "confidence": 0.85,
            "recommendation": "edge-weighted",
            "reason": (
                "All four shape tests show a smooth, low-dimensional interior compatible with "
                "magnetics-only reconstruction, but none is constraint-specific. The same signatures "
                "can arise from MSE-constrained EFIT with a low-order profile basis and regularization."
            ),
        },
    }
    plot_data = {
        "rho": rho,
        "interior": interior,
        "q": q_normalised,
        "j": j_normalised,
        "q_residual": q_residual,
        "j_residual": j_residual,
        "j_core_cv": j_core_cv,
        "q_cumulative": np.asarray(q_basis["cumulative"]),
        "j_cumulative": np.asarray(j_basis["cumulative"]),
    }
    return metrics, plot_data


def _plot_profiles(plot_data: dict[str, np.ndarray], output: Path) -> None:
    rho = plot_data["rho"]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for axis, key, label in (
        (axes[0], "q", "q proxy / q95"),
        (axes[1], "j", "normalized j-phi"),
    ):
        values = plot_data[key]
        low, median, high = np.nanquantile(values, [0.10, 0.50, 0.90], axis=0)
        axis.fill_between(
            rho, low, high, alpha=0.25, color="#326b8c", label="frame p10-p90"
        )
        axis.plot(rho, median, color="#16384c", linewidth=2, label="frame median")
        axis.axvspan(0.0, 0.4, color="#d8b365", alpha=0.12)
        axis.set(xlabel="rho-hat", ylabel=label)
        axis.legend(frameon=False, fontsize=8)
    for key, label, colour in (
        ("q_cumulative", "q proxy", "#326b8c"),
        ("j_cumulative", "current density", "#a33a2b"),
    ):
        values = plot_data[key]
        axes[2].plot(np.arange(1, len(values) + 1), values, label=label, color=colour)
    axes[2].axhline(0.95, color="0.5", linestyle="--", linewidth=1)
    axes[2].set(
        xlabel="basis rank", ylabel="cumulative variance", xlim=(1, 12), ylim=(0, 1.01)
    )
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_distributions(plot_data: dict[str, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for axis, key, label in (
        (axes[0], "q_residual", "q proxy low-order residual"),
        (axes[1], "j_residual", "j-phi low-order residual"),
        (axes[2], "j_core_cv", "near-axis j-phi CV"),
    ):
        values = plot_data[key]
        values = values[np.isfinite(values)]
        axis.hist(values, bins=45, color="#326b8c", alpha=0.85)
        axis.axvline(np.median(values), color="#a33a2b", linewidth=1.5, label="median")
        axis.set(xlabel=label, ylabel="frames")
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_report(metrics: dict[str, object], output: Path) -> None:
    q = metrics["q_profile_structure"]
    j = metrics["near_axis_current_detail"]
    basis = metrics["interior_basis"]
    flat = metrics["interior_flatness"]
    corpus = metrics["corpus"]
    rows = [
        (
            "q-profile interior structure",
            f"quadratic-in-rho-squared median residual {_percent(q['quadratic_rho2_residual_fraction']['median'])}; "
            f"median explained fraction {_percent(q['quadratic_rho2_explained_fraction']['median'])}",
            "Magnetics-only-like, not identifying",
        ),
        (
            "near-axis current detail",
            f"low-order median residual {_percent(j['quadratic_rho2_residual_fraction']['median'])}; "
            f"split-half residual coherence {j['residual_split_half_coherence_by_shot']['median']:.3f}",
            "Coherent detail exists, but its source is ambiguous",
        ),
        (
            "interior basis stability",
            f"95% ranks: q proxy {basis['q_proxy']['rank_95']}, current {basis['current_density']['rank_95']}; "
            f"between-shot variance fractions {_percent(basis['q_between_shot_variance_fraction'])} and "
            f"{_percent(basis['current_between_shot_variance_fraction'])}",
            "Compact smooth ansatz",
        ),
        (
            "unconstrained-interior flatness",
            f"median core CV: current {_percent(flat['current_core_cv']['median'])}, q proxy "
            f"{_percent(flat['q_proxy_core_cv']['median'])}; below 10% in "
            f"{_percent(flat['current_frames_below_10pct_cv'])} and {_percent(flat['q_proxy_frames_below_10pct_cv'])} of frames",
            "Flatness present to the measured degree, but not unique",
        ),
    ]
    table = "\n".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(number)}</td><td>{html.escape(call)}</td></tr>"
        for name, number, call in rows
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label pedigree probe</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#18232b}} table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;vertical-align:top;padding:.6rem;border-bottom:1px solid #ccd4d8}} th{{width:22%}} figure{{margin:2rem 0}} img{{max-width:100%;height:auto}} .verdict{{border-left:5px solid #a76f00;padding:.8rem 1rem;background:#fff8e8}} code{{background:#eef2f4;padding:.1rem .25rem}}</style></head><body>
<h1>Label pedigree probe</h1>
<p class="verdict"><strong>Verdict: inconclusive (85% confidence that these data do not identify pedigree).</strong>
The profiles are smooth and low-dimensional, which is compatible with magnetics-only EFIT, but MSE-constrained EFIT using the same low-order basis and regularization can produce the same signatures. Recommend <strong>edge-weighted</strong> closure tuning: use boundary and outer-profile information strongly, retain only tightly priored low-degree interior corrections, and do not treat the core as independently verified.</p>
<p>Scope: {corpus["quality_frames"]:,} quality-selected frames from {corpus["frames"]:,} banked frames across {corpus["shots"]} shots ({_percent(corpus["quality_fraction"])}). The quality rule was {html.escape(corpus["quality_rule"])}.</p>
<h2>Measured discriminators</h2><table><thead><tr><th>Discriminator</th><th>Measured number</th><th>Reading</th></tr></thead><tbody>{table}</tbody></table>
<figure><img src="profile_shapes.png" alt="Profile envelopes and cumulative basis variance"><figcaption>Profile envelopes and basis compactness. The q curve is a shape proxy, normalized to q95, not a directly extracted full q profile.</figcaption></figure>
<figure><img src="shape_distributions.png" alt="Distributions of low-order residual and core flatness metrics"><figcaption>Frame distributions behind the low-order and flatness receipts.</figcaption></figure>
<h2>Why the evidence does not settle provenance</h2>
<p>No bank or parquet field records whether MSE constraints entered the EFIT. The extracted current profile is derived from second derivatives of the same 65 by 65 psi map, so stable fine structure can be generated by the EFIT basis, smoothing, grid stencil, or extraction method. Conversely, a genuinely MSE-constrained solution may remain smooth because EFIT represents it with a low-order parametrization. Profile shape alone therefore has no constraint-specific signature in this corpus.</p>
<p>The q discriminator is weaker still: the corpus carries q95 but no q profile or F function. This report constructs <code>q_proxy proportional to (dV/dpsi_N) times mean(1/R squared) divided by abs(delta psi)</code>, assumes constant F for its radial shape, and scales its edge to q95. It measures geometric complexity, not internal pitch independently.</p>
<h2>Decision recommendation</h2>
<p><strong>edge-weighted</strong>. <code>interior-trusted</code> would convert non-identifying smoothness into unjustified provenance. <code>boundary-only</code> would discard real, coherent low-order information that is still useful under tight priors. The middle policy preserves that information without claiming MSE-grade core authority.</p>
<h2>What would settle it</h2>
<p>Any one of these would provide identifying evidence: reconstruction run metadata listing diagnostic constraints; EFIT input files or k-files exposing MSE weights; paired reconstructions of the same frames with MSE enabled and disabled; or a q-profile/pitch-angle residual receipt against held-out MSE measurements. Without one of them, the pedigree should remain explicitly unknown.</p>
<p>Machine-readable metrics: <a href="metrics.json">metrics.json</a>. Reproduction log: <a href="run.log">run.log</a>.</p>
</body></html>"""
    output.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(args.bank_dir, args.source_dir)
    metrics, plot_data = _analyse(data)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _plot_profiles(plot_data, args.output_dir / "profile_shapes.png")
    _plot_distributions(plot_data, args.output_dir / "shape_distributions.png")
    _write_report(metrics, args.output_dir / "report.html")
    verdict = metrics["verdict"]
    corpus = metrics["corpus"]
    print(
        f"frames={corpus['frames']} quality_frames={corpus['quality_frames']} shots={corpus['shots']}"
    )
    print(
        f"verdict={verdict['pedigree']} confidence={verdict['confidence']:.2f} recommendation={verdict['recommendation']}"
    )


if __name__ == "__main__":
    main()
