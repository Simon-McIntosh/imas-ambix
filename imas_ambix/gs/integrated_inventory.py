"""Integrated-input feasibility scoping for the GS-grounded engine (T9).

Why this module exists
----------------------
T6 showed that the magnetics-only latent cannot ground the *internal* current
profile (near-vacuum c_plasma ratio 0.98) — this is the classic
equilibrium-reconstruction under-determination: external magnetics fix the
boundary, total current and the low moments, but NOT p′/FF′ internally.  The
fix (user direction, comment ``c-s8-integrated-direction``) is to *widen the
engine's input set* with INTERNAL diagnostics (MSE → current/q; Thomson →
pressure) plus broadly-available emission (bolometer, soft-xray) and the
visible camera, so the integrated Bayesian filter + GS soft prior can ground
the latent — a learned, temporal Minerva/IDA-style equilibrium+kinetic
inference.

Co-availability is the *binding* constraint (as it was in S7.1): widening the
input set shrinks the co-available corpus, and a too-small calibration set
makes split-conformal coverage high-variance.  This module is the FEASIBILITY
scoping that quantifies that trade-off so the orchestrator can LOCK the
integrated input set.  It does NOT retrain anything and does NOT decide the
input set — it recommends.

What it computes
----------------
1. A per-diagnostic inventory: candidate group(s), native sampling rate, the
   GS internal-profile quantity each constrains, and level-1 shot coverage.
2. A co-availability matrix: for each candidate input combo, the co-available
   shot count vs the Dα target (``xim``) — both whole-corpus and restricted to
   the GS-geometry-campaign envelope — then split through the *locked v0 OOD
   box* (joint_p84, by-current-density) into train / calibration / test sizes,
   with the conformal-viability flag (cal < ~200).
3. A multi-rate time-alignment plan per diagnostic vs the 1000 Hz engine grid.
4. A camera-feature plan (rbb fed as a compact feature, not raw pixels).
5. A recommendation: 2-3 concrete integrated-input-set options with corpus N's.

Reuse (import-only — this module owns nothing it imports):
    * :class:`imas_ambix.statespace.inventory.InventoryResult` — the S7.1
      per-shot family co-availability (loaded from the persisted artifact).
    * :class:`imas_ambix.statespace.splits.RegimeBox` / ``build_splits`` /
      ``CorpusNReport`` — the locked train/cal/ood split machinery.
    * The persisted regime scalars + the locked v0 OOD box (so the expensive
      per-shot amc/ane read is NOT repeated, and the by-current-density axis
      stays identical across combos and with the v0 baseline).

Locked decisions honoured
-------------------------
* Raw signals only.  ``efm`` is GEOMETRY-only (T1); ``esm``/``xdc`` excluded;
  ``amm`` (Omaha) currents excluded.  Solver outputs are NEVER inputs/labels.
* Held-out target stays Dα (``xim``).
* regime-split axis = by-current-density (the locked v0 joint_p84 box).
* The integrated input set SUPERSEDES the Stage-1 ``mag+ane`` lock for the
  *grounded* engine.

Usage
-----
    from imas_ambix.gs.integrated_inventory import build_feasibility
    report = build_feasibility()
    report.save(Path("imas_ambix/gs/artifacts/gs_integrated_feasibility.json"))

    # or via the CLI entrypoint:
    #   uv run python -m imas_ambix.gs.integrated_inventory
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_ambix.statespace.inventory import InventoryResult
    from imas_ambix.statespace.splits import RegimeBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical artifact locations (persisted by earlier stages — import-only)
# ---------------------------------------------------------------------------

MANIFEST_DIR = Path("/work/projects/imas_gpu/mast/manifests")
FAMILY_INVENTORY = MANIFEST_DIR / "statespace_family_inventory.json"
REGIME_SCALARS = MANIFEST_DIR / "statespace_regime_scalars.json"
SPLITS_DALPHA_V0 = MANIFEST_DIR / "statespace_splits_dalpha_v0.json"

ARTIFACT_DEFAULT = (
    Path(__file__).parent / "artifacts" / "gs_integrated_feasibility.json"
)

# The engine model grid (Hz).  Locked at 1000 Hz to preserve ELM/Dα structure
# (see statespace/baseline.py docstring) — the alignment plan is relative to
# this grid.
ENGINE_GRID_HZ = 1000.0

# Conformal calibration-set viability floor.  Below this the split-conformal
# coverage estimate is high-variance (matches statespace/splits.py heuristic).
CAL_VIABILITY_FLOOR = 200

# The GS-geometry-campaign shot-range envelope (union of the 3 signatures'
# sampled spans from gs_geometry_summary.json: [11764,12342] ∪ [12417,13349]
# ∪ [12533,30473]).  Used as a CHEAP proxy for "matches a GS-geometry
# campaign" — exact per-shot setup-signature matching (opening efm) is
# deferred to T10, where the corpus is already small.  efm-coverage (14633)
# >> the binding MSE coverage (4882), so this restriction is non-binding (see
# the artifact's gs_envelope_note).
GS_CAMPAIGN_SHOT_MIN = 11764
GS_CAMPAIGN_SHOT_MAX = 30473

# ---------------------------------------------------------------------------
# Per-diagnostic inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticSpec:
    """One candidate input diagnostic and what it constrains for the GS state.

    Attributes
    ----------
    name:
        Human label (e.g. ``"MSE"``).
    groups:
        Level-1 Zarr group name(s).  A tuple > 1 means "any-of" (e.g. the
        three Thomson systems) unless ``require_all`` is set.
    require_all:
        When True, the diagnostic counts as present only if EVERY group is
        present (rare — used only where physics needs both core+edge).
    native_hz:
        Representative native sampling rate (Hz), measured from the corpus.
    constrains:
        The GS internal-profile / state quantity this diagnostic constrains.
    align:
        The proposed alignment strategy vs the 1000 Hz engine grid.
    note:
        Provenance / caveat.
    """

    name: str
    groups: tuple[str, ...]
    native_hz: float
    constrains: str
    align: str
    require_all: bool = False
    note: str = ""


# The candidate diagnostics.  Rates are MEASURED from representative level-1
# shots (see the module commit message / the artifact's `measured_on` field).
# `ama`/`amb`/`amc` (magnetics) and `ane` (interferometer) are the v0 baseline
# inputs; `xim` is the Dα held-out TARGET (never an input).
DIAGNOSTICS: dict[str, DiagnosticSpec] = {
    "magnetics": DiagnosticSpec(
        name="magnetics (ama+amb+amc)",
        groups=("ama", "amb", "amc"),
        require_all=True,
        native_hz=5000.0,  # amc 0.2 ms grid
        constrains="boundary + total Iₚ + low flux moments (NOT internal p′/FF′)",
        align="interp_to_grid",
        note="v0 baseline input. amm (Omaha MHz) excluded per lock.",
    ),
    "interferometer": DiagnosticSpec(
        name="interferometer (ane)",
        groups=("ane",),
        native_hz=66000.0,  # ~15 µs grid
        constrains="line-integrated nₑ (weak pressure-scale prior)",
        align="decimate_then_interp",
        note="v0 baseline input.",
    ),
    "mse": DiagnosticSpec(
        name="MSE (ams)",
        groups=("ams",),
        native_hz=2000.0,  # 0.5 ms grid
        constrains=(
            "magnetic pitch angle → INTERNAL current / q-profile "
            "(THE constraint magnetics lack — the p′/FF′ current term)"
        ),
        align="interp_to_grid",
        note=(
            "~15-35 polarimetry channels (acoeff trailing dim). The single "
            "most physically valuable internal-current constraint; also the "
            "BINDING co-availability constraint (≈4882 shots)."
        ),
    ),
    "thomson_any": DiagnosticSpec(
        name="Thomson any (ayc|atm|aye)",
        groups=("ayc", "atm", "aye"),
        require_all=False,
        native_hz=240.0,  # ~4.2 ms grid — SPARSE/asynchronous
        constrains="Tₑ(r), nₑ(r) → kinetic PRESSURE p(ψ) (the p′ term in GS)",
        align="hold_at_native_cadence",
        note=(
            "any-of the three TS systems (combined ayc / core atm / edge aye). "
            "SPARSE (~5 ms) vs the 1 ms grid → the GS prior weights p(ψ) only "
            "at measurement times, not every grid step."
        ),
    ),
    "thomson_combined": DiagnosticSpec(
        name="Thomson combined (ayc)",
        groups=("ayc",),
        native_hz=240.0,
        constrains="Tₑ(r), nₑ(r) → kinetic PRESSURE p(ψ) (full-profile)",
        align="hold_at_native_cadence",
        note="combined-system TS — the cleanest single full-profile pressure source.",
    ),
    "thomson_core": DiagnosticSpec(
        name="Thomson core (atm)",
        groups=("atm",),
        native_hz=200.0,
        constrains="core Tₑ(r), nₑ(r) → on-axis pressure",
        align="hold_at_native_cadence",
        note="core-only TS — highest single-TS coverage (6122) but core-only.",
    ),
    "bolometer": DiagnosticSpec(
        name="bolometer (abm)",
        groups=("abm",),
        native_hz=2500.0,  # 0.4 ms grid
        constrains="radiated power (edge/impurity/boundary radiation)",
        align="interp_to_grid",
        note="~32 channels. Broadly available (12166).",
    ),
    "soft_xray": DiagnosticSpec(
        name="soft-xray (xsx)",
        groups=("xsx",),
        native_hz=500000.0,  # 2 µs grid — very fast
        constrains="emissivity (core profile shape / tomography)",
        align="decimate_aggregate_then_interp",
        note=(
            "VERY fast (500 kHz, 300k samples/chan) → MUST decimate/aggregate "
            "to the 1 ms grid before ingest. Broadly available (13944)."
        ),
    ),
    "cxrs": DiagnosticSpec(
        name="CXRS (act)",
        groups=("act",),
        native_hz=200.0,  # 5 ms grid — SPARSE
        constrains="Tᵢ(r), toroidal rotation (ion pressure / rotation prior)",
        align="hold_at_native_cadence",
        note="SPARSE (~5 ms). Secondary kinetic constraint (7991).",
    ),
    "camera_visible": DiagnosticSpec(
        name="visible camera (rbb)",
        groups=("rbb",),
        native_hz=1000.0,  # ~300 frames over ~0.3 s
        constrains="boundary / emission pattern (shape prior, via a FEATURE)",
        align="feature_then_interp",
        note=(
            "FRAMES (T,1024,1024) uint16 — fed as a COMPACT feature, NEVER raw "
            "pixels into the RKN latent (see camera_feature_plan)."
        ),
    ),
    "ir_camera": DiagnosticSpec(
        name="IR cameras (rir/rit)",
        groups=("rir", "rit"),
        require_all=False,
        native_hz=0.0,
        constrains="(divertor/target heat load — EXCLUDED)",
        align="excluded",
        note=(
            "ESSENTIALLY ABSENT in level-1 (rir≈25, rit≈14 shots). Requiring "
            "IR as an input collapses the corpus → EXCLUDED."
        ),
    ),
}

# Convenience group sets
MAG_GROUPS: tuple[str, ...] = ("ama", "amb", "amc")
TARGET_GROUP = "xim"  # Dα held-out target (and its analysis siblings ada/aim)
TARGET_NOTE = "dalpha_primary (xim)"


# ---------------------------------------------------------------------------
# Input-combo definitions (the columns of the co-availability matrix)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputCombo:
    """One candidate integrated input set.

    ``require_groups`` must ALL be present; ``any_of`` (if non-empty) requires
    at least one of its members present (used for the Thomson-any semantics).
    """

    key: str
    label: str
    require_groups: tuple[str, ...]
    any_of: tuple[str, ...] = ()
    rationale: str = ""


def default_combos() -> list[InputCombo]:
    """Return the candidate input combos (incremental, internal-profile first)."""
    return [
        InputCombo(
            key="v0_baseline",
            label="mag+ane (v0 baseline)",
            require_groups=(*MAG_GROUPS, "ane"),
            rationale=(
                "The Stage-1 input-modality-v0 lock the grounded engine supersedes."
            ),
        ),
        InputCombo(
            key="core_mse",
            label="mag+ane+MSE",
            require_groups=(*MAG_GROUPS, "ane", "ams"),
            rationale="Add the internal-current/q constraint (MSE) magnetics lack.",
        ),
        InputCombo(
            key="core_internal",
            label="mag+ane+MSE+TS-any",
            require_groups=(*MAG_GROUPS, "ane", "ams"),
            any_of=("ayc", "atm", "aye"),
            rationale=(
                "Internal-profile CORE: current/q (MSE) + pressure p(ψ) (Thomson). "
                "TS-any maximises N at near-zero cost beyond MSE."
            ),
        ),
        InputCombo(
            key="core_internal_combined",
            label="mag+ane+MSE+Thomson(ayc)",
            require_groups=(*MAG_GROUPS, "ane", "ams", "ayc"),
            rationale=(
                "Internal core with the cleanest single full-profile pressure source."
            ),
        ),
        InputCombo(
            key="core_plus_emission",
            label="mag+ane+MSE+TS-any+bolo+SXR",
            require_groups=(*MAG_GROUPS, "ane", "ams", "abm", "xsx"),
            any_of=("ayc", "atm", "aye"),
            rationale=(
                "Internal core + broadly-available emission (radiation + SXR shape)."
            ),
        ),
        InputCombo(
            key="full",
            label="mag+ane+MSE+TS-any+bolo+SXR+camera",
            require_groups=(*MAG_GROUPS, "ane", "ams", "abm", "xsx", "rbb"),
            any_of=("ayc", "atm", "aye"),
            rationale="Maximal integrated input set incl. the visible-camera feature.",
        ),
    ]


# ---------------------------------------------------------------------------
# Co-availability counting
# ---------------------------------------------------------------------------


def _combo_shots(
    inv: InventoryResult,
    combo: InputCombo,
    *,
    restrict_gs_envelope: bool = False,
) -> list[int]:
    """Shots co-available for *combo* AND the Dα target.

    A shot qualifies when every ``require_groups`` is present, at least one of
    ``any_of`` (if non-empty) is present, and the target group is present.
    """
    req = frozenset((*combo.require_groups, TARGET_GROUP))
    any_of = frozenset(combo.any_of)
    out: list[int] = []
    for sid, grps in inv.shot_groups.items():
        gset = frozenset(grps)
        if not req.issubset(gset):
            continue
        if any_of and not (any_of & gset):
            continue
        if restrict_gs_envelope:
            if not (GS_CAMPAIGN_SHOT_MIN <= sid <= GS_CAMPAIGN_SHOT_MAX):
                continue
            if "efm" not in gset:  # GS geometry source must exist for the shot
                continue
        out.append(sid)
    return sorted(out)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class IntegratedFeasibilityReport:
    """The compact T9 feasibility artifact."""

    diagnostic_inventory: list[dict] = field(default_factory=list)
    coavailability_matrix: list[dict] = field(default_factory=list)
    alignment_plan: list[dict] = field(default_factory=list)
    camera_feature_plan: dict = field(default_factory=dict)
    ir_absence: dict = field(default_factory=dict)
    recommendation: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": "gs_integrated_feasibility_v0",
            "meta": self.meta,
            "diagnostic_inventory": self.diagnostic_inventory,
            "coavailability_matrix": self.coavailability_matrix,
            "alignment_plan": self.alignment_plan,
            "camera_feature_plan": self.camera_feature_plan,
            "ir_absence": self.ir_absence,
            "recommendation": self.recommendation,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Integrated feasibility artifact saved to %s", path)

    def matrix_table(self) -> str:
        """Compact ASCII table of the co-availability matrix."""
        hdr = (
            f"{'combo':<34} {'N_total':>8} {'N_gs':>7} "
            f"{'N_train':>8} {'N_cal':>7} {'N_ood':>6} {'cal_ok':>7}"
        )
        lines = [hdr, "-" * len(hdr)]
        for r in self.coavailability_matrix:
            lines.append(
                f"{r['label']:<34} {r['n_total_coavailable']:>8} "
                f"{r['n_gs_envelope']:>7} {r['n_train']:>8} "
                f"{r['n_calibration']:>7} {r['n_test_ood_regime']:>6} "
                f"{'yes' if r['cal_adequate'] else 'NO':>7}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loaders (import-only reuse of persisted artifacts)
# ---------------------------------------------------------------------------


def _load_inventory(path: Path = FAMILY_INVENTORY) -> InventoryResult:
    from imas_ambix.statespace.inventory import InventoryResult  # noqa: PLC0415

    return InventoryResult.load(path)


def _load_regime_scalars(path: Path = REGIME_SCALARS) -> dict[int, dict[str, float]]:
    """Load the persisted per-shot {ip_mean, ne_mean} (the expensive S7.1 read)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if "regime_scalars" in d:  # tolerate a wrapped form
        d = d["regime_scalars"]
    return {int(k): v for k, v in d.items()}


def _load_locked_ood_box(path: Path = SPLITS_DALPHA_V0) -> RegimeBox:
    """Load the LOCKED v0 OOD box (joint_p84, by-current-density)."""
    from imas_ambix.statespace.splits import RegimeBox  # noqa: PLC0415

    d = json.loads(path.read_text(encoding="utf-8"))
    b = d["ood_box"]
    return RegimeBox(
        ip_min=b["ip_min_kA"],
        ip_max=b["ip_max_kA"],
        ne_min=b["ne_min_1e19"],
        ne_max=b["ne_max_1e19"],
        description=b.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_feasibility(
    *,
    inventory: InventoryResult | None = None,
    regime_scalars: dict[int, dict[str, float]] | None = None,
    ood_box: RegimeBox | None = None,
    combos: list[InputCombo] | None = None,
    cal_fraction: float = 0.12,
    seed: int = 42,
) -> IntegratedFeasibilityReport:
    """Build the integrated-input feasibility report.

    All heavy inputs default to the persisted artifacts so the call performs
    NO per-shot Zarr reads (the regime scalars — the expensive amc/ane read —
    are reused from S7.1).  The OOD box is the LOCKED v0 joint_p84 box, applied
    identically to every combo so the columns stay comparable with each other
    and with the v0 baseline.

    Parameters left as ``None`` are loaded from the canonical manifest paths;
    tests inject small synthetic objects instead.
    """
    from imas_ambix.statespace.splits import build_splits  # noqa: PLC0415

    inv = inventory if inventory is not None else _load_inventory()
    scalars = regime_scalars if regime_scalars is not None else _load_regime_scalars()
    box = ood_box if ood_box is not None else _load_locked_ood_box()
    combo_list = combos if combos is not None else default_combos()

    report = IntegratedFeasibilityReport()

    # --- per-diagnostic inventory ------------------------------------------
    coverage = inv.group_coverage()
    for spec in DIAGNOSTICS.values():
        if spec.require_all:
            present = inv.shots_with_all_groups(*spec.groups)
            n_cov = len(present)
            cov_mode = "all-of"
        else:
            # any-of coverage
            gset = frozenset(spec.groups)
            n_cov = sum(
                1 for grps in inv.shot_groups.values() if gset & frozenset(grps)
            )
            cov_mode = "any-of" if len(spec.groups) > 1 else "single"
        report.diagnostic_inventory.append(
            {
                "name": spec.name,
                "groups": list(spec.groups),
                "coverage_mode": cov_mode,
                "per_group_coverage": {g: coverage.get(g, 0) for g in spec.groups},
                "n_shots": n_cov,
                "native_hz": spec.native_hz,
                "constrains": spec.constrains,
                "alignment": spec.align,
                "note": spec.note,
            }
        )
        if spec.align != "excluded":
            report.alignment_plan.append(
                {
                    "diagnostic": spec.name,
                    "native_hz": spec.native_hz,
                    "engine_grid_hz": ENGINE_GRID_HZ,
                    "rate_ratio_vs_grid": round(spec.native_hz / ENGINE_GRID_HZ, 3)
                    if spec.native_hz
                    else None,
                    "strategy": spec.align,
                    "rationale": _alignment_rationale(spec),
                }
            )

    # --- co-availability matrix --------------------------------------------
    for combo in combo_list:
        shots_total = _combo_shots(inv, combo, restrict_gs_envelope=False)
        shots_gs = _combo_shots(inv, combo, restrict_gs_envelope=True)
        # Build splits on the GS-envelope-restricted set (the corpus the
        # grounded engine will actually train on), with the LOCKED box.
        splits = build_splits(
            co_available_shots=shots_gs,
            regime_scalars=scalars,
            held_out_family="dalpha",
            input_groups=list(combo.require_groups),
            ood_box=box,
            cal_fraction=cal_fraction,
            seed=seed,
        )
        report.coavailability_matrix.append(
            {
                "key": combo.key,
                "label": combo.label,
                "target": TARGET_NOTE,
                "require_groups": list(combo.require_groups),
                "any_of": list(combo.any_of),
                "n_total_coavailable": len(shots_total),
                "n_gs_envelope": len(shots_gs),
                "n_train": splits.n_train,
                "n_calibration": splits.n_cal,
                "n_test_ood_regime": splits.n_ood,
                "cal_adequate": splits.n_cal >= CAL_VIABILITY_FLOOR,
                "rationale": combo.rationale,
            }
        )

    # --- camera-feature plan ------------------------------------------------
    report.camera_feature_plan = _camera_feature_plan()

    # --- IR absence ---------------------------------------------------------
    rir = coverage.get("rir", 0)
    rit = coverage.get("rit", 0)
    report.ir_absence = {
        "groups": {"rir": rir, "rit": rit},
        "n_shots_any_ir": len(inv.shots_with_all_groups("rir"))
        + len(inv.shots_with_all_groups("rit")),
        "decision": "EXCLUDED",
        "reason": (
            f"IR cameras are essentially absent in level-1 (rir={rir}, rit={rit} "
            "shots out of ~17k). Requiring IR as an input collapses the corpus to "
            "single digits — excluded as a required input."
        ),
    }

    # --- recommendation -----------------------------------------------------
    report.recommendation = _recommendation(report.coavailability_matrix)

    # --- meta ---------------------------------------------------------------
    report.meta = {
        "task": "T9 integrated-input feasibility scoping",
        "n_shots_inventory": inv.n_shots,
        "n_shots_with_regime_scalars": sum(
            1 for v in scalars.values() if "ip_mean" in v and "ne_mean" in v
        ),
        "target": TARGET_NOTE,
        "regime_split_axis": "by-current-density (LOCKED v0 joint_p84 OOD box)",
        "ood_box": box.to_dict(),
        "engine_grid_hz": ENGINE_GRID_HZ,
        "cal_viability_floor": CAL_VIABILITY_FLOOR,
        "gs_envelope_note": (
            f"GS-geometry restriction uses the cheap proxy efm-present AND shot "
            f"in [{GS_CAMPAIGN_SHOT_MIN},{GS_CAMPAIGN_SHOT_MAX}] (the union of the "
            "3 campaign sampled-spans). It is NON-BINDING: all MSE/Thomson shots "
            "fall inside the envelope and have efm, so n_gs_envelope ≈ "
            "n_total_coavailable. Exact per-shot setup-signature matching is "
            "deferred to T10 (cheap there — the corpus is already small)."
        ),
        "locks_honoured": [
            "raw signals only (no efm/esm/xdc/amm outputs as inputs or labels)",
            "efm GEOMETRY-only (T1)",
            "target = Dα (xim)",
            "regime-split = by-current-density",
            "integrated input set SUPERSEDES mag+ane for the grounded engine",
        ],
        "open_decisions_surfaced": ["uq-level-v0", "extrapolation-coordinates"],
        "sources": {
            "family_inventory": str(FAMILY_INVENTORY),
            "regime_scalars": str(REGIME_SCALARS),
            "splits_dalpha_v0_locked_box": str(SPLITS_DALPHA_V0),
        },
    }

    return report


def _alignment_rationale(spec: DiagnosticSpec) -> str:
    ratio = spec.native_hz / ENGINE_GRID_HZ if spec.native_hz else 0.0
    if spec.align == "hold_at_native_cadence":
        return (
            f"native {spec.native_hz:.0f} Hz is ~{1 / ratio:.1f}× SLOWER than the "
            f"{ENGINE_GRID_HZ:.0f} Hz grid → the model sees a held/interpolated "
            "value between samples; the GS prior should weight this constraint at "
            "measurement times only (sparse-likelihood), not every grid step."
        )
    if spec.align == "decimate_aggregate_then_interp":
        return (
            f"native {spec.native_hz:.0f} Hz is ~{ratio:.0f}× FASTER than the grid → "
            "aggregate (mean/RMS within each 1 ms bin) then place on the grid; "
            "feeding raw is wasteful and aliases fast MHD."
        )
    if spec.align == "decimate_then_interp":
        return (
            f"native {spec.native_hz:.0f} Hz >> grid -> decimate to grid (anti-alias)."
        )
    if spec.align == "feature_then_interp":
        return (
            "frames → extract a compact per-frame feature (see camera_feature_plan), "
            "then interp the feature vector onto the grid."
        )
    if spec.align == "interp_to_grid":
        return (
            f"native {spec.native_hz:.0f} Hz ≥ grid → linear interp onto the grid "
            "(resample_to_grid)."
        )
    return ""


def _camera_feature_plan() -> dict:
    return {
        "principle": (
            "Feed rbb as a COMPACT per-frame feature vector aligned to the engine "
            "grid — NEVER raw (T,1024,1024) pixels into the RKN latent (that would "
            "dwarf the latent dim and destabilise the filter)."
        ),
        "options": [
            {
                "name": "magvit2_token_pool",
                "detail": (
                    "Reuse the existing Open-MAGVIT2 frame tokenizer "
                    "(imas_ambix.tokenizer.frames.OpenMagvit2Tokenizer / the "
                    "in-process stream_encode.py path): 256→16×16 LFQ tokens per "
                    "frame; pool/embed the 16×16 token grid to a small (~16-64 dim) "
                    "per-frame vector, then interp onto the 1 ms grid."
                ),
                "pros": "reuses a validated encoder + the corpus encode pipeline.",
                "cons": "token grid → vector pooling needs a learned/avg projection.",
            },
            {
                "name": "camera_boundary_edge",
                "detail": (
                    "Reuse statespace.camera_boundary: Sobel emission-edge → a "
                    "~36-dim boundary-radius(angle) vector per frame (pixel-space "
                    "shape proxy). Cheap, no GPU, interpretable."
                ),
                "pros": "no model, CPU-only, directly a shape prior.",
                "cons": "pixel-space (no camera calibration → not a physical LCFS).",
            },
            {
                "name": "small_cnn_embedding",
                "detail": (
                    "A small CNN (e.g. 4-conv) trained jointly with the engine to "
                    "emit a ~32-dim per-frame embedding."
                ),
                "pros": "task-tuned feature.",
                "cons": "adds trainable params + a frame loader to the engine.",
            },
        ],
        "recommendation": (
            "Start with camera_boundary_edge (CPU, zero new training surface) as a "
            "shape prior; promote to magvit2_token_pool if the camera earns lift. "
            "Either way the camera is a SECONDARY/optional input — it costs ~1k "
            "shots of corpus (see the matrix) and is not part of the internal-"
            "profile core."
        ),
    }


def _recommendation(matrix: list[dict]) -> dict:
    """Assemble 2-3 concrete integrated-input-set options from the matrix."""
    by_key = {r["key"]: r for r in matrix}

    def opt(key: str, headline: str, why: str) -> dict:
        r = by_key[key]
        return {
            "option": headline,
            "combo_key": key,
            "label": r["label"],
            "n_total_coavailable": r["n_total_coavailable"],
            "n_train": r["n_train"],
            "n_calibration": r["n_calibration"],
            "n_test_ood_regime": r["n_test_ood_regime"],
            "cal_adequate": r["cal_adequate"],
            "why": why,
        }

    return {
        "headline": (
            "MSE is the BINDING co-availability constraint: adding MSE drops the "
            "corpus from ~13.3k (mag+ane) to ~4.7k, but MSE is the ONE diagnostic "
            "that directly constrains the internal current/q profile magnetics "
            "cannot see — so it is non-negotiable for grounding. Thomson-any then "
            "costs almost nothing beyond MSE (MSE shots overwhelmingly carry TS)."
        ),
        "options": [
            opt(
                "core_internal",
                "RECOMMENDED — internal-profile core (MSE + Thomson-any)",
                "Adds BOTH the current/q constraint (MSE) and the pressure p(ψ) "
                "constraint (Thomson) — the two GS internal-profile terms — at the "
                "minimal corpus cost beyond MSE alone. Comfortable calibration set. "
                "This is the smallest set that can actually ground the latent.",
            ),
            opt(
                "core_plus_emission",
                "OPTION B — internal core + emission (add bolometer + SXR)",
                "Adds broadly-available radiation (abm) + core-shape SXR (xsx) for a "
                "richer integrated inference, at a modest further corpus cost. SXR "
                "needs 500 kHz→1 kHz aggregation. Good if emission earns lift.",
            ),
            opt(
                "full",
                "OPTION C — full integrated set (+ visible-camera feature)",
                "Maximal input set incl. the rbb camera FEATURE. Smallest corpus; "
                "camera is a secondary shape prior. Choose only if the camera "
                "feature is shown to help and the cal set stays viable.",
            ),
        ],
        "do_not_use": {
            "core_internal_combined": (
                "mag+ane+MSE+Thomson(ayc) specifically restricts to the COMBINED TS "
                "system and roughly halves the core corpus vs TS-any for no physics "
                "gain over any-TS pressure — prefer TS-any."
            ),
        },
        "orchestrator_action": (
            "LOCK ONE integrated-input-set option above as the grounded-engine input "
            "modality (it supersedes input-modality-v0=mag+ane), then dispatch T10 "
            "(integrated-grounding retrain) gated on the locked set. uq-level-v0 and "
            "extrapolation-coordinates remain OPEN and must be settled before T10."
        ),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="GS integrated-input feasibility scoping (T9)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ARTIFACT_DEFAULT,
        help="Output artifact path (compact JSON).",
    )
    args = p.parse_args(argv)

    report = build_feasibility()
    report.save(args.out)

    print("\n=== Per-diagnostic inventory ===")
    for d in report.diagnostic_inventory:
        print(
            f"  {d['name']:<32} n={d['n_shots']:>6}  "
            f"{d['native_hz']:>8.0f} Hz  → {d['constrains']}"
        )
    print("\n=== Co-availability matrix (target = Dα/xim, locked OOD box) ===")
    print(report.matrix_table())
    print("\n=== IR absence ===")
    print(f"  {report.ir_absence['reason']}")
    print("\n=== Recommendation ===")
    print(f"  {report.recommendation['headline']}")
    for o in report.recommendation["options"]:
        print(
            f"  - {o['option']}: N={o['n_total_coavailable']} "
            f"train={o['n_train']} cal={o['n_calibration']} "
            f"ood={o['n_test_ood_regime']} cal_ok={o['cal_adequate']}"
        )
    print(f"\nArtifact written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
