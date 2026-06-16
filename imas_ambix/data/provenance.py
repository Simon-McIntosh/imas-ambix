"""Field-level provenance for MAST Level-2 inputs.

This module homes the single source of truth for *where a Level-2 field
came from* and *whether it is admissible as a world-model input*. Both
the conditioning leakage guard
(:mod:`imas_ambix.camdyn.conditioning`) and the L2 inventory manifest
import from here so the rules live in exactly one place.

The discriminator (lead-corrected principle)
---------------------------------------------
The thing we must keep out of the inputs is **code-reconstructed state**
and anything derived from it — *not* the bare source label. A forward
world model is supposed to *produce* the reconstruction (ψ, j_φ, q,
boundary, globals); feeding it any of those (or a scalar that embeds the
reconstructed boundary) is leakage. But a model also *needs the control
intent* to predict the machine response, so **planned / demanded
waveforms** (the pulse-schedule references, the feed-forward coil
voltage, the gas-valve demands) are legitimate, a-priori-known inputs.

Each L2 field is therefore classified into one of:

``input``
    Measured diagnostic (magnetics, interferometer, bolometer, gas
    flows, soft-x-ray emission, NBI/neutron summary, ...). Admissible.
``planned-action``
    A planned / demanded / feed-forward control waveform from the pulse
    schedule (``XDC_`` demand/setpoint/feed-forward fields). Known a
    priori, NOT a reconstruction. Admissible — these were wrongly banned
    before.
``banned``
    Code-reconstructed equilibrium state (``EFM_``/``ESM_``) or a scalar
    derived from it (the ESM-derived ``line_average_n_e`` and
    ``greenwald_density`` embed the EFIT boundary), or any ``XDC_`` field
    that encodes an error/residual against the *achieved/reconstructed*
    state. Rejected.
``probe-target``
    Dα filter-spectrometer (``XIM_``) — a downstream probe target, kept
    clearly separable and default-off, never a default input.
``infra``
    Static machine geometry / sensor description (no ``uda_name``, or a
    geometry channel). Not a per-shot signal; carried for context only.

The classifier keys off the **L1 source prefix recovered from
``uda_name``** plus the IMAS group/variable name. ``uda_name`` appears in
two written forms in the corpus — an underscore form (``EFM_PSI(R,Z)``,
``XDC_GAS_F_G1``) and a slash/path form (``/xdc/gas/f/tc5a``,
``/xsx/HCAM/L/1``); :func:`source_of_uda` normalises both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# uda_name → Level-1 source prefix
# ---------------------------------------------------------------------------
#
# uda_name carries the L1 acquisition source as its leading token. Two
# written forms occur in the FAIR-MAST L2 corpus:
#   underscore : "EFM_PSI(R,Z)", "AMC_PLASMA CURRENT", "XSX_HCAML#1"
#   slash/path : "/xdc/gas/f/tc5a", "/xsx/HCAM/L/1", "xbt/channel01"
# Both reduce to a 3-letter source code by taking the first token,
# stripping a leading slash, splitting on '_' / '/' / whitespace, and
# upper-casing.

_NONALNUM = re.compile(r"[^A-Za-z0-9]")


def source_of_uda(uda_name: str | None) -> str | None:
    """Return the upper-cased L1 source prefix for a ``uda_name``.

    Handles both the underscore form (``EFM_PSI(R,Z)`` → ``EFM``) and the
    slash/path form (``/xdc/gas/f/tc5a`` → ``XDC``; ``xbt/channel01`` →
    ``XBT``). Returns ``None`` for a missing/empty ``uda_name``.
    """
    if not uda_name:
        return None
    s = str(uda_name).strip()
    if not s:
        return None
    s = s.lstrip("/")
    # First token before any separator (_, /, whitespace, or '(' etc.).
    head = re.split(r"[_/\s(#]", s, maxsplit=1)[0]
    head = _NONALNUM.sub("", head).upper()
    return head or None


# Source prefix → Level-1 acquisition system. Mirrors paths.LEVEL1_SOURCES
# but keyed by the *uda* prefix actually written into L2 fields (some L2
# groups merge several L1 sources, e.g. magnetics ← AMA/AMB/AMC/ASM/XMB/
# XMC/XMO, summary ← AMC/ABM/ANB/ANU/ESM).
UDA_SOURCE_MAP: dict[str, str] = {
    # --- magnetics (measured field/flux probes + raw voltages) ---
    "AMA": "magnetics",
    "AMB": "magnetics",
    "AMC": "magnetics",  # also pf_active coil/sol current + summary.ip
    "AMH": "magnetics",
    "AMM": "magnetics",
    "ASM": "magnetics",  # saddle coils
    "XMA": "magnetics_raw",
    "XMB": "magnetics_raw",
    "XMC": "magnetics_raw",
    "XMO": "magnetics_raw",  # omaha
    # --- interferometer (line density) ---
    "ANE": "interferometer",
    # --- bolometer (radiated power) ---
    "ABM": "bolometer",
    # --- gas injection (measured flows) ---
    "AGA": "gas_injection",
    # --- soft x-rays (emission) ---
    "XSX": "soft_x_rays",
    # --- visible spectrometer ---
    "XIM": "spectrometer_visible",  # Dα filter spectrometer (probe target)
    "ADG": "spectrometer_visible",  # density-gradient (measured/derived)
    "XBT": "spectrometer_visible",  # beam-emission spectroscopy (BES)
    # --- NBI ---
    "ANB": "nbi",
    # --- neutron ---
    "ANU": "neutron_diagnostic",
    # --- charge exchange ---
    "ACT": "charge_exchange",
    # --- Thomson scattering ---
    "ATM": "thomson_scattering",
    "AYC": "thomson_scattering",
    "AYE": "thomson_scattering",
    # --- reconstructed equilibrium (BANNED) ---
    "EFM": "equilibrium_efit",
    "ESM": "equilibrium_solovev",
    # --- pulse schedule / discharge control (planned actions) ---
    "XDC": "pulse_schedule",
}
"""uda-prefix → Level-1 acquisition system. Keyed by the prefix found in
the L2 ``uda_name``; several L2 groups merge multiple L1 sources."""

# Reconstructed-equilibrium sources — these *are* the code-reconstructed
# state the world model is meant to produce.
RECONSTRUCTED_SOURCES: frozenset[str] = frozenset({"EFM", "ESM"})

# Pulse-schedule (control) source — planned unless the field is a
# reconstruction-residual (see _is_reconstruction_residual).
PLANNED_SOURCE = "XDC"

# Probe-target sources (default-off, kept separable).
PROBE_TARGET_SOURCES: frozenset[str] = frozenset({"XIM"})


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

Classification = str  # one of the labels below; str for JSON-friendliness

INPUT: Classification = "input"
PLANNED_ACTION: Classification = "planned-action"
BANNED: Classification = "banned"
PROBE_TARGET: Classification = "probe-target"
INFRA: Classification = "infra"

ALL_CLASSIFICATIONS: tuple[Classification, ...] = (
    INPUT,
    PLANNED_ACTION,
    BANNED,
    PROBE_TARGET,
    INFRA,
)

# Reconstruction-derived scalars: measured-source labels would *look*
# admissible, but the value embeds the EFIT/Solov'ev reconstruction. The
# canonical pair lives in ``summary`` and is sourced from ESM_ (so the
# source check already bans them), but we name them explicitly so the
# intent is documented and a future relabelling cannot silently re-admit
# them.
RECONSTRUCTION_DERIVED_FIELDS: frozenset[tuple[str, str]] = frozenset(
    {
        ("summary", "line_average_n_e"),  # ESM_NE_BAR — embeds EFIT boundary
        ("summary", "greenwald_density"),  # ESM_N_GREENWALD — uses minor radius
    }
)

# Tokens in a uda_name / variable name that mark an XDC field as a
# residual/error against the achieved (reconstructed) state rather than a
# pure demand. Anything matching → BANNED even though the source is XDC.
_RESIDUAL_TOKENS = re.compile(
    r"(err|error|resid|residual|achiev|deviation|mismatch|fluxerr)",
    re.IGNORECASE,
)

# Tokens that positively mark an XDC field as a demand / setpoint /
# feed-forward command (the legitimate planned action). Used for the
# clear-cut authorise; an XDC field matching neither demand nor residual
# is treated conservatively (see classify_l2_field).
_DEMAND_TOKENS = re.compile(
    r"(ipref|nelref|_ip_|_density_|_pf_f|_gas_f|_gas_t|/ip/|/density/"
    r"|/pf/f|/gas/f|/gas/t|ref|demand|setpoint|target|feedforward|feed_forward"
    r"|/f/|_f_|_t_)",
    re.IGNORECASE,
)


def _is_reconstruction_residual(uda_name: str | None, var: str) -> bool:
    """True if an XDC field encodes an error/residual vs achieved state."""
    hay = f"{uda_name or ''} {var}"
    return bool(_RESIDUAL_TOKENS.search(hay))


def _looks_like_demand(uda_name: str | None, var: str) -> bool:
    """True if an XDC field looks like a demand / setpoint / feed-forward."""
    hay = f"{uda_name or ''} {var}"
    return bool(_DEMAND_TOKENS.search(hay))


@dataclass(frozen=True)
class FieldClassification:
    """Result of classifying one L2 field.

    Attributes
    ----------
    classification:
        One of :data:`ALL_CLASSIFICATIONS`.
    source:
        L1 source prefix recovered from ``uda_name`` (e.g. ``AMC``,
        ``EFM``, ``XDC``), or ``None`` for geometry/infra fields with no
        ``uda_name``.
    level1_system:
        The acquisition system the source maps to
        (:data:`UDA_SOURCE_MAP`), or ``None``.
    reason:
        Short human-readable justification (for the manifest / review).
    review:
        True when the field could not be classified confidently and was
        defaulted to ``banned`` pending human review.
    """

    classification: Classification
    source: str | None
    level1_system: str | None
    reason: str
    review: bool = False


def classify_l2_field(
    group: str,
    var: str,
    uda_name: str | None,
) -> FieldClassification:
    """Classify one L2 field by the reconstruction-vs-plan principle.

    Parameters
    ----------
    group:
        IMAS group / IDS name (e.g. ``magnetics``, ``equilibrium``,
        ``pulse_schedule``, ``summary``).
    var:
        Variable name within the group.
    uda_name:
        The field's ``uda_name`` attribute (underscore or slash form), or
        ``None`` for geometry / static-description fields.

    Returns
    -------
    FieldClassification
        The label plus the recovered source, system, and a reason. When a
        field is genuinely ambiguous it is defaulted to ``banned`` with
        ``review=True`` so it surfaces for human review rather than
        silently leaking.
    """
    source = source_of_uda(uda_name)
    system = UDA_SOURCE_MAP.get(source) if source else None

    # 1. No uda_name → static geometry / machine description (infra).
    if source is None:
        return FieldClassification(
            INFRA,
            None,
            None,
            "no uda_name — static geometry / machine description",
        )

    # 2. Reconstruction-derived scalars (explicit field list) → BANNED
    #    even before the source check, so the intent is documented.
    if (group, var) in RECONSTRUCTION_DERIVED_FIELDS:
        return FieldClassification(
            BANNED,
            source,
            system,
            "reconstruction-derived scalar — embeds the EFIT/Solov'ev boundary",
        )

    # 3. Code-reconstructed equilibrium state (EFM_/ESM_) → BANNED.
    if source in RECONSTRUCTED_SOURCES:
        return FieldClassification(
            BANNED,
            source,
            system,
            "code-reconstructed equilibrium state (the world model must "
            "produce this, not consume it)",
        )

    # 4. Pulse-schedule (XDC_) — planned action unless it is a
    #    reconstruction residual/error against the achieved state.
    if source == PLANNED_SOURCE:
        if _is_reconstruction_residual(uda_name, var):
            return FieldClassification(
                BANNED,
                source,
                system,
                "pulse-schedule field encodes an error/residual against "
                "the achieved/reconstructed state",
            )
        if _looks_like_demand(uda_name, var):
            return FieldClassification(
                PLANNED_ACTION,
                source,
                system,
                "planned/demanded control waveform (known a priori — "
                "feed-forward command / setpoint)",
            )
        # XDC, not a residual, but no positive demand marker → ambiguous.
        # Default to BANNED and flag for review (fail safe).
        return FieldClassification(
            BANNED,
            source,
            system,
            "ambiguous pulse-schedule field (no clear demand marker) — "
            "defaulted to banned pending review",
            review=True,
        )

    # 5. Probe-target sources (Dα filter spectrometer) → probe-target.
    if source in PROBE_TARGET_SOURCES:
        return FieldClassification(
            PROBE_TARGET,
            source,
            system,
            "downstream probe target — default-off, kept separable",
        )

    # 6. Known measured/diagnostic source → input.
    if system is not None:
        return FieldClassification(
            INPUT,
            source,
            system,
            f"measured diagnostic ({system})",
        )

    # 7. Unknown source prefix → default to BANNED for review (fail safe).
    return FieldClassification(
        BANNED,
        source,
        None,
        f"unknown source prefix {source!r} — defaulted to banned pending review",
        review=True,
    )


def is_admissible_input(group: str, var: str, uda_name: str | None) -> bool:
    """True if a field may be used as a world-model input.

    Admissible = ``input`` (measured) or ``planned-action`` (pulse-schedule
    demand). Probe targets, infra, and anything banned are NOT admissible
    as a default input.
    """
    return classify_l2_field(group, var, uda_name).classification in (
        INPUT,
        PLANNED_ACTION,
    )
