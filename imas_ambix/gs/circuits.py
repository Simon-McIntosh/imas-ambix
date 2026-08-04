"""MAST PF-coil power-supply / circuit description.

This module answers two questions the geometry and operator modules do not:
*which power supply drives which coil, and by what raw-channel path*,
and *is every coil actually powered, or are some circuits structural/passive*.
It is sourced from the MAST control-system machine description —
``pfSystems.xml`` (``efitpp-jet/utilities/machine/mast/inputPreparation/
standard_input/pfSystems.xml``) — which enumerates 23 ``pfCircuit`` / 24
``pfCoil`` / 23 ``pfSupply`` entries for the ORIGINAL MAST device (shots
~11000-30000; **not** MAST-U).  The values below are transcribed from that
file (hardcoded here, same pattern as ``operator._PF_COIL_CENTROID`` — the
source is a static machine description, not per-shot data) and independently
cross-validated against the ``efm`` static geometry (``fcoil_circ`` /
``fcoil_xmult`` — see :mod:`imas_ambix.gs.geometry`) and the raw ``amc``
channel listing on sample shots spanning both in-use ``fcoil`` discretisations
(938- and 1004-filament campaigns).

Two circuit families
---------------------
* **13 active circuits** (``pfCircuit`` id 1-13) — each has its own dedicated
  power supply and raw ``amc`` current channel(s).  Circuit 1 (``ohmic``) is
  the one series circuit: ``solenoid1`` + ``solenoid2`` share one supply
  (``amc_sol current``).  Circuits 2-13 (P2IU/P2OU/P2IL/P2OL, P3U/P3L,
  P4U/P4L, P5U/P5L, P6U/P6L) each have their own supply.  For 10 of the 13
  (everything except ``sol``/``p6u``/``p6l``) the ``amc`` group carries BOTH a
  ``*_feed_current`` (raw per-turn supply current, kA) and a
  ``*_coil_current`` channel; measured on four held-out shots (18502-18505,
  ``docs/mast-coil-circuits.html`` §3), ``coil_current = feed_current ×
  (supply_scaling_a / 1000)`` **exactly** (integer ratio, zero residual) — the
  ``coil_current`` channel is the feed current already multiplied by the
  coil's turn count into amp-turns, matching
  :data:`imas_ambix.gs.operator._KA_TURN_TO_A`'s documented convention.  For
  ``sol``/``p6u``/``p6l`` only one channel exists and is used directly.
* **10 case circuits** (``pfCircuit`` id 14-23) — passive/induced structural
  conductors (the coil casings for P2U/P2L/P3U/P3L/P4U/P4L/P5U/P5L/P6U/P6L),
  each on its OWN nominal ``pfSupply`` entry but NOT independently driven: 8
  of them (P2U/P2L/P3U/.../P5L) have a real measured ``amc`` case-current
  channel (small induced current, not a controllable actuator); the P6U/P6L
  cases have **no** ``amc`` channel at all — ``pfSystems.xml`` states outright
  "unknown for MAST, constrained to 0" (``scalingFactor=0``) and this is
  confirmed by their absence from every sampled shot's ``amc`` listing.

Why the classifier needs this table and not just geometry
--------------------------------------------------------
A case sits a couple of centimetres from the winding it encloses, so a
nearest-centroid rule cannot tell the two apart.  Measured directly (three
sample shots, two ``fcoil`` signatures): each of the 8 non-P6 case circuits'
filament centroid lands within the 8 cm match radius of its co-located active
coil — the "P2U case" filaments, R≈0.50 Z≈1.77, sit 2 cm from the P2IU coil
centroid.  On geometry alone every one of them reads as a redundant
discretisation of the active coil and would be merged into that coil's ``G_pf``
column, driven by the coil's amp-turn channel instead of by its own much
smaller induced case current.

:attr:`CaseCircuit.geometry_confusable_with` records which active ``coil_label``
each case circuit is confusable with, and
:func:`case_circuit_for_active_coil` resolves the correspondence the other way.
:func:`imas_ambix.gs.operator.classify_circuits` consults it before accepting a
centroid match, so a case circuit keeps its own column and its own measured
``*_case_current`` channel.  Only this transcribed id correspondence can make
that distinction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACT_NAME = "mast_pf_circuits.json"

MAST_ACTIVE_CIRCUIT_COUNT = 13
MAST_CASE_CIRCUIT_COUNT = 10


@dataclass(frozen=True)
class ActiveCircuit:
    """One MAST PF active circuit: dedicated supply, own ``amc`` channel(s).

    ``coil_label`` is the canonical geometry key shared with
    :data:`imas_ambix.gs.operator._PF_COIL_AMC` /
    :data:`imas_ambix.gs.operator._PF_COIL_CENTROID` — this module's
    ``preferred_current_channel`` and the operator's ``_PF_COIL_AMC`` are
    pinned equal by test.
    """

    circuit_id: int
    name: str
    coil_label: str
    pf_coil_names: tuple[str, ...]
    series: bool
    supply_id: int
    supply_signal_name: str
    supply_scaling_a: float
    amc_channel_prefix: str
    l1_feed_channel: str | None
    l1_coil_channel: str

    @property
    def turns(self) -> float:
        """Coil turn count implied by the supply scaling (``scalingFactor/1000``).

        Confirmed against raw data: ``coil_current / feed_current`` measured
        on four held-out shots equals this value exactly (zero residual) for
        every circuit that carries both channels.
        """
        return self.supply_scaling_a / 1000.0

    def preferred_current_channel(self) -> str:
        """The GS-path (amp-turn) actuator channel — see module docstring."""
        return self.l1_coil_channel


@dataclass(frozen=True)
class CaseCircuit:
    """One MAST PF-coil-case (structural/induced) circuit — see module docstring."""

    circuit_id: int
    name: str
    coils_encased: str
    supply_id: int
    supply_signal_name: str
    supply_scaling_a: float
    l1_case_channel: str | None
    constrained_zero: bool
    geometry_confusable_with: str


# --- The 13 active circuits (pfSystems.xml pfCircuit id 1-13) ----------
#
# name, coil_label, pf_coil_names, series, supply_id, supply_signal_name,
# supply_scaling_a, amc_channel_prefix, l1_feed_channel, l1_coil_channel

_ACTIVE_ROWS: tuple[
    tuple[int, str, str, tuple[str, ...], bool, int, str, float, str, str | None, str],
    ...,
] = (
    (
        1,
        "ohmic",
        "sol",
        ("solenoid1", "solenoid2"),
        True,
        1,
        "amc_sol current",
        1000.0,
        "sol",
        None,
        "sol_current",
    ),
    (
        2,
        "P2IU",
        "p2iu",
        ("p2iu",),
        False,
        2,
        "amc_p2iu feed current",
        12000.0,
        "p2iu",
        "p2iu_feed_current",
        "p2iu_coil_current",
    ),
    (
        3,
        "P2OU",
        "p2ou",
        ("p2ou",),
        False,
        3,
        "amc_p2ou feed current",
        8000.0,
        "p2ou",
        "p2ou_feed_current",
        "p2ou_coil_current",
    ),
    (
        4,
        "P2IL",
        "p2il",
        ("p2il",),
        False,
        4,
        "amc_p2il feed current",
        12000.0,
        "p2il",
        "p2il_feed_current",
        "p2il_coil_current",
    ),
    (
        5,
        "P2OL",
        "p2ol",
        ("p2ol",),
        False,
        5,
        "amc_p2ol feed current",
        8000.0,
        "p2ol",
        "p2ol_feed_current",
        "p2ol_coil_current",
    ),
    (
        6,
        "P3U",
        "p3u",
        ("p3u",),
        False,
        6,
        "amc_p3u feed current",
        8000.0,
        "p3u",
        "p3u_feed_current",
        "p3u_coil_current",
    ),
    (
        7,
        "P3L",
        "p3l",
        ("p3l",),
        False,
        7,
        "amc_p3l feed current",
        8000.0,
        "p3l",
        "p3l_feed_current",
        "p3l_coil_current",
    ),
    (
        8,
        "P4U",
        "p4u",
        ("p4u",),
        False,
        8,
        "amc_p4u feed current",
        23000.0,
        "p4u",
        "p4u_feed_current",
        "p4u_coil_current",
    ),
    (
        9,
        "P4L",
        "p4l",
        ("p4l",),
        False,
        9,
        "amc_p4l feed current",
        23000.0,
        "p4l",
        "p4l_feed_current",
        "p4l_coil_current",
    ),
    (
        10,
        "P5U",
        "p5u",
        ("p5u",),
        False,
        10,
        "amc_p5u feed current",
        23000.0,
        "p5u",
        "p5u_feed_current",
        "p5u_coil_current",
    ),
    (
        11,
        "P5L",
        "p5l",
        ("p5l",),
        False,
        11,
        "amc_p5l feed current",
        23000.0,
        "p5l",
        "p5l_feed_current",
        "p5l_coil_current",
    ),
    (
        12,
        "P6U",
        "p6u",
        ("p6u",),
        False,
        12,
        "amc_p6u current",
        1000.0,
        "p6u",
        None,
        "p6u_current",
    ),
    (
        13,
        "P6L",
        "p6l",
        ("p6l",),
        False,
        13,
        "amc_p6l current",
        1000.0,
        "p6l",
        None,
        "p6l_current",
    ),
)

# --- The 10 case circuits (pfSystems.xml pfCircuit id 14-23) -----------
#
# name, coils_encased, supply_id, supply_signal_name, supply_scaling_a,
# l1_case_channel, constrained_zero, geometry_confusable_with

_CASE_ROWS: tuple[tuple[int, str, str, int, str, float, str | None, bool, str], ...] = (
    (
        14,
        "P2U case current",
        "P2U (P2IU+P2OU)",
        14,
        "amc_p2u case current",
        1000.0,
        "p2u_case_current",
        False,
        "p2iu",
    ),
    (
        15,
        "P2L case current",
        "P2L (P2IL+P2OL)",
        15,
        "amc_p2l case current",
        1000.0,
        "p2l_case_current",
        False,
        "p2il",
    ),
    (
        16,
        "P3U case current",
        "P3U",
        16,
        "amc_p3u case current",
        1000.0,
        "p3u_case_current",
        False,
        "p3u",
    ),
    (
        17,
        "P3L case current",
        "P3L",
        17,
        "amc_p3l case current",
        1000.0,
        "p3l_case_current",
        False,
        "p3l",
    ),
    (
        18,
        "P4U case current",
        "P4U",
        18,
        "amc_p4u case current",
        1000.0,
        "p4u_case_current",
        False,
        "p4u",
    ),
    (
        19,
        "P4L case current",
        "P4L",
        19,
        "amc_p4l case current",
        1000.0,
        "p4l_case_current",
        False,
        "p4l",
    ),
    (
        20,
        "P5U case current",
        "P5U",
        20,
        "amc_p5u case current",
        1000.0,
        "p5u_case_current",
        False,
        "p5u",
    ),
    (
        21,
        "P5L case current",
        "P5L",
        21,
        "amc_p5l case current",
        1000.0,
        "p5l_case_current",
        False,
        "p5l",
    ),
    (
        22,
        "P6U case current",
        "P6U",
        22,
        "amc_p6u case current",
        0.0,
        None,
        True,
        "p6u",
    ),
    (
        23,
        "P6L case current",
        "P6L",
        23,
        "amc_p6l case current",
        0.0,
        None,
        True,
        "p6l",
    ),
)

_ACTIVE_CIRCUITS: tuple[ActiveCircuit, ...] = tuple(
    ActiveCircuit(*row) for row in _ACTIVE_ROWS
)
_CASE_CIRCUITS: tuple[CaseCircuit, ...] = tuple(CaseCircuit(*row) for row in _CASE_ROWS)

_BY_COIL_LABEL: dict[str, ActiveCircuit] = {c.coil_label: c for c in _ACTIVE_CIRCUITS}


# --- Public API ---------------------------------------------------------


def active_circuits() -> tuple[ActiveCircuit, ...]:
    """All 13 active MAST PF circuits (own supply + amc channel)."""
    return _ACTIVE_CIRCUITS


def case_circuits() -> tuple[CaseCircuit, ...]:
    """All 10 MAST PF coil-case (structural/induced) circuits."""
    return _CASE_CIRCUITS


def circuits() -> dict[str, tuple[ActiveCircuit, ...] | tuple[CaseCircuit, ...]]:
    """Both circuit families, keyed ``"active"`` / ``"case"``."""
    return {"active": _ACTIVE_CIRCUITS, "case": _CASE_CIRCUITS}


def active_circuit_for_coil(coil_label: str) -> ActiveCircuit:
    """Look up the active circuit for a canonical coil label (e.g. ``"p3u"``).

    Raises ``KeyError`` with the valid label set if ``coil_label`` is unknown —
    verify-and-flag, never silently return a default.
    """
    try:
        return _BY_COIL_LABEL[coil_label]
    except KeyError:
        raise KeyError(
            f"unknown MAST PF coil label {coil_label!r}; known labels: "
            f"{sorted(_BY_COIL_LABEL)}"
        ) from None


def preferred_current_channel(coil_label: str) -> str:
    """The preferred (amp-turn) ``amc`` channel for a coil — see module docstring."""
    return active_circuit_for_coil(coil_label).preferred_current_channel()


def case_circuit_for_active_coil(coil_label: str) -> CaseCircuit | None:
    """The case circuit that :func:`imas_ambix.gs.operator.classify_circuits`
    will fold into ``coil_label``'s ``G_pf`` column, if any."""
    for cc in _CASE_CIRCUITS:
        if cc.geometry_confusable_with == coil_label:
            return cc
    return None


def verify_amc_channels(amc_channels: Sequence[str]) -> dict[str, list[str]]:
    """Cross-check every circuit's channel(s) against an observed ``amc`` listing.

    Returns ``{"missing_active": [...], "missing_case": [...],
    "unexpectedly_present_zero_channels": [...]}`` — the last catches a
    surprise if a P6-case channel ever DOES show up (would contradict the
    "constrained to 0 / unmeasured" machine-description claim).
    """
    avail = set(amc_channels)
    missing_active: list[str] = []
    for ac in _ACTIVE_CIRCUITS:
        if ac.l1_coil_channel not in avail:
            missing_active.append(ac.l1_coil_channel)
        if ac.l1_feed_channel is not None and ac.l1_feed_channel not in avail:
            missing_active.append(ac.l1_feed_channel)
    missing_case: list[str] = []
    unexpected_zero: list[str] = []
    for cc in _CASE_CIRCUITS:
        if cc.constrained_zero:
            # a zero/unmeasured case channel should NOT appear in amc.
            probe = f"{cc.coils_encased.split()[0].lower()}_case_current"
            if probe in avail:
                unexpected_zero.append(probe)
        elif cc.l1_case_channel not in avail:
            missing_case.append(cc.l1_case_channel or "")
    return {
        "missing_active": missing_active,
        "missing_case": missing_case,
        "unexpectedly_present_zero_channels": unexpected_zero,
    }


# --- Artifact I/O --------------------------------------------------------


def to_dict() -> dict[str, object]:
    """The full machine-readable description as a plain dict."""
    return {
        "schema": "mast-pf-circuits-v0",
        "source": (
            "efitpp-jet/utilities/machine/mast/inputPreparation/standard_input/"
            "pfSystems.xml (23 pfCircuit / 24 pfCoil / 23 pfSupply, MAST — not "
            "MAST-U); cross-validated against efm fcoil_circ/fcoil_xmult and "
            "raw amc channel presence (docs/mast-coil-circuits.html)"
        ),
        "n_active_circuits": len(_ACTIVE_CIRCUITS),
        "n_case_circuits": len(_CASE_CIRCUITS),
        "active_circuits": [asdict(c) for c in _ACTIVE_CIRCUITS],
        "case_circuits": [asdict(c) for c in _CASE_CIRCUITS],
    }


def write_artifact(out_path: Path | None = None) -> Path:
    """Write the circuit description as JSON under ``gs/artifacts/``."""
    out_path = out_path or (_ARTIFACT_DIR / _ARTIFACT_NAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_dict(), indent=2))
    return out_path


if __name__ == "__main__":
    path = write_artifact()
    print(f"wrote {path}")
