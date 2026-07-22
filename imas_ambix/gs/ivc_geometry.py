"""Era-keyed 3-D geometry of MAST's non-axisymmetric coils and sensor loops.

Companion data + builders for :mod:`imas_ambix.gs.filaments3d`.  Where that
module supplies the *kernels* (straight-segment + arc Biot-Savart, flux linkage,
mutual inductance), this module supplies the *conductor centrelines and sensor
loop topology* of the toroidally-broken machine — the pieces the axisymmetric
elliptic-ring model cannot represent:

* the ex-vessel **error-field correction coils** (EFCC) — energised in nearly
  every shot (``error_field_02``/``error_field_05``), the abundant validation
  source;
* the in-vessel **ELM / RMP coils** — a lower row installed first, an upper row
  added later, so the conductor complement is *era-dependent*;
* the outer **saddle detector loops** — toroidally-partial pickups that (unlike
  the full flux loops) see the n != 0 applied field.

Everything is plain versioned in-tree data (module-level constants) plus small
builders that return picture-frame polylines / ``Conductor`` objects the kernels
(:mod:`imas_ambix.gs.filaments3d`) consume directly.  No I/O, no external
geometry files — the form :func:`~imas_ambix.gs.filaments3d.picture_frame` accepts.

Geometry provenance
-------------------
* **EFCC** — Kirk et al., arXiv:1312.6507 (CCFE, 2013): four ex-vessel coils at
  major radius ~2.9 m, each spanning 83 deg toroidally with 3 turns and 15
  kA-turn max, wired as two opposite-in-series pairs, ``EFCC_2_8`` (sector
  centres 45/225 deg) and ``EFCC_5_11`` (sector centres 315/135 deg), on
  independent supplies.  Sign convention: a positive ``EFCC_2`` current gives
  B_r < 0 at sector 2.  The **vertical extent is not pinned by the paper** — it
  is carried as a declared parameter (:data:`EFCC_Z_HALF`) with an uncertainty
  band; the sensor-coupling validation constrains it from measured data.
* **ELM / RMP** — Kirk et al.: in-vessel coils between P4 and P5; a lower row of
  12 (all 12 sectors) installed first, an upper row of 6 (odd sectors only)
  added later; 5.6 kA-turn max.  The in-vessel radial/vertical placement and the
  per-coil toroidal span are not pinned to the millimetre by the literature —
  the nominal values below carry an explicit uncertainty note and any downstream
  absolute fit treats the worst-known dimensions as bounded nuisances.

Era keying
----------
The conductor complement changes at two documented campaign boundaries (shot
numbers), so the coil set is a function of shot:

* ``shot < ELM_LOWER_FIRST_SHOT``      -> EFCC only (no in-vessel ELM/RMP set);
* ``ELM_LOWER_FIRST_SHOT <= shot``     -> EFCC + 12 lower ELM coils;
* ``ELM_UPPER_FIRST_SHOT <= shot``     -> EFCC + 12 lower + 6 upper ELM coils.

Loop topology
-------------
Layer-B (energised n != 0) immunity depends on whether a magnetic sensor is a
*complete* toroidal loop (integrates the n != 0 vector potential to zero) or a
*partial* one (sees it fully).  :func:`loop_topology` records that per sensor
family so the forward model never applies an energised-field term to an immune
loop, nor omits it from an exposed saddle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imas_ambix.gs import filaments3d as f3d

# --- EFCC (ex-vessel error-field correction coils) -------------------------
#: major radius of the ex-vessel EFCC coils [m] (Kirk et al., arXiv:1312.6507).
EFCC_RADIUS = 2.9
#: toroidal span of each EFCC coil [deg].
EFCC_SPAN_DEG = 83.0
#: turns per EFCC coil.
EFCC_TURNS = 3
#: maximum pair current [kA-turn].
EFCC_MAX_KAT = 15.0
#: half-height of the EFCC picture frame [m].  NOT pinned by Kirk et al.; a
#: declared nuisance carried with the uncertainty band below and constrained by
#: the sensor-coupling validation (``scripts/efcc_absolute_coupling.py``).
EFCC_Z_HALF = 1.4
#: plausible half-height range for the uncertainty sweep [m].
EFCC_Z_HALF_RANGE = (0.8, 2.0)
#: overall current-direction sign so a positive pair current reproduces Kirk et
#: al.'s published convention (positive ``EFCC_2`` -> B_r < 0 at sector 2).  The
#: bare picture-frame arcs run phi-increasing; this bakes the machine sign in.
EFCC_POLARITY = -1

#: The two independently-supplied EFCC pairs.  Each entry: the amc drive channel
#: -> the two sector centres [deg] wired opposite-in-series (n=1) with their
#: series signs, and the human coil-pair name.
EFCC_PAIRS: dict[str, dict] = {
    "error_field_02": {
        "name": "EFCC_2_8",
        "centres_deg": (45.0, 225.0),
        "signs": (+1, -1),
    },
    "error_field_05": {
        "name": "EFCC_5_11",
        "centres_deg": (315.0, 135.0),
        "signs": (+1, -1),
    },
}

# --- In-vessel ELM / RMP coils --------------------------------------------
#: first shot with the lower ELM/RMP row (12 coils, all sectors) installed.
ELM_LOWER_FIRST_SHOT = 19031
#: first shot with the upper ELM/RMP row (6 coils, odd sectors) added.
ELM_UPPER_FIRST_SHOT = 25404
#: number of toroidal sectors on MAST.
N_SECTORS = 12
#: nominal in-vessel major radius of the ELM/RMP coils [m] (between P4 and P5).
#: Uncertainty: +/- ~0.05 m — not pinned to the mm by the literature.
ELM_RADIUS = 1.45
#: nominal vertical band of the lower row (below the midplane) [m].
ELM_LOWER_Z = (-0.95, -0.45)
#: nominal vertical band of the upper row (above the midplane) [m].
ELM_UPPER_Z = (0.45, 0.95)
#: toroidal span of each ELM coil [deg] (a sector arc with an inter-coil gap).
ELM_SPAN_DEG = 24.0
#: turns per ELM coil.
ELM_TURNS = 1
#: maximum ELM/RMP coil current [kA-turn].
ELM_MAX_KAT = 5.6

# --- Sensor loop topology --------------------------------------------------
#: full toroidal loops (integrate n != 0 to zero -> immune to layer B).
_FULL_LOOP_PREFIXES = ("fl_", "silop", "flcc", "fl")
#: toroidally-partial loops (fully exposed to n != 0 -> need the layer-B term).
_PARTIAL_LOOP_PREFIXES = ("sad_out", "sad_", "saddle", "xmb")


@dataclass(frozen=True)
class CoilSpec:
    """One picture-frame coil: sector centre, span, radius, vertical band, turns.

    ``build`` returns the densely-sampled closed centreline
    (:func:`imas_ambix.gs.filaments3d.picture_frame`) the kernels integrate.
    """

    name: str
    centre_deg: float
    span_deg: float
    radius: float
    z_lo: float
    z_hi: float
    turns: int = 1

    def build(self, *, n_arc: int = 80, n_leg: int = 40) -> np.ndarray:
        return f3d.picture_frame(
            np.deg2rad(self.centre_deg),
            np.deg2rad(self.span_deg),
            self.radius,
            self.z_lo,
            self.z_hi,
            n_arc=n_arc,
            n_leg=n_leg,
        )


@dataclass(frozen=True)
class CoilPair:
    """An opposite-in-series coil pair on one drive channel (an n=1 saddle pair)."""

    drive_channel: str
    name: str
    coils: tuple[CoilSpec, ...]
    series_signs: tuple[int, ...]
    polarity: int = 1

    def build(self, **kw) -> list[tuple[np.ndarray, int]]:
        """Return ``[(polyline, series_current_sign), ...]`` for the pair."""
        return [
            (c.build(**kw), self.polarity * s)
            for c, s in zip(self.coils, self.series_signs, strict=True)
        ]


@dataclass(frozen=True)
class CoilSet:
    """The era-resolved non-axisymmetric coil complement for one shot."""

    shot: int
    era: str
    efcc_pairs: dict[str, CoilPair]
    elm_coils: tuple[CoilSpec, ...]

    @property
    def n_elm(self) -> int:
        return len(self.elm_coils)

    def summary(self) -> dict:
        return {
            "shot": self.shot,
            "era": self.era,
            "n_efcc_pairs": len(self.efcc_pairs),
            "n_efcc_coils": sum(len(p.coils) for p in self.efcc_pairs.values()),
            "n_elm_coils": self.n_elm,
        }


# --- EFCC builders ---------------------------------------------------------


def efcc_pairs(*, z_half: float = EFCC_Z_HALF) -> dict[str, CoilPair]:
    """Build the two EFCC drive pairs at the published ex-vessel geometry.

    ``z_half`` overrides the (unpinned) vertical half-extent — the knob the
    sensor-coupling validation sweeps to constrain it from data.
    """
    out: dict[str, CoilPair] = {}
    for chan, spec in EFCC_PAIRS.items():
        coils = tuple(
            CoilSpec(
                name=f"{spec['name']}_s{int(round(c / 30.0)) % N_SECTORS}",
                centre_deg=c,
                span_deg=EFCC_SPAN_DEG,
                radius=EFCC_RADIUS,
                z_lo=-z_half,
                z_hi=+z_half,
                turns=EFCC_TURNS,
            )
            for c in spec["centres_deg"]
        )
        out[chan] = CoilPair(
            drive_channel=chan,
            name=spec["name"],
            coils=coils,
            series_signs=spec["signs"],
            polarity=EFCC_POLARITY,
        )
    return out


# --- ELM / RMP builders ----------------------------------------------------


def _elm_row(
    z_lo: float, z_hi: float, sectors: range | tuple, tag: str
) -> tuple[CoilSpec, ...]:
    """Build one toroidal row of ELM coils, one per named sector centre."""
    return tuple(
        CoilSpec(
            name=f"elm_{tag}_s{s}",
            centre_deg=(s + 0.5) * (360.0 / N_SECTORS),
            span_deg=ELM_SPAN_DEG,
            radius=ELM_RADIUS,
            z_lo=z_lo,
            z_hi=z_hi,
            turns=ELM_TURNS,
        )
        for s in sectors
    )


def elm_coils_for_era(shot: int) -> tuple[CoilSpec, ...]:
    """Era-keyed in-vessel ELM/RMP coil complement (0, 12, or 18 coils)."""
    if shot < ELM_LOWER_FIRST_SHOT:
        return ()
    lower = _elm_row(*ELM_LOWER_Z, range(N_SECTORS), "l")
    if shot < ELM_UPPER_FIRST_SHOT:
        return lower
    # upper row: odd sectors only (6 coils)
    upper = _elm_row(*ELM_UPPER_Z, tuple(range(1, N_SECTORS, 2)), "u")
    return lower + upper


def era_name(shot: int) -> str:
    """Human label for the coil era a shot falls in."""
    if shot < ELM_LOWER_FIRST_SHOT:
        return "pre-invessel (EFCC only)"
    if shot < ELM_UPPER_FIRST_SHOT:
        return "lower row (12 in-vessel coils)"
    return "lower+upper rows (18 in-vessel coils)"


def coil_set_for_shot(shot: int, *, z_half: float = EFCC_Z_HALF) -> CoilSet:
    """Assemble the full era-resolved non-axisymmetric coil set for ``shot``."""
    return CoilSet(
        shot=int(shot),
        era=era_name(int(shot)),
        efcc_pairs=efcc_pairs(z_half=z_half),
        elm_coils=elm_coils_for_era(int(shot)),
    )


# --- Loop topology ---------------------------------------------------------


def loop_topology(channel: str) -> str:
    """Classify a magnetic sensor channel by its layer-B exposure.

    Returns ``"full_loop"`` (complete toroidal loop — immune to n != 0),
    ``"partial_loop"`` (toroidally-partial saddle — fully exposed), or
    ``"probe"`` (a point pickup — exposed, orientation-dependent).
    """
    c = channel.lower()
    if c.startswith(_PARTIAL_LOOP_PREFIXES):
        return "partial_loop"
    if c.startswith(_FULL_LOOP_PREFIXES):
        return "full_loop"
    return "probe"


def is_immune_to_energised_field(channel: str) -> bool:
    """True iff ``channel`` is a complete toroidal loop (rejects n != 0)."""
    return loop_topology(channel) == "full_loop"


__all__ = [
    "EFCC_RADIUS",
    "EFCC_SPAN_DEG",
    "EFCC_TURNS",
    "EFCC_MAX_KAT",
    "EFCC_Z_HALF",
    "EFCC_Z_HALF_RANGE",
    "EFCC_POLARITY",
    "EFCC_PAIRS",
    "ELM_LOWER_FIRST_SHOT",
    "ELM_UPPER_FIRST_SHOT",
    "ELM_MAX_KAT",
    "N_SECTORS",
    "CoilSpec",
    "CoilPair",
    "CoilSet",
    "efcc_pairs",
    "elm_coils_for_era",
    "era_name",
    "coil_set_for_shot",
    "loop_topology",
    "is_immune_to_energised_field",
]
