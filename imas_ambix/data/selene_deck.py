"""Convert the facility SELENE decks into per-phase IDS netCDF descriptions.

The JT-60SA machine description exists as SELENE input decks on the facility's
PSRC run tree: ``EQSLE.DATA`` (the master deck whose ``12 PF COIL`` block, ``NV``
vessel block, TFC arc segments and limiter/first-wall segments the equilibrium
code reads), ``geo.in`` (the ``getseldata`` geometry with its named sensor
blocks) and ``coil_vv_OP*.dat`` (the coil/vessel decks). This module parses
those grammars into provenance-carrying elements, writes the pf_active,
pf_passive, magnetics, wall and tf IDS netCDF for a phase at DD 4.1.1, and
records a ``receipt.json`` naming each source file's path, sha256 and line
ranges together with the converter's git commit.

Every parsed element carries a :class:`Provenance` naming the source path, the
file's sha256 and the 1-based inclusive line range it was also read from, so the
converted IDS can be re-derived from the deck it was built from.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

DD_VERSION = "4.1.1"
CONVERTER_MODULE = "imas_ambix/data/selene_deck.py"
MU0 = 4.0e-7 * np.pi
GEOMETRY_TYPE_RECTANGLE = 2


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Provenance:
    """The source file and the 1-based inclusive line range of an element."""

    source: str
    sha256: str
    line_start: int
    line_end: int

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }

    def line_range(self) -> list[int]:
        return [self.line_start, self.line_end]


def sha256_of(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def converter_git_commit() -> str:
    """The git commit of the tree the converter is imported from."""
    here = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return result.stdout.strip()


# --------------------------------------------------------------------------
# Parsed element types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PfElement:
    """One rectangular conductor element of a PF coil."""

    turns: float
    r: float
    z: float
    dr: float
    dz: float
    provenance: Provenance

    @property
    def width(self) -> float:
        return 2.0 * self.dr

    @property
    def height(self) -> float:
        return 2.0 * self.dz


@dataclass
class PfCoil:
    """A named PF coil (CS/EF/FPPC) and its deck elements."""

    name: str
    deck_label: str
    elements: list[PfElement] = field(default_factory=list)


@dataclass(frozen=True)
class VesselFilament:
    """One passive vessel filament carrying its resistivity."""

    turns: float
    r: float
    z: float
    dr: float
    dz: float
    resistivity: float
    provenance: Provenance


@dataclass(frozen=True)
class Segment:
    """A contour segment: a straight line or a circular arc."""

    kind: str  # "line" or "arc"
    params: tuple[float, ...]
    comment: str
    provenance: Provenance

    def line_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.kind != "line":
            raise ValueError("only line segments have endpoints")
        r1, z1, r2, z2 = self.params[:4]
        return (r1, z1), (r2, z2)


@dataclass(frozen=True)
class Sensor:
    """A poloidal probe (with angle) or a flux loop position."""

    r: float
    z: float
    angle: float | None
    provenance: Provenance


@dataclass
class EqSleDeck:
    """The master EQSLE.DATA deck, parsed."""

    path: str
    sha256: str
    pf_coils: list[PfCoil]
    vessel: list[VesselFilament]
    tfc_inside: list[Segment]
    tfc_outside: list[Segment]
    limiter_and_first_wall: list[Segment]


@dataclass
class GeoIn:
    """The getseldata geo.in sensor geometry, parsed."""

    path: str
    sha256: str
    probes: list[Sensor]
    flux_loops: list[Sensor]


@dataclass
class CoilVesselDeck:
    """A coil_vv deck, of which the vessel block is parsed."""

    path: str
    sha256: str
    vessel: list[VesselFilament]


# --------------------------------------------------------------------------
# Grammar helpers
# --------------------------------------------------------------------------
def _read_lines(path: Path) -> list[str]:
    # Decks are ASCII/Latin-1 Fortran text; keep the bytes that decode.
    return Path(path).read_text(encoding="latin-1").splitlines()


def _numeric_prefix(tokens: Sequence[str]) -> tuple[list[float], str]:
    """Split a deck row into its leading numeric fields and trailing comment."""
    nums: list[float] = []
    for idx, tok in enumerate(tokens):
        try:
            nums.append(float(tok))
        except ValueError:
            return nums, " ".join(tokens[idx:]).strip()
    return nums, ""


_PF_TO_NAME = {
    1: "CS1", 2: "CS2", 3: "CS3", 4: "CS4",
    5: "EF1", 6: "EF2", 7: "EF3", 8: "EF4", 9: "EF5", 10: "EF6",
}


def _canonical_coil_name(deck_label: str, position: int) -> str:
    """Map a deck label onto the machine's CS/EF/FPPC naming."""
    label = deck_label.strip()
    if label.lower().replace("_", "-") == "fppc-up":
        return "FPPC_UP"
    if label.lower().replace("_", "-") == "fppc-down":
        return "FPPC_DOWN"
    if label.startswith("PF") and label[2:].isdigit():
        return _PF_TO_NAME.get(int(label[2:]), label)
    if label:
        return label
    return _PF_TO_NAME.get(position + 1, f"PF{position + 1}")


def _parse_pf_coils(
    lines: list[str], header_idx: int, source: str, sha: str
) -> tuple[list[PfCoil], int]:
    counts = [int(t) for t in lines[header_idx + 1].split()]
    idx = header_idx + 2
    coils: list[PfCoil] = []
    for nelem in counts:
        elements: list[PfElement] = []
        deck_label = ""
        for k in range(nelem):
            line_no = idx + 1  # 1-based
            nums, comment = _numeric_prefix(lines[idx].split())
            turns, r, z, dr, dz = nums[:5]
            if k == 0 and comment:
                deck_label = comment.split()[0]
            elements.append(
                PfElement(
                    turns=abs(float(turns)),
                    r=float(r),
                    z=float(z),
                    dr=float(dr),
                    dz=float(dz),
                    provenance=Provenance(source, sha, line_no, line_no),
                )
            )
            idx += 1
        coils.append(
            PfCoil(
                name=_canonical_coil_name(deck_label, len(coils)),
                deck_label=deck_label,
                elements=elements,
            )
        )
    return coils, idx


def _parse_vessel(
    lines: list[str], header_idx: int, source: str, sha: str
) -> tuple[list[VesselFilament], int]:
    nv = int(lines[header_idx].split()[0])
    idx = header_idx + 1
    vessel: list[VesselFilament] = []
    for _ in range(nv):
        line_no = idx + 1
        nums, _ = _numeric_prefix(lines[idx].split())
        turns, r, z, dr, dz, rho = nums[:6]
        vessel.append(
            VesselFilament(
                turns=float(turns),
                r=float(r),
                z=float(z),
                dr=float(dr),
                dz=float(dz),
                resistivity=float(rho),
                provenance=Provenance(source, sha, line_no, line_no),
            )
        )
        idx += 1
    return vessel, idx


def _find_vessel_header(lines: list[str], start: int) -> int:
    for i in range(start, len(lines)):
        tokens = lines[i].split()
        if not tokens or not tokens[0].isdigit():
            continue
        if "NV" in " ".join(tokens[1:]):
            return i
    raise ValueError("no NV vessel block header found")


def _parse_segments(
    lines: list[str], start: int, source: str, sha: str
) -> tuple[list[Segment], list[Segment], list[Segment]]:
    """Parse the trailing contour table into wall, TFC inside and outside."""
    wall: list[Segment] = []
    tfc_inside: list[Segment] = []
    by_section = {"wall": wall, "inside": tfc_inside, "outside": []}
    section = "wall"
    for idx in range(start, len(lines)):
        line_no = idx + 1
        nums, comment = _numeric_prefix(lines[idx].split())
        if len(nums) < 5:
            continue
        kind_code = int(round(nums[-1]))
        if kind_code == 1:
            kind, params = "line", tuple(float(v) for v in nums[:4])
        elif kind_code == 2 and len(nums) >= 6:
            kind, params = "arc", tuple(float(v) for v in nums[:5])
        else:
            continue
        # A section label sits on the first row of its run; the rows after it
        # carry no comment and continue the section.
        if "TFC INSIDE" in comment:
            section = "inside"
        elif "TFC OUTSIDE" in comment:
            section = "outside"
        seg = Segment(
            kind=kind,
            params=params,
            comment=comment,
            provenance=Provenance(source, sha, line_no, line_no),
        )
        by_section[section].append(seg)
    return wall, tfc_inside, by_section["outside"]


def parse_eqsle_deck(path: Path | str) -> EqSleDeck:
    """Parse an EQSLE.DATA-grammar deck."""
    path = Path(path)
    lines = _read_lines(path)
    sha = sha256_of(path)
    source = str(path)
    pf_header = None
    for i, line in enumerate(lines):
        if "PF COIL" in line:
            pf_header = i
            break
    if pf_header is None:
        raise ValueError("no 'PF COIL' block in deck")
    coils, after_coils = _parse_pf_coils(lines, pf_header, source, sha)
    nv_header = _find_vessel_header(lines, after_coils)
    vessel, after_vessel = _parse_vessel(lines, nv_header, source, sha)
    wall, tfc_in, tfc_out = _parse_segments(lines, after_vessel, source, sha)
    return EqSleDeck(
        path=source,
        sha256=sha,
        pf_coils=coils,
        vessel=vessel,
        tfc_inside=tfc_in,
        tfc_outside=tfc_out,
        limiter_and_first_wall=wall,
    )


def _parse_named_blocks(
    lines: list[str],
) -> list[tuple[str, int, list[tuple[list[float], int]]]]:
    """Scan a geo.in-grammar file into (label, declared_count, rows) blocks."""
    blocks: list[tuple[str, int, list[tuple[list[float], int]]]] = []
    current: list[tuple[list[float], int]] | None = None
    for idx, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        line_no = idx + 1
        first = tokens[0].lstrip("+-")
        is_header = (
            first.isdigit()
            and len(tokens) >= 2
            and any(ch.isalpha() for ch in " ".join(tokens[1:]))
        )
        if is_header:
            label = " ".join(tokens[1:]).strip()
            current = []
            blocks.append((label, int(first), current))
            continue
        nums, _ = _numeric_prefix(tokens)
        if current is None or not nums:
            continue
        current.append((nums, line_no))
    return blocks


def parse_geo_in(path: Path | str) -> GeoIn:
    """Parse the Magnetic Probe and Flux Loop blocks of a geo.in file."""
    path = Path(path)
    lines = _read_lines(path)
    sha = sha256_of(path)
    source = str(path)
    blocks = _parse_named_blocks(lines)

    def _find(fragment: str) -> list[tuple[list[float], int]]:
        for label, count, rows in blocks:
            if fragment.lower() in label.lower():
                # The header declares the element count; rows after it with no
                # header of their own belong to a later, unnamed table.
                return rows[:count]
        raise ValueError(f"no block matching {fragment!r}")

    probes = [
        Sensor(
            r=float(nums[0]),
            z=float(nums[1]),
            angle=float(nums[2]) if len(nums) > 2 else None,
            provenance=Provenance(source, sha, line_no, line_no),
        )
        for nums, line_no in _find("Magnetic Probe")
    ]
    loops = [
        Sensor(
            r=float(nums[0]),
            z=float(nums[1]),
            angle=None,
            provenance=Provenance(source, sha, line_no, line_no),
        )
        for nums, line_no in _find("Flux Loop")
    ]
    return GeoIn(path=source, sha256=sha, probes=probes, flux_loops=loops)


def parse_coil_vv_deck(path: Path | str) -> CoilVesselDeck:
    """Parse the vessel block of a coil_vv-grammar deck."""
    path = Path(path)
    lines = _read_lines(path)
    sha = sha256_of(path)
    source = str(path)
    nv_header = _find_vessel_header(lines, 0)
    vessel, _ = _parse_vessel(lines, nv_header, source, sha)
    return CoilVesselDeck(path=source, sha256=sha, vessel=vessel)


def derive_b_field_phi_vacuum_r(current: float, turns: int, r0: float) -> float:
    """Toroidal vacuum field at the major radius from a declared relation.

    ``B_phi(R0) = mu0 * turns * current / (2 * pi * R0)``.
    """
    return MU0 * turns * current / (2.0 * np.pi * r0)


# --------------------------------------------------------------------------
# IDS construction
# --------------------------------------------------------------------------
def _static_header(ids) -> None:
    from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT

    ids.ids_properties.homogeneous_time = IDS_TIME_MODE_INDEPENDENT


def build_pf_active(factory, deck: EqSleDeck):
    ids = factory.new("pf_active")
    _static_header(ids)
    ids.coil.resize(len(deck.pf_coils))
    ids.circuit.resize(len(deck.pf_coils))
    for i, coil in enumerate(deck.pf_coils):
        ids.coil[i].name = coil.name
        ids.coil[i].element.resize(len(coil.elements))
        for j, el in enumerate(coil.elements):
            out = ids.coil[i].element[j]
            out.turns_with_sign = abs(float(el.turns))
            out.geometry.geometry_type = GEOMETRY_TYPE_RECTANGLE
            out.geometry.rectangle.r = float(el.r)
            out.geometry.rectangle.z = float(el.z)
            out.geometry.rectangle.width = float(el.width)
            out.geometry.rectangle.height = float(el.height)
        # Winding direction lives in the circuit, never in signed turns.
        ids.circuit[i].connections = np.array([[i + 1]], dtype=np.int32)
    return ids


def build_pf_passive(factory, vessel: Sequence[VesselFilament]):
    ids = factory.new("pf_passive")
    _static_header(ids)
    ids.loop.resize(1)
    ids.loop[0].element.resize(len(vessel))
    for j, fil in enumerate(vessel):
        out = ids.loop[0].element[j]
        out.turns_with_sign = abs(float(fil.turns))
        out.geometry.geometry_type = GEOMETRY_TYPE_RECTANGLE
        out.geometry.rectangle.r = float(fil.r)
        out.geometry.rectangle.z = float(fil.z)
        out.geometry.rectangle.width = 2.0 * float(fil.dr)
        out.geometry.rectangle.height = 2.0 * float(fil.dz)
    if vessel:
        ids.loop[0].resistivity = float(np.mean([f.resistivity for f in vessel]))
    return ids


def build_magnetics(factory, geo: GeoIn):
    ids = factory.new("magnetics")
    _static_header(ids)
    ids.b_field_pol_probe.resize(len(geo.probes))
    for i, probe in enumerate(geo.probes):
        out = ids.b_field_pol_probe[i]
        out.name = f"MP{i + 1}"
        out.position.r = float(probe.r)
        out.position.z = float(probe.z)
        if probe.angle is not None:
            # geo.in carries degrees; the DD leaf is in radians.
            out.poloidal_angle = float(np.deg2rad(probe.angle))
    ids.flux_loop.resize(len(geo.flux_loops))
    for i, loop in enumerate(geo.flux_loops):
        out = ids.flux_loop[i]
        out.name = f"FL{i + 1}"
        out.position.resize(1)
        out.position[0].r = float(loop.r)
        out.position[0].z = float(loop.z)
    return ids


def _wall_outline(wall: Sequence[Segment]) -> tuple[list[float], list[float]]:
    r_pts: list[float] = []
    z_pts: list[float] = []
    for seg in wall:
        if seg.kind != "line":
            continue
        (r1, z1), (r2, z2) = seg.line_points()
        if not r_pts or abs(r_pts[-1] - r1) > 1e-9 or abs(z_pts[-1] - z1) > 1e-9:
            r_pts.append(r1)
            z_pts.append(z1)
        r_pts.append(r2)
        z_pts.append(z2)
    return r_pts, z_pts


def build_wall(factory, wall: Sequence[Segment]):
    ids = factory.new("wall")
    _static_header(ids)
    r_pts, z_pts = _wall_outline(wall)
    ids.description_2d.resize(1)
    ids.description_2d[0].limiter.unit.resize(1)
    ids.description_2d[0].limiter.unit[0].outline.r = np.array(r_pts, dtype=float)
    ids.description_2d[0].limiter.unit[0].outline.z = np.array(z_pts, dtype=float)
    return ids


def _tf_section_points(segments: Sequence[Segment]):
    r_pts: list[float] = []
    z_pts: list[float] = []
    for seg in segments:
        if seg.kind == "line":
            (r1, z1), (r2, z2) = seg.line_points()
            r_pts.extend([r1, r2])
            z_pts.extend([z1, z2])
        else:
            r_pts.append(seg.params[0])
            z_pts.append(seg.params[1])
    return np.array(r_pts, dtype=float), np.array(z_pts, dtype=float)


def build_tf(
    factory,
    deck: EqSleDeck,
    *,
    r0: float = 2.96,
    coils_n: int = 18,
    turns: int = 0,
):
    ids = factory.new("tf")
    _static_header(ids)
    ids.r0 = float(r0)
    ids.coils_n = int(coils_n)
    ids.coil.resize(1)
    ids.coil[0].turns = int(turns)
    ids.coil[0].conductor.resize(2)
    for c, segments in enumerate((deck.tfc_inside, deck.tfc_outside)):
        r_pts, z_pts = _tf_section_points(segments)
        elements = ids.coil[0].conductor[c].elements
        for point in ("start_points", "end_points"):
            arr = getattr(elements, point)
            arr.r = r_pts
            arr.z = z_pts
            arr.phi = np.zeros_like(r_pts)
        elements.types = np.ones_like(r_pts, dtype=np.int32)
    return ids


# --------------------------------------------------------------------------
# Receipt and phase writer
# --------------------------------------------------------------------------
def _line_ranges(provenances: Sequence[Provenance]) -> list[list[int]]:
    ranges = sorted({(p.line_start, p.line_end) for p in provenances})
    return [[start, end] for start, end in ranges]


def _deck_provenances(deck: EqSleDeck) -> list[Provenance]:
    provs = [e.provenance for coil in deck.pf_coils for e in coil.elements]
    provs += [f.provenance for f in deck.vessel]
    provs += [s.provenance for s in deck.tfc_inside]
    provs += [s.provenance for s in deck.tfc_outside]
    provs += [s.provenance for s in deck.limiter_and_first_wall]
    return provs


def build_receipt(
    *,
    phase: str,
    eqsle: EqSleDeck,
    geo: GeoIn,
    coil_vv: CoilVesselDeck | None,
    outputs: dict[str, str],
    converter_commit: str,
) -> dict:
    """Compose the receipt naming every source's path, sha256 and line ranges."""
    sources = [
        {
            "role": "eqsle_deck",
            "path": eqsle.path,
            "sha256": eqsle.sha256,
            "line_ranges": _line_ranges(_deck_provenances(eqsle)),
            "blocks": {
                "PF COIL": _line_ranges(
                    [e.provenance for c in eqsle.pf_coils for e in c.elements]
                ),
                "NV": _line_ranges([f.provenance for f in eqsle.vessel]),
                "TFC": _line_ranges(
                    [s.provenance for s in eqsle.tfc_inside]
                    + [s.provenance for s in eqsle.tfc_outside]
                ),
                "wall": _line_ranges(
                    [s.provenance for s in eqsle.limiter_and_first_wall]
                ),
            },
        },
        {
            "role": "geo_in",
            "path": geo.path,
            "sha256": geo.sha256,
            "line_ranges": _line_ranges(
                [s.provenance for s in geo.probes + geo.flux_loops]
            ),
            "blocks": {
                "Magnetic Probe": _line_ranges([s.provenance for s in geo.probes]),
                "Flux Loop": _line_ranges([s.provenance for s in geo.flux_loops]),
            },
        },
    ]
    if coil_vv is not None:
        sources.append(
            {
                "role": "coil_vv_deck",
                "path": coil_vv.path,
                "sha256": coil_vv.sha256,
                "line_ranges": _line_ranges([f.provenance for f in coil_vv.vessel]),
                "blocks": {
                    "NV": _line_ranges([f.provenance for f in coil_vv.vessel])
                },
            }
        )
    return {
        "phase": phase,
        "dd_version": DD_VERSION,
        "converter": {
            "module": CONVERTER_MODULE,
            "git_commit": converter_commit,
        },
        "sources": sources,
        "outputs": outputs,
    }


def write_phase_description(
    *,
    phase: str,
    out_dir: Path | str,
    eqsle: EqSleDeck,
    geo: GeoIn,
    coil_vv: CoilVesselDeck | None = None,
    converter_commit: str | None = None,
    tf_current: float | None = None,
    r0: float = 2.96,
    coils_n: int = 18,
) -> dict:
    """Write the per-phase IDS netCDF and the receipt into ``out_dir/phase``."""
    import imas

    phase_dir = Path(out_dir) / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    factory = imas.IDSFactory(DD_VERSION)
    commit = converter_commit or converter_git_commit()

    vessel = coil_vv.vessel if coil_vv is not None else eqsle.vessel
    # The deck carries no explicit TF conductor count; the conductor geometry
    # is what it states, so the turn count is not inferred from it.
    tf_turns = 0

    ids_map = {
        "pf_active": build_pf_active(factory, eqsle),
        "pf_passive": build_pf_passive(factory, vessel),
        "magnetics": build_magnetics(factory, geo),
        "wall": build_wall(factory, eqsle.limiter_and_first_wall),
        "tf": build_tf(factory, eqsle, r0=r0, coils_n=coils_n, turns=tf_turns),
    }
    outputs: dict[str, str] = {}
    for name, ids in ids_map.items():
        path = phase_dir / f"{name}.nc"
        with imas.DBEntry(path, "w", dd_version=DD_VERSION) as entry:
            entry.put(ids)
        outputs[name] = str(path)

    receipt = build_receipt(
        phase=phase,
        eqsle=eqsle,
        geo=geo,
        coil_vv=coil_vv,
        outputs=outputs,
        converter_commit=commit,
    )
    if tf_current is not None:
        receipt["tf_b_field_phi_vacuum_r"] = derive_b_field_phi_vacuum_r(
            tf_current, coils_n, r0
        )
    receipt_path = phase_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    receipt["receipt_path"] = str(receipt_path)
    return receipt
