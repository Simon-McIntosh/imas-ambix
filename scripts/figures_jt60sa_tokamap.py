"""Render JT-60SA catalogue and signal rules beside hand-built tokamap entries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pygments import lex
from pygments.lexers import JsonLexer
from pygments.styles import get_style_by_name

if TYPE_CHECKING:
    from pygments.token import Token

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "maps/jt-60sa"
TOKAMAP = Path("/work/projects/imas_gpu/jt60sa/tokamap/hand-built")
DRAFT_TOKAMAP = Path("/work/projects/imas_gpu/jt60sa/tokamap/draft-fixture")
HANDOFF = ROOT / "tests/fixtures/mapping_handoff_example.json"
SVG_DIR = ROOT / "docs/figures/jt60sa-tokamap"
PNG_DIR = Path("/work/projects/imas_gpu/jt60sa/presentation")
matplotlib.rcParams["svg.hashsalt"] = "jt60sa-tokamap"
FRIENDLY = get_style_by_name("friendly")


@dataclass(frozen=True)
class Example:
    label: str
    group: str
    key: str
    source_kind: str
    identifier: str


SYSTEMS: dict[str, tuple[str, tuple[Example, ...]]] = {
    "pickup-probes": (
        "Pickup probe 10",
        (
            Example(
                "Probe 10 field",
                "magnetics",
                "b_field_pol_probe[9]/field/data",
                "signal",
                "magPbTC10",
            ),
        ),
    ),
    "flux-loops": (
        "Flux loops: FL7 absolute and FL1 differential",
        (
            Example(
                "FL7 absolute",
                "magnetics",
                "flux_loop[6]/flux/data",
                "signal",
                "magFlxLp7",
            ),
            Example(
                "FL1 differential",
                "magnetics",
                "flux_loop[27]/flux/data",
                "signal",
                "magFlxLp1",
            ),
        ),
    ),
    "pf-coils": (
        "PF coil current and CS1 geometry",
        (
            Example(
                "CS1 current",
                "pf_active",
                "coil[0]/current/data",
                "signal",
                "curCS1LKAT",
            ),
            Example(
                "CS1 rectangle radius",
                "pf_active",
                "coil[#]/element[#]/geometry/rectangle/r",
                "binding",
                "jt60sa-op1-pf-active-cs1-r",
            ),
        ),
    ),
    "passive-structure": (
        "Passive vessel rectangle",
        (
            Example(
                "Vessel rectangle radius",
                "pf_passive",
                "loop[#]/element[#]/geometry/rectangle/r",
                "binding",
                "jt60sa-op1-pf-passive-vv-r",
            ),
        ),
    ),
    "tf": (
        "Toroidal field coil",
        (Example("TF current", "tf", "coil[0]/current/data", "signal", "cur1TFLKAT"),),
    ),
    "wall": (
        "Wall limiter outline",
        (
            Example(
                "Limiter radius",
                "wall",
                "description_2d[#]/limiter/unit[#]/outline/r",
                "binding",
                "jt60sa-op1-wall-limiter-outline-r",
            ),
        ),
    ),
    "equilibrium": (
        "Equilibrium safety factor",
        (
            Example(
                "Axis safety factor",
                "equilibrium",
                "time_slice[#]/global_quantities/q_axis",
                "signal",
                "QAXIS",
            ),
        ),
    ),
}

SOURCE_FIELDS = {
    "signal": (
        "semantic_id",
        "source_group",
        "source_array",
        "target_path",
        "target_index",
        "source_unit",
        "target_unit",
        "transformation",
        "channel_factor",
        "validation_state",
    ),
    "binding": (
        "name",
        "source_group",
        "source_array",
        "source_role",
        "dd_path",
        "struct_array_entry",
        "source_unit",
        "target_unit",
        "sign_convention",
    ),
}

COLORS = {
    "plain": "#262b32",
    "muted": "#626a73",
}


def _json_excerpt(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _source(
    example: Example, catalogue: dict, signals: dict[str, list[dict]]
) -> tuple[str, str]:
    if example.source_kind == "signal":
        candidates = [
            item
            for item in signals[example.group]
            if item["source_array"] == example.identifier
        ]
        source_path = MAP / "maps" / f"{example.group}.json"
    else:
        candidates = [
            item
            for item in catalogue["binding_sets"][0]["bindings"]
            if item["name"] == example.identifier
        ]
        source_path = MAP / "machine_map.json"
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {example.identifier} in {source_path}, got {len(candidates)}"
        )
    selected = {
        key: candidates[0][key]
        for key in SOURCE_FIELDS[example.source_kind]
        if key in candidates[0]
    }
    return _json_excerpt(selected), str(source_path.relative_to(ROOT))


def _entry(example: Example) -> tuple[str | None, str]:
    path = TOKAMAP / example.group / "100001" / "mappings.json"
    entries = json.loads(path.read_text())
    entry = entries.get(example.key)
    if entry is None:
        return None, str(path)
    fields = ("map_type", "args", "scale")
    if example.source_kind == "signal":
        fields = ("map_type", "data_source", "args", "scale")
    selected = {example.key: {key: entry[key] for key in fields if key in entry}}
    return _json_excerpt(selected), str(path)


def _draft(example: Example, handoff: dict) -> tuple[str | None, str | None]:
    if example.source_kind != "signal":
        return None, None
    rows = [
        row
        for ids in handoff["ids"]
        if ids["ids_name"] == example.group
        for row in ids["signals"]
        if row["source_array"] == example.identifier
    ]
    if not rows:
        return None, None
    if len(rows) != 1:
        raise ValueError(f"expected one hand-off row for {example.identifier}")
    path = DRAFT_TOKAMAP / example.group / "100001" / "mappings.json"
    indexed = json.loads(path.read_text()).get(example.key)
    if indexed is None or indexed["args"]["source_array"] != example.identifier:
        raise ValueError(
            f"draft export does not index {example.identifier} at {example.key}"
        )
    if "validation_state=draft" not in indexed["comment"]:
        raise ValueError(f"draft export lacks draft state for {example.identifier}")
    row = rows[0]
    excerpt = {
        "source_group": row["source_group"],
        "source_array": row["source_array"],
        "member_identifier": row["member_identifier"],
        "target_path": row["target_path"],
        "source_units": row["source_units"],
        "target_units": row["target_units"],
        "confidence": row["confidence"],
        "indexed_target": example.key,
        "validation_state": "draft",
        "scale": indexed["scale"],
    }
    return _json_excerpt(excerpt), str(path)


def _colour(token: Token) -> str:
    colour = FRIENDLY.style_for_token(token)["color"]
    return f"#{colour}" if colour else COLORS["plain"]


def _code(ax, code: str, x: float, y: float) -> None:
    row = 0
    col = 0
    for token, value in lex(code, JsonLexer()):
        colour = _colour(token)
        chunks = value.split("\n")
        for index, chunk in enumerate(chunks):
            if chunk:
                ax.text(
                    x + 10.4 * col,
                    y - 22.0 * row,
                    chunk,
                    color=colour,
                    family="DejaVu Sans Mono",
                    fontsize=12.5,
                    va="top",
                )
                col += len(chunk)
            if index < len(chunks) - 1:
                row += 1
                col = 0


def _save(fig, name: str) -> tuple[Path, Path]:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    svg = SVG_DIR / f"{name}.svg"
    png = PNG_DIR / f"{name}.png"
    fig.savefig(svg, format="svg", facecolor="white", metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(png, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    return svg, png


def _render_json(
    slug: str, title: str, pairs: list[tuple[Example, str, str | None, str | None]]
) -> tuple[Path, Path]:
    heights = [
        max(len(left.splitlines()), len((right or "").splitlines()), 10) * 22
        + (len(draft.splitlines()) * 22 + 55 if draft else 0)
        + 84
        for _, left, right, draft in pairs
    ]
    height = sum(heights) + 35
    fig = plt.figure(figsize=(14, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(
        28,
        height - 18,
        title,
        fontsize=20,
        weight="bold",
        va="top",
        color=COLORS["plain"],
    )
    y = height - 62
    for (example, left, right, draft), block_height in zip(pairs, heights, strict=True):
        source_x, entry_x = 28, 720
        ax.text(28, y, example.label, fontsize=15, va="top", color=COLORS["plain"])
        ax.text(
            source_x,
            y - 27,
            "Catalogue / signal map",
            fontsize=12,
            va="top",
            color=COLORS["muted"],
        )
        ax.text(
            entry_x,
            y - 27,
            "Tokamap mappings.json",
            fontsize=12,
            va="top",
            color=COLORS["muted"],
        )
        _code(ax, left, source_x, y - 51)
        if right:
            _code(ax, right, entry_x, y - 51)
        else:
            ax.text(
                entry_x,
                y - 51,
                "No hand-built tokamap entry for this target",
                fontsize=13,
                va="top",
                color=COLORS["muted"],
            )
        if draft:
            draft_top = (
                y
                - max(len(left.splitlines()), len((right or "").splitlines()), 10) * 22
                - 45
            )
            ax.text(
                28,
                draft_top,
                "Fixture-driven hand-off + indexed draft",
                fontsize=12,
                va="top",
                color=COLORS["muted"],
            )
            _code(ax, draft, 28, draft_top - 24)
        y -= block_height
    return _save(fig, slug)


def _arrow(
    ax, start: tuple[int, int], end: tuple[int, int], dashed: bool = False
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=2,
            linestyle="--" if dashed else "-",
            color=COLORS["plain"],
        )
    )


def _render_chain() -> tuple[Path, Path]:
    height = 740
    fig = plt.figure(figsize=(14, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, height)
    ax.axis("off")

    def label(x: int, y: int, heading: str, detail: str, colour: str = "plain") -> None:
        ax.text(
            x, y, heading, fontsize=16, weight="bold", va="top", color=COLORS[colour]
        )
        ax.text(x, y - 30, detail, fontsize=12, va="top", color=COLORS["muted"])

    label(30, 675, "imas-codex discovery", "facility signals")
    label(380, 675, "Candidates", "source ↔ DD path")
    label(720, 675, "Generated mapping", "status=generated")
    label(1070, 675, "Hand-off JSON", "fixture-driven input")
    ax.text(
        1070,
        620,
        "tests/fixtures/mapping_handoff_example.json",
        fontsize=10,
        color=COLORS["muted"],
    )
    _arrow(ax, (300, 635), (355, 635), True)
    _arrow(ax, (645, 635), (695, 635), True)
    _arrow(ax, (995, 635), (1045, 635), True)
    label(30, 465, "Machine description", "maps/jt-60sa/machine_map.json")
    label(500, 465, "imas-ambix import", "member identity → DD index")
    label(1020, 465, "JT-60SA bundle", "maps/jt-60sa/bundle.json")
    _arrow(ax, (365, 425), (470, 425))
    _arrow(ax, (840, 425), (990, 425), True)
    _arrow(ax, (1200, 605), (1200, 520), True)
    _arrow(ax, (1200, 520), (715, 520), True)
    _arrow(ax, (715, 520), (715, 485), True)
    label(30, 285, "Tokamap export", "imas-ambix maps tokamap jt-60sa")
    label(625, 285, "Pulse IDS writer", "imas-alambic write E101154")
    label(1060, 285, "Validation", "required before promotion")
    _arrow(ax, (1090, 395), (1090, 320))
    _arrow(ax, (1020, 400), (300, 320), True)
    _arrow(ax, (1125, 400), (820, 320))
    label(
        30,
        105,
        "Draft tokamap export",
        "/work/projects/imas_gpu/jt60sa/tokamap/draft-fixture/",
    )
    label(625, 105, "Pulse IDS + WriteReceipt", "~/public/imasdb/jt-60sa/")
    _arrow(ax, (240, 245), (240, 145), True)
    _arrow(ax, (840, 245), (840, 145))
    ax.text(
        30,
        550,
        "Dashed = generated draft; solid = catalogue or writer path",
        fontsize=12,
        color=COLORS["muted"],
    )
    return _save(fig, "chain")


def _render_writer() -> tuple[Path, Path]:
    height = 660
    fig = plt.figure(figsize=(14, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, height)
    ax.axis("off")

    def label(x: int, y: int, heading: str, detail: str) -> None:
        ax.text(
            x, y, heading, fontsize=17, weight="bold", va="top", color=COLORS["plain"]
        )
        ax.text(x, y - 30, detail, fontsize=12, va="top", color=COLORS["muted"])

    label(30, 610, "Phase description", "maps/jt-60sa/machine_description/OP1/*.nc")
    label(505, 610, "Bundle signal maps", "maps/jt-60sa/maps/*.json")
    label(1000, 610, "EDDB channels", "VirtualZarrView + read_channel")
    _arrow(ax, (280, 565), (280, 400))
    _arrow(ax, (680, 565), (680, 400))
    _arrow(ax, (1190, 565), (1190, 400))
    ax.text(
        30,
        355,
        "imas-alambic write E101154 --machine jt-60sa --maps maps/jt-60sa",
        fontsize=18,
        family="DejaVu Sans Mono",
        color=COLORS["plain"],
    )
    ax.text(
        30,
        315,
        "--out ~/public/imasdb/jt-60sa/",
        fontsize=18,
        family="DejaVu Sans Mono",
        color=COLORS["plain"],
    )
    ax.text(
        30,
        260,
        "Settings precedence: CLI flag → IMAS_ALAMBIC_* → home / bundle role → default",
        fontsize=14,
        color=COLORS["muted"],
    )
    _arrow(ax, (720, 235), (720, 175))
    label(30, 140, "pulse_path", "~/public/imasdb/jt-60sa/101154_0.nc")
    label(700, 140, "WriteReceipt", "IDS names, written leaves, exclusions")
    label(30, 65, "Alternative IDS root", "/work/projects/imas_gpu/jt60sa/ids/")
    label(700, 65, "Legacy shared description", "~/public/imasdb/jt-60sa_md/*.h5")
    return _save(fig, "writer-workflow")


def main() -> None:
    catalogue = json.loads((MAP / "machine_map.json").read_text())
    handoff = json.loads(HANDOFF.read_text())
    signals = {
        group: json.loads((MAP / "maps" / f"{group}.json").read_text())["signals"]
        for group in ("magnetics", "pf_active", "tf", "equilibrium")
    }
    pngs: list[Path] = []
    gaps: list[str] = []
    draft_count = 0
    for slug, (title, examples) in SYSTEMS.items():
        pairs = []
        for example in examples:
            source, source_path = _source(example, catalogue, signals)
            entry, entry_path = _entry(example)
            draft, draft_path = _draft(example, handoff)
            if entry is not None:
                source_record = json.loads(source)
                target_record = json.loads(entry)[example.key]
                for field in ("source_group", "source_array"):
                    if target_record["args"][field] != source_record[field]:
                        raise ValueError(
                            f"{example.key}: {field} differs from {source_path}"
                        )
                index = source_record.get("target_index")
                if index is not None and f"[{index}]" not in example.key:
                    raise ValueError(f"{example.key}: target index {index} is absent")
            pairs.append((example, source, entry, draft))
            if entry is None:
                gaps.append(f"{slug}: {example.key}")
            if draft is not None:
                draft_count += 1
                print(f"fixture-driven {slug}: {source_path}")
                print(f"  hand-built: {entry_path}")
                print(f"  indexed draft: {draft_path}")
        svg, png = _render_json(slug, title, pairs)
        pngs.append(png)
    svg, png = _render_chain()
    pngs.append(png)
    svg, png = _render_writer()
    pngs.append(png)
    print(f"rendered {len(SYSTEMS)} JSON figures and 2 flow charts")
    print(f"PNGs={len(pngs)} gaps={len(gaps)} draft_rows={draft_count}")
    for path in pngs:
        print(f"{path} 2800x{_png_height(path)} {path.stat().st_size} bytes")


def _png_height(path: Path) -> int:
    from matplotlib.image import imread

    return int(imread(path).shape[0])


if __name__ == "__main__":
    main()
