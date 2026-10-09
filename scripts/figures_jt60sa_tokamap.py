"""Render JT-60SA catalogue and signal rules beside hand-built tokamap entries."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pygments import lex
from pygments.lexers import JsonLexer
from pygments.token import Comment, Keyword, Name, Number, Punctuation, String, Token

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "maps/jt-60sa"
TOKAMAP = Path("/work/projects/imas_gpu/jt60sa/tokamap/hand-built")
SVG_DIR = ROOT / "docs/figures/jt60sa-tokamap"
PNG_DIR = Path("/work/projects/imas_gpu/jt60sa/presentation")
FRAGMENT = (
    ROOT / "docs/evidence/fragments/jt60sa-tokamap-from-generated-maps/"
    "tokamap-system-figures.html"
)
NODE = "tokamap-system-figures"


@dataclass(frozen=True)
class Example:
    label: str
    group: str
    key: str
    source_kind: str
    identifier: str


SYSTEMS: dict[str, tuple[str, tuple[Example, ...]]] = {
    "pickup-probes": (
        "Pickup probe 1",
        (
            Example(
                "Probe 1 field",
                "magnetics",
                "b_field_pol_probe[0]/field/data",
                "signal",
                "magPbTC1",
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
    "key": "#24547f",
    "string": "#69523b",
    "number": "#306a59",
    "keyword": "#763e79",
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


def _colour(token: Token) -> str:
    if token in Name.Tag or token in Name.Attribute:
        return COLORS["key"]
    if token in String:
        return COLORS["string"]
    if token in Number:
        return COLORS["number"]
    if token in Keyword:
        return COLORS["keyword"]
    if token in Comment:
        return COLORS["muted"]
    if token in Punctuation:
        return COLORS["plain"]
    return COLORS["plain"]


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
    fig.savefig(svg, format="svg", facecolor="white")
    fig.savefig(png, format="png", dpi=200, facecolor="white")
    plt.close(fig)
    return svg, png


def _render_json(
    slug: str, title: str, pairs: list[tuple[Example, str, str | None]]
) -> tuple[Path, Path]:
    heights = [
        max(len(left.splitlines()), len((right or "").splitlines()), 10) * 22 + 84
        for _, left, right in pairs
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
    for (example, left, right), block_height in zip(pairs, heights, strict=True):
        ax.text(28, y, example.label, fontsize=15, va="top", color=COLORS["plain"])
        ax.text(
            28,
            y - 27,
            "Catalogue / signal map",
            fontsize=12,
            va="top",
            color=COLORS["muted"],
        )
        ax.text(
            720,
            y - 27,
            "Tokamap mappings.json",
            fontsize=12,
            va="top",
            color=COLORS["muted"],
        )
        _code(ax, left, 28, y - 51)
        if right:
            _code(ax, right, 720, y - 51)
        else:
            ax.text(
                720,
                y - 51,
                "No hand-built tokamap entry for this target",
                fontsize=13,
                va="top",
                color=COLORS["muted"],
            )
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
    height = 690
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

    label(40, 630, "imas-codex discovery", "Facility signals → candidates")
    label(495, 630, "Generated mapping", "Draft target paths")
    label(975, 630, "Hand-off JSON", "Awaiting JT-60SA hand-off")
    _arrow(ax, (330, 605), (470, 605), True)
    _arrow(ax, (805, 605), (950, 605), True)
    label(975, 480, "imas-ambix import", "Catalogue identity → array index")
    _arrow(ax, (1090, 570), (1090, 520), True)
    label(40, 465, "Machine description", "maps/jt-60sa/machine_map.json")
    label(40, 335, "Hand-built signal maps", "maps/jt-60sa/maps/*.json")
    label(560, 335, "JT-60SA bundle", "Catalogue + indexed signal rules")
    _arrow(ax, (365, 430), (540, 340))
    _arrow(ax, (365, 315), (540, 315))
    _arrow(ax, (975, 440), (830, 350), True)
    label(990, 315, "Corpus validation", "Required for promotion")
    _arrow(ax, (1130, 415), (1130, 345), True)
    _arrow(ax, (980, 305), (830, 305), True)
    label(40, 175, "Tokamap export", "imas-ambix maps tokamap jt-60sa")
    label(730, 175, "Pulse IDS writer", "imas-alambic write")
    _arrow(ax, (610, 290), (300, 205))
    _arrow(ax, (740, 290), (920, 205))
    label(40, 65, "hand-built/", "/work/projects/imas_gpu/jt60sa/tokamap/")
    label(730, 65, "Pulse IDS + receipt", "~/public/imasdb/jt-60sa/")
    _arrow(ax, (235, 145), (235, 95))
    _arrow(ax, (925, 145), (925, 95))
    ax.text(
        425,
        540,
        "dashed: draft path awaiting hand-off",
        fontsize=12,
        color=COLORS["muted"],
    )
    return _save(fig, "chain")


def main() -> None:
    catalogue = json.loads((MAP / "machine_map.json").read_text())
    signals = {
        group: json.loads((MAP / "maps" / f"{group}.json").read_text())["signals"]
        for group in ("magnetics", "pf_active", "tf", "equilibrium")
    }
    sections: list[str] = []
    pngs: list[Path] = []
    gaps: list[str] = []
    for slug, (title, examples) in SYSTEMS.items():
        pairs = []
        detail = []
        for example in examples:
            source, source_path = _source(example, catalogue, signals)
            entry, entry_path = _entry(example)
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
            pairs.append((example, source, entry))
            if entry is None:
                gaps.append(f"{slug}: {example.key}")
            safe_label = html.escape(example.label)
            safe_source_path = html.escape(source_path)
            safe_entry_path = html.escape(entry_path)
            safe_key = html.escape(example.key)
            safe_source = html.escape(source)
            safe_entry = html.escape(
                entry or "No hand-built tokamap entry for this target"
            )
            detail.append(
                f'<h3 id="{NODE}-{slug}-{len(detail) + 1}">{safe_label}</h3>'
                f"<p>Source: <code>{safe_source_path}</code>; tokamap: "
                f"<code>{safe_entry_path}</code>, key <code>{safe_key}</code>.</p>"
                f"<p>Catalogue or signal map excerpt:</p><pre>{safe_source}</pre>"
                f"<p>Tokamap excerpt:</p><pre>{safe_entry}</pre>"
            )
        svg, png = _render_json(slug, title, pairs)
        pngs.append(png)
        gap_note = (
            " Missing hand-built entry is shown explicitly."
            if any(p[2] is None for p in pairs)
            else ""
        )
        image_source = f"/imas-ambix/figures/jt60sa-tokamap/{svg.name}"
        safe_title = html.escape(title)
        sections.append(
            f'<figure id="{NODE}-{slug}"><img src="{image_source}" '
            f'alt="{safe_title}: source JSON beside tokamap JSON">'
            f"<figcaption>{safe_title}. OP1 catalogue and shot 100001 tokamap; "
            "selected fields are copied verbatim from the source records."
            f"{gap_note}</figcaption></figure>" + "".join(detail)
        )
    svg, png = _render_chain()
    pngs.append(png)
    image_source = f"/imas-ambix/figures/jt60sa-tokamap/{svg.name}"
    sections.append(
        f'<figure id="{NODE}-chain"><img src="{image_source}" '
        'alt="JT-60SA discovery, catalogue, tokamap and pulse IDS chain">'
        "<figcaption>Solid arrows show the current hand-built path. "
        "Dashed arrows show the draft hand-off path awaiting generated "
        "JT-60SA mappings; corpus validation precedes promotion."
        "</figcaption></figure>"
    )
    png_list = "".join(
        f"<li><code>{html.escape(str(path))}</code> — 2800 × "
        f"{_png_height(path)} px, {path.stat().st_size:,} bytes</li>"
        for path in pngs
    )
    fragment = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="docs-project" content="imas-ambix">'
        '<meta name="reckon-type" content="evidence">'
        '<meta name="plan-slug" content="jt60sa-tokamap-from-generated-maps">'
        '</head><body><main class="plan-doc">'
        f'<section id="{NODE}-landing"><h2>JT-60SA system map renders</h2>'
        "<p>These figures compare selected fields from the hand-built catalogue "
        "or signal map with the corresponding tokamap entry. Each selected "
        "value is copied verbatim into "
        "both the image and selectable excerpt.</p>"
        + "".join(sections)
        + f'<h3 id="{NODE}-pngs">Presentation PNGs</h3><ul>{png_list}</ul>'
        + (f"<p>Hand-built gaps: {html.escape(', '.join(gaps))}.</p>" if gaps else "")
        + "</section></main></body></html>\n"
    )
    FRAGMENT.parent.mkdir(parents=True, exist_ok=True)
    FRAGMENT.write_text(fragment)
    print(f"rendered {len(SYSTEMS)} JSON figures and chain")
    print(f"PNGs={len(pngs)} gaps={len(gaps)}")
    for path in pngs:
        print(f"{path} {path.stat().st_size} bytes")


def _png_height(path: Path) -> int:
    from matplotlib.image import imread

    return int(imread(path).shape[0])


if __name__ == "__main__":
    main()
