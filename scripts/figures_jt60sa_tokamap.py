"""Render JT-60SA catalogue and signal rules beside hand-built tokamap entries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import imas
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
DRAFT_ROOT = Path("/work/projects/imas_gpu/jt60sa/tokamap")
DRAFT_TOKAMAP = DRAFT_ROOT / "draft-fixture"
HANDOFF = ROOT / "tests/fixtures/mapping_handoff_example.json"
LIVE_HANDOFF = Path(
    "/work/projects/imas_gpu/jt60sa/handoff/jt-60sa-mapping-handoff.json"
)
SVG_DIR = ROOT / "docs/figures/jt60sa-tokamap"
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
    "red": "#a62f35",
}


@dataclass(frozen=True)
class Target:
    path: str
    dd_version: str
    units: str
    data_type: str
    coordinates: str
    documentation: str
    dd_label: str | None
    rule_label: str
    history: str = ""

    @property
    def mismatch(self) -> bool:
        return self.dd_label is not None and self.rule_label != self.dd_label

    @property
    def dd_display(self) -> str:
        return self.dd_label or f"none at DD {self.dd_version}"


def _dd_label(metadata, relative_path: str) -> str | None:
    components = relative_path.split("/")
    for length in range(len(components), 0, -1):
        label = getattr(
            metadata["/".join(components[:length])], "cocos_label_transformation", None
        )
        if label:
            return str(label)
    return None


def _target(
    example: Example,
    source: str,
    factories: dict[str, imas.IDSFactory],
    dd_version: str,
) -> Target:
    rule = json.loads(source)
    relative = re.sub(r"\[(?:#|\d+)\]", "", example.key)
    metadata = factories[dd_version].new(example.group).metadata
    node = metadata[relative]
    label = _dd_label(metadata, relative)
    rule_label = rule.get(
        "transformation", f"sign convention {rule.get('sign_convention', 'none')}"
    )
    history = ""
    if example.group == "magnetics" and relative == "flux_loop/flux/data":
        earlier = [
            _dd_label(factories[v].new(example.group).metadata, relative)
            for v in ("3.28.1", "3.42.0")
        ]
        removed = _dd_label(factories["4.0.0"].new(example.group).metadata, relative)
        if (
            earlier != ["psi_like", "psi_like"]
            or removed is not None
            or label is not None
        ):
            raise ValueError(
                f"unexpected flux-loop COCOS history: {earlier}, {removed}, {label}"
            )
        history = "psi_like in DD 3.28.1–3.42, removed in 4.0.0"
    documentation = str(node.documentation or "").split(".", 1)[0].strip()
    coordinates = ", ".join(str(value) for value in node.coordinates) or "none"
    return Target(
        f"{example.group}/{node.path_string}",
        dd_version,
        str(node.units or "none"),
        str(node.data_type.value),
        coordinates,
        documentation,
        label,
        str(rule_label),
        history,
    )


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


def _save(fig, name: str) -> Path:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    svg = SVG_DIR / f"{name}.svg"
    fig.savefig(svg, format="svg", facecolor="white", metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    return svg


def _render_json(
    slug: str,
    title: str,
    pairs: list[tuple[Example, str, str | None, str | None, Target]],
) -> Path:
    heights = [
        max(len(left.splitlines()), len((right or "").splitlines()), 10) * 22
        + (len(draft.splitlines()) * 22 + 55 if draft else 0)
        + 214
        for _, left, right, draft, _ in pairs
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
    for (example, left, right, draft, target), block_height in zip(
        pairs, heights, strict=True
    ):
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
        panel_top = y - block_height + 154
        ax.text(
            28,
            panel_top,
            f"Data Dictionary target · DD {target.dd_version}",
            fontsize=15,
            weight="bold",
            va="top",
            color=COLORS["plain"],
        )
        ax.text(
            28,
            panel_top - 28,
            target.path,
            fontsize=13,
            va="top",
            color=COLORS["plain"],
            family="DejaVu Sans Mono",
        )
        ax.text(
            28,
            panel_top - 53,
            f"units: {target.units}    type: {target.data_type}"
            f"    coordinates: {target.coordinates}",
            fontsize=12,
            va="top",
            color=COLORS["muted"],
        )
        ax.text(
            28,
            panel_top - 77,
            f"documentation: {target.documentation}",
            fontsize=12,
            va="top",
            color=COLORS["muted"],
        )
        dd_text = target.dd_display
        label_colour = COLORS["red"] if target.mismatch else COLORS["plain"]
        ax.text(
            28,
            panel_top - 101,
            f"DD COCOS: {dd_text}    rule: {target.rule_label}",
            fontsize=12,
            va="top",
            color=label_colour,
        )
        if target.history:
            ax.text(
                28,
                panel_top - 125,
                target.history,
                fontsize=12,
                va="top",
                color=COLORS["muted"],
            )
        elif target.mismatch:
            ax.text(
                28,
                panel_top - 125,
                f"Mismatch: DD {target.dd_label}; rule {target.rule_label}",
                fontsize=12,
                va="top",
                color=COLORS["red"],
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


def _render_chain() -> Path:
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


def _render_writer() -> Path:
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


def _statistics(
    catalogue: dict, factories: dict[str, imas.IDSFactory]
) -> tuple[list[dict], str]:
    partition = "101174"
    draft = DRAFT_ROOT / "draft"
    if not draft.is_dir():
        draft = DRAFT_TOKAMAP
    handoff_path = LIVE_HANDOFF if draft.name == "draft" else HANDOFF
    handoff = json.loads(handoff_path.read_text())
    handoff_groups = {item["ids_name"]: item for item in handoff["ids"]}
    input_rows = sum(
        len(item["signals"]) + len(item["unexpanded"]) for item in handoff["ids"]
    )
    rows = []
    for group_dir in sorted(TOKAMAP.iterdir()):
        hand_path = group_dir / partition / "mappings.json"
        if not hand_path.is_file():
            continue
        group = group_dir.name
        hand = json.loads(hand_path.read_text())
        draft_path = draft / group / partition / "mappings.json"
        if (
            draft.name == "draft"
            and draft_path.stat().st_mtime_ns < handoff_path.stat().st_mtime_ns
        ):
            raise ValueError(
                f"{draft_path} predates {handoff_path}; regenerate the draft"
            )
        drafted = json.loads(draft_path.read_text())

        def machine(item: dict) -> bool:
            return str(item.get("data_source", "")).startswith("file://")

        description = sum(machine(item) for item in hand.values())
        signals = len(hand) - description
        draft_signals = sum(not machine(item) for item in drafted.values())
        # The partition's mapping comments identify the catalogue set used for it.
        binding_sets = catalogue["binding_sets"]
        chosen = max(
            binding_sets,
            key=lambda item: sum(
                binding["name"] in str(entry.get("comment", ""))
                for binding in item["bindings"]
                for entry in hand.values()
            ),
        )
        metadata = factories[catalogue["dd_version"]].new(group).metadata
        withheld = sum(
            bool(_dd_label(metadata, binding["dd_path"].split("/", 1)[1]))
            and catalogue["source_cocos"] == 0
            and not any(
                binding["name"] in str(entry.get("comment", ""))
                for entry in hand.values()
            )
            for binding in chosen["bindings"]
            if binding["dd_path"].startswith(group + "/")
        )
        handoff_group = handoff_groups.get(group, {})
        input_group_rows = len(handoff_group.get("signals", [])) + len(
            handoff_group.get("unexpanded", [])
        )
        generated = [
            item
            for item in drafted.values()
            if "validation_state=draft" in str(item.get("comment", ""))
        ]
        signal_ids = {item["signal_id"] for item in handoff_group.get("signals", [])}
        for item in generated:
            identity = re.search(r"semantic_id=([^;]+)", item["comment"])
            if identity is None or identity.group(1) not in signal_ids:
                raise ValueError(f"{group}: draft entry is absent from the hand-off")
        if len(generated) > input_group_rows:
            raise ValueError(f"{group}: draft entries exceed hand-off rows")
        unresolved = input_group_rows - len(generated)
        rows.append(
            dict(
                group=group,
                description=description,
                signals=signals,
                draft_signals=draft_signals,
                withheld=withheld,
                unresolved=unresolved,
            )
        )
    if not rows or sum(row["description"] + row["signals"] for row in rows) == 0:
        raise ValueError("hand-built partition is empty; cannot interpret zero counts")
    if input_rows == 0:
        raise ValueError(
            "hand-off has no signal rows; cannot interpret unresolved zero"
        )
    return rows, draft.name


def _render_statistics(rows: list[dict], draft_name: str) -> Path:
    kinds = (
        ("description", "machine description", "#4f6982"),
        ("signals", "hand-built signals", "#558a78"),
        ("draft_signals", f"{draft_name} export signals", "#9a6c8c"),
        ("withheld", "COCOS withheld", "#a67849"),
        ("unresolved", "unresolved hand-off", "#a62f35"),
    )
    fig, ax = plt.subplots(figsize=(14, 11), dpi=100)
    fig.subplots_adjust(left=0.17, right=0.72, top=0.96, bottom=0.07)
    positions = []
    for group_index, row in enumerate(rows):
        for kind_index, (key, label, colour) in enumerate(kinds):
            y = group_index * 6 + kind_index
            count = row[key]
            ax.barh(y, count, height=0.68, color=colour)
            ax.text(
                count + 0.6,
                y,
                f"{count}  {label}",
                va="center",
                color=colour,
                fontsize=15,
            )
        positions.append(group_index * 6 + 2)
    ax.set_yticks(positions, [row["group"] for row in rows], fontsize=17)
    ax.set_xlim(0, max(row[key] for row in rows for key, _, _ in kinds) * 1.85 + 2)
    ax.invert_yaxis()
    ax.set_xlabel("entries or rows [count]", fontsize=20)
    ax.tick_params(axis="x", labelsize=16, width=1.2)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(False)
    return _save(fig, "ids-statistics")


def main() -> None:
    catalogue = json.loads((MAP / "machine_map.json").read_text())
    handoff = json.loads(HANDOFF.read_text())
    factories = {
        version: imas.IDSFactory(version)
        for version in (catalogue["dd_version"], "3.28.1", "3.42.0", "4.0.0")
    }
    signals = {
        group: json.loads((MAP / "maps" / f"{group}.json").read_text())["signals"]
        for group in ("magnetics", "pf_active", "tf", "equilibrium")
    }
    svgs: list[Path] = []
    targets: list[tuple[str, Target]] = []
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
            target = _target(example, source, factories, catalogue["dd_version"])
            targets.append((slug, target))
            pairs.append((example, source, entry, draft, target))
            if entry is None:
                gaps.append(f"{slug}: {example.key}")
            if draft is not None:
                draft_count += 1
                print(f"fixture-driven {slug}: {source_path}")
                print(f"  hand-built: {entry_path}")
                print(f"  indexed draft: {draft_path}")
        svgs.append(_render_json(slug, title, pairs))
    svgs.extend((_render_chain(), _render_writer()))
    rows, draft_name = _statistics(catalogue, factories)
    svgs.append(_render_statistics(rows, draft_name))
    print(
        f"rendered {len(SYSTEMS)} system figures, 2 flow charts and 1 statistics figure"
    )
    print(f"SVGs={len(svgs)} gaps={len(gaps)} fixture_draft_rows={draft_count}")
    print(f"partition=101174 draft_source={draft_name}")
    for row in rows:
        print(
            "stats "
            + row["group"]
            + " "
            + " ".join(
                f"{key}={row[key]}"
                for key in (
                    "description",
                    "signals",
                    "draft_signals",
                    "withheld",
                    "unresolved",
                )
            )
        )
    for slug, target in targets:
        print(
            f"target {slug}: {target.path} "
            f"DD={target.dd_display} rule={target.rule_label}"
        )
    for path in svgs:
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
