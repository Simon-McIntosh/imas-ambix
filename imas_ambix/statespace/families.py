"""Diagnostic-family classification and leakage audit.

Defines the physical-quantity family taxonomy for MAST level-1 data
and provides empirical leakage-audit tooling that enumerates EVERY
channel across ALL groups that measures or is derived from a candidate
held-out physical quantity.

The leakage audit is the hard precondition for the orchestrator's target
lock. A held-out FAMILY must be defined by PHYSICAL QUANTITY, not by
Zarr group name — otherwise within-family channels in other groups act
as a shortcut and invalidate the cross-family evaluation.

Usage
-----
    from imas_ambix.statespace.families import (
        FAMILY_GROUPS,
        LEAKAGE_MAP,
        classify_group,
        build_leakage_audit,
    )

    # Enumerate all channels that leak Dα across ALL groups:
    leaking = LEAKAGE_MAP["dalpha"]   # pre-built from empirical scan
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from imas_ambix.statespace.inventory import InventoryResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Family taxonomy
# ---------------------------------------------------------------------------

# Canonical physical-quantity families (input + candidate-target families).
# Each family maps to the primary Zarr groups that carry it.
# This is the "intended" mapping — the leakage audit below finds ALL groups.
FAMILY_GROUPS: dict[str, list[str]] = {
    "magnetics": ["ama", "amb", "amc", "amh", "amm", "asm"],
    "magnetics_raw": ["xma", "xmb", "xmc", "xmo"],
    "dalpha": ["xim", "ada", "aim"],  # primary groups — leakage audit expands this
    "interferometer": ["ane"],
    "thomson_scattering": ["atm", "ayc", "aye"],
    "bolometer": ["abm"],
    "soft_xray": ["xsx"],
    "hard_xray": ["ahx"],
    "mse": ["ams"],
    "charge_exchange": ["act"],
    "langmuir": ["alp"],
    "neutron": ["anu"],
    "camera_visible": ["rba", "rbb", "rbc", "rco", "rgb", "rgc", "rca"],
    "camera_ir": ["rir", "rit", "air", "ait"],
    "nbi": ["anb"],
    "gas": ["aga"],
    # Analysis/derived — not input families but inventoried
    "dalpha_analysis": ["ada", "aim"],  # overlaps with dalpha (leakage)
    "ir_analysis": ["air"],
    "soft_xray_sawtooth": ["asx"],
    "equilibrium_sawtooth": ["esx"],
    "density_gradient": ["adg"],
    "microwave_reflectometry": ["aoe"],
    # Excluded families (solvers + control)
    "equilibrium_efit": ["efm"],
    "equilibrium_solovev": ["esm"],
    "pulse_schedule": ["xdc"],
}

# Inverse map: group → primary family
_GROUP_TO_FAMILY: dict[str, str] = {}
for _fam, _groups in FAMILY_GROUPS.items():
    for _g in _groups:
        if _g not in _GROUP_TO_FAMILY:
            _GROUP_TO_FAMILY[_g] = _fam


def classify_group(group: str) -> str:
    """Return the primary family name for a Zarr group name.

    Returns 'unknown' for groups not in the taxonomy.
    """
    return _GROUP_TO_FAMILY.get(group, "unknown")


# ---------------------------------------------------------------------------
# Dα leakage patterns (empirical, verified against real shots)
# ---------------------------------------------------------------------------

# Channel name patterns (lowercase) that indicate a Dα measurement or
# Dα-derived quantity.  Any channel matching one of these patterns in ANY
# group is a potential leakage source for the Dα target.
_DALPHA_CHANNEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"da_"),  # filterscope naming (da_hm10_t, da_bo10, …)
    re.compile(r"dalpha"),  # ada group (dalpha_integrated, …)
    re.compile(r"halpha"),  # Hα / Dα filterscope aliases
    re.compile(r"h_alpha"),
    re.compile(r"d_alpha"),
    re.compile(r"balmer"),  # Balmer series (Dα is n=3→2)
    re.compile(r"pellet_halpha"),  # pellet Hα trigger (pellet_halpha_2 in xim)
]

# Channel name patterns for magnetics leakage
_MAGNETICS_CHANNEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^b_"),  # magnetic-field channel
    re.compile(r"^br_"),
    re.compile(r"^bz_"),
    re.compile(r"^bp_"),
    re.compile(r"^bt_"),
    re.compile(r"ip$"),  # plasma current
    re.compile(r"plasma_current"),
    re.compile(r"flux"),
    re.compile(r"saddle"),
    re.compile(r"mirnov"),
    re.compile(r"rogowski"),
    re.compile(r"diamagnetic"),
]

# Camera-boundary leakage patterns (in addition to primary groups)
_CAMERA_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"edge"),          # edge_pixel, edge_radius (xho)
    re.compile(r"boundary"),      # any boundary-derived channel
    re.compile(r"ridge"),         # ridge_pixel (xho)
    re.compile(r"rp_radius"),     # arp/rp_radius
]

# Map candidate target → leakage channel patterns
_TARGET_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "dalpha": _DALPHA_CHANNEL_PATTERNS,
    "magnetics": _MAGNETICS_CHANNEL_PATTERNS,
    "camera_boundary": _CAMERA_BOUNDARY_PATTERNS,
}

# ---------------------------------------------------------------------------
# Empirical leakage audit
# ---------------------------------------------------------------------------


@dataclass
class ChannelLeakageEntry:
    """One leaking channel found in an empirical scan."""

    group: str
    channel: str
    matched_pattern: str
    example_shot: int


@dataclass
class LeakageAudit:
    """Machine-readable leakage audit for ALL candidate targets.

    The audit enumerates every (group, channel) pair that measures or
    derives from a candidate held-out physical quantity, so the
    orchestrator can hold them ALL out together.

    Attributes
    ----------
    target_leakage:
        {candidate_target -> [list of ChannelLeakageEntry objects]}
    scanned_shots:
        Shot IDs used in the empirical scan.
    notes:
        Human-readable notes about the audit.
    """

    target_leakage: dict[str, list[ChannelLeakageEntry]] = field(default_factory=dict)
    scanned_shots: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scanned_shots": self.scanned_shots,
            "notes": self.notes,
            "target_leakage": {
                target: [
                    {
                        "group": e.group,
                        "channel": e.channel,
                        "matched_pattern": e.matched_pattern,
                        "example_shot": e.example_shot,
                    }
                    for e in entries
                ]
                for target, entries in self.target_leakage.items()
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Leakage audit saved to %s", path)

    def summary(self) -> str:
        """Return a human-readable summary of the leakage audit."""
        lines = ["Leakage Audit Summary", "=" * 40]
        for target, entries in self.target_leakage.items():
            lines.append(f"\n  Target: {target}")
            by_group: dict[str, list[str]] = {}
            for e in entries:
                by_group.setdefault(e.group, []).append(e.channel)
            for group, channels in sorted(by_group.items()):
                lines.append(f"    {group}: {', '.join(sorted(channels))}")
            lines.append(
                f"    --> {len(entries)} leaking channels across {len(by_group)} groups"
            )
        return "\n".join(lines)


def _scan_shot_channels(
    shot_zarr_path: Path,
    groups_to_scan: list[str],
) -> dict[str, list[str]]:
    """Return {group: [channel_names]} by listing subdirectory names.

    Channels are the leaf subdirectories (or arrays) inside each group.
    Pure filesystem listing — no Zarr open.
    """
    result: dict[str, list[str]] = {}
    for group in groups_to_scan:
        grp_path = shot_zarr_path / group
        if not grp_path.is_dir():
            continue
        channels = sorted(p.name for p in grp_path.iterdir() if p.is_dir())
        result[group] = channels
    return result


def build_leakage_audit(
    inventory: InventoryResult,
    n_scan_shots: int = 100,
    seed: int = 42,
) -> LeakageAudit:
    """Empirically scan channels in ``n_scan_shots`` rich shots to find leaks.

    For each candidate target (dalpha, magnetics), scan the channel names
    in EVERY group across the scanned shots and flag any channel whose
    name matches a leakage pattern.

    Parameters
    ----------
    inventory:
        Pre-built :class:`~imas_ambix.statespace.inventory.InventoryResult`.
    n_scan_shots:
        Number of shots to scan.  Rich shots (many groups) are prioritised.
        Defaults to 100 — enough to cover all known channel variants.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    LeakageAudit
        Populated leakage audit with all discovered leaking channels.
    """
    from imas_ambix.data.paths import LEVEL1_DIR

    # seed is accepted for API consistency but currently the scan is deterministic
    _ = seed  # future: use rng for random shot selection if n_scan_shots is small

    # Prioritise rich shots (many groups) for maximum channel coverage
    all_shot_ids = sorted(inventory.shot_groups.keys())
    group_counts = {sid: len(grps) for sid, grps in inventory.shot_groups.items()}
    # Sort descending by group count, take top n_scan_shots
    top_shots = sorted(all_shot_ids, key=lambda s: group_counts[s], reverse=True)
    scan_shots = top_shots[:n_scan_shots]
    logger.info("Scanning %d rich shots for leakage channels", len(scan_shots))

    # Collect all (group, channel) pairs seen across scanned shots
    seen_channels: dict[str, set[str]] = {}  # group -> set of channel names
    example_shots: dict[tuple[str, str], int] = {}  # (group, channel) -> shot_id

    for sid in scan_shots:
        groups = inventory.shot_groups[sid]
        shot_path = LEVEL1_DIR / f"{sid}.zarr"
        channels_by_group = _scan_shot_channels(shot_path, list(groups))
        for grp, channels in channels_by_group.items():
            if grp not in seen_channels:
                seen_channels[grp] = set()
            for ch in channels:
                if ch not in seen_channels[grp]:
                    seen_channels[grp].add(ch)
                    example_shots[(grp, ch)] = sid

    logger.info(
        "Scanned %d groups, %d unique (group, channel) pairs",
        len(seen_channels),
        sum(len(v) for v in seen_channels.values()),
    )

    # Now pattern-match for each target
    audit = LeakageAudit(scanned_shots=sorted(scan_shots))
    audit.notes.append(
        f"Empirical scan of {len(scan_shots)} richest shots "
        f"({min(scan_shots)}–{max(scan_shots)})"
    )

    # Primary family groups that MUST always be held out (regardless of channel
    # names).  The hold-out set = primary_family_groups ∪ pattern-matched groups.
    # Pattern matching alone misses groups whose channels don't name-match but
    # ARE the physical quantity (e.g. ama/amb/amh/amm/asm coil currents don't
    # contain 'b_' / 'flux' / 'plasma_current' but ARE magnetics measurements).
    _primary_family_groups: dict[str, frozenset[str]] = {  # noqa: N806 — module-level constant inside fn
        "dalpha": frozenset({"xim", "ada", "aim"}),
        "magnetics": frozenset(
            {"ama", "amb", "amc", "amh", "amm", "asm", "xma", "xmb", "xmc", "xmo"}
        ),
        "camera_boundary": frozenset(
            # rbb is the primary target; xho is a pre-computed boundary analysis
            # from rbb (edge_pixel/edge_radius) — leaks the target directly.
            # air/ait = IR camera heat-load analysis (different physical quantity
            # but derived from boundary position); arp has rp_gap_efit (EFIT-
            # derived and already excluded), rp_radius (position near boundary).
            {"rbb", "xho", "rca"}  # rca = single-frame visible camera
        ),
    }

    for target, patterns in _TARGET_PATTERNS.items():
        entries: list[ChannelLeakageEntry] = []
        # Step 1: add ALL channels from primary family groups
        primary_groups = _primary_family_groups.get(target, frozenset())
        for grp in sorted(primary_groups):
            if grp not in seen_channels:
                # Group was not present in scanned shots; still mark it as leaking
                entries.append(
                    ChannelLeakageEntry(
                        group=grp,
                        channel="*",
                        matched_pattern="primary_family_group",
                        example_shot=-1,
                    )
                )
                continue
            for ch in sorted(seen_channels[grp]):
                entries.append(
                    ChannelLeakageEntry(
                        group=grp,
                        channel=ch,
                        matched_pattern="primary_family_group",
                        example_shot=example_shots.get((grp, ch), -1),
                    )
                )
        # Step 2: pattern-match for additional leaking groups OUTSIDE the
        # primary family (cross-group leakage)
        seen_key = {(e.group, e.channel) for e in entries}
        for grp, channels in sorted(seen_channels.items()):
            if grp in primary_groups:
                continue  # already included above
            for ch in sorted(channels):
                if (grp, ch) in seen_key:
                    continue
                ch_lower = ch.lower()
                for pat in patterns:
                    if pat.search(ch_lower):
                        entries.append(
                            ChannelLeakageEntry(
                                group=grp,
                                channel=ch,
                                matched_pattern=pat.pattern,
                                example_shot=example_shots.get((grp, ch), -1),
                            )
                        )
                        seen_key.add((grp, ch))
                        break  # only record first matching pattern per channel

        audit.target_leakage[target] = entries
        logger.info(
            "Target '%s': %d leaking channels across %d groups",
            target,
            len(entries),
            len({e.group for e in entries}),
        )

    # Add per-target notes
    dalpha_entries = audit.target_leakage.get("dalpha", [])
    dalpha_groups = sorted({e.group for e in dalpha_entries})
    audit.notes.append(
        f"Dα leakage: channels found in groups {dalpha_groups}. "
        "Hold-out MUST exclude all of these groups, not just xim. "
        "NOTE: ada/inner_integrated, inner_peak_radius, geo_full are also "
        "Dα-derived (in the ada group) but not explicitly listed because the "
        "entire ada group is held out."
    )

    mag_entries = audit.target_leakage.get("magnetics", [])
    mag_primary = sorted(
        {e.group for e in mag_entries if e.matched_pattern == "primary_family_group"}
    )
    mag_cross = sorted(
        {
            g
            for g in {e.group for e in mag_entries}
            if g not in set(mag_primary)
        }
    )
    audit.notes.append(
        f"Magnetics leakage: primary groups {mag_primary} "
        f"+ cross-group pattern matches in {mag_cross}. "
        "NOTE: regime split uses Ip from amc — if magnetics is the target, "
        "the split axis itself leaks. Flag this to the orchestrator."
    )

    cam_entries = audit.target_leakage.get("camera_boundary", [])
    if cam_entries:
        cam_groups = sorted({e.group for e in cam_entries})
        audit.notes.append(
            f"Camera-boundary leakage: groups {cam_groups} include xho "
            "(pre-computed edge_radius/edge_pixel — directly leaks camera boundary). "
            "Also exclude rba/rbc/rco/rgb/rgc (other visible cameras — correlated "
            "emission)."
        )

    return audit


# ---------------------------------------------------------------------------
# Hold-out group sets (used to define the actual held-out family)
# ---------------------------------------------------------------------------


def held_out_groups(target: str, leakage_audit: LeakageAudit) -> frozenset[str]:
    """Return the COMPLETE set of groups to hold out for *target*.

    This is the UNION of all groups that contain a leaking channel for
    the target.  Every group in this set must be masked from inputs when
    *target* is the prediction target.

    Parameters
    ----------
    target:
        Candidate held-out family (e.g. ``"dalpha"``, ``"magnetics"``).
    leakage_audit:
        Populated :class:`LeakageAudit`.

    Returns
    -------
    frozenset[str]
        Complete set of Zarr group names to hold out.
    """
    entries = leakage_audit.target_leakage.get(target, [])
    return frozenset(e.group for e in entries)
