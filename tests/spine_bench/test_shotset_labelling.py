"""A stamp must be named by the shot set it measured, never by a set it did not.

The results directory is keyed by shot-set label, so a mislabelled stamp is
indistinguishable from the frozen evolution metric it is not.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from imas_ambix.spine_bench.runner import write_yaml
from imas_ambix.spine_bench.schema import (
    EnvInfo,
    MachineInfo,
    ShotStamp,
    SpineBenchmarkStamp,
)
from imas_ambix.spine_bench.shots import (
    AD_HOC_SHOTSET_VERSION,
    FROZEN_SHOTSET,
    SHOTSET_VERSION,
    BenchShot,
    resolve_shotset_version,
)


def test_the_frozen_set_is_labelled_frozen_when_no_override_is_given():
    assert resolve_shotset_version(None) == SHOTSET_VERSION


def test_a_bench_shot_requires_a_machine():
    """The machine is declared and never inferred, so its absence is refused."""
    with pytest.raises(ValidationError):
        BenchShot(shot_id=21978, role="flat-top representative")


def test_the_frozen_shots_declare_the_mast_machine():
    assert {shot.machine for shot in FROZEN_SHOTSET} == {"mast"}
    assert len(FROZEN_SHOTSET) == 6


def test_an_override_naming_exactly_the_frozen_set_keeps_the_frozen_label():
    """Passing the frozen shots explicitly measures the frozen metric."""
    assert resolve_shotset_version(list(FROZEN_SHOTSET)) == SHOTSET_VERSION


def test_a_subset_of_the_frozen_shots_is_labelled_ad_hoc():
    """The three-shot override that must never again claim the frozen name."""
    subset = list(FROZEN_SHOTSET[:3])
    assert resolve_shotset_version(subset) == AD_HOC_SHOTSET_VERSION


def test_an_extra_shot_is_labelled_ad_hoc():
    extended = [
        *FROZEN_SHOTSET,
        BenchShot(machine="mast", shot_id=99999, role="ad-hoc"),
    ]
    assert resolve_shotset_version(extended) == AD_HOC_SHOTSET_VERSION


def test_a_reordered_frozen_set_is_labelled_ad_hoc():
    """Order is part of the pin: metrics are medians over the set as recorded."""
    reordered = list(reversed(FROZEN_SHOTSET))
    assert resolve_shotset_version(reordered) == AD_HOC_SHOTSET_VERSION


def test_the_frozen_shots_under_a_different_role_are_labelled_ad_hoc():
    """The command line cannot silently re-role the frozen set."""
    reroled = [
        BenchShot(machine=s.machine, shot_id=s.shot_id, role="ad-hoc")
        for s in FROZEN_SHOTSET
    ]
    assert resolve_shotset_version(reroled) == AD_HOC_SHOTSET_VERSION


def test_the_frozen_ids_and_roles_under_another_machine_are_labelled_ad_hoc():
    """A coincident shot id on another machine must never claim MAST's label."""
    other_machine = [
        BenchShot(machine="jt60sa", shot_id=s.shot_id, role=s.role)
        for s in FROZEN_SHOTSET
    ]
    assert resolve_shotset_version(other_machine) == AD_HOC_SHOTSET_VERSION
    assert resolve_shotset_version(other_machine) != SHOTSET_VERSION


def test_the_cli_refuses_an_override_that_does_not_name_a_machine():
    """--shots without --machine is refused by the parser, not assumed to be MAST."""
    from scripts.spine_benchmark import build_parser

    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--shots", "21978"])
    assert excinfo.value.code == 2


def test_the_cli_does_not_require_a_machine_for_the_frozen_set():
    """With no --shots the frozen set is used, so --machine is not needed."""
    from scripts.spine_benchmark import build_parser

    args = build_parser().parse_args([])
    assert args.shots == ""
    assert args.machine is None


def _stamp(shotset_version: str) -> SpineBenchmarkStamp:
    return SpineBenchmarkStamp(
        shotset_version=shotset_version,
        created_utc="2026-01-01T00:00:00+00:00",
        machine=MachineInfo(hostname="testhost.example", platform="linux"),
        env=EnvInfo(
            python_version="3.14.0",
            git_commit="0123456789abcdef",
            git_dirty=False,
        ),
        shots=[ShotStamp(shot_id=21978, role="ad-hoc", substrate="greens-matvec")],
    )


def test_an_ad_hoc_stamp_is_written_under_a_filename_that_says_so(tmp_path):
    """The discriminator is in the path, so a directory listing cannot mislead."""
    path = write_yaml(_stamp(AD_HOC_SHOTSET_VERSION), tmp_path)
    assert AD_HOC_SHOTSET_VERSION in path.name
    assert SHOTSET_VERSION not in path.name
    assert yaml.safe_load(path.read_text())["shotset_version"] == (
        AD_HOC_SHOTSET_VERSION
    )


def test_a_frozen_stamp_keeps_the_historical_filename_shape(tmp_path):
    """The frozen stamp's name must not change: committed stamps are compared by it."""
    path = write_yaml(_stamp(SHOTSET_VERSION), tmp_path)
    assert path.name == f"physics-spine-{SHOTSET_VERSION}-0123456789-testhost.yaml"


def test_an_ad_hoc_stamp_can_never_collide_with_the_frozen_stamp_filename(tmp_path):
    """Same commit and host, different set: two files, not one overwritten."""
    frozen = write_yaml(_stamp(SHOTSET_VERSION), tmp_path)
    ad_hoc = write_yaml(_stamp(AD_HOC_SHOTSET_VERSION), tmp_path)
    assert frozen != ad_hoc
    assert frozen.exists() and ad_hoc.exists()
