"""A stamp must be named by the shot set it measured, never by a set it did not.

The results directory is keyed by shot-set label, so a mislabelled stamp is
indistinguishable from the frozen evolution metric it is not.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    FROZEN_SHOTSETS,
    JT60SA_FROZEN_SHOTSET,
    JT60SA_SHOTSET_VERSION,
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
        BenchShot(machine="jt-60sa", shot_id=s.shot_id, role=s.role)
        for s in FROZEN_SHOTSET
    ]
    assert resolve_shotset_version(other_machine) == AD_HOC_SHOTSET_VERSION
    assert resolve_shotset_version(other_machine) != SHOTSET_VERSION


def test_the_frozen_sets_are_keyed_by_machine_and_carry_their_own_labels():
    """MAST's names are its row; JT-60SA's row is a second set, not a relabel."""
    assert set(FROZEN_SHOTSETS) == {"mast", "jt-60sa"}
    assert FROZEN_SHOTSETS["mast"].shots == FROZEN_SHOTSET
    assert FROZEN_SHOTSETS["mast"].version == SHOTSET_VERSION
    assert FROZEN_SHOTSETS["jt-60sa"].shots == JT60SA_FROZEN_SHOTSET
    assert FROZEN_SHOTSETS["jt-60sa"].version == JT60SA_SHOTSET_VERSION
    assert JT60SA_SHOTSET_VERSION not in (SHOTSET_VERSION, AD_HOC_SHOTSET_VERSION)


def test_every_shotset_key_is_the_machine_its_shots_declare():
    """The key must name the machine its row is for, so a lookup cannot miss."""
    for key, row in FROZEN_SHOTSETS.items():
        assert row.machine == key
        assert {shot.machine for shot in row.shots} == {key}


def test_the_jt60sa_key_names_the_machine_the_artifact_layer_builds():
    """One slug across the route: the shot set and the description artifact agree."""
    from imas_ambix.gs import description_artifact

    assert "jt-60sa" in FROZEN_SHOTSETS
    assert description_artifact.MACHINE == "jt-60sa"


def test_the_jt60sa_frozen_set_is_labelled_by_its_own_machine():
    """A JT-60SA run resolves to its own label, never ad-hoc and never MAST's."""
    assert resolve_shotset_version(list(JT60SA_FROZEN_SHOTSET)) == (
        JT60SA_SHOTSET_VERSION
    )
    assert resolve_shotset_version(list(JT60SA_FROZEN_SHOTSET)) != (
        AD_HOC_SHOTSET_VERSION
    )


def test_the_jt60sa_row_is_the_psrc_present_shots_of_the_op1_survey():
    """The set is drawn from the survey, so its ids are the survey's ids."""
    survey = (
        Path(__file__).parents[2]
        / "docs"
        / "evidence"
        / "fragments"
        / "jt60sa-machine-map"
        / "jtmm-op1-shot-survey.json"
    )
    rows = json.loads(survey.read_text())
    expected = [int(row["shot"].lstrip("E")) for row in rows if row["psrc_present"]]
    assert [shot.shot_id for shot in JT60SA_FROZEN_SHOTSET] == expected


def test_a_jt60sa_shot_whose_digits_match_a_mast_id_is_not_labelled_mast():
    """The digits collide but the machine does not: the row decides the label.

    MAST's ids declaring JT-60SA must resolve against JT-60SA's row (ad-hoc, since
    they are not its frozen shots) and never under MAST's label; the genuine
    JT-60SA set resolves to the JT-60SA label rather than ad-hoc.
    """
    colliding = [
        BenchShot(machine="jt-60sa", shot_id=s.shot_id, role=s.role)
        for s in FROZEN_SHOTSET
    ]
    assert resolve_shotset_version(colliding) == AD_HOC_SHOTSET_VERSION
    assert resolve_shotset_version(colliding) != SHOTSET_VERSION
    assert resolve_shotset_version(list(JT60SA_FROZEN_SHOTSET)) == (
        JT60SA_SHOTSET_VERSION
    )


def test_a_machine_with_no_frozen_set_resolves_ad_hoc():
    """An unmapped machine borrows no other machine's label."""
    stranger = [BenchShot(machine="d3d", shot_id=21978, role="ad-hoc")]
    assert resolve_shotset_version(stranger) == AD_HOC_SHOTSET_VERSION


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
