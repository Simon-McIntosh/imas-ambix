"""Tests for splitting the pooled sensor misfit across the channels carrying it.

The solve itself needs the level-1 mirror, so what is pinned here is the part
that decides what the decomposition SAYS: how the shares are computed, how a
channel is compared with its own noise floor, and what happens to a channel with
no floor measured in its units.
"""

from __future__ import annotations

from imas_ambix.spine_bench.channel_gap import (
    _median_of_shot_medians,
    _pooled_floor,
    _pooled_slice_rms,
    summarise_channels,
)


def test_the_shares_add_to_one_so_the_split_accounts_for_the_whole_misfit():
    rows = summarise_channels(
        {"obv06": [3.0, 4.0], "obr01": [1.0], "fl_p4u_1": [2.0]},
        {"obv06": [3e-3, 4e-3], "obr01": [1e-3], "fl_p4u_1": [2e-3]},
        {},
    )

    assert sum(row["share_of_mean_square"] for row in rows) == 1.0


def test_the_share_is_of_the_mean_square_not_of_the_rms():
    """A channel scored on one slice must not outrank one carrying more total.

    Both channels here have the same rms; the first carries twice the misfit
    because it was scored on twice as many slices, and the ranking has to see it.
    """
    rows = summarise_channels(
        {"often": [2.0, 2.0], "once": [2.0]},
        {"often": [1e-3, 1e-3], "once": [1e-3]},
        {},
    )

    assert [row["channel"] for row in rows] == ["often", "once"]
    assert rows[0]["rms_whitened"] == rows[1]["rms_whitened"]
    assert rows[0]["share_of_mean_square"] == 2.0 / 3.0


def test_a_channel_is_reported_as_a_multiple_of_its_own_measured_floor():
    rows = summarise_channels({"obv06": [1.0]}, {"obv06": [4.0e-3]}, {"obv06": 4.0e-4})

    assert rows[0]["gap_over_floor"] == 10.0
    assert rows[0]["noise_floor"] == 4.0e-4
    assert rows[0]["unit"] == "T"


def test_a_channel_with_no_measured_floor_reports_none_rather_than_a_number():
    """A flux loop carries webers, so the field floor is not its floor."""
    rows = summarise_channels({"fl_p4u_1": [1.0]}, {"fl_p4u_1": [2.0e-3]}, {})

    assert rows[0]["gap_over_floor"] is None
    assert rows[0]["noise_floor"] is None
    assert rows[0]["unit"] == "Wb"


def test_the_stamp_metric_is_reproduced_by_medians_of_shot_medians():
    """The stamp takes a median per shot and then a median over shots, so a single
    pooled median over every slice is a DIFFERENT number and would look like a
    disagreement with the gate rather than a different average of the same data."""
    per_shot = {21978: [1.0, 2.0, 9.0], 21983: [3.0, 4.0]}

    assert _median_of_shot_medians(per_shot) == 2.75
    assert _pooled_slice_rms(per_shot) == 3.0


def test_no_scored_slice_reports_not_a_number_rather_than_zero():
    """Zero would read as a perfect fit."""
    assert _median_of_shot_medians({}) != _median_of_shot_medians({})
    assert _pooled_slice_rms({}) != _pooled_slice_rms({})


def test_the_pooled_floor_is_the_quadratic_mean_of_the_channel_floors():
    """A residual pooled over the array is a quadratic mean, so its floor must be
    the same kind of average or the two are not comparable."""
    assert _pooled_floor({"a": 3.0, "b": 4.0}) == 3.5355339059327378
    assert _pooled_floor({}) != _pooled_floor({})  # no floors measured -> NaN
