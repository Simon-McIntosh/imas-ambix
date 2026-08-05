"""Offline tests for the acquisition-range setting the magnetics read divides out.

Nineteen MAST probe channels were recorded at more than one range setting, so the
same field arrives as a different stored amplitude depending on which shot is
read.  These pin what the read does about it: a channel with a measured setting is
referred to one, a channel without one is left exactly as published, and every
read says which of the two happened.  The table is injected here, so nothing
touches the ``/work`` mirror.
"""

from __future__ import annotations

import numpy as np
from nova.imas.mast_block_scale import BlockScale, BlockScaleTable

from imas_ambix.latent.data import divide_out_acquisition_scale

STEPPING = "obv04"
STEADY = "obr01"


def _table() -> BlockScaleTable:
    """A channel that steps by a rung beside one that never moves.

    ``STEPPING`` holds the reference setting over the first block and half of it
    over the second, which is the case the correction exists for: the second
    block's amplitudes are twice the first's for the same field.
    """
    return BlockScaleTable.create(
        [
            BlockScale(channel=STEPPING, scale=1.0, shots=(100, 101), rung=1.0),
            BlockScale(channel=STEPPING, scale=2.0, shots=(200, 201), rung=2.0),
            BlockScale(channel=STEADY, scale=1.0, shots=(100, 201), rung=1.0),
        ],
        route="test",
    )


def _values() -> np.ndarray:
    return np.array([[4.0, 7.0], [6.0, 9.0]], dtype=np.float64)


def test_a_measured_step_is_divided_out_of_the_stepping_channel():
    values, warrants = divide_out_acquisition_scale(
        _values(), [STEPPING, STEADY], 200, table=_table()
    )
    assert values[:, 0].tolist() == [2.0, 3.0]  # the rung of two removed
    assert warrants[0].channel == STEPPING
    assert warrants[0].disposition == "measured"
    assert warrants[0].scale == 2.0


def test_the_steady_channel_is_untouched_on_the_same_read():
    values, warrants = divide_out_acquisition_scale(
        _values(), [STEPPING, STEADY], 200, table=_table()
    )
    assert values[:, 1].tolist() == [7.0, 9.0]
    assert warrants[1].channel == STEADY
    assert warrants[1].scale == 1.0


def test_the_reference_block_leaves_the_stepping_channel_alone():
    values, warrants = divide_out_acquisition_scale(
        _values(), [STEPPING, STEADY], 100, table=_table()
    )
    assert values[:, 0].tolist() == [4.0, 6.0]
    assert warrants[0].disposition == "measured"
    assert warrants[0].scale == 1.0


def test_a_shot_inside_a_block_but_not_measured_on_it_is_still_corrected():
    """Both ends of the block agree, so there is no switch to place inside it."""
    values, warrants = divide_out_acquisition_scale(
        _values(), [STEPPING], 2005, table=_table_spanning()
    )
    assert values[:, 0].tolist() == [2.0, 3.0]
    assert warrants[0].disposition == "bracketed"


def _table_spanning() -> BlockScaleTable:
    return BlockScaleTable.create(
        [BlockScale(channel=STEPPING, scale=2.0, shots=(2000, 2010), rung=2.0)],
        route="test",
    )


def test_a_shot_in_the_gap_between_two_blocks_is_read_as_published():
    """The switch is somewhere in the gap and no shot in there says where."""
    values, warrants = divide_out_acquisition_scale(
        _values(), [STEPPING], 150, table=_table()
    )
    assert values[:, 0].tolist() == [4.0, 6.0]
    assert warrants[0].disposition == "unmeasured"
    assert warrants[0].candidates == (1.0, 2.0)


def test_a_step_that_is_not_a_ladder_rung_is_refused_rather_than_rounded():
    table = BlockScaleTable.create(
        [BlockScale(channel=STEPPING, scale=1.37, shots=(300, 301), rung=float("nan"))],
        route="test",
    )
    values, warrants = divide_out_acquisition_scale(
        _values(), [STEPPING], 300, table=table
    )
    assert values[:, 0].tolist() == [4.0, 6.0]
    assert warrants[0].disposition == "refused"


def test_a_channel_the_table_never_measured_is_read_as_published():
    values, warrants = divide_out_acquisition_scale(
        _values(), ["fl_p4u_1", STEADY], 200, table=_table()
    )
    assert values[:, 0].tolist() == [4.0, 6.0]
    assert warrants[0].disposition == "unmeasured"
    assert warrants[0].candidates == ()


def test_the_warrants_come_back_in_the_column_order_that_was_read():
    """A summary keyed by position would mislabel which channel was divided."""
    channels = [STEADY, "fl_p4u_1", STEPPING]
    _values_out, warrants = divide_out_acquisition_scale(
        np.ones((2, 3)), channels, 200, table=_table()
    )
    assert [row.channel for row in warrants] == channels


def test_an_empty_table_reads_the_archive_exactly_as_published():
    values, warrants = divide_out_acquisition_scale(
        _values(), [STEPPING, STEADY], 200, table=BlockScaleTable()
    )
    assert values.tolist() == _values().tolist()
    assert {row.disposition for row in warrants} == {"unmeasured"}


def test_the_promoted_table_is_what_a_read_applies_by_default():
    """A default of no correction would make the read silently campaign-dependent."""
    from nova.imas.mast_block_scale import promoted_block_scales

    table = promoted_block_scales()
    assert len(table.stepping) == 19
    channel = table.corrected[0]
    block = next(row for row in table.blocks[channel] if row.rung != 1.0)
    values, warrants = divide_out_acquisition_scale(
        np.ones((1, 1)), [channel], block.first_shot
    )
    assert warrants[0].scale == block.rung
    assert values[0, 0] == 1.0 / block.rung
