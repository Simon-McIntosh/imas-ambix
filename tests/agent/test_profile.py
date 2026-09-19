"""Profile parallelism arithmetic.

Cards are ``tensor_parallel * data_parallel``. Every check here exists because
that product is easy to break silently: the failure shows up as a scheduler
request for a card count nobody chose, not as a configuration error.
"""

from __future__ import annotations

from imas_ambix.agent.profile import SiteConfig, load_profile


def test_cards_are_tensor_width_times_replicas():
    """--gpus names CARDS, and cards are tensor_parallel * data_parallel.

    Assigning the card count straight to tensor_parallel multiplied the request
    by the replica count, so `--gpus 4` on a two-replica profile asked for
    eight. The failure is silent at the profile layer and surfaces as a
    scheduler request nobody made.
    """
    from imas_ambix.agent.cli import _tensor_width

    profile = load_profile("deepseek-v4-flash")
    assert profile.engine.tensor_parallel * profile.engine.data_parallel == (
        profile.slurm.gpus
    )
    assert _tensor_width(profile, 8) == 8 // profile.engine.data_parallel


def test_a_card_count_the_replicas_do_not_divide_is_refused():
    """Rounding would serve a topology nobody asked for."""
    import click
    import pytest as _pytest

    from imas_ambix.agent.cli import _tensor_width

    profile = load_profile("deepseek-v4-flash")
    if profile.engine.data_parallel > 1:
        with _pytest.raises(click.BadParameter):
            _tensor_width(profile, profile.engine.data_parallel + 1)


def test_a_single_engine_emits_no_data_parallel_flag():
    """An unset profile must produce the command it produced before the option.

    Otherwise a change made for an unrelated reason becomes attributable to
    this one, which is exactly what a clean before-and-after cannot survive.
    """
    from imas_ambix.agent.slurm import _build_serve_command

    profile = load_profile("glm-5-2")
    assert profile.engine.data_parallel == 1
    assert "--data-parallel-size" not in _build_serve_command(profile, SiteConfig())
