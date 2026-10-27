"""The gpu-watch view, moved in from the dotfiles script.

The renderer lives here so the launcher the operator runs holds no figures of
its own; :mod:`imas_ambix.agent.gpu_watch.view` carries the view, and its
:func:`imas_ambix.agent.gpu_watch.view.main` is the ``gpu-watch`` console
script.
"""

from __future__ import annotations

from imas_ambix.agent.gpu_watch.view import (
    DEFAULT_WINDOW_S,
    NO_JOB,
    Reading,
    card_figures,
    conditions,
    main,
    render,
    render_text,
    sample_readings,
)

__all__ = [
    "DEFAULT_WINDOW_S",
    "NO_JOB",
    "Reading",
    "card_figures",
    "conditions",
    "main",
    "render",
    "render_text",
    "sample_readings",
]
