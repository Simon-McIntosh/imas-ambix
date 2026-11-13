"""The gpu-watch view: the GPU node's cards, the served lane and the queue.

:mod:`imas_ambix.agent.gpu_watch.view` renders the view and its
:func:`~imas_ambix.agent.gpu_watch.view.main` is the ``gpu-watch`` console
script; :mod:`imas_ambix.agent.gpu_watch.sampler` is the reader that runs on
the node inside a job step.
"""

from __future__ import annotations

from imas_ambix.agent.gpu_watch.view import main, render

__all__ = ["main", "render"]
