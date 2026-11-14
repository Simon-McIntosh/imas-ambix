"""ambix-watch: the GPU servers, the lanes they serve and their queues.

:mod:`~imas_ambix.agent.gpu_watch.app` is the console script and the page
switching; :mod:`~imas_ambix.agent.gpu_watch.fleet` reads the scheduler and
the cards; :mod:`~imas_ambix.agent.gpu_watch.lane` reads the serving document;
:mod:`~imas_ambix.agent.gpu_watch.panels` draws the pages; and
:mod:`~imas_ambix.agent.gpu_watch.sampler` is the reader that runs on a server
inside a job step.
"""

from __future__ import annotations

from imas_ambix.agent.gpu_watch.app import main, render

__all__ = ["main", "render"]
