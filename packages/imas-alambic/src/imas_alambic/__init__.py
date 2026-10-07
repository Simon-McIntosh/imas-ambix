"""imas-alambic — the slim IMAS write engine.

Machine maps, signal maps, the transform engine, the virtual Zarr view, the
EDDB read path and the COCOS convention contract, with no dependency on the
world-model training stack.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("imas-alambic")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
