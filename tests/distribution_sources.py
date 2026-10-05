"""Shared resolution of an installed distribution's editable checkout.

The declaration census in ``test_declared_dependencies`` and the Nova
dependency-contract test in ``tests.physics.test_dependency_contract`` both
need the filesystem path an editable path install points at.  The path is
carried in the install's ``direct_url.json`` as a file URL, where reserved
characters are percent-encoded; it must be decoded before it is handed to the
filesystem, or a checkout whose path contains a space or another reserved
character resolves to a directory that does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlsplit


def editable_checkout(direct_url_json: str) -> Path | None:
    """The checkout a ``direct_url.json`` file URL points at, decoded.

    Returns ``None`` when the record is missing, malformed, or not a file URL
    (a registry install carries no checkout).
    """
    try:
        direct_url = json.loads(direct_url_json)
    except (TypeError, ValueError):
        return None
    url = direct_url.get("url")
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    if parts.scheme != "file":
        return None
    return Path(unquote(parts.path))
