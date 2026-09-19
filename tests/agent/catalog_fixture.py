"""One anonymous catalog origin, shared by the launcher tests.

The generated launcher fetches its release catalog from an origin the caller names, so a
test that runs the deployed script stands up an origin answering that fetch. Several of
them also record what the launcher asked for.

The raw form is the general one: it serves whatever payload, status and response headers
a caller hands it, which is what an answer that redirects needs. The items form is the
common shape, a list of catalog cards.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from imas_ambix.agent.profile import SiteConfig

# One recorded request: the path asked for, and the headers it carried as sent.
Request = tuple[str, dict[str, str]]
Origin = tuple[SiteConfig, list[Request]]


@contextmanager
def serve_catalog(
    payload: object,
    *,
    status: int = 200,
    response_headers: dict[str, str] | None = None,
) -> Iterator[Origin]:
    """Serve one anonymous catalog, recording every request's path and headers.

    ``payload`` is sent as JSON unless it is already bytes, for a body that is not a
    JSON document. ``status`` and ``response_headers`` are what a caller needs when the
    answer is not a catalog.
    """
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    requests: list[Request] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, dict(self.headers)))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in (response_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield SiteConfig(global_origin=f"http://{host}:{port}"), requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def serve_catalog_items(items: Sequence[dict[str, object]]) -> Iterator[Origin]:
    """Serve a list of catalog cards, the shape most launcher tests send."""
    with serve_catalog({"data": list(items)}) as origin:
        yield origin
