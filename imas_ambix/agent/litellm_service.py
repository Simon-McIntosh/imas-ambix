# ruff: noqa: E501
"""Generate the per-user LiteLLM service and its runtime environment helper.

The helper reads credentials only when the service starts and writes a mode-600
file beneath the user's runtime directory.  The generated unit is installed
only by the explicit OpenRouter deployment path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_ambix.agent.profile import SiteConfig

LITELLM_PORT = 18399


def generate_litellm_env_helper(site: SiteConfig) -> str:
    """Render the runtime credential-file builder."""
    local_key_file = str(site.api_key_file)
    return f"""#!/usr/bin/env bash
# Generated credential helper for the per-user clive proxy.
set -euo pipefail
umask 077
_out="${{XDG_RUNTIME_DIR:-/tmp}}/imas-ambix-llm.env"
_or=""
[[ -r "$HOME/.config/openrouter/key" ]] && _or="$(tr -d '[:space:]' < "$HOME/.config/openrouter/key")"
_local="no-auth"
if [[ -r "{local_key_file}" ]]; then
    _key="$(grep -E '^[[:space:]]*AMBIX_AGENT_API_KEY[[:space:]]*=' "{local_key_file}" 2>/dev/null | tail -1 | sed -E 's/[^=]*=//; s/^["'"'"']//; s/["'"'"']$//' || true)"
    [[ -n "$_key" ]] && _local="$_key"
fi
{{
    printf 'OPENROUTER_API_KEY=%s\\n' "$_or"
    printf 'AMBIX_LOCAL_KEY=%s\\n' "$_local"
}} > "$_out"
"""


def generate_litellm_service(site: SiteConfig) -> str:
    """Render the loopback-only per-user LiteLLM systemd unit."""
    config_path = str(site.litellm_config_path)
    helper_path = str(site.litellm_env_helper_path)
    return f"""# Generated per-user LiteLLM proxy for clive OpenRouter routing.
[Unit]
Description=imas-ambix LiteLLM proxy (per-user)
After=network.target

[Service]
Type=simple
ExecStartPre=/bin/bash {helper_path}
EnvironmentFile=-%t/imas-ambix-llm.env
ExecStart=%h/.local/bin/litellm --config {config_path} --host 127.0.0.1 --port {LITELLM_PORT}
ExecStopPost=/bin/rm -f %t/imas-ambix-llm.env
Restart=on-failure
RestartSec=5
TimeoutStartSec=120

[Install]
WantedBy=default.target
"""
