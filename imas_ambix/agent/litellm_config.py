"""Generate the secret-free LiteLLM routing config used by ``clive``.

The site-global native release uses LiteLLM's OpenAI-compatible transport.
OpenRouter routes use the
provider form matching each model family; Claude routes retain the Anthropic
endpoint so Anthropic message features remain intact.  Keys are environment
references and are resolved only by the per-user service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_ambix.agent.profile import SiteConfig


def generate_litellm_config(site: SiteConfig, native_release: str) -> str:
    """Render the proxy YAML for one native release and OpenRouter."""
    global_url = f"{site.global_origin}/v1"
    return f"""# Generated LiteLLM routing config for the per-user clive proxy.
# Keys are loaded from the service environment; this file contains no secrets.

model_list:
  - model_name: {native_release}
    litellm_params:
      model: openai/{native_release}
      api_base: {global_url}
      api_key: os.environ/AMBIX_LOCAL_KEY
    model_info:
      description: "{native_release} — site-global native release"

  - model_name: or-opus-4.8
    litellm_params:
      model: anthropic/claude-opus-4.8
      api_base: https://openrouter.ai/api
      api_key: os.environ/OPENROUTER_API_KEY
    model_info:
      description: "Claude Opus 4.8 via OpenRouter"

  - model_name: or-sonnet-4.6
    litellm_params:
      model: anthropic/claude-sonnet-4.6
      api_base: https://openrouter.ai/api
      api_key: os.environ/OPENROUTER_API_KEY
    model_info:
      description: "Claude Sonnet 4.6 via OpenRouter"

  - model_name: or-gpt-5.5
    litellm_params:
      model: openai/gpt-5.5
      api_base: https://openrouter.ai/api/v1
      api_key: os.environ/OPENROUTER_API_KEY
    model_info:
      description: "GPT-5.5 via OpenRouter"

  - model_name: or-glm-5.2
    litellm_params:
      model: openai/z-ai/glm-5.2
      api_base: https://openrouter.ai/api/v1
      api_key: os.environ/OPENROUTER_API_KEY
    model_info:
      description: "GLM-5.2 via OpenRouter"

litellm_settings:
  drop_params: true

general_settings:
  disable_spend_logs: true
"""
