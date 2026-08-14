"""llama.cpp / llama-swap provider profile (user plugin).

Registers the ``llamacpp`` provider so llama.cpp endpoints - a bare
``llama-server`` or a llama-swap proxy in front of one - get their own
profile instead of riding the generic ``custom`` profile. The bundled
``custom`` profile lists ``llamacpp`` among its aliases; user plugins
register after bundled ones (last-writer-wins), so the self-claimed
``llamacpp`` alias below repoints that lookup at this profile. Removing
this plugin directory restores stock resolution to ``custom``.

Skeleton stage: registration and identity only. Server probing,
reasoning-control emission, and discovery hooks land in later tasks.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class LlamaCppProfile(ProviderProfile):
    """llama.cpp servers, bare or behind llama-swap."""

    # Custom-provider entries named llamacpp / llama-swap activate this
    # profile: providers.resolve_provider_profile does its requested-first
    # lookup only for profiles carrying this opt-in.
    activates_on_requested_provider = True

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Route hermes reasoning effort to llama-server's chat template.

        llama-server reads reasoning controls from the request's top-level
        ``chat_template_kwargs`` (Jinja variables for the served template);
        the OpenAI SDK merges extra_body into the JSON body top level, so
        the mapping goes through extra_body, never an OpenAI parameter.
        Thinking-off (enable_thinking=false) is wired separately.
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if isinstance(reasoning_config, dict):
            effort = str(reasoning_config.get("effort") or "").strip().lower()
            enabled = reasoning_config.get("enabled", True)
            if enabled is not False and effort and effort != "none":
                extra_body["chat_template_kwargs"] = {"reasoning_effort": effort}
        return extra_body, top_level

    def probe_server_caps(self, *, base_url=None, model=None, timeout=3.0):
        """Detect server kind and served-template capabilities (read-only).

        Never dispatches a request that could start a non-resident model
        on llama-swap - see probe.py for the safety contract.
        """
        from . import probe

        return probe.probe_model(base_url or self.base_url, model, timeout=timeout)


llamacpp = LlamaCppProfile(
    name="llamacpp",
    # "llamacpp" is self-claimed on purpose: it steals the alias the
    # bundled custom profile registered, which is what makes lookups by
    # that name resolve here. "llama-swap" is new and ours alone.
    aliases=(
        "llamacpp",
        "llama-swap",
    ),
    display_name="llama.cpp",
    description="Local llama.cpp server (bare llama-server or llama-swap proxy)",
    env_vars=(),  # local endpoint - no API key required
    base_url="",  # user-configured
    auth_type="api_key",
    # Parity with the custom profile this shadows: a generous floor used
    # only when the user has not set model.max_tokens.
    default_max_tokens=65536,
)

register_provider(llamacpp)
