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

import logging
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger("providers.llamacpp")

# hermes effort levels in intensity order, for nearest-level clamping
_EFFORT_RANK = {
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
    "ultra": 7,
}


def _clamp_effort(effort: str, caps) -> str | None:
    """Map a hermes effort level onto what the served template tolerates.

    Returns the wire value, or None to omit the kwarg entirely.
    caps None (cold/unknown model, bare endpoint without props) keeps the
    verbatim passthrough - the user's explicit config is honored when the
    template cannot be inspected.
    """
    if caps is None:
        return effort
    if not caps.has_reasoning_effort:
        return None
    if effort in caps.accepted_efforts:
        return effort
    if effort in caps.remapped_efforts:
        # the template would rewrite it anyway; do it client-side so the
        # wire always carries a value from the accepted set
        return caps.remapped_efforts[effort]
    rank = _EFFORT_RANK.get(effort)
    ranked = [e for e in caps.accepted_efforts if e in _EFFORT_RANK]
    if rank is None or not ranked:
        return None
    return min(ranked, key=lambda e: (abs(_EFFORT_RANK[e] - rank), _EFFORT_RANK[e]))


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
            if enabled is False or effort == "none":
                # Thinking off. enable_thinking must be a JSON boolean - a
                # string "false" is truthy inside the Jinja template and can
                # 400 server-side. Omit only when the served template
                # provably has no toggle; unknown caps still emit (an unused
                # Jinja var is inert on non-thinking templates).
                emit_toggle = True
                base_url = context.get("base_url") or self.base_url
                if base_url:
                    try:
                        from . import probe

                        result = probe.probe_model(base_url, context.get("model"))
                        if (
                            result.caps is not None
                            and not result.caps.supports_thinking_toggle
                        ):
                            emit_toggle = False
                    except Exception as exc:
                        logger.debug(
                            "llamacpp thinking-off probe failed: %s", exc
                        )
                if emit_toggle:
                    extra_body["chat_template_kwargs"] = {
                        "enable_thinking": False
                    }
            elif effort:
                wire_effort: str | None = effort
                base_url = context.get("base_url") or self.base_url
                if base_url:
                    try:
                        from . import probe

                        result = probe.probe_model(base_url, context.get("model"))
                        wire_effort = _clamp_effort(effort, result.caps)
                    except Exception as exc:
                        logger.debug(
                            "llamacpp effort clamp skipped (probe failed): %s", exc
                        )
                if wire_effort != effort:
                    logger.info(
                        "llamacpp: reasoning_effort %r -> %s (served template)",
                        effort,
                        wire_effort if wire_effort else "omitted",
                    )
                if wire_effort:
                    extra_body["chat_template_kwargs"] = {
                        "reasoning_effort": wire_effort
                    }
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
