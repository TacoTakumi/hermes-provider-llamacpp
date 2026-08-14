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

from providers import register_provider
from providers.base import ProviderProfile


class LlamaCppProfile(ProviderProfile):
    """llama.cpp servers, bare or behind llama-swap."""


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
