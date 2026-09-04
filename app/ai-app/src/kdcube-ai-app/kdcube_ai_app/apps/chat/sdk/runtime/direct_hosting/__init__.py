"""Reusable SDK support for hosting agent cores in a direct Python process."""

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (
    ConfiguredSkills,
    ConfiguredTool,
    activate_configured_skills,
    agent_instructions,
    configured_skills,
    configured_tools,
    enabled_tool_ids,
    require_supported_tools,
    verify_docker_image,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.evidence import (
    ConsoleEmitter,
    utc_now,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.infrastructure import (
    activate_platform_descriptors,
    direct_harness_config,
    platform_exec_profile,
    postgres_label,
    postgres_url,
    redis_url,
    storage_uri,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.local_setup import (
    configure,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.model_service import (
    build_model_service,
)

__all__ = [
    "ConfiguredSkills",
    "ConfiguredTool",
    "ConsoleEmitter",
    "activate_configured_skills",
    "activate_platform_descriptors",
    "agent_instructions",
    "build_model_service",
    "configure",
    "configured_skills",
    "configured_tools",
    "direct_harness_config",
    "enabled_tool_ids",
    "platform_exec_profile",
    "postgres_label",
    "postgres_url",
    "redis_url",
    "require_supported_tools",
    "storage_uri",
    "utc_now",
    "verify_docker_image",
]
