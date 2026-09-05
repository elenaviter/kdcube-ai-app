"""Reusable SDK support for hosting agent cores in a direct Python process."""

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.configuration import (
    ConfiguredSkills,
    ConfiguredTool,
    activate_configured_skills,
    agent_instructions,
    configured_run_directory,
    configured_skills,
    configured_tool_settings,
    configured_tools,
    configured_web_search,
    enabled_tool_ids,
    require_supported_tools,
    verify_docker_image,
    verify_playwright_chromium,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.evidence import (
    ConsoleEmitter,
    print_evidence_summary,
    utc_now,
    write_evidence_index,
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
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.native_tool_bindings import (
    NativeToolBindings,
    NativeToolSource,
    resolve_native_tool_bindings,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.tool_runtime import (
    DirectToolRuntime,
)
from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.workspace import (
    DirectTurnWorkspace,
)

__all__ = [
    "ConfiguredSkills",
    "ConfiguredTool",
    "ConsoleEmitter",
    "DirectToolRuntime",
    "DirectTurnWorkspace",
    "NativeToolBindings",
    "NativeToolSource",
    "activate_configured_skills",
    "activate_platform_descriptors",
    "agent_instructions",
    "build_model_service",
    "configure",
    "configured_run_directory",
    "configured_skills",
    "configured_tool_settings",
    "configured_tools",
    "configured_web_search",
    "direct_harness_config",
    "enabled_tool_ids",
    "platform_exec_profile",
    "print_evidence_summary",
    "postgres_label",
    "postgres_url",
    "redis_url",
    "resolve_native_tool_bindings",
    "require_supported_tools",
    "storage_uri",
    "utc_now",
    "verify_docker_image",
    "verify_playwright_chromium",
    "write_evidence_index",
]
