"""Build the SDK model service from the standard platform descriptors."""

from __future__ import annotations

from kdcube_ai_app.infra.service_hub.inventory import (
    ConfigRequest,
    ModelServiceBase,
    create_workflow_config,
    resolve_config_request_secrets,
)


async def build_model_service(
    *,
    role: str,
    check_only: bool,
) -> ModelServiceBase:
    request = ConfigRequest()
    if not check_only:
        request = await resolve_config_request_secrets(request)
    workflow = create_workflow_config(request)
    selected = workflow.ensure_role(role)
    provider = str(selected.get("provider") or "").strip()
    credential = {
        "openai": ("openai_api_key", "platform.services.openai.api_key"),
        "anthropic": ("claude_api_key", "platform.services.anthropic.api_key"),
        "google": ("google_api_key", "platform.services.google.api_key"),
        "openrouter": ("openrouter_api_key", "platform.services.openrouter.api_key"),
    }.get(provider)
    if not check_only and credential and not getattr(workflow, credential[0], None):
        raise RuntimeError(f"secret {credential[1]!r} is not set in secrets.yaml")
    return ModelServiceBase(workflow)
