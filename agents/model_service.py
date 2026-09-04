"""Descriptor-driven model-service construction for direct examples."""

from __future__ import annotations

import os
from typing import Any

from kdcube_ai_app.infra.service_hub.inventory import (
    ConfigRequest,
    ModelServiceBase,
    create_workflow_config,
)


def build_model_service(
    config: dict[str, Any],
    *,
    role: str,
    check_only: bool,
) -> ModelServiceBase:
    model = dict(config.get("model") or {})
    provider = str(model.get("provider") or "openai")
    model_name = str(model.get("name") or "gpt-5-mini")
    key_ref = str(model.get("api_key_ref") or "OPENAI_API_KEY")
    api_key = os.environ.get(key_ref, "")
    if not api_key and not check_only:
        raise RuntimeError(f"secret {key_ref!r} is not set")
    kwargs: dict[str, Any] = {
        "role_models": {role: {"provider": provider, "model": model_name}}
    }
    field = {
        "openai": "openai_api_key",
        "anthropic": "claude_api_key",
        "google": "google_api_key",
        "openrouter": "openrouter_api_key",
    }.get(provider)
    if field:
        kwargs[field] = api_key or "check-only"
    return ModelServiceBase(create_workflow_config(ConfigRequest(**kwargs)))
