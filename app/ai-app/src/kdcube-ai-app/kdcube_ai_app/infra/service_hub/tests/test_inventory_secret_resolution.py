from __future__ import annotations

import pytest

import kdcube_ai_app.infra.service_hub.inventory as inventory


@pytest.mark.asyncio
async def test_model_secret_resolution_keeps_bundle_keys_relative(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    async def _get_secret(key, *, bundle_id=None, **_kwargs):
        calls.append((key, bundle_id))
        if key == "b:services.openai.api_key":
            return "bundle-openai"
        return None

    monkeypatch.setattr(inventory, "get_secret", _get_secret)

    result = await inventory.resolve_config_request_secrets(
        inventory.ConfigRequest(),
        bundle_id="workspace@1-0",
    )

    assert result.openai_api_key == "bundle-openai"
    assert calls[0] == ("b:services.openai.api_key", "workspace@1-0")
    assert all(not key.startswith("b:platform.") for key, _bundle_id in calls)
    assert (
        "platform.services.anthropic.api_key",
        None,
    ) in calls
