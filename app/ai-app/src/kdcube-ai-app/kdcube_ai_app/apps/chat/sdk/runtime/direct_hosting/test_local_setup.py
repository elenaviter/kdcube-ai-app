from __future__ import annotations

from kdcube_ai_app.apps.chat.sdk.runtime.direct_hosting.local_setup import (
    _secret_descriptor,
)


def test_generated_secrets_use_platform_qualified_descriptor_paths() -> None:
    document = _secret_descriptor(
        {
            "platform": {
                "services": {"anthropic": {"api_key": None}},
                "infra": {
                    "postgres": {"password": None},
                    "redis": {"password": None},
                },
            }
        },
        provider="anthropic",
        provider_key="provider-secret",
        postgres="postgres-secret",
        redis="redis-secret",
    )

    assert "services" not in document
    assert "infra" not in document
    assert document["platform"]["services"]["anthropic"]["api_key"] == (
        "provider-secret"
    )
    assert document["platform"]["infra"]["postgres"]["password"] == ("postgres-secret")
    assert document["platform"]["infra"]["redis"]["password"] == "redis-secret"
