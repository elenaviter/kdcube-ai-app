from __future__ import annotations

import pytest
from kdcube_ai_app.apps.chat.proc.rest.management.secret_contracts import (
    BUNDLE_SCOPE,
    MAX_SECRET_VALUE_BYTES,
    PLATFORM_SCOPE,
    SecretTarget,
    validate_secret_value,
)


def test_exact_platform_and_bundle_secret_resources() -> None:
    platform = SecretTarget(
        scope=PLATFORM_SCOPE,
        key="services.brave.api_key",
    )
    bundle = SecretTarget(
        scope=BUNDLE_SCOPE,
        bundle_id="workspace@1-0",
        key="provider.api-key",
    )

    assert platform.provider_key == "services.brave.api_key"
    assert platform.resource(tenant="tenant-a", project="project-a") == (
        "urn:kdcube:management:secret:tenant-a:project-a:platform:_:"
        "services.brave.api_key"
    )
    assert bundle.provider_key == (
        "bundles.workspace@1-0.secrets.provider.api-key"
    )
    assert bundle.resource(tenant="tenant-a", project="project-a") == (
        "urn:kdcube:management:secret:tenant-a:project-a:bundle:"
        "workspace@1-0:provider.api-key"
    )


@pytest.mark.parametrize(
    ("scope", "bundle_id", "key"),
    [
        (PLATFORM_SCOPE, "workspace@1-0", "services.api_key"),
        (BUNDLE_SCOPE, "", "services.api_key"),
        (PLATFORM_SCOPE, "", "bundles.workspace.secrets.key"),
        (PLATFORM_SCOPE, "", "users.someone.secrets.key"),
        (PLATFORM_SCOPE, "", "services..api_key"),
        (PLATFORM_SCOPE, "", "__keys"),
        (BUNDLE_SCOPE, "workspace@1-0", "provider.__keys"),
        (BUNDLE_SCOPE, "../workspace", "provider.api_key"),
    ],
)
def test_secret_target_rejects_scope_escape_and_metadata_keys(
    scope: str,
    bundle_id: str,
    key: str,
) -> None:
    with pytest.raises(ValueError):
        SecretTarget(scope=scope, bundle_id=bundle_id, key=key)


def test_secret_value_requires_string_with_bounded_utf8_size() -> None:
    assert validate_secret_value("a" * MAX_SECRET_VALUE_BYTES)

    with pytest.raises(ValueError, match="must not be empty"):
        validate_secret_value("")
    with pytest.raises(ValueError, match="must be a string"):
        validate_secret_value(123)
    with pytest.raises(ValueError, match="must not exceed"):
        validate_secret_value("x" * (MAX_SECRET_VALUE_BYTES + 1))
    with pytest.raises(ValueError, match="must not exceed"):
        validate_secret_value("\u00e9" * (MAX_SECRET_VALUE_BYTES // 2 + 1))


def test_secret_target_rejects_non_text_protocol_fields() -> None:
    with pytest.raises(ValueError, match="must be text"):
        SecretTarget(scope=123, key="services.brave.api_key")
    with pytest.raises(ValueError, match="must be text"):
        SecretTarget(scope=PLATFORM_SCOPE, key=123)
