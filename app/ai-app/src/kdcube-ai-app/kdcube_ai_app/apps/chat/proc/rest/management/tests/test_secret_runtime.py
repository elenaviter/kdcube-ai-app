from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from kdcube_ai_app.apps.chat.proc.rest.management.secret_contracts import (
    BUNDLE_SCOPE,
    MAX_SECRET_VALUE_BYTES,
    PLATFORM_SCOPE,
    SecretTarget,
)
from kdcube_ai_app.apps.chat.proc.rest.management.secret_runtime import (
    KDCubeSecretRuntime,
    ManagementSecretNotFound,
    ManagementSecretsProviderReadOnly,
    ManagementSecretsProviderUnavailable,
)
from starlette.requests import Request


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.published: list[tuple[str, str]] = []

    async def set(self, key: str, value: str, **_kwargs: Any) -> bool:
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        return 1 if self.values.pop(key, None) is not None else 0

    async def publish(self, channel: str, value: str) -> int:
        self.published.append((channel, value))
        return 1


class _Manager:
    provider_type = "fixture"

    def __init__(self, *, writable: bool = True) -> None:
        self.data: dict[str, str] = {}
        self.writable = writable

    async def get_secret(self, key: str) -> str | None:
        return self.data.get(key)

    async def get_secret_strict(self, key: str) -> str | None:
        return await self.get_secret(key)

    def can_write(self) -> bool:
        return self.writable

    async def set_secret(self, key: str, value: str) -> None:
        self.data[key] = value

    async def delete_secret(self, key: str) -> None:
        self.data.pop(key, None)

    async def set_many(self, values: dict[str, str]) -> None:
        self.data.update(values)


def _request(redis: _Redis) -> Request:
    app = SimpleNamespace(state=SimpleNamespace(redis_async=redis))
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("runtime.example", 443),
            "app": app,
        }
    )


@pytest.fixture()
def declared_bundle(monkeypatch):
    from kdcube_ai_app.apps.chat.proc.rest.management import secret_runtime

    async def _registry(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(bundles={"workspace@1-0": object()})

    monkeypatch.setattr(secret_runtime, "load_registry", _registry)


@pytest.mark.asyncio
async def test_bundle_secret_lifecycle_invalidates_derived_inventory_cache(
    declared_bundle,
) -> None:
    del declared_bundle
    redis = _Redis()
    manager = _Manager()
    runtime = KDCubeSecretRuntime(
        _request(redis),
        tenant="tenant-a",
        project="project-a",
        manager=manager,
    )
    target = SecretTarget(
        scope=BUNDLE_SCOPE,
        bundle_id="workspace@1-0",
        key="provider.api_key",
    )
    redis.values[
        "kdcube:config:bundles:secrets:tenant-a:project-a:workspace@1-0"
    ] = json.dumps([target.provider_key])

    created = await runtime.write(
        target,
        value="secret-canary",
        caller_profile="devops-agent",
    )
    metadata = await runtime.metadata(target)
    disclosed = await runtime.read(target)
    deleted = await runtime.delete(target, caller_profile="devops-agent")

    metadata_key = "bundles.workspace@1-0.secrets.__keys"
    assert created == {
        "scope": "bundle",
        "bundle_id": "workspace@1-0",
        "key": "provider.api_key",
        "created": True,
        "provider": "fixture",
        "state": "stored",
    }
    assert metadata["exists"] is True
    assert disclosed["value"] == "secret-canary"
    assert deleted["existed"] is True
    assert target.provider_key not in manager.data
    assert metadata_key not in manager.data
    assert redis.values == {}
    assert redis.published
    assert "secret-canary" not in str(created)
    assert "secret-canary" not in str(metadata)
    assert "secret-canary" not in str(deleted)
    assert "secret-canary" not in str(redis.values)
    assert "secret-canary" not in str(redis.published)


@pytest.mark.asyncio
async def test_bundle_write_does_not_mutate_legacy_inventory_record(
    declared_bundle,
) -> None:
    del declared_bundle
    manager = _Manager()
    metadata_key = "bundles.workspace@1-0.secrets.__keys"
    manager.data[metadata_key] = json.dumps(
        ["bundles.workspace@1-0.secrets.existing.token"]
    )
    runtime = KDCubeSecretRuntime(
        _request(_Redis()),
        tenant="tenant-a",
        project="project-a",
        manager=manager,
    )

    await runtime.write(
        SecretTarget(
            scope=BUNDLE_SCOPE,
            bundle_id="workspace@1-0",
            key="new.token",
        ),
        value="new-value",
        caller_profile="devops-agent",
    )

    assert json.loads(manager.data[metadata_key]) == [
        "bundles.workspace@1-0.secrets.existing.token"
    ]
    assert manager.data["bundles.workspace@1-0.secrets.new.token"] == "new-value"


@pytest.mark.asyncio
async def test_platform_secret_does_not_enter_bundle_metadata() -> None:
    manager = _Manager()
    runtime = KDCubeSecretRuntime(
        _request(_Redis()),
        tenant="tenant-a",
        project="project-a",
        manager=manager,
    )
    target = SecretTarget(
        scope=PLATFORM_SCOPE,
        key="services.brave.api_key",
    )

    await runtime.write(
        target,
        value="platform-canary",
        caller_profile="devops-agent",
    )

    assert manager.data == {"services.brave.api_key": "platform-canary"}


@pytest.mark.asyncio
async def test_missing_secret_and_read_only_provider_fail_closed(
    declared_bundle,
) -> None:
    del declared_bundle
    manager = _Manager(writable=False)
    runtime = KDCubeSecretRuntime(
        _request(_Redis()),
        tenant="tenant-a",
        project="project-a",
        manager=manager,
    )
    target = SecretTarget(
        scope=BUNDLE_SCOPE,
        bundle_id="workspace@1-0",
        key="provider.api_key",
    )

    with pytest.raises(ManagementSecretNotFound):
        await runtime.read(target)
    with pytest.raises(ManagementSecretsProviderReadOnly):
        await runtime.write(
            target,
            value="must-not-write",
            caller_profile="devops-agent",
        )
    assert manager.data == {}


@pytest.mark.asyncio
async def test_provider_exception_text_is_normalized_before_orchestration() -> None:
    marker = "provider-secret-marker"

    class _ExplodingManager(_Manager):
        async def set_secret(self, key: str, value: str) -> None:
            raise RuntimeError(f"provider failed with {value}")

        async def delete_secret(self, key: str) -> None:
            raise RuntimeError(f"provider failed with {marker}")

    runtime = KDCubeSecretRuntime(
        _request(_Redis()),
        tenant="tenant-a",
        project="project-a",
        manager=_ExplodingManager(),
    )
    target = SecretTarget(
        scope=PLATFORM_SCOPE,
        key="services.brave.api_key",
    )

    with pytest.raises(ManagementSecretsProviderUnavailable) as write_error:
        await runtime.write(
            target,
            value=marker,
            caller_profile="devops-agent",
        )
    with pytest.raises(ManagementSecretsProviderUnavailable) as delete_error:
        await runtime.delete(target, caller_profile="devops-agent")

    assert marker not in str(write_error.value)
    assert marker not in str(delete_error.value)


@pytest.mark.asyncio
async def test_strict_provider_read_failure_is_not_reported_as_absence() -> None:
    class _UnavailableManager(_Manager):
        async def get_secret(self, key: str) -> str | None:
            return None

        async def get_secret_strict(self, key: str) -> str | None:
            raise RuntimeError("provider-response-must-not-escape")

    runtime = KDCubeSecretRuntime(
        _request(_Redis()),
        tenant="tenant-a",
        project="project-a",
        manager=_UnavailableManager(),
    )
    target = SecretTarget(
        scope=PLATFORM_SCOPE,
        key="services.brave.api_key",
    )

    with pytest.raises(ManagementSecretsProviderUnavailable) as captured:
        await runtime.metadata(target)

    assert str(captured.value) == (
        "The configured secrets provider could not read the secret"
    )
    assert "provider-response-must-not-escape" not in str(captured.value)


@pytest.mark.asyncio
async def test_management_read_rejects_oversized_existing_value() -> None:
    manager = _Manager()
    target = SecretTarget(
        scope=PLATFORM_SCOPE,
        key="services.fixture.api_key",
    )
    manager.data[target.provider_key] = "x" * (MAX_SECRET_VALUE_BYTES + 1)
    runtime = KDCubeSecretRuntime(
        _request(_Redis()),
        tenant="tenant-a",
        project="project-a",
        manager=manager,
    )

    assert (await runtime.metadata(target))["exists"] is True
    with pytest.raises(ManagementSecretsProviderUnavailable):
        await runtime.read(target)
