import json
import logging
from types import SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.proc.rest.integrations import integrations


class _FakeRedis:
    def __init__(self):
        self.data = {}
        self.published = []

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value):
        self.data[key] = value

    async def delete(self, key):
        return 1 if self.data.pop(key, None) is not None else 0

    async def publish(self, channel, message):
        self.published.append((channel, message))
        return 1


class _FakeSecretsManager:
    def __init__(self):
        self.set_many_calls = []
        self.delete_many_calls = []
        self.values = {}

    def can_write(self):
        return True

    async def set_many(self, values):
        self.set_many_calls.append(dict(values))
        self.values.update(values)

    async def delete_many(self, keys):
        self.delete_many_calls.append(sorted(keys))
        for key in keys:
            self.values.pop(key, None)

    async def set_secret(self, key, value):
        self.values[key] = value

    async def delete_secret(self, key):
        self.values.pop(key, None)

    async def get_secret(self, key):
        return self.values.get(key)

    async def list_secret_keys(self, metadata_key):
        prefix = metadata_key[: -len("__keys")]
        return sorted(
            key
            for key in self.values
            if key.startswith(prefix) and not key.endswith(".__keys")
        )


class _FailingSecretsManager(_FakeSecretsManager):
    async def set_many(self, _values):
        raise integrations.SecretsManagerWriteError(
            "provider included secret-canary in its failure"
        )


def _failing_secrets_manager(_settings):
    raise integrations.SecretsManagerError(
        "provider initialization included secret-canary"
    )


def _request_with_redis(redis):
    state = SimpleNamespace(redis_async=redis)
    app = SimpleNamespace(state=state)
    return SimpleNamespace(app=app)


@pytest.mark.asyncio
async def test_set_bundle_secrets_uses_provider_derived_inventory(monkeypatch):
    redis = _FakeRedis()
    request = _request_with_redis(redis)
    session = SimpleNamespace(username="tester", user_id="user-1")
    manager = _FakeSecretsManager()

    monkeypatch.setattr(integrations, "get_settings", lambda: SimpleNamespace(TENANT="tenant-a", PROJECT="project-a"))
    monkeypatch.setattr(integrations, "get_secrets_manager", lambda _settings: manager)

    result = await integrations.set_bundle_secrets(
        "bundle@1",
        integrations.BundleSecretsUpdateRequest(
            mode="set",
            secrets={"openai": {"api_key": "sk-test"}},
        ),
        request,
        session,
    )

    assert result["mode"] == "set"
    assert result["keys"] == ["bundles.bundle@1.secrets.openai.api_key"]
    assert manager.set_many_calls == [{"bundles.bundle@1.secrets.openai.api_key": "sk-test"}]
    assert "bundles.bundle@1.secrets.__keys" not in manager.values
    assert result["inventory_state"] == "current"
    assert json.loads(
        redis.data["kdcube:config:bundles:secrets:tenant-a:project-a:bundle@1"]
    ) == ["bundles.bundle@1.secrets.openai.api_key"]
    channel, message = redis.published[-1]
    payload = json.loads(message)
    assert channel == "kdcube:config:bundles:secrets:update:tenant-a:project-a"
    assert payload["type"] == "bundles.secrets.update"
    assert payload["scope"] == "bundle"
    assert payload["mode"] == "set"
    assert payload["bundle_id"] == "bundle@1"
    assert payload["tenant"] == "tenant-a"
    assert payload["project"] == "project-a"
    assert payload["keys"] == ["bundles.bundle@1.secrets.openai.api_key"]
    assert "sk-test" not in message

    result = await integrations.set_bundle_secrets(
        "bundle@1",
        integrations.BundleSecretsUpdateRequest(
            mode="clear",
            secrets={"openai": {"api_key": None}},
        ),
        request,
        session,
    )

    assert result["mode"] == "clear"
    assert manager.delete_many_calls == [["bundles.bundle@1.secrets.openai.api_key"]]
    assert "bundles.bundle@1.secrets.__keys" not in manager.values
    assert json.loads(
        redis.data["kdcube:config:bundles:secrets:tenant-a:project-a:bundle@1"]
    ) == []
    channel, message = redis.published[-1]
    payload = json.loads(message)
    assert channel == "kdcube:config:bundles:secrets:update:tenant-a:project-a"
    assert payload["scope"] == "bundle"
    assert payload["mode"] == "clear"
    assert payload["keys"] == ["bundles.bundle@1.secrets.openai.api_key"]
    assert "sk-test" not in message


@pytest.mark.asyncio
async def test_set_current_user_bundle_secrets_uses_current_user_scope(monkeypatch):
    redis = _FakeRedis()
    request = _request_with_redis(redis)
    session = SimpleNamespace(username="tester", user_id="user-1")
    manager = _FakeSecretsManager()

    monkeypatch.setattr(integrations, "get_settings", lambda: SimpleNamespace(TENANT="tenant-a", PROJECT="project-a"))
    monkeypatch.setattr(integrations, "get_secrets_manager", lambda _settings: manager)

    result = await integrations.set_current_user_bundle_secrets(
        "tenant-a",
        "project-a",
        "bundle@1",
        integrations.UserBundleSecretsUpdateRequest(
            mode="set",
            secrets={"anthropic": {"api_key": "sk-user"}},
        ),
        request,
        session,
    )

    expected_key = "users.user-1.bundles.bundle@1.secrets.anthropic.api_key"
    expected_meta = "users.user-1.bundles.bundle@1.secrets.__keys"
    assert result["mode"] == "set"
    assert result["inventory_state"] == "current"
    assert manager.set_many_calls == [{expected_key: "sk-user"}]
    assert expected_meta not in manager.values
    assert json.loads(
        redis.data["kdcube:config:bundles:user-secrets:tenant-a:project-a:bundle@1:user-1"]
    ) == [expected_key]
    channel, message = redis.published[-1]
    payload = json.loads(message)
    assert channel == "kdcube:config:bundles:secrets:update:tenant-a:project-a"
    assert payload["type"] == "bundles.secrets.update"
    assert payload["scope"] == "user"
    assert payload["mode"] == "set"
    assert payload["bundle_id"] == "bundle@1"
    assert payload["tenant"] == "tenant-a"
    assert payload["project"] == "project-a"
    assert payload["user_id"] == "user-1"
    assert payload["keys"] == [expected_key]
    assert "sk-user" not in message
    assert "keys" not in result
    assert "stored_keys" not in result


@pytest.mark.asyncio
async def test_bundle_secret_write_failure_is_secret_safe(
    monkeypatch,
    caplog,
):
    marker = "secret-canary"
    manager = _FailingSecretsManager()
    monkeypatch.setattr(
        integrations,
        "get_settings",
        lambda: SimpleNamespace(TENANT="tenant-a", PROJECT="project-a"),
    )
    monkeypatch.setattr(
        integrations,
        "get_secrets_manager",
        lambda _settings: manager,
    )
    caplog.set_level(logging.ERROR)

    with pytest.raises(integrations.HTTPException) as raised:
        await integrations.set_bundle_secrets(
            "bundle@1",
            integrations.BundleSecretsUpdateRequest(
                mode="set",
                secrets={"provider": {"api_key": marker}},
            ),
            _request_with_redis(_FakeRedis()),
            SimpleNamespace(username="tester", user_id="user-1"),
        )

    assert raised.value.status_code == 502
    assert raised.value.detail == "Failed to store secrets"
    assert marker not in str(raised.value)
    assert marker not in caplog.text


@pytest.mark.asyncio
async def test_user_secret_write_failure_is_secret_safe(
    monkeypatch,
    caplog,
):
    marker = "secret-canary"
    manager = _FailingSecretsManager()
    monkeypatch.setattr(
        integrations,
        "get_settings",
        lambda: SimpleNamespace(TENANT="tenant-a", PROJECT="project-a"),
    )
    monkeypatch.setattr(
        integrations,
        "get_secrets_manager",
        lambda _settings: manager,
    )
    caplog.set_level(logging.ERROR)

    with pytest.raises(integrations.HTTPException) as raised:
        await integrations.set_current_user_bundle_secrets(
            "tenant-a",
            "project-a",
            "bundle@1",
            integrations.UserBundleSecretsUpdateRequest(
                mode="set",
                secrets={"provider": {"api_key": marker}},
            ),
            _request_with_redis(_FakeRedis()),
            SimpleNamespace(username="tester", user_id="user-1"),
        )

    assert raised.value.status_code == 502
    assert raised.value.detail == "Failed to store user secrets"
    assert marker not in str(raised.value)
    assert marker not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["bundle", "user"])
async def test_secret_provider_initialization_failure_is_secret_safe(
    scope,
    monkeypatch,
    caplog,
):
    marker = "secret-canary"
    monkeypatch.setattr(
        integrations,
        "get_settings",
        lambda: SimpleNamespace(TENANT="tenant-a", PROJECT="project-a"),
    )
    monkeypatch.setattr(
        integrations,
        "get_secrets_manager",
        _failing_secrets_manager,
    )
    caplog.set_level(logging.ERROR)

    with pytest.raises(integrations.HTTPException) as raised:
        if scope == "bundle":
            await integrations.set_bundle_secrets(
                "bundle@1",
                integrations.BundleSecretsUpdateRequest(
                    mode="set",
                    secrets={"provider": {"api_key": "value"}},
                ),
                _request_with_redis(_FakeRedis()),
                SimpleNamespace(username="tester", user_id="user-1"),
            )
        else:
            await integrations.set_current_user_bundle_secrets(
                "tenant-a",
                "project-a",
                "bundle@1",
                integrations.UserBundleSecretsUpdateRequest(
                    mode="set",
                    secrets={"provider": {"api_key": "value"}},
                ),
                _request_with_redis(_FakeRedis()),
                SimpleNamespace(username="tester", user_id="user-1"),
            )

    assert raised.value.status_code == 503
    assert raised.value.detail == "Secrets provider is unavailable"
    assert marker not in str(raised.value)
    assert marker not in caplog.text
