# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
import dotenv

from kdcube_ai_app.apps.chat.sdk import config as sdk_config

os.environ["KDCUBE_STORAGE_PATH"] = "/tmp/kdcube-test-storage"
os.environ["ENABLE_DATABASE"] = "false"
dotenv.load_dotenv = lambda *args, **kwargs: False
dotenv.find_dotenv = lambda *args, **kwargs: ""
sdk_config.get_settings.cache_clear()

from kdcube_ai_app.apps.chat.proc import web_app
from kdcube_ai_app.infra.plugin import bundle_loader, bundle_store


class _FakeRedis:
    def __init__(self, *, acquire: bool, acquire_bundle: bool | None = None) -> None:
        self.acquire = acquire
        self.acquire_bundle = acquire_bundle
        self.calls: list[tuple] = []
        self.values: dict[str, str] = {}

    async def set(self, key, value, ex=None, nx=None):
        self.calls.append(("set", key, value, ex, nx))
        if not nx:
            self.values[key] = value
            return True
        is_bundle_preload_lock = str(key).count(":") > 5
        acquire = self.acquire_bundle if is_bundle_preload_lock and self.acquire_bundle is not None else self.acquire
        if not acquire:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        self.calls.append(("get", key))
        return self.values.get(key)

    async def delete(self, key):
        self.calls.append(("delete", key))
        self.values.pop(key, None)
        return 1

    async def expire(self, key, seconds):
        self.calls.append(("expire", key, seconds))
        return 1 if key in self.values else 0


def _app(redis: _FakeRedis) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            bundle_git_task=None,
            redis_async=redis,
            pg_pool=object(),
            bundles_preload_ready=False,
            bundles_preload_errors=None,
        )
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        TENANT="tenant-a",
        PROJECT="project-a",
        BUNDLES_PRELOAD_LOCK_TTL_SECONDS=45,
        PLATFORM=SimpleNamespace(
            APPLICATIONS=SimpleNamespace(
                BUNDLES_PRELOAD_LOCK_TTL_SECONDS=45,
                BUNDLES_PRELOAD_BUNDLE_LOCK_TTL_SECONDS=30,
            )
        ),
    )


def _registry(items: dict) -> bundle_store.BundlesRegistry:
    return bundle_store.BundlesRegistry(
        default_bundle_id=next(iter(items.keys()), None),
        bundles={
            bid: bundle_store.BundleEntry(id=bid, **entry)
            for bid, entry in items.items()
        },
    )


def _manifest(*, widgets: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        ui_widgets=tuple(SimpleNamespace(alias=alias) for alias in widgets),
        api_endpoints=(),
        mcp_endpoints=(),
        ui_main=None,
        on_message=None,
        on_job=None,
        scheduled_jobs=(),
    )


def test_static_preload_generation_tracks_source_runtime_and_props() -> None:
    entry = bundle_store.BundleEntry(
        id="bundle.demo",
        path="/tmp/demo",
        module="entrypoint",
        singleton=False,
    )
    base = web_app._bundle_preload_generation(
        "bundle.demo",
        entry,
        static_source_fingerprint="source-a",
        static_runtime_generation="runtime-a",
        static_props_fingerprint="props-a",
    )
    assert base != web_app._bundle_preload_generation(
        "bundle.demo",
        entry,
        static_source_fingerprint="source-b",
        static_runtime_generation="runtime-a",
        static_props_fingerprint="props-a",
    )
    assert base != web_app._bundle_preload_generation(
        "bundle.demo",
        entry,
        static_source_fingerprint="source-a",
        static_runtime_generation="runtime-b",
        static_props_fingerprint="props-a",
    )
    assert base != web_app._bundle_preload_generation(
        "bundle.demo",
        entry,
        static_source_fingerprint="source-a",
        static_runtime_generation="runtime-a",
        static_props_fingerprint="props-b",
    )


@pytest.fixture(autouse=True)
def _no_authoritative_props(monkeypatch):
    async def _props(**kwargs):
        del kwargs
        return {}

    monkeypatch.setattr(web_app, "get_bundle_props_from_authority", _props)


@pytest.mark.asyncio
async def test_preload_bundles_loop_acquires_and_releases_leader_lock(monkeypatch):
    redis = _FakeRedis(acquire=True)
    app = _app(redis)
    preload_calls: list[tuple[str, str]] = []

    async def _fake_preload(spec, bundle_spec, **kwargs):
        preload_calls.append((spec.path, bundle_spec.id))
        assert kwargs["tenant"] == "tenant-a"
        assert kwargs["project"] == "project-a"

    monkeypatch.setattr(web_app, "get_settings", _settings)
    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "module": "entrypoint", "singleton": False}})

    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(bundle_loader, "load_bundle_manifest", lambda *args, **kwargs: _manifest())

    await web_app._preload_bundles_loop(app)

    key = web_app.CONFIG.BUNDLES.PRELOAD_LOCK_FMT.format(tenant="tenant-a", project="project-a")
    assert preload_calls == [("/tmp/demo", "bundle.demo")]
    assert app.state.bundles_preload_ready is True
    assert app.state.bundles_preload_errors == {}
    assert app.state.bundles_preload_status["bundle.demo"]["status"] == "succeeded"
    assert app.state.bundles_preload_status["bundle.demo"]["duration_ms"] >= 0
    assert redis.calls[0] == ("set", key, redis.calls[0][2], 45, True)
    assert ("delete", key) in redis.calls


@pytest.mark.asyncio
async def test_preload_bundles_loop_still_preloads_when_another_instance_holds_lock(monkeypatch):
    redis = _FakeRedis(acquire=False, acquire_bundle=True)
    app = _app(redis)
    preload_calls: list[str] = []

    async def _fake_preload(spec, bundle_spec, **kwargs):
        del spec, bundle_spec, kwargs
        preload_calls.append("called")

    monkeypatch.setattr(web_app, "get_settings", _settings)
    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "module": "entrypoint", "singleton": False}})

    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(bundle_loader, "load_bundle_manifest", lambda *args, **kwargs: _manifest())

    await web_app._preload_bundles_loop(app)

    key = web_app.CONFIG.BUNDLES.PRELOAD_LOCK_FMT.format(tenant="tenant-a", project="project-a")
    assert preload_calls == ["called"]
    assert app.state.bundles_preload_ready is True
    assert app.state.bundles_preload_errors == {}
    assert redis.calls[0] == ("set", key, redis.calls[0][2], 45, True)
    assert any(call[0] == "delete" and str(call[1]).endswith(":bundle.demo") for call in redis.calls)


@pytest.mark.asyncio
async def test_preload_bundles_loop_skips_bundle_when_bundle_lock_is_held(monkeypatch):
    redis = _FakeRedis(acquire=True, acquire_bundle=False)
    app = _app(redis)
    preload_calls: list[str] = []

    async def _fake_preload(spec, bundle_spec, **kwargs):
        del spec, kwargs
        preload_calls.append(bundle_spec.id)

    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({
            "bundle.a": {"path": "/tmp/a", "module": "entrypoint", "singleton": False},
            "bundle.b": {"path": "/tmp/b", "module": "entrypoint", "singleton": False},
        })

    monkeypatch.setattr(web_app, "get_settings", _settings)
    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(bundle_loader, "load_bundle_manifest", lambda *args, **kwargs: _manifest())

    await web_app._preload_bundles_loop(app)

    assert preload_calls == []
    assert app.state.bundles_preload_ready is True
    assert app.state.bundles_preload_errors == {}
    assert app.state.bundles_preload_skipped_locked == 2
    assert app.state.bundles_preload_skipped_claimed == 2
    assert app.state.bundles_preload_status["bundle.a"]["status"] == "skipped_claimed"
    assert app.state.bundles_preload_status["bundle.b"]["status"] == "skipped_claimed"


@pytest.mark.asyncio
async def test_preload_bundles_loop_skips_bundle_when_generation_done(monkeypatch):
    redis = _FakeRedis(acquire=True)
    app = _app(redis)
    preload_calls: list[str] = []
    entry = bundle_store.BundleEntry(id="bundle.demo", path="/tmp/demo", module="entrypoint", singleton=False)
    generation = web_app._bundle_preload_generation("bundle.demo", entry)
    done_key = web_app.CONFIG.BUNDLES.PRELOAD_BUNDLE_DONE_FMT.format(
        tenant="tenant-a",
        project="project-a",
        bundle_id="bundle.demo",
        generation=generation,
    )
    redis.values[done_key] = generation

    async def _fake_preload(spec, bundle_spec, **kwargs):
        del spec, kwargs
        preload_calls.append(bundle_spec.id)

    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return bundle_store.BundlesRegistry(
            default_bundle_id="bundle.demo",
            bundles={"bundle.demo": entry},
        )

    monkeypatch.setattr(web_app, "get_settings", _settings)
    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(bundle_loader, "load_bundle_manifest", lambda *args, **kwargs: _manifest())

    await web_app._preload_bundles_loop(app)

    assert preload_calls == []
    assert app.state.bundles_preload_ready is True
    assert app.state.bundles_preload_errors == {}
    assert app.state.bundles_preload_skipped_done == 1
    assert app.state.bundles_preload_status["bundle.demo"]["status"] == "skipped_done"


@pytest.mark.asyncio
async def test_preload_bundles_loop_heartbeats_long_bundle_claim(monkeypatch):
    redis = _FakeRedis(acquire=True)
    app = _app(redis)

    async def _fake_preload(spec, bundle_spec, **kwargs):
        del spec, bundle_spec, kwargs
        await asyncio.sleep(0.45)

    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "module": "entrypoint", "singleton": False}})

    monkeypatch.setattr(web_app, "get_settings", _settings)
    monkeypatch.setattr(web_app, "_bundle_preload_lock_ttl_seconds", lambda: 1)
    monkeypatch.setattr(web_app, "_bundle_preload_bundle_lock_ttl_seconds", lambda: 1)
    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(bundle_loader, "load_bundle_manifest", lambda *args, **kwargs: _manifest())

    await web_app._preload_bundles_loop(app)

    bundle_lock_key = web_app.CONFIG.BUNDLES.PRELOAD_BUNDLE_LOCK_FMT.format(
        tenant="tenant-a",
        project="project-a",
        bundle_id="bundle.demo",
    )
    assert any(call == ("expire", bundle_lock_key, 1) for call in redis.calls)
    assert app.state.bundles_preload_status["bundle.demo"]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_preload_bundles_loop_publishes_static_deployment_in_shadow_mode(monkeypatch):
    redis = _FakeRedis(acquire=True)
    app = _app(redis)
    workflow = object()
    module = object()
    deployments: list[dict] = []

    async def _fake_preload(spec, bundle_spec, **kwargs):
        del spec, bundle_spec, kwargs
        return workflow, module

    async def _fake_deploy(**kwargs):
        deployments.append(kwargs)
        return SimpleNamespace(deployment_signature="deployment-signature")

    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "module": "entrypoint", "singleton": False}})

    monkeypatch.setattr(web_app, "get_settings", _settings)
    monkeypatch.setattr(web_app, "static_widget_deployment_enabled", lambda settings=None: True)
    monkeypatch.setattr(web_app, "static_widget_runtime_generation", lambda: "runtime-a")

    async def _source_fingerprint(spec):
        del spec
        return "source-a"

    monkeypatch.setattr(web_app, "app_source_fingerprint", _source_fingerprint)
    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(web_app, "deploy_loaded_bundle_static_surfaces", _fake_deploy)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(bundle_loader, "load_bundle_manifest", lambda *args, **kwargs: _manifest())

    await web_app._preload_bundles_loop(app)

    assert len(deployments) == 1
    assert deployments[0]["workflow"] is workflow
    assert deployments[0]["module"] is module
    assert deployments[0]["tenant"] == "tenant-a"
    assert deployments[0]["project"] == "project-a"
    assert app.state.bundles_preload_status["bundle.demo"]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_preload_bundles_loop_reports_static_widget_without_decorator(monkeypatch):
    redis = _FakeRedis(acquire=True)
    app = _app(redis)
    preload_calls: list[str] = []
    evict_calls: list[str] = []

    async def _fake_preload(spec, bundle_spec, **kwargs):
        del spec, kwargs
        preload_calls.append(bundle_spec.id)

    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "module": "entrypoint", "singleton": False}})

    async def _props(**kwargs):
        del kwargs
        return {"ui": {"widgets": {"copilot_webapp": {"enabled": True}}}}

    def _evict(spec, **kwargs):
        del kwargs
        evict_calls.append(spec.path)
        return {"evicted_modules": 0, "evicted_singletons": 0, "evicted_manifests": 0, "sys_modules_deleted": 0}

    monkeypatch.setattr(web_app, "get_settings", _settings)
    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(web_app, "get_bundle_props_from_authority", _props)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(bundle_loader, "load_bundle_manifest", lambda *args, **kwargs: _manifest())
    monkeypatch.setattr(bundle_loader, "evict_bundle_scope", _evict)

    await web_app._preload_bundles_loop(app)

    assert preload_calls == ["bundle.demo"]
    assert evict_calls == ["/tmp/demo"]
    assert app.state.bundles_preload_ready is True
    assert "bundle.demo" in app.state.bundles_preload_errors
    assert "copilot_webapp" in app.state.bundles_preload_errors["bundle.demo"]


@pytest.mark.asyncio
async def test_preload_bundles_loop_accepts_static_widget_backed_by_decorator(monkeypatch):
    redis = _FakeRedis(acquire=True)
    app = _app(redis)

    async def _fake_preload(spec, bundle_spec, **kwargs):
        del spec, bundle_spec, kwargs

    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "module": "entrypoint", "singleton": False}})

    async def _props(**kwargs):
        del kwargs
        return {"ui": {"widgets": {"copilot_webapp": {"enabled": True}}}}

    monkeypatch.setattr(web_app, "get_settings", _settings)
    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)
    monkeypatch.setattr(web_app, "get_bundle_props_from_authority", _props)
    monkeypatch.setattr(bundle_loader, "preload_bundle_async", _fake_preload)
    monkeypatch.setattr(
        bundle_loader,
        "load_bundle_manifest",
        lambda *args, **kwargs: _manifest(widgets=("copilot_webapp",)),
    )

    await web_app._preload_bundles_loop(app)

    assert app.state.bundles_preload_ready is True
    assert app.state.bundles_preload_errors == {}


@pytest.mark.asyncio
async def test_initial_git_bundle_prefetch_marks_ready_after_success(monkeypatch):
    app = _app(_FakeRedis(acquire=True))

    monkeypatch.setattr(web_app, "_git_prefetch_enabled", lambda: True)
    monkeypatch.setattr(web_app, "_git_resolution_enabled", lambda: True)
    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "repo": "https://example.invalid/repo.git"}})

    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)

    async def _fake_prefetch(app_obj, registry=None):
        assert "bundle.demo" in (registry.bundles or {})
        app_obj.state.bundle_git_ready = True
        app_obj.state.bundle_git_errors = {}

    monkeypatch.setattr(web_app, "_prefetch_git_bundles_loop", _fake_prefetch)

    await web_app._initial_git_bundle_prefetch(app)

    assert app.state.bundle_git_ready is True
    assert app.state.bundle_git_errors == {}
    assert app.state.bundle_git_task is None


@pytest.mark.asyncio
async def test_initial_git_bundle_prefetch_preserves_prefetch_errors(monkeypatch):
    app = _app(_FakeRedis(acquire=True))

    monkeypatch.setattr(web_app, "_git_prefetch_enabled", lambda: True)
    monkeypatch.setattr(web_app, "_git_resolution_enabled", lambda: True)
    async def _load_runtime_registry(redis_arg, tenant, project):
        del redis_arg, tenant, project
        return _registry({"bundle.demo": {"path": "/tmp/demo", "repo": "https://example.invalid/repo.git"}})

    monkeypatch.setattr(web_app, "load_bundle_runtime_registry", _load_runtime_registry)

    async def _fake_prefetch(app_obj, registry=None):
        assert "bundle.demo" in (registry.bundles or {})
        app_obj.state.bundle_git_ready = False
        app_obj.state.bundle_git_errors = {"bundle.demo": "boom"}

    monkeypatch.setattr(web_app, "_prefetch_git_bundles_loop", _fake_prefetch)

    await web_app._initial_git_bundle_prefetch(app)

    assert app.state.bundle_git_ready is False
    assert app.state.bundle_git_errors == {"bundle.demo": "boom"}
    assert app.state.bundle_git_task is None
