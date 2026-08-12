from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.sites.registry import (
    SiteRegistryError,
    compile_application_site_catalog,
)
from kdcube_ai_app.apps.chat.sdk.solutions.sites.runtime import (
    ApplicationSiteCatalogRuntime,
    load_application_site_catalog,
    publish_application_site_catalog,
    refresh_application_site_catalog,
    site_catalog_generation_key,
    site_catalog_key,
    site_catalog_update_channel,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.messages: list[tuple[str, str]] = []
        self.counters: dict[str, int] = {}

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str):
        self.values[key] = value

    async def publish(self, channel: str, value: str):
        self.messages.append((channel, value))

    async def incr(self, key: str):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def eval(
        self,
        _script: str,
        _numkeys: int,
        catalog_key: str,
        generation_key: str,
        channel: str,
        payload: str,
    ):
        generation = await self.incr(generation_key)
        decoded = json.loads(payload)
        decoded["generation"] = generation
        encoded = json.dumps(decoded, separators=(",", ":"), sort_keys=True)
        await self.set(catalog_key, encoded)
        await self.publish(channel, encoded)
        return encoded


@pytest.mark.asyncio
async def test_projection_is_stored_and_published() -> None:
    redis = FakeRedis()
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
    )

    projected = await publish_application_site_catalog(redis, catalog)
    restored = await load_application_site_catalog(
        redis,
        tenant="tenant-a",
        project="project-a",
    )

    key = site_catalog_key(tenant="tenant-a", project="project-a")
    channel = site_catalog_update_channel(tenant="tenant-a", project="project-a")
    generation_key = site_catalog_generation_key(tenant="tenant-a", project="project-a")
    assert restored == projected
    assert projected.generation == 1
    assert redis.counters[generation_key] == 1
    assert json.loads(redis.values[key])["revision"] == catalog.revision
    assert redis.messages == [(channel, redis.values[key])]


def test_runtime_replaces_only_changed_revision() -> None:
    runtime = ApplicationSiteCatalogRuntime()
    catalog = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
    )

    assert runtime.replace(catalog) is True
    assert runtime.replace(catalog) is False
    assert runtime.snapshot() == catalog


def test_runtime_rejects_older_generation() -> None:
    runtime = ApplicationSiteCatalogRuntime()
    older = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
        generation=1,
    )
    newer = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
        generation=2,
    )

    assert runtime.replace(newer) is True
    assert runtime.replace(older) is False
    assert runtime.snapshot() == newer


@pytest.mark.asyncio
async def test_refresh_projects_resolved_application_target(monkeypatch) -> None:
    from kdcube_ai_app.infra.plugin import bundle_store

    async def _get_bundle_props(_redis, *, tenant, project, bundle_id):
        assert (tenant, project, bundle_id) == ("tenant-a", "project-a", "site@1")
        return {
            "ui": {
                "main_view": {
                    "site": {"enabled": True, "alias": "site", "default": True},
                },
            },
        }

    monkeypatch.setattr(bundle_store, "get_bundle_props", _get_bundle_props)
    redis = FakeRedis()
    runtime = ApplicationSiteCatalogRuntime()

    catalog = await refresh_application_site_catalog(
        redis,
        tenant="tenant-a",
        project="project-a",
        applications={
            "site@1": SimpleNamespace(
                model_dump=lambda: {
                    "id": "site@1",
                    "path": "/managed/site@1",
                    "module": "entrypoint",
                    "singleton": True,
                },
            ),
        },
        runtime=runtime,
    )

    assert catalog.generation == 1
    assert catalog.sites[0].target is not None
    assert catalog.sites[0].target.path == "/managed/site@1"
    assert catalog.sites[0].target.module == "entrypoint"
    assert catalog.sites[0].target.singleton is True
    assert runtime.snapshot() == catalog


@pytest.mark.asyncio
async def test_rejected_route_prefix_conflict_preserves_last_valid_runtime(monkeypatch) -> None:
    from kdcube_ai_app.infra.plugin import bundle_store

    async def _get_bundle_props(_redis, *, tenant, project, bundle_id):
        assert (tenant, project, bundle_id) == ("tenant-a", "project-a", "site@1")
        return {
            "ui": {
                "main_view": {
                    "site": {"enabled": True, "alias": "site", "default": True},
                },
            },
        }

    monkeypatch.setattr(bundle_store, "get_bundle_props", _get_bundle_props)
    redis = FakeRedis()
    runtime = ApplicationSiteCatalogRuntime()
    last_valid = compile_application_site_catalog(
        tenant="tenant-a",
        project="project-a",
        sites=[],
        generation=7,
    )
    assert runtime.replace(last_valid) is True

    with pytest.raises(SiteRegistryError, match="proxy.route_prefix is '/'"):
        await refresh_application_site_catalog(
            redis,
            tenant="tenant-a",
            project="project-a",
            applications={
                "site@1": SimpleNamespace(
                    model_dump=lambda: {
                        "id": "site@1",
                        "path": "/managed/site@1",
                        "module": "entrypoint",
                        "singleton": True,
                    },
                ),
            },
            runtime=runtime,
            route_prefix="/",
        )

    assert runtime.snapshot() == last_valid
