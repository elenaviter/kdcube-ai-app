# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.proc.app_lifecycle import runtime
from kdcube_ai_app.apps.chat.proc.app_lifecycle.runtime import ProcApplicationLifecycle
from kdcube_ai_app.infra.plugin.app_readiness import (
    ApplicationReadinessMode,
    ApplicationReadinessRegistry,
)
from kdcube_ai_app.infra.plugin.bundle_store import BundleEntry, BundlesRegistry


def _registry(path: Path, *, readiness: str = "independent") -> BundlesRegistry:
    return BundlesRegistry(
        default_bundle_id="app@1-0",
        bundles={
            "app@1-0": BundleEntry(
                id="app@1-0",
                path=str(path),
                module="entrypoint",
                singleton=True,
                service={"readiness": readiness},
            )
        },
    )


def _patch_preparation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    started: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
) -> list[str]:
    calls: list[str] = []

    async def _props(**kwargs):
        del kwargs
        return {"feature": {"enabled": True}}

    async def _resolve(application_id, payload, **kwargs):
        del application_id, kwargs
        return payload

    async def _upsert(*args, **kwargs):
        del args, kwargs

    async def _preload(spec, bundle_spec, **kwargs):
        del bundle_spec, kwargs
        calls.append(spec.id)
        if started is not None:
            started.set()
        if release is not None:
            await release.wait()
        return object(), object()

    async def _validate(**kwargs):
        del kwargs

    async def _deploy(**kwargs):
        del kwargs

    monkeypatch.setattr(runtime, "get_bundle_props_from_authority", _props)
    monkeypatch.setattr(runtime, "resolve_git_bundle_entry_async", _resolve)
    monkeypatch.setattr(runtime, "upsert_bundles_async", _upsert)
    monkeypatch.setattr(runtime, "preload_bundle_async", _preload)
    monkeypatch.setattr(runtime, "validate_prepared_application_manifest", _validate)
    monkeypatch.setattr(runtime, "deploy_loaded_bundle_app_resources", _deploy)
    return calls


def _lifecycle(registry: ApplicationReadinessRegistry) -> ProcApplicationLifecycle:
    return ProcApplicationLifecycle(
        tenant="tenant-a",
        project="project-a",
        redis=object(),
        pg_pool=object(),
        concurrency=2,
        retry_initial_seconds=0,
        retry_max_seconds=0,
        registry=registry,
    )


@pytest.mark.asyncio
async def test_reconcile_publishes_state_without_waiting_for_application_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "app"
    source.mkdir()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = _patch_preparation(monkeypatch, started=started, release=release)
    registry = ApplicationReadinessRegistry()
    lifecycle = _lifecycle(registry)

    await lifecycle.reconcile(_registry(source))
    await asyncio.wait_for(started.wait(), timeout=1)

    snapshot = registry.snapshot(
        tenant="tenant-a",
        project="project-a",
        application_id="app@1-0",
    )
    assert snapshot is not None
    assert snapshot.state.value == "preparing"
    assert registry.aggregate(tenant="tenant-a", project="project-a").ready is True

    release.set()
    await asyncio.wait_for(lifecycle.wait_for_current(), timeout=1)
    assert registry.require_ready(
        tenant="tenant-a",
        project="project-a",
        application_id="app@1-0",
    )
    assert calls == ["app@1-0"]
    await lifecycle.shutdown()


@pytest.mark.asyncio
async def test_required_policy_blocks_aggregate_but_does_not_change_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "app"
    source.mkdir()
    calls = _patch_preparation(monkeypatch)
    registry = ApplicationReadinessRegistry()
    lifecycle = _lifecycle(registry)

    await lifecycle.reconcile(_registry(source, readiness="independent"))
    await asyncio.wait_for(lifecycle.wait_for_current(), timeout=1)
    before = registry.snapshot(
        tenant="tenant-a",
        project="project-a",
        application_id="app@1-0",
    )
    assert before is not None

    await lifecycle.reconcile(_registry(source, readiness="required"))
    after = registry.snapshot(
        tenant="tenant-a",
        project="project-a",
        application_id="app@1-0",
    )
    assert after is not None
    assert after.readiness is ApplicationReadinessMode.REQUIRED
    assert after.desired_generation == before.desired_generation
    assert after.ready is True
    assert calls == ["app@1-0"]
    await lifecycle.shutdown()


@pytest.mark.asyncio
async def test_explicit_retry_reprepares_only_the_selected_application(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "app"
    source.mkdir()
    calls = _patch_preparation(monkeypatch)
    registry = ApplicationReadinessRegistry()
    lifecycle = _lifecycle(registry)

    await lifecycle.reconcile(_registry(source))
    await asyncio.wait_for(lifecycle.wait_for_current(), timeout=1)
    await lifecycle.retry("app@1-0")
    retrying = registry.snapshot(
        tenant="tenant-a",
        project="project-a",
        application_id="app@1-0",
    )
    assert retrying is not None
    assert retrying.ready is False
    assert retrying.state.value in {"pending", "preparing"}
    await asyncio.wait_for(lifecycle.wait_for_current(), timeout=1)

    assert calls == ["app@1-0", "app@1-0"]
    await lifecycle.shutdown()
