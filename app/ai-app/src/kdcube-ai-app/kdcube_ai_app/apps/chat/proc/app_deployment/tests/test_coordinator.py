# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.proc.app_deployment import coordinator
from kdcube_ai_app.apps.chat.proc.app_deployment.coordinator import (
    deploy_loaded_bundle_static_surfaces,
)
from kdcube_ai_app.infra.plugin.bundle_loader import (
    BundleInterfaceManifest,
    BundleSpec,
    UIWidgetSpec,
)


class _Workflow:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.bundle_props_defaults = {}
        self.bundle_props = {}
        self.config = SimpleNamespace(log_level="INFO")
        self.deploy_calls = 0
        self.build_calls = 0

    async def on_app_deploy(self, **kwargs) -> None:
        assert kwargs["tenant"] == "tenant-a"
        assert kwargs["project"] == "project-a"
        self.deploy_calls += 1

    async def _ensure_ui_build(self) -> None:
        self.build_calls += 1
        root = self.output_root / "ui" / "widgets" / "stats"
        root.mkdir(parents=True, exist_ok=True)
        (root / "index.html").write_text("<html>stats</html>", encoding="utf-8")

    def compute_ui_widget_signature(self, alias: str) -> str:
        return f"widget-source:{alias}"


@pytest.mark.asyncio
async def test_deployment_runs_once_and_persists_resolved_policy(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "entrypoint.py").write_text("APP = True\n", encoding="utf-8")
    storage_root = tmp_path / "storage"
    props = {
        "ui": {
            "widgets": {
                "stats": {
                    "src_folder": "ui/widgets/stats/src",
                    "build_command": "npm run build",
                }
            }
        },
        "surfaces": {
            "as_provider": {
                "widget": {
                    "stats": {
                        "visibility": {"user_types": ["registered"], "roles": ["finance"]},
                        "auth": {
                            "authority_id": "kdcube.platform",
                            "grants": ["stats:read"],
                            "provider_config": "must-not-enter-static-manifest",
                        },
                    }
                }
            }
        },
    }

    async def _descriptor_props(**kwargs):
        del kwargs
        return props

    async def _storage_root(**kwargs):
        del kwargs
        return storage_root

    monkeypatch.setattr(coordinator, "static_widget_deployment_enabled", lambda: True)
    monkeypatch.setattr(coordinator, "get_bundle_props_from_authority", _descriptor_props)
    monkeypatch.setattr(coordinator, "resolve_app_storage_root", _storage_root)
    monkeypatch.setattr(
        coordinator,
        "get_settings",
        lambda: SimpleNamespace(
            PLATFORM=SimpleNamespace(
                APPLICATIONS=SimpleNamespace(
                    BUNDLES_PRELOAD_LOCK_TTL_SECONDS=30,
                    BUNDLES_PRELOAD_BUNDLE_LOCK_TTL_SECONDS=30,
                )
            )
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "discover_bundle_interface_manifest",
        lambda workflow, bundle_id: BundleInterfaceManifest(
            bundle_id=bundle_id,
            ui_widgets=(UIWidgetSpec(method_name="stats_widget", alias="stats", icon={}),),
        ),
    )

    workflow = _Workflow(storage_root)
    bundle_spec = SimpleNamespace(
        id="app@1-0",
        path=str(source_root),
        module="entrypoint",
        singleton=True,
        repo=None,
        ref=None,
        subdir=None,
        git_commit="commit-1",
    )
    agentic_spec = BundleSpec(
        id="app@1-0",
        path=str(source_root),
        module="entrypoint",
        singleton=True,
    )

    first = await deploy_loaded_bundle_static_surfaces(
        workflow=workflow,
        module=SimpleNamespace(),
        agentic_spec=agentic_spec,
        bundle_spec=bundle_spec,
        tenant="tenant-a",
        project="project-a",
    )
    second = await deploy_loaded_bundle_static_surfaces(
        workflow=workflow,
        module=SimpleNamespace(),
        agentic_spec=agentic_spec,
        bundle_spec=bundle_spec,
        tenant="tenant-a",
        project="project-a",
    )

    assert first is not None and second is not None
    assert first.deployment_signature == second.deployment_signature
    assert workflow.deploy_calls == 1
    assert workflow.build_calls == 1
    assert first.widgets["stats"].roles == ["finance"]
    assert first.widgets["stats"].user_types == ["registered"]
    assert first.widgets["stats"].auth == {
        "authority_id": "kdcube.platform",
        "grants": ["stats:read"],
    }

    props["surfaces"]["as_provider"]["widget"]["stats"]["visibility"]["roles"] = ["admin"]
    third = await deploy_loaded_bundle_static_surfaces(
        workflow=workflow,
        module=SimpleNamespace(),
        agentic_spec=agentic_spec,
        bundle_spec=bundle_spec,
        tenant="tenant-a",
        project="project-a",
    )
    assert third is not None
    assert third.widgets["stats"].roles == ["admin"]
    assert workflow.deploy_calls == 2
    assert workflow.build_calls == 2
