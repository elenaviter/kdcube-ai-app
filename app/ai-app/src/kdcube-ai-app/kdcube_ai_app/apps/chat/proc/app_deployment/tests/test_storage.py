# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.proc.app_deployment.models import (
    AppStaticSurfaceManifest,
    DeployedWidgetSurface,
)
from kdcube_ai_app.apps.chat.proc.app_deployment.storage import (
    deployment_manifest_ready,
    invalidate_deployment_manifest,
    load_deployment_manifest,
    resolve_deployed_widget_target,
    write_deployment_manifest,
)


def _manifest() -> AppStaticSurfaceManifest:
    return AppStaticSurfaceManifest(
        tenant="tenant-a",
        project="project-a",
        bundle_id="app@1-0",
        source_generation="source-1",
        props_fingerprint="props-1",
        deployment_signature="deploy-1",
        generated_at="2026-07-27T00:00:00+00:00",
        widgets={
            "stats": DeployedWidgetSurface(
                alias="stats",
                method_name="stats_widget",
                static=True,
                artifact_relpath="ui/widgets/stats",
            )
        },
    )


@pytest.mark.asyncio
async def test_manifest_is_atomic_and_requires_built_entrypoint(tmp_path: Path) -> None:
    manifest = _manifest()
    await write_deployment_manifest(tmp_path, manifest)
    assert await load_deployment_manifest(tmp_path) == manifest
    assert await deployment_manifest_ready(tmp_path, expected_signature="deploy-1") is False

    widget_root = tmp_path / "ui" / "widgets" / "stats"
    widget_root.mkdir(parents=True)
    (widget_root / "index.html").write_text("<html></html>", encoding="utf-8")
    assert await deployment_manifest_ready(tmp_path, expected_signature="deploy-1") is True
    assert await deployment_manifest_ready(tmp_path, expected_signature="other") is False

    await invalidate_deployment_manifest(tmp_path)
    assert await load_deployment_manifest(tmp_path) is None


@pytest.mark.asyncio
async def test_widget_target_stays_inside_published_artifact(tmp_path: Path) -> None:
    widget_root = tmp_path / "ui" / "widgets" / "stats"
    widget_root.mkdir(parents=True)
    (widget_root / "index.html").write_text("<html></html>", encoding="utf-8")
    resolved_root, target = await resolve_deployed_widget_target(
        tmp_path,
        artifact_relpath="ui/widgets/stats",
        widget_path="missing.js",
    )
    assert resolved_root == widget_root.resolve()
    assert target == (widget_root / "index.html").resolve()

    with pytest.raises(ValueError):
        await resolve_deployed_widget_target(
            tmp_path,
            artifact_relpath="ui/widgets/stats",
            widget_path="../../../../secrets.yaml",
        )
