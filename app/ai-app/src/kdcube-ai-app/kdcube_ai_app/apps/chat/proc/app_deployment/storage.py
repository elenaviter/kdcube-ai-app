# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import asyncio
import os
import pathlib
import time

from kdcube_ai_app.apps.chat.proc.app_deployment.models import AppStaticSurfaceManifest
from kdcube_ai_app.infra.plugin.bundle_storage import storage_for_spec

DEPLOYMENT_DIRNAME = ".kdcube.app-deployment"
MANIFEST_FILENAME = "static-widget-surfaces.v1.json"
SIGNATURE_FILENAME = "static-widget-surfaces.v1.signature"


def deployment_directory(storage_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(storage_root) / DEPLOYMENT_DIRNAME


def deployment_manifest_path(storage_root: pathlib.Path) -> pathlib.Path:
    return deployment_directory(storage_root) / MANIFEST_FILENAME


def deployment_signature_path(storage_root: pathlib.Path) -> pathlib.Path:
    return deployment_directory(storage_root) / SIGNATURE_FILENAME


def _load_deployment_manifest_sync(storage_root: pathlib.Path) -> AppStaticSurfaceManifest | None:
    path = deployment_manifest_path(storage_root)
    try:
        return AppStaticSurfaceManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


async def load_deployment_manifest(storage_root: pathlib.Path) -> AppStaticSurfaceManifest | None:
    return await asyncio.to_thread(_load_deployment_manifest_sync, storage_root)


def _write_deployment_manifest_sync(
    storage_root: pathlib.Path,
    manifest: AppStaticSurfaceManifest,
) -> pathlib.Path:
    path = deployment_manifest_path(storage_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


async def write_deployment_manifest(
    storage_root: pathlib.Path,
    manifest: AppStaticSurfaceManifest,
) -> pathlib.Path:
    return await asyncio.to_thread(_write_deployment_manifest_sync, storage_root, manifest)


def _invalidate_deployment_manifest_sync(storage_root: pathlib.Path) -> None:
    """Remove the published pointer before a live source reload.

    Built widget files remain available to the legacy path. The deployed path
    falls back until the new generation publishes its complete manifest.
    """
    for path in (
        deployment_manifest_path(storage_root),
        deployment_signature_path(storage_root),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


async def invalidate_deployment_manifest(storage_root: pathlib.Path) -> None:
    """Invalidate the current publication pointer without blocking proc."""
    await asyncio.to_thread(_invalidate_deployment_manifest_sync, storage_root)


def _deployment_manifest_ready_sync(
    storage_root: pathlib.Path,
    *,
    expected_signature: str,
) -> bool:
    manifest = _load_deployment_manifest_sync(storage_root)
    if manifest is None or manifest.deployment_signature != expected_signature:
        return False
    root = pathlib.Path(storage_root).resolve()
    for widget in manifest.widgets.values():
        if not (widget.enabled and widget.static):
            continue
        if not widget.artifact_relpath:
            return False
        try:
            artifact_root = (root / widget.artifact_relpath).resolve()
            artifact_root.relative_to(root)
        except ValueError:
            return False
        if not (artifact_root / "index.html").is_file():
            return False
    return True


async def deployment_manifest_ready(
    storage_root: pathlib.Path,
    *,
    expected_signature: str,
) -> bool:
    return await asyncio.to_thread(
        _deployment_manifest_ready_sync,
        storage_root,
        expected_signature=expected_signature,
    )


def _resolve_deployed_widget_target_sync(
    storage_root: pathlib.Path,
    *,
    artifact_relpath: str,
    widget_path: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    root = pathlib.Path(storage_root).resolve()
    artifact_root = (root / artifact_relpath).resolve()
    artifact_root.relative_to(root)
    cleaned_path = str(widget_path or "index.html").strip().lstrip("/") or "index.html"
    target = (artifact_root / cleaned_path).resolve()
    target.relative_to(artifact_root)
    if target.is_dir():
        target = (target / "index.html").resolve()
        target.relative_to(artifact_root)
    if not target.exists():
        target = (artifact_root / "index.html").resolve()
        target.relative_to(artifact_root)
    return artifact_root, target


async def resolve_deployed_widget_target(
    storage_root: pathlib.Path,
    *,
    artifact_relpath: str,
    widget_path: str,
) -> tuple[pathlib.Path, pathlib.Path]:
    return await asyncio.to_thread(
        _resolve_deployed_widget_target_sync,
        storage_root,
        artifact_relpath=artifact_relpath,
        widget_path=widget_path,
    )


async def deployed_widget_target_is_file(target: pathlib.Path) -> bool:
    return await asyncio.to_thread(target.is_file)


async def read_deployed_widget_text(target: pathlib.Path) -> str:
    return await asyncio.to_thread(target.read_text, encoding="utf-8")


async def resolve_app_storage_root(
    *,
    spec: object,
    tenant: str,
    project: str,
    ensure: bool,
) -> pathlib.Path | None:
    return await asyncio.to_thread(
        storage_for_spec,
        spec=spec,
        tenant=tenant,
        project=project,
        ensure=ensure,
    )
