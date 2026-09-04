# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Bounded adapter from public delegated management to KDCube internals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from fastapi import Request
from kdcube_ai_app.apps.chat.proc.rest.management.service import (
    ManagementApplicationNotFound,
    ManagementRuntimeUnavailable,
)
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.auth.sessions import UserSession, UserType
from kdcube_ai_app.infra.plugin.app_readiness import application_readiness_registry
from kdcube_ai_app.infra.plugin.bundle_loader import BundleSpec, load_bundle_manifest
from kdcube_ai_app.infra.plugin.bundle_store import load_registry


def _segment(value: str) -> str:
    return quote(str(value or "").strip(), safe="-._~@")


def _base_path(*, tenant: str, project: str, application_id: str) -> str:
    return (
        "/api/integrations/bundles/"
        f"{_segment(tenant)}/{_segment(project)}/{_segment(application_id)}"
    )


class KDCubeManagementRuntime:
    def __init__(self, request: Request, *, tenant: str, project: str) -> None:
        self._request = request
        self._tenant = tenant
        self._project = project

    @property
    def _redis(self) -> Any:
        redis = getattr(self._request.app.state, "redis_async", None)
        if redis is None:
            raise ManagementRuntimeUnavailable("Redis is unavailable")
        return redis

    async def _registry(self) -> Any:
        try:
            return await load_registry(self._redis, self._tenant, self._project)
        except Exception as exc:
            raise ManagementRuntimeUnavailable("Application registry is unavailable") from exc

    async def inspect_deployment(self) -> Mapping[str, Any]:
        registry = await self._registry()
        aggregate = application_readiness_registry.aggregate(
            tenant=self._tenant,
            project=self._project,
        )
        applications: list[dict[str, Any]] = []
        for application_id, entry in sorted(registry.bundles.items()):
            snapshot = application_readiness_registry.snapshot(
                tenant=self._tenant,
                project=self._project,
                application_id=application_id,
            )
            service = getattr(entry, "service", None)
            required = getattr(service, "readiness", "independent") == "required"
            applications.append(
                {
                    "application_id": application_id,
                    "declared": True,
                    "preparation_state": (
                        snapshot.state.value if snapshot is not None else "unknown"
                    ),
                    "generation": (
                        snapshot.desired_generation if snapshot is not None else ""
                    ),
                    "readiness_required": required,
                }
            )
        settings = get_settings()
        return {
            "platform_release": str(settings.plain("platform.ref", default="") or ""),
            "readiness": "ready" if aggregate.ready else "not_ready",
            "applications": applications,
        }

    async def application_surfaces(self, application_id: str) -> Mapping[str, Any]:
        registry = await self._registry()
        entry = registry.bundles.get(application_id)
        if entry is None:
            raise ManagementApplicationNotFound(application_id)

        spec = BundleSpec(
            id=application_id,
            path=entry.path,
            module=entry.module,
            singleton=bool(entry.singleton),
        )
        try:
            manifest = load_bundle_manifest(spec, bundle_id=application_id)
            from kdcube_ai_app.apps.chat.proc.app_deployment.policy import (
                is_widget_enabled,
            )
            from kdcube_ai_app.apps.chat.proc.rest.integrations.integrations import (
                _authoritative_bundle_props,
                is_api_enabled,
                is_mcp_enabled,
            )
            from kdcube_ai_app.infra.plugin.bundle_loader import (
                apply_api_overrides,
                apply_mcp_overrides,
                apply_widget_overrides,
            )

            props = _authoritative_bundle_props(
                tenant=self._tenant,
                project=self._project,
                bundle_id=application_id,
            )
        except Exception as exc:
            raise ManagementRuntimeUnavailable(
                "Application surfaces could not be discovered"
            ) from exc

        base = _base_path(
            tenant=self._tenant,
            project=self._project,
            application_id=application_id,
        )
        apis = []
        for declared in manifest.api_endpoints:
            effective = apply_api_overrides(declared, props)
            if not is_api_enabled(props, effective):
                continue
            route = "public" if effective.route == "public" else "operations"
            apis.append(
                {
                    "alias": effective.alias,
                    "method": effective.http_method,
                    "path": f"{base}/{route}/{_segment(effective.alias)}",
                }
            )
        mcp = []
        for declared in manifest.mcp_endpoints:
            effective = apply_mcp_overrides(declared, props)
            if not is_mcp_enabled(props, effective):
                continue
            prefix = "public/mcp" if effective.route == "public" else "mcp"
            mcp.append(
                {
                    "alias": effective.alias,
                    "transport": effective.transport,
                    "path": f"{base}/{prefix}/{_segment(effective.alias)}",
                }
            )
        widgets = []
        for declared in manifest.ui_widgets:
            effective = apply_widget_overrides(declared, props)
            if not is_widget_enabled(props, effective):
                continue
            widgets.append(
                {
                    "alias": effective.alias,
                    "path": f"{base}/widgets/{_segment(effective.alias)}",
                }
            )
        jobs = [
            {"alias": item.alias or item.method_name}
            for item in manifest.scheduled_jobs
        ]
        if manifest.on_job is not None:
            jobs.append({"alias": "on_job"})
        messaging = []
        if manifest.on_message is not None:
            messaging.append({"kind": "on_message"})
        messaging.extend(
            {"kind": "data_bus", "subject": item.subject}
            for item in manifest.data_bus_handlers
        )
        return {
            "application_id": application_id,
            "surfaces": {
                "api": sorted(apis, key=lambda item: (item["alias"], item["method"])),
                "mcp": sorted(mcp, key=lambda item: item["alias"]),
                "widgets": sorted(widgets, key=lambda item: item["alias"]),
                "jobs": sorted(jobs, key=lambda item: item["alias"]),
                "messaging": sorted(
                    messaging,
                    key=lambda item: (item["kind"], str(item.get("subject") or "")),
                ),
            },
        }

    async def reload_application(
        self, application_id: str, *, caller_profile: str
    ) -> Mapping[str, Any]:
        registry = await self._registry()
        if application_id not in registry.bundles:
            raise ManagementApplicationNotFound(application_id)
        from kdcube_ai_app.apps.chat.proc.rest.integrations.integrations import (
            BundleReloadAuthorityRequest,
            _do_reload_bundles_from_authority,
        )

        session = UserSession(
            session_id="delegated-management",
            user_type=UserType.PRIVILEGED,
            user_id=caller_profile,
            username=caller_profile,
            roles=[],
            permissions=[],
        )
        try:
            outcome = await _do_reload_bundles_from_authority(
                self._request,
                session,
                BundleReloadAuthorityRequest(
                    tenant=self._tenant,
                    project=self._project,
                    bundle_id=application_id,
                ),
            )
        except Exception as exc:
            raise ManagementRuntimeUnavailable("Application reload failed") from exc
        changed = [
            item
            for item in (outcome.get("changed_bundle_ids") or [])
            if item == application_id
        ]
        snapshot = application_readiness_registry.snapshot(
            tenant=self._tenant,
            project=self._project,
            application_id=application_id,
        )
        return {
            "application_id": application_id,
            "state": "completed",
            "changed_application_ids": changed,
            "generation": snapshot.desired_generation if snapshot is not None else "",
        }


__all__ = ["KDCubeManagementRuntime"]
