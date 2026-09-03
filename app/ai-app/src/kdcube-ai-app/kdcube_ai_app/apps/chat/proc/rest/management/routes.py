# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Public HTTP routes for delegated management of a running KDCube."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from kdcube_ai_app.apps.chat.proc.rest.management.admission import (
    ConnectionHubAdmissionClient,
)
from kdcube_ai_app.apps.chat.proc.rest.management.config import (
    DelegatedManagementConfig,
)
from kdcube_ai_app.apps.chat.proc.rest.management.contracts import (
    INSPECT_OPERATION,
    RELOAD_OPERATION,
    SURFACES_OPERATION,
    management_error,
    management_resource,
    validate_application_id,
    validate_invocation_id,
)
from kdcube_ai_app.apps.chat.proc.rest.management.effect_ledger import (
    RedisEffectLedger,
)
from kdcube_ai_app.apps.chat.proc.rest.management.runtime import (
    KDCubeManagementRuntime,
)
from kdcube_ai_app.apps.chat.proc.rest.management.service import (
    DelegatedManagementService,
    ManagementResponse,
)
from kdcube_ai_app.apps.chat.sdk.config import get_secret, get_settings

router = APIRouter(prefix="/management/v1")


def _origin(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}".rstrip("/")


def _bearer(request: Request) -> str:
    scheme, separator, value = request.headers.get("authorization", "").partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return value.strip()


def _configuration() -> tuple[Any, DelegatedManagementConfig, str]:
    settings = get_settings()
    config = DelegatedManagementConfig.from_settings(settings)
    config.validate()
    resource = management_resource(config.tenant, config.project)
    return settings, config, resource


def _error_response(
    *,
    status_code: int,
    operation: str,
    resource: str,
    config: DelegatedManagementConfig,
    invocation_id: str,
    code: str,
    message: str,
    retryable: bool = False,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=management_error(
            operation=operation,
            resource=resource,
            tenant=config.tenant,
            project=config.project,
            invocation_id=invocation_id,
            code=code,
            message=message,
            retryable=retryable,
        ),
        headers=headers,
    )


async def _service(request: Request) -> tuple[DelegatedManagementService, str]:
    settings, config, resource = _configuration()
    if not config.enabled:
        raise RuntimeError("delegated_management_disabled")
    secret = str(
        await get_secret(
            f"b:{config.service_secret_ref}",
            bundle_id=config.connection_hub_app_id,
        )
        or ""
    )
    if len(secret.encode("utf-8")) < 32:
        raise RuntimeError("delegated_management_secret_unavailable")
    redis = getattr(request.app.state, "redis_async", None)
    if redis is None:
        raise RuntimeError("delegated_management_redis_unavailable")
    runtime_instance = str(getattr(settings, "INSTANCE_ID", None) or "").strip()
    if not runtime_instance:
        raise RuntimeError("delegated_management_runtime_identity_unavailable")
    return (
        DelegatedManagementService(
            tenant=config.tenant,
            project=config.project,
            resource=resource,
            runtime_instance=runtime_instance,
            admission=ConnectionHubAdmissionClient(
                admission_url=config.resolved_admission_url(settings),
                service_id=config.service_id,
                service_secret=secret,
                timeout_seconds=config.admission_timeout_seconds,
            ),
            ledger=RedisEffectLedger(
                redis,
                tenant=config.tenant,
                project=config.project,
                pending_seconds=config.effect_pending_seconds,
            ),
            runtime=KDCubeManagementRuntime(
                request,
                tenant=config.tenant,
                project=config.project,
            ),
        ),
        resource,
    )


async def _invoke(
    request: Request,
    *,
    operation: str,
    application_id: str = "",
    body: dict[str, Any] | None = None,
) -> JSONResponse:
    _settings, config, resource = _configuration()
    metadata_url = (
        f"{_origin(request)}/api/integrations/management/v1/"
        ".well-known/oauth-protected-resource"
    )
    bearer = _bearer(request)
    if not bearer:
        return _error_response(
            status_code=401,
            operation=operation,
            resource=resource,
            config=config,
            invocation_id="",
            code="delegated_bearer_missing",
            message="A Connection Hub delegated bearer is required.",
            headers={
                "WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'
            },
        )
    try:
        invocation_id = validate_invocation_id(
            request.headers.get("idempotency-key", "")
        )
    except ValueError as exc:
        return _error_response(
            status_code=400,
            operation=operation,
            resource=resource,
            config=config,
            invocation_id="",
            code="idempotency_key_invalid",
            message=str(exc),
        )
    try:
        service, _ = await _service(request)
    except RuntimeError as exc:
        code = str(exc)
        status = 404 if code == "delegated_management_disabled" else 503
        return _error_response(
            status_code=status,
            operation=operation,
            resource=resource,
            config=config,
            invocation_id=invocation_id,
            code=code,
            message="Delegated KDCube management is unavailable.",
            retryable=status == 503,
        )
    response: ManagementResponse = await service.execute(
        operation=operation,
        delegated_bearer=bearer,
        invocation_id=invocation_id,
        application_id=application_id,
        body=body,
    )
    return JSONResponse(status_code=response.status_code, content=dict(response.payload))


@router.get(
    "/.well-known/oauth-protected-resource",
    name="delegated_management_protected_resource_metadata",
)
async def protected_resource_metadata(request: Request) -> JSONResponse:
    _settings, config, resource = _configuration()
    if not config.enabled:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    origin = _origin(request)
    encoded = (
        quote(config.tenant, safe="-._~"),
        quote(config.project, safe="-._~"),
        quote(config.connection_hub_app_id, safe="-._~@"),
    )
    issuer = (
        f"{origin}/api/integrations/bundles/{encoded[0]}/{encoded[1]}/"
        f"{encoded[2]}/public/oauth"
    )
    return JSONResponse(
        {
            "resource": resource,
            "authorization_servers": [issuer],
            "bearer_methods_supported": ["header"],
            "kdcube_management_operations": {
                INSPECT_OPERATION: {
                    "method": "GET",
                    "path": "/api/integrations/management/v1/deployment",
                },
                SURFACES_OPERATION: {
                    "method": "GET",
                    "path": (
                        "/api/integrations/management/v1/applications/"
                        "{application_id}/surfaces"
                    ),
                },
                RELOAD_OPERATION: {
                    "method": "POST",
                    "path": (
                        "/api/integrations/management/v1/applications/"
                        "{application_id}/reload"
                    ),
                },
            },
        }
    )


@router.get("/deployment")
async def inspect_deployment(request: Request) -> JSONResponse:
    if await request.body():
        settings, config, resource = _configuration()
        del settings
        return _error_response(
            status_code=400,
            operation=INSPECT_OPERATION,
            resource=resource,
            config=config,
            invocation_id=request.headers.get("idempotency-key", ""),
            code="request_body_not_allowed",
            message="Deployment inspection accepts no request body.",
        )
    return await _invoke(request, operation=INSPECT_OPERATION)


@router.get("/applications/{application_id}/surfaces")
async def application_surfaces(
    application_id: str, request: Request
) -> JSONResponse:
    try:
        exact_application_id = validate_application_id(application_id)
    except ValueError as exc:
        _settings, config, resource = _configuration()
        return _error_response(
            status_code=400,
            operation=SURFACES_OPERATION,
            resource=resource,
            config=config,
            invocation_id=request.headers.get("idempotency-key", ""),
            code="application_id_invalid",
            message=str(exc),
        )
    if await request.body():
        _settings, config, resource = _configuration()
        return _error_response(
            status_code=400,
            operation=SURFACES_OPERATION,
            resource=resource,
            config=config,
            invocation_id=request.headers.get("idempotency-key", ""),
            code="request_body_not_allowed",
            message="Surface discovery accepts no request body.",
        )
    return await _invoke(
        request,
        operation=SURFACES_OPERATION,
        application_id=exact_application_id,
    )


@router.post("/applications/{application_id}/reload")
async def reload_application(application_id: str, request: Request) -> JSONResponse:
    try:
        exact_application_id = validate_application_id(application_id)
    except ValueError as exc:
        _settings, config, resource = _configuration()
        return _error_response(
            status_code=400,
            operation=RELOAD_OPERATION,
            resource=resource,
            config=config,
            invocation_id=request.headers.get("idempotency-key", ""),
            code="application_id_invalid",
            message=str(exc),
        )
    try:
        payload = json.loads((await request.body()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if payload != {}:
        _settings, config, resource = _configuration()
        return _error_response(
            status_code=400,
            operation=RELOAD_OPERATION,
            resource=resource,
            config=config,
            invocation_id=request.headers.get("idempotency-key", ""),
            code="reload_request_invalid",
            message="Application reload requires the exact JSON body {}.",
        )
    return await _invoke(
        request,
        operation=RELOAD_OPERATION,
        application_id=exact_application_id,
        body={},
    )


__all__ = ["router"]
