# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/proc/rest/integrations/integrations.py
import inspect
import asyncio
import logging
import os
import uuid
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Body
from pydantic import BaseModel

from kdcube_ai_app.apps.chat.api.resolvers import require_auth, REDIS_URL
from kdcube_ai_app.auth.AuthManager import RequireUser
from kdcube_ai_app.auth.sessions import UserSession
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.infra.redis.client import get_async_redis_client
from kdcube_ai_app.infra.service_hub.inventory import ConfigRequest, create_workflow_config
from kdcube_ai_app.apps.chat.sdk.protocol import (
    ChatTaskPayload,
    ChatTaskRouting,
    ChatTaskActor,
    ChatTaskUser,
    ChatTaskRequest,
)
from kdcube_ai_app.infra.plugin.bundle_registry import resolve_bundle_async
from kdcube_ai_app.infra.plugin.agentic_loader import AgenticBundleSpec, get_workflow_instance
import kdcube_ai_app.infra.namespaces as namespaces

logger = logging.getLogger("ChatProc.Integrations")
_integrations_limit: Optional[int] = None
_integrations_semaphore = None


def _resolve_integrations_limit() -> Optional[int]:
    # Prefer gateway config limits (component-aware).
    try:
        from kdcube_ai_app.infra.gateway.config import get_gateway_config
        cfg = get_gateway_config()
        limits_cfg = getattr(cfg, "limits", None)
        if limits_cfg and limits_cfg.max_integrations_ops_concurrency is not None:
            value = int(limits_cfg.max_integrations_ops_concurrency)
            return value if value > 0 else None
    except Exception:
        pass
    return None


def _get_integrations_semaphore():
    global _integrations_limit, _integrations_semaphore
    limit = _resolve_integrations_limit()
    if not limit:
        _integrations_limit = None
        _integrations_semaphore = None
        return None
    if _integrations_semaphore is None or _integrations_limit != limit:
        _integrations_limit = limit
        _integrations_semaphore = asyncio.Semaphore(limit)
    return _integrations_semaphore


def _get_app_redis(request: Request):
    redis = getattr(request.app.state, "redis_async", None)
    if redis is None:
        # fallback to router state if wired via mount_integrations_routers
        redis = getattr(router.state, "redis_async", None)
    if redis is None:
        raise RuntimeError("redis_async is not initialized on app.state")
    return redis


router = APIRouter()


class BundleSuggestionsRequest(BaseModel):
    bundle_id: Optional[str] = None
    conversation_id: Optional[str] = None
    config_request: Optional[ConfigRequest] = None


@router.post("/bundles/{tenant}/{project}/operations/{operation}")
async def call_bundle_op(
    tenant: str,
    project: str,
    request: Request,
    payload: BundleSuggestionsRequest = Body(default_factory=BundleSuggestionsRequest),
    operation: str = "suggestions",  # news, etc.
    session: UserSession = Depends(require_auth(RequireUser())),
):
    """
    Load (or reuse singleton) bundle instance and call its operation (e.g. suggestions()).
    Returns generic JSON from the bundle.
    """
    settings = get_settings()
    sem = _get_integrations_semaphore()
    if sem:
        async with sem:
            return await _call_bundle_op_inner(
                tenant=tenant,
                project=project,
                payload=payload,
                request=request,
                operation=operation,
                session=session,
            )
    return await _call_bundle_op_inner(
        tenant=tenant,
        project=project,
        payload=payload,
        request=request,
        operation=operation,
        session=session,
    )


async def _call_bundle_op_inner(
    *,
    tenant: str,
    project: str,
    payload: BundleSuggestionsRequest,
    request: Request,
    operation: str,
    session: UserSession,
):
    settings = get_settings()
    cfg_req = payload.config_request or ConfigRequest()

    # Ensure model defaults exist
    if not cfg_req.selected_model:
        cfg_req.selected_model = (namespaces.CONFIG.AGENTIC.DEFAULT_LLM_MODEL_CONFIG or {}).get("model_name", "gpt-4o-mini")
    if not cfg_req.selected_embedder:
        cfg_req.selected_embedder = (namespaces.CONFIG.AGENTIC.DEFAULT_EMBEDDING_MODEL_CONFIG or {}).get("model_name", "gpt-4o-mini")
    if not cfg_req.openai_api_key:
        cfg_req.openai_api_key = os.getenv("OPENAI_API_KEY")
    if not cfg_req.claude_api_key:
        cfg_req.claude_api_key = os.getenv("ANTHROPIC_API_KEY")

    if payload.bundle_id:
        cfg_req.agentic_bundle_id = payload.bundle_id

    tenant_id = cfg_req.tenant or tenant or settings.TENANT
    project_id = cfg_req.project or project or settings.PROJECT
    request_id = str(uuid.uuid4())

    # 1) Resolve bundle from registry
    spec_resolved = await resolve_bundle_async(cfg_req.agentic_bundle_id, override=None)
    if not spec_resolved:
        raise HTTPException(status_code=404, detail=f"Bundle {cfg_req.agentic_bundle_id} not found")

    # 2) Build workflow config
    wf_config = create_workflow_config(cfg_req)
    wf_config.ai_bundle_spec = spec_resolved

    spec = AgenticBundleSpec(
        path=spec_resolved.path,
        module=spec_resolved.module,
        singleton=bool(spec_resolved.singleton),
    )
    routing = ChatTaskRouting(
        session_id=session.session_id,
        bundle_id=spec_resolved.id,
    )
    comm_context = ChatTaskPayload(
        request=ChatTaskRequest(request_id=request_id),
        routing=routing,
        actor=ChatTaskActor(
            tenant_id=tenant_id,
            project_id=project_id,
        ),
        user=ChatTaskUser(
            user_type=session.user_type.value,
            user_id=session.user_id,
            username=session.username,
            fingerprint=session.fingerprint,
            roles=session.roles,
            permissions=session.permissions,
            timezone=session.request_context.user_timezone,
            utc_offset_min=session.request_context.user_utc_offset_min,
        ),
    )

    redis = _get_app_redis(request)
    try:
        workflow, _mod = get_workflow_instance(
            spec, wf_config, comm_context=comm_context, redis=redis,
        )
    except Exception as e:
        logger.exception(f"[call_bundle_op.{tenant}.{project}] Failed to load bundle {asdict(spec)}")
        try:
            admin_spec = await resolve_bundle_async("kdcube.admin", override=None)
            if not admin_spec:
                raise e
            wf_config.ai_bundle_spec = admin_spec
            admin = AgenticBundleSpec(
                path=admin_spec.path,
                module=admin_spec.module,
                singleton=bool(admin_spec.singleton),
            )
            workflow, _mod = get_workflow_instance(
                admin, wf_config, comm_context=comm_context, redis=redis,
            )
            spec_resolved = admin_spec
        except Exception:
            raise HTTPException(status_code=500, detail=f"Failed to load bundle: {e}")

    # 4) Call op() if available (support sync/async)
    if not hasattr(workflow, operation) or not callable(getattr(workflow, operation)):
        raise HTTPException(status_code=404, detail=f"Bundle does not support operation {operation}")

    try:
        user_id = session.user_id or session.fingerprint
        fn = getattr(workflow, operation)
        if inspect.iscoroutinefunction(fn):
            result = await fn(user_id=user_id, fingerprint=session.fingerprint)
        else:
            result = fn(user_id=user_id, fingerprint=session.fingerprint)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{operation}() failed: {e}")

    return {
        "status": "ok",
        "tenant": tenant_id,
        "project": project_id,
        "bundle_id": spec_resolved.id,
        "conversation_id": payload.conversation_id,
        operation: result,
    }
