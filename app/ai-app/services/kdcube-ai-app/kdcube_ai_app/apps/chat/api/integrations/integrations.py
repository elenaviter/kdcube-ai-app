# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from typing import Optional, Dict, Any
import logging
import json

from datetime import datetime
from pydantic import BaseModel
from fastapi import Depends, HTTPException
from fastapi import APIRouter

from kdcube_ai_app.apps.chat.api.resolvers import get_user_session_dependency, auth_without_pressure
from kdcube_ai_app.auth.sessions import UserSession

import kdcube_ai_app.infra.namespaces as namespaces

class AdminBundlesUpdateRequest(BaseModel):
    op: str = "merge"  # "replace" | "merge"
    bundles: Dict[str, Dict[str, Any]]
    default_bundle_id: Optional[str] = None

"""
Integrations API

File: api/integrations/integrations.py
"""


logger = logging.getLogger("KBMonitoring.API")

# Create router
router = APIRouter()


@router.get("/landing/bundles")
async def get_available_bundles(session: UserSession = Depends(get_user_session_dependency())):
    """
    Returns configured bundles for selection in the UI.
    Permission: chat user (read-only).
    """
    from kdcube_ai_app.infra.plugin.bundle_registry import get_all, get_default_id
    reg = get_all()
    default_id = get_default_id()
    return {
        "available_bundles": {
            bid: {
                "id": bid,
                "name": info.get("name"),
                "description": info.get("description"),
                "path": info.get("path"),
                "module": info.get("module"),
                "singleton": info.get("singleton", False),
            }
            for bid, info in reg.items()
        },
        "default_bundle_id": default_id
    }

@router.post("/admin/integrations/bundles", status_code=200)
async def admin_set_bundles(
        payload: AdminBundlesUpdateRequest,
        session: UserSession = Depends(auth_without_pressure())
):
    from kdcube_ai_app.infra.plugin.bundle_registry import set_registry, upsert_bundles, serialize_to_env, get_all, get_default_id
    from kdcube_ai_app.infra.plugin.agentic_loader import clear_agentic_caches

    # Apply locally (in-memory) first
    if payload.op == "replace":
        set_registry(payload.bundles, payload.default_bundle_id)
    elif payload.op == "merge":
        upsert_bundles(payload.bundles, payload.default_bundle_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid op; use 'replace' or 'merge'")

    # Reflect to env
    reg = get_all()
    default_id = get_default_id()
    serialize_to_env(reg, default_id)

    # Clear module caches so new bundles can load fresh
    clear_agentic_caches()

    # Broadcast to all servers via Redis pub/sub
    try:
        msg = {
            "type": "bundles.update",
            "op": payload.op,
            "bundles": payload.bundles,
            "default_bundle_id": payload.default_bundle_id,
            "updated_by": session.username or session.user_id or "unknown",
            "ts": datetime.utcnow().isoformat() + "Z"
        }
        await router.state.middleware.redis.publish(namespaces.CONFIG.CONFIG_CHANNEL,
                                                    json.dumps(msg))
    except Exception as e:
        logger.error(f"Failed to publish config update: {e}")
        # not fatal

    return {"status": "ok", "default_bundle_id": default_id, "count": len(reg)}



