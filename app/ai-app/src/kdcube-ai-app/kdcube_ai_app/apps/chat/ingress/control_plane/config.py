# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.infra.config.frontend_config import build_frontend_config as build_frontend_config_payload

router = APIRouter()
logger = logging.getLogger(__name__)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assembly_path() -> Path | None:
    explicit = _text(os.getenv("ASSEMBLY_YAML_DESCRIPTOR_PATH"))
    if explicit:
        return Path(explicit).expanduser()
    descriptors_dir = _text(os.getenv("PLATFORM_DESCRIPTORS_DIR"))
    if descriptors_dir:
        return Path(descriptors_dir).expanduser() / "assembly.yaml"
    default = Path("/config/assembly.yaml")
    return default if default.exists() else None


def _load_assembly_descriptor() -> dict[str, Any]:
    path = _assembly_path()
    if not path or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load assembly descriptor for frontend config: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _assembly_from_settings(settings: Any) -> dict[str, Any]:
    auth_cfg = getattr(settings, "AUTH", None)
    frontend_plain = settings.plain("frontend")
    assembly: dict[str, Any] = {
        "company": settings.plain("company"),
        "context": {
            "tenant": settings.TENANT,
            "project": settings.PROJECT,
        },
        "auth": {
            "type": settings.plain("auth.type"),
            "idp": getattr(settings, "AUTH_PROVIDER", ""),
            "id_token_header_name": _text(getattr(auth_cfg, "ID_TOKEN_HEADER_NAME", "")),
            "turnstile_development_token": settings.plain("auth.turnstile_development_token"),
            "cognito": {
                "region": _text(getattr(auth_cfg, "COGNITO_REGION", "")),
                "user_pool_id": _text(getattr(auth_cfg, "COGNITO_USER_POOL_ID", "")),
                "app_client_id": _text(getattr(auth_cfg, "COGNITO_APP_CLIENT_ID", "")),
            },
        },
        "proxy": {
            "route_prefix": settings.plain("proxy.route_prefix"),
        },
    }
    if isinstance(frontend_plain, dict):
        assembly["frontend"] = frontend_plain
    return assembly


def build_frontend_config() -> dict[str, Any]:
    settings = get_settings()
    assembly = _load_assembly_descriptor() or _assembly_from_settings(settings)
    auth_cfg = getattr(settings, "AUTH", None)

    return build_frontend_config_payload(
        tenant=settings.TENANT,
        project=settings.PROJECT,
        assembly=assembly,
        cognito_region=_text(getattr(auth_cfg, "COGNITO_REGION", "")),
        cognito_user_pool_id=_text(getattr(auth_cfg, "COGNITO_USER_POOL_ID", "")),
        cognito_app_client_id=_text(getattr(auth_cfg, "COGNITO_APP_CLIENT_ID", "")),
        routes_prefix=_text(settings.plain("proxy.route_prefix")) or None,
        company_name=_text(settings.plain("company")) or None,
        turnstile_development_token=_text(settings.plain("auth.turnstile_development_token")) or None,
        auth_token_cookie_name=_text(getattr(auth_cfg, "AUTH_TOKEN_COOKIE_NAME", "")) or None,
        id_token_cookie_name=_text(getattr(auth_cfg, "ID_TOKEN_COOKIE_NAME", "")) or None,
    )


# ---------------------------------------------------------------------------
# Serving the frontend config
#
# ``build_frontend_config`` performs synchronous, blocking I/O: it reads
# ``assembly.yaml`` (typically from a network mount such as EFS) and resolves
# plain-YAML settings on *every* call. Previously the route below was an
# ``async def`` that invoked it directly, so the read ran on the event loop.
# When the underlying storage stalled, a single request could wedge a uvicorn
# worker's event loop indefinitely; under low traffic the workers froze one by
# one until chat-ingress stopped serving *every* route.
#
# We now (a) run the blocking build in a worker thread so it can never block
# the event loop, (b) bound it with a timeout, and (c) cache the result with a
# short TTL, falling back to the last-known-good value if a rebuild fails or
# times out. The payload is derived from deployment-static descriptors, so a
# few seconds of staleness is harmless.
# ---------------------------------------------------------------------------

_CONFIG_TTL_SECONDS = float(os.getenv("CP_FRONTEND_CONFIG_TTL_SECONDS", "30") or 30)
_CONFIG_RETRY_SECONDS = float(os.getenv("CP_FRONTEND_CONFIG_RETRY_SECONDS", "5") or 5)
_CONFIG_BUILD_TIMEOUT_SECONDS = float(
    os.getenv("CP_FRONTEND_CONFIG_BUILD_TIMEOUT_SECONDS", "10") or 10
)

_config_lock = asyncio.Lock()
_config_cache: dict[str, Any] | None = None
_config_expiry: float = 0.0


async def _get_frontend_config() -> dict[str, Any]:
    """Return the frontend config without ever blocking the event loop.

    Serves a cached payload while it is fresh, rebuilds it off-thread when
    stale, and returns the last-known-good value if a rebuild fails or times
    out. This keeps a stalled descriptor read (e.g. a hung EFS mount) from
    freezing the uvicorn worker's event loop.
    """
    global _config_cache, _config_expiry

    if _config_cache is not None and time.monotonic() < _config_expiry:
        return _config_cache

    # A rebuild is needed. Only one runs at a time; concurrent callers that
    # already have a cached value serve it rather than piling up behind the lock
    # (and rather than spawning additional blocking build threads).
    if _config_cache is not None and _config_lock.locked():
        return _config_cache

    async with _config_lock:
        # Re-check: another coroutine may have refreshed while we waited.
        if _config_cache is not None and time.monotonic() < _config_expiry:
            return _config_cache
        try:
            fresh = await asyncio.wait_for(
                asyncio.to_thread(build_frontend_config),
                timeout=_CONFIG_BUILD_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Failed to (re)build frontend config")
            if _config_cache is not None:
                # Back off before retrying so a stalled backend doesn't cause a
                # tight rebuild loop; keep serving the last-known-good payload.
                _config_expiry = time.monotonic() + _CONFIG_RETRY_SECONDS
                return _config_cache
            raise
        _config_cache = fresh
        _config_expiry = time.monotonic() + _CONFIG_TTL_SECONDS
        return _config_cache


@router.get("/api/cp-frontend-config")
async def cp_frontend_config() -> JSONResponse:
    return JSONResponse(
        content=await _get_frontend_config(),
        headers={"Cache-Control": "no-store, no-cache"},
    )
