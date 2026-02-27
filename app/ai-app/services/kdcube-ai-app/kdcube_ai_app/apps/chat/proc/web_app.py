# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

# chat/proc/web_app.py
"""
Processor service: runs the queue processor and exposes minimal integrations API.
"""
import asyncio
import faulthandler
import logging
import os
import signal
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

# Default component identity for shared .env usage
os.environ.setdefault("GATEWAY_COMPONENT", "proc")

_ENV_DIR = Path(__file__).resolve().parent
load_dotenv(_ENV_DIR / ".env.proc", override=True)
load_dotenv(find_dotenv(usecwd=False))

# Ensure per-replica instance id is set (do not override explicit env)
os.environ.setdefault("INSTANCE_ID", f"proc-{uuid.uuid4().hex[:8]}")

from kdcube_ai_app.apps.utils.cors import configure_cors
from kdcube_ai_app.apps.middleware.gateway import STATE_FLAG, STATE_SESSION, STATE_USER_TYPE
from kdcube_ai_app.apps.middleware.token_extract import extract_auth_tokens_from_query_params
from kdcube_ai_app.infra.gateway.config import (
    get_gateway_config,
    apply_gateway_config_from_cache,
    subscribe_gateway_config_updates,
    GatewayConfigFactory,
    gateway_config_cache_key,
)
from kdcube_ai_app.infra.gateway.backpressure import create_atomic_chat_queue_manager
from kdcube_ai_app.infra.plugin.agentic_loader import AgenticBundleSpec
from kdcube_ai_app.infra.rendering.link_preview import close_shared_link_preview
from kdcube_ai_app.infra.rendering.shared_browser import close_shared_browser

from kdcube_ai_app.apps.chat.emitters import ChatRelayCommunicator
from kdcube_ai_app.apps.chat.api.resolvers import (
    get_fastapi_adapter,
    get_fast_api_accounting_binder,
    INSTANCE_ID,
    CHAT_APP_PORT,
    CHAT_PROCESSOR_PORT,
    REDIS_URL,
    get_pg_pool,
    get_conversation_system,
    get_redis_clients,
    close_redis_clients,
    get_redis_monitor_instance,
    get_heartbeats_mgr_and_middleware,
    get_external_request_processor,
    service_health_checker,
)
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.apps.chat.sdk.infra.economics.policy import EconomicsLimitException
from kdcube_ai_app.apps.chat.proc.rest.integrations import mount_integrations_routers
from kdcube_ai_app.infra.namespaces import CONFIG

# Ensure per-replica instance id is set (do not override explicit env)
os.environ.setdefault("INSTANCE_ID", f"proc-{uuid.uuid4().hex[:8]}")

import kdcube_ai_app.apps.utils.logging_config as logging_config
logging_config.configure_logging()
logger = logging.getLogger("ChatProc.WebApp")


def _install_crash_logging() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        logger.warning("Failed to enable faulthandler", exc_info=True)
    for sig in (signal.SIGTERM, signal.SIGABRT, signal.SIGSEGV):
        try:
            faulthandler.register(sig, all_threads=True)
        except Exception:
            pass

    def _excepthook(exc_type, exc, tb):
        logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook


_install_crash_logging()


def _get_uvicorn_workers_from_config() -> int:
    try:
        cfg = GatewayConfigFactory.create_from_env()
        return max(1, int(cfg.service_capacity.processes_per_instance))
    except Exception:
        logger.exception("Failed to resolve Uvicorn workers from gateway config; using 1")
        return 1


async def _safe_shutdown_step(name: str, coro, timeout: float = 5.0) -> None:
    try:
        await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Shutdown step timed out: %s (>%ss)", name, timeout)
    except Exception:
        logger.exception("Shutdown step failed: %s", name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Proc lifespan startup begin: port=%s pid=%s workers_cfg=%s reload_env=%s",
        CHAT_PROCESSOR_PORT,
        os.getpid(),
        _get_uvicorn_workers_from_config(),
        os.getenv("UVICORN_RELOAD", "0"),
    )
    app.state.shutting_down = False
    app.state.draining = False

    # register signal handlers for graceful drain (best-effort)
    try:
        import signal
        loop = asyncio.get_running_loop()

        def _enter_draining_mode():
            if not getattr(app.state, "draining", False):
                app.state.draining = True
                app.state.shutting_down = True
                logger.warning("Proc entering draining mode (SIGTERM/SIGINT).")

        for _sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(_sig, _enter_draining_mode)
            except Exception:
                signal.signal(_sig, lambda *_args: _enter_draining_mode())
    except Exception:
        pass

    try:
        # Gateway adapter (used by integrations auth)
        app.state.gateway_adapter = get_fastapi_adapter()
        settings = get_settings()
        cache_applied = await apply_gateway_config_from_cache(
            gateway_adapter=app.state.gateway_adapter,
            tenant=settings.TENANT,
            project=settings.PROJECT,
            redis_url=REDIS_URL,
        )
        if cache_applied:
            app.state.gateway_config_source = "redis-cache"
            logger.info(
                "Gateway config source: redis-cache tenant=%s project=%s key=%s",
                settings.TENANT,
                settings.PROJECT,
                gateway_config_cache_key(tenant=settings.TENANT, project=settings.PROJECT),
            )
        else:
            source = "env"
            if os.getenv("GATEWAY_CONFIG_JSON"):
                source = "env (GATEWAY_CONFIG_JSON)"
            app.state.gateway_config_source = source
            logger.info(
                "Gateway config source: %s tenant=%s project=%s",
                source,
                settings.TENANT,
                settings.PROJECT,
            )
        app.state.gateway_config_stop = asyncio.Event()
        app.state.gateway_config_task = asyncio.create_task(
            subscribe_gateway_config_updates(
                gateway_adapter=app.state.gateway_adapter,
                tenant=settings.TENANT,
                project=settings.PROJECT,
                redis_url=REDIS_URL,
                stop_event=app.state.gateway_config_stop,
            )
        )
        gateway_config = get_gateway_config()
        app.state.chat_queue_manager = create_atomic_chat_queue_manager(
            gateway_config.redis_url,
            gateway_config,
            app.state.gateway_adapter.gateway.throttling_monitor,
        )
        app.state.acc_binder = get_fast_api_accounting_binder()
    except Exception:
        logger.exception("Proc lifespan startup failed during gateway initialization")
        raise

    # Shared Redis pools + monitor
    try:
        app.state.redis_async, app.state.redis_async_decode, app.state.redis_sync = await get_redis_clients()
        logger.info("Redis pools ready (async/sync)")
    except Exception:
        logger.exception("Failed to initialize shared Redis pools")
        raise
    try:
        app.state.redis_monitor = await get_redis_monitor_instance()
    except Exception:
        logger.exception("Failed to start Redis connection monitor")
        raise

    # Relay communicator (processor publishes through it)
    app.state.chat_comm = ChatRelayCommunicator(redis_url=REDIS_URL, channel="chat.events")

    # Postgres
    app.state.pg_pool = await get_pg_pool()

    try:
        from kdcube_ai_app.apps.middleware.economics_role import EconomicsRoleResolver
        app.state.gateway_adapter.set_econ_role_resolver(
            EconomicsRoleResolver(
                pg_pool=app.state.pg_pool,
                tenant=settings.TENANT,
                project=settings.PROJECT,
            ).resolve_role_for_user_id
        )
    except Exception as e:
        logger.warning("Failed to attach economics role resolver: %s", e)

    port = CHAT_PROCESSOR_PORT

    async def agentic_app_func(comm_context: "ChatTaskPayload"):
        """
        Entry-point invoked by the processor.
        """
        import inspect
        from kdcube_ai_app.infra.plugin.bundle_registry import resolve_bundle_async
        from kdcube_ai_app.infra.plugin.agentic_loader import get_workflow_instance
        from kdcube_ai_app.infra.service_hub.inventory import ConfigRequest, create_workflow_config

        cfg_req = ConfigRequest(**(comm_context.config.values or {}))
        wf_config = create_workflow_config(cfg_req)
        bundle_id = comm_context.routing.bundle_id
        spec_resolved = await resolve_bundle_async(bundle_id, override=None)

        wf_config.ai_bundle_spec = spec_resolved
        spec = AgenticBundleSpec(
            path=spec_resolved.path,
            module=spec_resolved.module,
            singleton=bool(spec_resolved.singleton),
        )
        try:
            workflow, _ = get_workflow_instance(
                spec=spec,
                config=wf_config,
                comm_context=comm_context,
                pg_pool=app.state.pg_pool,
                redis=app.state.redis_async,
            )
        except Exception as e:
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
                workflow, _ = get_workflow_instance(
                    spec=admin,
                    config=wf_config,
                    comm_context=comm_context,
                    pg_pool=app.state.pg_pool,
                    redis=app.state.redis_async,
                )
            except Exception:
                raise

        state = {
            "request_id": comm_context.request.request_id,
            "tenant": comm_context.actor.tenant_id,
            "project": comm_context.actor.project_id,
            "user": comm_context.user.user_id,
            "user_type": comm_context.user.user_type,
            "session_id": comm_context.routing.session_id,
            "conversation_id": (comm_context.routing.conversation_id or comm_context.routing.session_id),
            "text": comm_context.request.message or (comm_context.request.payload or {}).get("text") or "",
            "attachments": (comm_context.request.payload or {}).get("attachments") or [],
            "turn_id": comm_context.routing.turn_id,
            "history": comm_context.request.chat_history or [],
            "final_answer": "",
            "followups": [],
            "step_logs": [],
            "start_time": comm_context.meta.created_at,
        }
        if hasattr(workflow, "set_state"):
            maybe = workflow.set_state(state)
            if inspect.isawaitable(maybe):
                await maybe

        params = dict(comm_context.request.payload or {})
        if "text" not in params and comm_context.request.message:
            params["text"] = comm_context.request.message
        command = comm_context.request.operation or params.pop("command", None)

        try:
            result = await (getattr(workflow, command)(**params) if (command and hasattr(workflow, command))
                            else workflow.run(**params))
            return result or {}
        except Exception as e:
            logger.error(traceback.format_exc())
            if not isinstance(e, EconomicsLimitException):
                return {"error_message": str(e), "final_answer": "An error occurred."}

    try:
        redis_async = app.state.redis_async
        from kdcube_ai_app.infra.metrics.pool_stats import build_pool_metadata

        def _heartbeat_metadata():
            return build_pool_metadata(
                pg_pool=app.state.pg_pool,
                redis_clients={
                    "async": app.state.redis_async,
                    "async_decode": app.state.redis_async_decode,
                    "sync": app.state.redis_sync,
                },
            )

        middleware, heartbeat_manager = get_heartbeats_mgr_and_middleware(
            service_type="chat",
            service_name="proc",
            port=port,
            redis_client=redis_async,
            metadata_provider=_heartbeat_metadata,
        )
        health_checker = service_health_checker(middleware)

        app.state.middleware = middleware
        app.state.heartbeat_manager = heartbeat_manager
        app.state.health_checker = health_checker

        conversation_browser, conversation_index, conversation_store = await get_conversation_system(app.state.pg_pool)
        app.state.conversation_browser = conversation_browser
        app.state.conversation_index = conversation_index
        app.state.conversation_store = conversation_store

        processor = get_external_request_processor(middleware, agentic_app_func, app, redis=redis_async)
        app.state.processor = processor

        await heartbeat_manager.start_heartbeat(interval=10)
        try:
            from kdcube_ai_app.infra.plugin.bundle_store import load_registry as _load_store_registry
            from kdcube_ai_app.infra.plugin.bundle_store import force_env_reset_if_requested
            from kdcube_ai_app.infra.plugin.bundle_registry import set_registry as _set_mem_registry
            reg = await force_env_reset_if_requested(
                redis_async,
                tenant=settings.TENANT,
                project=settings.PROJECT,
                actor="startup-env",
            )
            if not reg:
                reg = await _load_store_registry(redis_async)
            bundles_dict = {bid: entry.model_dump() for bid, entry in reg.bundles.items()}
            _set_mem_registry(bundles_dict, reg.default_bundle_id)
            logger.info(
                "Bundles registry loaded from Redis: %s items (default=%s)",
                len(bundles_dict),
                reg.default_bundle_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to load bundles registry from Redis; using env-only registry. %s",
                e,
            )

        await processor.start_processing()
    except Exception:
        logger.exception("Could not start processor service")
        raise

    # Bundle cleanup loop (proc only)
    try:
        from kdcube_ai_app.apps.chat.api.opex import routines as opex_routines
        app.state.bundle_cleanup_task = asyncio.create_task(opex_routines.bundle_cleanup_loop())
        logger.info("[Bundles] Cleanup task started (proc)")
    except Exception:
        logger.exception("Failed to start bundle cleanup loop")

    yield

    app.state.shutting_down = True
    if getattr(app.state, "processor", None):
        await _safe_shutdown_step("processor.stop_processing", app.state.processor.stop_processing(), timeout=10.0)
    if getattr(app.state, "heartbeat_manager", None):
        await _safe_shutdown_step("heartbeat.stop", app.state.heartbeat_manager.stop_heartbeat(), timeout=5.0)
    if getattr(app.state, "gateway_config_stop", None):
        app.state.gateway_config_stop.set()
    if getattr(app.state, "gateway_config_task", None):
        await _safe_shutdown_step("gateway_config_task", app.state.gateway_config_task, timeout=5.0)
    if getattr(app.state, "bundle_cleanup_task", None):
        app.state.bundle_cleanup_task.cancel()
        try:
            await app.state.bundle_cleanup_task
        except asyncio.CancelledError:
            pass
        app.state.bundle_cleanup_task = None
    if hasattr(app.state, "redis_monitor"):
        await _safe_shutdown_step("redis_monitor.stop", app.state.redis_monitor.stop(), timeout=5.0)
    if hasattr(app.state, "pg_pool"):
        await _safe_shutdown_step("pg_pool.close", app.state.pg_pool.close(), timeout=10.0)
    if hasattr(app.state, "redis_async"):
        await _safe_shutdown_step("redis_clients.close", close_redis_clients(), timeout=5.0)
    await close_shared_link_preview()
    await close_shared_browser()
    logger.info("Chat processor service stopped")


app = FastAPI(
    title="Chat Processor Service",
    description="Queue processor + integrations",
    lifespan=lifespan,
)

allowed_origins = configure_cors(app)


@app.middleware("http")
async def gateway_middleware(request: Request, call_next):
    if getattr(request.app.state, "draining", False) and not request.url.path.startswith("/health"):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is draining", "status": "draining"},
        )
    if request.method == "OPTIONS" or request.url.path.startswith(("/health", "/docs", "/openapi.json", "/favicon.ico")):
        return await call_next(request)

    if getattr(request.state, STATE_FLAG, False):
        return await call_next(request)

    try:
        # Inject auth tokens from query params if needed
        if request.url.path.startswith("/api/integrations/"):
            bearer_token, id_token = extract_auth_tokens_from_query_params(request.query_params)
            user_timezone = request.query_params.get("user_timezone")
            user_utc_offset_min = request.query_params.get("user_utc_offset_min")

            headers = MutableHeaders(scope=request.scope)
            if bearer_token and "authorization" not in {k.lower(): v for k, v in headers.items()}:
                headers["authorization"] = f"Bearer {bearer_token}"
            if id_token:
                headers[CONFIG.ID_TOKEN_HEADER_NAME] = id_token
            if user_timezone:
                headers[CONFIG.USER_TIMEZONE_HEADER_NAME] = user_timezone
            if user_utc_offset_min:
                headers[CONFIG.USER_UTC_OFFSET_MIN_HEADER_NAME] = user_utc_offset_min

        session = await app.state.gateway_adapter.process_by_policy(request)
        setattr(request.state, STATE_SESSION, session)
        setattr(request.state, STATE_USER_TYPE, session.user_type.value)
        setattr(request.state, STATE_FLAG, True)

        response = await call_next(request)
        response.headers["X-User-Type"] = session.user_type.value
        response.headers["X-Session-ID"] = session.session_id
        return response
    except HTTPException as e:
        headers = getattr(e, "headers", {})
        return JSONResponse(
            status_code=e.status_code,
            content=e.detail if isinstance(e.detail, dict) else {"detail": e.detail},
            headers=headers,
        )


@app.get("/health")
async def health():
    draining = getattr(app.state, "draining", False)
    payload = {
        "status": "draining" if draining else "ok",
        "draining": draining,
        "service": "chat-proc",
        "instance_id": INSTANCE_ID,
    }
    if draining:
        return JSONResponse(status_code=503, content=payload)
    return payload


# Mount integrations API at /api/integrations
mount_integrations_routers(app)


if __name__ == "__main__":
    import uvicorn

    try:
        faulthandler.enable()
    except Exception:
        pass

    workers = _get_uvicorn_workers_from_config()
    reload_enabled = os.getenv("UVICORN_RELOAD", "").lower() in {"1", "true", "yes", "on"}
    use_import_string = workers > 1 or reload_enabled
    app_target = "kdcube_ai_app.apps.chat.proc.web_app:app" if use_import_string else app

    run_kwargs = {
        "host": "0.0.0.0",
        "port": CHAT_PROCESSOR_PORT,
        "log_config": None,
        "log_level": None,
        "timeout_keep_alive": 60 * 60,
        "timeout_graceful_shutdown": 15,
    }
    if use_import_string:
        run_kwargs["workers"] = workers
        if reload_enabled:
            run_kwargs["reload"] = True

    logger.info(
        "Starting Uvicorn (proc): target=%s workers=%s reload=%s port=%s pid=%s",
        app_target,
        workers,
        reload_enabled,
        CHAT_PROCESSOR_PORT,
        os.getpid(),
    )
    uvicorn.run(app_target, **run_kwargs)
