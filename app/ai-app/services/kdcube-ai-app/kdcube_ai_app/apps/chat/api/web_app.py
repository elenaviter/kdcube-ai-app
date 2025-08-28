# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/web_app.py
"""
FastAPI chat application with modular Socket.IO integration and gateway protection
"""
import uuid
import time
import logging
import os

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from kdcube_ai_app.apps.middleware.logging.uvicorn import configure_logging
from kdcube_ai_app.infra.accounting.envelope import build_envelope_from_session, AccountingEnvelope

configure_logging()

from kdcube_ai_app.apps.middleware.gateway import STATE_FLAG, STATE_SESSION, STATE_USER_TYPE
from kdcube_ai_app.infra.gateway.backpressure import create_atomic_chat_queue_manager
from kdcube_ai_app.infra.gateway.circuit_breaker import CircuitBreakerError
from kdcube_ai_app.infra.gateway.config import get_gateway_config

# Import our simplified components
from kdcube_ai_app.apps.chat.api.resolvers import (
    get_fastapi_adapter, get_fast_api_accounting_binder, get_user_session_dependency,
    get_orchestrator, INSTANCE_ID, CHAT_APP_PORT, REDIS_URL, auth_without_pressure
)
from kdcube_ai_app.auth.sessions import UserType, UserSession
from kdcube_ai_app.apps.chat.reg import MODEL_CONFIGS, EMBEDDERS

from kdcube_ai_app.apps.chat.sdk.inventory import ConfigRequest, create_workflow_config, _mid, BundleState
from kdcube_ai_app.infra.orchestration.orchestration import IOrchestrator

# Import the modular Socket.IO handler
from kdcube_ai_app.apps.chat.api.socketio.chat import create_socketio_chat_handler

import kdcube_ai_app.apps.utils.logging_config as logging_config
logging_config.configure_logging()
logger = logging.getLogger(__name__)


# ================================
# APPLICATION SETUP
# ================================
# CORS setup
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:4000",
    "http://localhost:5173",
    "http://localhost:8050",
]
app_domain = os.environ.get("APP_DOMAIN")
if app_domain:
    allowed_origins.append(f"https://{app_domain}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified lifespan management"""
    # Startup
    logger.info(f"Chat service starting on port {CHAT_APP_PORT}")

    # Initialize gateway adapter and store in app state
    app.state.gateway_adapter = get_fastapi_adapter()
    gateway_config = get_gateway_config()
    app.state.chat_queue_manager = create_atomic_chat_queue_manager(
        gateway_config.redis_url,
        gateway_config,
        app.state.gateway_adapter.gateway.throttling_monitor  # Pass throttling monitor
    )
    app.state.acc_binder = get_fast_api_accounting_binder()

    # --- Heartbeats / processor (uses local queue processor) ---
    from kdcube_ai_app.apps.chat.api.resolvers import get_heartbeats_mgr_and_middleware, get_external_request_processor, \
        service_health_checker

    async def agentic_chat_handler(
            message: str,
            session_id: str,
            config: Dict,
            chat_history: Optional[List[Dict[str, str]]] = None,
            conversation_id: Optional[str] = None,
            turn_id: Optional[str] = None,
            envelope: AccountingEnvelope = None,
            emit_data = None,
            **kwargs
    ) -> Dict:
        sio = getattr(getattr(app.state, "socketio_handler", None), "sio", None)
        start_ts = datetime.now().isoformat()
        task_id = kwargs.get("task_id")
        # 1) announce start
        if sio:
            logger.info(f"agentic_chat_handler. session_id={session_id}, task_id={task_id}, message={message[:100]}...")
            await sio.emit("chat_start_handler", {
                "task_id": task_id,
                "message": message[:100] + "..." if len(message) > 100 else message,
                "timestamp": start_ts,
                "user_type": "unknown",
                "queue_stats": {}
            }, room=session_id)

        # 2) build config + workflow with a step emitter
        try:
            cfg_req = ConfigRequest(**(config or {}))
        except Exception:
            cfg_req = ConfigRequest.model_validate(config or {})

        wf_config = create_workflow_config(cfg_req)
        if not emit_data:
            emit_data = {}

        async def _emit_step(step: str, status: str, data: Dict[str, Any]):
            if not sio:
                return
            await sio.emit("chat_step", {
                "step": step,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "elapsed_time": None,
                "error": data.get("error"),
                **emit_data
            }, room=session_id)

        async def _emit_delta(delta: str, index: int, meta: dict):
            if not sio:
                return
            await sio.emit("chat_delta", {
                "delta": delta,
                "index": index,
                "timestamp": datetime.now().isoformat(),
                "meta": meta,
                **emit_data
            }, room=session_id)

        from kdcube_ai_app.infra.plugin.bundle_registry import resolve_bundle
        bundle_id = (config or {}).get("agentic_bundle_id")
        spec_resolved = resolve_bundle(bundle_id, override=None)
        if not spec_resolved:
            logger.info("Using built-in agentic workflow")
            from kdcube_ai_app.apps.chat.default_app.agentic_app import (
                ChatWorkflow, create_initial_state as built_in_create_initial_state
            )
            workflow = ChatWorkflow(wf_config, step_emitter=_emit_step, delta_emitter=_emit_delta)
            create_initial_state_fn = built_in_create_initial_state
        else:
            from kdcube_ai_app.infra.plugin.agentic_loader import AgenticBundleSpec, get_workflow_instance
            logger.info(f"Loading agentic bundle '{spec_resolved.id}' from {spec_resolved.path} module={spec_resolved.module}")
            spec = AgenticBundleSpec(
                path=spec_resolved.path,
                module=spec_resolved.module,
                singleton=bool(spec_resolved.singleton),
            )
            workflow, create_initial_state_fn, _ = get_workflow_instance(
                spec, wf_config, step_emitter=_emit_step, delta_emitter=_emit_delta
            )

        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        # 3) seed optional history into initial state (so first run has context)
        #    (We reuse your create_initial_state and pass it directly to graph)
        call_ctx = {
            "user_message": message,
            "workflow_config": wf_config,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "timestamp": start_ts,
            "config": config,
        }
        state = create_initial_state_fn(call_ctx)

        state["user"] = envelope.user_id
        state["request_id"] = envelope.request_id
        state["session_id"] = session_id
        state["tenant"] = envelope.tenant_id
        state["project"] = envelope.project_id
        state["text"] = message
        state["turn_id"] = turn_id
        state["conversation_id"] = conversation_id

        # Include a lightweight system prompt if passed in config
        system_prompt = (config or {}).get("system_prompt") or "You are a helpful assistant."
        state["messages"] = [SystemMessage(content=system_prompt, id=_mid("sys"))]

        for h in (chat_history or []):
            role = (h.get("role") or "").lower()
            content = h.get("content") or ""
            if not content:
                continue
            state["messages"].append(
                AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))

        state = BundleState(**state)
        # 4) run workflow
        try:
            result = await workflow.run(session_id, conversation_id, state)
        except Exception as e:
            error_message = str(e)
            # try to extract partial state if any
            logger.exception(e)
            result = {
                "final_answer": "I couldn’t complete your request.",
                "error_message": error_message
            }

        # # 5) send complete
        # if sio:
        #     await sio.emit("chat_complete", result, room=session_id)
        return result

    port = CHAT_APP_PORT
    process_id = os.getpid()

    # ================================
    # SOCKET.IO SETUP
    # ================================

    # Create modular Socket.IO chat handler
    try:
        socketio_handler = create_socketio_chat_handler(
            app=app,
            gateway_adapter=app.state.gateway_adapter,
            chat_queue_manager=app.state.chat_queue_manager,
            allowed_origins=allowed_origins,
            instance_id=INSTANCE_ID,
            redis_url=REDIS_URL
        )

        # Mount Socket.IO app if available
        socket_asgi_app = socketio_handler.get_asgi_app()
        if socket_asgi_app:
            app.mount("/socket.io", socket_asgi_app)
            app.state.socketio_handler = socketio_handler
            logger.info("Socket.IO chat handler mounted successfully")
        else:
            logger.warning("Socket.IO not available - chat handler disabled")

    except Exception as e:
        logger.error(f"Failed to setup Socket.IO chat handler: {e}")
        app.state.socketio_handler = None

    try:
        handler = agentic_chat_handler

        middleware, heartbeat_manager = get_heartbeats_mgr_and_middleware(port=port)
        processor = get_external_request_processor(middleware, handler, app)
        health_checker = service_health_checker(middleware)

        # Store in app state for monitoring endpoints
        app.state.middleware = middleware
        app.state.heartbeat_manager = heartbeat_manager
        app.state.processor = processor
        app.state.health_checker = health_checker

        # Start services
        await middleware.init_redis()
        await heartbeat_manager.start_heartbeat(interval=10)

        try:
            from kdcube_ai_app.infra.plugin.bundle_store import load_registry as _load_store_registry
            from kdcube_ai_app.infra.plugin.bundle_registry import set_registry as _set_mem_registry
            reg = await _load_store_registry(middleware.redis)
            bundles_dict = {bid: entry.model_dump() for bid, entry in reg.bundles.items()}
            _set_mem_registry(bundles_dict, reg.default_bundle_id)
            logger.info(f"Bundles registry loaded from Redis: {len(bundles_dict)} items (default={reg.default_bundle_id})")
        except Exception as e:
            logger.warning(f"Failed to load bundles registry from Redis; using env-only registry. {e}")

        await processor.start_processing()
        await health_checker.start_monitoring()

        logger.info(f"Chat process {process_id} started with enhanced gateway")

    except Exception as e:
        logger.warning(f"Could not start legacy middleware: {e}")

    yield

    # Shutdown
    if hasattr(app.state, 'heartbeat_manager'):
        await app.state.heartbeat_manager.stop_heartbeat()
    if hasattr(app.state, 'processor'):
        await app.state.processor.stop_processing()
    if hasattr(app.state, 'health_checker'):
        await app.state.health_checker.stop_monitoring()

    logger.info("Chat service stopped")


# Create FastAPI app
app = FastAPI(
    title="Chat API with Modular Socket.IO",
    description="Chat API with gateway integration and modular real-time Socket.IO streaming",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create orchestrator instance
orchestrator: IOrchestrator = get_orchestrator()


# ================================
# MIDDLEWARE
# ================================

@app.middleware("http")
async def gateway_middleware(request: Request, call_next):
    if request.url.path.startswith(("/profile", "/monitoring", "/admin", "/health", "/docs", "/openapi.json", "/favicon.ico")):
        return await call_next(request)

    # If already processed by a dependency earlier in the chain (rare but safe), skip
    if getattr(request.state, STATE_FLAG, False):
        return await call_next(request)

    try:
        session = await app.state.gateway_adapter.process_request(request, [])
        setattr(request.state, STATE_SESSION, session)
        setattr(request.state, STATE_USER_TYPE, session.user_type.value)
        setattr(request.state, STATE_FLAG, True)

        response = await call_next(request)

        # Add headers once
        response.headers["X-User-Type"] = session.user_type.value
        response.headers["X-Session-ID"] = session.session_id
        return response
    except HTTPException as e:
        headers = getattr(e, "headers", {})
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail}, headers=headers)


# ================================
# REQUEST/RESPONSE MODELS
# ================================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    config: Optional[ConfigRequest] = {}
    chat_history: Optional[List[ChatMessage]] = []  # <— added


class ChatResponse(BaseModel):
    status: str
    task_id: str
    session_id: str
    user_type: str
    message: str


# ================================
# UTILITY FUNCTIONS
# ================================

def convert_chat_history(chat_history: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert Pydantic chat history to dict format"""
    return [
        {
            "role": msg["role"] if isinstance(msg, dict) else msg.role,
            "content": msg["content"] if isinstance(msg, dict) else msg.content,
            "timestamp": msg.get("timestamp") if isinstance(msg, dict) else (
                        msg.timestamp or datetime.now().isoformat())
        }
        for msg in (chat_history or [])
    ]


# ================================
# ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    socketio_enabled = hasattr(app.state, 'socketio_handler') and app.state.socketio_handler is not None

    return {
        "name": "KDCube AI App Platform",
        "version": "3.0.0",
        "description": "Multitenant hosting for your AI applications",
        "features": [
        ],
        "available_models": list(MODEL_CONFIGS.keys()),
        "socketio_enabled": socketio_enabled,
        "endpoints": {
            "/landing/health": "Health check",
            "/landing/chat": "Legacy sync chat endpoint",
            "/landing/models": "Get available models",
            "/landing/test-embeddings": "Test custom embedding endpoint",
            "/landing/workflow-info": "Get workflow information",
            "/socket.io": "Socket.IO endpoint for real-time chat" if socketio_enabled else "Socket.IO disabled"
        }
    }


@app.post("/landing/test-embeddings")
async def check_embeddings_endpoint(request: ConfigRequest,
                                    session: UserSession = Depends(auth_without_pressure())):
    """Test embedding configuration"""
    try:
        from kdcube_ai_app.apps.chat.sdk.inventory import probe_embeddings
        return probe_embeddings(request)
    except Exception as e:
        import traceback
        logger.error(f"Error testing embeddings: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Error testing embeddings: {str(e)}",
                "embedder_id": request.selected_embedder
            }
        )


@app.get("/profile")
# think of replacing with auth_without_pressure
async def get_profile(session: UserSession = Depends(get_user_session_dependency())):
    """Get user profile - works for both anonymous and registered users"""
    if session.user_type in [UserType.REGISTERED, UserType.PRIVILEGED]:
        return {
            "user_type": "registered" if session.user_type == UserType.REGISTERED else "privileged",
            "username": session.username,
            "user_id": session.user_id,
            "roles": session.roles,
            "permissions": session.permissions,
            "session_id": session.session_id,
            "created_at": session.created_at
        }
    else:
        return {
            "user_type": "anonymous",
            "fingerprint": session.fingerprint[:8] + "...",
            "session_id": session.session_id,
            "created_at": session.created_at
        }


@app.post("/landing/chat")
async def chat_endpoint(
        payload: ChatRequest,  # rename to avoid shadowing fastapi.Request
        session: UserSession = Depends(get_user_session_dependency())
):
    """Main chat endpoint - supports both anonymous and registered users.
       Only enqueues; worker binds accounting from attached envelope."""
    try:
        # IDs
        task_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        session_id = payload.session_id or session.session_id

        # Build an accounting snapshot (no storage I/O here)
        # Try to infer tenant / project if you carry them in config; safe to leave None.
        cfg_dict = payload.config.model_dump() if payload.config else {}
        project_id = cfg_dict.get("project")
        # If you have a central gateway config, you can fetch tenant like this:
        try:
            from kdcube_ai_app.infra.gateway.config import get_gateway_config
            tenant_id = get_gateway_config().tenant_id
        except Exception:
            tenant_id = None

        acct_env = build_envelope_from_session(
            session=session,
            tenant_id=tenant_id,
            project_id=project_id,
            request_id=request_id,
            component="chat.rest",
            metadata={
                "entrypoint": "/landing/chat",
                "selected_model": cfg_dict.get("selected_model")
            },
        )

        # Prepare task payload for the orchestrator/worker
        task_data = {
            "task_id": task_id,
            "message": payload.message,
            "session_id": session_id,
            "config": cfg_dict,
            "chat_history": convert_chat_history(payload.chat_history or []),
            "user_type": session.user_type.value,
            "user_info": {
                "user_id": session.user_id,
                "username": session.username,
                "fingerprint": session.fingerprint,
                "roles": session.roles,
                "permissions": session.permissions,
            },
            "created_at": time.time(),
            "instance_id": INSTANCE_ID,
            "acct": acct_env.to_dict(),  # <— accounting envelope for downstream worker
            "kdcube_path": os.environ.get("KDCUBE_STORAGE_PATH"),
            "target_room": session_id  # <— explicit for clarity
        }

        # Use the gateway-provided context we stored in the session at auth time
        context = session.request_context

        # Atomic enqueue with backpressure protection
        chat_queue_manager = app.state.chat_queue_manager
        success, reason, stats = await chat_queue_manager.enqueue_chat_task_atomic(
            session.user_type,
            task_data,
            session,
            context,
            "/landing/chat"
        )

        if not success:
            retry_after = 30 if "anonymous" in reason else 45 if "registered" in reason else 60
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "System under pressure",
                    "reason": reason,
                    "retry_after": retry_after,
                    "stats": stats
                },
                headers={"Retry-After": str(retry_after)}
            )

        # Keep the existing response shape for compatibility
        return ChatResponse(
            status="processing_started",
            task_id=task_id,
            session_id=session_id,
            user_type=session.user_type.value,
            message=f"Request queued for {session.user_type.value} user"
        )

    except CircuitBreakerError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable",
                "reason": f"Circuit breaker '{e.circuit_name}' is open",
                "retry_after": e.retry_after,
                "circuit_breaker": e.circuit_name
            },
            headers={"Retry-After": str(e.retry_after)}
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/landing/models")
async def get_available_models(session: UserSession = Depends(get_user_session_dependency())):
    """Get available model configurations"""
    return {
        "available_models": {
            model_id: {
                "id": model_id,
                "name": config["model_name"],
                "provider": config["provider"],
                "has_classifier": config["has_classifier"],
                "description": config["description"]
            }
            for model_id, config in MODEL_CONFIGS.items()
        },
        "default_model": "gpt-4o"
    }


@app.get("/landing/embedders")
async def get_available_embedders(session: UserSession = Depends(get_user_session_dependency())):
    """Get available embedding configurations"""
    available_embedders = {
        "available_embedders": {
            embedder_id: {
                "id": embedder_id,
                "provider": config["provider"],
                "model": config["model_name"],
                "dimension": config["dim"],
                "description": config["description"]
            }
            for embedder_id, config in EMBEDDERS.items()
        },
        "default_embedder": "openai-text-embedding-3-small",
        "providers": {
            "openai": {
                "name": "OpenAI",
                "description": "OpenAI's embedding models",
                "requires_api_key": True,
                "requires_endpoint": False
            },
            "custom": {
                "name": "Custom/HuggingFace",
                "description": "Custom embedding endpoints (HuggingFace, etc.)",
                "requires_api_key": False,
                "requires_endpoint": True
            }
        }
    }
    return available_embedders


# ================================
# MONITORING ENDPOINTS
# ================================

@app.get("/health")
async def health_check():
    """Basic health check"""
    socketio_status = "enabled" if hasattr(app.state, 'socketio_handler') and app.state.socketio_handler else "disabled"

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "instance_id": INSTANCE_ID,
        "port": CHAT_APP_PORT,
        "socketio_status": socketio_status,
        "modular_architecture": True
    }

@app.get("/debug/session")
async def debug_session(session: UserSession = Depends(get_user_session_dependency())):
    """Debug endpoint to see current session"""
    return {
        "session": session.__dict__,
        "user_type": session.user_type.value
    }

# ================================
# ERROR HANDLERS
# ================================

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    """Enhanced exception handler that records circuit breaker failures"""
    logger.exception(f"Unhandled exception in {request.url.path}: {exc}")

    # Record failure in appropriate circuit breakers if it's a service error
    if hasattr(app.state, 'gateway_adapter'):
        try:
            # You could record failures in relevant circuit breakers here
            # based on the type of exception and endpoint
            pass
        except Exception as cb_error:
            logger.error(f"Error recording circuit breaker failure: {cb_error}")

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


@app.exception_handler(CircuitBreakerError)
async def circuit_breaker_exception_handler(request: Request, exc: CircuitBreakerError):
    """Handle circuit breaker errors gracefully"""
    logger.warning(f"Circuit breaker '{exc.circuit_name}' blocked request to {request.url.path}")

    return JSONResponse(
        status_code=503,
        content={
            "detail": "Service temporarily unavailable due to circuit breaker",
            "circuit_breaker": exc.circuit_name,
            "retry_after": exc.retry_after,
            "message": "The service is experiencing issues and is temporarily unavailable. Please try again later."
        },
        headers={"Retry-After": str(exc.retry_after)}
    )


# Mount monitoring routers
from kdcube_ai_app.apps.chat.api.monitoring import mount_monitoring_routers
mount_monitoring_routers(app)

# Mount integrations router
from kdcube_ai_app.apps.chat.api.integrations import mount_integrations_routers
mount_integrations_routers(app)

# ================================
# RUN APPLICATION
# ================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=CHAT_APP_PORT,
        log_level="info"
    )
