# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# socketio/chat.py
"""
Modular Socket.IO chat handler with gateway integration
"""
import os
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import socketio
from pydantic import BaseModel

from kdcube_ai_app.auth.sessions import UserSession, UserType, RequestContext
from kdcube_ai_app.infra.accounting.envelope import build_envelope_from_session
from kdcube_ai_app.infra.gateway.rate_limiter import RateLimitError
from kdcube_ai_app.infra.gateway.backpressure import BackpressureError
from kdcube_ai_app.infra.gateway.circuit_breaker import CircuitBreakerError

from kdcube_ai_app.apps.chat.inventory import create_workflow_config, ConfigRequest

logger = logging.getLogger(__name__)

class StepUpdate(BaseModel):
    step: str
    status: str  # "started", "completed", "error"
    timestamp: str
    data: Optional[Dict[str, Any]] = None
    elapsed_time: Optional[str] = None
    error: Optional[str] = None


class SocketIOChatHandler:
    """Socket.IO chat handler with gateway integration"""

    def __init__(self, app, gateway_adapter, chat_queue_manager, allowed_origins, instance_id, redis_url):
        self.app = app
        self.gateway_adapter = gateway_adapter
        self.chat_queue_manager = chat_queue_manager
        self.allowed_origins = allowed_origins
        self.instance_id = instance_id

        self.redis_url = redis_url

        # Create Socket.IO server
        self.sio = self._create_socketio_server()
        self._setup_event_handlers()

    def _create_socketio_server(self):
        """Create Socket.IO server with Redis manager"""
        try:
            mgr = socketio.AsyncRedisManager(self.redis_url)
            logger.info(f"Using Redis manager for Socket.IO: {self.redis_url}")

            sio = socketio.AsyncServer(
                cors_allowed_origins=self.allowed_origins,
                async_mode='asgi',
                client_manager=mgr,
                logger=True,
                engineio_logger=True
            )
            return sio

        except ImportError:
            logger.warning("socketio not installed. Socket.IO support disabled.")
            return None

    def _setup_event_handlers(self):
        """Setup Socket.IO event handlers"""
        if not self.sio:
            return

        @self.sio.on('connect')
        async def handle_connect(sid, environ, auth):
            return await self._handle_connect(sid, environ, auth)

        @self.sio.on('disconnect')
        async def handle_disconnect(sid):
            return await self._handle_disconnect(sid)

        @self.sio.on('chat_message')
        async def handle_chat_message(sid, data):
            return await self._handle_chat_message(sid, data)

        @self.sio.on('ping')
        async def handle_ping(sid, data):
            return await self._handle_ping(sid, data)

    async def _handle_connect(self, sid, environ, auth):
        """
        WebSocket connect handler.

        Contract:
          - Clients MUST pass `auth.user_session_id` obtained from REST (/profile).
          - Optional: `auth.bearer_token` can be supplied for extra validation; if present and
            the decoded user does not match the stored session, the connection is rejected.
          - Optional: `auth.project`, `auth.tenant` can be sent and will be stored alongside
            the socket session (not used for auth).
        """
        logger.info(f"WS connect attempt: sid={sid}")

        # ---- 0) Allowlist origin check ---------------------------------------------------------
        origin = environ.get("HTTP_ORIGIN")
        if self.allowed_origins not in (None, [], ["*"]):
            if not origin or (origin not in self.allowed_origins and "*" not in self.allowed_origins):
                logger.warning(f"WS connect rejected for {sid}: origin '{origin}' not allowed")
                return False
        logger.debug(f"WS origin accepted: {origin}")

        # ---- 2) Require a REST-provisioned session id ------------------------------------------
        user_session_id = (auth or {}).get("user_session_id")
        if not user_session_id:
            logger.warning(f"WS connect rejected for {sid}: missing user_session_id")
            return False

        # ---- 3) Load the existing session from session manager ---------------------------------
        try:
            session = await self.gateway_adapter.gateway.session_manager.get_session_by_id(user_session_id)
            if not session:
                logger.warning(f"WS connect rejected for {sid}: unknown session_id={user_session_id}")
                return False
        except Exception as e:
            logger.error(f"WS connect failed to load session {user_session_id} for {sid}: {e}")
            return False

        # ---- 4) Optional bearer token validation against the loaded session --------------------
        # If a token is supplied, verify it belongs to this session's user (for registered/privileged).
        try:
            bearer_token = (auth or {}).get("bearer_token")
            id_token = (auth or {}).get("id_token")

            if bearer_token and session.user_type.value != "anonymous":
                # Authenticate token via the gateway's auth_manager
                user = await self.gateway_adapter.gateway.auth_manager.authenticate_with_both(bearer_token, id_token)
                claimed_user_id = getattr(user, "sub", None) or user.username
                if session.user_id and claimed_user_id and session.user_id != claimed_user_id:
                    logger.warning(
                        f"WS connect rejected for {sid}: token user_id '{claimed_user_id}' "
                        f"does not match session user_id '{session.user_id}'"
                    )
                    return False
        except Exception as e:
            logger.error(f"WS bearer token validation failed for {sid}: {e}")
            return False

        # ---- 5) Apply gateway protections (rate limit + backpressure) for connect --------------
        try:
            # Build a RequestContext for recording purposes
            client_ip = environ.get("REMOTE_ADDR", environ.get("HTTP_X_FORWARDED_FOR", "unknown"))
            user_agent = environ.get("HTTP_USER_AGENT", "")
            auth_header = f"Bearer {bearer_token}" if (bearer_token) else None

            context = RequestContext(
                client_ip=client_ip,
                user_agent=user_agent,
                authorization_header=auth_header
            )

            # Use the managers directly (do NOT call process_request which might create a new session)
            endpoint = "/socket.io/connect"
            await self.gateway_adapter.gateway.rate_limiter.check_and_record(session, context, endpoint)
            await self.gateway_adapter.gateway.backpressure_manager.check_capacity(
                session.user_type, session, context, endpoint
            )

            logger.info(f"WS connect gateway checks passed: sid={sid}, session={session.session_id}, type={session.user_type.value}")

        except RateLimitError as e:
            logger.warning(f"WS connect rate-limited: sid={sid}, detail={e.message}")
            # Emit a one-off error to the new socket and reject connection
            try:
                await self.sio.emit("chat_error", {
                    "error": f"Rate limit exceeded: {e.message}",
                    "retry_after": e.retry_after,
                    "timestamp": datetime.now().isoformat()
                }, to=sid)
            except Exception:
                pass
            return False

        except BackpressureError as e:
            logger.warning(f"WS connect rejected by backpressure: sid={sid}, detail={e.message}")
            try:
                await self.sio.emit("chat_error", {
                    "error": f"System under pressure: {e.message}",
                    "retry_after": e.retry_after,
                    "timestamp": datetime.now().isoformat()
                }, to=sid)
            except Exception:
                pass
            return False

        except CircuitBreakerError as e:
            logger.warning(f"WS connect rejected by circuit breaker: sid={sid}, circuit={e.circuit_name}")
            try:
                await self.sio.emit("chat_error", {
                    "error": f"Service temporarily unavailable: {e.message}",
                    "retry_after": e.retry_after,
                    "timestamp": datetime.now().isoformat()
                }, to=sid)
            except Exception:
                pass
            return False

        except Exception as e:
            logger.error(f"WS connect gateway check failed for {sid}: {e}")
            try:
                await self.sio.emit("chat_error", {
                    "error": "System check failed",
                    "timestamp": datetime.now().isoformat()
                }, to=sid)
            except Exception:
                pass
            return False

        # ---- 6) Save socket session & join room = session_id -----------------------------------
        try:
            # Persist a compact session dict + any useful client-provided tags (project/tenant)
            session_dict = session.serialize_to_dict()
            socket_meta = {
                "user_session": session_dict,
                "authenticated": session.user_type.value != "anonymous",
                "project": (auth or {}).get("project"),
                "tenant": (auth or {}).get("tenant"),
            }
            await self.sio.save_session(sid, socket_meta)

            # Join the per-session room for later REST->WS emissions
            await self.sio.enter_room(sid, session.session_id)

            # Send a small ack to the client
            await self.sio.emit("session_info", {
                "session_id": session.session_id,
                "user_type": session.user_type.value,
                "user_id": session.user_id,
                "username": session.username,
                "project": socket_meta.get("project"),
                "tenant": socket_meta.get("tenant"),
                "connected_at": datetime.now().isoformat()
            }, to=sid)

            logger.info(f"WS connected: sid={sid} attached to session={session.session_id} (room joined)")
            return True

        except Exception as e:
            logger.error(f"WS connect finalization failed for {sid}: {e}")
            return False


    async def _handle_disconnect(self, sid):
        """Handle Socket.IO disconnection"""
        logger.info(f'Chat client disconnected: {sid}')

    async def _handle_chat_message(self, sid, data):
        """
        Handle a chat message from a Socket.IO client.

        - Validates payload
        - Reconstructs user session (from socket session)
        - Runs gateway checks (rate limit, backpressure, circuit breakers)
        - Builds an AccountingEnvelope snapshot and attaches it to the task
        - Enqueues the task for the orchestrator to process later
        - Emits 'chat_start' (ack) and 'chat_queued'
        """
        logger.info(f"Received chat message from {sid}: {data.get('message', '')[:100]}...")

        try:
            # 1) Pull the session we saved at connect time
            socket_session = await self.sio.get_session(sid)
            user_session_data = socket_session.get('user_session', {})

            # 2) Basic validation
            if not data or "message" not in data or "config" not in data:
                error_msg = 'Missing message or config in request'
                logger.error(f"Chat validation error for {sid}: {error_msg}")
                await self.sio.emit('chat_error', {
                    'error': error_msg
                }, room=sid)
                return

            message = data["message"]
            chat_history = data.get("chat_history", [])
            config_data = data["config"]

            # 3) Rebuild our UserSession (lightweight)
            session = UserSession(
                session_id=user_session_data.get('session_id', 'unknown'),
                user_type=UserType(user_session_data.get('user_type', 'anonymous')),
                fingerprint=user_session_data.get('fingerprint', 'unknown'),
                user_id=user_session_data.get('user_id'),
                username=user_session_data.get('username'),
                roles=user_session_data.get('roles', []),
                permissions=user_session_data.get('permissions', [])
            )

            # 4) Build a RequestContext for gateway checks
            context = RequestContext(
                client_ip="socket.io",
                user_agent="socket.io-client",
                authorization_header=None  # Already authenticated at connect time
            )

            # 5) Gateway protection (rate-limit + backpressure)
            try:
                await self.gateway_adapter.gateway.rate_limiter.check_and_record(
                    session, context, "/socket.io/chat"
                )
                await self.gateway_adapter.gateway.backpressure_manager.check_capacity(
                    session.user_type, session, context, "/socket.io/chat"
                )
                logger.info(f"Gateway check passed for socket {sid} ({session.user_type.value})")

            except RateLimitError as e:
                logger.warning(f"Rate limit exceeded for socket {sid}: {e.message}")
                await self.sio.emit('chat_error', {
                    'error': f'Rate limit exceeded: {e.message}',
                    'retry_after': e.retry_after,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                return

            except BackpressureError as e:
                logger.warning(f"Backpressure limit exceeded for socket {sid}: {e.message}")
                await self.sio.emit('chat_error', {
                    'error': f'System under pressure: {e.message}',
                    'retry_after': e.retry_after,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                return

            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker open for socket {sid}: {e.message}")
                await self.sio.emit('chat_error', {
                    'error': f'Service temporarily unavailable: {e.message}',
                    'retry_after': e.retry_after,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                return

            except Exception as e:
                logger.error(f"Gateway check failed for socket {sid}: {e}")
                await self.sio.emit('chat_error', {
                    'error': 'System check failed',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                return

            # 6) Parse config (only to capture fields; execution is offloaded)
            config_request = ConfigRequest(**config_data)
            if not config_request.selected_model:
                config_request.selected_model = ""

            # Optional: infer project/tenant for accounting if you carry them in config
            project_id = getattr(config_request, "project", None) or data.get("project")
            tenant_id = getattr(self.gateway_adapter.gateway.gateway_config, "tenant_id", None)

            # 7) Convert history to normalized list
            history_dict = self._convert_chat_history(chat_history)

            # 8) Build the AccountingEnvelope snapshot (we attach this to the task)
            request_id = str(uuid.uuid4())
            acct_envelope = build_envelope_from_session(
                session=session,
                tenant_id=tenant_id,
                project_id=project_id,
                request_id=request_id,
                component="chat.socket",   # base component for socket-level enqueues
                metadata={
                    "socket_id": sid,
                    "entrypoint": "/socket.io/chat",
                    "selected_model": getattr(config_request, "selected_model", None),
                },
            )

            # 9) Prepare task payload for orchestrator
            task_id = str(uuid.uuid4())
            task_data = {
                "task_id": task_id,
                "message": message,
                "session_id": session.session_id,
                "config": config_request.model_dump(),
                "chat_history": history_dict,
                "user_type": session.user_type.value,
                "user_info": {
                    "user_id": session.user_id,
                    "username": session.username,
                    "fingerprint": session.fingerprint,
                    "roles": session.roles,
                    "permissions": session.permissions
                },
                "created_at": time.time(),
                "instance_id": self.instance_id,
                "socket_id": sid,                          # so the worker can emit back to this client
                "acct": acct_envelope.to_dict(),           # <-- accounting snapshot for the worker
                "kdcube_path": os.environ.get("KDCUBE_STORAGE_PATH"),  # worker uses this to init storage backend
            }

            # 10) Atomic enqueue with backpressure accounting
            success, reason, stats = await self.chat_queue_manager.enqueue_chat_task_atomic(
                session.user_type,
                task_data,
                session,
                context,
                "/socket.io/chat"
            )

            if not success:
                retry_after = 30 if "anonymous" in reason else 45 if "registered" in reason else 60
                logger.warning(f"Chat task rejected for socket {sid}: {reason}")
                await self.sio.emit('chat_error', {
                    'error': 'System under pressure - request rejected',
                    'reason': reason,
                    'retry_after': retry_after,
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                return

            # 11) Acknowledge to client (no processing here)
            start_data = {
                'task_id': task_id,
                'message': message[:100] + "..." if len(message) > 100 else message,
                'timestamp': datetime.now().isoformat(),
                'user_type': session.user_type.value,
                'queue_stats': stats
            }
            logger.info(f"Sending chat_start to {sid}")
            await self.sio.emit('chat_start', start_data, room=sid)

            # Optional: a second lightweight event to include IDs useful to the client
            await self.sio.emit('chat_queued', {
                'task_id': task_id,
                'request_id': request_id,
                'tenant_id': tenant_id,
                'project_id': project_id,
                'selected_model': getattr(config_request, "selected_model", None),
                'timestamp': datetime.now().isoformat()
            }, room=sid)

            logger.info(f"Chat task {task_id} enqueued successfully for socket {sid}")

            # NOTE: The orchestrator/worker will later:
            #  - bind accounting using the attached 'acct' payload,
            #  - run your workflow,
            #  - emit 'chat_step' and 'chat_complete' back to this 'socket_id'.

        except Exception as e:
            import traceback
            logger.error(f"Error processing chat message for {sid}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            try:
                await self.sio.emit('chat_error', {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                logger.info(f"Error message sent to {sid}")
            except Exception as emit_error:
                logger.error(f"Failed to send error message to {sid}: {str(emit_error)}")

    async def _handle_ping(self, sid, data):
        """Handle ping for connection testing"""
        await self.sio.emit('pong', {'timestamp': datetime.now().isoformat()}, room=sid)

    def _convert_chat_history(self, chat_history):
        """Convert chat history to dict format"""
        return [
            {
                "role": msg["role"] if isinstance(msg, dict) else msg.role,
                "content": msg["content"] if isinstance(msg, dict) else msg.content,
                "timestamp": msg.get("timestamp") if isinstance(msg, dict) else (msg.timestamp or datetime.now().isoformat())
            }
            for msg in chat_history
        ]

    def get_asgi_app(self):
        """Get the Socket.IO ASGI app for mounting"""
        if self.sio:
            return socketio.ASGIApp(self.sio)
        return None


def create_socketio_chat_handler(app, gateway_adapter, chat_queue_manager, allowed_origins, instance_id, redis_url):
    """Factory function to create Socket.IO chat handler"""
    return SocketIOChatHandler(
        app=app,
        gateway_adapter=gateway_adapter,
        chat_queue_manager=chat_queue_manager,
        allowed_origins=allowed_origins,
        instance_id=instance_id,
        redis_url=redis_url
    )