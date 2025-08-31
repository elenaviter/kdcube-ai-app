# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/api/socketio/chat.py
"""
Modular Socket.IO chat handler with gateway integration and Redis relay.
Redis pub/sub listener relays chat events (chat.events) to clients
uses a standardized ChatTaskPayload schema (chat/sdk/protocol.py).
"""

from __future__ import annotations
import os
import uuid
import time
import logging
from datetime import datetime
from typing import Any, Optional, Dict

import socketio

from kdcube_ai_app.apps.chat.api.resolvers import get_tenant
from kdcube_ai_app.auth.sessions import UserSession, UserType, RequestContext
from kdcube_ai_app.infra.accounting.envelope import build_envelope_from_session
from kdcube_ai_app.infra.gateway.rate_limiter import RateLimitError
from kdcube_ai_app.infra.gateway.backpressure import BackpressureError
from kdcube_ai_app.infra.gateway.circuit_breaker import CircuitBreakerError

from kdcube_ai_app.apps.chat.sdk.protocol import (
    ChatTaskPayload, ChatTaskMeta, ChatTaskRouting, ChatTaskActor, ChatTaskUser,
    ChatTaskRequest, ChatTaskConfig, ChatTaskAccounting,
    ServiceCtx, ConversationCtx,
)
from kdcube_ai_app.apps.chat.emitters import ChatRelayCommunicator

logger = logging.getLogger(__name__)


class SocketIOChatHandler:
    """
    Socket.IO chat handler with FULL gateway gating + Redis relay.
    Emits to clients only via ChatRelayCommunicator (same one the processor uses).
    """

    def __init__(
        self,
        app,
        gateway_adapter,
        chat_queue_manager,
        allowed_origins,
        instance_id: str,
        redis_url: str,
        chat_comm: ChatRelayCommunicator,   # ← SAME communicator used by processor
    ):
        self.app = app
        self.gateway_adapter = gateway_adapter
        self.chat_queue_manager = chat_queue_manager
        self.allowed_origins = allowed_origins
        self.instance_id = instance_id
        self.redis_url = redis_url

        self._comm = chat_comm
        self._listener_started = False

        self.sio = self._create_socketio_server()
        self._setup_event_handlers()

    # ---------- Socket.IO core ----------

    def _create_socketio_server(self):
        try:
            mgr = socketio.AsyncRedisManager(self.redis_url)
            sio = socketio.AsyncServer(
                cors_allowed_origins=self.allowed_origins,
                async_mode="asgi",
                client_manager=mgr,
                logger=True,
                engineio_logger=True,
            )
            return sio
        except Exception as e:
            logger.exception("Socket.IO init failed: %s", e)
            return None

    def _setup_event_handlers(self):
        if not self.sio:
            return

        @self.sio.on("connect")
        async def _on_connect(sid, environ, auth):
            return await self._handle_connect(sid, environ, auth)

        @self.sio.on("disconnect")
        async def _on_disconnect(sid):
            return await self._handle_disconnect(sid)

        @self.sio.on("chat_message")
        async def _on_chat_message(sid, data):
            return await self._handle_chat_message(sid, data)

        @self.sio.on("ping")
        async def _on_ping(sid, data):
            await self.sio.emit("pong", {"timestamp": datetime.utcnow().isoformat() + "Z"}, to=sid)

    # ---------- Relay (pub/sub -> socket) ----------

    async def _on_pubsub_message(self, message: dict):
        """
        Relay events published by workers/processors to connected sockets.
        { event, data, target_sid?, session_id? }
        """
        try:
            event = message.get("event")
            data = message.get("data") or {}
            target_sid = message.get("target_sid")
            session_id = message.get("session_id")
            if not event:
                return
            if target_sid:
                await self.sio.emit(event, data, room=target_sid)
            elif session_id:
                await self.sio.emit(event, data, room=session_id)
        except Exception as e:
            logger.error("[chat relay] emit failed: %s", e)

    async def start(self):
        if self._listener_started or not self.sio:
            return
        await self._comm.subscribe(self._on_pubsub_message)
        self._listener_started = True
        logger.info("Socket.IO chat handler subscribed to relay channel.")

    async def stop(self):
        if not self._listener_started:
            return
        await self._comm.unsubscribe()
        self._listener_started = False

    # ---------- CONNECT with GATING (restored) ----------

    async def _handle_connect(self, sid, environ, auth):
        logger.info("WS connect attempt sid=%s", sid)

        # origin allowlist
        origin = environ.get("HTTP_ORIGIN")
        if self.allowed_origins not in (None, [], ["*"]):
            if not origin or (origin not in self.allowed_origins and "*" not in self.allowed_origins):
                logger.warning("WS connect rejected: origin '%s' not allowed", origin)
                return False

        user_session_id = (auth or {}).get("user_session_id")
        if not user_session_id:
            logger.warning("WS connect rejected: missing user_session_id")
            return False

        # load session
        try:
            session = await self.gateway_adapter.gateway.session_manager.get_session_by_id(user_session_id)
            if not session:
                logger.warning("WS connect rejected: unknown session_id=%s", user_session_id)
                return False
        except Exception as e:
            logger.error("WS connect failed to load session %s: %s", user_session_id, e)
            return False

        # optional bearer validation (registered/privileged)
        try:
            bearer_token = (auth or {}).get("bearer_token")
            id_token = (auth or {}).get("id_token")
            if bearer_token and session.user_type.value != "anonymous":
                user = await self.gateway_adapter.gateway.auth_manager.authenticate_with_both(bearer_token, id_token)
                claimed_user_id = getattr(user, "sub", None) or user.username
                if session.user_id and claimed_user_id and session.user_id != claimed_user_id:
                    logger.warning(
                        "WS connect rejected: bearer user_id '%s' != session user_id '%s'",
                        claimed_user_id, session.user_id
                    )
                    return False
        except Exception as e:
            logger.error("WS bearer validation failed: %s", e)
            return False

        # gateway protections (rate-limit + backpressure) — tracked on this endpoint
        try:
            client_ip = environ.get("REMOTE_ADDR", environ.get("HTTP_X_FORWARDED_FOR", "unknown"))
            user_agent = environ.get("HTTP_USER_AGENT", "")
            auth_header = f"Bearer {(auth or {}).get('bearer_token')}" if (auth or {}).get("bearer_token") else None

            context = RequestContext(client_ip=client_ip, user_agent=user_agent, authorization_header=auth_header)
            endpoint = "/socket.io/connect"

            await self.gateway_adapter.gateway.rate_limiter.check_and_record(session, context, endpoint)
            await self.gateway_adapter.gateway.backpressure_manager.check_capacity(
                session.user_type, session, context, endpoint
            )
            logger.info("WS connect gateway checks passed: sid=%s session=%s type=%s",
                        sid, session.session_id, session.user_type.value)

        except RateLimitError as e:
            logger.warning("WS connect rate-limited: %s", e.message)
            try:
                await self.sio.emit("chat_error", {
                    "error": f"Rate limit exceeded: {e.message}",
                    "retry_after": e.retry_after,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, to=sid)
            finally:
                return False

        except BackpressureError as e:
            logger.warning("WS connect backpressure: %s", e.message)
            try:
                await self.sio.emit("chat_error", {
                    "error": f"System under pressure: {e.message}",
                    "retry_after": e.retry_after,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, to=sid)
            finally:
                return False

        except CircuitBreakerError as e:
            logger.warning("WS connect circuit breaker '%s'", e.circuit_name)
            try:
                await self.sio.emit("chat_error", {
                    "error": f"Service temporarily unavailable: {e.message}",
                    "retry_after": e.retry_after,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, to=sid)
            finally:
                return False

        except Exception as e:
            logger.error("WS connect gateway failure: %s", e)
            try:
                await self.sio.emit("chat_error", {
                    "error": "System check failed",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }, to=sid)
            finally:
                return False

        # save socket session & join per-session room
        try:
            socket_meta = {
                "user_session": session.serialize_to_dict(),
                "authenticated": session.user_type.value != "anonymous",
                "project": (auth or {}).get("project"),
                "tenant": (auth or {}).get("tenant"),
            }
            await self.sio.save_session(sid, socket_meta)
            await self.sio.enter_room(sid, session.session_id)

            await self.sio.emit("session_info", {
                "session_id": session.session_id,
                "user_type": session.user_type.value,
                "user_id": session.user_id,
                "username": session.username,
                "project": socket_meta.get("project"),
                "tenant": socket_meta.get("tenant"),
                "connected_at": datetime.utcnow().isoformat() + "Z"
            }, to=sid)

            logger.info("WS connected: sid=%s -> room=%s", sid, session.session_id)
            return True

        except Exception as e:
            logger.error("WS connect finalization failed: %s", e)
            return False

    async def _handle_disconnect(self, sid):
        logger.info("Chat client disconnected: %s", sid)

    # ---------- CHAT MESSAGE with GATING (restored) ----------

    async def _handle_chat_message(self, sid, data: Dict[str, Any]):
        logger.info("chat_message sid=%s '%s'...", sid, (data or {}).get("message", "")[:100])
        try:
            socket_session = await self.sio.get_session(sid)
            user_session_data = (socket_session or {}).get("user_session", {})

            # basic validation
            message = (data or {}).get("text") or (data or {}).get("message")
            if not message:
                svc = ServiceCtx(request_id=str(uuid.uuid4()))
                conv = ConversationCtx(
                    session_id=user_session_data.get("session_id", "unknown"),
                    conversation_id=data.get("conversation_id") or user_session_data.get("session_id", "unknown"),
                    turn_id=f"turn_{uuid.uuid4().hex[:8]}",
                )
                self._comm.emit_error(svc, conv, error='Missing "message"', target_sid=sid, session_id=conv.session_id)
                return

            # rebuild lightweight UserSession
            session = UserSession(
                session_id=user_session_data.get("session_id", "unknown"),
                user_type=UserType(user_session_data.get("user_type", "anonymous")),
                fingerprint=user_session_data.get("fingerprint", "unknown"),
                user_id=user_session_data.get("user_id"),
                username=user_session_data.get("username"),
                roles=user_session_data.get("roles", []),
                permissions=user_session_data.get("permissions", []),
            )

            # gateway protections for /socket.io/chat (tracked)
            context = RequestContext(client_ip="socket.io", user_agent="socket.io-client", authorization_header=None)
            try:
                endpoint = "/socket.io/chat"
                await self.gateway_adapter.gateway.rate_limiter.check_and_record(session, context, endpoint)
                await self.gateway_adapter.gateway.backpressure_manager.check_capacity(
                    session.user_type, session, context, endpoint
                )
                logger.info("gateway ok sid=%s session=%s", sid, session.session_id)

            except RateLimitError as e:
                svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
                conv = ConversationCtx(session_id=session.session_id, conversation_id=data.get("conversation_id") or session.session_id, turn_id=f"turn_{uuid.uuid4().hex[:8]}")
                self._comm.emit_error(svc, conv, error=f"Rate limit exceeded: {e.message}", target_sid=sid, session_id=session.session_id)
                return

            except BackpressureError as e:
                svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
                conv = ConversationCtx(session_id=session.session_id, conversation_id=data.get("conversation_id") or session.session_id, turn_id=f"turn_{uuid.uuid4().hex[:8]}")
                self._comm.emit_error(svc, conv, error=f"System under pressure: {e.message}", target_sid=sid, session_id=session.session_id)
                return

            except CircuitBreakerError as e:
                svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
                conv = ConversationCtx(session_id=session.session_id, conversation_id=data.get("conversation_id") or session.session_id, turn_id=f"turn_{uuid.uuid4().hex[:8]}")
                self._comm.emit_error(svc, conv, error=f"Service temporarily unavailable: {e.message}", target_sid=sid, session_id=session.session_id)
                return

            except Exception as e:
                svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
                conv = ConversationCtx(session_id=session.session_id, conversation_id=data.get("conversation_id") or session.session_id, turn_id=f"turn_{uuid.uuid4().hex[:8]}")
                self._comm.emit_error(svc, conv, error="System check failed", target_sid=sid, session_id=session.session_id)
                logger.error("gateway check failed: %s", e)
                return
            tenant_id = data.get("tenant_id") or get_tenant()
            # accounting envelope
            request_id = str(uuid.uuid4())

            agentic_bundle_id = data.get("bundle_id")
            from kdcube_ai_app.infra.plugin.bundle_registry import resolve_bundle
            spec_resolved = resolve_bundle(agentic_bundle_id, override=None)
            agentic_bundle_id = spec_resolved.id if spec_resolved else None

            acct_env = build_envelope_from_session(
                session=session,
                tenant_id=tenant_id,
                project_id=data.get("project"),
                request_id=request_id,
                component="chat.socket",
                app_bundle_id=agentic_bundle_id,
                metadata={"socket_id": sid, "entrypoint": "/socket.io/chat"},
            ).to_dict()

            # assemble task payload (protocol model)
            task_id = str(uuid.uuid4())
            turn_id = data.get("turn_id") or f"turn_{uuid.uuid4().hex[:8]}"
            conversation_id = data.get("conversation_id") or session.session_id


            if not spec_resolved:
                svc = ServiceCtx(request_id=request_id, user=session.user_id, project=data.get("project"))
                conv = ConversationCtx(session_id=session.session_id, conversation_id=conversation_id, turn_id=turn_id)
                self._comm.emit_error(svc, conv, error=f"Unknown bundle_id '{agentic_bundle_id}'", target_sid=sid, session_id=session.session_id)
                return

            payload = ChatTaskPayload(
                meta=ChatTaskMeta(task_id=task_id, created_at=time.time(), instance_id=self.instance_id),
                routing=ChatTaskRouting(
                    session_id=session.session_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    socket_id=sid,
                    bundle_id=spec_resolved.id,
                ),
                actor=ChatTaskActor(
                    tenant_id=data.get("tenant_id") or get_tenant(),
                    project_id=data.get("project"),
                ),
                user=ChatTaskUser(
                    user_type=session.user_type.value,
                    user_id=session.user_id,
                    username=session.username,
                    fingerprint=session.fingerprint,
                    roles=session.roles,
                    permissions=session.permissions,
                ),
                request=ChatTaskRequest(
                    message=message,
                    chat_history=data.get("chat_history") or [],
                    operation=data.get("operation") or data.get("command"),
                    invocation=data.get("invocation"),
                    payload=data.get("payload") or {},       # ← generic Any pass-through
                ),
                config=ChatTaskConfig(values=data.get("config") or {}),
                accounting=ChatTaskAccounting(envelope=acct_env),
            )

            # atomic enqueue with backpressure accounting
            success, reason, stats = await self.chat_queue_manager.enqueue_chat_task_atomic(
                session.user_type,
                payload.model_dump(),   # ← serialize protocol model
                session,
                context,
                "/socket.io/chat",
            )
            if not success:
                svc = ServiceCtx(request_id=request_id, user=session.user_id, project=data.get("project"))
                conv = ConversationCtx(session_id=session.session_id, conversation_id=conversation_id, turn_id=turn_id)
                self._comm.emit_error(svc, conv, error=f"System under pressure - request rejected ({reason})", target_sid=sid, session_id=session.session_id)
                return

            # ack to client via communicator (same envelope as processor emits)
            svc = ServiceCtx(request_id=request_id, user=session.user_id, project=data.get("project"))
            conv = ConversationCtx(session_id=session.session_id, conversation_id=conversation_id, turn_id=turn_id)
            self._comm.emit_start(svc, conv, message=(message[:100] + "..." if len(message) > 100 else message), queue_stats=stats, target_sid=sid, session_id=session.session_id)

        except Exception as e:
            logger.exception("chat_message error: %s", e)
            try:
                svc = ServiceCtx(request_id=str(uuid.uuid4()))
                conv = ConversationCtx(session_id="unknown", conversation_id="unknown", turn_id=f"turn_{uuid.uuid4().hex[:8]}")
                self._comm.emit_error(svc, conv, error=str(e), target_sid=sid)
            except Exception:
                pass

    # ---------- ASGI app ----------

    def get_asgi_app(self):
        return socketio.ASGIApp(self.sio) if self.sio else None


def create_socketio_chat_handler(
    app,
    gateway_adapter,
    chat_queue_manager,
    allowed_origins,
    instance_id,
    redis_url,
    chat_comm: ChatRelayCommunicator,
):
    return SocketIOChatHandler(
        app=app,
        gateway_adapter=gateway_adapter,
        chat_queue_manager=chat_queue_manager,
        allowed_origins=allowed_origins,
        instance_id=instance_id,
        redis_url=redis_url,
        chat_comm=chat_comm,
    )
