# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/api/sse/chat.py

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from kdcube_ai_app.apps.chat.api.resolvers import get_tenant, get_user_session_dependency
from kdcube_ai_app.auth.sessions import UserSession, UserType, RequestContext
from kdcube_ai_app.infra.gateway.rate_limiter import RateLimitError
from kdcube_ai_app.infra.gateway.backpressure import BackpressureError
from kdcube_ai_app.infra.gateway.circuit_breaker import CircuitBreakerError

from kdcube_ai_app.apps.chat.emitters import ChatRelayCommunicator
from kdcube_ai_app.apps.chat.sdk.protocol import (
    ChatTaskPayload, ChatTaskMeta, ChatTaskRouting, ChatTaskActor, ChatTaskUser,
    ChatTaskRequest, ChatTaskConfig, ChatTaskAccounting,
    ServiceCtx, ConversationCtx,
)
from kdcube_ai_app.infra.accounting.envelope import build_envelope_from_session

from kdcube_ai_app.tools.file_text_extractor import DocumentTextExtractor
from kdcube_ai_app.infra.gateway.safe_preflight import PreflightConfig, preflight_async

logger = logging.getLogger(__name__)

KEEPALIVE_SECONDS = 3
def _iso() -> str: return datetime.utcnow().isoformat() + "Z"

def _sse_frame(event: str, data: Dict[str, Any], *, event_id: Optional[str] = None) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    lines: List[str] = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    for line in payload.splitlines():
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines) + "\n"


# -----------------------------
# In-process SSE Hub (fan-out)
# -----------------------------

@dataclass(frozen=True)
class Client:
    session_id: str
    stream_id: Optional[str]
    queue: asyncio.Queue[str]      # queue of SSE frames (strings)

class SSEHub:
    """
    One Redis relay subscription per process -> fan out to all connected SSE clients.
    """
    def __init__(self, chat_comm: ChatRelayCommunicator):
        self.chat_comm = chat_comm
        self._by_session: Dict[str, List[Client]] = {}
        self._lock = asyncio.Lock()
        self._relay_started = False

    async def start(self):
        # Just mark that the hub is ready. We'll start the Redis listener
        # lazily on the first real subscription.
        if self._relay_started:
            logger.info("[SSEHub] start() called but already started")
            return

        logger.info(
            "[SSEHub] start() – hub initialized; listener will start on first session subscription"
        )
        self._relay_started = True

    async def stop(self):
        if not self._relay_started:
            return


        # Unblock all generators by pushing a sentinel into their queues
        async with self._lock:
            for clients in self._by_session.values():
                for c in clients:
                    try:
                        c.queue.put_nowait(": shutdown\n\n")
                    except Exception:
                        pass

        await self.chat_comm.unsubscribe()
        self._relay_started = False
        logger.info("[SSEHub] unsubscribed from relay channel(s)")

    async def register(self, client: Client):
        async with self._lock:
            lst = self._by_session.setdefault(client.session_id, [])
            was_empty = not lst
            lst.append(client)

        if was_empty:
            # First client for this session on this worker → subscribe to its channel
            session_ch = f"chat.events.{client.session_id}"
            await self.chat_comm._comm.subscribe_add(session_ch)
            logger.info(
                "[SSEHub] subscribe for session=%s channel=%s",
                client.session_id,
                session_ch,
            )

            # Ensure Redis listener is running.
            # Safe to call many times; start_listener() returns immediately if already running.
            await self.chat_comm._comm.start_listener(self._on_relay)

        logger.info(
            "[SSEHub] register session=%s total=%d",
            client.session_id,
            len(self._by_session[client.session_id]),
        )

    async def unregister(self, client: Client):

        remove_channel = None
        async with self._lock:
            lst = self._by_session.get(client.session_id, [])
            new = [c for c in lst if c is not client]
            if new:
                self._by_session[client.session_id] = new
                logger.info("[SSEHub] unregister session=%s total=%d", client.session_id, len(new))
            else:
                self._by_session.pop(client.session_id, None)
                remove_channel = f"chat.events.{client.session_id}"
                logger.info("[SSEHub] unregister session=%s total=0 -> removed", client.session_id)

        if remove_channel:
            await self.chat_comm._comm.unsubscribe_some(remove_channel)
            logger.info("[SSEHub] unsubscribed from channel=%s", remove_channel)

    # Relay callback invoked by ChatRelayCommunicator
    async def _on_relay(self, message: dict):
        """
        message = { event, data, target_sid?, session_id? }
        """
        try:
            event = message.get("event")
            data = message.get("data") or {}
            target_sid = message.get("target_sid")
            room = message.get("session_id")

            if not event or not room:
                return  # we only fan-out messages scoped to a session room. ignore malformed / global messages

            # First check if we even have listeners for this session
            async with self._lock:
                recipients = list(self._by_session.get(room, []))

            if not recipients:
                # Nothing to do on this worker
                return

            # Only now build the SSE frame
            frame = _sse_frame(event, data, event_id=str(uuid.uuid4()))

            if target_sid:
                # DM: only client with matching stream_id
                for c in recipients:
                    if c.stream_id and c.stream_id == target_sid:
                        self._enqueue(c, frame)
            else:
                # Broadcast to all clients in the same session
                for c in recipients:
                    self._enqueue(c, frame)

        except Exception as e:
            logger.error("[SSEHub] relay fan-out failed: %s", e)

    def _enqueue(self, client: Client, frame: str):
        # Small bounded queues per client; drop oldest on overflow
        q = client.queue
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(frame)
            except Exception:
                pass


def create_sse_router(
    *,
    app,
    gateway_adapter,
    chat_queue_manager,
    instance_id: str,
    redis_url: str,
    chat_comm: ChatRelayCommunicator,
) -> APIRouter:
    """
    Mount with:
        app.state.sse_hub = SSEHub(chat_comm)
        app.add_event_handler("startup", app.state.sse_hub.start)
        app.add_event_handler("shutdown", app.state.sse_hub.stop)
        app.include_router(create_sse_router(...), prefix="/sse", tags=["SSE"])
    """
    router = APIRouter()
    app.state.sse_enabled = True

    # Ensure hub exists on app
    if not hasattr(app.state, "sse_hub"):
        app.state.sse_hub = SSEHub(chat_comm)

    # ---------- STREAM ----------
    @router.get("/stream")
    async def sse_stream(
        request: Request,
        session: UserSession = Depends(get_user_session_dependency()),
        # Query auth (parity with Socket.IO connect)
        user_session_id: Optional[str] = None,
        bearer_token: Optional[str] = None,
        id_token: Optional[str] = None,
        project: Optional[str] = None,
        tenant: Optional[str] = None,
        # direct-message id for this connection
        stream_id: Optional[str] = None,
    ):
        # --- Resolve session exactly like WS connect ---
        if user_session_id:
            try:
                sess = await gateway_adapter.gateway.session_manager.get_session_by_id(user_session_id)
                if not sess:
                    raise HTTPException(status_code=401, detail="Unknown session")
                session = sess  # override the cookie-derived session
            except HTTPException:
                raise
            except Exception as e:
                logger.error("SSE load session failed for id=%s: %s", user_session_id, e)
                raise HTTPException(status_code=401, detail="Invalid session")
        # Optional bearer validation (same semantics as WS)
        try:
            if bearer_token and session.user_type.value != "anonymous":
                user = await gateway_adapter.gateway.auth_manager.authenticate_with_both(bearer_token, id_token)
                claimed_user_id = getattr(user, "sub", None) or getattr(user, "username", None)
                if session.user_id and claimed_user_id and session.user_id != claimed_user_id:
                    # Emit error into the stream (DM) and reject
                    svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
                    conv = ConversationCtx(session_id=session.session_id,
                                           conversation_id=session.session_id,
                                           turn_id=f"turn_{uuid.uuid4().hex[:8]}")
                    await chat_comm.emit_error(svc, conv,
                        error="Token user mismatch",
                        target_sid=stream_id,
                        session_id=session.session_id)
                    raise HTTPException(status_code=401, detail="Token user mismatch")
        except HTTPException:
            raise
        except Exception as e:
            logger.error("SSE bearer validation failed: %s", e)
            # Emit error before rejecting
            svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv,
                error="Invalid token",
                target_sid=stream_id,
                session_id=session.session_id)
            raise HTTPException(status_code=401, detail="Invalid token")

        # Gateway protections for opening a stream
        try:
            context = RequestContext(
                client_ip="sse",
                user_agent=request.headers.get("user-agent",""),
                authorization_header=(f"Bearer {bearer_token}" if bearer_token else None)
            )
            await gateway_adapter.gateway.rate_limiter.check_and_record(session, context, "/sse/stream")
            await gateway_adapter.gateway.backpressure_manager.check_capacity(session.user_type, session, context, "/sse/stream")
        except RateLimitError as e:
            svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv,
                error=f"Rate limit exceeded: {e.message}",
                target_sid=stream_id, session_id=session.session_id)
            raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded", "retry_after": e.retry_after})
        except BackpressureError as e:
            svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv,
                error=f"System under pressure: {e.message}",
                target_sid=stream_id, session_id=session.session_id)
            raise HTTPException(status_code=503, detail={"error": "System under pressure", "retry_after": e.retry_after})
        except CircuitBreakerError as e:
            svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv,
                error=f"Service temporarily unavailable: {e.message}",
                target_sid=stream_id, session_id=session.session_id)
            raise HTTPException(status_code=503, detail={"error": "Service temporarily unavailable", "retry_after": e.retry_after})

        # Prepare per-connection queue (bounded)
        max_q = int(os.getenv("SSE_CLIENT_QUEUE", "100"))
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=max_q)
        client = Client(session_id=session.session_id, stream_id=stream_id, queue=q)

        # Register client
        await app.state.sse_hub.register(client)

        async def gen():

            # Initial ready
            hello = {
                "timestamp": _iso(),
                "session_id": session.session_id,
                "user_type": session.user_type.value,
                **({"stream_id": stream_id} if stream_id else {}),
                **({"tenant": tenant} if tenant else {}),
                **({"project": project} if project else {}),
            }
            yield _sse_frame("ready", hello, event_id=str(uuid.uuid4()))

            try:
                while True:
                    # 1) if server is shutting down, exit
                    if getattr(app.state, "shutting_down", False):
                        logger.info("[sse_stream] Server shutting down; closing SSE for session=%s", session.session_id)
                        break

                    # 2) if client disconnected, exit
                    try:
                        if await request.is_disconnected():
                            logger.info("[sse_stream] Client disconnected; closing SSE for session=%s", session.session_id)
                            break
                    except RuntimeError:
                        # request might be finalized already in some edge cases
                        break

                    try:
                        # Wait up to KEEPALIVE_SECONDS for the next frame
                        frame = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_SECONDS)
                        yield frame
                    except asyncio.TimeoutError:
                        # On timeout, if shutting down, don't send keepalive, just break
                        if getattr(app.state, "shutting_down", False):
                            logger.info("[sse_stream] Timeout during shutdown; closing SSE for session=%s", session.session_id)
                            break
                        # No frames in this window → send keepalive
                        yield ": keepalive\n\n"

            except asyncio.CancelledError:
                # Uvicorn/gunicorn is cancelling us as part of shutdown
                logger.info("[sse_stream] Cancelled; closing SSE for session=%s", session.session_id)
                # Re-raise so Uvicorn/Starlette know this request is done
                raise
            finally:
                await app.state.sse_hub.unregister(client)
                logger.info("[sse_stream] Cleaned up: session=%s", session.session_id)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",      # nginx
                # "Connection": "keep-alive",
            },
        )

    # ---------- helpers (attachments) ----------
    async def _extract_attachments_from_multipart(
        files: List[UploadFile],
        attachment_meta_json: Optional[str],
        max_mb: int
    ) -> List[Dict[str, Any]]:
        max_bytes = max_mb * 1024 * 1024
        attachment_meta = []
        if attachment_meta_json:
            try:
                attachment_meta = json.loads(attachment_meta_json) or []
            except Exception:
                attachment_meta = []

        out: List[Dict[str, Any]] = []
        enable_av = os.getenv("APP_AV_SCAN", "1") == "1"
        av_timeout = float(os.getenv("APP_AV_TIMEOUT_S", "3.0"))
        cfg = PreflightConfig(av_scan=enable_av, av_timeout_s=av_timeout)
        extractor = DocumentTextExtractor()

        by_name = {m.get("filename") or m.get("name"): m for m in (attachment_meta or [])}

        for f in (files or []):
            try:
                raw = await f.read()
            except Exception:
                continue
            if not raw:
                continue
            if len(raw) > max_bytes:
                logger.warning("attachment '%s' rejected: %d > max %d", f.filename, len(raw), max_bytes)
                continue

            pf = await preflight_async(raw, f.filename, f.content_type or "application/octet-stream", cfg)
            if not pf.allowed:
                logger.warning("attachment '%s' rejected by preflight: %s", f.filename, pf.reasons)
                continue

            try:
                text, info = extractor.extract(raw, f.filename or "file", f.content_type or "application/octet-stream")
            except Exception as ex:
                logger.error("extract failed for '%s': %s", f.filename, ex)
                continue

            meta_in = by_name.get(f.filename) or {}
            out.append({
                "name": f.filename,
                "mime": info.mime,
                "ext": info.ext,
                "size": len(raw),
                "meta": {**(info.meta or {}), **({k:v for k,v in meta_in.items() if k not in ("filename","name")})},
                "warnings": info.warnings,
                "text": text,
            })
        return out

    def _merge_attachments_into_message(message: str, attachments_text: List[Dict[str, Any]]) -> str:
        if not attachments_text:
            return message
        lines = [message, "ATTACHMENTS:"]
        for idx, a in enumerate(attachments_text, start=1):
            lines.append(f"{idx}. Name: {a['name']}; Mime: {a['mime']}")
            lines.append(a["text"])
            lines.append("...")
        return "\n".join(lines)

    # ---------- CHAT (enqueue) ----------
    @router.post("/chat")
    async def sse_chat(
        request: Request,
        session: UserSession = Depends(get_user_session_dependency()),
        stream_id: Optional[str] = None,

        # multipart support (attachments)
        message: Optional[str] = Form(None),
        attachment_meta: Optional[str] = Form(None),
        files: List[UploadFile] = File(default=[]),
    ):
        # Gateway checks for /sse/chat
        ctx = RequestContext(client_ip="sse", user_agent=request.headers.get("user-agent",""), authorization_header=None)
        try:
            await gateway_adapter.gateway.rate_limiter.check_and_record(session, ctx, "/sse/chat")
            await gateway_adapter.gateway.backpressure_manager.check_capacity(session.user_type, session, ctx, "/sse/chat")
        except RateLimitError as e:
            svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv, error=f"Rate limit exceeded: {e.message}",
                                 target_sid=stream_id, session_id=session.session_id)
            raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded", "retry_after": e.retry_after})
        except BackpressureError as e:
            svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv, error=f"System under pressure: {e.message}",
                                 target_sid=stream_id, session_id=session.session_id)
            raise HTTPException(status_code=503, detail={"error": "System under pressure", "retry_after": e.retry_after})
        except CircuitBreakerError as e:
            svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv, error=f"Service temporarily unavailable: {e.message}",
                                 target_sid=stream_id, session_id=session.session_id)
            raise HTTPException(status_code=503, detail={"error": "Service temporarily unavailable", "retry_after": e.retry_after})

        # Parse body (JSON or multipart)
        if message is not None:
            try:
                body = json.loads(message)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid 'message' JSON in multipart form")
            body_attachment_meta = attachment_meta
        else:
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body")
            body_attachment_meta = (body or {}).get("attachment_meta")

        message_data = (body or {}).get("message") or {}

        # Accept either:
        #  - { message: "hello" }
        #  - { text: "hello" }
        #  - { message: { msg: "hello" } }   ← defensive: unwrap common fields
        raw_msg = message_data.get("text")
        if raw_msg is None:
            raw_msg = message_data.get("message")

        # If message is an object (e.g., a structured log), try to unwrap a text-y field
        if isinstance(raw_msg, dict):
            text = (
                    raw_msg.get("text")
                    or raw_msg.get("message")
                    or raw_msg.get("msg")
                    or ""
            )
        else:
            text = raw_msg or ""

        text = str(text).strip()
        if files and not body_attachment_meta:
            body_attachment_meta = "[]"

        # Attachments (multipart only)
        attachments_text: List[Dict[str, Any]] = []
        if files:
            try:
                max_mb = int(os.environ.get("CHAT_MAX_UPLOAD_MB", "20"))
            except Exception:
                max_mb = 20
            attachments_text = await _extract_attachments_from_multipart(files, body_attachment_meta, max_mb)

        if attachments_text:
            text = _merge_attachments_into_message(text, attachments_text)

        if not text:
            svc = ServiceCtx(request_id=str(uuid.uuid4()))
            conv = ConversationCtx(session_id=session.session_id,
                                   conversation_id=message_data.get("conversation_id") or session.session_id,
                                   turn_id=f"turn_{uuid.uuid4().hex[:8]}")
            await chat_comm.emit_error(svc, conv, error='Missing "message"',
                                 target_sid=stream_id, session_id=session.session_id)
            raise HTTPException(status_code=400, detail='Missing "message"')

        tenant_id = message_data.get("tenant") or message_data.get("tenant_id") or get_tenant()
        project_id = message_data.get("project")
        request_id = str(uuid.uuid4())
        provided_bundle_id = message_data.get("bundle_id")

        from kdcube_ai_app.infra.plugin.bundle_registry import resolve_bundle
        spec_resolved = resolve_bundle(provided_bundle_id, override=None)
        bundle_id = spec_resolved.id if spec_resolved else None

        acct_env = build_envelope_from_session(
            session=session,
            tenant_id=tenant_id,
            project_id=project_id,
            request_id=request_id,
            component="chat.sse",
            app_bundle_id=bundle_id,
            metadata={"stream_id": stream_id, "entrypoint": "/sse/chat"},
        ).to_dict()

        task_id = str(uuid.uuid4())
        turn_id = message_data.get("turn_id") or f"turn_{uuid.uuid4().hex[:8]}"
        conversation_id = message_data.get("conversation_id") or session.session_id
        ext_config = (message_data.get("config") or {}) | {}
        if "tenant" not in ext_config: ext_config["tenant"] = tenant_id
        if "project" not in ext_config and project_id: ext_config["project"] = project_id

        svc = ServiceCtx(request_id=request_id, user=session.user_id, project=project_id, tenant=tenant_id)
        conv = ConversationCtx(session_id=session.session_id, conversation_id=conversation_id, turn_id=turn_id)

        if not spec_resolved:
            await chat_comm.emit_error(svc, conv, error=f"Unknown bundle_id '{provided_bundle_id}'", target_sid=stream_id, session_id=session.session_id)
            return

        payload = ChatTaskPayload(
            meta=ChatTaskMeta(task_id=task_id, created_at=time.time(), instance_id=instance_id),
            routing=ChatTaskRouting(
                session_id=session.session_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                socket_id=stream_id,  # DM target = this SSE connection
                bundle_id=bundle_id,
            ),
            actor=ChatTaskActor(tenant_id=tenant_id, project_id=project_id),
            user=ChatTaskUser(
                user_type=session.user_type.value,
                user_id=session.user_id,
                username=session.username,
                fingerprint=session.fingerprint,
                roles=session.roles,
                permissions=session.permissions,
            ),
            request=ChatTaskRequest(
                message=text,
                chat_history=message_data.get("chat_history") or [],
                operation=message_data.get("operation") or message_data.get("command"),
                invocation=message_data.get("invocation"),
                payload=message_data.get("payload") or {},
            ),
            config=ChatTaskConfig(values=ext_config),
            accounting=ChatTaskAccounting(envelope=acct_env),
        )

        # Best-effort “in_progress” mark
        try:
            set_res = await app.state.conversation_browser.set_conversation_state(
                tenant=payload.actor.tenant_id,
                project=payload.actor.project_id,
                user_id=payload.user.user_id,
                conversation_id=payload.routing.conversation_id,
                new_state="in_progress",
                by_instance=instance_id,
                request_id=request_id,
                last_turn_id=payload.routing.turn_id,
                require_not_in_progress=True,
                user_type=payload.user.user_type,
                bundle_id=payload.routing.bundle_id,
            )
        except Exception:
            set_res = {"ok": True, "updated_at": _iso()}

        # Enqueue
        success, reason, stats = await chat_queue_manager.enqueue_chat_task_atomic(
            session.user_type,
            payload.model_dump(),
            session,
            ctx,
            "/sse/chat",
        )
        if not success:
            # rollback state
            try:
                await app.state.conversation_browser.set_conversation_state(
                    tenant=payload.actor.tenant_id,
                    project=payload.actor.project_id,
                    user_id=payload.user.user_id,
                    conversation_id=payload.routing.conversation_id,
                    new_state="idle",
                    by_instance=instance_id,
                    request_id=request_id,
                    last_turn_id=payload.routing.turn_id,
                    require_not_in_progress=False,
                    user_type=payload.user.user_type,
                    bundle_id=payload.routing.bundle_id,
                )
            except Exception:
                pass

            retry_after = 30 if session.user_type == UserType.ANONYMOUS else (45 if session.user_type == UserType.REGISTERED else 60)
            svc = ServiceCtx(request_id=request_id, user=session.user_id, project=project_id, tenant=tenant_id)
            conv = ConversationCtx(session_id=session.session_id, conversation_id=conversation_id, turn_id=turn_id)
            await chat_comm.emit_error(
                svc, conv,
                error=f"System under pressure - request rejected ({reason})",
                target_sid=stream_id,
                session_id=session.session_id,
            )
            raise HTTPException(status_code=503,
                                detail={"error": "System under pressure", "reason": reason, "retry_after": retry_after},
                                headers={"Retry-After": str(retry_after)})

        await chat_comm.emit_start(
            svc, conv,
            message=(text[:100] + "..." if len(text) > 100 else text),
            queue_stats=stats,
            target_sid=stream_id,                 # DM to this tab
            session_id=session.session_id,        # and visible to other tabs of same session if needed
        )

        return {
            "status": "processing_started",
            "task_id": task_id,
            "session_id": session.session_id,
            "user_type": session.user_type.value,
            "message": "Queued; streaming via SSE",
        }

    # ---------- conv_status.get (parity) ----------
    @router.post("/conv_status.get")
    async def sse_conv_status_get(
        data: Dict[str, Any],
        session: UserSession = Depends(get_user_session_dependency()),
        stream_id: Optional[str] = None,
    ):
        conv_id = (data or {}).get("conversation_id") or session.session_id
        row = None
        try:
            row = await app.state.conversation_browser.idx.get_conversation_state_row(
                user_id=session.user_id, conversation_id=conv_id
            )
        except Exception:
            pass

        state = "idle" if not row else ("in_progress" if "conv.state:in_progress" in row.get("tags", []) else "error" if "conv.state:error" in row.get("tags", []) else "idle")
        updated_at = (row["ts"].isoformat() + "Z") if row else _iso()
        current_turn_id = (row.get("payload", {}) or {}).get("last_turn_id") if row else None

        # emit status via relay so all session tabs can see it
        from kdcube_ai_app.apps.chat.sdk.protocol import ChatTaskRouting
        agentic_bundle_id = data.get("bundle_id")
        from kdcube_ai_app.infra.plugin.bundle_registry import resolve_bundle
        spec_resolved = resolve_bundle(agentic_bundle_id, override=None)

        routing = ChatTaskRouting(
            session_id=session.session_id,
            conversation_id=conv_id,
            turn_id=current_turn_id,
            socket_id=stream_id,
            bundle_id=spec_resolved.id,
        )
        svc = ServiceCtx(request_id=str(uuid.uuid4()), user=session.user_id)
        conv = ConversationCtx(session_id=session.session_id, conversation_id=conv_id, turn_id=current_turn_id or f"turn_{uuid.uuid4().hex[:8]}")
        await app.state.chat_comm.emit_conv_status(
            svc, conv, routing,
            state=state, updated_at=updated_at, current_turn_id=current_turn_id,
            target_sid=stream_id  # DM this requester; omit to broadcast to all session tabs
        )
        return {"conversation_id": conv_id, "state": state, "updated_at": updated_at, "current_turn_id": current_turn_id}

    return router
