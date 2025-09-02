# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/emitters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Callable, Awaitable, Dict
import os

from kdcube_ai_app.apps.chat.sdk.protocol import (
    ChatEnvelope, ServiceCtx, ConversationCtx, _iso_now
)
from kdcube_ai_app.infra.orchestration.app.communicator import ServiceCommunicator


# map protocol type → client socket event
_EVENT_MAP = {
    "chat.start": "chat_start",
    "chat.step": "chat_step",
    "chat.delta": "chat_delta",
    "chat.complete": "chat_complete",
    "chat.error": "chat_error",
}


class ChatRelayCommunicator:
    """
    Single interface for publishing and subscribing chat events (via Redis).
    The web Socket.IO relay subscribes and forwards to browser clients.
    """

    def __init__(
            self,
            *,
            redis_url: Optional[str] = None,
            channel: str = "chat.events",
            orchestrator_identity: Optional[str] = None,
            comm: Optional[ServiceCommunicator] = None,
    ):
        redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._comm = comm or ServiceCommunicator(
            redis_url=redis_url,
            orchestrator_identity=orchestrator_identity
                                  or os.environ.get(
                "ORCHESTRATOR_IDENTITY",
                f"kdcube_orchestrator_{os.environ.get('ORCHESTRATOR_TYPE', 'dramatiq')}",
            ),
        )
        self._channel = channel

    # ---------- publish ----------

    def _pub(self, env: ChatEnvelope, *, target_sid: Optional[str], session_id: Optional[str]):
        # Always publish TYPE-MAPPED event so the relay only emits
        event = _EVENT_MAP[env.type]
        self._comm.pub(
            event=event,
            data=env.dump_model(),
            target_sid=target_sid,
            session_id=session_id,
            channel=self._channel,
        )

    def emit_envelope(self, env: ChatEnvelope, *, target_sid: Optional[str] = None, session_id: Optional[str] = None):
        """Low-level: publish a prebuilt envelope."""
        self._pub(env, target_sid=target_sid, session_id=session_id)

    def emit_start(self, service: ServiceCtx, conv: ConversationCtx, *, message: str, queue_stats: Dict[str, Any] | None = None, target_sid: Optional[str] = None, session_id: Optional[str] = None):
        self._pub(ChatEnvelope.start(service, conv, message=message, queue_stats=queue_stats), target_sid=target_sid, session_id=session_id)

    def emit_step(self, service: ServiceCtx, conv: ConversationCtx, *, step: str, status: str, title: Optional[str] = None, data: Any = None, agent: Optional[str] = None, target_sid: Optional[str] = None, session_id: Optional[str] = None):
        self._pub(ChatEnvelope.step(service, conv, step=step, status=status, title=title, data=data, agent=agent), target_sid=target_sid, session_id=session_id)

    def emit_delta(self, service: ServiceCtx, conv: ConversationCtx, *, text: str, index: int, marker: str = "answer", target_sid: Optional[str] = None, session_id: Optional[str] = None):
        self._pub(ChatEnvelope.delta(service, conv, text=text, index=index, marker=marker), target_sid=target_sid, session_id=session_id)

    def emit_complete(self, service: ServiceCtx, conv: ConversationCtx, *, data: Any = None, target_sid: Optional[str] = None, session_id: Optional[str] = None):
        self._pub(ChatEnvelope.complete(service, conv, data=data), target_sid=target_sid, session_id=session_id)

    def emit_error(self, service: ServiceCtx, conv: ConversationCtx, *, error: str, title: Optional[str] = "Workflow Error", step: str = "workflow", target_sid: Optional[str] = None, session_id: Optional[str] = None):
        self._pub(ChatEnvelope.error(service, conv, error=error, title=title, step=step), target_sid=target_sid, session_id=session_id)

    # ---------- binding helpers (nice for processors) ----------

    class _Bound:
        def __init__(self, parent: "ChatRelayCommunicator", service: ServiceCtx, conv: ConversationCtx, *, target_sid: Optional[str], session_id: Optional[str]):
            self._p = parent
            self._svc = service
            self._conv = conv
            self._sid = target_sid
            self._room = session_id

        @property
        def service(self) -> ServiceCtx:
            return self._svc

        @property
        def conversation(self) -> ConversationCtx:
            return self._conv

        @property
        def target_sid(self) -> Optional[str]:
            return self._sid

        @property
        def session_id(self) -> Optional[str]:
            return self._room

        def emit_start(self, message: str, queue_stats: Dict[str, Any] | None = None): self._p.emit_start(self._svc, self._conv, message=message, queue_stats=queue_stats, target_sid=self._sid, session_id=self._room)
        def emit_step(self, step: str, status: str, *, title: Optional[str] = None, data: Any = None, agent: Optional[str] = None): self._p.emit_step(self._svc, self._conv, step=step, status=status, title=title, data=data, agent=agent, target_sid=self._sid, session_id=self._room)
        def emit_delta(self, text: str, index: int, *, marker: str = "answer"): self._p.emit_delta(self._svc, self._conv, text=text, index=index, marker=marker, target_sid=self._sid, session_id=self._room)
        def emit_complete(self, data: Any | None = None): self._p.emit_complete(self._svc, self._conv, data=data or {}, target_sid=self._sid, session_id=self._room)
        def emit_error(self, error: str, *, title: Optional[str] = "Workflow Error", step: str = "workflow"): self._p.emit_error(self._svc, self._conv, error=error, title=title, step=step, target_sid=self._sid, session_id=self._room)

        def make_emitters(self) -> tuple[Callable[[str, str, Dict[str, Any] | None], Awaitable[None]], Callable[[str, int, Dict[str, Any] | None], Awaitable[None]]]:
            """
            Returns (step_emitter, delta_emitter) that workflows expect.
            """
            async def step_emitter(step: str, status: str, payload: Dict[str, Any] | None = None):
                p = payload or {}
                self.emit_step(step, status, title=p.get("title"), data=p.get("data") if "data" in p else p, agent=p.get("agent"))

            async def delta_emitter(text: str, idx: int, meta: Dict[str, Any] | None = None):
                marker = (meta or {}).get("marker", "answer")
                self.emit_delta(text, idx, marker=marker)

            return step_emitter, delta_emitter

    def bind(self, *, service: ServiceCtx, conversation: ConversationCtx, target_sid: Optional[str] = None, session_id: Optional[str] = None) -> "_Bound":
        return ChatRelayCommunicator._Bound(self, service, conversation, target_sid=target_sid, session_id=session_id)

    # ---------- subscribe / relay ----------

    async def subscribe(self, callback):
        await self._comm.subscribe(self._channel)
        await self._comm.start_listener(callback)

    async def unsubscribe(self):
        await self._comm.stop_listener()

@dataclass
class ChatCommunicator:
    """
    Unified chat communicator that:
      - knows your service & conversation context
      - builds standard envelopes
      - publishes via a transport emitter (relay/socket/etc)
    """
    emitter: Any                             # ChatRelayEmitter | SocketIOEmitter | NoopEmitter
    service: Dict[str, Any]                  # {request_id, tenant, project, user}
    conversation: Dict[str, Any]             # {session_id, conversation_id, turn_id, socket_id?}
    room: Optional[str] = None               # default fan-out room (session_id)
    target_sid: Optional[str] = None         # optional exact socket target

    def __post_init__(self):
        # default room = session_id
        self.room = self.room or self.conversation.get("session_id")
        self.target_sid = self.target_sid or self.conversation.get("socket_id")

    # ---------- low-level ----------
    async def emit(self, event: str, data: dict):
        await self.emitter.emit(
            event=event,
            data=data,
            room=self.room,
            target_sid=self.target_sid,
            session_id=self.conversation.get("session_id"),
        )

    # ---------- envelopes ----------
    def _base_env(self, typ: str) -> Dict[str, Any]:
        return {
            "type": typ,
            "timestamp": _iso_now(),
            "service": dict(self.service or {}),
            "conversation": {
                "session_id": self.conversation.get("session_id"),
                "conversation_id": self.conversation.get("conversation_id"),
                "turn_id": self.conversation.get("turn_id"),
            },
            "event": {"step": "event", "status": "update"},
        }

    async def emit_enveloped(self, env: dict):
        """Pass-through for already-formed envelopes (route by .type)."""
        typ = (env or {}).get("type")
        route = {
            "chat.start": "chat_start",
            "chat.step": "chat_step",
            "chat.delta": "chat_delta",
            "chat.complete": "chat_complete",
            "chat.error": "chat_error",
        }.get(typ, "chat_step")
        await self.emit(route, env)

    # ---------- high-level helpers ----------
    async def start(self, *, message: str, queue_stats: Optional[dict] = None):
        env = self._base_env("chat.start")
        env["event"] = {"step": "turn", "status": "started", "title": "Turn Started"}
        env["data"] = {"message": message, "queue_stats": queue_stats or {}}
        await self.emit("chat_start", env)

    async def step(self, *, step: str, status: str, title: Optional[str] = None,
                   agent: Optional[str] = None, data: Optional[dict] = None, markdown: Optional[str] = None):
        env = self._base_env("chat.step")
        env["event"].update({"step": step, "status": status, "title": title, "agent": agent})
        env["data"] = data or {}
        if markdown:
            env["event"]["markdown"] = markdown
        await self.emit("chat_step", env)

    async def delta(self, *, text: str, index: int, marker: str = "answer", agent: str = "assistant", completed: bool = False, **kwargs):
        env = self._base_env("chat.delta")
        env["event"].update({"agent": agent, "step": "stream", "status": "running", "title": "Assistant Delta"})
        env["delta"] = {"text": text, "marker": marker, "index": int(index), "completed": completed }
        # back-compat mirrors
        env["text"] = text
        env["idx"] = int(index)
        if kwargs:
            env["extra"] = kwargs
        await self.emit("chat_delta", env)

    async def complete(self, *, data: dict):
        env = self._base_env("chat.complete")
        env["event"].update({"agent": "answer_generator", "step": "stream", "status": "completed", "title": "Turn Completed"})
        env["data"] = data or {}
        await self.emit("chat_complete", env)

    async def error(self, *, message: str, data: Optional[dict] = None):
        env = self._base_env("chat.error")
        env["event"].update({"step": "workflow", "status": "error", "title": "Workflow Error"})
        env["data"] = {"error": message, **(data or {})}
        await self.emit("chat_error", env)