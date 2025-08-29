# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from typing import Any, Optional, Callable, Awaitable

StepEmitter = Callable[[str, str, dict], Awaitable[None]]
DeltaEmitter = Callable[[str, int, dict], Awaitable[None]]

class NoopEmitter:
    async def emit(self, event: str, data: dict, *, room: Optional[str] = None, **_):
        return

class SocketIOEmitter:
    def __init__(self, sio):
        self.sio = sio
    async def emit(self, event: str, data: dict, *, room: Optional[str] = None, **_):
        await self.sio.emit(event, data, room=room)

# -------------------------------
# Redis relay emitter (preferred)
# -------------------------------
import os
from kdcube_ai_app.infra.orchestration.app.communicator import ServiceCommunicator

class ChatRelayEmitter:
    """
    Publishes chat events to Redis; a Socket.IO listener on web nodes relays to clients.

    Usage parity with SocketIOEmitter:
        await emitter.emit("chat_step", {...}, room=session_id)
    If you know the exact target SID (socket_id), pass target_sid=... to avoid duplicates.
    """

    def __init__(
        self,
        communicator: Optional[ServiceCommunicator] = None,
        *,
        channel: str = "chat.events",
        orchestrator_identity: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.comm = communicator or ServiceCommunicator(
            redis_url=redis_url,
            orchestrator_identity=orchestrator_identity
            or os.environ.get(
                "ORCHESTRATOR_IDENTITY",
                f"kdcube_orchestrator_{os.environ.get('ORCHESTRATOR_TYPE', 'dramatiq')}",
            ),
        )
        self.channel = channel

    async def emit(
        self,
        event: str,
        data: dict,
        *,
        room: Optional[str] = None,
        target_sid: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        room: logical room (use a session_id here)
        target_sid: specific socket id (best to avoid cross-node duplicates)
        session_id: optional explicit session id (if you don't pass room)
        """
        # avoid publishing both sid AND session to prevent duplicate fan-out
        pub_session = session_id or (None if target_sid else room)
        try:
            self.comm.pub(
                event=event,
                target_sid=target_sid,
                session_id=pub_session,
                data=data,
                channel=self.channel,
            )
        except Exception:
            # keep emitter failures non-fatal in chat flows
            pass
