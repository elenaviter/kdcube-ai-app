# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from typing import Any, Optional, Callable, Awaitable

StepEmitter = Callable[[str, str, dict], Awaitable[None]]
DeltaEmitter = Callable[[str, int, dict], Awaitable[None]]

class NoopEmitter:
    async def emit(self, event: str, data: dict, *, room: Optional[str] = None):
        return

class SocketIOEmitter:
    def __init__(self, sio):
        self.sio = sio
    async def emit(self, event: str, data: dict, *, room: Optional[str] = None):
        await self.sio.emit(event, data, room=room)
