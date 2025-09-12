# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chatbot/sdk/comm/emitters.py

from __future__ import annotations
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from kdcube_ai_app.apps.chat.emitters import ChatCommunicator

def _compose_md(md: str) -> Dict[str, Any]:
    return {"markdown": md, "compose": {"blocks": [{"type": "md", "text": md}]}}

class DeltaPayload(BaseModel):
    text: str
    index: int
    marker: Literal["thinking", "answer"] = "answer"
    completed: bool = False
    agent: Optional[str] = None

class StepPayload(BaseModel):
    step: str
    status: Literal["started", "completed", "error", "skipped"] = "completed"
    title: Optional[str] = None
    markdown: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    agent: Optional[str] = None

class AIBEmitters:
    def __init__(self, comm: ChatCommunicator):
        self.comm = comm

    async def delta(self, p: DeltaPayload) -> None:
        await self.comm.delta(
            text=p.text,
            index=p.index,
            marker=p.marker,
            completed=p.completed,
            agent=p.agent,
        )

    async def step(self, p: StepPayload) -> None:
        data = dict(p.data or {})
        if p.markdown:
            data = {**_compose_md(p.markdown), **data}
        await self.comm.step(step=p.step, status=p.status, title=p.title, data=data)

    async def event(self,
                    agent: str | None,
                    type: str,                   # e.g. "chat.followups"
                    title: str | None = None,
                    step: str | None = None,
                    data: dict | None = None,
                    markdown: str | None = None,
                    route: str | None = None,    # optional override for socket event name
                    status: str = "update") -> None:
        # passthrough to the communicator, no extra flags/compose
        await self.comm.event(agent=agent, type=type, title=title, data=data,
                              markdown=markdown, route=route, status=status,
                              step=step)

    async def followups(self, items: list[str], *, agent: str = "answer_generator") -> None:
        await self.comm.event(
            agent=agent,
            type="chat.followups",
            title="Follow-ups: User Shortcuts",
            status="completed",
            data={"items": items},
        )
