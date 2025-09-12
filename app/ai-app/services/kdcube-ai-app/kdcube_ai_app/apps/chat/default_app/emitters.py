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

class SimpleEmitters:
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

    async def event(
            self,
            *,
            type: str,
            title: str | None = None,
            step: str = "event",
            status: str = "completed",
            agent: str | None = None,
            data: dict | None = None,
            markdown: str | None = None,
            compose: bool = False,
    ) -> None:
        await self.comm.event(
            type=type,
            title=title,
            step=step,
            status=status,
            agent=agent,
            data=data or {},
            markdown=markdown,
            compose=compose,
        )

    async def followups(self, items: List[str], *, agent: str = "answer_generator") -> None:
        # chips
        await self.event(
            type="chat.followups",
            title="Follow-ups: User Shortcuts",
            step="followups",
            status="completed",
            agent=agent,
            data={"items": items},
        )
        # optional visible card
        md = "### Suggested next actions\n\n" + "\n".join(f"- {s}" for s in items) if items else "_No follow-ups._"
        await self.step(StepPayload(step="followups", status="completed" if items else "skipped",
                                    title="🧠 Follow-ups", markdown=md, agent=agent))
