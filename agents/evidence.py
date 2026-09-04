"""Local evidence sinks shared by the direct agent examples."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsoleEmitter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def emit(self, *, event: str, data: dict, **_: Any) -> None:
        row = {"socket_event": event, "data": data, "recorded_at": utc_now()}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        kind = str(data.get("type") or event)
        detail = data.get("delta") if kind == "chat.delta" else data.get("event")
        print(f"[communicator] {kind}: {json.dumps(detail or {}, ensure_ascii=False, default=str)}")
