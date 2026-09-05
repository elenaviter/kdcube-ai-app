"""Local evidence sinks for directly hosted agents."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsoleEmitter:
    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.path = path
        self.echo = bool(echo)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def emit(self, *, event: str, data: dict, **_: Any) -> None:
        row = {"socket_event": event, "data": data, "recorded_at": utc_now()}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        kind = str(data.get("type") or event)
        detail = data.get("delta") if kind == "chat.delta" else data.get("event")
        rendered = json.dumps(detail or {}, ensure_ascii=False, default=str)
        if self.echo:
            print(f"[communicator] {kind}: {rendered}")


def _local_storage_root(storage_uri: str) -> Path | None:
    parsed = urlparse(str(storage_uri or ""))
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def _accounting_files(
    *,
    storage_uri: str,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str,
    turn_id: str,
) -> list[str]:
    root = _local_storage_root(storage_uri)
    base = root / "accounting" / tenant / project if root is not None else None
    if base is None or not base.is_dir():
        return []
    pattern = f"cb|{user_id}|{conversation_id}|{turn_id}|*.json"
    return [path.resolve().as_uri() for path in sorted(base.rglob(pattern))]


def write_evidence_index(
    path: Path,
    *,
    config: Any,
    conversation_id: str,
    turns: list[Any],
    conversation_records: list[dict[str, Any]],
    adapter_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a convenience index over the run's authoritative KDCube records."""
    records_by_turn = {
        str(row.get("turn_id") or ""): row
        for row in conversation_records
        if isinstance(row, dict)
    }
    turn_rows: list[dict[str, Any]] = []
    for turn in turns:
        turn_id = str(getattr(turn, "turn_id", "") or "")
        record = records_by_turn.get(turn_id) or {}
        turn_rows.append(
            {
                "turn_id": turn_id,
                "turn_log": {
                    key: record.get(key)
                    for key in ("hosted_uri", "rn", "key")
                    if record.get(key)
                },
                "accounting": {
                    "events": list(getattr(turn, "accounting_events", []) or []),
                    "durable_files": _accounting_files(
                        storage_uri=config.storage_uri,
                        tenant=config.tenant,
                        project=config.project,
                        user_id=config.user_id,
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                    ),
                    "redis_role": "live per-turn mirror with TTL",
                },
                "user_attachments": list(
                    getattr(turn, "user_attachments", []) or []
                ),
                "assistant_files": list(
                    getattr(turn, "assistant_files", []) or []
                ),
                "execution_snapshots": list(
                    getattr(turn, "execution_snapshots", []) or []
                ),
            }
        )
    root = _local_storage_root(config.storage_uri)
    payload = {
        "schema": "kdcube.direct-agent-evidence.v1",
        "created_at": utc_now(),
        "adapter": config.agent_id,
        "identity": {
            "tenant": config.tenant,
            "project": config.project,
            "user_id": config.user_id,
            "bundle_id": config.bundle_id,
            "conversation_id": conversation_id,
        },
        "storage": {
            "uri": config.storage_uri,
            "local_root": str(root) if root is not None else None,
            "authority": (
                "The paths and URIs listed below are the durable records. This "
                "evidence.json file is only a navigation index."
            ),
            "areas": {
                "accounting": "accounting/<tenant>/<project>/...",
                "turn_logs": "cb/tenants/<tenant>/projects/<project>/conversation/...",
                "attachments_and_outputs": "cb/tenants/<tenant>/projects/<project>/attachments/...",
                "execution_archives": "cb/tenants/<tenant>/projects/<project>/executions/...",
            },
        },
        "turns": turn_rows,
        "adapter_evidence": dict(adapter_evidence or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def print_evidence_summary(path: Path, payload: dict[str, Any]) -> None:
    storage = payload.get("storage") or {}
    print(f"[evidence] index: {path}")
    print(f"[evidence] KDCube storage: {storage.get('local_root') or storage.get('uri')}")
    for turn in payload.get("turns") or []:
        print(
            "[evidence] "
            f"{turn.get('turn_id')}: "
            f"accounting={len((turn.get('accounting') or {}).get('events') or [])}, "
            f"attachments={len(turn.get('user_attachments') or [])}, "
            f"files={len(turn.get('assistant_files') or [])}, "
            f"executions={len(turn.get('execution_snapshots') or [])}"
        )


__all__ = [
    "ConsoleEmitter",
    "print_evidence_summary",
    "utc_now",
    "write_evidence_index",
]
