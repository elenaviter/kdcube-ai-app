from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json
from datetime import datetime


@dataclass
class TurnLog:
    """
    Minimal turn log: timestamp + ordered timeline blocks for this turn.
    """
    turn_id: str
    ts: str
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    end_ts: str = ""
    sources_used: List[int] = field(default_factory=list)
    blocks_count: int = 0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "ts": self.ts,
            "blocks": list(self.blocks or []),
            "end_ts": self.end_ts,
            "sources_used": list(self.sources_used or []),
            "blocks_count": int(self.blocks_count or 0),
            "tokens": int(self.tokens or 0),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TurnLog":
        return cls(
            turn_id=str(payload.get("turn_id") or ""),
            ts=str(payload.get("ts") or ""),
            blocks=list(payload.get("blocks") or []),
            end_ts=str(payload.get("end_ts") or ""),
            sources_used=list(payload.get("sources_used") or []),
            blocks_count=int(payload.get("blocks_count") or 0),
            tokens=int(payload.get("tokens") or 0),
        )

    def to_markdown(self) -> str:
        md = f"## Turn {self.turn_id} (timestamp: {self.ts})\n\n"
        return md

    @staticmethod
    def _ts_key(ts: str) -> float:
        if not ts:
            return 0.0
        try:
            s = str(ts).strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            try:
                return float(ts)
            except Exception:
                return 0.0

    @classmethod
    def _feedback_summary(cls, feedbacks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not feedbacks:
            return {}
        try:
            latest = max(feedbacks, key=lambda fb: cls._ts_key(str(fb.get("ts") or "")))
            return {
                "count": len(feedbacks),
                "last_ts": latest.get("ts"),
                "last_reaction": latest.get("reaction"),
                "last_origin": latest.get("origin"),
                "last_text": latest.get("text") or "",
            }
        except Exception:
            return {}

    @classmethod
    def build_index_text(cls, payload: Dict[str, Any]) -> str:
        """
        Build compact JSON summary for artifact:turn.log index row.
        Includes: ts, end_ts, sources_used, blocks_count, tokens, feedback summary (if present).
        """
        if not isinstance(payload, dict):
            return ""
        try:
            blocks = payload.get("blocks") or []
            blocks_count = payload.get("blocks_count")
            if not isinstance(blocks_count, int):
                try:
                    blocks_count = len(blocks)
                except Exception:
                    blocks_count = 0
            tokens_val = payload.get("tokens")
            if not isinstance(tokens_val, int):
                tokens_val = 0
            sources_used = payload.get("sources_used") or []
            if not isinstance(sources_used, list):
                sources_used = []
            summary: Dict[str, Any] = {
                "turn_id": payload.get("turn_id"),
                "ts": payload.get("ts"),
                "end_ts": payload.get("end_ts") or "",
                "sources_used": sources_used,
                "blocks_count": blocks_count,
                "tokens": tokens_val,
            }
            tl = payload.get("turn_log") or {}
            if isinstance(tl, dict):
                feedbacks = tl.get("feedbacks") or []
                if isinstance(feedbacks, list) and feedbacks:
                    fb_summary = cls._feedback_summary(feedbacks)
                    if fb_summary:
                        summary["feedback"] = fb_summary
            return json.dumps(summary, ensure_ascii=False)
        except Exception:
            return ""
