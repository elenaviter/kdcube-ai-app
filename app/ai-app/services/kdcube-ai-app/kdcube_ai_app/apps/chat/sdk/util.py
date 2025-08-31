# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/util.py
import time, orjson, hashlib, re, json, unicodedata
from typing import Any, List, Dict, Optional, Union
from datetime import datetime, timezone

from pydantic import BaseModel

from kdcube_ai_app.apps.chat.sdk.protocol import ChatHistoryMessage


# ---------- small general helpers ----------

def now_ms() -> int:
    return int(time.time() * 1000)

def json_dumps(data: Any) -> str:
    return orjson.dumps(
        data,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
    ).decode()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# def slug(s: str) -> str:
#     return re.sub(r"[^a-z0-9\-]+", "-", s.lower()).strip("-")

def slug(s: str) -> str:
    # fold accents → ascii, then normalize
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)   # turn any run of non-alnum into a single dash
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def _safe_schema_name(name: str) -> str:
    """
    Same semantics as ops/deployment SQL tool to ensure we land in the same schema.
    """
    sanitized = re.sub(r'[^A-Za-z0-9_]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_').lower()
    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"_{sanitized}" if sanitized else "_schema"
    return sanitized[:63]

def _make_project_schema(tenant: str, project: str) -> str:
    return f"kdcube_{tenant}_{_safe_schema_name(project)}"

# ---------- follow-up parsing (kept) ----------

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_followups_tail(raw: str) -> List[str]:
    """
    Accepts:
      - JSON object with key 'followup' or 'followups'
      - or a raw JSON array of strings
      - allows content to be wrapped in ```json fences
      - ignores any text before the final JSON block
    """
    if not raw:
        return []
    candidate = _strip_code_fences(raw)
    m_obj = re.search(r"\{[\s\S]*\}\s*$", candidate)
    if m_obj:
        try:
            obj = json.loads(m_obj.group(0))
            vals = obj.get("followup") or obj.get("followups") or []
            if isinstance(vals, list):
                return [str(x).strip() for x in vals if str(x).strip()]
        except Exception:
            pass
    m_arr = re.search(r"\[[\s\S]*\]\s*$", candidate)
    if m_arr:
        try:
            arr = json.loads(m_arr.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return []

def _extract_json_block(text: str) -> Optional[str]:
    """Strip ```json fences and return the innermost {...} block."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[ \t]*([jJ][sS][oO][nN])?[ \t]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return None

def _json_loads_loose(text: str):
    """Best-effort JSON loader that tolerates code fences and chatter."""
    try:
        return json.loads(text)
    except Exception:
        pass
    block = _extract_json_block(text)
    if block:
        try:
            return json.loads(block)
        except Exception:
            block_nc = re.sub(r",(\s*[}\]])", r"\1", block)
            try:
                return json.loads(block_nc)
            except Exception:
                return None
    return None

# ---------- simple markdown generation (NO type/stage mapping) ----------

def _truncate(s: str, n: int = 500) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + "…"

def ensure_event_markdown(evt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure evt['markdown'] exists based purely on fields present in the event.
    We do NOT guess types or map names. We just render what the producer sent.

    Expected (but not required) keys:
      - title: str
      - agent: str
      - step: str
      - status: str
      - message: str
      - data: dict | list | str | number
    """
    if evt.get("markdown"):
        return evt

    title  = evt.get("title") or (evt.get("type") or "Event").replace(".", " ").title()
    agent  = evt.get("agent")
    step   = evt.get("step")
    status = evt.get("status")
    note   = evt.get("message")
    data   = evt.get("data")

    lines: List[str] = [f"**Message:** {title}"]
    if agent:  lines.append(f"**Agent:** `{agent}`")
    if step:   lines.append(f"**Step:** `{step}`")
    if status: lines.append(f"**Status:** `{status}`")
    if note:   lines.append(f"**Note:** {_truncate(str(note), 1200)}")

    # Render data in a helpful, generic way
    if isinstance(data, dict):
        # show frequently useful fields if present
        # queries
        if isinstance(data.get("queries"), list):
            qs = data["queries"]
            lines.append(f"**Queries:** {len(qs)}")
            for q in qs[:8]:
                item = q.get("query") if isinstance(q, dict) else str(q)
                lines.append(f"* {_truncate(str(item), 200)}")
            if len(qs) > 8:
                lines.append(f"* … +{len(qs) - 8} more")
        # items/sources
        if isinstance(data.get("items"), list):
            items = data["items"]
            lines.append(f"**Items:** {len(items)}")
            for m in items[:5]:
                if isinstance(m, dict):
                    label = m.get("title") or m.get("summary") or m.get("heading") or m.get("text") or m.get("content") or str(m)
                else:
                    label = str(m)
                lines.append(f"* {_truncate(label, 160)}")
            if len(items) > 5:
                lines.append(f"* … +{len(items) - 5} more")
        # scalars overview
        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        if scalars:
            lines.append("**Details:**")
            for k, v in list(scalars.items())[:12]:
                lines.append(f"- {k}: `{_truncate(str(v), 200)}`")

        # raw block
        lines.append("```json")
        lines.append(_truncate(json.dumps(data, ensure_ascii=False, indent=2), 1200))
        lines.append("```")

    elif isinstance(data, list):
        lines.append(f"**Items:** {len(data)}")
        for x in data[:10]:
            lines.append(f"* {_truncate(str(x), 160)}")
        if len(data) > 10:
            lines.append(f"* … +{len(data) - 10} more")

    elif data is not None:
        lines.append(f"**Data:** {_truncate(str(data), 1200)}")

    evt["markdown"] = "\n".join(lines)
    return evt

# ---------- safe JSON for wire ----------

def _to_json_safe(x):
    import datetime as _dt
    if isinstance(x, dict):
        return {k: _to_json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_json_safe(v) for v in x]
    if isinstance(x, tuple):
        return [_to_json_safe(v) for v in x]
    if isinstance(x, set):
        return [_to_json_safe(v) for v in x]
    if isinstance(x, _dt.datetime):
        return (x.isoformat() if x.tzinfo else x.isoformat() + "Z")
    if isinstance(x, _dt.date):
        return x.isoformat()
    return x

# -----------------------------
# Utilities
# -----------------------------

def normalize_history(items: Optional[Union[List[Dict[str, Any]], List[ChatHistoryMessage]]]) -> List[ChatHistoryMessage]:
    if not items:
        return []
    out: List[ChatHistoryMessage] = []
    for it in items:
        if isinstance(it, ChatHistoryMessage):
            out.append(it)
            continue
        if isinstance(it, dict):
            role = (it.get("role") or "user").lower()
            content = it.get("content") or ""
            ts = it.get("timestamp")
            out.append(ChatHistoryMessage(role=role, content=content, timestamp=ts))
    return out


def history_as_dicts(items: List[ChatHistoryMessage]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for h in (items or []):
        out.append({
            "role": h.role,
            "content": h.content,
            "timestamp": h.timestamp or datetime.now().isoformat()
        })
    return out

def _json_schema_of(model: type[BaseModel]) -> str:
    """Pretty JSON Schema for a Pydantic model (v2 with v1 fallback)."""
    try:
        schema = model.model_json_schema()   # Pydantic v2
    except Exception:
        schema = model.schema()              # Pydantic v1 fallback
    return json.dumps(schema, indent=2, ensure_ascii=False)

from collections.abc import Mapping, MutableSequence, Set as _Set

def _deep_merge(a, b):
    """
    Recursively merge b into a without mutating inputs.
    Rules:
      - dict vs dict  -> recursive merge (right side wins on conflicts)
      - list/tuple    -> concatenate
      - set/frozenset -> union
      - other         -> b overwrites a
    """
    # fast path: exact same object
    if a is b:
        return a

    # dict-like
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = dict(a)
        for k, bv in b.items():
            av = out.get(k, None)
            if k in out:
                out[k] = _deep_merge(av, bv)
            else:
                out[k] = bv
        return out

    # sequences (keep type of left operand where possible)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        cat = list(a) + list(b)
        return type(a)(cat) if isinstance(a, tuple) else cat

    # sets
    if isinstance(a, _Set) and isinstance(b, _Set):
        return type(a)(set(a) | set(b))

    # fallback: overwrite
    return b

def _today_str() -> str:
    # UTC; if you have user’s tz, inject it in prompts separately
    return datetime.now(timezone.utc).date().isoformat()