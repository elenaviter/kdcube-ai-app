# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations
from typing import Optional, Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage

from kdcube_ai_app.apps.chat.sdk.inventory import (
    Config,
    ModelServiceBase,   # we rely on the base router
    AgentLogger,
    _mid,
)
from kdcube_ai_app.tools.serialization import json_safe

BUNDLE_ID = "kdcube.demo.1"


class ThematicBotModelService(ModelServiceBase):
    """Thin adapter that relies on ModelServiceBase's router for role-based clients."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = AgentLogger("ThematicBotModelService", config.log_level)
        self.logger.log_step(
            "model_service_initialized",
            {
                "role_models": dict(config.role_models or {}),
            },
        )


# ---- app state helpers ----

APP_STATE_KEYS = [
    "context",
    "user_message",
    "is_our_domain",
    "classification_reasoning",
    "rag_queries",
    "retrieved_docs",
    "reranked_docs",
    "final_answer",
    "thinking",
    "followups",
    "turn_log",
    "error_message",
    "format_fix_attempts",
    "search_hits",
    "execution_id",
    "start_time",
    "step_logs",
    "performance_metrics",
]

def project_app_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in APP_STATE_KEYS:
        if k == "context":
            ctx = dict(state.get("context") or {})
            ctx.setdefault("bundle", BUNDLE_ID)
            out["context"] = json_safe(ctx)
        else:
            out[k] = json_safe(state.get(k))
    return out


def _history_to_seed_messages(history: Optional[List[Dict[str, Any]]]) -> List[AnyMessage]:
    """
    Convert a simple [{role, content}] history into LangChain messages.
    Accepts roles: 'system', 'user', 'assistant'.
    Unknown roles are ignored.
    """
    out: List[AnyMessage] = []
    for h in history or []:
        role = (h.get("role") or "").strip().lower()
        content = (h.get("content") or "").strip()
        if not content:
            continue
        mid = h.get("id") or _mid(role or "msg")
        if role == "system":
            out.append(SystemMessage(content=content, id=mid))
        elif role == "user":
            out.append(HumanMessage(content=content, id=mid))
        elif role == "assistant":
            out.append(AIMessage(content=content, id=mid))
        # else: ignore unknown roles
    return out
