# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/protocol.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# -----------------------------
# History / client-side request
# -----------------------------

class ChatHistoryMessage(BaseModel):
    role: str = Field(default="user")
    content: str
    timestamp: Optional[str] = None


class ClientRequest(BaseModel):
    """
    Client-intent block.

    operation: optional generic method to call on the bundle (e.g., "suggestions")
    invocation: "sync" | "async" (advisory, used by REST route to enqueue or run inline)
    message: primary text input (for chat-style workflows)
    payload: arbitrary JSON payload for generic operations
    chat_history: optional conversational history (normalized)
    """
    operation: Optional[str] = None
    invocation: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    chat_history: List[ChatHistoryMessage] = Field(default_factory=list)


# -----------------------------
# Identity / routing / context
# -----------------------------

class TaskMeta(BaseModel):
    task_id: str
    created_at: float
    instance_id: Optional[str] = None
    source: Optional[str] = None  # "socket" | "rest" | etc.


class RoutingInfo(BaseModel):
    session_id: str
    conversation_id: Optional[str] = None
    turn_id: Optional[str] = None
    socket_id: Optional[str] = None  # exact Socket.IO SID (preferred for relay)


class ActorInfo(BaseModel):
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None
    user_type: Optional[str] = None  # "anonymous" | "registered" | "privileged"


class UserInfo(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None
    fingerprint: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)


class ConfigBlock(BaseModel):
    """Opaque config used to build the workflow (ConfigRequest model_dump goes here)."""
    values: Dict[str, Any] = Field(default_factory=dict)


class AccountingBlock(BaseModel):
    """Accounting envelope (as dict)."""
    envelope: Dict[str, Any] = Field(default_factory=dict)


class EnvBlock(BaseModel):
    """Environment / infra hints."""
    kdcube_path: Optional[str] = None


# -----------------------------
# Top-level standardized payload
# -----------------------------

class ChatTaskPayload(BaseModel):
    meta: TaskMeta
    routing: RoutingInfo
    actor: ActorInfo
    user: UserInfo
    request: ClientRequest
    config: ConfigBlock
    accounting: AccountingBlock
    env: EnvBlock


def build_chat_task_payload(
        *,
        task_id: str,
        created_at: float,
        instance_id: Optional[str],
        source: str,
        session_id: str,
        conversation_id: Optional[str],
        turn_id: Optional[str],
        socket_id: Optional[str],
        tenant_id: Optional[str],
        project_id: Optional[str],
        user_type: Optional[str],
        user_id: Optional[str],
        username: Optional[str],
        fingerprint: Optional[str],
        roles: Optional[List[str]],
        permissions: Optional[List[str]],
        message: Optional[str],
        chat_history: Optional[Union[List[Dict[str, Any]], List[ChatHistoryMessage]]],
        config_values: Dict[str, Any],
        accounting_envelope: Dict[str, Any],
        kdcube_path: Optional[str],
        operation: Optional[str] = None,
        invocation: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
) -> ChatTaskPayload:

    from kdcube_ai_app.apps.chat.sdk.util import normalize_history
    req = ClientRequest(
        operation=operation,
        invocation=invocation,
        message=message,
        payload=payload,
        chat_history=normalize_history(chat_history),
    )
    return ChatTaskPayload(
        meta=TaskMeta(task_id=task_id, created_at=created_at, instance_id=instance_id, source=source),
        routing=RoutingInfo(session_id=session_id, conversation_id=conversation_id, turn_id=turn_id, socket_id=socket_id),
        actor=ActorInfo(tenant_id=tenant_id, project_id=project_id, user_type=user_type),
        user=UserInfo(user_id=user_id, username=username, fingerprint=fingerprint, roles=roles or [], permissions=permissions or []),
        request=req,
        config=ConfigBlock(values=config_values or {}),
        accounting=AccountingBlock(envelope=accounting_envelope or {}),
        env=EnvBlock(kdcube_path=kdcube_path),
    )
