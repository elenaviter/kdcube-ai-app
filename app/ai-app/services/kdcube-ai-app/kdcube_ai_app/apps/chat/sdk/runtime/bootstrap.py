# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# kdcube_ai_app/apps/chat/sdk/runtime/bootstrap.py
from __future__ import annotations
import os

from kdcube_ai_app.apps.chat.emitters import ChatRelayCommunicator, ChatCommunicator, _RelayEmitterAdapter
from kdcube_ai_app.apps.chat.sdk.runtime.portable_spec import PortableSpec
from kdcube_ai_app.infra.service_hub.inventory import (
    ConfigRequest, create_workflow_config, ModelServiceBase
)

import importlib, sys, base64
from typing import Any, Dict, Optional
from contextvars import ContextVar

try:
    import cloudpickle
except Exception:
    cloudpickle = None

def _decode_value(packed: dict) -> Any:
    kind = packed.get("kind")
    data = packed.get("data")
    if kind == "json":
        return data
    if kind == "pickle_b64" and cloudpickle is not None:
        return cloudpickle.loads(base64.b64decode(data.encode("ascii")))
    if kind == "repr":
        # last resort; you may choose to leave it as string
        return data
    return data

def _find_cv_in_child(entry: dict) -> Optional[ContextVar]:
    # 1) Prefer module.attr identity
    mod = entry.get("module")
    attr = entry.get("attr")
    if mod and attr:
        try:
            m = importlib.import_module(mod)
            maybe = getattr(m, attr, None)
            if isinstance(maybe, ContextVar):
                return maybe
        except Exception:
            pass
    # 2) Fallback: scan known ContextVars by unique name
    want_name = entry.get("name")
    if want_name:
        for mname, m in list(sys.modules.items()):
            try:
                for a, v in vars(m).items():
                    if isinstance(v, ContextVar) and getattr(v, "name", None) == want_name:
                        return v
            except Exception:
                continue
    return None

def restore_all_contextvars(snapshot: dict) -> dict:
    restored, missing = 0, 0
    for e in (snapshot or {}).get("entries", []):
        cv = _find_cv_in_child(e)
        if not cv:
            missing += 1
            continue
        try:
            cv.set(_decode_value(e["value"]))
            restored += 1
        except Exception:
            missing += 1
    return {"restored": restored, "missing": missing}


def apply_env(env_map: Dict[str, str]):
    for k, v in (env_map or {}).items():
        os.environ[k] = v

def make_model_service(spec: PortableSpec) -> ModelServiceBase:
    cfg_req = ConfigRequest(**spec.model_config.__dict__)
    cfg = create_workflow_config(cfg_req)
    return ModelServiceBase(cfg)

def make_chat_comm(spec: PortableSpec) -> Optional[ChatCommunicator]:
    if not spec.comm:
        return None
    relay = ChatRelayCommunicator(channel=spec.comm.channel)  # redis url picked from ENV by ctor
    emitter = _RelayEmitterAdapter(relay)
    return ChatCommunicator(
        emitter=emitter,
        service=spec.comm.service,
        conversation=spec.comm.conversation,
        room=spec.comm.room,
        target_sid=spec.comm.target_sid
    )

def make_registry(spec: PortableSpec) -> Dict[str, Any]:
    """
    Build runtime registry objects inside the child process.
    Do NOT serialize/ship connection pools; instantiate clients here.
    """
    reg: Dict[str, Any] = {}
    try:
        # Create a KBClient with no pool; it will lazy-init its own pool using env settings.
        from kdcube_ai_app.apps.chat.sdk.retrieval.kb_client import KBClient
        reg["kb_client"] = KBClient(pool=None)
    except Exception:
        # If KB is optional in some runs, it's fine to skip; tools will error only when called.
        pass
    return reg

def bind_into_module(module, *, svc: ModelServiceBase, registry: Any = None, integrations: Dict[str, Any] | None = None):
    try:
        if hasattr(module, "bind_service"):
            module.bind_service(svc)
    except Exception:
        pass
    try:
        if hasattr(module, "bind_registry") and registry is not None:
            module.bind_registry(registry)
    except Exception:
        pass
    try:
        if hasattr(module, "bind_integrations") and integrations is not None:
            module.bind_integrations(integrations)
    except Exception:
        pass

def bootstrap_from_spec(spec_json: str, *, tool_module) -> Dict[str, Any]:
    spec = PortableSpec.from_json(spec_json)

    # 1) env
    apply_env(spec.env_passthrough)

    # 2) preload modules that define ContextVars, then restore CVs
    if spec.cv_snapshot:
        try:
            for e in (spec.cv_snapshot or {}).get("entries", []):
                mod = (e or {}).get("module")
                if mod:
                    try:
                        importlib.import_module(mod)
                    except Exception:
                        pass
        except Exception:
            pass
        restore_all_contextvars(spec.cv_snapshot)

    # 3) services
    svc = make_model_service(spec)

    # NEW: build registry in the child (kb_client without pool; lazy-init later)
    registry = make_registry(spec)

    bind_into_module(
        tool_module,
        svc=svc,
        registry=registry,
        integrations=(spec.integrations.__dict__ if spec.integrations else None),
    )

    # 4) communicator (optional; only if tools emit deltas)
    comm = make_chat_comm(spec)
    if comm:
        from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import set_comm
        set_comm(comm)

    return {"ok": True}
