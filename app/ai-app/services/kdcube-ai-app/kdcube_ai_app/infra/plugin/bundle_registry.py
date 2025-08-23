# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter
# kdcube_ai_app/infra/plugin/bundle_registry.py

from __future__ import annotations
import json, os, threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

_REG_LOCK = threading.RLock()
_REGISTRY: Dict[str, Dict[str, Any]] = {}
_DEFAULT_ID: Optional[str] = None

@dataclass
class BundleSpec:
    id: str
    name: Optional[str] = None
    path: str = ""
    module: Optional[str] = None
    singleton: bool = False
    description: Optional[str] = None

ENV_JSON = "AGENTIC_BUNDLES_JSON"
ENV_DEFAULT_ID = "AGENTIC_DEFAULT_BUNDLE_ID"

def _bool_env(v: Optional[str]) -> bool:
    return v in {"1", "true", "True", "YES", "yes"}

def _normalize(d: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure required keys exist
    d = dict(d)
    d["id"] = d.get("id") or d.get("key") or d.get("name")
    if not d.get("id"):
        raise ValueError("BundleSpec missing 'id'")
    if not d.get("path"):
        raise ValueError(f"BundleSpec '{d['id']}' missing 'path'")
    d.setdefault("singleton", bool(d.get("singleton", False)))
    return d


def load_from_env() -> None:
    global _REGISTRY, _DEFAULT_ID
    with _REG_LOCK:
        raw = os.getenv(ENV_JSON)
        data = json.loads(raw)
        reg: Dict[str, Dict[str, Any]] = {}
        for k, v in (data or {}).items():
            item = _normalize({"id": k, **(v or {})})
            reg[item["id"]] = item
        _REGISTRY = reg

        default_env = os.getenv(ENV_DEFAULT_ID)
        if default_env:
            _DEFAULT_ID = default_env
        elif _DEFAULT_ID not in (_REGISTRY.keys()):
            # If default missing, pick first or None
            _DEFAULT_ID = next(iter(_REGISTRY.keys()), None)

def serialize_to_env(registry: Dict[str, Dict[str, Any]], default_id: Optional[str]) -> None:
    """Sets os.environ variables (does not persist to disk)."""
    with _REG_LOCK:
        os.environ[ENV_JSON] = json.dumps(registry, ensure_ascii=False)
        if default_id:
            os.environ[ENV_DEFAULT_ID] = default_id
        else:
            os.environ.pop(ENV_DEFAULT_ID, None)

def get_all() -> Dict[str, Dict[str, Any]]:
    with _REG_LOCK:
        return {k: dict(v) for k, v in _REGISTRY.items()}

def get_default_id() -> Optional[str]:
    with _REG_LOCK:
        return _DEFAULT_ID

def set_registry(registry: Dict[str, Dict[str, Any]], default_id: Optional[str]) -> None:
    global _REGISTRY, _DEFAULT_ID
    with _REG_LOCK:
        # normalize & replace
        new_reg: Dict[str, Dict[str, Any]] = {}
        for k, v in (registry or {}).items():
            item = _normalize({"id": k, **(v or {})})
            new_reg[item["id"]] = item
        _REGISTRY = new_reg
        _DEFAULT_ID = default_id if default_id in _REGISTRY else (next(iter(_REGISTRY), None))

def upsert_bundles(partial: Dict[str, Dict[str, Any]], default_id: Optional[str]) -> None:
    """Merge update."""
    global _REGISTRY, _DEFAULT_ID
    with _REG_LOCK:
        reg = dict(_REGISTRY)
        for k, v in (partial or {}).items():
            item = _normalize({"id": k, **(v or {})})
            reg[item["id"]] = {**reg.get(item["id"], {}), **item}
        _REGISTRY = reg
        if default_id:
            _DEFAULT_ID = default_id if default_id in _REGISTRY else _DEFAULT_ID

def resolve_bundle(bundle_id: Optional[str], override: Optional[Dict[str, Any]] = None) -> Optional[BundleSpec]:
    """Return the effective BundleSpec from (id OR override)."""
    with _REG_LOCK:
        if override and override.get("path"):
            d = _normalize({
                "id": override.get("id") or "override",
                "path": override["path"],
                "module": override.get("module"),
                "singleton": bool(override.get("singleton", False)),
                "name": override.get("name"),
                "description": override.get("description"),
            })
            return BundleSpec(**d)
        bid = bundle_id or _DEFAULT_ID
        if not bid or bid not in _REGISTRY:
            return None
        return BundleSpec(**_REGISTRY[bid])
