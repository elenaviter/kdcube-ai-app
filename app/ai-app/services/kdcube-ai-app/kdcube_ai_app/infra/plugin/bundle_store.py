# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter
# kdcube_ai_app/infra/plugin/bundle_store.py

from __future__ import annotations
import json, os, time
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
import kdcube_ai_app.infra.namespaces as namespaces
from kdcube_ai_app.apps.chat.sdk.config import get_settings

REDIS_KEY_FMT = namespaces.CONFIG.BUNDLES.BUNDLE_MAPPING_KEY_FMT
REDIS_CHANNEL_FMT = namespaces.CONFIG.BUNDLES.UPDATE_CHANNEL
ADMIN_BUNDLE_ID = "kdcube.admin"
_EXAMPLES_REL_PATH = Path("apps/chat/sdk/examples/bundles")


def _admin_bundle_entry() -> "BundleEntry":
    root = Path(__file__).resolve().parent
    return BundleEntry(
        id=ADMIN_BUNDLE_ID,
        name="KDCube Admin",
        path=str(root),
        module="admin_bundle.entrypoint",
        singleton=True,
        description="Built-in admin-only bundle",
    )

def _examples_root() -> Path:
    return (Path(__file__).resolve().parents[2] / _EXAMPLES_REL_PATH).resolve()

def _examples_enabled() -> bool:
    try:
        settings = get_settings()
        return bool(settings.BUNDLES_INCLUDE_EXAMPLES)
    except Exception:
        raw = os.getenv("BUNDLES_INCLUDE_EXAMPLES", "1").lower()
        return raw in {"1", "true", "yes", "on"}

def _load_example_bundles() -> Dict[str, "BundleEntry"]:
    if not _examples_enabled():
        return {}
    root = _examples_root()
    if not root.exists():
        return {}
    bundles: Dict[str, BundleEntry] = {}
    for item in root.iterdir():
        if not item.is_dir():
            continue
        if item.name in {"data", "__pycache__"}:
            continue
        if not (item / "entrypoint.py").exists():
            continue
        bid = item.name
        bundles[bid] = BundleEntry(
            id=bid,
            name=bid,
            path=str(item),
            module="entrypoint",
            singleton=False,
            description="Built-in example bundle",
        )
    return bundles

def _merge_example_bundles(reg: "BundlesRegistry") -> tuple["BundlesRegistry", bool]:
    examples = _load_example_bundles()
    if not examples:
        return reg, False
    updated = False
    merged = BundlesRegistry(
        default_bundle_id=reg.default_bundle_id,
        bundles=dict(reg.bundles),
    )
    for bid, entry in examples.items():
        if bid not in merged.bundles:
            merged.bundles[bid] = entry
            updated = True
    return merged, updated


def _ensure_admin_bundle(reg: "BundlesRegistry") -> "BundlesRegistry":
    if ADMIN_BUNDLE_ID not in reg.bundles:
        reg = BundlesRegistry(
            default_bundle_id=reg.default_bundle_id,
            bundles=dict(reg.bundles),
        )
        reg.bundles[ADMIN_BUNDLE_ID] = _admin_bundle_entry()
    if not reg.default_bundle_id or reg.default_bundle_id not in reg.bundles:
        reg.default_bundle_id = ADMIN_BUNDLE_ID
    return reg

class BundleEntry(BaseModel):
    id: str
    name: Optional[str] = None
    path: str                     # container-visible absolute path, e.g. /bundles/...
    module: Optional[str] = None
    singleton: Optional[bool] = False
    description: Optional[str] = None
    repo: Optional[str] = None
    ref: Optional[str] = None
    subdir: Optional[str] = None
    git_commit: Optional[str] = None

class BundlesRegistry(BaseModel):
    default_bundle_id: Optional[str] = None
    bundles: Dict[str, BundleEntry] = Field(default_factory=dict)

def _tp_from_env() -> Tuple[str,str]:
    settings = get_settings()
    return settings.TENANT, settings.PROJECT

def redis_key(tenant: Optional[str]=None, project: Optional[str]=None) -> str:
    t,p = tenant, project
    if not t or not p:
        t2,p2 = _tp_from_env()
        t = t or t2
        p = p or p2
    return REDIS_KEY_FMT.format(tenant=t, project=p)

def update_channel(tenant: Optional[str]=None, project: Optional[str]=None) -> str:
    t, p = tenant, project
    if not t or not p:
        t2, p2 = _tp_from_env()
        t = t or t2
        p = p or p2
    return REDIS_CHANNEL_FMT.format(tenant=t, project=p)

async def load_registry(redis, tenant: Optional[str] = None, project: Optional[str] = None) -> BundlesRegistry:
    """
    Load per-tenant/project registry from Redis.
    If the key is missing OR contains an 'effectively empty' registry
    (e.g. {"default_bundle_id": null, "bundles": {}}), seed once from env.
    """
    key = redis_key(tenant, project)
    raw = await redis.get(key)

    # Helper: parse raw -> BundlesRegistry (tolerant to shape)
    def _parse(raw_bytes) -> BundlesRegistry:
        try:
            return BundlesRegistry.model_validate_json(raw_bytes)
        except ValidationError:
            data = json.loads(raw_bytes)
            return BundlesRegistry(**data)

    if raw:
        reg = _parse(raw)
        # ---empty registry stored -> seed from env once ---
        if not reg.bundles or len(reg.bundles) == 0:
            seeded = await seed_from_env_if_any(redis, tenant, project)
            if seeded and seeded.bundles:
                seeded, _ = _merge_example_bundles(seeded)
                return _ensure_admin_bundle(seeded)
            # keep returning the empty registry if seeding didn't produce anything
            reg, updated = _merge_example_bundles(reg)
            reg = _ensure_admin_bundle(reg)
            if updated:
                await save_registry(redis, reg, tenant, project)
            return reg
        reg, updated = _merge_example_bundles(reg)
        reg = _ensure_admin_bundle(reg)
        if updated:
            await save_registry(redis, reg, tenant, project)
        return reg

    # Key absent -> try seeding once
    seeded = await seed_from_env_if_any(redis, tenant, project)
    if seeded:
        seeded, _ = _merge_example_bundles(seeded)
        return _ensure_admin_bundle(seeded)
    reg, updated = _merge_example_bundles(BundlesRegistry())
    reg = _ensure_admin_bundle(reg)
    if updated:
        await save_registry(redis, reg, tenant, project)
    return reg

async def save_registry(redis, reg: BundlesRegistry, tenant: Optional[str]=None, project: Optional[str]=None) -> None:
    key = redis_key(tenant, project)
    reg = _ensure_admin_bundle(reg)
    await redis.set(key, reg.model_dump_json())

async def publish_update(redis, reg: BundlesRegistry, *, tenant: Optional[str]=None, project: Optional[str]=None, op: str="merge", actor: Optional[str]=None):
    t,p = tenant, project
    if not t or not p: t,p = _tp_from_env()
    payload = {
        "tenant": t, "project": p,
        "op": op, "ts": time.time(),
        # attach full registry snapshot
        "registry": reg.model_dump()
    }
    if actor: payload["actor"] = actor
    await redis.publish(update_channel(t, p), json.dumps(payload, ensure_ascii=False))

async def seed_from_env_if_any(redis, tenant: Optional[str] = None, project: Optional[str] = None) -> Optional[BundlesRegistry]:
    """
    Seed Redis with bundles mapping from AGENTIC_BUNDLES_JSON env var, if present.
    Accepts either:
      - flat mapping: {"mybundle": {...}, ...}
      - new: {"default_bundle_id": "...", "bundles": { ... }}
    """
    data = _load_env_json(strict=False)
    if not data:
        return None

    try:
        if "bundles" not in data:
            bundles_dict = data
            default_id = next(iter(bundles_dict.keys()), None)
        else:
            bundles_dict = data.get("bundles") or {}
            default_id = data.get("default_bundle_id") or (next(iter(bundles_dict.keys()), None))

        reg = BundlesRegistry(
            default_bundle_id=default_id,
            bundles={bid: _to_entry(bid, b) for bid, b in bundles_dict.items()}
        )
        if not reg.bundles:
            return None
        reg, _ = _merge_example_bundles(reg)
        reg = _ensure_admin_bundle(reg)
        await save_registry(redis, reg, tenant, project)
        return reg

    except Exception:
        return None

async def reset_registry_from_env(redis, tenant: Optional[str] = None, project: Optional[str] = None) -> BundlesRegistry:
    """
    Force-reload from AGENTIC_BUNDLES_JSON and overwrite Redis for (tenant, project).
    Raise ValueError if env is missing or invalid.
    """
    data = _load_env_json(strict=True)
    if "bundles" not in data:
        bundles_dict = data
        default_id = next(iter(bundles_dict.keys()), None)
    else:
        bundles_dict = data.get("bundles") or {}
        default_id = data.get("default_bundle_id") or (next(iter(bundles_dict.keys()), None))

    if not bundles_dict:
        raise ValueError("AGENTIC_BUNDLES_JSON has no bundles")

    reg = BundlesRegistry(
        default_bundle_id=default_id,
        bundles={bid: _to_entry(bid, b) for bid, b in bundles_dict.items()}
    )
    reg, _ = _merge_example_bundles(reg)
    reg = _ensure_admin_bundle(reg)
    await save_registry(redis, reg, tenant, project)
    return reg


async def force_env_reset_if_requested(
    redis,
    tenant: Optional[str] = None,
    project: Optional[str] = None,
    actor: Optional[str] = None,
) -> Optional[BundlesRegistry]:
    """
    If BUNDLES_FORCE_ENV_ON_STARTUP is set, overwrite Redis registry from env once.
    Uses a Redis lock to avoid multiple workers doing it concurrently.
    """
    settings = get_settings()
    if not settings.BUNDLES_FORCE_ENV_ON_STARTUP:
        return None

    t, p = tenant, project
    if not t or not p:
        t, p = _tp_from_env()

    lock_key = namespaces.CONFIG.BUNDLES.ENV_SYNC_LOCK_FMT.format(tenant=t, project=p)
    try:
        acquired = await redis.set(lock_key, "1", nx=True, ex=settings.BUNDLES_FORCE_ENV_LOCK_TTL_SECONDS)
    except Exception:
        acquired = False

    if not acquired:
        return None

    reg = await reset_registry_from_env(redis, t, p)
    await publish_update(redis, reg, tenant=t, project=p, op="replace", actor=actor or "startup-env")
    return reg


def _to_entry(bid: str, v: Dict[str, Any]) -> BundleEntry:
    """Normalize incoming dict -> BundleEntry."""
    unsupported_keys = {"git_url", "git_ref", "git_subdir", "git_repo"}
    if any(k in v for k in unsupported_keys):
        raise ValueError("Use repo/ref/subdir only; git_* keys are not supported.")
    repo = v.get("repo")
    path_val = v.get("path") or ""
    ref = v.get("ref")
    subdir = v.get("subdir")
    if not path_val and repo:
        try:
            from kdcube_ai_app.infra.plugin.git_bundle import compute_git_bundle_paths
            paths = compute_git_bundle_paths(
                bundle_id=bid,
                git_url=repo,
                git_ref=ref,
                git_subdir=subdir,
            )
            path_val = str(paths.bundle_root)
        except Exception:
            path_val = ""
    return BundleEntry(
        id=bid,
        name=v.get("name"),
        path=path_val or v.get("path") or "",
        module=v.get("module"),
        singleton=bool(v.get("singleton", False)),
        description=v.get("description"),
        repo=repo,
        ref=ref,
        subdir=subdir,
        git_commit=v.get("git_commit"),
    )

def _load_env_json(strict: bool) -> Optional[Dict[str, Any]]:
    raw = os.getenv("AGENTIC_BUNDLES_JSON")
    if not raw:
        if strict:
            raise ValueError("AGENTIC_BUNDLES_JSON is not set")
        return None
    raw = raw.strip()
    if raw.startswith("{") or raw.startswith("["):
        return json.loads(raw)
    path = Path(raw).expanduser()
    if not path.exists():
        if strict:
            raise ValueError(f"AGENTIC_BUNDLES_JSON file not found: {path}")
        return None
    text = path.read_text()
    data: Optional[Dict[str, Any]]
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ValueError("PyYAML is required to load YAML bundle descriptors.") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not data:
        return None
    # If a release descriptor is provided, extract the bundles section.
    if isinstance(data, dict) and "bundles" in data:
        bundles_block = data.get("bundles") or {}
        if isinstance(bundles_block, dict) and "items" in bundles_block:
            items = bundles_block.get("items") or []
            bundles: Dict[str, Any] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                bid = item.get("id")
                if not bid:
                    raise ValueError("Bundle item missing 'id' in release descriptor.")
                bundles[bid] = dict(item)
            return {
                "default_bundle_id": bundles_block.get("default_bundle_id"),
                "bundles": bundles,
            }
    return data

def apply_update(
        current: BundlesRegistry,
        op: str,
        bundles_patch: Dict[str, Dict[str, Any]],
        default_bundle_id: Optional[str] = None,
) -> BundlesRegistry:
    """
    Apply a 'replace' or 'merge' patch to the current registry and
    return a NEW BundlesRegistry (does not persist).
    """
    if op not in ("replace", "merge"):
        raise ValueError("Invalid op; use 'replace' or 'merge'")

    if op == "replace":
        new_map = {bid: _to_entry(bid, v) for bid, v in (bundles_patch or {}).items()}
        new_default = (
            default_bundle_id if (default_bundle_id in new_map) else (next(iter(new_map), None))
        )
        return _ensure_admin_bundle(BundlesRegistry(default_bundle_id=new_default, bundles=new_map))

    # merge
    new_map: Dict[str, BundleEntry] = dict(current.bundles)
    for bid, v in (bundles_patch or {}).items():
        new_map[bid] = _to_entry(bid, v)

    new_default = default_bundle_id or current.default_bundle_id
    if new_default and new_default not in new_map:
        new_default = next(iter(new_map), None)

    return _ensure_admin_bundle(BundlesRegistry(default_bundle_id=new_default, bundles=new_map))
