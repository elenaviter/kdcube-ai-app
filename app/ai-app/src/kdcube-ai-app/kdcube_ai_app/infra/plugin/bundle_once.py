# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import pathlib
import shutil
import socket
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Optional

_log = logging.getLogger("kdcube.plugin.bundle_once")

OnceStatus = Literal[
    "already_current",
    "ran",
    "became_current",
    "lock_existing",
    "lock_timeout_existing",
]


@dataclass(frozen=True)
class SharedStorageOnceResult:
    status: OnceStatus
    storage_root: pathlib.Path
    operation: str
    lock_path: pathlib.Path
    signature_path: pathlib.Path
    ran: bool = False


class SharedStorageOnceTimeout(TimeoutError):
    pass


_LOCK_HEARTBEAT_FILE = "heartbeat"


def _sanitize_segment(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in (raw or ""))
    safe = safe.strip("-_.")
    return safe or "operation"


def _logger_log(logger: Any, message: str, level: str = "INFO") -> None:
    if logger is None:
        getattr(_log, level.lower(), _log.info)(message)
        return
    log_fn = getattr(logger, "log", None)
    if callable(log_fn):
        try:
            log_fn(message, level)
            return
        except TypeError:
            pass
    std_fn = getattr(logger, level.lower(), None)
    if callable(std_fn):
        std_fn(message)


async def _run_action(action: Callable[[], Awaitable[Any]]) -> Any:
    value = action()
    if not inspect.isawaitable(value):
        raise TypeError("run_once_for_shared_bundle_storage action must return an awaitable")
    return await value


async def _run_ready(ready: Callable[[], bool | Awaitable[bool]]) -> bool:
    if inspect.iscoroutinefunction(ready):
        value = ready()
    else:
        value = await asyncio.to_thread(ready)
    if inspect.isawaitable(value):
        value = await value
    return bool(value)


async def _signature_current(
    *,
    signature_path: pathlib.Path,
    signature: str,
    ready: Callable[[], bool | Awaitable[bool]],
) -> bool:
    try:
        current = await asyncio.to_thread(signature_path.read_text, encoding="utf-8")
        return current.strip() == signature and await _run_ready(ready)
    except Exception:
        return False


def _write_signature(signature_path: pathlib.Path, signature: str) -> None:
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = signature_path.with_name(f".{signature_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp_path.write_text(f"{signature}\n", encoding="utf-8")
    tmp_path.replace(signature_path)


def _lock_freshness_mtime(lock_path: pathlib.Path) -> float:
    heartbeat_path = lock_path / _LOCK_HEARTBEAT_FILE
    try:
        return heartbeat_path.stat().st_mtime
    except OSError:
        return lock_path.stat().st_mtime


def _remove_stale_lock(lock_path: pathlib.Path, *, lock_ttl_seconds: float) -> bool:
    try:
        stale = (time.time() - _lock_freshness_mtime(lock_path)) > lock_ttl_seconds
    except OSError:
        stale = True
    if not stale:
        return False
    shutil.rmtree(lock_path, ignore_errors=True)
    return True


def _lock_age_seconds(lock_path: pathlib.Path) -> Optional[float]:
    try:
        return max(0.0, time.time() - _lock_freshness_mtime(lock_path))
    except OSError:
        return None


def _touch_lock_heartbeat(lock_path: pathlib.Path) -> None:
    heartbeat_path = lock_path / _LOCK_HEARTBEAT_FILE
    heartbeat_path.write_text(f"{time.time()}\n", encoding="utf-8")


async def _heartbeat_lock(lock_path: pathlib.Path, *, interval_seconds: float) -> None:
    interval = max(0.01, float(interval_seconds))
    while True:
        try:
            await asyncio.to_thread(_touch_lock_heartbeat, lock_path)
        except Exception:
            pass
        await asyncio.sleep(interval)


def _read_lock_owner(lock_path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads((lock_path / "owner.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _lock_owner_summary(lock_path: pathlib.Path) -> str:
    owner = _read_lock_owner(lock_path)
    if not owner:
        return "owner=unknown"
    pieces = []
    for key in ("host", "pid", "operation", "bundle_id", "kind"):
        value = owner.get(key)
        if value is not None and value != "":
            pieces.append(f"{key}={value}")
    return "owner=" + ",".join(pieces) if pieces else "owner=unknown"


async def run_once_for_shared_bundle_storage(
    *,
    storage_root: str | os.PathLike[str],
    operation: str,
    signature_path: str | os.PathLike[str],
    signature: str,
    ready: Callable[[], bool | Awaitable[bool]],
    action: Callable[[], Awaitable[Any]],
    logger: Optional[Any] = None,
    owner_metadata: Optional[dict[str, Any]] = None,
    lock_wait_seconds: float = 600.0,
    lock_ttl_seconds: float = 900.0,
    lock_heartbeat_interval_seconds: float | None = None,
    poll_interval_seconds: float = 0.25,
    allow_existing_while_locked: bool = False,
    allow_existing_on_timeout: bool = True,
    log_prefix: str = "[bundle.once]",
) -> SharedStorageOnceResult:
    """
    Run an idempotent storage-scoped action once for all processes sharing a bundle storage root.

    This is async-first so request handlers and bundle lifecycle hooks can await it.

    The action is considered current when both are true:
    - signature_path contains `signature`
    - ready() returns True; it may be synchronous or awaitable

    The lock is a directory under `<storage_root>/.kdcube.once/`, which works for
    local filesystems and shared mounts such as EFS. Filesystem operations run
    outside the proc event loop.
    """
    root, sig_path = await asyncio.to_thread(
        lambda: (
            pathlib.Path(storage_root).expanduser().resolve(),
            pathlib.Path(signature_path).expanduser().resolve(),
        )
    )
    op = _sanitize_segment(operation)
    lock_path = root / ".kdcube.once" / f"{op}.lock"

    if await _signature_current(signature_path=sig_path, signature=signature, ready=ready):
        _logger_log(logger, f"{log_prefix} skipped: signature cache hit op={op} storage={root}", "INFO")
        return SharedStorageOnceResult("already_current", root, op, lock_path, sig_path, ran=False)

    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(lock_path.parent.mkdir, parents=True, exist_ok=True)
    deadline = time.time() + max(0.0, float(lock_wait_seconds))
    lock_acquired = False
    last_wait_log_at = 0.0

    while True:
        if await _signature_current(signature_path=sig_path, signature=signature, ready=ready):
            _logger_log(logger, f"{log_prefix} skipped: became current op={op} storage={root}", "INFO")
            return SharedStorageOnceResult("became_current", root, op, lock_path, sig_path, ran=False)

        try:
            await asyncio.to_thread(lock_path.mkdir, parents=True, exist_ok=False)
            lock_acquired = True
            owner = dict(owner_metadata or {})
            owner.update(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "created_at": time.time(),
                    "operation": op,
                    "signature": signature,
                }
            )
            try:
                await asyncio.to_thread(
                    (lock_path / "owner.json").write_text,
                    json.dumps(owner, sort_keys=True),
                    encoding="utf-8",
                )
                await asyncio.to_thread(_touch_lock_heartbeat, lock_path)
            except Exception:
                pass
            _logger_log(logger, f"{log_prefix} lock acquired op={op} storage={root}", "INFO")
            break
        except FileExistsError:
            owner_summary = await asyncio.to_thread(_lock_owner_summary, lock_path)
            removed_stale = await asyncio.to_thread(
                _remove_stale_lock,
                lock_path,
                lock_ttl_seconds=float(lock_ttl_seconds),
            )
            if removed_stale:
                _logger_log(
                    logger,
                    f"{log_prefix} removed stale lock op={op} storage={root} {owner_summary}",
                    "WARNING",
                )
                continue
            if allow_existing_while_locked and await _run_ready(ready):
                age = await asyncio.to_thread(_lock_age_seconds, lock_path)
                age_part = f" lock_age_sec={age:.1f}" if age is not None else ""
                _logger_log(
                    logger,
                    f"{log_prefix} lock held; using existing output op={op} storage={root}{age_part} {owner_summary}",
                    "WARNING",
                )
                return SharedStorageOnceResult("lock_existing", root, op, lock_path, sig_path, ran=False)
            if time.time() >= deadline:
                if allow_existing_on_timeout and await _run_ready(ready):
                    _logger_log(
                        logger,
                        f"{log_prefix} lock wait timed out; using existing output "
                        f"op={op} storage={root} {owner_summary}",
                        "WARNING",
                    )
                    return SharedStorageOnceResult("lock_timeout_existing", root, op, lock_path, sig_path, ran=False)
                raise SharedStorageOnceTimeout(f"{op} lock wait timed out for storage={root}")
            now = time.time()
            if last_wait_log_at <= 0.0 or (now - last_wait_log_at) >= 5.0:
                age = await asyncio.to_thread(_lock_age_seconds, lock_path)
                age_part = f" lock_age_sec={age:.1f}" if age is not None else ""
                _logger_log(
                    logger,
                    f"{log_prefix} waiting for lock op={op} storage={root}{age_part} {owner_summary}",
                    "WARNING",
                )
                last_wait_log_at = now
            await asyncio.sleep(max(0.01, float(poll_interval_seconds)))

    heartbeat_task: asyncio.Task | None = None
    try:
        if await _signature_current(signature_path=sig_path, signature=signature, ready=ready):
            _logger_log(logger, f"{log_prefix} skipped: signature cache hit under lock op={op} storage={root}", "INFO")
            return SharedStorageOnceResult("became_current", root, op, lock_path, sig_path, ran=False)

        heartbeat_interval = (
            float(lock_heartbeat_interval_seconds)
            if lock_heartbeat_interval_seconds is not None
            else min(max(1.0, float(lock_ttl_seconds) / 3.0), 10.0)
        )
        heartbeat_task = asyncio.create_task(
            _heartbeat_lock(lock_path, interval_seconds=heartbeat_interval),
            name=f"bundle-once-heartbeat:{op}",
        )
        await _run_action(action)
        if not await _run_ready(ready):
            raise RuntimeError(f"{op} action completed but ready() is false for storage={root}")
        await asyncio.to_thread(_write_signature, sig_path, signature)
        _logger_log(logger, f"{log_prefix} done: op={op} storage={root}", "INFO")
        return SharedStorageOnceResult("ran", root, op, lock_path, sig_path, ran=True)
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        if lock_acquired:
            await asyncio.to_thread(shutil.rmtree, lock_path, ignore_errors=True)
