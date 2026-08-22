# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Trusted parent bridge for workspace tools hosted in a child process.

Claude Code starts local MCP servers as subprocesses. Those children can copy
ordinary ``conv:fi:`` bytes, but they do not own the active ContextBrowser,
namespace rehosters, conversation hosting service, or request authority. This
short-lived Unix-socket broker keeps those objects in the parent process while
letting the child request bounded operations:

* materialize one object locator through the parent's authorized resolver;
* publish selected current-turn ``files/...`` paths through the parent's host.
* stage one semantic turn summary for successful-turn persistence.

The child supplies refs and paths only. Identity and credentials never cross the
socket. A random bearer and a mode-0600 socket bind requests to this one turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import secrets
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
    MaterializedWorkspaceSource,
    WorkspaceSourceResolver,
)


WorkspacePublisher = Callable[..., Awaitable[Sequence[Mapping[str, Any]] | Mapping[str, Any]]]
WorkspaceSummaryContributor = Callable[..., Awaitable[Mapping[str, Any]] | Mapping[str, Any]]
_MAX_MESSAGE_BYTES = 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 120.0


class WorkspaceBrokerError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class WorkspaceBroker:
    socket_path: Path
    token: str
    server: asyncio.AbstractServer
    temp_root: Path
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.close()
        await self.server.wait_closed()
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        shutil.rmtree(self.temp_root, ignore_errors=True)

    async def __aenter__(self) -> "WorkspaceBroker":
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()


def _source_payload(source: MaterializedWorkspaceSource | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, MaterializedWorkspaceSource):
        return {
            "requested_ref": source.requested_ref,
            "resolved_ref": source.resolved_ref,
            "local_path": str(source.local_path),
            "object_ref": source.object_ref,
            "mime": source.mime,
            "kind": source.kind,
        }
    if isinstance(source, Mapping):
        return dict(source)
    raise WorkspaceBrokerError(
        "invalid_source_resolver_result",
        "trusted workspace resolver returned an unsupported result",
    )


async def start_workspace_broker(
    *,
    source_resolver: Optional[WorkspaceSourceResolver] = None,
    publisher: Optional[WorkspacePublisher] = None,
    summary_contributor: Optional[WorkspaceSummaryContributor] = None,
) -> WorkspaceBroker:
    """Start one turn-scoped broker and return its child binding."""
    temp_root = Path(tempfile.mkdtemp(prefix="kdcube-ws-broker-"))
    os.chmod(temp_root, 0o700)
    socket_path = temp_root / "broker.sock"
    token = secrets.token_urlsafe(32)

    async def _reply(writer: asyncio.StreamWriter, payload: Mapping[str, Any]) -> None:
        writer.write((json.dumps(dict(payload), ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=_REQUEST_TIMEOUT_SECONDS)
            if not raw or len(raw) > _MAX_MESSAGE_BYTES:
                raise WorkspaceBrokerError("invalid_request", "workspace broker request is empty or too large")
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise WorkspaceBrokerError("invalid_request", "workspace broker request must be an object")
            supplied = str(request.get("token") or "")
            if not secrets.compare_digest(supplied, token):
                raise WorkspaceBrokerError("unauthorized", "workspace broker token is invalid")

            operation = str(request.get("operation") or "").strip()
            if operation == "materialize":
                if source_resolver is None:
                    raise WorkspaceBrokerError(
                        "source_resolver_unavailable",
                        "this hosted runtime has no trusted owner-ref resolver",
                    )
                ref = str(request.get("ref") or "").strip()
                if not ref:
                    raise WorkspaceBrokerError("missing_ref", "materialization requires a ref")
                staging_dir = temp_root / "staging" / uuid.uuid4().hex
                staging_dir.mkdir(parents=True, exist_ok=False)
                resolved = source_resolver(ref=ref, staging_dir=staging_dir)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                if resolved is None:
                    raise WorkspaceBrokerError("source_not_materialized", f"could not materialize {ref}")
                await _reply(writer, {"ok": True, "source": _source_payload(resolved)})
            elif operation == "publish":
                if publisher is None:
                    raise WorkspaceBrokerError(
                        "publisher_unavailable",
                        "this hosted runtime has no conversation file publisher",
                    )
                raw_paths = request.get("paths")
                if not isinstance(raw_paths, list):
                    raise WorkspaceBrokerError("invalid_paths", "publish requires a list of files/... paths")
                paths = [str(path or "").strip() for path in raw_paths if str(path or "").strip()]
                published = publisher(paths=paths)
                if inspect.isawaitable(published):
                    published = await published
                await _reply(writer, {"ok": True, "published": published or []})
            elif operation == "record_turn_summary":
                if summary_contributor is None:
                    raise WorkspaceBrokerError(
                        "turn_summary_unavailable",
                        "this hosted runtime has no turn-summary contribution binding",
                    )
                contribution = summary_contributor(
                    summary=str(request.get("summary") or ""),
                    refs=request.get("refs"),
                    phrases=request.get("phrases"),
                    entities=request.get("entities"),
                )
                if inspect.isawaitable(contribution):
                    contribution = await contribution
                await _reply(writer, {"ok": True, "contribution": contribution or {}})
            else:
                raise WorkspaceBrokerError(
                    "unsupported_operation",
                    f"unsupported workspace broker operation: {operation or '<none>'}",
                )
        except Exception as error:
            await _reply(
                writer,
                {
                    "ok": False,
                    "error": getattr(error, "code", "workspace_broker_failed"),
                    "message": str(error),
                },
            )
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    try:
        server = await asyncio.start_unix_server(
            _handle,
            path=str(socket_path),
            limit=_MAX_MESSAGE_BYTES,
        )
        os.chmod(socket_path, 0o600)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return WorkspaceBroker(
        socket_path=socket_path,
        token=token,
        server=server,
        temp_root=temp_root,
    )


async def request_workspace_broker(
    *,
    socket_path: str,
    token: str,
    operation: str,
    payload: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Call the trusted parent from a local workspace MCP child."""
    if not socket_path or not token:
        raise WorkspaceBrokerError("broker_unavailable", "trusted workspace broker is not configured")
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(path=socket_path),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    try:
        request = {"token": token, "operation": operation, **dict(payload or {})}
        writer.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()
        raw = await asyncio.wait_for(reader.readline(), timeout=_REQUEST_TIMEOUT_SECONDS)
        response = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(response, dict) or not response.get("ok"):
            raise WorkspaceBrokerError(
                str(response.get("error") or "workspace_broker_failed"),
                str(response.get("message") or "trusted workspace broker failed"),
            )
        if operation == "materialize":
            return response.get("source")
        if operation == "record_turn_summary":
            return response.get("contribution")
        return response.get("published")
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


def broker_source_resolver(*, socket_path: str, token: str) -> WorkspaceSourceResolver:
    async def _resolve(*, ref: str, staging_dir: Path) -> Any:
        del staging_dir
        try:
            return await request_workspace_broker(
                socket_path=socket_path,
                token=token,
                operation="materialize",
                payload={"ref": ref},
            )
        except WorkspaceBrokerError as error:
            if error.code == "source_resolver_unavailable":
                # A publisher-only binding still lets the child use the
                # built-in conv:fi reader. Owner namespaces remain denied by
                # resolve_workspace_source when no trusted resolver exists.
                return None
            raise

    return _resolve


__all__ = [
    "WorkspaceBroker",
    "WorkspaceBrokerError",
    "WorkspacePublisher",
    "WorkspaceSummaryContributor",
    "broker_source_resolver",
    "request_workspace_broker",
    "start_workspace_broker",
]
