# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Append-only, secret-safe audit records.

Every vault operation, allowed or denied, appends one line: who (deployment
id and certificate fingerprint), what (operation, reference DIGEST, expected
and resulting generation), correlation (request id), outcome (code), and
when. Never the secret name, never a value, never a backend message. The
digest lets an operator correlate a reference across records without the
audit file becoming a directory of secret names.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class AuditEvent:
    time: float
    deployment_id: str
    fingerprint: str
    application: str
    operation: str
    reference_digest: str
    request_id: str
    code: str
    generation: int | None = None
    expected_generation: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "deployment_id": self.deployment_id,
            "fingerprint": self.fingerprint,
            "application": self.application,
            "operation": self.operation,
            "reference_digest": self.reference_digest,
            "request_id": self.request_id,
            "code": self.code,
            "generation": self.generation,
            "expected_generation": self.expected_generation,
        }


class AuditSink(Protocol):
    def append(self, event: AuditEvent) -> None: ...


class MemoryAuditSink:
    """Test sink."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)


class FileAuditSink:
    """JSON lines, append-only (O_APPEND), fsynced per event. The file is
    service-owned; rotation is the host's log policy, not the vault's."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AuditEvent) -> None:
        line = json.dumps(event.to_dict(), sort_keys=True) + "\n"
        with self._lock:
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)


def event_now(**fields: Any) -> AuditEvent:
    return AuditEvent(time=time.time(), **fields)


__all__ = ["AuditEvent", "AuditSink", "FileAuditSink", "MemoryAuditSink", "event_now"]
