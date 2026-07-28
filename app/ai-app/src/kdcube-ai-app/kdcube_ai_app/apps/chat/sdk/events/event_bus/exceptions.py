# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations


class EventLaneStateLockTimeout(TimeoutError):
    """Raised when a transient event-lane state lock cannot be acquired."""

    def __init__(
        self,
        *,
        state_key: str,
        lock_key: str,
        operation: str,
        timeout_seconds: float,
        holder_operation: str = "",
        holder_pid: str = "",
        holder_task: str = "",
        holder_acquired_at: str = "",
        pttl_ms: int | None = None,
    ) -> None:
        self.state_key = str(state_key or "")
        self.lock_key = str(lock_key or "")
        self.operation = str(operation or "unspecified")
        self.timeout_seconds = float(timeout_seconds)
        self.holder_operation = str(holder_operation or "")
        self.holder_pid = str(holder_pid or "")
        self.holder_task = str(holder_task or "")
        self.holder_acquired_at = str(holder_acquired_at or "")
        self.pttl_ms = pttl_ms
        holder = self.holder_operation or "unknown"
        ttl = str(pttl_ms) if pttl_ms is not None else "unknown"
        super().__init__(
            "event lane state lock timed out"
            f" operation={self.operation}"
            f" timeout_seconds={self.timeout_seconds:.3f}"
            f" state_key={self.state_key}"
            f" lock_key={self.lock_key}"
            f" holder_operation={holder}"
            f" holder_pid={self.holder_pid or 'unknown'}"
            f" holder_task={self.holder_task or 'unknown'}"
            f" holder_acquired_at={self.holder_acquired_at or 'unknown'}"
            f" pttl_ms={ttl}"
        )


class ExternalEventLaneWakeIgnored(Exception):
    """Raised when a lane wake is valid but no processor turn should run."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason or "ignored")


class ExternalEventLaneTurnSuperseded(Exception):
    """Raised when a running turn lost ownership of its conversation event lane."""

    def __init__(
        self,
        *,
        turn_id: str,
        owner_turn_id: str = "",
        handler_status: str = "",
        conversation_id: str = "",
        phase: str = "",
    ) -> None:
        self.turn_id = str(turn_id or "")
        self.owner_turn_id = str(owner_turn_id or "")
        self.handler_status = str(handler_status or "")
        self.conversation_id = str(conversation_id or "")
        self.phase = str(phase or "")
        super().__init__(
            "external event lane turn superseded"
            f" turn_id={self.turn_id or '<empty>'}"
            f" owner_turn_id={self.owner_turn_id or '<empty>'}"
            f" handler_status={self.handler_status or '<empty>'}"
            f" phase={self.phase or '<empty>'}"
        )
