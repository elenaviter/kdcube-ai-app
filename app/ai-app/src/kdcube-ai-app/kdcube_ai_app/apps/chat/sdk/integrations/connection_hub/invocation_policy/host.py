"""Compose the portable invocation-policy service over shared bundle storage."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from connection_hub.invocation_policy import (
    BundleStorageInvocationPolicyStore,
    InvocationPolicyConflict,
    InvocationPolicyService,
)
from kdcube_ai_app.storage.observed_file_locks import (
    ObservedFileLockTimeout,
    observed_file_lock_async,
)


@asynccontextmanager
async def _invocation_policy_mutation_lock(**kwargs: Any):
    try:
        async with observed_file_lock_async(**kwargs) as metadata:
            yield metadata
    except ObservedFileLockTimeout as exc:
        raise InvocationPolicyConflict("invocation_policy_lock_timeout") from exc


def build_invocation_policy_service(*, storage_root: Any) -> InvocationPolicyService:
    return InvocationPolicyService(
        store=BundleStorageInvocationPolicyStore(storage_root),
        mutation_lock=_invocation_policy_mutation_lock,
    )


__all__ = ["build_invocation_policy_service"]
