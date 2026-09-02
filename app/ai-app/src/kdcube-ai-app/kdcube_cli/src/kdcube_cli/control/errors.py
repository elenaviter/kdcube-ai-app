# SPDX-License-Identifier: MIT
from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Sequence


class ControlErrorCode(str, Enum):
    UNSUPPORTED_CAPABILITY = "target.unsupported_capability"
    MISSING_TARGET = "target.missing"
    AMBIGUOUS_TARGET = "target.ambiguous"
    INVALID_DESCRIPTOR = "descriptor.invalid"
    DOCKER_UNAVAILABLE = "docker.unavailable"
    OPERATION_FAILED = "operation.failed"
    APPLICATION_MISSING = "application.missing"
    SURFACE_MISSING = "application.surface_missing"
    SURFACE_AMBIGUOUS = "application.surface_ambiguous"


class KDCubeControlError(RuntimeError):
    """Base error returned by the reusable deployment-target control API."""

    def __init__(
        self,
        code: ControlErrorCode,
        summary: str,
        *,
        recovery: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.code = code
        self.summary = str(summary)
        self.recovery = MappingProxyType(dict(recovery or {}))
        super().__init__(self.summary)


class UnsupportedCapabilityError(KDCubeControlError):
    def __init__(self, target_id: str, capability: str) -> None:
        self.target_id = target_id
        self.capability = capability
        super().__init__(
            ControlErrorCode.UNSUPPORTED_CAPABILITY,
            f"Target {target_id!r} does not support {capability}.",
            recovery={"capability": capability, "target_id": target_id},
        )


class MissingTargetError(KDCubeControlError):
    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        super().__init__(
            ControlErrorCode.MISSING_TARGET,
            f"Deployment target was not found: {target_id}",
            recovery={"target_id": target_id},
        )


class AmbiguousTargetError(KDCubeControlError):
    def __init__(self, target_id: str, candidates: Sequence[str]) -> None:
        self.target_id = target_id
        self.candidates = tuple(str(candidate) for candidate in candidates)
        super().__init__(
            ControlErrorCode.AMBIGUOUS_TARGET,
            f"Deployment target is ambiguous: {target_id}",
            recovery={
                "target_id": target_id,
                "candidates": "\n".join(self.candidates),
            },
        )


class InvalidDescriptorError(KDCubeControlError):
    def __init__(self, descriptor: str, reason: str) -> None:
        self.descriptor = descriptor
        self.reason = reason
        super().__init__(
            ControlErrorCode.INVALID_DESCRIPTOR,
            f"Invalid descriptor {descriptor}: {reason}",
            recovery={"descriptor": descriptor},
        )


class DockerUnavailableError(KDCubeControlError):
    def __init__(self, summary: Optional[str] = None) -> None:
        super().__init__(
            ControlErrorCode.DOCKER_UNAVAILABLE,
            summary or "Docker is unavailable or is not responding.",
            recovery={"action": "Start or repair Docker, then retry."},
        )


class OperationFailedError(KDCubeControlError):
    def __init__(
        self,
        operation: str,
        target_id: str,
        summary: str,
        *,
        returncode: Optional[int] = None,
        recovery: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.operation = operation
        self.target_id = target_id
        self.returncode = returncode
        values = {"operation": operation, "target_id": target_id}
        values.update(dict(recovery or {}))
        super().__init__(ControlErrorCode.OPERATION_FAILED, summary, recovery=values)


class ApplicationNotFoundError(KDCubeControlError):
    def __init__(self, target_id: str, bundle_id: str) -> None:
        self.target_id = target_id
        self.bundle_id = bundle_id
        super().__init__(
            ControlErrorCode.APPLICATION_MISSING,
            f"Application {bundle_id!r} is not installed on target {target_id!r}.",
            recovery={"target_id": target_id, "bundle_id": bundle_id},
        )


class ApplicationSurfaceNotFoundError(KDCubeControlError):
    def __init__(self, bundle_id: str, selector: str) -> None:
        self.bundle_id = bundle_id
        self.selector = selector
        super().__init__(
            ControlErrorCode.SURFACE_MISSING,
            f"Application {bundle_id!r} has no surface matching {selector!r}.",
            recovery={"bundle_id": bundle_id, "selector": selector},
        )


class AmbiguousApplicationSurfaceError(KDCubeControlError):
    def __init__(self, bundle_id: str, candidates: Sequence[str]) -> None:
        self.bundle_id = bundle_id
        self.candidates = tuple(str(candidate) for candidate in candidates)
        super().__init__(
            ControlErrorCode.SURFACE_AMBIGUOUS,
            f"Application {bundle_id!r} has more than one matching surface.",
            recovery={
                "bundle_id": bundle_id,
                "candidates": "\n".join(self.candidates),
            },
        )
