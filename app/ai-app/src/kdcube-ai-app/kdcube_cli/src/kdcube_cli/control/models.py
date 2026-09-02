# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Tuple


class TargetKind(str, Enum):
    LOCAL = "local"
    ENDPOINT = "endpoint"


class TargetCapability(str, Enum):
    INITIALIZE = "initialize"
    START = "start"
    STOP = "stop"
    REFRESH = "refresh"
    DESCRIPTOR_CHANGES = "descriptor_changes"
    APPLICATION_RELOAD = "application_reload"
    LOGS = "logs"
    STATUS = "status"
    RESOLVE_ENDPOINTS = "resolve_endpoints"
    OPEN = "open"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SurfaceKind(str, Enum):
    WIDGET = "widget"
    MAIN_UI = "main_ui"
    MCP = "mcp"
    OPERATION = "operation"


class ControlEventKind(str, Enum):
    PROGRESS = "progress"
    COMMAND = "command"


@dataclass(frozen=True)
class DeploymentTargetRef:
    target_id: str
    kind: TargetKind
    tenant: Optional[str] = None
    project: Optional[str] = None
    workdir: Optional[Path] = None
    endpoint: Optional[str] = None

    @classmethod
    def local(
        cls,
        workdir: Path,
        *,
        tenant: Optional[str] = None,
        project: Optional[str] = None,
    ) -> "DeploymentTargetRef":
        resolved = Path(workdir).expanduser().resolve()
        return cls(
            target_id=f"local:{resolved}",
            kind=TargetKind.LOCAL,
            tenant=tenant,
            project=project,
            workdir=resolved,
        )

    @classmethod
    def endpoint_target(
        cls,
        endpoint: str,
        *,
        tenant: str,
        project: str,
        target_id: Optional[str] = None,
    ) -> "DeploymentTargetRef":
        normalized = str(endpoint or "").strip().rstrip("/")
        return cls(
            target_id=target_id or f"endpoint:{normalized}",
            kind=TargetKind.ENDPOINT,
            tenant=str(tenant or "").strip() or None,
            project=str(project or "").strip() or None,
            endpoint=normalized,
        )

    def __post_init__(self) -> None:
        if self.kind == TargetKind.LOCAL:
            if self.workdir is None or self.endpoint is not None:
                raise ValueError("A local target requires workdir and has no endpoint.")
        elif self.kind == TargetKind.ENDPOINT:
            if not self.endpoint or self.workdir is not None:
                raise ValueError("An endpoint target requires endpoint and has no workdir.")


@dataclass(frozen=True)
class TargetCapabilities:
    supported: FrozenSet[TargetCapability]

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported", frozenset(self.supported))

    def supports(self, capability: TargetCapability) -> bool:
        return capability in self.supported

    def as_mapping(self) -> Mapping[str, bool]:
        return MappingProxyType(
            {capability.value: capability in self.supported for capability in TargetCapability}
        )


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: DiagnosticSeverity
    summary: str
    recovery: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recovery", MappingProxyType(dict(self.recovery)))


@dataclass(frozen=True)
class ReleaseCoordinates:
    platform_ref: Optional[str] = None
    install_mode: Optional[str] = None
    docker_namespace: Optional[str] = None


@dataclass(frozen=True)
class LocalTargetPaths:
    workdir: Path
    config_dir: Path
    data_dir: Path
    logs_dir: Path
    docker_dir: Path
    repo_root: Path
    assembly_path: Optional[Path] = None
    bundles_path: Optional[Path] = None
    compose_mode: Optional[str] = None
    host_bundles_path: Optional[str] = None
    container_bundles_root: Optional[str] = None
    host_managed_bundles_path: Optional[str] = None
    container_managed_bundles_root: Optional[str] = None
    host_bundle_storage_path: Optional[str] = None
    container_bundle_storage_root: Optional[str] = None
    host_exec_workspace_path: Optional[str] = None
    container_exec_workspace_root: Optional[str] = None
    host_react_debug_path: Optional[str] = None
    container_react_debug_root: Optional[str] = None


@dataclass(frozen=True)
class ApplicationRef:
    bundle_id: str

    def __post_init__(self) -> None:
        normalized = str(self.bundle_id or "").strip()
        if not normalized:
            raise ValueError("bundle_id is required")
        object.__setattr__(self, "bundle_id", normalized)


@dataclass(frozen=True)
class SurfaceSelector:
    kind: Optional[SurfaceKind] = None
    alias: Optional[str] = None

    def label(self) -> str:
        kind = self.kind.value if self.kind is not None else "any"
        alias = str(self.alias or "*").strip() or "*"
        return f"{kind}:{alias}"


@dataclass(frozen=True)
class ApplicationSurface:
    application: ApplicationRef
    kind: SurfaceKind
    alias: str
    route: str
    url: str
    declared: bool = True
    openable: bool = False

    @property
    def surface_id(self) -> str:
        return f"{self.kind.value}:{self.alias}"


@dataclass(frozen=True)
class ApplicationStatus:
    reference: ApplicationRef
    name: Optional[str]
    installed: bool
    enabled: bool
    surfaces: Tuple[ApplicationSurface, ...] = ()
    source_ref: Optional[str] = None
    diagnostics: Tuple[Diagnostic, ...] = ()


@dataclass(frozen=True)
class TargetStatus:
    reference: DeploymentTargetRef
    capabilities: TargetCapabilities
    reachable: Optional[bool]
    initialized: Optional[bool]
    running: Optional[bool]
    release: ReleaseCoordinates
    applications: Tuple[ApplicationStatus, ...] = ()
    diagnostics: Tuple[Diagnostic, ...] = ()
    local_paths: Optional[LocalTargetPaths] = None
    default_application_id: Optional[str] = None
    public_base_url: Optional[str] = None


@dataclass(frozen=True)
class ControlEvent:
    kind: ControlEventKind
    message: str


@dataclass(frozen=True)
class OperationResult:
    target: DeploymentTargetRef
    operation: str
    changed: bool
    running: Optional[bool] = None
    url: Optional[str] = None
    diagnostics: Tuple[Diagnostic, ...] = ()


@dataclass(frozen=True)
class LocalInitializationRequest:
    descriptor_source: Optional[Path] = None
    install_mode: str = "release"
    release_ref: Optional[str] = None
    docker_namespace: Optional[str] = None
    parameterize_defaults: bool = False


@dataclass(frozen=True)
class LocalStartRequest:
    build: bool = False


@dataclass(frozen=True)
class LocalStopRequest:
    remove_volumes: bool = False
