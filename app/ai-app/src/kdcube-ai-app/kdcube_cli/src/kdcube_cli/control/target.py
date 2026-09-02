# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from kdcube_cli.control.models import (
    ApplicationRef,
    ApplicationSurface,
    DeploymentTargetRef,
    OperationResult,
    SurfaceSelector,
    TargetCapabilities,
    TargetStatus,
)


BrowserOpener = Callable[[str], bool]


@runtime_checkable
class DeploymentTarget(Protocol):
    @property
    def reference(self) -> DeploymentTargetRef:
        ...

    @property
    def capabilities(self) -> TargetCapabilities:
        ...

    def describe(self) -> TargetStatus:
        ...

    def resolve_surface(
        self,
        application: ApplicationRef,
        selector: SurfaceSelector,
    ) -> ApplicationSurface:
        ...

    def application_url(
        self,
        application: ApplicationRef,
        selector: SurfaceSelector,
    ) -> str:
        ...

    def open_application(
        self,
        application: ApplicationRef,
        selector: SurfaceSelector,
        *,
        opener: BrowserOpener = ...,
    ) -> OperationResult:
        ...
