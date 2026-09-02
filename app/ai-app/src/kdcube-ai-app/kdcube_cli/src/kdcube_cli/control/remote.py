# SPDX-License-Identifier: MIT
from __future__ import annotations

import webbrowser
from typing import Callable

from kdcube_cli.control.errors import (
    ApplicationSurfaceNotFoundError,
    OperationFailedError,
    UnsupportedCapabilityError,
)
from kdcube_cli.control.models import (
    ApplicationRef,
    ApplicationSurface,
    DeploymentTargetRef,
    Diagnostic,
    DiagnosticSeverity,
    OperationResult,
    ReleaseCoordinates,
    SurfaceSelector,
    TargetCapabilities,
    TargetCapability,
    TargetKind,
    TargetStatus,
)
from kdcube_cli.control.surfaces import build_application_surface, normalize_endpoint


BrowserOpener = Callable[[str], bool]

ENDPOINT_CAPABILITIES = TargetCapabilities(
    frozenset({TargetCapability.RESOLVE_ENDPOINTS, TargetCapability.OPEN})
)


class EndpointDeploymentTarget:
    """Endpoint-only target for an already deployed KDCube application server."""

    def __init__(self, reference: DeploymentTargetRef) -> None:
        if reference.kind != TargetKind.ENDPOINT or not reference.endpoint:
            raise ValueError("EndpointDeploymentTarget requires an endpoint target reference.")
        endpoint = normalize_endpoint(reference.endpoint)
        if not reference.tenant or not reference.project:
            raise ValueError("Endpoint targets require tenant and project coordinates.")
        self._reference = DeploymentTargetRef.endpoint_target(
            endpoint,
            tenant=reference.tenant,
            project=reference.project,
            target_id=reference.target_id,
        )

    @property
    def reference(self) -> DeploymentTargetRef:
        return self._reference

    @property
    def capabilities(self) -> TargetCapabilities:
        return ENDPOINT_CAPABILITIES

    def describe(self) -> TargetStatus:
        return TargetStatus(
            reference=self.reference,
            capabilities=self.capabilities,
            reachable=None,
            initialized=None,
            running=None,
            release=ReleaseCoordinates(),
            diagnostics=(
                Diagnostic(
                    code="remote.endpoint_only",
                    severity=DiagnosticSeverity.INFO,
                    summary=(
                        "This target resolves application endpoints; remote status and management "
                        "are not defined."
                    ),
                    recovery={"management": "Use a target with an authorized management API."},
                ),
            ),
            public_base_url=self.reference.endpoint,
        )

    def status(self) -> TargetStatus:
        self._unsupported(TargetCapability.STATUS)
        raise AssertionError("unreachable")

    def initialize(self, *_args: object, **_kwargs: object) -> OperationResult:
        self._unsupported(TargetCapability.INITIALIZE)
        raise AssertionError("unreachable")

    def start(self, *_args: object, **_kwargs: object) -> OperationResult:
        self._unsupported(TargetCapability.START)
        raise AssertionError("unreachable")

    def stop(self, *_args: object, **_kwargs: object) -> OperationResult:
        self._unsupported(TargetCapability.STOP)
        raise AssertionError("unreachable")

    def refresh(self, *_args: object, **_kwargs: object) -> OperationResult:
        self._unsupported(TargetCapability.REFRESH)
        raise AssertionError("unreachable")

    def resolve_surface(
        self,
        application: ApplicationRef,
        selector: SurfaceSelector,
    ) -> ApplicationSurface:
        if selector.kind is None or not str(selector.alias or "").strip():
            raise ApplicationSurfaceNotFoundError(application.bundle_id, selector.label())
        return build_application_surface(
            self.reference,
            application,
            selector.kind,
            str(selector.alias).strip(),
            endpoint=str(self.reference.endpoint),
            declared=False,
        )

    def application_url(
        self,
        application: ApplicationRef,
        selector: SurfaceSelector,
    ) -> str:
        surface = self.resolve_surface(application, selector)
        if not surface.openable:
            raise ApplicationSurfaceNotFoundError(
                application.bundle_id,
                f"openable:{surface.surface_id}",
            )
        return surface.url

    def open_application(
        self,
        application: ApplicationRef,
        selector: SurfaceSelector,
        *,
        opener: BrowserOpener = webbrowser.open,
    ) -> OperationResult:
        url = self.application_url(application, selector)
        try:
            opened = bool(opener(url))
        except Exception as exc:
            raise OperationFailedError(
                "open",
                self.reference.target_id,
                f"The browser could not open the application URL: {type(exc).__name__}",
            ) from exc
        if not opened:
            raise OperationFailedError(
                "open",
                self.reference.target_id,
                "The browser did not accept the application URL.",
            )
        return OperationResult(
            target=self.reference,
            operation="open",
            changed=False,
            url=url,
        )

    def _unsupported(self, capability: TargetCapability) -> None:
        raise UnsupportedCapabilityError(self.reference.target_id, capability.value)
