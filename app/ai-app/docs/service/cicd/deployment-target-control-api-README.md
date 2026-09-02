---
id: repo:kdcube/app/ai-app/docs/service/cicd/deployment-target-control-api-README.md
title: "KDCube Deployment Target Control API"
summary: "Documents the supported typed Python API for selecting a KDCube deployment target, inspecting local applications, controlling local lifecycle, and resolving local or remote application surfaces without importing CLI internals."
tags: ["service", "cicd", "cli", "python-api", "deployment-target", "application-control"]
keywords: ["kdcube_cli.control", "local deployment target", "endpoint deployment target", "application surface", "structured control error", "Connection Hub CLI"]
see_also:
  - repo:kdcube/app/ai-app/docs/service/cicd/cli-README.md
  - repo:kdcube/app/ai-app/docs/configuration/bundles-descriptor-README.md
  - repo:kdcube/app/ai-app/docs/arch/security-and-trust-model-README.md
---
# KDCube Deployment Target Control API

`kdcube_cli.control` is the supported Python boundary for a product CLI that
needs to locate a KDCube app server, inspect its installed applications, manage
a supported local lifecycle, or resolve an application surface. The `kdcube`
executable uses the same boundary for local discovery, `info`, prepare-only
initialization, `start`, and `stop`.

The first distribution version carrying this boundary is
`kdcube-cli==2026.09.02.1733`. Before that package is published, coordinated
consumers must install this KDCube source checkout rather than depending on an
older package with the same imports absent.

The boundary returns immutable typed data and raises `KDCubeControlError`
subclasses. Argument parsing, Rich rendering, prompts, and `SystemExit` remain
in the executable layer.

## Target contract

`DeploymentTargetRef` represents either:

- a local namespaced runtime with a filesystem `workdir`; or
- an endpoint-only target with an HTTP(S) endpoint and no filesystem or Docker
  assumption.

Every adapter implements the runtime-checkable `DeploymentTarget` protocol:
`reference`, `capabilities`, `describe()`, surface resolution, URL resolution,
and browser open. `describe()` is the common non-mutating inspection boundary.
For a local target it returns the available runtime status; for an
endpoint-only target, reachability and initialization remain unknown because
no remote status protocol is claimed. A downstream test double can implement
this protocol without exposing a workdir or Docker behavior.

Ask `target.capabilities.supports(...)` before an operation. Unsupported
operations fail closed with `UnsupportedCapabilityError`.

| Capability | Local target | Endpoint-only target |
| --- | --- | --- |
| initialize | prepare a descriptor-owned runtime | unsupported |
| start / stop | Docker Compose lifecycle | unsupported |
| refresh | unsupported by this first library slice; the executable still composes existing steps | unsupported |
| descriptor changes | unsupported | unsupported |
| application reload | unsupported | unsupported |
| logs | unsupported | unsupported |
| status | local filesystem, descriptor, application, release, and optional Docker status | unsupported |
| resolve endpoints | installed descriptor inventory | explicit app/surface coordinates |
| open | browser UI surfaces | explicit browser UI surface |

Endpoint-only support is URL resolution for an already deployed app. It is not
remote deployment management. A future remote management implementation needs
an explicit app-server management API and an operator authorization type.
Delegated credentials issued for an application caller are not deployment
authority and are not accepted by this API.

## Local application resolution

The local target reads the staged, descriptor-owned `assembly.yaml`,
`bundles.yaml`, and `install-meta.json`. It derives public routes from declared
KDCube surface coordinates:

- `config.ui.widgets.<alias>` and `surfaces.as_provider.widget.<alias>` become
  `/public/widgets/<alias>`;
- `surfaces.as_provider.mcp.<alias>` becomes `/public/mcp/<alias>`;
- `surfaces.as_provider.api.<alias>` becomes `/operations/<alias>`;
- `config.ui.main_view` becomes the bundle static main-view route.

Application inventory accepts both the current `bundles: {items: ...}` wrapper
and the earlier unwrapped `items:` catalog retained by the executable's info
path.

The API does not inspect application source to guess product intent. When an
application has multiple browser surfaces, pass a `SurfaceSelector`; otherwise
resolution raises `AmbiguousApplicationSurfaceError` with candidate surface
identifiers.

```python
from pathlib import Path

from kdcube_cli.control import (
    ApplicationRef,
    LocalDeploymentTarget,
    SurfaceKind,
    SurfaceSelector,
    select_local_target,
)

reference = select_local_target(
    Path.home() / ".kdcube/kdcube-runtime/demo-tenant__demo-project"
)
target = LocalDeploymentTarget(reference)
surface = target.resolve_surface(
    ApplicationRef("connection-hub@1-0"),
    SurfaceSelector(kind=SurfaceKind.WIDGET, alias="connections_settings"),
)
print(surface.url)
```

The complete runnable example is
`kdcube_cli.examples.resolve_application_surface`.

## Endpoint-only resolution

An endpoint target requires tenant, project, app, surface kind, and alias. It
constructs standard KDCube coordinates and marks the surface `declared=False`
because this first target has no remote inventory protocol.

```python
from kdcube_cli.control import (
    ApplicationRef,
    DeploymentTargetRef,
    EndpointDeploymentTarget,
    SurfaceKind,
    SurfaceSelector,
)

target = EndpointDeploymentTarget(
    DeploymentTargetRef.endpoint_target(
        "https://runtime.example",
        tenant="acme",
        project="prod",
    )
)
url = target.application_url(
    ApplicationRef("connection-hub@1-0"),
    SurfaceSelector(kind=SurfaceKind.WIDGET, alias="connections_settings"),
)
```

## Initialization and lifecycle

`LocalDeploymentTarget.initialize()` takes a `LocalInitializationRequest`.
Configuration comes from the request's descriptor directory, or from the
selected platform repository's deployment descriptors. The operation prepares
the runtime and writes its staged runtime configuration; it does not start
Docker. Call `start()` explicitly after initialization.

`start()` and `stop()` preserve the local single-active-runtime lock and return
`OperationResult`. Callers can receive typed progress and command events through
an optional event sink. The API itself does not print.

## Errors and diagnostics

| Exception | Stable code |
| --- | --- |
| `UnsupportedCapabilityError` | `target.unsupported_capability` |
| `MissingTargetError` | `target.missing` |
| `AmbiguousTargetError` | `target.ambiguous` |
| `InvalidDescriptorError` | `descriptor.invalid` |
| `DockerUnavailableError` | `docker.unavailable` |
| `OperationFailedError` | `operation.failed` |
| `ApplicationNotFoundError` | `application.missing` |
| `ApplicationSurfaceNotFoundError` | `application.surface_missing` |
| `AmbiguousApplicationSurfaceError` | `application.surface_ambiguous` |

Exceptions and `Diagnostic` records contain a code, summary, and bounded
recovery coordinates. They do not carry descriptor secret values or raw
process output. `ApplicationStatus.source_ref` contains only the descriptor's
bundle release ref; repository URLs are not copied into public status.
