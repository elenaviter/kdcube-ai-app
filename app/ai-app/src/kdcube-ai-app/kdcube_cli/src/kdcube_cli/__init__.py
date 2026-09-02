# SPDX-License-Identifier: MIT
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kdcube_cli.control import (
        ApplicationRef,
        DeploymentTargetRef,
        EndpointDeploymentTarget,
        LocalDeploymentTarget,
        SurfaceKind,
        SurfaceSelector,
        discover_local_targets,
        select_local_target,
    )

__all__ = [
    "ApplicationRef",
    "DeploymentTargetRef",
    "EndpointDeploymentTarget",
    "LocalDeploymentTarget",
    "SurfaceKind",
    "SurfaceSelector",
    "discover_local_targets",
    "select_local_target",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("kdcube_cli.control"), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
