# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, Protocol

from kdcube_cli import installer as installer_mod
from kdcube_cli.control.errors import (
    InvalidDescriptorError,
    KDCubeControlError,
    MissingTargetError,
    OperationFailedError,
)
from kdcube_cli.control.models import (
    ControlEvent,
    ControlEventKind,
    DeploymentTargetRef,
    LocalInitializationRequest,
)
from kdcube_cli.control.surfaces import load_descriptor


EventSink = Callable[[ControlEvent], None]


class RuntimeInitializer(Protocol):
    def prepare(
        self,
        *,
        target: DeploymentTargetRef,
        repo_root: Path,
        request: LocalInitializationRequest,
        event_sink: Optional[EventSink],
    ) -> None:
        ...


class _EventConsole:
    """Duck-typed installer output adapter; it has no prompt implementation."""

    def __init__(self, event_sink: Optional[EventSink]) -> None:
        self._event_sink = event_sink

    def print(self, *values: object, **_kwargs: object) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            ControlEvent(
                kind=ControlEventKind.PROGRESS,
                message=" ".join(str(value) for value in values),
            )
        )

    def input(self, *_args: object, **_kwargs: object) -> str:
        raise RuntimeError("Interactive input is unavailable in the control API.")


_INSTALLER_ENV_LOCK = threading.RLock()


@contextmanager
def _installer_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Isolate the legacy installer's process environment compatibility layer."""

    with _INSTALLER_ENV_LOCK:
        previous = dict(os.environ)
        try:
            os.environ.update(values)
            yield
        finally:
            os.environ.clear()
            os.environ.update(previous)


class InstallerRuntimeInitializer:
    """Prepare a runtime through the existing non-interactive installer core."""

    def prepare(
        self,
        *,
        target: DeploymentTargetRef,
        repo_root: Path,
        request: LocalInitializationRequest,
        event_sink: Optional[EventSink],
    ) -> None:
        if target.workdir is None:
            raise MissingTargetError(target.target_id)
        descriptor_source = (
            Path(request.descriptor_source).expanduser().resolve()
            if request.descriptor_source is not None
            else (repo_root / "app" / "ai-app" / "deployment").resolve()
        )
        assembly_path = descriptor_source / "assembly.yaml"
        load_descriptor(assembly_path)

        parameterize_defaults = request.parameterize_defaults or request.descriptor_source is None
        environment = {
            "KDCUBE_CLI_NONINTERACTIVE": "1",
            "KDCUBE_INIT_PREPARE_ONLY": "1",
            "KDCUBE_DESCRIPTORS_LOCATION": str(descriptor_source),
            "KDCUBE_ASSEMBLY_DESCRIPTOR_PATH": str(assembly_path),
            "KDCUBE_ASSEMBLY_USER_SUPPLIED": "0" if parameterize_defaults else "1",
            "KDCUBE_DEFAULT_DESCRIPTOR_BOOTSTRAP": "1" if parameterize_defaults else "0",
            "KDCUBE_DRY_RUN_PRINT_ENV": "0",
        }
        if target.tenant and target.project:
            environment["KDCUBE_PRESET_TENANT"] = target.tenant
            environment["KDCUBE_PRESET_PROJECT"] = target.project
        try:
            with _installer_environment(environment):
                installer_mod.run_setup(
                    _EventConsole(event_sink),
                    repo_root=repo_root,
                    workdir=target.workdir,
                    install_mode=request.install_mode,
                    release_ref=request.release_ref,
                    docker_namespace=request.docker_namespace,
                    dry_run=True,
                )
        except KDCubeControlError:
            raise
        except FileNotFoundError as exc:
            raise InvalidDescriptorError(
                str(descriptor_source), "a required descriptor or platform file is missing"
            ) from exc
        except ValueError as exc:
            raise InvalidDescriptorError(
                str(descriptor_source), "descriptor validation failed"
            ) from exc
        except SystemExit as exc:
            raise OperationFailedError(
                "initialize",
                target.target_id,
                "Runtime initialization failed in the non-interactive preparation step.",
            ) from exc
        except Exception as exc:
            raise OperationFailedError(
                "initialize",
                target.target_id,
                f"Runtime initialization failed: {type(exc).__name__}",
            ) from exc
