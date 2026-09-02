# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import yaml

from kdcube_cli import installer as installer_mod
from kdcube_cli.control.errors import OperationFailedError
from kdcube_cli.control.execution import CommandRunner, SubprocessCommandRunner
from kdcube_cli.control.initialization import EventSink
from kdcube_cli.control.models import (
    ControlEvent,
    ControlEventKind,
    DeploymentTargetRef,
    LocalPlatformSourceRequest,
    PreparedPlatformSource,
    TargetKind,
)


def _emit(event_sink: Optional[EventSink], message: str) -> None:
    if event_sink is not None:
        event_sink(ControlEvent(kind=ControlEventKind.PROGRESS, message=message))


def _run(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    cwd: Optional[Path],
    operation: str,
    target_id: str,
    required: bool = True,
):
    try:
        result = runner.run(command, cwd=cwd, capture_output=True)
    except (OSError, ValueError) as exc:
        raise OperationFailedError(
            operation,
            target_id,
            "The KDCube platform source command could not be started.",
        ) from exc
    if required and result.returncode != 0:
        raise OperationFailedError(
            operation,
            target_id,
            "The KDCube platform source could not be prepared.",
            returncode=result.returncode,
        )
    return result


def _is_git_repository(
    runner: CommandRunner,
    path: Path,
    *,
    target_id: str,
) -> bool:
    if not path.exists():
        return False
    result = _run(
        runner,
        ("git", "-C", str(path), "rev-parse", "--show-toplevel"),
        cwd=None,
        operation="prepare_source",
        target_id=target_id,
        required=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return False
    return Path(result.stdout.strip()).expanduser().resolve() == path.resolve()


def _release_ref_from_yaml(text: str, *, target_id: str) -> str:
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OperationFailedError(
            "prepare_source",
            target_id,
            "The KDCube release record could not be read.",
        ) from exc
    platform = payload.get("platform") if isinstance(payload, dict) else None
    release_ref = (
        str(platform.get("ref") or "").strip() if isinstance(platform, dict) else ""
    )
    if not release_ref:
        raise OperationFailedError(
            "prepare_source",
            target_id,
            "The KDCube release record does not declare platform.ref.",
        )
    return release_ref


def _checkout_release(
    runner: CommandRunner,
    repo_root: Path,
    release_ref: str,
    *,
    target_id: str,
) -> None:
    _run(
        runner,
        ("git", "fetch", "--tags", "origin"),
        cwd=repo_root,
        operation="prepare_source",
        target_id=target_id,
        required=False,
    )
    for candidate in (
        release_ref,
        f"origin/{release_ref}",
        f"refs/tags/{release_ref}",
        f"tags/{release_ref}",
    ):
        result = _run(
            runner,
            ("git", "checkout", "--detach", candidate),
            cwd=repo_root,
            operation="prepare_source",
            target_id=target_id,
            required=False,
        )
        if result.returncode == 0:
            return
    raise OperationFailedError(
        "prepare_source",
        target_id,
        "The requested KDCube release is unavailable in the platform repository.",
        recovery={"release_ref": release_ref},
    )


def prepare_local_platform_source(
    target: DeploymentTargetRef,
    request: LocalPlatformSourceRequest = LocalPlatformSourceRequest(),
    *,
    runner: Optional[CommandRunner] = None,
    event_sink: Optional[EventSink] = None,
) -> PreparedPlatformSource:
    """Prepare the managed platform checkout used by a fresh local target."""

    if target.kind != TargetKind.LOCAL or target.workdir is None:
        raise ValueError("Platform source preparation requires a local target.")
    if request.upstream and str(request.release_ref or "").strip():
        raise OperationFailedError(
            "prepare_source",
            target.target_id,
            "Choose either upstream source or one release, not both.",
        )

    selected_runner = runner or SubprocessCommandRunner()
    repo_root = (target.workdir / "repo").resolve()
    descriptors = repo_root / "app" / "ai-app" / "deployment"
    target.workdir.mkdir(parents=True, exist_ok=True)

    if not _is_git_repository(selected_runner, repo_root, target_id=target.target_id):
        if repo_root.exists() and any(repo_root.iterdir()):
            raise OperationFailedError(
                "prepare_source",
                target.target_id,
                "The managed KDCube source directory exists but is not a Git repository.",
                recovery={"path": str(repo_root)},
            )
        normalized = installer_mod.normalize_git_repo_source(request.repository)
        _emit(event_sink, "Cloning the KDCube platform source.")
        _run(
            selected_runner,
            ("git", "clone", normalized, str(repo_root)),
            cwd=None,
            operation="prepare_source",
            target_id=target.target_id,
        )

    if request.upstream:
        _emit(event_sink, "Selecting the current KDCube upstream source.")
        _run(
            selected_runner,
            ("git", "fetch", "origin", "main"),
            cwd=repo_root,
            operation="prepare_source",
            target_id=target.target_id,
        )
        _run(
            selected_runner,
            ("git", "checkout", "--detach", "origin/main"),
            cwd=repo_root,
            operation="prepare_source",
            target_id=target.target_id,
        )
        resolved = _run(
            selected_runner,
            ("git", "rev-parse", "HEAD"),
            cwd=repo_root,
            operation="prepare_source",
            target_id=target.target_id,
        ).stdout.strip()
        if not resolved:
            raise OperationFailedError(
                "prepare_source",
                target.target_id,
                "The selected KDCube upstream revision could not be resolved.",
            )
        install_mode = "upstream"
    else:
        resolved = str(request.release_ref or "").strip()
        if not resolved:
            _emit(event_sink, "Resolving the latest KDCube release.")
            _run(
                selected_runner,
                ("git", "fetch", "origin", "main"),
                cwd=repo_root,
                operation="prepare_source",
                target_id=target.target_id,
            )
            release_record = _run(
                selected_runner,
                ("git", "show", "origin/main:release.yaml"),
                cwd=repo_root,
                operation="prepare_source",
                target_id=target.target_id,
            )
            resolved = _release_ref_from_yaml(
                release_record.stdout, target_id=target.target_id
            )
        _emit(event_sink, f"Selecting KDCube release {resolved}.")
        _checkout_release(
            selected_runner,
            repo_root,
            resolved,
            target_id=target.target_id,
        )
        install_mode = "skip" if request.build else "release"

    if (
        not (descriptors / "assembly.yaml").is_file()
        or not (
            repo_root / "app" / "ai-app" / "src" / "kdcube-ai-app" / "kdcube_ai_app"
        ).is_dir()
    ):
        raise OperationFailedError(
            "prepare_source",
            target.target_id,
            "The selected repository revision is not a KDCube platform source tree.",
        )

    return PreparedPlatformSource(
        repo_root=repo_root,
        descriptor_source=descriptors,
        release_ref=resolved,
        install_mode=install_mode,
    )
