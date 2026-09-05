# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from kdcube_cli import installer as installer_mod
from kdcube_cli.host_vault import (
    HOST_VAULT_BACKEND,
    HostVaultRuntimeConfig,
    compose_environment,
)


class HostVaultPrepareError(RuntimeError):
    def __init__(self, message: str, *, code: str, rollback: str = "not_needed") -> None:
        super().__init__(message)
        self.code = code
        self.rollback = rollback


@dataclass(frozen=True)
class HostVaultPrepareResult:
    dry_run: bool
    config_changed: bool
    broker_recreated: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "kdcube_cli.host_vault_prepare.v1",
            "status": "ready" if self.dry_run else "ok",
            "dry_run": self.dry_run,
            "config_changed": self.config_changed,
            "broker_recreated": self.broker_recreated,
            "source_provider": "secrets-file",
            "destination_backend": HOST_VAULT_BACKEND,
            "provider_changed": False,
            "source_deleted": False,
        }


class HostVaultPrepareRuntime(Protocol):
    def require_running(self) -> None: ...

    def recreate_broker(self) -> None: ...


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    mode: int

    @classmethod
    def capture(cls, path: Path) -> _FileSnapshot:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise HostVaultPrepareError(
                "Host-vault preparation requires the generated Compose environment file.",
                code="invalid_runtime_configuration",
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise HostVaultPrepareError(
                "Host-vault preparation requires a regular Compose environment file.",
                code="invalid_runtime_configuration",
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise HostVaultPrepareError(
                "Host-vault preparation could not read the generated Compose environment file.",
                code="invalid_runtime_configuration",
            ) from exc
        return cls(path=path, payload=payload, mode=metadata.st_mode & 0o777)

    def restore(self) -> None:
        _atomic_write(self.path, self.payload, mode=self.mode)


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        if os.name == "posix":
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


def _render_environment(path: Path, updates: Mapping[str, str]) -> bytes:
    try:
        env = installer_mod.load_env_file(path)
    except OSError as exc:
        raise HostVaultPrepareError(
            "Host-vault preparation could not parse the generated Compose environment file.",
            code="invalid_runtime_configuration",
        ) from exc
    for key, value in updates.items():
        installer_mod.update_env_value(env, key, value)
    return ("\n".join(env.lines).rstrip() + "\n").encode("utf-8")


def prepare_host_vault_shadow(
    *,
    config_dir: Path,
    config: HostVaultRuntimeConfig,
    runtime: HostVaultPrepareRuntime,
    dry_run: bool,
) -> HostVaultPrepareResult:
    if config.provider != "secrets-file" or config.backend != HOST_VAULT_BACKEND:
        raise HostVaultPrepareError(
            "Host-vault shadow preparation requires secrets-file with the host-vault backend.",
            code="invalid_shadow_configuration",
        )

    snapshot = _FileSnapshot.capture(Path(config_dir) / ".env")
    desired = _render_environment(snapshot.path, compose_environment(config))
    changed = desired != snapshot.payload
    runtime.require_running()

    if dry_run:
        return HostVaultPrepareResult(
            dry_run=True,
            config_changed=changed,
            broker_recreated=False,
        )

    try:
        if changed:
            _atomic_write(snapshot.path, desired, mode=snapshot.mode)
        runtime.recreate_broker()
    except HostVaultPrepareError as exc:
        if not changed:
            raise
        try:
            snapshot.restore()
            runtime.recreate_broker()
        except (HostVaultPrepareError, OSError) as rollback_exc:
            raise HostVaultPrepareError(
                "Host-vault shadow preparation failed and the previous broker could not be restored.",
                code="prepare_failed",
                rollback="failed",
            ) from rollback_exc
        raise HostVaultPrepareError(
            "Host-vault shadow preparation failed; the previous broker configuration was restored.",
            code=exc.code,
            rollback="restored",
        ) from exc
    except OSError as exc:
        try:
            snapshot.restore()
        except OSError as rollback_exc:
            raise HostVaultPrepareError(
                "Host-vault shadow preparation could not update or restore its generated configuration.",
                code="configuration_write_failed",
                rollback="failed",
            ) from rollback_exc
        raise HostVaultPrepareError(
            "Host-vault shadow preparation could not update its generated configuration.",
            code="configuration_write_failed",
            rollback="restored",
        ) from exc

    return HostVaultPrepareResult(
        dry_run=False,
        config_changed=changed,
        broker_recreated=True,
    )


class ComposeHostVaultPrepareRuntime:
    def __init__(
        self,
        *,
        docker_dir: Path,
        env_file: Path,
        timeout_seconds: float = 120.0,
        poll_seconds: float = 1.0,
    ) -> None:
        self._docker_dir = Path(docker_dir)
        self._env_file = Path(env_file)
        self._timeout = timeout_seconds
        self._poll = poll_seconds

    def _base(self, env_file: Path | None = None) -> list[str]:
        return ["docker", "compose", "--env-file", str(env_file or self._env_file)]

    def _run(
        self,
        command: list[str],
        *,
        env_file: Path,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                cwd=self._docker_dir,
                env=installer_mod.compose_env(env_file),
                text=True,
                capture_output=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HostVaultPrepareError(
                "A Docker Compose operation required by host-vault preparation could not run.",
                code="runtime_operation_failed",
            ) from exc

    def require_running(self) -> None:
        result = self._run(
            [*self._base(), "ps", "--services", "--status", "running"],
            env_file=self._env_file,
        )
        if result.returncode != 0:
            raise HostVaultPrepareError(
                "Host-vault preparation could not inspect the running deployment.",
                code="runtime_status_failed",
            )
        running = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        if "kdcube-secrets" not in running:
            raise HostVaultPrepareError(
                "Host-vault preparation requires the local kdcube-secrets service to be running.",
                code="runtime_not_running",
            )

    def _wait_for_broker(self, runtime_env: Path) -> None:
        deadline = time.monotonic() + self._timeout
        command = [
            *self._base(runtime_env),
            "exec",
            "-T",
            "kdcube-secrets",
            "python",
            "-c",
            (
                "import sys,urllib.request; "
                "r=urllib.request.urlopen('http://127.0.0.1:7777/health',timeout=2); "
                "sys.exit(0 if r.status==200 else 1)"
            ),
        ]
        while time.monotonic() < deadline:
            result = self._run(command, env_file=runtime_env)
            if result.returncode == 0:
                return
            time.sleep(self._poll)
        raise HostVaultPrepareError(
            "The host-vault shadow broker did not become healthy.",
            code="broker_not_ready",
        )

    def recreate_broker(self) -> None:
        runtime_env = installer_mod.write_env_overlay(
            self._env_file,
            installer_mod.generate_runtime_tokens(),
        )
        try:
            result = self._run(
                [
                    *self._base(runtime_env),
                    "up",
                    "-d",
                    "--force-recreate",
                    "--no-deps",
                    "kdcube-secrets",
                ],
                env_file=runtime_env,
            )
            if result.returncode != 0:
                raise HostVaultPrepareError(
                    "The host-vault shadow broker could not be recreated.",
                    code="broker_recreate_failed",
                )
            self._wait_for_broker(runtime_env)
        finally:
            runtime_env.unlink(missing_ok=True)


__all__ = [
    "ComposeHostVaultPrepareRuntime",
    "HostVaultPrepareError",
    "HostVaultPrepareResult",
    "prepare_host_vault_shadow",
]
