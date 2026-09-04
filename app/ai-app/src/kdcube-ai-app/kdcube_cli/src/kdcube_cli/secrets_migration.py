# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import yaml


class HostVaultStageError(RuntimeError):
    pass


class VerificationState(str, Enum):
    MATCH = "match"
    MISSING = "missing"
    DIFFERENT = "different"


@dataclass(frozen=True)
class FileSecretInventory:
    values: Mapping[str, str]
    skipped_placeholders: int


@dataclass(frozen=True)
class HostVaultStageResult:
    discovered: int
    created: int
    would_create: int
    already_matched: int
    skipped_placeholders: int
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "kdcube_cli.host_vault_stage.v1",
            "ok": True,
            "source_provider": "secrets-file",
            "destination_backend": "host-vault",
            "discovered": self.discovered,
            "created": self.created,
            "already_matched": self.already_matched,
            "skipped_placeholders": self.skipped_placeholders,
            "would_create": self.would_create,
            "dry_run": self.dry_run,
            "source_deleted": False,
            "provider_changed": False,
        }


class HostVaultDestination(Protocol):
    def verify(self, key: str, value: str) -> VerificationState: ...

    def create(self, key: str, value: str) -> None: ...


def _load_mapping(path: Path, *, required: bool) -> dict[str, object]:
    if path.is_symlink():
        raise HostVaultStageError("A secrets descriptor must not be a symbolic link.")
    if not path.exists():
        if required:
            raise HostVaultStageError("The staged secrets.yaml descriptor is missing.")
        return {}
    if not path.is_file():
        raise HostVaultStageError("A secrets descriptor is not a regular file.")
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise HostVaultStageError(
            "Secrets descriptors must be owner-only before host-vault staging (mode 0600)."
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise HostVaultStageError("A secrets descriptor could not be read safely.") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise HostVaultStageError("A secrets descriptor must contain a mapping.")
    return loaded


def _flatten(
    prefix: str,
    node: object,
    values: dict[str, str],
    *,
    is_placeholder: Callable[[str], bool],
) -> int:
    if node is None:
        return 0
    if isinstance(node, dict):
        skipped = 0
        for key, value in node.items():
            child = str(key or "").strip()
            if not child:
                continue
            child_prefix = f"{prefix}.{child}" if prefix else child
            skipped += _flatten(
                child_prefix,
                value,
                values,
                is_placeholder=is_placeholder,
            )
        return skipped
    if isinstance(node, list):
        skipped = 0
        for index, value in enumerate(node):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            skipped += _flatten(
                child_prefix,
                value,
                values,
                is_placeholder=is_placeholder,
            )
        return skipped
    text = str(node).strip()
    if not text:
        return 0
    if is_placeholder(text):
        return 1
    values[prefix] = text
    return 0


def load_file_secret_inventory(
    config_dir: Path,
    *,
    is_placeholder: Callable[[str], bool],
) -> FileSecretInventory:
    config_dir = Path(config_dir).expanduser().resolve()
    global_data = _load_mapping(config_dir / "secrets.yaml", required=True)
    bundle_data = _load_mapping(config_dir / "bundles.secrets.yaml", required=False)
    values: dict[str, str] = {}
    skipped = 0

    global_root = (
        global_data.get("secrets")
        if isinstance(global_data.get("secrets"), dict)
        else global_data
    )
    skipped += _flatten(
        "",
        global_root,
        values,
        is_placeholder=is_placeholder,
    )

    bundle_root = (
        bundle_data.get("bundles")
        if isinstance(bundle_data.get("bundles"), dict)
        else bundle_data
    )
    items = bundle_root.get("items") if isinstance(bundle_root, dict) else None
    if items is not None and not isinstance(items, list):
        raise HostVaultStageError("bundles.secrets.yaml items must be a list.")
    for item in items or []:
        if not isinstance(item, dict):
            raise HostVaultStageError("Each bundles.secrets.yaml item must be a mapping.")
        bundle_id = str(item.get("id") or "").strip()
        if not bundle_id:
            raise HostVaultStageError("Each bundles.secrets.yaml item requires an id.")
        secrets = item.get("secrets")
        if secrets is None:
            continue
        prefix = f"bundles.{bundle_id}.secrets"
        skipped += _flatten(
            prefix,
            secrets,
            values,
            is_placeholder=is_placeholder,
        )

    return FileSecretInventory(values=dict(sorted(values.items())), skipped_placeholders=skipped)


class ComposeHostVaultDestination:
    def __init__(
        self,
        *,
        docker_dir: Path,
        env_file: Path,
        environment: Mapping[str, str],
        timeout_seconds: float = 10.0,
        transient_attempts: int = 3,
        transient_delay_seconds: float = 1.0,
    ) -> None:
        self._docker_dir = Path(docker_dir)
        self._environment = dict(environment)
        self._timeout = timeout_seconds
        self._transient_attempts = max(1, int(transient_attempts))
        self._transient_delay = max(0.0, float(transient_delay_seconds))
        self._base_command = [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "exec",
            "-T",
            "kdcube-secrets",
            "python",
            "/app/secretsctl.py",
        ]

    def _run(self, arguments: list[str], *, stdin: str) -> subprocess.CompletedProcess[str]:
        last_timeout: subprocess.TimeoutExpired | None = None
        for attempt in range(self._transient_attempts):
            try:
                result = subprocess.run(
                    [*self._base_command, *arguments],
                    cwd=self._docker_dir,
                    env=self._environment,
                    input=stdin,
                    text=True,
                    capture_output=True,
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                last_timeout = exc
                result = None
            except OSError as exc:
                raise HostVaultStageError(
                    "The host-vault broker command could not run."
                ) from exc
            if result is not None and result.returncode != 5:
                return result
            if attempt + 1 < self._transient_attempts:
                time.sleep(self._transient_delay)
        if last_timeout is not None:
            raise HostVaultStageError(
                "The host-vault broker command timed out."
            ) from last_timeout
        if result is not None:
            return result
        raise HostVaultStageError("The host-vault broker command could not run.")

    def verify(self, key: str, value: str) -> VerificationState:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        result = self._run(["verify", key, "--sha256-stdin"], stdin=digest)
        state_by_code = {
            0: VerificationState.MATCH,
            3: VerificationState.MISSING,
            4: VerificationState.DIFFERENT,
        }
        state = state_by_code.get(result.returncode)
        if state is None:
            raise HostVaultStageError("The host-vault broker could not verify a value.")
        return state

    def create(self, key: str, value: str) -> None:
        result = self._run(["set", key, "--stdin", "--if-absent"], stdin=value)
        if result.returncode != 0:
            raise HostVaultStageError("The host-vault broker refused a create-only write.")


def stage_file_secrets(
    inventory: FileSecretInventory,
    destination: HostVaultDestination,
    *,
    dry_run: bool,
) -> HostVaultStageResult:
    states = {
        key: destination.verify(key, value)
        for key, value in inventory.values.items()
    }
    differing = sum(state is VerificationState.DIFFERENT for state in states.values())
    if differing:
        raise HostVaultStageError(
            "Host-vault staging refused to overwrite existing values "
            f"({differing} conflict(s))."
        )

    missing = [key for key, state in states.items() if state is VerificationState.MISSING]
    matched = len(states) - len(missing)
    if dry_run:
        return HostVaultStageResult(
            discovered=len(states),
            created=0,
            would_create=len(missing),
            already_matched=matched,
            skipped_placeholders=inventory.skipped_placeholders,
            dry_run=True,
        )

    created = 0
    for key in missing:
        value = inventory.values[key]
        try:
            destination.create(key, value)
        except HostVaultStageError as exc:
            if destination.verify(key, value) is VerificationState.MATCH:
                matched += 1
                continue
            raise HostVaultStageError(
                "Host-vault staging stopped after a create-only write was refused; "
                f"{created} new value(s) remain safely staged."
            ) from exc
        if destination.verify(key, value) is not VerificationState.MATCH:
            raise HostVaultStageError(
                "Host-vault staging stopped after readback verification failed; "
                f"{created + 1} new value(s) may remain staged."
            )
        created += 1

    for key, value in inventory.values.items():
        if destination.verify(key, value) is not VerificationState.MATCH:
            raise HostVaultStageError(
                "Host-vault staging completed writes but final verification failed."
            )

    return HostVaultStageResult(
        discovered=len(states),
        created=created,
        would_create=0,
        already_matched=matched,
        skipped_placeholders=inventory.skipped_placeholders,
        dry_run=False,
    )


__all__ = [
    "ComposeHostVaultDestination",
    "FileSecretInventory",
    "HostVaultStageError",
    "HostVaultStageResult",
    "VerificationState",
    "load_file_secret_inventory",
    "stage_file_secrets",
]
