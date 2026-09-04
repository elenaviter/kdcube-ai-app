from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from kdcube_cli.secrets_migration import (
    ComposeHostVaultDestination,
    HostVaultStageError,
    VerificationState,
    load_file_secret_inventory,
    stage_file_secrets,
)


class _MemoryDestination:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.created: list[str] = []

    def verify(self, key: str, value: str) -> VerificationState:
        if key not in self.values:
            return VerificationState.MISSING
        if self.values[key] == value:
            return VerificationState.MATCH
        return VerificationState.DIFFERENT

    def create(self, key: str, value: str) -> None:
        if key in self.values:
            raise HostVaultStageError("create refused")
        self.values[key] = value
        self.created.append(key)


def _write_inventory(config_dir: Path) -> None:
    config_dir.mkdir()
    secrets = config_dir / "secrets.yaml"
    secrets.write_text(
        """
services:
  brave:
    api_key: brave-canary
  optional:
    token: <FILL_ME>
infra:
  redis:
    password: redis-canary
""".strip()
        + "\n",
        encoding="utf-8",
    )
    bundles = config_dir / "bundles.secrets.yaml"
    bundles.write_text(
        """
bundles:
  version: '1'
  items:
    - id: connection-hub@1-0
      secrets:
        connections:
          token: connector-canary
          optional: <FILL_ME>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    secrets.chmod(0o600)
    bundles.chmod(0o600)


def _is_placeholder(value: str) -> bool:
    return "<" in value and ">" in value


def test_inventory_matches_file_provider_key_shape_and_filters_placeholders(tmp_path):
    config_dir = tmp_path / "config"
    _write_inventory(config_dir)

    inventory = load_file_secret_inventory(
        config_dir,
        is_placeholder=_is_placeholder,
    )

    bundle_key = "bundles.connection-hub@1-0.secrets.connections.token"
    assert inventory.values == {
        bundle_key: "connector-canary",
        "bundles.connection-hub@1-0.secrets.__keys": f'["{bundle_key}"]',
        "infra.redis.password": "redis-canary",
        "services.brave.api_key": "brave-canary",
    }
    assert inventory.skipped_placeholders == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes are required")
def test_inventory_refuses_non_owner_only_source(tmp_path):
    config_dir = tmp_path / "config"
    _write_inventory(config_dir)
    (config_dir / "secrets.yaml").chmod(0o644)

    with pytest.raises(HostVaultStageError, match="owner-only"):
        load_file_secret_inventory(config_dir, is_placeholder=_is_placeholder)


def test_stage_is_idempotent_and_dry_run_has_no_effect(tmp_path):
    config_dir = tmp_path / "config"
    _write_inventory(config_dir)
    inventory = load_file_secret_inventory(config_dir, is_placeholder=_is_placeholder)
    destination = _MemoryDestination()

    dry = stage_file_secrets(inventory, destination, dry_run=True)
    assert dry.created == 0
    assert dry.would_create == len(inventory.values)
    assert destination.values == {}

    first = stage_file_secrets(inventory, destination, dry_run=False)
    assert first.created == len(inventory.values)
    assert first.would_create == 0

    second = stage_file_secrets(inventory, destination, dry_run=False)
    assert second.created == 0
    assert second.already_matched == len(inventory.values)


def test_stage_refuses_all_writes_when_one_destination_value_differs(tmp_path):
    config_dir = tmp_path / "config"
    _write_inventory(config_dir)
    inventory = load_file_secret_inventory(config_dir, is_placeholder=_is_placeholder)
    destination = _MemoryDestination({"services.brave.api_key": "other-value"})

    with pytest.raises(HostVaultStageError, match=r"1 conflict\(s\)") as exc_info:
        stage_file_secrets(inventory, destination, dry_run=False)

    assert destination.created == []
    assert "brave-canary" not in str(exc_info.value)
    assert "other-value" not in str(exc_info.value)


def test_compose_destination_keeps_values_and_digests_out_of_arguments(monkeypatch, tmp_path):
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "verify" in command:
            return subprocess.CompletedProcess(command, 3, stdout="missing\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    destination = ComposeHostVaultDestination(
        docker_dir=tmp_path,
        env_file=tmp_path / ".env",
        environment={"PATH": "/usr/bin"},
    )
    secret = "must-not-enter-process-arguments"

    assert destination.verify("services.fixture.token", secret) is VerificationState.MISSING
    destination.create("services.fixture.token", secret)

    assert len(calls) == 2
    assert all(secret not in command for command, _kwargs in calls)
    verify_command, verify_kwargs = calls[0]
    create_command, create_kwargs = calls[1]
    assert verify_command[-3:] == ["verify", "services.fixture.token", "--sha256-stdin"]
    assert len(str(verify_kwargs["input"])) == 64
    assert str(verify_kwargs["input"]) not in verify_command
    assert create_command[-4:] == [
        "set",
        "services.fixture.token",
        "--stdin",
        "--if-absent",
    ]
    assert create_kwargs["input"] == secret


def test_compose_destination_retries_transient_broker_failure(monkeypatch, tmp_path):
    returncodes = iter((5, 3))
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            next(returncodes),
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("kdcube_cli.secrets_migration.time.sleep", sleeps.append)
    destination = ComposeHostVaultDestination(
        docker_dir=tmp_path,
        env_file=tmp_path / ".env",
        environment={"PATH": "/usr/bin"},
        transient_delay_seconds=0.25,
    )

    assert destination.verify("services.fixture.token", "value") is VerificationState.MISSING
    assert len(calls) == 2
    assert sleeps == [0.25]
