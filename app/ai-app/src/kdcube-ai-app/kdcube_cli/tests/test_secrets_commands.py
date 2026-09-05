from __future__ import annotations

import argparse
import io
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from kdcube_cli.management.errors import ManagementCliError
from kdcube_cli.management.models import ManagementResult
from kdcube_cli.secrets_backend import backend_status
from kdcube_cli.secrets_commands import run_management_secret_command
from kdcube_cli.secrets_parser import configure_secrets_parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    secrets = commands.add_parser("secrets")

    def add_quiet(command: argparse.ArgumentParser) -> None:
        command.add_argument("-q", "--quiet", action="store_true")

    configure_secrets_parser(
        secrets,
        add_quiet=add_quiet,
        default_path=Path("/platform"),
    )
    return parser


class _SuccessfulClient:
    calls: ClassVar[list[tuple[Any, str]]] = []
    result: ClassVar[dict[str, Any] | None] = None

    def __init__(self, *, transport):
        self.transport = transport

    async def execute(self, request, *, bearer):
        self.__class__.calls.append((request, bearer))
        result = self.__class__.result
        assert isinstance(result, dict)
        return ManagementResult(
            operation=request.operation,
            resource=request.resource,
            invocation_id=request.invocation_id,
            replay=False,
            authority={"access_id": "caller-access"},
            result=result,
        )


@pytest.fixture(autouse=True)
def _reset_client():
    _SuccessfulClient.calls = []
    _SuccessfulClient.result = None


def test_parser_exposes_backend_neutral_commands_and_host_vault_alias():
    parser = _parser()

    canonical = parser.parse_args(
        ["secrets", "backend", "host-vault", "stage", "--dry-run"]
    )
    compatibility = parser.parse_args(["secrets", "host-vault", "stage", "--dry-run"])
    metadata = parser.parse_args(
        [
            "secrets",
            "metadata",
            "services.fixture.token",
            "--scope",
            "platform",
            "--endpoint",
            "https://kdcube.example.test",
            "--tenant",
            "tenant-a",
            "--project",
            "project-a",
        ]
    )

    assert canonical.secrets_backend_action == "host-vault"
    assert compatibility.secrets_command == "host-vault"
    assert canonical.secrets_action == compatibility.secrets_action == "stage"
    assert metadata.secrets_command == "metadata"
    assert not hasattr(metadata, "secrets_backend_action")


def test_set_reads_bearer_then_exact_value_from_stdin_without_printing_either(
    monkeypatch,
    capsys,
):
    import kdcube_cli.secrets_commands as commands

    caller = "caller-bearer-canary"
    secret = "provider-secret-canary\nsecond-line"
    args = _parser().parse_args(
        [
            "secrets",
            "set",
            "services.fixture.token",
            "--scope",
            "platform",
            "--endpoint",
            "https://kdcube.example.test",
            "--tenant",
            "tenant-a",
            "--project",
            "project-a",
            "--credential-stdin",
            "--value-stdin",
        ]
    )
    _SuccessfulClient.result = {
        "scope": "platform",
        "key": "services.fixture.token",
        "state": "stored",
        "created": True,
        "provider": "host-vault",
    }
    monkeypatch.setattr(commands, "ManagementClient", _SuccessfulClient)
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{caller}\n{secret}"))

    exit_code = run_management_secret_command(
        args,
        local_workdir=None,
        tenant="tenant-a",
        project="project-a",
    )

    assert exit_code == 0
    request, bearer = _SuccessfulClient.calls[0]
    assert bearer == caller
    assert request.body["value"] == secret
    output = capsys.readouterr().out
    assert caller not in output
    assert secret not in output
    assert json.loads(output)["result"]["state"] == "stored"


def test_get_writes_private_file_without_printing_value(monkeypatch, capsys, tmp_path):
    import kdcube_cli.secrets_commands as commands

    caller = "caller-bearer-canary"
    secret = "provider-secret-canary"
    output_path = tmp_path / "disclosed.txt"
    args = _parser().parse_args(
        [
            "secrets",
            "get",
            "services.fixture.token",
            "--scope",
            "platform",
            "--endpoint",
            "https://kdcube.example.test",
            "--tenant",
            "tenant-a",
            "--project",
            "project-a",
            "--credential-stdin",
            "--output",
            str(output_path),
        ]
    )
    _SuccessfulClient.result = {
        "scope": "platform",
        "key": "services.fixture.token",
        "value": secret,
    }
    monkeypatch.setattr(commands, "ManagementClient", _SuccessfulClient)
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{caller}\n"))

    exit_code = run_management_secret_command(
        args,
        local_workdir=None,
        tenant="tenant-a",
        project="project-a",
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == secret
    if os.name == "posix":
        assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    output = capsys.readouterr().out
    assert caller not in output
    assert secret not in output
    assert json.loads(output)["result"]["output"] == str(output_path.absolute())


def test_local_target_is_derived_from_runtime_metadata(monkeypatch, capsys, tmp_path):
    import kdcube_cli.secrets_commands as commands

    workdir = tmp_path / "tenant-a__project-a"
    config = workdir / "config"
    config.mkdir(parents=True)
    (config / ".env").write_text("KDCUBE_PROXY_HTTP_PORT=8088\n", encoding="utf-8")
    (config / "install-meta.json").write_text(
        json.dumps({"tenant": "tenant-a", "project": "project-a"}),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "secrets",
            "metadata",
            "services.fixture.token",
            "--scope",
            "platform",
            "--workdir",
            str(workdir),
            "--credential-stdin",
        ]
    )
    _SuccessfulClient.result = {
        "scope": "platform",
        "key": "services.fixture.token",
        "exists": True,
        "provider": "secrets-file",
        "writable": True,
    }
    monkeypatch.setattr(commands, "ManagementClient", _SuccessfulClient)
    monkeypatch.setattr(sys, "stdin", io.StringIO("caller-bearer\n"))

    assert (
        run_management_secret_command(
            args,
            local_workdir=workdir,
            tenant="",
            project="",
        )
        == 0
    )

    request, _ = _SuccessfulClient.calls[0]
    assert request.target.public_base_url == "http://localhost:8088"
    assert request.target.tenant == "tenant-a"
    assert request.target.project == "project-a"
    assert json.loads(capsys.readouterr().out)["result"]["provider"] == ("secrets-file")


def test_remote_target_requires_exact_coordinates():
    args = _parser().parse_args(
        [
            "secrets",
            "metadata",
            "services.fixture.token",
            "--scope",
            "platform",
            "--endpoint",
            "https://kdcube.example.test",
        ]
    )

    with pytest.raises(ManagementCliError) as exc_info:
        run_management_secret_command(
            args,
            local_workdir=None,
            tenant="",
            project="",
        )

    assert exc_info.value.code == "management_coordinates_required"


def test_backend_status_distinguishes_shadow_from_authoritative_host_vault(tmp_path):
    workdir = tmp_path / "tenant-a__project-a"
    config = workdir / "config"
    config.mkdir(parents=True)
    (config / "install-meta.json").write_text(
        json.dumps({"tenant": "tenant-a", "project": "project-a"}),
        encoding="utf-8",
    )
    assembly = {
        "context": {"tenant": "tenant-a", "project": "project-a"},
        "secrets": {
            "provider": "secrets-file",
            "service": {
                "backend": "host-vault",
                "host_vault": {
                    "address": "host.docker.internal:7781",
                    "server_name": "host.docker.internal",
                    "identity_dir": "/private/identity",
                },
            },
        },
    }
    (config / "assembly.yaml").write_text(
        yaml.safe_dump(assembly),
        encoding="utf-8",
    )

    shadow = backend_status(workdir=workdir)
    assert shadow["authoritative_store"] == {
        "kind": "descriptor-files",
        "evidence": "staged-descriptor",
        "runtime_verified": False,
    }
    assert shadow["host_vault"]["state"] == "shadow-configured"
    assert shadow["descriptor_values_authoritative"] is True

    assembly["secrets"]["provider"] = "secrets-service"
    (config / "assembly.yaml").write_text(
        yaml.safe_dump(assembly),
        encoding="utf-8",
    )
    active = backend_status(workdir=workdir)
    assert active["authoritative_store"]["kind"] == "host-vault"
    assert active["host_vault"]["state"] == "active"
    assert active["descriptor_values_authoritative"] is False


def test_backend_status_requires_exact_local_coordinates(tmp_path):
    workdir = tmp_path / "runtime"
    config = workdir / "config"
    config.mkdir(parents=True)
    (config / "assembly.yaml").write_text(
        yaml.safe_dump({"secrets": {"provider": "secrets-file"}}),
        encoding="utf-8",
    )

    with pytest.raises(ManagementCliError) as exc_info:
        backend_status(workdir=workdir)

    assert exc_info.value.code == "management_local_target_invalid"
