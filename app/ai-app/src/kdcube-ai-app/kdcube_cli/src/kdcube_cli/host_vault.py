# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

EPHEMERAL_BACKEND = "ephemeral"
HOST_VAULT_BACKEND = "host-vault"
DEPLOYMENT_SECRET_APPLICATION = "kdcube-runtime"
HOST_VAULT_ACTIVATION_MARKER = ".host-vault-activation.pending.json"
IDENTITY_FILENAMES = (
    "host-vault-client.crt",
    "host-vault-client.key",
    "host-vault-ca.crt",
)


class HostVaultConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class HostVaultRuntimeConfig:
    provider: str
    backend: str
    tenant: str
    project: str
    address: str = ""
    server_name: str = ""
    identity_dir: Path | None = None
    exec_network_mode: str = ""

    @property
    def enabled(self) -> bool:
        return self.backend == HOST_VAULT_BACKEND

    @property
    def identity_paths(self) -> tuple[Path, Path, Path]:
        if self.identity_dir is None:
            raise HostVaultConfigurationError(
                "secrets.service.host_vault.identity_dir is required"
            )
        return (
            self.identity_dir / IDENTITY_FILENAMES[0],
            self.identity_dir / IDENTITY_FILENAMES[1],
            self.identity_dir / IDENTITY_FILENAMES[2],
        )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return str(value or "").strip()


def _provider_name(value: object) -> str:
    provider = _text(value).lower().replace("_", "-")
    if provider in {"local", "service", "sidecar", "secrets-service"}:
        return "secrets-service"
    if provider in {"file", "yaml", "yaml-file", "secrets-file"}:
        return "secrets-file"
    if provider in {"aws", "aws-sm", "awssm"}:
        return "aws-sm"
    if provider in {"memory", "in-memory", "inmemory"}:
        return "in-memory"
    return provider


def _backend_name(value: object) -> str:
    backend = _text(value).lower().replace("_", "-") or EPHEMERAL_BACKEND
    if backend in {"memory", "transient", "ephemeral-memory"}:
        return EPHEMERAL_BACKEND
    return backend


def _validate_address(address: str) -> None:
    if "://" in address:
        raise HostVaultConfigurationError(
            "secrets.service.host_vault.address must be host:port without a URL scheme"
        )
    try:
        parsed = urlsplit(f"//{address}")
        port = parsed.port
    except ValueError as exc:
        raise HostVaultConfigurationError(
            "secrets.service.host_vault.address must contain a valid port"
        ) from exc
    if not parsed.hostname or port is None or parsed.username or parsed.password:
        raise HostVaultConfigurationError(
            "secrets.service.host_vault.address must be host:port; bracket IPv6 literals"
        )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise HostVaultConfigurationError(
            "secrets.service.host_vault.address must not contain a path, query, or fragment"
        )


def config_from_assembly(assembly: Mapping[str, object]) -> HostVaultRuntimeConfig:
    secrets = _mapping(assembly.get("secrets"))
    service = _mapping(secrets.get("service"))
    vault = _mapping(service.get("host_vault"))
    context = _mapping(assembly.get("context"))
    platform = _mapping(assembly.get("platform"))
    services = _mapping(platform.get("services"))
    proc = _mapping(services.get("proc"))
    exec_config = _mapping(proc.get("exec"))

    provider = _provider_name(secrets.get("provider"))
    backend = _backend_name(service.get("backend"))
    identity_text = _text(vault.get("identity_dir"))
    identity_dir = Path(identity_text).expanduser() if identity_text else None
    if identity_dir is not None and identity_dir.is_absolute():
        identity_dir = identity_dir.resolve()
    config = HostVaultRuntimeConfig(
        provider=provider,
        backend=backend,
        tenant=_text(context.get("tenant")),
        project=_text(context.get("project")),
        address=_text(vault.get("address")),
        server_name=_text(vault.get("server_name")),
        identity_dir=identity_dir,
        exec_network_mode=_text(exec_config.get("py_code_exec_network_mode")),
    )
    validate_configuration(config, check_identity=False)
    return config


def validate_configuration(
    config: HostVaultRuntimeConfig,
    *,
    check_identity: bool,
    workdir: Path | None = None,
) -> None:
    if config.backend not in {EPHEMERAL_BACKEND, HOST_VAULT_BACKEND}:
        raise HostVaultConfigurationError(
            "secrets.service.backend must be 'ephemeral' or 'host-vault'"
        )
    if not config.enabled:
        return
    if config.provider not in {"secrets-file", "secrets-service"}:
        raise HostVaultConfigurationError(
            "secrets.service.backend 'host-vault' requires secrets.provider "
            "'secrets-file' for shadow staging or 'secrets-service' for active use"
        )
    if not config.tenant or not config.project:
        raise HostVaultConfigurationError(
            "context.tenant and context.project are required for the host-vault backend"
        )
    if not config.address:
        raise HostVaultConfigurationError(
            "secrets.service.host_vault.address is required"
        )
    _validate_address(config.address)
    if not config.server_name:
        raise HostVaultConfigurationError(
            "secrets.service.host_vault.server_name is required"
        )
    if config.identity_dir is None or not config.identity_dir.is_absolute():
        raise HostVaultConfigurationError(
            "secrets.service.host_vault.identity_dir must be an absolute host path"
        )
    if config.exec_network_mode.lower() != "auto":
        raise HostVaultConfigurationError(
            "the local host-vault backend requires "
            "platform.services.proc.exec.py_code_exec_network_mode 'auto'"
        )
    if workdir is not None:
        runtime_root = Path(workdir).expanduser().resolve()
        if (
            config.identity_dir == runtime_root
            or runtime_root in config.identity_dir.parents
        ):
            raise HostVaultConfigurationError(
                "the host-vault deployment identity must be stored outside the KDCube workdir"
            )
    if not check_identity:
        return
    for path in config.identity_paths:
        if path.is_symlink() or not path.is_file():
            raise HostVaultConfigurationError(
                f"host-vault deployment identity is incomplete: {path.name} is missing"
            )
    key_path = config.identity_paths[1]
    if os.name == "posix" and key_path.stat().st_mode & 0o077:
        raise HostVaultConfigurationError(
            "host-vault-client.key must not be accessible by group or other users"
        )


def compose_environment(config: HostVaultRuntimeConfig) -> dict[str, str]:
    values = {
        "KDCUBE_SECRETS_SERVICE_BACKEND": config.backend,
        "KDCUBE_HOST_VAULT_ADDR": "",
        "KDCUBE_HOST_VAULT_SERVER_NAME": "",
        "KDCUBE_SECRETS_TENANT": "",
        "KDCUBE_SECRETS_PROJECT": "",
        "HOST_KDCUBE_HOST_VAULT_CLIENT_CERT_PATH": "",
        "HOST_KDCUBE_HOST_VAULT_CLIENT_KEY_PATH": "",
        "HOST_KDCUBE_HOST_VAULT_CA_PATH": "",
    }
    if not config.enabled:
        return values
    cert_path, key_path, ca_path = config.identity_paths
    values.update(
        {
            "KDCUBE_HOST_VAULT_ADDR": config.address,
            "KDCUBE_HOST_VAULT_SERVER_NAME": config.server_name,
            "KDCUBE_SECRETS_TENANT": config.tenant,
            "KDCUBE_SECRETS_PROJECT": config.project,
            "HOST_KDCUBE_HOST_VAULT_CLIENT_CERT_PATH": str(cert_path),
            "HOST_KDCUBE_HOST_VAULT_CLIENT_KEY_PATH": str(key_path),
            "HOST_KDCUBE_HOST_VAULT_CA_PATH": str(ca_path),
        }
    )
    return values


def validate_assembly_for_start(
    assembly: Mapping[str, object],
    *,
    workdir: Path,
) -> HostVaultRuntimeConfig:
    marker = Path(workdir).expanduser().resolve() / "config" / HOST_VAULT_ACTIVATION_MARKER
    if marker.exists() or marker.is_symlink():
        raise HostVaultConfigurationError(
            "an interrupted host-vault activation is pending; run "
            "`kdcube secrets host-vault recover --yes` before starting the runtime"
        )
    config = config_from_assembly(assembly)
    validate_configuration(config, check_identity=True, workdir=workdir)
    return config


__all__ = [
    "DEPLOYMENT_SECRET_APPLICATION",
    "EPHEMERAL_BACKEND",
    "HOST_VAULT_ACTIVATION_MARKER",
    "HOST_VAULT_BACKEND",
    "HostVaultConfigurationError",
    "HostVaultRuntimeConfig",
    "compose_environment",
    "config_from_assembly",
    "validate_assembly_for_start",
    "validate_configuration",
]
