"""Non-secret inspection of a local deployment's selected secret backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kdcube_cli.control.local_runtime import local_target_identity
from kdcube_cli.host_vault import HOST_VAULT_ACTIVATION_MARKER
from kdcube_cli.management.errors import ManagementCliError

_KNOWN_PROVIDERS = frozenset({"secrets-file", "secrets-service", "aws-sm", "in-memory"})
_KNOWN_SERVICE_BACKENDS = frozenset({"ephemeral", "host-vault"})


def backend_status(
    *,
    workdir: Path,
    tenant: str = "",
    project: str = "",
) -> dict[str, Any]:
    resolved = workdir.expanduser().resolve()
    assembly_path = resolved / "config" / "assembly.yaml"
    try:
        document = yaml.safe_load(assembly_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ManagementCliError(
            "secrets_backend_descriptor_invalid",
            "The selected runtime has no readable assembly secret configuration.",
        ) from exc
    if not isinstance(document, dict):
        raise ManagementCliError(
            "secrets_backend_descriptor_invalid",
            "The selected runtime has no readable assembly secret configuration.",
        )
    secrets = document.get("secrets")
    if not isinstance(secrets, dict):
        secrets = {}
    provider = str(secrets.get("provider") or "in-memory").strip().lower()
    service = secrets.get("service")
    if not isinstance(service, dict):
        service = {}
    service_backend = str(service.get("backend") or "ephemeral").strip().lower()
    if provider not in _KNOWN_PROVIDERS or service_backend not in (
        _KNOWN_SERVICE_BACKENDS
    ):
        raise ManagementCliError(
            "secrets_backend_descriptor_invalid",
            "The selected runtime declares an unsupported secret backend.",
        )

    resolved_tenant, resolved_project = local_target_identity(resolved)
    exact_tenant = str(resolved_tenant or "").strip()
    exact_project = str(resolved_project or "").strip()
    if not exact_tenant or not exact_project:
        raise ManagementCliError(
            "management_local_target_invalid",
            "The selected local KDCube runtime has no deployment coordinates.",
        )
    if (tenant and tenant != exact_tenant) or (project and project != exact_project):
        raise ManagementCliError(
            "management_coordinate_mismatch",
            "The selected workdir belongs to different KDCube coordinates.",
        )

    authoritative = {
        "secrets-file": "descriptor-files",
        "secrets-service": (
            "host-vault" if service_backend == "host-vault" else "ephemeral-service"
        ),
        "aws-sm": "aws-secrets-manager",
        "in-memory": "process-memory",
    }[provider]
    if provider == "secrets-service" and service_backend == "host-vault":
        migration_state = "active"
    elif provider == "secrets-file" and service_backend == "host-vault":
        migration_state = "shadow-configured"
    else:
        migration_state = "not-applicable"
    host_vault = service.get("host_vault")
    if not isinstance(host_vault, dict):
        host_vault = {}

    return {
        "schema": "kdcube_cli.secrets_backend_status.v1",
        "target": {
            "tenant": exact_tenant,
            "project": exact_project,
            "workdir": str(resolved),
        },
        "configuration": {
            "provider": provider,
            "service_backend": service_backend,
            "source": str(assembly_path),
        },
        "authoritative_store": {
            "kind": authoritative,
            "evidence": "staged-descriptor",
            "runtime_verified": False,
        },
        "descriptor_files": {
            "platform": {
                "path": str(resolved / "config" / "secrets.yaml"),
                "exists": (resolved / "config" / "secrets.yaml").is_file(),
            },
            "bundles": {
                "path": str(resolved / "config" / "bundles.secrets.yaml"),
                "exists": (resolved / "config" / "bundles.secrets.yaml").is_file(),
            },
        },
        "host_vault": {
            "state": migration_state,
            "address_configured": bool(str(host_vault.get("address") or "").strip()),
            "server_name_configured": bool(
                str(host_vault.get("server_name") or "").strip()
            ),
            "identity_configured": bool(
                str(host_vault.get("identity_dir") or "").strip()
            ),
            "recovery_pending": (
                resolved / "config" / HOST_VAULT_ACTIVATION_MARKER
            ).exists(),
        },
        "descriptor_values_authoritative": provider == "secrets-file",
    }


__all__ = ["backend_status"]
