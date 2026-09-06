from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from kdcube_ai_app.apps.chat.proc.rest.management.secret_contracts import (
    SECRET_OPERATIONS,
    SECRET_RESOURCE_SELECTOR,
)


def _deployment_root() -> Path:
    return Path(__file__).resolve().parents[9] / "deployment"


def _connection_hub_config() -> Mapping[str, Any]:
    document = yaml.safe_load(
        (_deployment_root() / "bundles.yaml").read_text(encoding="utf-8")
    )
    item = next(
        row
        for row in document["bundles"]["items"]
        if row["id"] == "connection-hub@1-0"
    )
    return item["config"]


def test_shipped_local_descriptor_exposes_secret_management_contract() -> None:
    delegated = _connection_hub_config()["connections"][
        "delegated_credentials"
    ]
    oauth = delegated["oauth"]
    admission = delegated["admission"]["services"]["kdcube-management"]

    capabilities = {row["grant"]: row for row in oauth["capabilities"]}
    assert set(SECRET_OPERATIONS).issubset(capabilities)

    secret_resource = next(
        row
        for row in oauth["resources"]
        if row["resource"] == SECRET_RESOURCE_SELECTOR
    )
    assert secret_resource["admin_only"] is True
    assert secret_resource["resource_selection"] is True
    assert set(secret_resource["operations"]) == set(SECRET_OPERATIONS)

    assert SECRET_RESOURCE_SELECTOR in admission["resources"]
    assert not set(SECRET_OPERATIONS).intersection(
        admission["request_bound_operations"]
    )


def test_shipped_local_descriptor_enables_bounded_human_export() -> None:
    assembly = yaml.safe_load(
        (_deployment_root() / "assembly.yaml").read_text(encoding="utf-8")
    )
    export = assembly["management"]["secret_export"]

    assert export == {
        "enabled": True,
        "required_assurance": "session_confirmation",
        "max_evidence_age_seconds": 300,
        "transaction_ttl_seconds": 180,
        "consumed_tombstone_seconds": 600,
        "max_targets": 4096,
        "max_total_value_bytes": 1048576,
    }

    human = assembly["management"]["human_approval"]
    assert human["fresh_authentication_provider"] == "auto"
    assert human["cognito"]["managed_login"] is False
    assert human["google"]["client_id"] == ""
    assert human["webauthn"] == {
        "enabled": True,
        "rp_id": "",
        "rp_name": "KDCube",
        "allowed_origins": [],
        "credential_policy": "verified_passkey",
        "trusted_attestation_root_files": {},
        "timeout_milliseconds": 60000,
        "max_credentials_per_user": 8,
    }
