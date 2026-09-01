from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any

import pytest
import yaml


def _bundles_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundle_config_from_path(path: Path, bundle_id: str) -> dict[str, Any]:
    descriptor = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bundle = next(
        item
        for item in descriptor["bundles"]["items"]
        if item.get("id") == bundle_id
    )
    return bundle["config"]


def _bundle_config(bundle_dir: str, bundle_id: str) -> dict[str, Any]:
    path = _bundles_root() / bundle_dir / "config" / "bundles.template.yaml"
    return _bundle_config_from_path(path, bundle_id)


def _connection_hub_config() -> dict[str, Any]:
    ecosystem_repo = str(os.environ.get("APP_ECOSYSTEM_REPO") or "").strip()
    if not ecosystem_repo:
        pytest.skip(
            "APP_ECOSYSTEM_REPO is not set; Connection Hub app contract is "
            "verified from its companion checkout"
        )
    path = (
        Path(ecosystem_repo).expanduser().resolve()
        / "products"
        / "connection-hub"
        / "apps"
        / "connection-hub@1-0"
        / "config"
        / "bundles.template.yaml"
    )
    if not path.is_file():
        pytest.fail(f"Connection Hub app descriptor does not exist: {path}")
    return _bundle_config_from_path(path, "connection-hub@1-0")


def _connected_account_requirements(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        requirements = node.get("connected_accounts")
        if isinstance(requirements, list):
            yield from (item for item in requirements if isinstance(item, dict))
        for value in node.values():
            yield from _connected_account_requirements(value)
    elif isinstance(node, list):
        for value in node:
            yield from _connected_account_requirements(value)


def test_connected_account_requirements_reference_declared_connector_apps():
    workspace = _bundle_config(
        "workspace@2026-03-31-13-36",
        "workspace@2026-03-31-13-36",
    )
    connection_hub = _connection_hub_config()
    providers = connection_hub["connections"]["delegated_to_kdcube"]["providers"]
    requirements = list(_connected_account_requirements(workspace))

    assert requirements
    for requirement in requirements:
        provider_id = requirement["provider_id"]
        connector_app_id = requirement.get("connector_app_id")
        provider = providers[provider_id]

        if not connector_app_id:
            continue

        connector_app = provider["connector_apps"][connector_app_id]
        provider_claims = provider.get("claims") or {}
        allowed_claims = set(connector_app.get("allowed_claims") or [])
        for claim in requirement.get("claims") or []:
            assert claim in provider_claims
            assert claim in allowed_claims


def test_connection_hub_template_declares_google_productivity_contract():
    connection_hub = _connection_hub_config()
    connections = connection_hub["connections"]
    google = connections["delegated_to_kdcube"]["providers"]["google"]
    connector = google["connector_apps"]["gmail"]

    expected_claims = {
        "sheets:read",
        "sheets:write",
        "docs:read",
        "docs:write",
        "docs:comment",
    }
    assert expected_claims <= set(connector["allowed_claims"])
    assert expected_claims <= set(google["claims"])
    assert "https://www.googleapis.com/auth/drive.file" in google["claims"][
        "sheets:write"
    ]["provider_scopes"]

    oauth = connections["delegated_credentials"]["oauth"]
    resource = next(
        item
        for item in oauth["resources"]
        if item.get("label") == "KDCube named services MCP"
    )
    assert "docs" in resource["named_services"]["namespaces"]
    assert {
        "named_services_upsert",
        "named_services_host_file",
        "named_services_delete",
    } <= set(resource["tools"])
