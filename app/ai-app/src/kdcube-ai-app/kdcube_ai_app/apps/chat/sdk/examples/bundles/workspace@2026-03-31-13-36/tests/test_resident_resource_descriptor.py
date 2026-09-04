from __future__ import annotations

from pathlib import Path

import yaml

from kdcube_ai_app.apps.chat.sdk.runtime.agent_inventory import (
    agent_capabilities_catalog,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources import (
    delegated_mcp_bindings_from_catalog,
    resident_agent_ceiling_from_bundle_props,
)


MANAGED_KNOWLEDGE = (
    "*/api/integrations/bundles/*/*/knowledge@1-0/public/mcp/"
    "knowledge_managed*"
)


def _bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bundle_props() -> dict:
    path = _bundle_root() / "config" / "bundles.template.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return next(
        item["config"]
        for item in document["bundles"]["items"]
        if item.get("id") == "workspace@2026-03-31-13-36"
    )


def test_workspace_declares_one_ceiling_for_managed_and_user_owned_resources():
    props = _bundle_props()
    catalog = agent_capabilities_catalog(
        props,
        "main",
        bundle_root=_bundle_root(),
    )
    declared, _servers, _aliases = delegated_mcp_bindings_from_catalog(catalog)
    ceiling = resident_agent_ceiling_from_bundle_props(
        props,
        tenant="tenant-a",
        project="project-a",
        application="workspace@2026-03-31-13-36",
        agent_id="main",
        declared_resource_ids=declared,
    )

    knowledge = next(row for row in catalog["mcp"] if row["server_id"] == "knowledge")
    assert knowledge["authority_source"] == "delegated_card"
    assert knowledge["resource_id"] == MANAGED_KNOWLEDGE
    assert knowledge["claims"] == ["knowledge:read"]
    assert ceiling.declared_resource_ids == (MANAGED_KNOWLEDGE,)
    assert len(ceiling.resource_families) == 1
    family = ceiling.resource_families[0]
    assert family.resource_patterns == ("urn:connection-hub:remote-mcp:*",)
    assert family.endpoint_schemes == ()
    assert family.endpoint_hosts == ()
    assert family.max_resources == 8
    assert family.max_tools_per_resource == 64
