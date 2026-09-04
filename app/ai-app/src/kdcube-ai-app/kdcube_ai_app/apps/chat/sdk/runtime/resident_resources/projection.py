# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Wire projection helpers for an effective resident resource inventory."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.models import (
    EffectiveResidentInventory,
)


def effective_resource_catalog(
    inventory: EffectiveResidentInventory,
) -> list[dict[str, Any]]:
    """Return stable resource rows for capability views and selection clamps."""

    return [resource.to_dict() for resource in inventory.resources]


def resident_projection_base_catalog(
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep direct MCP rows in ``mcp``; Card-governed rows live in resources."""

    out = copy.deepcopy(dict(catalog))
    rows = out.get("mcp")
    if isinstance(rows, list):
        out["mcp"] = [
            row
            for row in rows
            if not (
                isinstance(row, Mapping)
                and str(row.get("authority_source") or "").strip()
                == "delegated_card"
            )
        ]
    return out


def attach_effective_resource_catalog(
    catalog: Mapping[str, Any],
    inventory: EffectiveResidentInventory,
) -> dict[str, Any]:
    """Attach one live row per resource without mutating the base catalog."""

    out = resident_projection_base_catalog(catalog)
    out["resources"] = effective_resource_catalog(inventory)
    out["resource_rejections"] = [entry.to_dict() for entry in inventory.rejected]
    return out


__all__ = [
    "attach_effective_resource_catalog",
    "effective_resource_catalog",
    "resident_projection_base_catalog",
]
