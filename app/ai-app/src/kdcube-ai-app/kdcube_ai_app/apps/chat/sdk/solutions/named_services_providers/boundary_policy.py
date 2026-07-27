# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def as_mapping(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def clean_namespace(value: Any) -> str:
    return str(value or "").strip().lower().rstrip(":")


def _operation_policy(
    operation_policies: Mapping[str, Any],
    operation: str,
) -> tuple[bool, dict[str, Any]]:
    """Resolve an exact operation policy with an action-parent fallback.

    Named-service providers dispatch every bounded action through
    ``object.action``. A boundary may nevertheless publish narrower policy
    keys such as ``object.action.send``. Exact entries win. An older catalog
    containing only the parent ``object.action`` entry remains compatible, but
    that parent cannot open undeclared actions beside an exact catalog.
    """

    key = str(operation or "").strip()
    policies = dict(operation_policies or {})
    if key in policies:
        return True, as_mapping(policies.get(key))
    exact_action_catalog = any(
        str(candidate).startswith("object.action.") for candidate in policies
    )
    if (
        key.startswith("object.action.")
        and not exact_action_catalog
        and "object.action" in policies
    ):
        return True, as_mapping(policies.get("object.action"))
    return False, {}


@dataclass(frozen=True)
class NamespaceBoundaryPolicy:
    """Authority/grant boundary for one named-service namespace.

    The policy is intentionally transport-neutral. MCP, ReAct tools, APIs,
    jobs, widgets, and Data Bus handlers can all ask the same question:

    `for namespace N and tool T/operation O, which grants are required?`
    """

    namespace: str
    label: str = ""
    description: str = ""
    authority_id: str = ""
    provider_configs: tuple[Mapping[str, Any], ...] = ()
    tools: Mapping[str, Mapping[str, Any]] | None = None

    @classmethod
    def from_config(cls, namespace: str, value: Mapping[str, Any]) -> "NamespaceBoundaryPolicy":
        data = dict(value or {})
        providers = data.get("providers") or data.get("provider_configs") or ()
        if isinstance(providers, Mapping):
            providers = [providers]
        return cls(
            namespace=namespace,
            label=str(data.get("label") or namespace),
            description=str(data.get("description") or ""),
            authority_id=str(data.get("authority_id") or data.get("authority") or ""),
            provider_configs=tuple(
                dict(item) for item in (providers or ()) if isinstance(item, Mapping)
            ),
            tools=as_mapping(data.get("tools")),
        )

    def tool_configured(self, tool_name: str) -> bool:
        return str(tool_name or "").strip() in dict(self.tools or {})

    def operation_configured(self, *, tool_name: str, operation: str) -> bool:
        policy = as_mapping(dict(self.tools or {}).get(str(tool_name or "").strip()))
        operation_policies = as_mapping(policy.get("operations"))
        if not operation_policies:
            return True
        configured, _ = _operation_policy(operation_policies, operation)
        return configured

    def grants_for(self, *, tool_name: str, operation: str) -> tuple[str, ...]:
        policies = dict(self.tools or {})
        policy = as_mapping(policies.get(str(tool_name or "").strip()))
        operation_policies = as_mapping(policy.get("operations"))
        if operation_policies:
            configured, operation_policy = _operation_policy(
                operation_policies, operation
            )
            if configured:
                return tuple(as_list(operation_policy.get("grants") or operation_policy.get("scopes")))
        return tuple(as_list(policy.get("grants") or policy.get("scopes")))

    def label_for(self, *, tool_name: str, operation: str) -> str:
        policies = dict(self.tools or {})
        tool_key = str(tool_name or "").strip()
        policy = as_mapping(policies.get(tool_key))
        operation_policies = as_mapping(policy.get("operations"))
        if operation_policies:
            configured, operation_policy = _operation_policy(
                operation_policies, operation
            )
            if configured:
                text = str(operation_policy.get("label") or "").strip()
                if text:
                    return text
        return str(policy.get("label") or tool_key).strip()

    def description_for(self, *, tool_name: str, operation: str) -> str:
        policies = dict(self.tools or {})
        policy = as_mapping(policies.get(str(tool_name or "").strip()))
        operation_policies = as_mapping(policy.get("operations"))
        if operation_policies:
            configured, operation_policy = _operation_policy(
                operation_policies, operation
            )
            if configured:
                text = str(operation_policy.get("description") or "").strip()
                if text:
                    return text
        return str(policy.get("description") or "").strip()

    def authority_for(self, *, tool_name: str, operation: str) -> str:
        policies = dict(self.tools or {})
        policy = as_mapping(policies.get(str(tool_name or "").strip()))
        operation_policies = as_mapping(policy.get("operations"))
        if operation_policies:
            configured, operation_policy = _operation_policy(
                operation_policies, operation
            )
            if configured:
                text = str(operation_policy.get("authority_id") or operation_policy.get("authority") or "").strip()
                if text:
                    return text
        text = str(policy.get("authority_id") or policy.get("authority") or "").strip()
        return text or self.authority_id

    def to_public_dict(self) -> dict[str, Any]:
        tools = {}
        for name, policy in dict(self.tools or {}).items():
            data = as_mapping(policy)
            public = {
                "operation": str(data.get("operation") or ""),
                "label": str(data.get("label") or name),
                "description": str(data.get("description") or ""),
                "authority_id": str(data.get("authority_id") or data.get("authority") or self.authority_id),
                "grants": as_list(data.get("grants") or data.get("scopes")),
            }
            operation_policies = as_mapping(data.get("operations"))
            if operation_policies:
                public["operations"] = {
                    str(operation): {
                        "label": str(as_mapping(op_policy).get("label") or operation),
                        "description": str(as_mapping(op_policy).get("description") or ""),
                        "authority_id": str(
                            as_mapping(op_policy).get("authority_id")
                            or as_mapping(op_policy).get("authority")
                            or data.get("authority_id")
                            or data.get("authority")
                            or self.authority_id
                        ),
                        "grants": as_list(as_mapping(op_policy).get("grants") or as_mapping(op_policy).get("scopes"))
                    }
                    for operation, op_policy in operation_policies.items()
                }
            tools[str(name)] = public
        return {
            "namespace": self.namespace,
            "label": self.label,
            "description": self.description,
            "authority_id": self.authority_id,
            "tools": tools,
        }


class NamedServiceBoundaryCatalog:
    """Descriptor-backed namespace/tool boundary catalog."""

    def __init__(self, config: Mapping[str, Any]):
        self._config = dict(config or {})
        self._namespaces = self._load_namespaces()

    def _load_namespaces(self) -> dict[str, NamespaceBoundaryPolicy]:
        raw_namespaces = as_mapping(self._config.get("namespaces"))
        out: dict[str, NamespaceBoundaryPolicy] = {}
        for raw_namespace, raw_policy in raw_namespaces.items():
            namespace = clean_namespace(raw_namespace)
            if not namespace or not isinstance(raw_policy, Mapping):
                continue
            out[namespace] = NamespaceBoundaryPolicy.from_config(namespace, raw_policy)
        return out

    def list_public(self) -> list[dict[str, Any]]:
        return [policy.to_public_dict() for policy in self._namespaces.values()]

    def namespace_names(self) -> list[str]:
        return sorted(self._namespaces)

    def policy_for(self, namespace: str) -> NamespaceBoundaryPolicy | None:
        return self._namespaces.get(clean_namespace(namespace))


__all__ = [
    "NamedServiceBoundaryCatalog",
    "NamespaceBoundaryPolicy",
    "as_list",
    "as_mapping",
    "clean_namespace",
]
