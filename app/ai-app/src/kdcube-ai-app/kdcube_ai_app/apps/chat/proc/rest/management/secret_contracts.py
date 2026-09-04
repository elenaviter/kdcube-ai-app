# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Exact secret targets exposed through delegated KDCube management."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from kdcube_ai_app.apps.chat.proc.rest.management.contracts import (
    validate_application_id,
)

SECRET_METADATA_OPERATION = "kdcube.management.secret.metadata.read"
SECRET_READ_OPERATION = "kdcube.management.secret.value.read"
SECRET_WRITE_OPERATION = "kdcube.management.secret.value.write"
SECRET_DELETE_OPERATION = "kdcube.management.secret.delete"
SECRET_OPERATIONS = (
    SECRET_METADATA_OPERATION,
    SECRET_READ_OPERATION,
    SECRET_WRITE_OPERATION,
    SECRET_DELETE_OPERATION,
)
SECRET_RESOURCE_SELECTOR = "urn:kdcube:management:secret:*:*:*:*:*"

PLATFORM_SCOPE = "platform"
BUNDLE_SCOPE = "bundle"
SECRET_SCOPES = frozenset({PLATFORM_SCOPE, BUNDLE_SCOPE})
MAX_SECRET_VALUE_BYTES = 64 * 1024

_SECRET_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]{0,511}$")


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be text")  # noqa: TRY004 - protocol validation
    return value.strip()


def validate_secret_key(value: Any, *, scope: str) -> str:
    key = _text(value)
    if (
        not _SECRET_KEY_RE.fullmatch(key)
        or ".." in key
        or key.endswith((".", ".__keys"))
        or key == "__keys"
    ):
        raise ValueError("key must be one exact non-metadata dotted secret key")
    if scope == PLATFORM_SCOPE and key.startswith(("bundles.", "users.")):
        raise ValueError("platform secret key cannot enter another secret scope")
    return key


def validate_secret_value(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")  # noqa: TRY004 - protocol validation
    if value == "":
        raise ValueError("value must not be empty; delete the secret instead")
    if len(value.encode("utf-8")) > MAX_SECRET_VALUE_BYTES:
        raise ValueError(
            f"value must not exceed {MAX_SECRET_VALUE_BYTES} UTF-8 bytes"
        )
    return value


@dataclass(frozen=True)
class SecretTarget:
    """One exact provider key inside a deployment-owned secret scope."""

    scope: str
    key: str
    bundle_id: str = ""

    def __post_init__(self) -> None:
        scope = _text(self.scope).lower()
        if scope not in SECRET_SCOPES:
            raise ValueError("scope must be platform or bundle")
        bundle_id = _text(self.bundle_id)
        if scope == BUNDLE_SCOPE:
            bundle_id = validate_application_id(bundle_id)
        elif bundle_id:
            raise ValueError("bundle_id is valid only for bundle secrets")
        key = validate_secret_key(self.key, scope=scope)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "bundle_id", bundle_id)
        object.__setattr__(self, "key", key)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SecretTarget:
        return cls(
            scope=value.get("scope", ""),
            bundle_id=value.get("bundle_id", ""),
            key=value.get("key", ""),
        )

    @property
    def provider_key(self) -> str:
        if self.scope == BUNDLE_SCOPE:
            return f"bundles.{self.bundle_id}.secrets.{self.key}"
        return self.key

    @property
    def scope_id(self) -> str:
        return self.bundle_id if self.scope == BUNDLE_SCOPE else "_"

    def resource(self, *, tenant: str, project: str) -> str:
        segments = (
            tenant,
            project,
            self.scope,
            self.scope_id,
            self.key,
        )
        encoded = [quote(_text(item), safe="-._~@") for item in segments]
        if any(not item for item in encoded):
            raise ValueError("secret resource identity is incomplete")
        return "urn:kdcube:management:secret:" + ":".join(encoded)

    def approval_context(self) -> dict[str, str]:
        context = {"secret_scope": self.scope, "secret_key": self.key}
        if self.bundle_id:
            context["bundle_id"] = self.bundle_id
        return context

    def public_dict(self) -> dict[str, str]:
        result = {"scope": self.scope, "key": self.key}
        if self.bundle_id:
            result["bundle_id"] = self.bundle_id
        return result


__all__ = [
    "BUNDLE_SCOPE",
    "MAX_SECRET_VALUE_BYTES",
    "PLATFORM_SCOPE",
    "SECRET_DELETE_OPERATION",
    "SECRET_METADATA_OPERATION",
    "SECRET_OPERATIONS",
    "SECRET_READ_OPERATION",
    "SECRET_RESOURCE_SELECTOR",
    "SECRET_SCOPES",
    "SECRET_WRITE_OPERATION",
    "SecretTarget",
    "validate_secret_key",
    "validate_secret_value",
]
