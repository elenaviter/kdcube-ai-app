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
USER_SCOPE = "user"
SECRET_SCOPES = frozenset({PLATFORM_SCOPE, BUNDLE_SCOPE, USER_SCOPE})
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
    if scope == PLATFORM_SCOPE and not key.startswith("platform."):
        raise ValueError("platform secret key must begin with platform.")
    if scope in {BUNDLE_SCOPE, USER_SCOPE} and key.startswith(
        ("platform.", "bundles.", "users.")
    ):
        raise ValueError(f"{scope} secret key must be relative to its scope")
    return key


def _scope_id(value: Any, *, name: str) -> str:
    candidate = _text(value)
    if (
        not _SECRET_KEY_RE.fullmatch(candidate)
        or "." in candidate
        or candidate in {"_", "__keys"}
    ):
        raise ValueError(f"{name} must be one exact identifier")
    return candidate


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
    user_id: str = ""

    def __post_init__(self) -> None:
        scope = _text(self.scope).lower()
        if scope not in SECRET_SCOPES:
            raise ValueError("scope must be platform, bundle, or user")
        bundle_id = _text(self.bundle_id)
        user_id = _text(self.user_id)
        if scope == BUNDLE_SCOPE:
            bundle_id = validate_application_id(bundle_id)
            if user_id:
                raise ValueError("user_id is valid only for user secrets")
        elif scope == USER_SCOPE:
            user_id = _scope_id(user_id, name="user_id")
            if bundle_id:
                bundle_id = validate_application_id(bundle_id)
        elif bundle_id or user_id:
            raise ValueError(
                "bundle_id and user_id are valid only for their secret scopes"
            )
        key = validate_secret_key(self.key, scope=scope)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "bundle_id", bundle_id)
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "key", key)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SecretTarget:
        return cls(
            scope=value.get("scope", ""),
            bundle_id=value.get("bundle_id", ""),
            user_id=value.get("user_id", ""),
            key=value.get("key", ""),
        )

    @classmethod
    def from_provider_key(cls, value: str) -> SecretTarget:
        key = _text(value)
        if key.startswith("platform."):
            return cls(scope=PLATFORM_SCOPE, key=key)
        if key.startswith("bundles.") and ".secrets." in key:
            identity, tail = key[len("bundles.") :].split(".secrets.", 1)
            return cls(scope=BUNDLE_SCOPE, bundle_id=identity, key=tail)
        if key.startswith("users."):
            user_path = key[len("users.") :]
            bundle_marker = user_path.find(".bundles.")
            secrets_marker = user_path.find(".secrets.")
            if bundle_marker >= 0 and (
                secrets_marker < 0 or bundle_marker < secrets_marker
            ):
                user_id, bundle_path = user_path.split(".bundles.", 1)
                if ".secrets." not in bundle_path:
                    raise ValueError("user bundle secret provider key is invalid")
                bundle_id, tail = bundle_path.split(".secrets.", 1)
                return cls(
                    scope=USER_SCOPE,
                    user_id=user_id,
                    bundle_id=bundle_id,
                    key=tail,
                )
            if ".secrets." in user_path:
                user_id, tail = user_path.split(".secrets.", 1)
                return cls(scope=USER_SCOPE, user_id=user_id, key=tail)
        raise ValueError("secret provider key is not canonical")

    @property
    def provider_key(self) -> str:
        if self.scope == BUNDLE_SCOPE:
            return f"bundles.{self.bundle_id}.secrets.{self.key}"
        if self.scope == USER_SCOPE:
            if self.bundle_id:
                return (
                    f"users.{self.user_id}.bundles.{self.bundle_id}.secrets."
                    f"{self.key}"
                )
            return f"users.{self.user_id}.secrets.{self.key}"
        return self.key

    @property
    def scope_id(self) -> str:
        if self.scope == BUNDLE_SCOPE:
            return self.bundle_id
        if self.scope == USER_SCOPE:
            suffix = f"~{self.bundle_id}" if self.bundle_id else ""
            return f"{self.user_id}{suffix}"
        return "_"

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
        if self.user_id:
            context["user_id"] = self.user_id
        return context

    def public_dict(self) -> dict[str, str]:
        result = {"scope": self.scope, "key": self.key}
        if self.bundle_id:
            result["bundle_id"] = self.bundle_id
        if self.user_id:
            result["user_id"] = self.user_id
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
    "USER_SCOPE",
    "validate_secret_key",
    "validate_secret_value",
]
