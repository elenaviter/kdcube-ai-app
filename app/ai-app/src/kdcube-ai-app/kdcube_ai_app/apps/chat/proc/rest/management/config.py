# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Descriptor-owned configuration for delegated KDCube management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "on", "enabled"}


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


@dataclass(frozen=True)
class DelegatedManagementConfig:
    enabled: bool
    tenant: str
    project: str
    connection_hub_app_id: str
    service_id: str
    service_secret_ref: str
    admission_url: str
    admission_timeout_seconds: float = 10.0
    effect_pending_seconds: float = 120.0

    @classmethod
    def from_settings(cls, settings: Any) -> "DelegatedManagementConfig":
        plain = settings.plain
        return cls(
            enabled=_bool(plain("management.delegated.enabled", default=False)),
            tenant=_text(getattr(settings, "TENANT", "")),
            project=_text(getattr(settings, "PROJECT", "")),
            connection_hub_app_id=_text(
                plain(
                    "management.delegated.connection_hub.app_id",
                    default="connection-hub@1-0",
                )
            ),
            service_id=_text(
                plain(
                    "management.delegated.connection_hub.service_id",
                    default="kdcube-management",
                )
            ),
            service_secret_ref=_text(
                plain(
                    "management.delegated.connection_hub.service_secret_ref",
                    default=(
                        "connections.delegated_credentials.admission.services."
                        "kdcube-management.signing_secret"
                    ),
                )
            ),
            admission_url=_text(
                plain(
                    "management.delegated.connection_hub.admission_url",
                    default="",
                )
            ),
            admission_timeout_seconds=_positive_float(
                plain(
                    "management.delegated.connection_hub.timeout_seconds",
                    default=10,
                ),
                10.0,
            ),
            effect_pending_seconds=_positive_float(
                plain("management.delegated.effect_pending_seconds", default=120),
                120.0,
            ),
        )

    def resolved_admission_url(self, settings: Any) -> str:
        if self.admission_url:
            return self.admission_url
        port = int(getattr(settings, "CHAT_PROCESSOR_PORT", None) or 8020)
        parts = (
            quote(self.tenant, safe="-._~"),
            quote(self.project, safe="-._~"),
            quote(self.connection_hub_app_id, safe="-._~@"),
        )
        return (
            f"http://127.0.0.1:{port}/api/integrations/bundles/"
            f"{parts[0]}/{parts[1]}/{parts[2]}/public/delegated_admission"
        )

    def validate(self) -> None:
        if not self.tenant or not self.project:
            raise ValueError("configured tenant/project are required")
        if not self.connection_hub_app_id:
            raise ValueError("Connection Hub application id is required")
        if not self.service_id or not self.service_secret_ref:
            raise ValueError("Connection Hub protected-service identity is required")


__all__ = ["DelegatedManagementConfig"]
