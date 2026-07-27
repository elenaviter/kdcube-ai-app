# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import os
from typing import Any, Literal

from kdcube_ai_app.apps.chat.sdk.config import get_settings

StaticWidgetDeliveryMode = Literal["legacy", "shadow", "deployed"]

LEGACY_MODE: StaticWidgetDeliveryMode = "legacy"
SHADOW_MODE: StaticWidgetDeliveryMode = "shadow"
DEPLOYED_MODE: StaticWidgetDeliveryMode = "deployed"
VALID_MODES = frozenset({LEGACY_MODE, SHADOW_MODE, DEPLOYED_MODE})


def static_widget_runtime_generation() -> str:
    """Return the immutable runtime release identity when the launcher provides one."""
    return str(
        os.getenv("PLATFORM_REF")
        or os.getenv("APP_IMAGE_TAG")
        or os.getenv("IMAGE_TAG")
        or "unversioned"
    ).strip()


def normalize_static_widget_delivery_mode(value: Any) -> StaticWidgetDeliveryMode:
    mode = str(value or LEGACY_MODE).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(
            "platform.services.proc.bundles.static_widget_delivery_mode must be "
            "legacy, shadow, or deployed"
        )
    return mode  # type: ignore[return-value]


def static_widget_delivery_mode(settings: Any | None = None) -> StaticWidgetDeliveryMode:
    resolved = settings or get_settings()
    applications = getattr(getattr(resolved, "PLATFORM", None), "APPLICATIONS", None)
    raw = getattr(applications, "STATIC_WIDGET_DELIVERY_MODE", LEGACY_MODE)
    return normalize_static_widget_delivery_mode(raw)


def static_widget_deployment_enabled(settings: Any | None = None) -> bool:
    return static_widget_delivery_mode(settings) in {SHADOW_MODE, DEPLOYED_MODE}


def deployed_static_widget_serving_enabled(settings: Any | None = None) -> bool:
    return static_widget_delivery_mode(settings) == DEPLOYED_MODE
