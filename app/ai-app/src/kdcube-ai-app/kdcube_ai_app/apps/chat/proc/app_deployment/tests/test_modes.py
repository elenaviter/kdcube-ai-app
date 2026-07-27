# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kdcube_ai_app.apps.chat.proc.app_deployment.modes import (
    deployed_static_widget_serving_enabled,
    normalize_static_widget_delivery_mode,
    static_widget_deployment_enabled,
    static_widget_delivery_mode,
)
from kdcube_ai_app.apps.chat.sdk.config_scopes import ApplicationsConfig


def _settings(mode: str | None = None):
    applications = SimpleNamespace()
    if mode is not None:
        applications.STATIC_WIDGET_DELIVERY_MODE = mode
    return SimpleNamespace(PLATFORM=SimpleNamespace(APPLICATIONS=applications))


def test_missing_mode_preserves_legacy_serving() -> None:
    assert static_widget_delivery_mode(_settings()) == "legacy"
    assert static_widget_deployment_enabled(_settings()) is False


def test_shadow_deploys_without_enabling_fast_serving() -> None:
    settings = _settings("shadow")
    assert static_widget_deployment_enabled(settings) is True
    assert deployed_static_widget_serving_enabled(settings) is False


def test_deployed_enables_deployment_and_fast_serving() -> None:
    settings = _settings("deployed")
    assert static_widget_deployment_enabled(settings) is True
    assert deployed_static_widget_serving_enabled(settings) is True


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy, shadow, or deployed"):
        normalize_static_widget_delivery_mode("cdn")


def test_applications_config_defaults_to_legacy_and_rejects_unknown_mode() -> None:
    assert ApplicationsConfig().STATIC_WIDGET_DELIVERY_MODE == "legacy"
    with pytest.raises(ValidationError):
        ApplicationsConfig(STATIC_WIDGET_DELIVERY_MODE="cdn")
