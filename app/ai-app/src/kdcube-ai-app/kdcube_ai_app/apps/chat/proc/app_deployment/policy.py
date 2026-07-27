# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from typing import Any, Mapping

from kdcube_ai_app.apps.chat.sdk.solutions.chat import (
    DEFAULT_CHAT_WIDGET_ALIAS,
    default_chat_widget_config,
)
from kdcube_ai_app.infra.plugin.bundle_loader import UIWidgetSpec, bundle_default_chat

_DISABLED_PROP_VALUES: frozenset[str] = frozenset({"false", "disable", "disabled", "off", "0"})


def is_truthy_enabled(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).strip().lower() not in _DISABLED_PROP_VALUES


def enabled_section(props: Mapping[str, Any] | None, kind: str) -> Mapping[str, Any] | None:
    section = (props or {}).get("enabled")
    if not isinstance(section, Mapping):
        return None
    sub = section.get(kind)
    return sub if isinstance(sub, Mapping) else None


def is_bundle_enabled(props: Mapping[str, Any] | None) -> bool:
    section = (props or {}).get("enabled")
    if not isinstance(section, Mapping):
        return True
    return is_truthy_enabled(section.get("bundle"))


def is_widget_enabled(props: Mapping[str, Any] | None, spec: UIWidgetSpec) -> bool:
    sub = enabled_section(props, "widget")
    if spec.alias == DEFAULT_CHAT_WIDGET_ALIAS and (sub is None or spec.alias not in sub):
        return bundle_default_chat(props)
    if sub is None:
        return True
    return is_truthy_enabled(sub.get(spec.alias))


def raw_static_widget_config(
    props: Mapping[str, Any] | None,
    *,
    widget_alias: str,
) -> dict[str, Any] | None:
    ui_cfg = props.get("ui") if isinstance(props, Mapping) else None
    raw_widgets = ui_cfg.get("widgets") if isinstance(ui_cfg, Mapping) else None
    cfg = raw_widgets.get(widget_alias) if isinstance(raw_widgets, Mapping) else None
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if widget_alias == DEFAULT_CHAT_WIDGET_ALIAS and bundle_default_chat(props):
        return default_chat_widget_config()
    return None


def static_widget_explicitly_disabled(
    props: Mapping[str, Any] | None,
    *,
    widget_alias: str,
) -> bool:
    cfg = raw_static_widget_config(props, widget_alias=widget_alias)
    return isinstance(cfg, dict) and "enabled" in cfg and not is_truthy_enabled(cfg.get("enabled"))


def static_widget_config(
    props: Mapping[str, Any] | None,
    *,
    widget_alias: str,
) -> dict[str, Any] | None:
    cfg = raw_static_widget_config(props, widget_alias=widget_alias)
    if not isinstance(cfg, dict):
        return None
    if "enabled" in cfg and not is_truthy_enabled(cfg.get("enabled")):
        return None
    has_source = bool(str(cfg.get("src_folder") or cfg.get("source_dir") or "").strip())
    has_build = bool(str(cfg.get("build_command") or "").strip())
    return cfg if has_source and has_build else None
