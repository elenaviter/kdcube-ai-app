# SPDX-License-Identifier: MIT

from __future__ import annotations

from kdcube_ai_app.infra.plugin.bundle_loader import (
    APIEndpointSpec,
    MCPEndpointSpec,
    OnMessageSpec,
    UIWidgetSpec,
    apply_api_overrides,
    apply_bundle_overrides,
    apply_mcp_overrides,
    apply_widget_overrides,
    bundle_default_chat,
    canonical_provider_surface_path,
    provider_surface_auth,
    BundleInterfaceManifest,
)


def test_provider_surface_policy_is_route_and_method_aware() -> None:
    props = {
        "surfaces": {
            "as_provider": {
                "api": {
                    "operations": {
                        "admin_data": {
                            "POST": {
                                "visibility": {
                                    "user_types": ["privileged"],
                                    "roles": ["kdcube:role:super-admin"],
                                },
                                "auth": {
                                    "authority_id": "platform",
                                    "grants": ["admin:data"],
                                },
                            },
                        },
                    },
                    "public": {
                        "admin_data": {
                            "GET": {
                                "visibility": {
                                    "user_types": [],
                                    "roles": [],
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    operations = APIEndpointSpec(
        method_name="admin_data",
        alias="admin_data",
        http_method="POST",
        route="operations",
        user_types=(),
        roles=(),
    )
    public = APIEndpointSpec(
        method_name="admin_data_public",
        alias="admin_data",
        http_method="GET",
        route="public",
        user_types=("registered",),
        roles=("legacy",),
    )

    effective_operations = apply_api_overrides(operations, props)
    effective_public = apply_api_overrides(public, props)

    assert effective_operations.user_types == ("privileged",)
    assert effective_operations.roles == ("kdcube:role:super-admin",)
    assert effective_public.user_types == ()
    assert effective_public.roles == ()
    assert provider_surface_auth(
        props,
        "api",
        alias="admin_data",
        http_method="POST",
        route="operations",
    ) == {"authority_id": "platform", "grants": ["admin:data"]}
    assert canonical_provider_surface_path(
        "api",
        alias="admin_data",
        http_method="POST",
        route="operations",
    ) == "surfaces.as_provider.api.operations.admin_data.POST"


def test_provider_surface_policy_allows_api_alias_level_defaults() -> None:
    props = {
        "surfaces": {
            "as_provider": {
                "api": {
                    "operations": {
                        "status": {
                            "visibility": {
                                "user_types": [],
                                "roles": ["kdcube:role:super-admin"],
                            },
                            "auth": {
                                "authority_id": "platform",
                                "grants": ["status:read"],
                            },
                        },
                    },
                },
            },
        },
    }
    spec = APIEndpointSpec(
        method_name="status",
        alias="status",
        http_method="GET",
        route="operations",
        user_types=("registered",),
        roles=(),
    )

    effective = apply_api_overrides(spec, props)

    assert effective.user_types == ()
    assert effective.roles == ("kdcube:role:super-admin",)
    assert provider_surface_auth(
        props,
        "api",
        alias="status",
        http_method="GET",
        route="operations",
    ) == {"authority_id": "platform", "grants": ["status:read"]}


def test_provider_surface_policy_overrides_operation_csrf_in_both_directions() -> None:
    enabled_props = {
        "surfaces": {
            "as_provider": {
                "api": {
                    "operations": {
                        "save": {"POST": {"csrf": True}},
                    },
                },
            },
        },
    }
    disabled_props = {
        "surfaces": {
            "as_provider": {
                "api": {
                    "operations": {
                        "save": {"csrf": False},
                    },
                },
            },
        },
    }

    default_off = APIEndpointSpec(
        method_name="save",
        alias="save",
        http_method="POST",
        route="operations",
    )
    default_on = APIEndpointSpec(
        method_name="save",
        alias="save",
        http_method="POST",
        route="operations",
        csrf=True,
    )

    assert apply_api_overrides(default_off, enabled_props).csrf is True
    assert apply_api_overrides(default_on, disabled_props).csrf is False


def test_provider_surface_method_policy_overrides_alias_defaults_per_field() -> None:
    props = {
        "surfaces": {
            "as_provider": {
                "api": {
                    "operations": {
                        "save": {
                            "csrf": True,
                            "auth": {
                                "authority_id": "platform",
                                "grants": ["settings:write"],
                            },
                            "POST": {
                                "csrf": False,
                                "visibility": {"roles": ["kdcube:role:editor"]},
                            },
                        },
                    },
                },
            },
        },
    }
    spec = APIEndpointSpec(
        method_name="save",
        alias="save",
        http_method="POST",
        route="operations",
    )

    resolved = apply_api_overrides(spec, props)

    assert resolved.csrf is False
    assert resolved.roles == ("kdcube:role:editor",)
    assert provider_surface_auth(
        props,
        "api",
        alias="save",
        http_method="POST",
        route="operations",
    ) == {
        "authority_id": "platform",
        "grants": ["settings:write"],
    }


def test_provider_surface_policy_controls_bundle_widget_and_mcp() -> None:
    props = {
        "surfaces": {
            "as_provider": {
                "bundle": {"visibility": {"allowed_roles": ["kdcube:role:editor"]}},
                "widget": {
                    "settings": {
                        "visibility": {
                            "user_types": ["privileged"],
                            "roles": ["kdcube:role:super-admin"],
                        },
                    },
                },
                "mcp": {
                    "knowledge": {
                        "auth": {
                            "mode": "managed",
                            "authority_id": "custom.identity",
                            "grants": ["knowledge:read"],
                        },
                    },
                },
            },
        },
    }

    manifest = BundleInterfaceManifest(bundle_id="example@1-0")
    widget = UIWidgetSpec(method_name="settings", alias="settings", icon={})
    mcp = MCPEndpointSpec(
        method_name="knowledge",
        alias="knowledge",
        auth={"mode": "header", "header": "X-Knowledge-MCP-Token"},
    )

    assert apply_bundle_overrides(manifest, props).allowed_roles == ("kdcube:role:editor",)
    assert apply_widget_overrides(widget, props).user_types == ("privileged",)
    assert apply_widget_overrides(widget, props).roles == ("kdcube:role:super-admin",)
    assert apply_mcp_overrides(mcp, props).auth == {
        "mode": "managed",
        "authority_id": "custom.identity",
        "grants": ["knowledge:read"],
    }


def test_default_chat_is_descriptor_declared_not_code_inherited() -> None:
    # Every SDK-based entrypoint inherits a reactive handler, so on_message
    # alone must never turn the chat surface on.
    with_handler = BundleInterfaceManifest(
        bundle_id="example@1-0", on_message=OnMessageSpec(method_name="run")
    )
    declared = {"surfaces": {"as_provider": {"bundle": {"default_chat": True}}}}

    assert with_handler.default_chat is False
    assert apply_bundle_overrides(with_handler, {}).default_chat is False
    assert apply_bundle_overrides(with_handler, declared).default_chat is True

    # Declaring the chat surface without a reactive handler serves nothing.
    without_handler = BundleInterfaceManifest(bundle_id="example@1-0")
    assert apply_bundle_overrides(without_handler, declared).default_chat is False

    assert bundle_default_chat(declared) is True
    assert bundle_default_chat({}) is False
    assert bundle_default_chat(None) is False
