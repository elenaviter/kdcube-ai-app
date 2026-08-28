# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

import copy

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.catalog.authorization import (
    CAPABILITY_NAMED_SERVICE_NAMESPACE,
    CAPABILITY_NAMED_SERVICE_OPERATION,
    CAPABILITY_OUTER_OPERATION,
    CAPABILITY_RESOURCE,
    CAPABILITY_RESOURCE_CLAIM,
    DENIAL_CODE,
    DENIAL_REASON,
    ActiveCatalogCapabilities,
    CapabilityRequest,
    CardProvenance,
    authorize_current_capability,
    catalog_unavailable_denial,
)
from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.catalog.models import (
    CatalogDocument,
)

SELECTOR = "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/named_services*"
REQUEST_URL = (
    "https://kdcube.example/api/integrations/bundles/acme/prod/"
    "kdcube-services@1-0/public/mcp/named_services"
)

CONNECTIONS = {
    "delegated_credentials": {
        "oauth": {
            "enabled": True,
            "resources": [
                {
                    "resource": SELECTOR,
                    "label": "KDCube named services MCP",
                    "tools": {
                        "named_services_schema": {"grants": ["named_services:use"]},
                        "named_services_search": {"grants": ["named_services:use"]},
                    },
                    "named_services": {
                        "namespaces": {
                            "mail": {
                                "tools": {
                                    "schema": {
                                        "operation": "object.schema",
                                        "grants": ["named_services:use"],
                                    },
                                    "search": {
                                        "operation": "object.search",
                                        "grants": ["named_services:use", "mail:read"],
                                    },
                                },
                            },
                            "linkedin": {
                                "tools": {
                                    "call": {
                                        "operations": {
                                            "object.action.publish": {"grants": ["linkedin:write"]},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            ],
        },
    },
}

PROVENANCE = CardProvenance(
    access_id="oauth-5aa44826664a0bdd",
    card_revision=8,
    catalog_version="delegated_catalog_2026-08-09-09-00-00-000_a1b2c3d4e5f6",
)


def _catalog(connections=None) -> ActiveCatalogCapabilities:
    return ActiveCatalogCapabilities(
        CatalogDocument.build(connections if connections is not None else CONNECTIONS)
    )


def _authorize(request, *, connections=None):
    return authorize_current_capability(
        catalog=_catalog(connections), provenance=PROVENANCE, request=request
    )


def _without_namespace(name: str) -> dict:
    trimmed = copy.deepcopy(CONNECTIONS)
    resource = trimmed["delegated_credentials"]["oauth"]["resources"][0]
    resource["named_services"]["namespaces"].pop(name)
    return trimmed


# -- what the active catalog still offers --------------------------------------


@pytest.mark.parametrize(
    "request_",
    [
        CapabilityRequest(kind=CAPABILITY_RESOURCE, resource=SELECTOR, request_resource=REQUEST_URL),
        CapabilityRequest(
            kind=CAPABILITY_RESOURCE_CLAIM,
            resource=SELECTOR,
            request_resource=REQUEST_URL,
            claim="named_services:use",
        ),
        CapabilityRequest(
            kind=CAPABILITY_OUTER_OPERATION,
            resource=SELECTOR,
            request_resource=REQUEST_URL,
            surface="mcp",
            outer_operation="named_services_schema",
        ),
        CapabilityRequest(
            kind=CAPABILITY_NAMED_SERVICE_NAMESPACE,
            resource=SELECTOR,
            request_resource=REQUEST_URL,
            surface="mcp",
            namespace="mail",
        ),
        CapabilityRequest(
            kind=CAPABILITY_NAMED_SERVICE_OPERATION,
            resource=SELECTOR,
            request_resource=REQUEST_URL,
            surface="mcp",
            namespace="mail",
            operation="object.schema",
        ),
    ],
)
def test_a_capability_the_catalog_still_offers_is_allowed(request_):
    assert _authorize(request_) is None


def test_nested_operations_mapping_is_read_as_well_as_the_flat_form():
    allowed = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="linkedin",
        operation="object.action.publish",
    )
    assert _authorize(allowed) is None


def test_a_namespace_key_is_matched_after_normalization():
    request_ = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_NAMESPACE,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="Mail:",
    )
    assert _authorize(request_) is None


# -- what the active catalog no longer offers ----------------------------------


def test_a_removed_namespace_denies_and_names_the_whole_path():
    request_ = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        outer_operation="named_services_schema",
        namespace="mail",
        operation="object.schema",
    )
    denial = _authorize(request_, connections=_without_namespace("mail"))

    assert denial is not None
    assert denial["ok"] is False
    assert denial["error"]["code"] == DENIAL_CODE
    assert denial["error"]["retryable"] is False
    ret = denial["ret"]
    assert ret["reason"] == DENIAL_REASON
    assert ret["access_id"] == PROVENANCE.access_id
    assert ret["card_revision"] == 8
    assert ret["card_catalog_version"] == PROVENANCE.catalog_version
    assert ret["active_catalog_version"] != PROVENANCE.catalog_version
    assert ret["recovery"]["retry_same_request"] is False
    assert ret["requested_capability"] == {
        "kind": CAPABILITY_NAMED_SERVICE_OPERATION,
        "resource": SELECTOR,
        "request_resource": REQUEST_URL,
        "surface": "mcp",
        "outer_operation": "named_services_schema",
        "namespace": "mail",
        "operation": "object.schema",
    }


def test_a_removed_operation_denies_while_its_namespace_survives():
    trimmed = copy.deepcopy(CONNECTIONS)
    namespaces = trimmed["delegated_credentials"]["oauth"]["resources"][0]["named_services"]["namespaces"]
    namespaces["mail"]["tools"].pop("schema")

    denied = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="mail",
        operation="object.schema",
    )
    still_offered = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_NAMESPACE,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="mail",
    )

    assert _authorize(denied, connections=trimmed) is not None
    assert _authorize(still_offered, connections=trimmed) is None


def test_a_removed_claim_denies_even_though_the_resource_survives():
    trimmed = copy.deepcopy(CONNECTIONS)
    resource = trimmed["delegated_credentials"]["oauth"]["resources"][0]
    resource["named_services"]["namespaces"]["mail"]["tools"]["search"]["grants"] = [
        "named_services:use"
    ]

    denied = CapabilityRequest(
        kind=CAPABILITY_RESOURCE_CLAIM,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        claim="mail:read",
    )
    resource_itself = CapabilityRequest(
        kind=CAPABILITY_RESOURCE, resource=SELECTOR, request_resource=REQUEST_URL
    )

    assert _authorize(denied, connections=trimmed) is not None
    assert _authorize(resource_itself, connections=trimmed) is None


def test_a_removed_outer_operation_denies():
    trimmed = copy.deepcopy(CONNECTIONS)
    trimmed["delegated_credentials"]["oauth"]["resources"][0]["tools"].pop("named_services_schema")
    request_ = CapabilityRequest(
        kind=CAPABILITY_OUTER_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        outer_operation="named_services_schema",
    )
    assert _authorize(request_, connections=trimmed) is not None


def test_a_removed_resource_denies_every_kind_beneath_it():
    empty = {"delegated_credentials": {"oauth": {"enabled": True, "resources": []}}}
    for request_ in (
        CapabilityRequest(kind=CAPABILITY_RESOURCE, resource=SELECTOR, request_resource=REQUEST_URL),
        CapabilityRequest(
            kind=CAPABILITY_RESOURCE_CLAIM,
            resource=SELECTOR,
            request_resource=REQUEST_URL,
            claim="named_services:use",
        ),
        CapabilityRequest(
            kind=CAPABILITY_NAMED_SERVICE_OPERATION,
            resource=SELECTOR,
            request_resource=REQUEST_URL,
            surface="mcp",
            namespace="mail",
            operation="object.schema",
        ),
    ):
        assert _authorize(request_, connections=empty) is not None


def test_a_namespace_block_emptied_of_namespaces_is_a_removal_not_an_absent_section():
    emptied = copy.deepcopy(CONNECTIONS)
    resource = emptied["delegated_credentials"]["oauth"]["resources"][0]
    resource["named_services"]["namespaces"] = {}

    request_ = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="mail",
        operation="object.schema",
    )
    assert _authorize(request_, connections=emptied) is not None


@pytest.mark.parametrize(
    "withdraw",
    [
        pytest.param(lambda r: r.pop("named_services"), id="block-deleted"),
        pytest.param(lambda r: r.__setitem__("named_services", {}), id="block-emptied"),
    ],
)
def test_a_resource_that_publishes_no_named_services_offers_no_inner_operation(withdraw):
    # The named-service dimension is the offer itself: a row that publishes no
    # namespace offers no inner operation. Claims and outer operations keep
    # reading an unenumerated dimension as no ceiling.
    withdrawn = copy.deepcopy(CONNECTIONS)
    withdraw(withdrawn["delegated_credentials"]["oauth"]["resources"][0])

    request_ = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="mail",
        operation="object.schema",
    )
    assert _authorize(request_, connections=withdrawn) is not None


def test_a_resource_that_enumerates_no_tools_carries_no_outer_ceiling():
    without_tools = {
        "delegated_credentials": {
            "oauth": {
                "enabled": True,
                "resources": [{"resource": "*", "grants": ["kdcube:role:super-admin"]}],
            }
        }
    }
    request_ = CapabilityRequest(
        kind=CAPABILITY_OUTER_OPERATION,
        resource="*",
        request_resource=REQUEST_URL,
        surface="rest",
        outer_operation="anything",
    )
    assert _authorize(request_, connections=without_tools) is None


def test_an_empty_catalog_body_denies_rather_than_permitting_everything():
    request_ = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="mail",
        operation="object.schema",
    )
    assert _authorize(request_, connections={}) is not None


# -- denial shape ---------------------------------------------------------------


def test_the_path_carries_only_the_fields_its_kind_requires():
    request_ = CapabilityRequest(
        kind=CAPABILITY_RESOURCE_CLAIM,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        outer_operation="named_services_schema",
        claim="mail:read",
        namespace="mail",
        operation="object.schema",
    )
    empty = {"delegated_credentials": {"oauth": {"enabled": True, "resources": []}}}
    path = _authorize(request_, connections=empty)["ret"]["requested_capability"]

    assert path == {
        "kind": CAPABILITY_RESOURCE_CLAIM,
        "resource": SELECTOR,
        "request_resource": REQUEST_URL,
        "claim": "mail:read",
    }


def test_a_denial_never_names_the_leaf_operation_alone():
    request_ = CapabilityRequest(
        kind=CAPABILITY_NAMED_SERVICE_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        namespace="mail",
        operation="object.schema",
    )
    path = _authorize(request_, connections=_without_namespace("mail"))["ret"][
        "requested_capability"
    ]
    assert {"resource", "surface", "namespace", "operation"}.issubset(path)


def test_unavailability_is_retryable_and_distinct_from_removal():
    payload = catalog_unavailable_denial("cache_unavailable")
    assert payload["ok"] is False
    assert payload["error"]["retryable"] is True
    assert payload["error"]["code"] != DENIAL_CODE
    assert payload["ret"]["reason"] == "cache_unavailable"


# -- the outer dimension and the all-resource admin row ------------------------
#
# Two rows read the same absent `tools` block and must not answer the same way.
# A door that stopped publishing its outer tools has withdrawn them. The
# all-resource admin row never published any, and is reachable only by a card
# whose own selector is `*` (config.card_selector_config).

ADMIN_SELECTOR = "*"
UNLISTED_URL = (
    "https://kdcube.example/api/integrations/bundles/acme/prod/"
    "unlisted@1-0/public/mcp/anything"
)


_WITHDRAWALS = [
    pytest.param(lambda r: r.pop("tools"), id="block-deleted"),
    pytest.param(lambda r: r.__setitem__("tools", {}), id="block-emptied"),
]


def _without_outer_tools(withdraw=lambda r: r.__setitem__("tools", {})) -> dict:
    trimmed = copy.deepcopy(CONNECTIONS)
    withdraw(trimmed["delegated_credentials"]["oauth"]["resources"][0])
    return trimmed


def _with_admin_row(connections=None) -> dict:
    extended = copy.deepcopy(connections if connections is not None else CONNECTIONS)
    extended["delegated_credentials"]["oauth"]["resources"].append(
        {
            "resource": ADMIN_SELECTOR,
            "label": "All KDCube resources",
            "admin_only": True,
            "grants": ["kdcube:admin"],
        }
    )
    return extended


@pytest.mark.parametrize("withdraw", _WITHDRAWALS)
def test_an_outer_operation_the_catalog_withdrew_is_denied(withdraw):
    # Deleting the block and emptying it are the same withdrawal: a door that
    # publishes no tool offers none.
    request_ = CapabilityRequest(
        kind=CAPABILITY_OUTER_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        outer_operation="named_services_search",
    )
    denial = _authorize(request_, connections=_without_outer_tools(withdraw))

    assert denial is not None
    assert denial["error"]["code"] == DENIAL_CODE
    assert denial["ret"]["requested_capability"]["outer_operation"] == "named_services_search"


def test_the_all_resource_admin_row_carries_no_outer_ceiling():
    request_ = CapabilityRequest(
        kind=CAPABILITY_OUTER_OPERATION,
        resource=ADMIN_SELECTOR,
        request_resource=UNLISTED_URL,
        surface="mcp",
        outer_operation="anything_at_all",
    )
    assert _authorize(request_, connections=_with_admin_row()) is None


# -- the claim ceiling is the row's grants, written or derived -----------------
#
# `resources[].grants` is the door's claim ceiling; when it is not written it is
# derived from the grants its tools and namespaces require. There is no third
# state in which a door carries no claim ceiling, and no source describes one for
# the all-resource row either — unlike outer tools, a claim has no second source:
# the guard reduces stored claims through `resource_claims` alone.


def _claim_request(claim: str = "mail:read") -> CapabilityRequest:
    return CapabilityRequest(
        kind=CAPABILITY_RESOURCE_CLAIM,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        claim=claim,
    )


def _with_row(row: dict) -> dict:
    return {"delegated_credentials": {"oauth": {"enabled": True, "resources": [row]}}}


_CLAIM_CEILINGS = [
    pytest.param({"resource": SELECTOR, "grants": ["mail:read", "x:y"]}, True, id="declared-present"),
    pytest.param({"resource": SELECTOR, "grants": ["x:y"]}, False, id="declared-removed"),
    pytest.param({"resource": SELECTOR, "grants": []}, False, id="emptied"),
    pytest.param({"resource": SELECTOR}, False, id="deleted"),
    pytest.param({"resource": "*", "grants": []}, False, id="all-resource-row-emptied"),
]


@pytest.mark.parametrize("row,allowed", _CLAIM_CEILINGS)
def test_a_claim_the_catalog_no_longer_offers_is_denied(row, allowed):
    denial = _authorize(_claim_request(), connections=_with_row(row))
    assert (denial is None) is allowed


@pytest.mark.parametrize("row,_allowed", _CLAIM_CEILINGS)
def test_both_readers_of_the_claim_ceiling_agree(row, _allowed):
    """`permits` and `resource_claims` read one dimension from one row.

    They disagreed while an empty ceiling read as no ceiling in one and as
    nothing available in the other, and the guard consults both: the loop that
    finds a withdrawn claim is keyed on `resource_claims`, then asks `permits` to
    build the denial. A disagreement there yields no denial body at all.
    """
    catalog = _catalog(_with_row(row))
    request_ = _claim_request()

    assert catalog.permits(request_) is (
        _clean_claim(request_.claim) in catalog.resource_claims(request_)
    )


def _clean_claim(value: str) -> str:
    return str(value or "").strip()


def test_a_withdrawn_door_is_not_rescued_by_the_admin_row():
    request_ = CapabilityRequest(
        kind=CAPABILITY_OUTER_OPERATION,
        resource=SELECTOR,
        request_resource=REQUEST_URL,
        surface="mcp",
        outer_operation="named_services_search",
    )
    denial = _authorize(
        request_, connections=_with_admin_row(_without_outer_tools())
    )

    assert denial is not None
    assert denial["error"]["code"] == DENIAL_CODE
