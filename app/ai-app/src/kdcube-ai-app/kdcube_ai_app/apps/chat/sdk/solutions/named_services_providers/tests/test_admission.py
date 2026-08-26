# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Admission is mandatory and invocation-scoped at named-service dispatch."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceAdmission,
    NamedServiceAdmissionDecision,
    NamedServiceAdmissionSelector,
    NamedServiceEndpoint,
    NamedServiceRequest,
    NamedServiceResponse,
    NamedServiceStreamResult,
    call_named_service_endpoint,
    call_named_service_endpoint_stream,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.admission import (
    DELEGATED_SELECTOR_BEARER,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.transports import (
    api_client,
)


def _request() -> NamedServiceRequest:
    return NamedServiceRequest(namespace="mail", operation="object.search")


def _endpoint() -> NamedServiceEndpoint:
    return NamedServiceEndpoint(transport="module", namespace="mail")


class _SequencedAuthorizer:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0

    async def authorize(self, request):
        assert request.namespace == "mail"
        self.calls += 1
        return self.decisions.pop(0)


def _delegated(authorizer) -> NamedServiceAdmission:
    return NamedServiceAdmission.delegated(
        selector=NamedServiceAdmissionSelector(
            mode="delegated",
            source="test.admission",
            delegated_kind=DELEGATED_SELECTOR_BEARER,
            access_id="oauth-test",
            client_id="claude",
            grantor_user_id="user-1",
            delegate_identity="integration:claude:user-1",
        ),
        authorizer=authorizer,
    )


def _denial(code="denied") -> NamedServiceAdmissionDecision:
    return NamedServiceAdmissionDecision.deny(
        NamedServiceResponse.error_response(
            code=code,
            message="Admission denied the invocation.",
            status=403,
            namespace="mail",
        )
    )


async def test_dispatch_requires_an_explicit_admission_argument():
    with pytest.raises(TypeError, match="admission"):
        await call_named_service_endpoint(_endpoint(), _request())  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="admission"):
        await call_named_service_endpoint_stream(_endpoint(), _request())  # type: ignore[call-arg]


async def test_delegated_denial_happens_before_provider_discovery(monkeypatch):
    authorizer = _SequencedAuthorizer([_denial("card_revoked")])

    async def _never(*_args, **_kwargs):
        raise AssertionError("provider discovery must not run after admission denial")

    monkeypatch.setattr(api_client, "_resolve_endpoint_from_discovery", _never)
    response = await call_named_service_endpoint(
        _endpoint(),
        NamedServiceRequest(
            namespace="mail",
            operation="object.search",
            context={"authority": "application", "governed": False},
        ),
        admission=_delegated(authorizer),
    )

    assert response.ok is False
    assert response.error.code == "card_revoked"
    assert authorizer.calls == 1


async def test_application_admission_dispatches_without_hub_lookup(monkeypatch):
    async def _resolved(endpoint, request):
        return endpoint

    async def _module_call(endpoint, request):
        return NamedServiceResponse.ok_response(namespace=request.namespace)

    async def _hub_never(*_args, **_kwargs):
        raise AssertionError("application authority must not query Connection Hub")

    monkeypatch.setattr(api_client, "_resolve_endpoint_from_discovery", _resolved)
    monkeypatch.setattr(api_client, "_call_module_endpoint", _module_call)
    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.solutions.connections.named_service_admission."
        "HubNamedServiceAdmissionAuthorizer.authorize",
        _hub_never,
    )

    response = await call_named_service_endpoint(
        _endpoint(),
        NamedServiceRequest(
            namespace="mail",
            operation="object.search",
            context={"admission": {"mode": "delegated", "access_id": "forged"}},
        ),
        admission=NamedServiceAdmission.application(source="test.application"),
    )

    assert response.ok is True


def test_admission_is_not_part_of_named_service_request_serialization():
    request = NamedServiceRequest(
        namespace="mail",
        operation="object.search",
        context={"source": "test"},
    )
    admission = _delegated(_SequencedAuthorizer([]))

    serialized = request.to_dict()

    assert "admission" not in serialized
    assert admission.selector.access_id not in str(serialized)


async def test_each_invocation_resolves_delegated_authority_again(monkeypatch):
    authorizer = _SequencedAuthorizer([_denial("first"), _denial("revoked")])
    admission = _delegated(authorizer)

    async def _never(*_args, **_kwargs):
        raise AssertionError("provider discovery must not run after admission denial")

    monkeypatch.setattr(api_client, "_resolve_endpoint_from_discovery", _never)
    first = await call_named_service_endpoint(
        _endpoint(), _request(), admission=admission
    )
    second = await call_named_service_endpoint(
        _endpoint(), _request(), admission=admission
    )

    assert first.error.code == "first"
    assert second.error.code == "revoked"
    assert authorizer.calls == 2


async def test_stream_authorizes_once_and_resets_scope_before_consumption(monkeypatch):
    scope_state = {"bound": False, "enters": 0, "exits": 0}

    class _Scope:
        @contextmanager
        def bind(self):
            scope_state["bound"] = True
            scope_state["enters"] += 1
            try:
                yield
            finally:
                scope_state["bound"] = False
                scope_state["exits"] += 1

    decision = NamedServiceAdmissionDecision.allow(execution_scope=_Scope())
    authorizer = _SequencedAuthorizer([decision])

    async def _resolved(endpoint, request):
        return endpoint

    async def _chunks():
        assert scope_state["bound"] is False
        yield b"one"
        assert scope_state["bound"] is False
        yield b"two"

    async def _module_call(endpoint, request):
        assert scope_state["bound"] is True
        return NamedServiceStreamResult(
            response=NamedServiceResponse.ok_response(namespace="mail"),
            chunks=_chunks(),
            filename="mail.bin",
        )

    monkeypatch.setattr(api_client, "_resolve_endpoint_from_discovery", _resolved)
    monkeypatch.setattr(api_client, "_call_module_endpoint_raw", _module_call)

    result = await call_named_service_endpoint_stream(
        _endpoint(), _request(), admission=_delegated(authorizer)
    )

    assert scope_state == {"bound": False, "enters": 1, "exits": 1}
    assert [chunk async for chunk in result.chunks] == [b"one", b"two"]
    assert authorizer.calls == 1
    assert scope_state == {"bound": False, "enters": 1, "exits": 1}


def test_production_dispatch_calls_name_their_admission():
    """A new caller cannot silently restore ambient application authority."""

    sdk_root = Path(api_client.__file__).resolve().parents[3]
    dispatch_names = {
        "call_named_service_endpoint",
        "call_named_service_endpoint_stream",
    }
    missing: list[str] = []
    for path in sdk_root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if called not in dispatch_names:
                continue
            if not any(keyword.arg == "admission" for keyword in node.keywords):
                missing.append(f"{path.relative_to(sdk_root)}:{node.lineno}")

    assert missing == []
