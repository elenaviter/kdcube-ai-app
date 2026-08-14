# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# A bundle MCP surface answers a plain request/response call. The framing it
# answers in decides whether the caller receives it at all: an event stream is
# delivered byte-for-byte by the deployment's own proxy and STILL lost by the
# client when a tunnel or HTTP/2 edge closes the stream badly, while the server
# logs a happy 200. These tests pin the two halves of the fix — ask the SDK for
# JSON framing, and re-frame a single-message stream we did not build.

import json

from kdcube_ai_app.apps.chat.proc.rest.integrations.integrations import (
    _coerce_bundle_mcp_asgi_app,
    _collapse_single_message_event_stream,
)

ACCEPT_BOTH = "application/json, text/event-stream"


def _sse(*payloads: dict) -> bytes:
    lines = []
    for payload in payloads:
        lines.append("event: message")
        lines.append(f"data: {json.dumps(payload)}")
        lines.append("")
    return "\n".join(lines).encode("utf-8")


class _Server:
    """Stands in for an SDK MCPServer/FastMCP with the modern factory."""

    def __init__(self):
        self.kwargs = None

    def streamable_http_app(self, *, streamable_http_path="/mcp", json_response=False, stateless_http=False):
        self.kwargs = {"json_response": json_response, "stateless_http": stateless_http}
        return object()


class _LegacyServer:
    def __init__(self):
        self.kwargs = None

    def streamable_http_app(self):
        self.kwargs = {}
        return object()


def test_sdk_app_is_asked_for_json_framing():
    server = _Server()
    _coerce_bundle_mcp_asgi_app(server, transport="streamable-http")
    assert server.kwargs == {"json_response": True, "stateless_http": True}


def test_legacy_factory_without_kwargs_still_builds():
    server = _LegacyServer()
    _coerce_bundle_mcp_asgi_app(server, transport="streamable-http")
    assert server.kwargs == {}


def test_single_message_stream_is_reframed_as_json():
    payload = {"jsonrpc": "2.0", "id": "1", "result": {"content": [{"text": "ok"}]}}
    collapsed = _collapse_single_message_event_stream(
        _sse(payload), content_type="text/event-stream", accept=ACCEPT_BOTH,
    )
    assert collapsed is not None
    assert json.loads(collapsed) == payload


def test_multi_message_stream_is_passed_through_untouched():
    # Collapsing this would DROP the progress notification.
    body = _sse(
        {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}},
        {"jsonrpc": "2.0", "id": "1", "result": {}},
    )
    assert _collapse_single_message_event_stream(
        body, content_type="text/event-stream", accept=ACCEPT_BOTH,
    ) is None


def test_client_that_asked_only_for_a_stream_keeps_its_stream():
    body = _sse({"jsonrpc": "2.0", "id": "1", "result": {}})
    assert _collapse_single_message_event_stream(
        body, content_type="text/event-stream", accept="text/event-stream",
    ) is None


def test_json_body_is_left_alone():
    body = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {}}).encode("utf-8")
    assert _collapse_single_message_event_stream(
        body, content_type="application/json", accept=ACCEPT_BOTH,
    ) is None
