from __future__ import annotations

from collections.abc import Sequence

import pytest
from kdcube_ai_app.apps.chat.proc.rest.management.http_input import (
    ManagementRequestBodyError,
    read_bounded_body,
    read_json_object,
)
from starlette.requests import Request


def _request(
    chunks: Sequence[bytes],
    *,
    content_type: str = "application/json",
    content_length: str | None = None,
    extra_headers: Sequence[tuple[bytes, bytes]] = (),
) -> Request:
    headers = [(b"content-type", content_type.encode("ascii"))]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode("ascii")))
    headers.extend(extra_headers)
    remaining = list(chunks)

    async def receive() -> dict:
        body = remaining.pop(0) if remaining else b""
        return {
            "type": "http.request",
            "body": body,
            "more_body": bool(remaining),
        }

    return Request({"type": "http", "headers": headers}, receive)


@pytest.mark.asyncio
async def test_bounded_body_accepts_chunks_only_inside_limit() -> None:
    request = _request([b'{"a":', b"1}"], content_length="7")

    assert await read_json_object(request, maximum_bytes=7) == {"a": 1}


@pytest.mark.asyncio
async def test_bounded_body_rejects_declared_and_streamed_overflow() -> None:
    declared = _request([b"{}"], content_length="8")
    streamed = _request([b"1234", b"5"])

    with pytest.raises(ManagementRequestBodyError, match="too large"):
        await read_bounded_body(
            declared,
            maximum_bytes=7,
            media_type="application/json",
        )
    with pytest.raises(ManagementRequestBodyError, match="too large"):
        await read_bounded_body(
            streamed,
            maximum_bytes=4,
            media_type="application/json",
        )


@pytest.mark.asyncio
async def test_json_object_rejects_ambiguous_or_nonstandard_input() -> None:
    duplicate = _request([b'{"key":"first","key":"second"}'])
    nonstandard = _request([b'{"value":NaN}'])
    wrong_type = _request([b"{}"], content_type="text/plain")
    duplicate_type = _request(
        [b"{}"],
        extra_headers=((b"content-type", b"application/json"),),
    )
    encoded = _request(
        [b"{}"],
        extra_headers=((b"content-encoding", b"gzip"),),
    )

    for request in (
        duplicate,
        nonstandard,
        wrong_type,
        duplicate_type,
        encoded,
    ):
        with pytest.raises(ManagementRequestBodyError):
            await read_json_object(request, maximum_bytes=1024)
