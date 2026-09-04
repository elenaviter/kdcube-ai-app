# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Bounded, unambiguous HTTP request-body parsing for management routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request


class ManagementRequestBodyError(ValueError):
    pass


def _reject_json_constant(_value: str) -> None:
    raise ManagementRequestBodyError("JSON constants are not supported")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManagementRequestBodyError("JSON object keys must be unique")
        result[key] = value
    return result


async def read_bounded_body(
    request: Request,
    *,
    maximum_bytes: int,
    media_type: str,
) -> bytes:
    """Read one required body without allocating beyond the declared bound."""

    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    content_types = request.headers.getlist("content-type")
    if len(content_types) != 1:
        raise ManagementRequestBodyError("request content type is invalid")
    actual_media_type = content_types[0].partition(";")[0].strip().lower()
    if actual_media_type != media_type:
        raise ManagementRequestBodyError("request content type is invalid")
    if request.headers.getlist("content-encoding"):
        raise ManagementRequestBodyError("request content encoding is unsupported")

    lengths = request.headers.getlist("content-length")
    if len(lengths) > 1:
        raise ManagementRequestBodyError("request content length is invalid")
    if lengths:
        raw_length = lengths[0].strip()
        if not raw_length.isdecimal():
            raise ManagementRequestBodyError("request content length is invalid")
        if int(raw_length) > maximum_bytes:
            raise ManagementRequestBodyError("request body is too large")

    body = bytearray()
    async for chunk in request.stream():
        if len(chunk) > maximum_bytes - len(body):
            raise ManagementRequestBodyError("request body is too large")
        body.extend(chunk)
    if not body:
        raise ManagementRequestBodyError("request body is required")
    return bytes(body)


async def read_json_object(
    request: Request,
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    raw = await read_bounded_body(
        request,
        maximum_bytes=maximum_bytes,
        media_type="application/json",
    )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManagementRequestBodyError("request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ManagementRequestBodyError("request body must be one JSON object")
    return value


__all__ = [
    "ManagementRequestBodyError",
    "read_bounded_body",
    "read_json_object",
]
