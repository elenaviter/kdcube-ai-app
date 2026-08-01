# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Search document files through the KDCube Google Docs SDK only.

This focused diagnostic calls ``execute_google_docs_operation`` and never
constructs a Google Drive request itself. Use ``debug_search.py`` when the SDK
result must be compared with raw provider controls.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kdcube_ai_app.apps.chat.sdk.integrations.google.docs_proxy import (
    execute_google_docs_operation,
)


HERE = Path(__file__).resolve().parent
_SENSITIVE_KEYS = {
    "access_token",
    "authorization",
    "refresh_token",
    "token",
}


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(HERE / ".env", override=False)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _limit(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 20
    return max(1, min(parsed, 50))


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<redacted>"
                if str(key).casefold() in _SENSITIVE_KEYS
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _print_json(label: str, value: Any) -> None:
    print(f"\n--- {label} ---")
    print(json.dumps(_redact(value), ensure_ascii=False, indent=2, default=str))


def _result_items(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ret = result.get("ret")
    if not isinstance(ret, Mapping):
        return []
    return [
        item
        for item in (ret.get("items") or [])
        if isinstance(item, Mapping)
    ]


async def main() -> int:
    _load_local_env()
    query = _clean(os.getenv("KDCUBE_DOCS_QUERY")) or "26_006"
    limit = _limit(os.getenv("KDCUBE_DOCS_LIMIT"))
    access_token = _clean(os.getenv("GOOGLE_ACCESS_TOKEN"))

    print("Google Docs SDK search (read-only)")
    print(f"query={query!r} limit={limit}")
    print("The access token is loaded locally and never printed.")

    if not access_token:
        print(
            "Set GOOGLE_ACCESS_TOKEN in the Run Configuration or local .env.",
            file=sys.stderr,
        )
        return 2

    search = await execute_google_docs_operation(
        operation="search",
        access_token=access_token,
        payload={"query": query, "limit": limit},
    )
    _print_json("SDK search", search)
    if not search.get("ok"):
        return 1

    items = _result_items(search)
    if not items:
        print("\nSDK result: no matching native document or import source.")
        return 0

    selected = next(
        (item for item in items if item.get("exact_title_match")),
        items[0],
    )
    document_id = _clean(selected.get("document_id"))
    if not document_id:
        print("\nSDK result did not contain a document_id.", file=sys.stderr)
        return 1

    source = await execute_google_docs_operation(
        operation="get_source",
        access_token=access_token,
        payload={"document_ref": document_id},
    )
    _print_json("SDK selected source metadata", source)
    return 0 if source.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
