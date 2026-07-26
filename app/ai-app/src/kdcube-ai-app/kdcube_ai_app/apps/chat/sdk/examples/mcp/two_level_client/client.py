# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# chat/sdk/examples/mcp/two_level_client/client.py
#
# A runnable, plain MCP client that narrates KDCube's two-level consent
# lifecycle. This is a THIN driver over the reference classifier in two_level.py.

"""Connect to a KDCube governed MCP door as a plain MCP client and narrate
what happens.

The demonstration is the narration: connect over streamable-http with a
delegated bearer, list the tool roster, call one tool, and print which
authorization level (if any) denied the call and exactly how the user fixes it.
A bare 403 never happens - every denial arrives as a structured, actionable
two-level consent envelope.

Run from ``app/ai-app/src/kdcube-ai-app``::

    python3 -m kdcube_ai_app.apps.chat.sdk.examples.mcp.two_level_client.client \\
        --url  "$KDCUBE_MCP_URL" \\
        --bearer "$KDCUBE_MCP_BEARER" \\
        --tool productivity_slack_search \\
        --query "quarterly planning"

``--url`` / ``--bearer`` fall back to ``KDCUBE_MCP_URL`` / ``KDCUBE_MCP_BEARER``.
See README.md for how to obtain a delegated bearer (connect via OAuth, or mint a
bounded automation token in Connection Hub).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.examples.mcp.two_level_client.two_level import (
    ConsentOutcome,
    classify_tool_result,
    result_payload_from_call_tool,
)

DEFAULT_TOOL = "productivity_slack_search"


# --------------------------------------------------------------------------
# transport - mirrors runtime/mcp/mcp_adapter.py exactly
# --------------------------------------------------------------------------

async def _list_and_call(
    *,
    url: str,
    bearer: str,
    tool: str,
    params: Mapping[str, Any],
) -> tuple[list[dict], Mapping[str, Any]]:
    """Open one streamable-http session: initialize, list tools, call one tool.

    `mcp` is imported lazily so this module imports cleanly even where the SDK
    is absent (the classifier and its tests need no transport).
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {bearer}"}
    async with streamablehttp_client(url, headers=headers) as (read, write, *_rest):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            roster = [
                {
                    "id": getattr(t, "name", "") or getattr(t, "id", ""),
                    "description": (getattr(t, "description", "") or "").strip(),
                }
                for t in (getattr(listed, "tools", None) or [])
            ]

            result = await session.call_tool(tool, dict(params))
            payload = result_payload_from_call_tool(result)
            return roster, payload


# --------------------------------------------------------------------------
# narration - the demonstration
# --------------------------------------------------------------------------

def _print_roster(roster: list[dict]) -> None:
    print(f"\nTool roster ({len(roster)} tools):")
    if not roster:
        print("  (the door listed no tools for this bearer)")
        return
    for entry in roster:
        desc = entry.get("description") or ""
        suffix = f" - {desc}" if desc else ""
        print(f"  - {entry['id']}{suffix}")


def _preview(payload: Mapping[str, Any]) -> str:
    body = payload.get("ret")
    if body is None:
        body = {k: v for k, v in payload.items() if k != "ok"} or payload
    text = json.dumps(body, ensure_ascii=False, default=str)
    return text if len(text) <= 480 else text[:477] + "..."


def narrate(outcome: ConsentOutcome, *, tool: str) -> None:
    """Print the two-level narration for one classified tool result.

    Each denial names its level and its fix. There is NO retry loop here: the
    door's contract is relay-and-stop - the user acts (or, for
    account_required, the caller resends with an account_id), never a blind
    replay hammering the door.
    """
    print("\n" + "=" * 72)

    if outcome.level == 0 and outcome.ok:
        print(f"SUCCESS  '{tool}'")
        print("  level 1 passed (the caller's own grant)")
        print("  level 2 passed (the connected account + per-account binding)")
        print("  result: " + _preview(outcome.raw))
        print("=" * 72)
        return

    if outcome.level == 1:
        claims = ", ".join(outcome.claims) if outcome.claims else "(unnamed)"
        print(f"LEVEL 1 - the caller's own grant   code={outcome.code}")
        print(f"  Missing grant(s): {claims}")
        print(f"  Fix: {outcome.next_action}")
        print("  (a level-1 grant miss is NEVER fixed by a blind resend - do not retry)")
        print("=" * 72)
        return

    if outcome.level == 2:
        print(f"LEVEL 2 - the connected account + binding   reason={outcome.reason}   "
              f"retry_hint={str(outcome.retry_hint).lower()}")
        if outcome.provider_id:
            print(f"  Provider: {outcome.provider_id}")
        if outcome.claims:
            print(f"  Claim(s): {', '.join(outcome.claims)}")
        if outcome.resend_with_account_id:
            ids = outcome.candidate_account_ids()
            print("  Candidate accounts:")
            for cand in outcome.candidates:
                aid = cand.get("account_id", "")
                label = cand.get("label") or cand.get("email") or cand.get("workspace") or ""
                print(f"    - account_id={aid}" + (f"  ({label})" if label else ""))
            hint = ids[0] if ids else "<one of candidates>"
            print(f"  Fix: resend the SAME call with account_id={hint}")
        else:
            print(f"  Fix: {outcome.next_action}")
        print("  (relay this to the user; do not retry blindly)")
        print("=" * 72)
        return

    # level 0, not ok: unrecognised / opaque failure - degraded gracefully.
    print(f"UNCLASSIFIED  code={outcome.code}")
    print(f"  {outcome.next_action}")
    print("=" * 72)


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="two_level_client",
        description="Consume a KDCube governed MCP door and narrate its two-level consent lifecycle.",
    )
    parser.add_argument("--url", default=os.environ.get("KDCUBE_MCP_URL", ""),
                        help="The governed door URL (or env KDCUBE_MCP_URL). "
                             "e.g. <base>/api/integrations/bundles/<tenant>/<project>/"
                             "kdcube-services@1-0/public/mcp/productivity")
    parser.add_argument("--bearer", default=os.environ.get("KDCUBE_MCP_BEARER", ""),
                        help="The delegated bearer token (or env KDCUBE_MCP_BEARER). "
                             "See README.md for how to obtain one.")
    parser.add_argument("--tool", default=DEFAULT_TOOL,
                        help=f"Tool id to call (default: {DEFAULT_TOOL}).")
    parser.add_argument("--query", default="quarterly planning",
                        help="Query passed as the tool's `query` param (search tools).")
    parser.add_argument("--account-id", default="",
                        help="Optional account_id to resend a level-2 account_required call with.")
    return parser.parse_args(argv)


def _missing_config_message(args: argparse.Namespace) -> str:
    missing = []
    if not args.url:
        missing.append("--url (or KDCUBE_MCP_URL)")
    if not args.bearer:
        missing.append("--bearer (or KDCUBE_MCP_BEARER)")
    return (
        "Missing required configuration: " + ", ".join(missing) + ".\n"
        "  --url    a KDCube governed door, e.g.\n"
        "           <base>/api/integrations/bundles/<tenant>/<project>/"
        "kdcube-services@1-0/public/mcp/productivity\n"
        "  --bearer a delegated bearer token for that door.\n"
        "How to obtain a bearer: connect via OAuth (dynamic client registration),\n"
        "or mint a bounded automation token in Connection Hub. See this example's\n"
        "README.md and the authenticated-MCP reference it links."
    )


def _build_params(args: argparse.Namespace) -> dict:
    params: dict = {}
    if args.query:
        params["query"] = args.query
    # account_id lets you demonstrate the level-2 account_required resolution:
    # the SAME call, resent with the account picked from the candidates.
    if args.account_id:
        params["account_id"] = args.account_id
    return params


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.url or not args.bearer:
        print(_missing_config_message(args), file=sys.stderr)
        return 2

    print(f"Connecting to governed door: {args.url}")
    print(f"Calling tool: {args.tool}  params: {_build_params(args) or '{}'}")

    try:
        roster, payload = asyncio.run(
            _list_and_call(
                url=args.url,
                bearer=args.bearer,
                tool=args.tool,
                params=_build_params(args),
            )
        )
    except ImportError:
        print(
            "The `mcp` SDK is not installed. Install it (pip install mcp) to run "
            "the live demo; the classifier in two_level.py needs no transport.",
            file=sys.stderr,
        )
        return 3
    except Exception as exc:  # transport / auth / network - report, don't retry
        print(f"\nCould not complete the MCP call: {exc}", file=sys.stderr)
        print(
            "If this is an authentication failure, the bearer may be missing, "
            "expired, or scoped to a different door. See README.md.",
            file=sys.stderr,
        )
        return 4

    _print_roster(roster)
    outcome = classify_tool_result(payload)
    narrate(outcome, tool=args.tool)
    # Exit code reflects the level so the demo is scriptable: 0 ok, else the level.
    return 0 if outcome.ok else (outcome.level or 5)


if __name__ == "__main__":
    raise SystemExit(main())
