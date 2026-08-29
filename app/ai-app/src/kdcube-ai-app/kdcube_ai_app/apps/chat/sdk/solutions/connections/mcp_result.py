# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Chat-side post-processing for KDCube MCP tool results — shared by every
consumer.

An agent that consumes a KDCube-served MCP surface runs INSIDE a chat turn, so
two things a raw tool result cannot do on its own must happen on the caller
side: a consent denial has to become a chat banner, and a file has to reach the
user as a card (never a signed URL the model re-types). This module is that
post-processor, applied ONCE by the SDK MCP loader, so no bundle re-implements
it and there is nothing to drift.

It is driven ENTIRELY by the result's self-describing consent block — the same
contract every KDCube MCP surface returns:

  * ``delegated_consent_required`` — the AGENT's own grant is missing; the block
    carries ``agent_client_id``, ``resource``, ``claims``, ``namespace`` and (for
    a hosted agent) the one-click ``grant`` action.
  * ``needs_connected_account_consent`` — the user's PROVIDER account is
    missing/expired; the block carries ``provider_id``, ``connector_app_id``,
    ``claims``, ``namespace``, ``url``.

Both banners are raised through the SAME shared announce the NATIVE named-service
path uses, so the two surfaces cannot diverge. An external client (Claude Code)
has no chat lane, so the announce is a no-op and the result — which already
carries the Connection Hub link — flows to it unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

DELEGATED_CONSENT_REQUIRED = "delegated_consent_required"
NEEDS_CONNECTED_ACCOUNT_CONSENT = "needs_connected_account_consent"
_MARKERS = ('"download"', f'"{DELEGATED_CONSENT_REQUIRED}"', f'"{NEEDS_CONNECTED_ACCOUNT_CONSENT}"', '"consent"')


def _error_code(parsed: Mapping[str, Any]) -> str:
    err = parsed.get("error")
    if isinstance(err, Mapping):
        return str(err.get("code") or "").strip()
    return str(err or "").strip()


def _consent_block(parsed: Mapping[str, Any]) -> Mapping[str, Any]:
    """The self-describing consent block, wherever the surface placed it —
    top-level ``consent`` or nested under ``error.details.consent``."""
    top = parsed.get("consent")
    if isinstance(top, Mapping):
        return top
    err = parsed.get("error") if isinstance(parsed.get("error"), Mapping) else {}
    details = err.get("details") if isinstance(err.get("details"), Mapping) else {}
    block = details.get("consent")
    return block if isinstance(block, Mapping) else {}


async def announce_result_consent(parsed: Dict[str, Any]) -> Dict[str, Any] | None:
    """Raise the chat banner a tool result's consent block asks for.

    Returns a model-safe result to substitute (agent-grant: the explainable
    consent result), or the payload itself when it announced but keeps the
    original content (connected-account: the result already carries the link),
    or None when there is no consent to raise. Never raises."""
    try:
        code = _error_code(parsed)
        if code not in (DELEGATED_CONSENT_REQUIRED, NEEDS_CONNECTED_ACCOUNT_CONSENT):
            return None
        block = _consent_block(parsed)
        namespace = str(block.get("namespace") or parsed.get("namespace") or "").strip()
        operation = str(block.get("operation") or parsed.get("operation") or "").strip()

        if code == NEEDS_CONNECTED_ACCOUNT_CONSENT:
            from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.consent import (
                raise_named_service_consent_demand,
            )

            if not namespace:
                logger.warning(
                    "[mcp-result] connected-account consent has NO namespace in the block — "
                    "the surface should self-describe it (provider=%s)",
                    block.get("provider_id"),
                )
            logger.info("[mcp-result] connected-account consent -> banner: namespace=%s", namespace)
            await raise_named_service_consent_demand(parsed, namespace=namespace, tool_name=namespace)
            # Handled; keep the original result (its link/instructions reach the
            # model and an external client). Return it so callers know it was
            # handled (an empty-return would read as "nothing happened").
            return parsed

        # delegated_consent_required — the agent's own grant. The block is
        # authoritative (the door enriched it from the bearer's credential).
        claims = [str(c) for c in (block.get("claims") or parsed.get("missing_grants") or []) if str(c or "").strip()]
        client_id = str(block.get("agent_client_id") or "").strip()
        resource = str(block.get("resource") or "").strip()
        if not claims or not client_id or not resource:
            logger.warning(
                "[mcp-result] agent-grant consent not announceable from block: "
                "client=%r resource=%r claims=%s namespace=%s — the surface must self-describe it",
                client_id, resource, claims, namespace,
            )
            return None
        from kdcube_ai_app.apps.chat.sdk.solutions.connections.mcp_consent import (
            announce_agent_consent,
            mcp_consent_from_denial,
        )

        logger.info(
            "[mcp-result] agent-grant consent -> banner: client=%s resource=%s "
            "claims=%s namespace=%s operation=%s",
            client_id, resource, claims, namespace, operation,
        )
        consent = mcp_consent_from_denial(
            {"status": 403, "reason": "authority_mismatch"},
            resource=resource,
            claims=claims,
            tool_name=str(block.get("tool_name") or namespace),
            agent_client_id=client_id,
            namespace=namespace,
            operation=operation,
        )
        await announce_agent_consent(consent)
        return consent.to_tool_result()
    except Exception:  # pragma: no cover - post-processing is best-effort
        logger.info("[mcp-result] consent announce failed (non-fatal)", exc_info=True)
        return None


def _postprocessable(text: str) -> Dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw.startswith("{") or not any(marker in raw for marker in _MARKERS):
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _process_dict(parsed: Dict[str, Any]) -> Dict[str, Any] | None:
    consent = await announce_result_consent(parsed)
    if consent is not None:
        return consent
    from kdcube_ai_app.apps.chat.sdk.solutions.widgets.send_to_user import deliver_result_files

    delivered = await deliver_result_files(parsed)
    return delivered if delivered is not parsed else None


async def _process_text(text: str) -> str | None:
    parsed = _postprocessable(text)
    if parsed is None:
        return None
    replaced = await _process_dict(parsed)
    return json.dumps(replaced, ensure_ascii=False) if replaced is not None else None


async def _process_content(content: Any) -> Any | None:
    # MCP bindings may return a string, a structured dict, or content blocks.
    if isinstance(content, str):
        return await _process_text(content)
    if isinstance(content, dict):
        direct = await _process_dict(content)
        if direct is not None:
            return direct
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            new_text = await _process_text(content["text"])
            return {**content, "text": new_text} if new_text is not None else None
        nested = content.get("content")
        if isinstance(nested, (str, dict, list)):
            replaced = await _process_content(nested)
            return {**content, "content": replaced} if replaced is not None else None
        return None
    if isinstance(content, list):
        changed = False
        out: List[Any] = []
        for item in content:
            replacement = None
            if isinstance(item, str):
                replacement = await _process_text(item)
            elif isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str):
                new_text = await _process_text(item["text"])
                if new_text is not None:
                    replacement = {**item, "text": new_text}
            if replacement is not None:
                out.append(replacement)
                changed = True
            else:
                out.append(item)
        return out if changed else None
    return None


def _unprocessed_sentinel(result: Any) -> None:
    # Loud when a result CARRIES a consent denial or download URL but was not
    # post-processed (unhandled shape) — the user saw nothing.
    raw = repr(result)
    for marker in (DELEGATED_CONSENT_REQUIRED, NEEDS_CONNECTED_ACCOUNT_CONSENT, "download_token"):
        if marker in raw:
            logger.warning(
                "[mcp-result] result carries %r but was NOT post-processed (shape %s) — "
                "no banner/file card reached the user",
                marker, type(result).__name__,
            )
            return


def bind_chat_result_handling(tools: List[Any]) -> List[Any]:
    """Wrap each MCP tool so its result is post-processed for the chat surface:
    consent denials become banners, files become cards, both through shared SDK.
    Applied ONCE by the loader; every consumer inherits it. Mutates each tool's
    coroutine in place; tools without one pass through."""
    def _wrap(orig: Any) -> Any:
        async def run(*args: Any, **kwargs: Any) -> Any:
            result = await orig(*args, **kwargs)
            try:
                if isinstance(result, tuple) and len(result) == 2:
                    replaced = await _process_content(result[0])
                    if replaced is not None:
                        return (replaced, result[1])
                else:
                    replaced = await _process_content(result)
                    if replaced is not None:
                        return replaced
                _unprocessed_sentinel(result)
            except Exception:  # pragma: no cover
                logger.warning("[mcp-result] post-process failed (non-fatal)", exc_info=True)
            return result

        return run

    wrapped = 0
    for tool in tools or []:
        orig = getattr(tool, "coroutine", None)
        if callable(orig):
            try:
                tool.coroutine = _wrap(orig)
                wrapped += 1
            except Exception:
                logger.info("[mcp-result] wrap failed for tool %s", getattr(tool, "name", "?"))
        else:
            logger.warning(
                "[mcp-result] tool %s has no coroutine — consent/delivery post-processing will NOT run for it",
                getattr(tool, "name", "?"),
            )
    if tools:
        logger.info("[mcp-result] %d/%d MCP tools bound for chat consent+delivery post-processing", wrapped, len(tools))
    return tools


__all__ = [
    "announce_result_consent",
    "bind_chat_result_handling",
]
