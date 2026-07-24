# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Token projection for a DRAFT agent configuration — the forge's cost meter.

Answers "what will the TOTAL system instruction weigh if this draft is
applied" by composing the REAL decision system text through the same
functions the runtime uses (``build_decision_system_text``), with the
draft's levers applied:

- instruction items (stored ``instr:`` refs expanded through the store);
- the tool catalog for the DRAFT tool roster — tool modules are loaded and
  introspected exactly as a running agent would see them, then rendered at
  the draft's ``tool_catalog_detail`` (full vs compact);
- the skill gallery on or off;
- subagents on (adds the delegation react tool) or off;
- multi-action protocol vs single-action;
- admin ``additional_instructions``.

The breakdown comes from DIFFS OF REAL COMPOSITIONS (full − no-gallery −
bare), never from summed estimates. Draft tool entries the projection
cannot load in-process (MCP connections, named-service kinds, unresolvable
refs) are reported in ``skipped`` — their catalog weight is NOT included,
and the response says so explicitly.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Optional

from kdcube_ai_app.apps.chat.sdk.util import token_count

logger = logging.getLogger(__name__)


def _instruction_parts(react_cfg: dict) -> tuple[Optional[str], Optional[list[str]], dict]:
    """The draft's instruction config → (body, items, facets)."""
    instructions = react_cfg.get("instructions")
    if isinstance(instructions, list):
        return None, [str(v) for v in instructions], {}
    if isinstance(instructions, str):
        return instructions, None, {}
    if isinstance(instructions, dict):
        blocks = instructions.get("blocks")
        items = [str(v) for v in blocks] if isinstance(blocks, list) else None
        body = instructions.get("body")
        return (str(body) if body else None), items, dict(instructions)
    return None, None, {}


def _subagents_enabled(react_cfg: dict) -> bool:
    subagents = react_cfg.get("subagents")
    if isinstance(subagents, dict):
        return bool(subagents.get("enabled", True))
    return bool(subagents)


async def _draft_adapters(
    tools_cfg: list,
    *,
    bundle_root: Optional[pathlib.Path],
) -> tuple[list[dict], list[str], list[str]]:
    """Load the draft tool roster in-process → (adapters, included_ids, skipped).

    One throwaway ToolSubsystem per tool connection so a single unresolvable
    entry skips (and is reported) instead of failing the whole projection.
    """
    from kdcube_ai_app.apps.chat.sdk.runtime.tool_subsystem import (
        ToolSubsystem,
        resolve_codegen_tools_specs,
    )

    adapters: list[dict] = []
    included_ids: list[str] = []
    skipped: list[str] = []

    for entry in tools_cfg or []:
        if not isinstance(entry, dict):
            skipped.append(f"{entry!r} (not a mapping)")
            continue
        kind = str(entry.get("kind") or "python").strip().lower()
        module = entry.get("module")
        ref = entry.get("ref")
        alias = str(
            entry.get("alias")
            or entry.get("name")
            or (str(module).rsplit(".", 1)[-1] if module else "")
            or (pathlib.Path(str(ref)).stem if ref else "")
        ).strip()
        if kind not in ("python", "module") or not (module or ref):
            skipped.append(f"{alias or '<unnamed>'} (kind={kind}: not loadable in-process)")
            continue
        raw_spec: dict[str, Any] = {"alias": alias}
        if module:
            raw_spec["module"] = str(module)
        else:
            raw_spec["ref"] = str(ref)
        try:
            specs = resolve_codegen_tools_specs([raw_spec], bundle_root=bundle_root)
            subsystem = ToolSubsystem(
                service=None,  # type: ignore[arg-type]  # projection never calls tools
                comm=None,  # type: ignore[arg-type]
                logger=None,
                bundle_spec=None,  # type: ignore[arg-type]  # specs pre-resolved
                context_rag_client=None,
                registry={},
                tools_specs=specs,
            )
            allowed = entry.get("allowed")
            allowed_names = [str(v) for v in allowed] if isinstance(allowed, list) else None
            # The alias map is ALWAYS passed: an omitted map narrows the
            # roster to the legacy io/ctx aliases, which is not this roster.
            # A None value means every tool of this alias.
            tool_adapters = await subsystem.react_tools(
                include_mcp=False,
                allowed_tool_names_by_alias={alias: allowed_names},
            )
            adapters.extend(tool_adapters)
            included_ids.extend(
                str(a.get("id")) for a in tool_adapters if a.get("id")
            )
        except Exception as exc:
            skipped.append(f"{alias or module or ref} ({exc})")
    return adapters, included_ids, skipped


async def project_agent_config(
    draft: dict,
    *,
    store: Any = None,
    bundle_root: Optional[pathlib.Path] = None,
    include_text: bool = False,
) -> dict:
    """Project the draft agent config → token breakdown of the REAL prompt.

    ``draft`` mirrors the two config roots the forge edits:
    ``{react: {...}, consumer: {...}, workspace_implementation?}``.
    """
    from kdcube_ai_app.apps.chat.sdk.solutions.agentic_config.instructions import (
        expand_instruction_items,
        has_custom_instruction_refs,
    )
    from kdcube_ai_app.apps.chat.sdk.solutions.react.v3.agents.decision import (
        build_decision_system_text,
    )

    react_cfg = draft.get("react") if isinstance(draft.get("react"), dict) else {}
    consumer_cfg = draft.get("consumer") if isinstance(draft.get("consumer"), dict) else {}

    body, items, facets = _instruction_parts(react_cfg)
    expanded = items
    if items and store is not None and has_custom_instruction_refs(items):
        expanded = await expand_instruction_items(items, store=store)

    tool_catalog_detail = str(facets.get("tool_catalog_detail") or "full")
    include_skill_gallery = bool(facets.get("include_skill_gallery", True))
    multi_action_mode = str(react_cfg.get("multi_action_mode") or "off")
    additional_instructions = react_cfg.get("additional_instructions")
    workspace_implementation = str(draft.get("workspace_implementation") or "custom")
    subagent_role = "parent" if _subagents_enabled(react_cfg) else None

    adapters, included_ids, skipped = await _draft_adapters(
        consumer_cfg.get("tools") or [],
        bundle_root=bundle_root,
    )

    def _compose(*, include_tool_catalog: bool, include_gallery: bool) -> str:
        return build_decision_system_text(
            adapters=adapters,
            infra_adapters=[],
            workspace_implementation=workspace_implementation,
            additional_instructions=(
                str(additional_instructions) if additional_instructions else None
            ),
            instruction_body=body,
            instruction_blocks=expanded,
            include_tool_catalog=include_tool_catalog,
            include_skill_gallery=include_gallery,
            tool_catalog_detail=tool_catalog_detail,
            multi_action_mode=multi_action_mode,
            subagent_role=subagent_role,
        )

    full_text = _compose(include_tool_catalog=True, include_gallery=include_skill_gallery)
    no_gallery = (
        _compose(include_tool_catalog=True, include_gallery=False)
        if include_skill_gallery
        else full_text
    )
    bare = _compose(include_tool_catalog=False, include_gallery=False)

    total = token_count(full_text)
    bare_tokens = token_count(bare)
    no_gallery_tokens = token_count(no_gallery)

    result: dict[str, Any] = {
        "tokens": {
            "total": total,
            "protocol_and_instructions": bare_tokens,
            "tool_catalog": max(0, no_gallery_tokens - bare_tokens),
            "skill_gallery": max(0, total - no_gallery_tokens),
        },
        "facets": {
            "tool_catalog_detail": tool_catalog_detail,
            "include_skill_gallery": include_skill_gallery,
            "multi_action_mode": multi_action_mode,
            "subagents": subagent_role == "parent",
            "workspace_implementation": workspace_implementation,
        },
        "tools": {
            "included_ids": included_ids,
            "skipped": skipped,
        },
        "items_expanded": expanded or [],
    }
    if include_text:
        result["text"] = full_text
    return result


__all__ = ["project_agent_config"]
