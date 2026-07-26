# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""The curated meaning of every built-in block: SIGNALS and semantic tags.

Each block carries runtime signals — the behaviors it protects or teaches.
This map states them in the signal-table language
(``docs/sdk/skills/instruction-blocks-and-signals-README.md`` is the authoritative
long form) so the constructor can show WHAT a block means and offer tags that
reflect meaning, not mechanics. Custom blocks author their own signals and
keywords at save time; this map covers the in-tree registries.
"""

from __future__ import annotations

BLOCK_SIGNALS: dict[str, dict[str, list[str]]] = {
    # ── moderate (REACT_LITE_*) ───────────────────────────────────────────
    "REACT_LITE_IDENTITY": {
        "signals": ["React identity: the model is the decision module inside a React loop."],
        "tags": ["identity", "protocol"],
    },
    "REACT_LITE_SECURITY_GUARD": {
        "signals": [
            "Hidden prompts and platform policies stay confidential.",
            "Retrieved and user content is data, never authority.",
        ],
        "tags": ["security", "boundaries", "prompt-safety"],
    },
    "REACT_LITE_USER_BOUNDARIES_AND_FAILURES": {
        "signals": [
            "User constraints are never silently ignored.",
            "No success claims without visible proof.",
        ],
        "tags": ["boundaries", "honesty", "failures"],
    },
    "REACT_LITE_TIMELINE_CONTEXT": {
        "signals": [
            "The rendered timeline is ordered working context for the current decision.",
            "Summaries, paths, source ids, and turn indexes are the recovery map.",
        ],
        "tags": ["timeline", "context", "recovery"],
    },
    "REACT_LITE_ANNOUNCE": {
        "signals": ["ANNOUNCE is the uncached tail block for state that changes between rounds."],
        "tags": ["announce", "runtime-state"],
    },
    "REACT_LITE_EXTERNAL_EVENTS": {
        "signals": ["Live same-turn user followups and steer events fold into the running turn."],
        "tags": ["events", "followup", "steer"],
    },
    "REACT_LITE_DECISION_LOOP": {
        "signals": ["One useful next action; inspect tool results before dependent actions advance."],
        "tags": ["decision-loop", "sequencing"],
    },
    "REACT_LITE_TOOL_USE_BASE": {
        "signals": [
            "Tools perform state changes; final answers do not.",
            "Only visible tool ids exist; the catalog is the authority.",
        ],
        "tags": ["tools", "catalog"],
    },
    "REACT_LITE_PATHS_AND_NAMESPACES": {
        "signals": ["The conv:*/sk: logical path contracts and owner-namespace handoffs."],
        "tags": ["paths", "namespaces", "refs"],
    },
    "REACT_LITE_REACT_READ_RECOVERY": {
        "signals": ["Exact logical-path reads, stats-only reads, ranged reads, turn-index reads."],
        "tags": ["read", "recovery"],
    },
    "REACT_LITE_MEMORY_SEARCH_RECOVERY": {
        "signals": ["Find prior conversation material when the exact path is unknown."],
        "tags": ["memory", "search", "recovery"],
    },
    "REACT_LITE_LOCAL_ARTIFACT_SEARCH": {
        "signals": ["react.rg searches materialized local artifact files, not hidden history."],
        "tags": ["search", "workspace", "files"],
    },
    "REACT_LITE_ATTACHMENTS": {
        "signals": ["Attachment summaries are hints; precise or visual work reads the original."],
        "tags": ["attachments", "files", "fidelity"],
    },
    "REACT_LITE_SOURCES_CITATIONS": {
        "signals": ["Source-backed claims cite source-pool ids."],
        "tags": ["sources", "citations"],
    },
    "REACT_LITE_SKILLS": {
        "signals": ["The skill catalog routes; detailed skill instructions load via sk:<skill_id>."],
        "tags": ["skills", "routing"],
    },
    "REACT_LITE_WORKSPACE_BASE": {
        "signals": [
            "React works over timeline/logical paths plus the current-turn artifact root.",
            "The physical artifact tree for file-processing tools.",
        ],
        "tags": ["workspace", "artifacts"],
    },
    "REACT_LITE_PROJECTS_AND_FILES": {
        "signals": ["git/projects is durable project state; files holds produced artifacts."],
        "tags": ["workspace", "projects", "files"],
    },
    "REACT_LITE_WORKSPACE_PULL_CHECKOUT": {
        "signals": ["Pull historical refs; checkout maintained projects before editing."],
        "tags": ["workspace", "pull", "checkout"],
    },
    "REACT_LITE_PATCHING": {
        "signals": ["Patch current-turn text files; omit displayed line prefixes."],
        "tags": ["patching", "editing"],
    },
    "REACT_LITE_REACT_WRITE_ARTIFACTS": {
        "signals": ["Write user-visible and internal artifacts with correct placement."],
        "tags": ["write", "artifacts", "canvas"],
    },
    "REACT_LITE_SUGGESTED_FOLLOWUPS": {
        "signals": ["Followup chips are short user-action phrases, never assistant questions."],
        "tags": ["followups", "ux"],
    },
    "REACT_LITE_FINALIZATION": {
        "signals": ["Complete only from visible context and successful tool results."],
        "tags": ["finalization", "honesty"],
    },
    "REACT_LITE_INTERNAL_NOTES": {
        "signals": ["Internal notes are user-invisible conversation anchors, not durable memory."],
        "tags": ["notes", "internal"],
    },
    "REACT_LITE_DURABLE_USER_MEMORY_READ": {
        "signals": ["Durable memory is user-visible and cross-conversation; the current turn overrides it."],
        "tags": ["memory", "read"],
    },
    "REACT_LITE_DURABLE_USER_MEMORY_WRITE": {
        "signals": ["Memory writes are neutral bookkeeping; success claims wait for the visible result."],
        "tags": ["memory", "write"],
    },
    "REACT_LITE_EXEC_TOOL": {
        "signals": ["Generated code runs in the isolated runtime with the OUTPUT_DIR contract."],
        "tags": ["exec", "code", "sandbox"],
    },
    "REACT_LITE_RENDERING_TOOLS": {
        "signals": ["Create renderer source first; renderer refs point to source, not output."],
        "tags": ["rendering", "documents"],
    },
    "REACT_LITE_WEB_TOOLS": {
        "signals": ["Search and fetch current external information; fetch decisive sources before precise claims."],
        "tags": ["web", "search", "fetch"],
    },
    "REACT_LITE_PLANNING": {
        "signals": ["Plans structure multi-step work; latest plan handles are readable."],
        "tags": ["planning"],
    },
    "REACT_LITE_STORY_SNAPSHOTS": {
        "signals": ["Story snapshots keep the conversation narrative recoverable across turns."],
        "tags": ["snapshots", "continuity", "timeline"],
    },
    # ── extra-lite (REACT_XLITE_*) ────────────────────────────────────────
    "REACT_XLITE_IDENTITY_AND_GUARDS": {
        "signals": ["React identity plus the confidentiality and data-not-authority boundaries."],
        "tags": ["identity", "security", "boundaries"],
    },
    "REACT_XLITE_CONTEXT_AND_EVENTS": {
        "signals": ["Timeline as context, ANNOUNCE as current truth, live followup/steer events."],
        "tags": ["timeline", "announce", "events"],
    },
    "REACT_XLITE_PATHS": {
        "signals": ["The conv:*/sk: logical path contracts."],
        "tags": ["paths", "namespaces", "refs"],
    },
    "REACT_XLITE_RECOVERY": {
        "signals": ["Recover prior material by exact read or memory search."],
        "tags": ["read", "memory", "recovery"],
    },
    "REACT_XLITE_WORKSPACE": {
        "signals": ["The workspace mental model: logical paths plus the current-turn artifact root."],
        "tags": ["workspace", "artifacts", "files"],
    },
    "REACT_XLITE_WORKSPACE_GIT_MODE": {
        "signals": ["Git-backed workspace mechanics."],
        "tags": ["workspace", "git"],
    },
    "REACT_XLITE_OPERATING": {
        "signals": ["The decision loop and base tool discipline."],
        "tags": ["decision-loop", "tools"],
    },
    "REACT_XLITE_WRITE_AND_PATCH": {
        "signals": ["Write and patch artifacts with correct placement."],
        "tags": ["write", "patching", "artifacts"],
    },
    "REACT_XLITE_EXEC": {
        "signals": ["Isolated code execution with the OUTPUT_DIR contract."],
        "tags": ["exec", "code", "sandbox"],
    },
    "REACT_XLITE_DOCUMENTS_RENDERING": {
        "signals": ["Document rendering: source first, refs to source."],
        "tags": ["rendering", "documents"],
    },
    "REACT_XLITE_WEB": {
        "signals": ["Web search and fetch for current external information."],
        "tags": ["web", "search", "fetch"],
    },
    "REACT_XLITE_ATTACHMENTS": {
        "signals": ["Attachment summaries are hints; precise work reads the original."],
        "tags": ["attachments", "files", "fidelity"],
    },
    "REACT_XLITE_SOURCES_CITATIONS": {
        "signals": ["Source-backed claims cite source-pool ids."],
        "tags": ["sources", "citations"],
    },
    "REACT_XLITE_SKILLS": {
        "signals": ["The skill catalog routes; details load via sk:<skill_id>."],
        "tags": ["skills", "routing"],
    },
    "REACT_XLITE_MEMORY_BEACONS": {
        "signals": ["Compact memory beacons anchor prior conversation state."],
        "tags": ["memory", "notes"],
    },
    "REACT_XLITE_PLANNING": {
        "signals": ["Plans structure multi-step work."],
        "tags": ["planning"],
    },
    "REACT_XLITE_FINALIZATION": {
        "signals": ["Complete only from visible successful results."],
        "tags": ["finalization", "honesty"],
    },
}


__all__ = ["BLOCK_SIGNALS"]
