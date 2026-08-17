---
id: ported-langgraph-agents@2026-07-13/docs/journal/2026-08-17-chat-reload-and-named-service-files
title: "2026-08-17 - chat reload recovery and named-service file uploads"
status: active
tags: ["ported-langgraph-agents", "journal", "chat-history", "named-services", "conv-fi", "slack"]
---

# 2026-08-17 - chat reload recovery and named-service file uploads

Context: while testing `lg-react` Slack direct-message file upload, one ReAct
turn hit the LangGraph recursion limit. The conversation fetch still contained
later persisted turns, but reload rendering stopped after the assistant-only
error turn. The same incident exposed that a named-service/MCP provider may
receive a durable `conv:fi:` file ref without sharing the live ReAct
`OUTDIR_CV` artifact workspace.

## What changed

The historical chat reducer now preserves the earlier turn timestamp when an
assistant-only historical message is applied. This keeps later fetched turns
visible after reload even when an earlier failed turn contains no `chat:user`
artifact.

The failed-turn recording path now reconstructs and stores the user prompt,
contexts, attachments, accepted user events, and batch id before the assistant
error completion when the normal turn-log path is skipped. A failure turn should
not persist only the assistant error if the user input was already accepted.

The `lg-react` stream adapter separately catches `GraphRecursionError` as a
terminal ReAct loop condition, emits a normal final assistant answer, records a
completed `react_loop_limit` step, and completes the turn. Visible text from
separate model rounds is separated with blank lines so tool preambles do not
collapse into unreadable text.

Named-service file upload now treats `conv:fi:` as a durable conversation file
reference, not as a process-local path. A shared helper,
`materialized_conversation_file_ref`, materializes the current actor's
`conv:fi:` bytes into a temporary provider workspace and then lets the provider
tool read the file normally. Slack `upload_file` uses this path when
`payload.file_path` starts with `conv:fi:`.

## Security semantics

Materialization is actor-scoped. It requires a bound tenant, project, and
current `user_id`, and the underlying resolver reads storage only under the
current user's owner key. Mentioning another user's `conv:fi:` string is not an
authorization grant and must not produce bytes.

`request_upload`/`staged_ref` remains a separate turnless-client path:
`request_upload` only reserves an upload slot. The caller must upload bytes to
the returned `upload_url` before `staged_ref` can be consumed. Existing
conversation files should use `upload_file` with `payload.file_path =
conv:fi:...`; small inline `content_base64` remains only a fallback for clients
that truly hold bytes.

## Validation

Python tests cover failed-turn persistence, stream recursion completion,
visible-round text boundaries, Slack named-service upload, the lower Slack file
loader, and `conv:fi:` owner scoping. Components-core tests cover historical
assistant-only turn ordering after reload.

The user refreshed KDCube and confirmed the Slack `conv:fi:` upload path works.
