---
id: ported-langgraph-agents@2026-07-13/docs/journal/2026-08-17-react-stream-stability
title: "2026-08-17 - lg-react stream stability for loop failures"
status: active
tags: ["ported-langgraph-agents", "journal", "lg-react", "streaming", "recursion-limit"]
---

# 2026-08-17 - lg-react stream stability for loop failures

Context: a live `lg-react` Slack upload turn reached LangGraph's ReAct loop
recursion limit after repeated tool attempts. The conversation fetch contained
the later turns, but the UI reload path was made fragile by an earlier
assistant-only failure turn, and the live stream surfaced the raw graph exception
as the terminal user-visible result.

## What changed

`platform/stream_prebuilt.py` now treats `GraphRecursionError` as a terminal
stream condition for the chat turn. The adapter emits a normal final answer,
adds a completed `react_loop_limit` step with the exception text, calls
`complete(final_answer=...)`, and returns the answer instead of letting the
exception escape to the platform error path.

The same adapter also separates visible text emitted by distinct ReAct model
rounds with a blank-line boundary. This keeps short tool preambles from multiple
model cycles readable when a provider streams visible content before tool calls.

## Validation

`tests/test_stream_prebuilt.py` covers the new terminal recursion path, the
visible-round boundary behavior, and the existing final-answer/tool-step
streaming rules.
