---
id: repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/turn-summary-contributions-README.md
title: "Hosted Agent Turn Summary Contributions"
summary: "The optional record_turn_summary capability through which a hosted foreign agent contributes one replaceable semantic summary to KDCube's shared TurnLog and conversation search without exposing its private checkpoint."
tags: ["runtime", "harness", "timeline", "turn-log", "conversation-search", "langgraph", "claude-code"]
updated_at: 2026-08-22
keywords: ["record_turn_summary", "turn summary contribution", "hosted agent summary", "foreign runtime context", "conv.working.summary", "searchable turn summary", "retrieval anchors", "turn_context"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/turn-log-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/conversation-artifacts-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/conversation/hosted-agent-conversation-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md
---
# Hosted Agent Turn Summary Contributions

A hosted foreign agent already contributes the user-visible facts KDCube can
observe at its boundary: folded user submissions, context events, attachments,
the final answer, produced conversation files, and emitted progress. An agent
may additionally know which outcome, decision, or artifact relationship should
be easy to recover later. The optional `record_turn_summary` tool lets it state
that semantic result deliberately.

This is a shared Agent Harness Timeline capability. It does not replace the
agent framework's checkpoint, compaction, or session memory.

## Model-facing contract

```text
record_turn_summary(
  summary="What was accomplished and what remains true",
  refs=["conv:fi:...", "<authorized owner ref>"],
  phrases=["exact phrase a person may search"],
  entities=["stable-name", "identifier"],
)
```

- `summary` is required and contains the reusable outcome, facts, and decisions.
- `refs` preserves locators needed to recover the work product. A ref remains a
  locator, not a credential; this tool does not resolve it or widen authority.
- `phrases` and `entities` become retrieval anchors for lexical and fuzzy
  discovery. The summary remains searchable without them.
- The writing rules are the same for every host and live in one place,
  `turn_summary_writing_guide` in `sdk/solutions/conversation/instructions.py`:
  name things by their searchable names (the user's wording, file names,
  identifiers), say what is new in this turn versus the earlier turns the model
  can see, and put verbatim re-quotable strings in `phrases` and
  turn-identifying proper nouns in `entities`. The hosted-agent prompt block
  `[SHARED TURN CONTEXT — record_turn_summary]` renders that guide and names the
  host's own search surface as the reader of the summary; the native ReAct
  `<channel:summary>` protocol renders the identical guide. See
  [Conversational Memory Search](../../../sdk/memory/conversational-memory-search-README.md)
  for why the query side depends on it.
- One draft exists per turn. Calling the tool again replaces the earlier draft.
- Trivial greetings and acknowledgements do not need a contribution.

## Lifecycle

```text
model tool call
    |
    | semantic arguments only
    v
trusted adapter binding
    |
    | stage/replace under current mutable turn state
    | no conversation id, user id, or durable write from the model
    v
agent continues and returns final_answer
    |
    v
framework-neutral successful-turn recorder
    |
    +--> TurnLog block
    |      type: conv.working.summary
    |      path: conv:ws:<turn>.conv.working.summary.attempt.1
    |
    +--> searchable assistant row
           tags: chat:summary, kind:working.summary,
                 summary_scope:turn, projection:minimal.turn.log
           embedding: best effort
           lexical/trigram text: always retained
           anchors_text: parsed from phrases/entities
```

Staging is intentionally not persistence. If the run errors before producing a
successful final answer, the draft is not committed as a completed-turn fact.
The existing TurnLog writer remains the single durable boundary and prevents a
tool call from creating orphan context outside the turn lifecycle.

## Configuration

The tool is an ordinary `kind: python` entry in the agent's consumer inventory:

```yaml
surfaces:
  as_consumer:
    agents:
      <agent-id>:
        tools:
          - name: turn_context
            kind: python
            alias: turn_context
            allowed: [record_turn_summary]
```

The descriptor entry is the administrator's ceiling. When the agent uses the
standard capabilities provider, the conversation user can disable the group or
the individual tool through the same `disabled.tools` selection used for other
Python tools. An undeclared or disabled tool is not bound.

## Adapter bindings

| Adapter | Model-facing binding | Trusted staging edge |
| --- | --- | --- |
| Hosted LangGraph tool-loop agent | ordinary LangChain `record_turn_summary` tool | the per-turn tool factory closes over trusted turn state |
| Harness-bound Claude Code | `turn_workspace` local stdio MCP tool | the per-turn authenticated Unix-socket broker invokes the trusted parent callback |

Claude Code may call the tool only when its wrapper opts in with
`turn_summary_enabled=True`; otherwise the specific MCP tool is denied and the
instruction block is absent. Direct app-owned Claude Code pipelines remain
unchanged. The ported LangGraph example declares the tool only for its
tool-calling `lg-react` agent; its fixed research graph has no model tool loop.

## Ownership boundary

The contributed summary is platform-owned shared conversation context. It is not
an automatic export of LangGraph checkpoint state, Claude session files, or a
framework-private compaction summary. Exporting private framework memory would
be a separate adapter decision. This contract records only content the agent
explicitly submits through the configured tool.

Implementation ownership:

- validation, replace semantics, and TurnLog block shape:
  `runtime/harness/timeline/contributions.py`;
- successful-turn persistence:
  `solutions/conversation/record.py`;
- searchable role projection:
  `solutions/conversation/ctx_rag.py`;
- Claude Code binding:
  `solutions/claude_code/harness_workspace.py` and the local workspace broker;
- LangGraph binding example:
  `examples/bundles/ported-langgraph-agents@2026-07-13/platform/turn_context.py`.
