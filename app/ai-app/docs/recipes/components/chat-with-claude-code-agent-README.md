---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-claude-code-agent-README.md
title: "Recipe: Chat With A Claude Code Agent"
summary: "End-to-end steps for putting the hosted Claude Code lane behind the chat component: declare the chat surface and the agent's runtime block, order the turn so refusals happen before spend, watch the lane so the run can be reached and stopped, and record a conversation that reloads without a ReAct timeline."
status: active
tags: ["recipes", "component", "chat", "claude-code", "hosted-agent", "run-to-completion"]
updated_at: 2026-09-04
keywords: ["default_chat", "accepts_steer", "accepts_followup", "live_control", "VITE_CHAT_AGENT_ID", "run-to-completion turn", "turn frame", "PreToolUse hook", "fail-closed spend gate", "in-band answer", "conversation record", "hosted agent chat"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-react-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/app-with-resident-coding-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/conversation/hosted-agent-conversation-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-langgraph-agent-README.md
  - repo:kdcube-ai-app/agents/claude/README.md
---

# Recipe: Chat With A Claude Code Agent

Use this when you want the chat component's ordinary conversation — a person
types, an agent answers — and the agent behind it is the **Claude Code CLI**
rather than a ReAct agent. The person gets file tools, Bash, skills and
subagents in the answer they asked for, without your app hosting a coding
surface.

If the work your app governs *is* a repository the user edits through the agent,
read [Build an app with a resident coding agent](../apps/app-with-resident-coding-agent-README.md)
instead — that recipe owns the git store, the desk beside the chat, and the
rulebook. This one is the chat-lane parallel to
[Chat with a ReAct agent](./chat-with-react-agent-README.md).

The difference that shapes everything below: **this lane's loop is not yours.**
The CLI decides when to call a tool and when to stop. You cannot fold a message
into its iteration mid-run, so anything the person does while it works has to
reach it another way, and anything you want recorded has to be recorded by you.

## 1. Declare the chat surface

The chat tile is inherited; the app only has to say that it hosts one.

```yaml
surfaces:
  as_provider:
    bundle:
      default_chat: true        # the platform routes conversations here
```

Do **not** declare a chat widget of your own alongside it: a second alias mounts
the same app twice and the scene shows a duplicate tile. Bind the inherited tile
to your agent id at build time (`VITE_CHAT_AGENT_ID`), or the widget drives the
generic `main` id — the app still answers, but the capabilities picker writes a
different agent's inventory than the one your turn lane resolves.

Product intent belongs in the descriptor, not inferred from code: an app that
inherits a chat-capable base class is not thereby a chat app.

## 2. Declare the agent's runtime block

These knobs shape the CLI run. The agent's *service* inventory is separate
(step 3).

```yaml
agent:
  claude_bin: "claude"
  model: "claude-opus-5"      # deployment default when the user picks nothing
  timeout_seconds: 900        # per-turn wall clock
  live_control: true          # reach and stop the run while it works (step 6)
```

`model` is the no-pick default and must also appear in the pickable list, or the
picker's "default" tag names a model no turn actually runs.

`timeout_seconds` is the number that makes the rest of this recipe matter: a
fifteen-minute ceiling means a person can be watching a wrong answer being built
for a quarter of an hour.

## 3. Declare what the composer may offer

The chat composer draws the stop and follow-up controls only for an agent that
says it can honour them. This is a per-agent declaration, not an assumption:

```yaml
capabilities:
  conversation:
    accepts_steer: true       # the stop control
    accepts_followup: true    # sending while a turn runs
  models: [...]               # the pickable list; the user picks per conversation
```

Both default to false. An agent that cannot be stopped should say so, and one
that can must say so — the controls exist in the widget either way, and silence
reads as "not supported".

The model list is the "user pays, user decides" boundary: the admin declares the
ceiling, the person picks their own quality-vs-spend point per conversation.

## 4. Order the turn so refusals cost nothing

Everything below happens in one `execute_core`. The order is the recipe — each
step is placed where it is because of what it prevents.

```
fold the pending lane        → the turn's whole input, in order
name the conversation        → first turn only, BEFORE the run
access rule                  → refuse in band
spend gate                   → refuse in band
capability set               → admin ceiling ∩ the person's pick
per-turn MCP resolution      → connect-required answered in band
workspace + support files    → per conversation, not per turn
session-store binding        → one CLI session lineage per conversation
seed live control            → the hook that can reach the run
the turn frame               → what the model actually reads
accounting boundary          → every priced call attributed
run, watched                 → the lane is polled while the CLI works
reconcile + answer           → what was delivered, what the person sees
```

Four placements are load-bearing:

**The fold takes the whole pending lane**, not the waking event. A single visible
send arrives as several lane events (prompt plus attachments and context refs),
and messages typed while the previous turn ran are still pending. Fold them all,
in lane order, and render every one in the frame — a fold that delivers three
messages and renders the first silently drops two.

**Name the conversation before the agent runs.** The title is generated from the
question, so it survives a turn that later errors. A title generated from the
answer is a conversation that stays "Untitled" precisely when something went
wrong.

**Refusals precede spend.** The access rule and the fail-closed spend gate answer
*in band* — a sentence in the chat saying what is missing and who can fix it —
before any priced call. Fail-open on a missing roles carrier is not allowed: no
roles reads as not authorised.

**Failures are answered, never silent.** A run that produces no answer says so
with its status and exit code, and says that nothing was retried. An empty
assistant bubble is the one outcome the person cannot act on.

## 5. Give the run a workspace it can keep

The workspace is **per conversation**, not per turn: a new one every turn
re-creates the prompt cache and throws away the CLI's session. The support files
it needs — the resolved MCP config with this turn's bearer, the instruction file,
trust — are written into it before the run.

The CLI's session transcript is kept in a git-backed session store so the
conversation survives a worker moving. Two consequences to design for, both
measured:

- resume after a kill is **clean** — a run killed mid-tool-loop resumes on the
  same session id with its memory intact, so the timeout above is safe
- the store **empties and restores its checkout every turn**, so nothing per-turn
  may be seeded inside it (see step 6)

## 6. Reach the run while it works

The CLI's loop is not yours, so the platform watches rather than seizes: a
read-only poll of the conversation lane during the turn, and a `PreToolUse` hook
that fires before every tool call. What the hook answers is what reaches a run in
flight:

- **`allow` + a reason** — the model reads what the person said and keeps working
- **`deny` + a reason** — the tool call does not happen, so the model answers
  with what it has: one more round, enforced, nothing killed

Streaming stdin does not work for this (measured: the message is ingested and
acknowledged in milliseconds, and the running turn never acts on it).

Three rules the implementation must keep:

1. **Seed the hook OUTSIDE the session store's checkout.** That directory is
   emptied and reset to the previous turn's snapshot at the start of every run;
   a buffer seeded inside it is deleted and replaced by the previous turn's —
   including its stop flag *and* the turn id that matches it, so every guard
   agrees and every tool call of every later turn is refused. The symptom is
   total and looks like a permission policy to the agent, which will ask about
   it rather than route around it.
2. **Stamp the buffer with its turn, and put that id on the hook's command
   line.** The workspace outlives the turn; a stop belongs to one run.
3. **Reconcile what was delivered.** A hook that fired means the model read those
   words inside this turn, so the lane can close them as answered. What no hook
   reached — a run sitting in one long tool call — stays pending and is folded
   into the next turn. Delivered and arrived are different things, and only the
   buffer knows which.

A stop is a boundary, not a discard: messages sent before it stay pending and
are read by the next turn a person starts; they simply do not start one.

## 7. Record a conversation that reloads

This lane has no ReAct timeline, so nothing writes the conversation record for
you. The turn must persist: the user block **per message** (a folded turn has
several, each with its own id, arrival time and event type), the assistant
answer, any files the run hosted, the title, and the turn timing. Without the
per-message blocks a folded turn reloads as one message and the rest vanish —
having been visible live, which is the worst version of that bug.

## Verify

- a plain question answers, and the answer streams
- tool activity shows as steps, not as the answer
- the stop control is drawn, and pressing it ends the run at the next tool call
  with the agent reporting what it had finished
- **the turn after a stop runs normally** — this is the regression test that
  matters; if its tool calls are refused, the hook is reading a stale buffer
- a message sent mid-turn reaches the run, or waits and is folded into the next
  turn, but is never lost
- reload shows every message of a folded turn, in the order they were sent
- a conversation gets a title on its first turn

The direct reference at
[`agents/claude`](../../../../../agents/claude/README.md) constructs
`ClaudeCodeAgent` around the local Claude Code executable. It runs two turns in
one stable Claude session, records real communicator and accounting events,
and creates a PDF and XLSX in a local workspace. Redis holds the accounting
mirror; Postgres and configured storage hold the common KDCube conversation
record. Claude's stable CLI session remains its private continuation. Its
Bash/file tools run at the trusted local Claude Code process boundary.

## Common failures

| Symptom | Cause |
| --- | --- |
| no stop control in the composer | `accepts_steer` not declared |
| every turn after a stop is refused | live-control files seeded inside the session store's checkout |
| duplicate chat tile in the scene | app declares its own chat widget alongside `default_chat` |
| picker writes an inventory the turn ignores | tile not bound to the agent id at build |
| conversation stays "Untitled" | title generated after the run, or from the answer |
| messages vanish on reload | one user block per turn instead of per message |
| empty assistant bubble | a failed run that did not answer in band |

## Read next

- [Claude Code Agent](../../sdk/agents/claude/claude-code-README.md) — the runner:
  CLI invocation, streaming, session resume, allowed tools, live control
- [Workspace bootstrap](../../sdk/agents/claude/claude-code-workspace-bootstrap-README.md)
  — the workspace, the session store, and what may not be seeded into it
- [Hosted agent conversation](../../sdk/solutions/conversation/hosted-agent-conversation-README.md)
  — the record a lane without a ReAct timeline owes the reload
- [Reactive turn delivery](../../sdk/events/reactive-turn-delivery-README.md) —
  the per-lane table: what each runtime can be told mid-turn
