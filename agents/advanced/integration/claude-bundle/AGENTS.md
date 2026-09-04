---
id: harness-claude-demo@1-0/agents
title: "Claude Harness Demo Builder-Agent Onboarding"
summary: "Maintenance boundaries for the focused Claude Code Agent Harness demonstration bundle."
status: active
tags: ["agents", "claude-code", "harness", "demonstration"]
see_also:
  - "repo:kdcube-ai-app/agents/advanced/integration/README.md"
  - "repo:kdcube-ai-app/agents/claude/README.md"
  - "repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md"
---

# Claude Harness Demo Builder-Agent Onboarding

Keep `entrypoint.py` as the composition root and turn behavior under
`services/`. Bundle code uses package-relative imports. Platform SDK imports
are absolute.

The host owns caller identity, economics, conversation recording, live-lane
reconciliation, session persistence, and file publication. The Claude child
receives only its bound workspace configuration and a resolved credential in
subprocess memory. Never serialize that credential into workspace files,
events, logs, or descriptors.

The standard-library artifact writer is deterministic support code, not an
alternative agent loop. Changes must keep its focused tests and the shared
runtime-backed two-turn demonstration passing.
