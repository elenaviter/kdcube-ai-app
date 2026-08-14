---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/app-with-agents-README.md
title: "Build An App With Several Agents"
summary: "Builder recipe for declaring an arbitrary number of agents in app config with stable ids, configuring each one separately, wiring tools per runtime for React and Claude Code, exposing the model picker, and letting per-agent grants and the chat consent card follow from the agent id."
status: active
tags: ["recipes", "app", "agents", "react", "claude-code", "consent", "model-pick"]
updated_at: 2026-08-14
keywords: ["surfaces.as_consumer.agents", "agent id", "default_agent", "delegated client id", "per-agent grant", "consent card", "model picker", "ClaudeCodeWorkspaceConfig"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/bundles-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-agent-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/consume-mcp-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/agent-acting-for-user/agent-acting-for-user-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/what-i-should-know-about-app-README.md
---

# Build An App With Several Agents

Use this recipe when one app serves more than one agent — a general assistant
plus a research agent, a chat agent plus a scheduled automation agent, a React
agent plus a Claude Code subprocess — and each one needs its own capabilities,
its own model options, and its own consent boundary.

The outcome is one declaration with three readers:

```text
app config: surfaces.as_consumer.agents.<agent_id>
        |
        +-> the runtime: what this agent may call
        |
        +-> the picker: what this user may narrow or choose
        |
        `-> Connection Hub: which entity the user grants and revokes
```

KDCube still uses **bundle** in literal identifiers such as `bundles.yaml`,
`bundle_id`, and `@bundle_entrypoint`. In builder-facing prose, app and bundle
refer to the same deployable unit.

## 1. Declare The Agents, And Mean The Ids

Agents are a map under `surfaces.as_consumer`, keyed by agent id. Declare as
many as the product needs:

```yaml
# bundles.yaml
bundles:
  items:
    - id: "my.app@1-0"
      config:
        surfaces:
          as_consumer:
            default_agent: assistant
            agents:
              assistant:
                tools:
                  - id: web
                    kind: python
                    module: kdcube_ai_app.apps.chat.sdk.tools.web_tools
                    alias: web_tools
                    allowed: [web_search, web_fetch]
              researcher:
                tools:
                  - id: knowledge
                    kind: mcp
                    server_id: knowledge
                    alias: knowledge
                    allowed: ["*"]
              night_shift:
                tools:
                  - id: task_service
                    kind: named_service
                    alias: named_services
                    namespaces:
                      task:
                        allowed: [object.list, object.search, object.upsert]
```

Three agents, three inventories, no sharing between them. `default_agent`
names the id that serves a caller that names none — a chat turn without an
agent, a transport that carries no agent id, a widget that asks for the default
capabilities.

Choose ids the way you choose a database identifier, because that is what they
are. The id is the agent's identity outside this file: the platform derives the
agent's delegated-client identity from it, and per-agent grants are keyed by
that identity. Renaming `researcher` to `research_agent` creates a different
entity, and every grant users already gave the old id stops applying. Full
rules, including the fallback keys the resolver tries when an id is absent:
[Bundles Descriptor](../../configuration/bundles-descriptor-README.md).

Read the agent id from runtime context rather than a constant, so scheduled
runs, reactive events, and chat turns all resolve the same agent:

```python
agent_id = self.runtime_ctx.agent_id
application = self.config.ai_bundle_spec.id
```

## 2. Let The Administrator Configure Each Agent

Everything declared for an agent is an **inventory**: what an administrator
grants that agent. A signed-in user may narrow it for their own turns, and can
never enable anything outside it.

| Decision | Where it is declared | Who decides at runtime |
| --- | --- | --- |
| which tools exist for this agent | `agents.<id>.tools` | administrator |
| which skills this agent may load | `agents.<id>.skills` | administrator |
| which models this agent may answer on | the agent's model-pick declaration (section 4) | administrator sets the list, the user picks inside it |
| which instruction set / presentation | `config.react.<id>.instruction_profiles` | administrator sets the options, the user picks |
| whether the agent may delegate to subagents | `config.react.<id>.subagents` | administrator offers it, the user toggles it |
| which of the above are active this turn | the user's saved selection | the user |

Two agents in one app can therefore differ in authority without any code
branching: the same MCP server can appear read-only under one agent and
read/write under another. That pattern, and the allow-list grammar behind it,
is
[Connect An MCP Service To A KDCube Agent](consume-mcp-service-README.md).

Non-tool runtime behavior for an agent lives in the react block for the same
id (`config.react.<agent_id>`), so an agent's tools and its model policy are
declared under the same name.

## 3. Wire Tools Per Runtime — React And Claude Code Do Not Share

The two agent runtimes read different configuration, and neither inherits the
other's wiring. Decide per capability, per runtime.

**React** takes its catalog from the declaration in section 1. The workflow
resolves that inventory and passes it to the builder:

```python
from kdcube_ai_app.apps.chat.sdk.runtime.tool_config import (
    agent_tool_config_from_bundle_props,
)

tool_config = agent_tool_config_from_bundle_props(
    self.bundle_props, agent_id, bundle_root=BUNDLE_ROOT,
)
react = self.build_react(
    mod_tools_spec=tool_config.tool_specs,
    mcp_tools_spec=tool_config.mcp_tool_specs,
    tools_runtime=tool_config.tool_runtime,
    tool_traits=tool_config.tool_traits,
)
```

**Claude Code** takes Claude's built-in tools plus whatever MCP configuration
is written into its workspace. It does not read `surfaces.as_consumer`, and it
does not see React tools or React skills. Build its workspace explicitly:

```python
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import (
    ClaudeCodeAgentConfig,
    ClaudeCodeWorkspaceConfig,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import (
    claude_code_mcp_servers,
    resolve_turn_mcp,
)

drops: dict[str, str] = {}
server_map = await resolve_turn_mcp(
    self, connections, agent_id=agent_id, application=application, drop_sink=drops,
)
servers = claude_code_mcp_servers(server_map)

workspace_config = ClaudeCodeWorkspaceConfig(
    mcp_servers=servers,
    enabled_mcp_servers=tuple(servers),
    allowed_tools=("mcp__knowledge__search", "mcp__knowledge__read_document"),
    denied_tools=("Bash", "WebFetch"),
    instructions_markdown=CLAUDE_MD,
)
```

The SDK writes `.mcp.json`, `.claude/settings.local.json`, and `CLAUDE.md` into
`workspace_path` from that config. The app still owns which URL, which token,
which built-ins are allowed, and which instructions are safe. Input-by-input
mapping and the skills story:
[Claude Code Agent Inputs](../../sdk/bundle/bundle-agent-integration-README.md#claude-code-agent-inputs).

A capability needed by both runtimes is two pieces of work. The per-capability
table is
[Wire Each Capability Per Runtime](../../sdk/bundle/bundle-agent-integration-README.md#1a-wire-each-capability-per-runtime).

## 4. Expose The Model Picker Per Agent

The pickable list, the saved pick, and the wire operations that serve them are
platform-owned. The app declares the allowed list for each agent that should
offer the choice; declaring nothing keeps the picker hidden for that agent.

For a ReAct agent:

```yaml
config:
  react:
    researcher:
      supported_models:
        - { model: claude-sonnet-4-6, provider: anthropic, label: Sonnet 4.6 }
        - { model: claude-haiku-4-5, provider: anthropic, label: Haiku 4.5 }
```

For an agent hosted on another framework, declare the generic provider on the
agent block:

```yaml
config:
  surfaces:
    as_consumer:
      agents:
        night_shift:
          capability_provider: simple_model_pick
          capabilities:
            models:
              role: night_shift.answer
              default: claude-sonnet-4-6
              supported:
                - { model: claude-sonnet-4-6, provider: anthropic, label: Sonnet 4.6 }
                - { model: claude-haiku-4-5, provider: anthropic, label: Haiku 4.5 }
```

React applies the user's pick on its own. A framework that calls through the
model router binds a role overlay for the turn. Claude Code takes a model
*name*, so the app resolves the pick and passes it into the agent config with
its own fallback:

```python
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import (
    resolve_turn_model_pick,
)

pick = await resolve_turn_model_pick(self, state, agent_id)
model = (pick or {}).get("model") or self.bundle_prop("my_app.claude.model", "sonnet")
config = ClaudeCodeAgentConfig(
    agent_name=agent_id,
    workspace_path=workspace_path,
    model=model,
    workspace_config=workspace_config,
)
```

`resolve_turn_model_pick` returns `None` when the user picked nothing or the
stored pick has fallen outside the declared list, which is why the app supplies
the default. The full per-runtime chain is
[Per-User Model Pick](../../sdk/bundle/bundle-agent-integration-README.md#per-user-model-pick).

The list is the administrator's ceiling and the pick is the user's decision
inside it; the user's turns are billed to the user, so keep the list honest
about what each option costs in quality and spend.

## 5. Let Grants And Consent Follow From The Id

A connection marked `delegated: true` calls a KDCube surface as the signed-in
user, under a grant that belongs to *this agent of this app*:

```text
agents.<agent_id>  ->  kdcube-agent:<bundle_id>:<agent_id>
                            |
                            +-> grant record (user, agent, resources)
                            +-> per-turn bearer for delegated connections
                            `-> "Delegated by KDCube" entry the user revokes
```

Nothing extra is wired for this. Pass the same agent id and the running app id
into the resolution seams and the identity is derived for you. What the user
sees depends on it:

| State | What the user sees |
| --- | --- |
| grant present | the tool runs; the entity is listed in Connection Hub under the agent's name |
| grant missing, identity present in the demand | a chat card reading **Grant access**, opening the per-agent grant pane prefilled with the agent, resource, and claims — one click and the turn continues |
| identity missing or malformed in the demand | the generic **Open Connection Hub** card, landing on the connect-an-account view, where there is nothing matching to approve |

The third row is the failure mode worth testing on purpose: the agent is
correctly blocked, the user is correctly prompted, and the prompt leads
nowhere. It comes from an agent id that was empty, renamed, or different from
the one used when the connection was resolved.

Consent stays demand-driven: it rises when the model actually attempts the
tool, with that tool's claims, not as a turn-start union. Identity model, grant
record, and the demand chain:
[Agents Acting On Behalf Of The User](../../sdk/solutions/connections/agent-acting-for-user/agent-acting-for-user-README.md).

## 6. Verify

1. Ask each agent by id and confirm it answers with its own tool catalog, not
   another agent's.
2. Call with no agent id and confirm `default_agent` serves the turn.
3. Open the capabilities picker per agent; confirm the tool, skill, and model
   inventory matches what that agent's block declares.
4. Narrow one capability as a user, run a turn, and confirm the narrowed
   capability is absent while the administrator's declaration is unchanged.
5. Pick a non-default model and confirm the turn runs on it — for Claude Code,
   confirm the resolved model is the one passed to the CLI.
6. Revoke one agent's grant in Connection Hub and confirm its sibling agent
   still works.
7. With the grant revoked, trigger the delegated tool and confirm the chat card
   offers a one-click grant for that agent, and that approving it unblocks the
   same turn's retry.
8. For a Claude Code agent, inspect the generated `.mcp.json` and
   `.claude/settings.local.json` in the workspace and confirm they list exactly
   the servers and tools intended for that run.

For local source changes, remember that the runtime executes its staged copy of
the platform and app. Refresh or reload the staged source before judging the
result.

## Common Failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| an agent sees another agent's tools | the requested id is absent from the map and resolution fell through to a fallback key | declare the id explicitly |
| consent card says "Open Connection Hub" and leads nowhere | the demand carried no agent client identity | pass the real agent id and app id into the resolution seam |
| users must re-approve everything after a release | an agent id was renamed, producing a new client identity | keep ids stable; treat a rename as a migration |
| Claude Code cannot call a tool the React agent uses | the capability was wired for React only | wire it into the Claude workspace as MCP plus allowed tools |
| the model picker is invisible | the agent declares no allowed model list | declare it on the agent's own block |
| a stale model pick silently reverts | the pick fell outside the current allowed list and resolved to none | expected; the deployment default routes the turn |

## Done Means

- Every agent the app serves is declared under its own id, with `default_agent`
  naming one of them.
- Each agent's inventory reflects the authority that agent should have, and
  differs where the product needs it to differ.
- Capabilities needed by both runtimes are wired twice, deliberately.
- The model picker shows a list the administrator chose, per agent.
- Consent, grants, and revocation are per agent, and a missing grant produces a
  card the user can actually act on.

## Read Next

- [Bundles Descriptor](../../configuration/bundles-descriptor-README.md)
- [Bundle Agent Integration](../../sdk/bundle/bundle-agent-integration-README.md)
- [Connect An MCP Service To A KDCube Agent](consume-mcp-service-README.md)
- [Agents Acting On Behalf Of The User](../../sdk/solutions/connections/agent-acting-for-user/agent-acting-for-user-README.md)
- [What I Should Know Before Writing a KDCube App](../what-i-should-know-about-app-README.md)
