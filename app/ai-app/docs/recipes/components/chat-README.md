---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-README.md
title: "Recipe: Chat Widget"
summary: "Wire the framework-neutral KDCube chat component to an app-declared native ReAct, LangGraph, Claude Code, or other hosted agent adapter."
status: current
tags: ["recipes", "components", "chat", "agent-harness", "scene"]
updated_at: 2026-09-04
keywords: ["chat widget", "hosted agent chat", "native ReAct", "LangGraph", "Claude Code", "communicator"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-react-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-langgraph-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-claude-code-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
---
# Recipe: Chat Widget

The chat widget is an app widget that hosts a KDCube conversation and
participates in the scene as a context producer and context consumer. Its agent
may be native ReAct, LangGraph, Claude Code, or another adapter that honors the
hosted conversation contract. The widget does not know the framework's loop or
memory internals.

Read [Architecture Of What You Build](../../arch/architecture-of-what-you-build-README.md)
first for the app/service-provider map. This recipe covers the component and
transport surface. Use the matching agent recipe for backend behavior:
[native ReAct](./chat-with-react-agent-README.md),
[LangGraph](./chat-with-langgraph-agent-README.md), or
[Claude Code](./chat-with-claude-code-agent-README.md).

## Runtime Shape

```text
chat iframe
  UI shell
    timeline
    steps
    artifacts/files/context chips
  event client
    conversation stream for this chat
    optional scene subscription messages
  context adapter
    drag context out
    attach context in
  hosted agent adapter
    tools
    artifacts
    named-service object pull/read
    accounting events
```

## Serve It From Your App

Four wiring points put this widget on your own app, backed by your own agent:

1. Register the widget on the entrypoint — an `@ui_widget(alias="my_chat", ...)`
   method (its body is a served-from-build placeholder), plus a
   `visibility.widget.my_chat` block in the descriptor for who may load it.
2. Declare the UI source: `config.ui.widgets.my_chat: chat_widget_ui_config()`
   from the entrypoint, or inline YAML with
   `src_folder: sdk://solutions/chat/ui/widget` and an `engine` selector.
   `engine` picks the implementation: `local` (in-tree engine + UI, no
   `npm://`), `package` (package engine, in-tree UI), `package-ui` (package
   engine + packaged `<Chat/>` UI; materializes `@kdcube/components-*` via
   `npm://` shared sources). Helper reference:
   [Chat Widget Solution](../../sdk/solutions/chat/chat-widget-solution-README.md);
   build mechanics:
   [Widget Integration](../../sdk/npm/widget-integration-README.md).
3. Bind the agent. The widget talks to the app it is served from; the agent
   key (`agentId`, default `main`) selects which declared adapter answers.
   Declare that adapter's capabilities and inventory using its recipe. The
   widget transport stays the same.
4. Declare the chat surface: `surfaces.as_provider.bundle.default_chat: true`
   in the descriptor makes the control plane draw the conversation UI for the
   app (absent means the app presents its widget scene instead). Auth rides
   the standard widget path: the iframe resolves tokens from the served route
   and the parent CONFIG handshake, and signed-in users get the composer "+"
   menu.

To mount the widget inside a composed scene instead of serving it standalone,
declare it as a scene component: [Scene recipe](./scene-README.md).

## Scene Boundary

When embedded, the scene provides:

- runtime origin, tenant, project, and app id;
- namespace presentation config;
- context attach commands;
- scene-delivered events only when configured for that widget.

When standalone, the chat widget uses its runtime URL/config and owns its direct streams.

## Context Drag And Attach

```text
chat context chip drag
  -> kdcube-context-drag-start
     contexts: [{ ref: "mem:record:..." }]

drop on canvas
  -> scene command: kdcube.surface.command target_surface=sdk.canvas.pinboard action=pin

drop on owning widget
  -> scene object.action(open)
  -> provider ui_event.target_surface
  -> scene command: kdcube.surface.command action=open

drop from canvas/memory/task onto chat
  -> scene command: kdcube.surface.command target_surface=sdk.chat.context action=attach
  -> chat adds canonical context chip
```

The attached context keeps its original object URI. Color and label come from namespace presentation config and object metadata, not from chat-specific fallbacks.

## Event Handling

Chat has two different event concerns:

- Conversation events: tool calls, tool results, assistant/user output, and chat-local accounting display.
- Scene events: events that the host fans out to widgets that registered subscriptions.

For cross-widget accounting refresh, the usage widget is the consumer. Chat may still render accounting for the active chat turn when the event belongs to that chat instance.

## Config

A scene should declare chat as a context drop target:

```json
{
  "contextDropTargets": {
    "workspace": {
      "surfaceRef": "website.chat",
      "accepts": "context",
      "dropEffect": "attach",
      "targetSurface": "sdk.chat.context",
      "action": "attach"
    }
  }
}
```

## Current Gaps

- Context chip and drag helpers now live in shared component packages; remaining
  chat-specific wiring should stay thin and host-driven.
- Chat-local filtering for accounting events should use stable agent/session identity metadata as multi-agent flows expand.

## Related Docs

- [Architecture Of What You Build](../../arch/architecture-of-what-you-build-README.md)
- [Component Recipes](./README.md)
- [Scene Recipe](./scene-README.md)
- [Chat With A ReAct Agent](./chat-with-react-agent-README.md)
- [Chat With A LangGraph Agent](./chat-with-langgraph-agent-README.md)
- [Chat With A Claude Code Agent](./chat-with-claude-code-agent-README.md)
- [Components Ecosystem Architecture](../../sdk/solutions/ecosystem-component/components-ecosystem-README.md)
- [Chat Widget Solution](../../sdk/solutions/chat/chat-widget-solution-README.md)
- [Widget Integration](../../sdk/npm/widget-integration-README.md)
- [Context Drag And Canvas Ingress](../../sdk/npm/components-core/context-drag-README.md)
- [Host Event Bus](../../sdk/npm/components-core/host-event-bus-README.md)
- [React Event Blocks](../../sdk/agents/react/event-blocks-README.md)
