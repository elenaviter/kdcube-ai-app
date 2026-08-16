# KDCube AI Platform + SDK

> Self-hosted platform and SDK for building **AI agents and chatbots** with streaming, tools, memory, and artifacts — built for multi-tenant, multi-user production.

<!-- TODO: add the bird-flight images you mentioned (provide exact paths/URLs) -->

## What this repo is

KDCube is the full stack for agentic apps:
- **Platform**: gateway, auth, queues, monitoring, storage, and economics.
- **SDK**: bundle authoring, streaming, tools, skills, memory, and ReAct runtime.
- **Runtime**: distributed worker execution with isolation for code and tools.

## The distributed ReAct runtime (no tool-calling model required)

We run a multi-user, distributed ReAct agent loop that does **not** require a tool-calling model.
It accepts structured JSON decisions and drives tools + execution at scale.

Core capabilities:
- JSON decision protocol compatible with any LLM that can emit valid JSON
- Tool-first or code-first execution, with planning as a tool
- Sources pool + structured citations (streamed and preserved)
- Timeline & artifacts persisted per turn (files, attachments, tool results)
- Context compaction for long conversations
- Isolated code execution and rendering tool support
- Streaming to chat timeline and canvas channels

## Quickstart (build an AI agent/chatbot)

- **Bundle developer guide:** `app/ai-app/docs/sdk/bundle/bundle-dev-README.md`
- **Reference bundle (React):** `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/react@2026-02-10-02-44`
- **Bundle system overview:** `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/infra/plugin/README.md`
- **Bundle examples index:** `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/README.md`

## Docs map (by domain)

**Architecture**
- Short: `app/ai-app/docs/arch/architecture-short.md`
- Long: `app/ai-app/docs/arch/architecture-long.md`

**Platform / Infra / Ops**
- Deployment (all‑in‑one): `app/ai-app/deployment/docker/all_in_one/README.md`
- Gateway & rate limits: `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/infra/gateway/gateway-README.md`
- Auth: `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/auth/auth-README.md`
- Monitoring & observability: `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/api/monitoring/README-monitoring-observability.md`

**Streaming & Comms**
- Comm system (REST/SSE/Socket.IO): `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/doc/comm-system.md`
- Comm implementation details: `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/comm/README-comm.md`
- SSE relay (fan‑out): `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/api/sse/CHAT-RELAY-SESSION-SUBSCR-SSE-SOCKETIO-FUNOUT.README.md`

**Economics / Usage**
- Economics usage (SDK): `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/infra/economics/economics-usage.md`
- OPEX aggregation (platform): `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/api/opex/README-AGGREGATIONS.md`
- Control plane management: `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/infra/control_plane/control-plane-management.md`

**SDK & Bundles**
- Bundle docs index: `app/ai-app/docs/sdk/bundle/bundle-index-README.md`
- Bundle developer guide: `app/ai-app/docs/sdk/bundle/bundle-dev-README.md`
- Bundle ops guide: `app/ai-app/docs/sdk/bundle/bundle-ops-README.md`
- SDK index: `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/doc/SDK-index.md`
- Bundle system (multi‑bundle runtime): `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/infra/plugin/README.md`

**Agent harness and ReAct**
- Shared harness docs: `app/ai-app/docs/runtime/harness/`
- Shared timeline contracts: `app/ai-app/docs/runtime/harness/timeline/`
- Shared workspace contracts: `app/ai-app/docs/runtime/harness/workspace/`
- ReAct adapter docs: `app/ai-app/docs/sdk/agents/react/`
- ReAct model context: `app/ai-app/docs/sdk/agents/react/react-context-README.md`

**Knowledge Base**
- KB overview: `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/knowledge_base/README.md`
