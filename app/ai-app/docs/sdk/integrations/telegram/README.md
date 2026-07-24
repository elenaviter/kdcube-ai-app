---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/telegram/README.md
title: "Telegram Integration Docs"
summary: "Index for the KDCube Telegram SDK integration docs."
tags: ["sdk", "integrations", "telegram"]
keywords: ["telegram integration", "telegram bot", "telegram webhook", "telegram mini app"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/telegram/telegram-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/telegram/telegram-webhook-submit-and-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/telegram/telegram-react-events-and-artifact-history-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/event-ingress-to-react-turn-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-conversation-events-and-react-output-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/telegram/telegram-external-prereq-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/email/README.md
---

# Telegram Integration Docs

Use these docs in this order:

- [Telegram SDK Integration](telegram-README.md) - bundle wiring checklist,
  reusable KDCube SDK modules, webhook submitter helpers, Bot API rendering,
  progress streaming, Mini App auth, widget operations, signed downloads, and
  the Telegram-to-`external_events[]` ingress contract. Mini App shells are
  served from the bundle public static widget route,
  `/api/integrations/bundles/{tenant}/{project}/{bundle_id}/public/widgets/{alias}/`.
- [Telegram Webhook Submit And Queued Delivery](telegram-webhook-submit-and-delivery-README.md) -
  exact runtime boundaries for bot messages: webhook acknowledgement, chat
  ingress submission, processor-side app execution, progress streaming, and
  final Telegram rendering/delivery for ReAct, LangGraph, or custom runners.
- [ReAct Events To Telegram And Artifact History](telegram-react-events-and-artifact-history-README.md) -
  exact conversion of ReAct/common `chat.*` events into Telegram progress,
  files, errors, and final messages; also documents why Telegram keeps every
  distinct file snapshot delivered live while web chat projects the latest
  artifact per turn and logical path.
- [Telegram External Prerequisites](telegram-external-prereq-README.md) - work
  that must happen outside KDCube before the integration can function,
  including BotFather setup, public HTTPS exposure, webhook registration,
  deployment secrets, Mini App settings, and web-client download requirements.
