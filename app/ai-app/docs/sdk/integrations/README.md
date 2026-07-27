---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/README.md
title: "SDK Integrations"
summary: "Index of reusable KDCube SDK integration packages and the shared provider-boundary contracts that apps can use instead of reimplementing integration mechanics."
tags: ["sdk", "integrations", "provider-errors", "email", "telegram", "bundles"]
keywords: ["sdk integrations", "email integration", "telegram integration", "bundle building blocks"]
updated_at: 2026-07-28
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/build/how-to-assemble-bundle-with-sdk-building-blocks-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/provider-error-contract-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/email/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/telegram/README.md
---
# SDK Integrations

SDK integrations are reusable provider and transport packages for bundles.

The bundle supplies product policy, route aliases, user-scope resolution, and
UI composition. The integration package supplies the mechanics.

| Integration | Use it for |
| --- | --- |
| [Provider Error And Observability Contract](provider-error-contract-README.md) | Required failure classification, safe client details, server logging, retries, and partial-result handling for every external provider integration. |
| [Connections](connections-README.md) *(design)* | The generic registry-driven framework for connecting external systems via OAuth in Settings (user-scoped tokens), generalizing the per-integration accounts/settings pattern below. Read this first when adding a new OAuth integration. |
| [Email](email/README.md) | Gmail OAuth/API, iCloud IMAP/SMTP, account settings, attachment materialization, Email MCP, Claude Code email processing, and email delivery helpers. |
| [LinkedIn](linkedin/README.md) | LinkedIn OAuth, UGC Posts API for text and image posts, content formatting helpers (`format_post_text`), image upload via Assets API. |
| [Telegram](telegram/README.md) | Webhooks, Bot API rendering, progress streaming, Mini App auth, user registry storage, widget operations, and signed downloads. |

For the broader bundle-builder selection map, start with
[How To Assemble A Bundle With SDK Building Blocks](../bundle/build/how-to-assemble-bundle-with-sdk-building-blocks-README.md).
