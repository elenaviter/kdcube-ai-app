---
id: kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/journal/2026-08-02-descriptor-reference-alignment.md
title: "Descriptor Reference Alignment"
summary: "Connection Hub and workspace templates now share the same connected-account path and Slack connector identity, with a regression check for every declared reference."
status: active
tags: ["connection-hub", "configuration", "delegated-credentials", "slack", "testing"]
keywords: ["connected account descriptor", "connector app reference", "slack-demo", "configuration alignment"]
see_also:
  - "repo:kdcube-ai-app/app/ai-app/deployment/bundles.yaml"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/config/bundles.template.yaml"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/workspace@2026-03-31-13-36/config/bundles.template.yaml"
---

# Descriptor Reference Alignment

## Problem

The workspace template requested Slack accounts through connector app `demo`,
while the maintained Connection Hub descriptors and secret path use
`slack-demo`. The Connection Hub template also placed
`delegated_to_kdcube` beside `connections`, while the runtime reads it from
`connections.delegated_to_kdcube`.

Both shapes are valid YAML, so syntax validation alone cannot detect this
class of configuration error.

## Correction

- `delegated_to_kdcube` now lives under `connections`, matching the runtime
  lookup and secret-ref contract.
- The Slack connector app is named `slack-demo` in the Connection Hub,
  workspace, and secret templates.
- The default-install workspace uses the same connector app identity.
- The source template carries the same Google Docs claims, operation-level
  named-service policy, and `drive.file` creation scope as the maintained
  descriptors.
- A regression test walks every workspace `connected_accounts` declaration
  and verifies its provider, connector app, and claims against the Connection
  Hub template.

The maintained environment descriptors were also checked for exact
Mail/Slack/Sheets/Docs operation policies. Environment-specific hosts,
credentials, schedules, model choices, and app origins remain owned by each
environment.

## Verification

- All maintained descriptor YAML and the affected source templates parse.
- The connected-account reference regression test passes.
- Source-template Google productivity claims and namespace coverage pass.
- Focused Slack tool and connector-resolution tests pass.
