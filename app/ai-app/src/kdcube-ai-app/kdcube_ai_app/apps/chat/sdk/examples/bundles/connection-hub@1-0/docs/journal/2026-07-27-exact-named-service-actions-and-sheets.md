---
id: connection-hub@1-0/journal/2026-07-27-exact-named-service-actions-and-sheets
title: "Exact Named-Service Actions and Google Sheets"
summary: "Records exact action-level delegation for Mail, Slack, and Sheets and the addition of the Sheets namespace to Connection Hub's consent catalog."
status: implemented
tags: ["connection-hub", "named-services", "delegation", "consent", "mail", "slack", "google-sheets"]
keywords: ["exact action grants", "connected account consent", "named-service catalog"]
see_also:
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/config/bundles.template.yaml"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/README.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/interface/README.md"
  - "repo:kdcube-ai-app/app/ai-app/docs/recipes/components/named-service-README.md"
---

# Exact Named-Service Actions and Google Sheets

## Problem

A single `object.action` permission was too broad for providers with several
mutations. A user must be able to grant sending mail without also granting
unrelated mail actions, or appending spreadsheet rows without granting every
spreadsheet mutation. Google Sheets also needed to appear in the same delegated
resource and connected-account catalog as Mail and Slack.

## Change

The descriptor now declares exact action identifiers below each provider's
`object.action` family. Examples include:

```text
mail    object.action.send
slack   object.action.post_message
sheets  object.action.append_rows
```

Discovery, the consent UI, grant creation, and the KDCube Services boundary use
the same identifiers. Connection Hub projects this descriptor-owned tree; it
does not derive action names from payloads or provider implementation code.

The descriptor also adds the `sheets` namespace, its read/write KDCube grants,
and its Google connected-account requirements. The two authorization gates stay
separate:

```text
Delegated by KDCube   caller may use this KDCube resource and exact operation
Delegated to KDCube   user connected Google and approved the provider claims
```

The delegated bearer contains only the KDCube grant. Google, Gmail, and Slack
credentials remain server-side and are resolved for the approving user when an
operation runs.

## Compatibility

The runtime accepts a legacy parent `object.action` policy only when no exact
action variants are declared for that namespace. Once exact variants exist, an
undeclared action is denied. This preserves older descriptors while making new
catalogs default-closed at action granularity.

## Verification

- The consent catalog exposes the descriptor's exact Mail, Slack, and Sheets
  action identifiers.
- Selecting one action stores only that operation and its required KDCube
  grants.
- Provider-account requirements remain a separate prerequisite and never copy
  provider credentials into the delegated grant.
- An undeclared action is denied when exact variants are configured.
