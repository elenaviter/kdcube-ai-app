---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/README.md
title: "Google Integration Docs"
summary: "Index for the KDCube Google SDK integration docs."
tags: ["sdk", "integrations", "google", "gmail", "sheets"]
keywords: ["google integration", "gmail integration", "sheets integration", "google oauth adapter", "provider scopes"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/google-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/integrations/google.md
---

# Google Integration Docs

One Google OAuth client, one `google` provider, and one `gmail` connector app
serve every Google service (Gmail, Sheets, and any Drive/Calendar/Docs service
added the same way). Each service only adds claims, scopes, and tools.

Use these docs in this order:

- [Google SDK Integration](google-README.md) - the SDK mechanics: the
  `google.oauth` adapter, connected-account credential resolution, and the scope
  machinery (claim -> provider-scope union, the read-write-supersedes-read-only
  collapse, and why creating a spreadsheet needs a Drive write scope).
- [Google Services Through KDCube (Gmail, Sheets)](../../../recipes/connections/integrations/google-service-README.md) -
  the end-to-end deployment recipe: provider claims, per-service wiring, connect,
  grant, and verification.

The external operator setup (Google Cloud project, OAuth client, redirect URIs,
and per-service API enablement) lives in the bundle-local
[Connection Hub Google setup](repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/integrations/google.md).
</invoke>
